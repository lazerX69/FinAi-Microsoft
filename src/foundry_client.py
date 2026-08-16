"""
FinAi için Microsoft Foundry Local istemcisi.

Bu modülün sorumlulukları:

- Foundry Local SDK yöneticisini güvenli biçimde başlatmak
- Yapılandırılan yerel modeli katalogdan bulmak
- Modeli gerektiğinde indirmek ve belleğe yüklemek
- Streaming veya tek parça metin üretmek
- Boş ya da geçersiz streaming paketlerini güvenle atlamak
- Aynı modelin tekrar tekrar yüklenmesini önlemek
- Uygulama kapanırken model kaynaklarını serbest bırakmak

Örnek kullanım:

    from src.foundry_client import FoundryClient

    client = FoundryClient()

    response = client.complete(
        [
            {
                "role": "system",
                "content": "Sen Türkçe cevap veren bir asistansın.",
            },
            {
                "role": "user",
                "content": "Enflasyon nedir?",
            },
        ]
    )

    print(response)
    client.close()

Komut satırı testi:

    python -m src.foundry_client

Özel test sorusu:

    python -m src.foundry_client "Bileşik faiz nedir?"
"""

from __future__ import annotations

import argparse
import atexit
import logging
import re
import sys
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from foundry_local_sdk import Configuration, FoundryLocalManager

from src.config import MODEL_ALIAS


LOGGER = logging.getLogger("finai.foundry_client")

DEFAULT_APP_NAME = "FinAi-Local-RAG"
DEFAULT_TEMPERATURE = 0.10
DEFAULT_TOP_P = 0.80
DEFAULT_MAX_TOKENS = 350

MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4_096

MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0

MIN_TOP_P = 0.0
MAX_TOP_P = 1.0

MAX_MESSAGE_LENGTH = 40_000
MAX_TOTAL_PROMPT_LENGTH = 80_000

VALID_MESSAGE_ROLES = frozenset(
    {
        "system",
        "user",
        "assistant",
        "tool",
    }
)


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """
    Yerel model metin üretim ayarları.

    Attributes:
        temperature:
            Yanıt rastlantısallığı. Düşük değerler RAG için daha tutarlıdır.
        top_p:
            Nucleus sampling eşiği.
        max_tokens:
            Üretilebilecek maksimum token sayısı.
    """

    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    max_tokens: int = DEFAULT_MAX_TOKENS

    def __post_init__(self) -> None:
        if isinstance(self.temperature, bool) or not isinstance(
            self.temperature,
            (int, float),
        ):
            raise TypeError("temperature sayısal bir değer olmalıdır.")

        if not MIN_TEMPERATURE <= float(self.temperature) <= MAX_TEMPERATURE:
            raise ValueError(
                f"temperature {MIN_TEMPERATURE} ile "
                f"{MAX_TEMPERATURE} arasında olmalıdır."
            )

        if isinstance(self.top_p, bool) or not isinstance(
            self.top_p,
            (int, float),
        ):
            raise TypeError("top_p sayısal bir değer olmalıdır.")

        if not MIN_TOP_P <= float(self.top_p) <= MAX_TOP_P:
            raise ValueError(
                f"top_p {MIN_TOP_P} ile {MAX_TOP_P} arasında olmalıdır."
            )

        if isinstance(self.max_tokens, bool) or not isinstance(
            self.max_tokens,
            int,
        ):
            raise TypeError("max_tokens bir tam sayı olmalıdır.")

        if not MIN_MAX_TOKENS <= self.max_tokens <= MAX_MAX_TOKENS:
            raise ValueError(
                f"max_tokens {MIN_MAX_TOKENS} ile "
                f"{MAX_MAX_TOKENS} arasında olmalıdır."
            )


@dataclass(frozen=True, slots=True)
class FoundryClientStatus:
    """FoundryClient çalışma durumunu temsil eder."""

    model_alias: str
    manager_initialized: bool
    model_resolved: bool
    model_loaded: bool
    client_ready: bool
    closed: bool


def configure_logging(verbose: bool = False) -> None:
    """Komut satırı çalıştırmaları için logging yapılandırır."""

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )


def normalize_text(value: str, *, field_name: str) -> str:
    """
    Kullanıcı veya mesaj metnini güvenli biçimde normalize eder.
    """

    if not isinstance(value, str):
        raise TypeError(f"{field_name} string olmalıdır.")

    normalized = value.replace("\x00", " ")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.strip()

    if not normalized:
        raise ValueError(f"{field_name} boş olamaz.")

    if len(normalized) > MAX_MESSAGE_LENGTH:
        raise ValueError(
            f"{field_name} maksimum {MAX_MESSAGE_LENGTH} karakter olabilir."
        )

    return normalized


def validate_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """
    Chat mesajlarını Foundry Local istemcisine göndermeden önce doğrular.

    Kabul edilen biçim:

        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
        ]

    Returns:
        Normalize edilmiş mesaj listesi.
    """

    if isinstance(messages, (str, bytes)) or not isinstance(
        messages,
        Sequence,
    ):
        raise TypeError("messages bir mesaj dizisi olmalıdır.")

    if not messages:
        raise ValueError("En az bir chat mesajı gereklidir.")

    validated_messages: list[dict[str, str]] = []
    total_length = 0

    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(
                f"Mesaj {index} sözlük benzeri bir nesne olmalıdır."
            )

        raw_role = message.get("role")
        raw_content = message.get("content")

        if not isinstance(raw_role, str):
            raise TypeError(f"Mesaj {index} role alanı string olmalıdır.")

        role = raw_role.strip().lower()

        if role not in VALID_MESSAGE_ROLES:
            raise ValueError(
                f"Mesaj {index} için geçersiz role: {raw_role!r}. "
                f"Desteklenen roller: {', '.join(sorted(VALID_MESSAGE_ROLES))}"
            )

        content = normalize_text(
            raw_content,
            field_name=f"Mesaj {index} content alanı",
        )

        total_length += len(content)

        if total_length > MAX_TOTAL_PROMPT_LENGTH:
            raise ValueError(
                "Mesajların toplam uzunluğu izin verilen sınırı aşıyor. "
                f"Maksimum: {MAX_TOTAL_PROMPT_LENGTH} karakter."
            )

        validated_messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    if not any(
        message["role"] == "user"
        for message in validated_messages
    ):
        raise ValueError("Mesajlarda en az bir user mesajı bulunmalıdır.")

    return validated_messages


def extract_stream_text(chunk: Any) -> str:
    """
    Foundry Local streaming paketinden metni güvenli biçimde çıkarır.

    Streaming tamamlanırken choices listesi boş olan kontrol paketleri
    gelebilir. Bu fonksiyon böyle paketlerde hata üretmek yerine boş string
    döndürür.
    """

    if chunk is None:
        return ""

    choices = getattr(chunk, "choices", None)

    if not choices:
        return ""

    try:
        first_choice = choices[0]
    except (IndexError, TypeError):
        return ""

    delta = getattr(first_choice, "delta", None)

    if delta is None:
        return ""

    content = getattr(delta, "content", None)

    if not isinstance(content, str):
        return ""

    return content


class FoundryClient:
    """
    Microsoft Foundry Local model yaşam döngüsünü yöneten istemci.

    Model lazy-loading yaklaşımıyla yalnızca ilk üretim çağrısında hazırlanır.
    Aynı nesne kullanıldığı sürece model tekrar yüklenmez.

    Args:
        model_alias:
            Foundry Local model katalog alias değeri.
        app_name:
            Foundry Local uygulama adı.
        settings:
            Varsayılan üretim ayarları.
        auto_download:
            Model cihazda bulunmuyorsa otomatik indirmeye izin verir.
        preload:
            True olduğunda model sınıf oluşturulurken hemen hazırlanır.
    """

    _manager_lock = threading.RLock()
    _generation_lock = threading.RLock()

    def __init__(
        self,
        *,
        model_alias: str = MODEL_ALIAS,
        app_name: str = DEFAULT_APP_NAME,
        settings: GenerationSettings | None = None,
        auto_download: bool = True,
        preload: bool = False,
    ) -> None:
        self.model_alias = normalize_text(
            model_alias,
            field_name="model_alias",
        )
        self.app_name = normalize_text(
            app_name,
            field_name="app_name",
        )

        if not isinstance(auto_download, bool):
            raise TypeError("auto_download boolean olmalıdır.")

        if not isinstance(preload, bool):
            raise TypeError("preload boolean olmalıdır.")

        self.settings = settings or GenerationSettings()
        self.auto_download = auto_download

        self._manager: Any | None = None
        self._model: Any | None = None
        self._chat_client: Any | None = None

        self._manager_initialized = False
        self._model_resolved = False
        self._model_loaded = False
        self._closed = False

        self._lifecycle_lock = threading.RLock()

        atexit.register(self.close)

        if preload:
            self.ensure_ready()

    @property
    def status(self) -> FoundryClientStatus:
        """İstemcinin güncel yaşam döngüsü durumunu döndürür."""

        return FoundryClientStatus(
            model_alias=self.model_alias,
            manager_initialized=self._manager_initialized,
            model_resolved=self._model_resolved,
            model_loaded=self._model_loaded,
            client_ready=self._chat_client is not None,
            closed=self._closed,
        )

    @property
    def is_ready(self) -> bool:
        """Modelin metin üretimine hazır olup olmadığını döndürür."""

        return (
            not self._closed
            and self._manager_initialized
            and self._model_resolved
            and self._model_loaded
            and self._chat_client is not None
        )

    def _ensure_not_closed(self) -> None:
        """Kapatılmış istemcinin yeniden kullanılmasını engeller."""

        if self._closed:
            raise RuntimeError(
                "FoundryClient kapatıldı ve yeniden kullanılamaz. "
                "Yeni bir FoundryClient örneği oluşturun."
            )

    def _initialize_manager(self) -> None:
        """
        Foundry Local yöneticisini process genelinde güvenli biçimde başlatır.
        """

        if self._manager_initialized and self._manager is not None:
            return

        with self._manager_lock:
            if self._manager_initialized and self._manager is not None:
                return

            LOGGER.info("Foundry Local yöneticisi başlatılıyor...")

            try:
                FoundryLocalManager.initialize(
                    Configuration(app_name=self.app_name)
                )
            except Exception as exc:
                # SDK daha önce aynı process içinde başlatılmış olabilir.
                LOGGER.debug(
                    "FoundryLocalManager.initialize sonucu: %s",
                    exc,
                )

            manager = FoundryLocalManager.instance

            if manager is None:
                raise RuntimeError(
                    "Foundry Local yöneticisi başlatılamadı."
                )

            self._manager = manager
            self._manager_initialized = True

            LOGGER.info("Foundry Local yöneticisi hazır.")

    def _resolve_model(self) -> None:
        """Yapılandırılan model alias'ını katalogdan bulur."""

        if self._model_resolved and self._model is not None:
            return

        self._initialize_manager()

        LOGGER.info(
            "Foundry Local modeli katalogdan aranıyor: %s",
            self.model_alias,
        )

        try:
            model = self._manager.catalog.get_model(self.model_alias)
        except Exception as exc:
            raise RuntimeError(
                f"Model kataloğu sorgulanamadı: {self.model_alias}"
            ) from exc

        if model is None:
            raise RuntimeError(
                f"Foundry Local kataloğunda model bulunamadı: "
                f"{self.model_alias}"
            )

        self._model = model
        self._model_resolved = True

        LOGGER.info("Foundry Local modeli bulundu: %s", self.model_alias)

    @staticmethod
    def _download_progress(progress: Any) -> None:
        """
        SDK indirme ilerlemesini loglar.

        SDK sürümüne göre progress int veya float olabilir.
        """

        if isinstance(progress, bool) or not isinstance(
            progress,
            (int, float),
        ):
            return

        numeric_progress = float(progress)

        # Bazı SDK sürümleri 0-1, bazıları 0-100 döndürebilir.
        if 0.0 <= numeric_progress <= 1.0:
            numeric_progress *= 100.0

        numeric_progress = max(0.0, min(100.0, numeric_progress))

        LOGGER.info(
            "Model indirme ilerlemesi: %.0f%%",
            numeric_progress,
        )

    def _download_model(self) -> None:
        """Modeli indirir veya mevcut cache kaydını doğrular."""

        self._resolve_model()

        if not self.auto_download:
            LOGGER.info(
                "Otomatik model indirme devre dışı; mevcut cache kullanılacak."
            )
            return

        LOGGER.info(
            "Model indiriliyor veya yerel cache kontrol ediliyor: %s",
            self.model_alias,
        )

        try:
            self._model.download(self._download_progress)
        except OSError as exc:
            raise RuntimeError(
                "Model indirilemedi. Disk alanını ve dosya izinlerini "
                "kontrol edin."
            ) from exc
        except Exception as exc:
            error_message = str(exc).lower()

            if (
                "diskte yeterli yer yok" in error_message
                or "not enough space" in error_message
                or "no space left" in error_message
            ):
                raise RuntimeError(
                    f"'{self.model_alias}' modeli için yeterli disk alanı "
                    "bulunmuyor. Yarım kalan model cache dosyalarını temizleyin "
                    "veya daha küçük bir model kullanın."
                ) from exc

            raise RuntimeError(
                f"Foundry Local modeli indirilemedi: {self.model_alias}"
            ) from exc

        LOGGER.info("Model indirme/cache kontrolü tamamlandı.")

    def _load_model(self) -> None:
        """Modeli belleğe yükler ve chat istemcisini oluşturur."""

        if self._model_loaded and self._chat_client is not None:
            return

        self._download_model()

        LOGGER.info(
            "Foundry Local modeli belleğe yükleniyor: %s",
            self.model_alias,
        )

        try:
            self._model.load()
        except Exception as exc:
            raise RuntimeError(
                f"Foundry Local modeli yüklenemedi: {self.model_alias}"
            ) from exc

        try:
            chat_client = self._model.get_chat_client()
        except Exception as exc:
            try:
                self._model.unload()
            except Exception:
                LOGGER.debug(
                    "Başarısız client oluşturma sonrasında model "
                    "boşaltılamadı.",
                    exc_info=True,
                )

            raise RuntimeError(
                "Foundry Local chat istemcisi oluşturulamadı."
            ) from exc

        if chat_client is None:
            try:
                self._model.unload()
            except Exception:
                LOGGER.debug(
                    "Boş client sonrasında model boşaltılamadı.",
                    exc_info=True,
                )

            raise RuntimeError(
                "Foundry Local geçerli bir chat istemcisi döndürmedi."
            )

        self._chat_client = chat_client
        self._model_loaded = True

        self._apply_settings(self.settings)

        LOGGER.info(
            "Foundry Local modeli üretime hazır: %s",
            self.model_alias,
        )

    def _apply_settings(
        self,
        settings: GenerationSettings,
    ) -> None:
        """Üretim ayarlarını SDK chat istemcisine uygular."""

        if self._chat_client is None:
            raise RuntimeError("Chat istemcisi henüz hazır değil.")

        client_settings = getattr(self._chat_client, "settings", None)

        if client_settings is None:
            raise RuntimeError(
                "Foundry Local chat istemcisinde settings alanı bulunamadı."
            )

        try:
            client_settings.temperature = float(settings.temperature)
            client_settings.top_p = float(settings.top_p)
            client_settings.max_tokens = settings.max_tokens
        except Exception as exc:
            raise RuntimeError(
                "Foundry Local üretim ayarları uygulanamadı."
            ) from exc

    def ensure_ready(self) -> None:
        """
        Model ve chat istemcisini metin üretimine hazırlar.

        Thread-safe ve idempotent çalışır. Model zaten hazırsa işlem yapmaz.
        """

        self._ensure_not_closed()

        if self.is_ready:
            return

        with self._lifecycle_lock:
            self._ensure_not_closed()

            if self.is_ready:
                return

            self._load_model()

    def stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        settings: GenerationSettings | None = None,
    ) -> Iterator[str]:
        """
        Model yanıtını parça parça üretir.

        Args:
            messages:
                Chat mesajları.
            settings:
                Yalnızca bu üretim çağrısında uygulanacak ayarlar.

        Yields:
            Boş olmayan metin parçaları.

        Raises:
            RuntimeError:
                Model yükleme veya üretim işlemi başarısız olursa.
        """

        validated_messages = validate_messages(messages)
        effective_settings = settings or self.settings

        if not isinstance(effective_settings, GenerationSettings):
            raise TypeError(
                "settings bir GenerationSettings örneği olmalıdır."
            )

        self.ensure_ready()

        with self._generation_lock:
            self._ensure_not_closed()
            self._apply_settings(effective_settings)

            LOGGER.info(
                "Foundry Local üretimi başlatıldı | model=%s | "
                "max_tokens=%d",
                self.model_alias,
                effective_settings.max_tokens,
            )

            started_at = time.perf_counter()
            generated_character_count = 0

            try:
                response_stream = (
                    self._chat_client.complete_streaming_chat(
                        validated_messages
                    )
                )

                for chunk in response_stream:
                    text = extract_stream_text(chunk)

                    if not text:
                        continue

                    generated_character_count += len(text)
                    yield text

            except GeneratorExit:
                LOGGER.debug(
                    "Streaming tüketici tarafından erken sonlandırıldı."
                )
                raise
            except Exception as exc:
                raise RuntimeError(
                    "Foundry Local streaming metin üretimi başarısız oldu."
                ) from exc
            finally:
                elapsed_seconds = time.perf_counter() - started_at

                LOGGER.info(
                    "Foundry Local üretimi tamamlandı | karakter=%d | "
                    "süre=%.2f saniye",
                    generated_character_count,
                    elapsed_seconds,
                )

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        settings: GenerationSettings | None = None,
    ) -> str:
        """
        Model yanıtını tek bir string olarak üretir.

        Boş yanıtlar hata kabul edilir.
        """

        parts = list(
            self.stream(
                messages,
                settings=settings,
            )
        )

        response = "".join(parts).strip()

        if not response:
            raise RuntimeError(
                "Foundry Local modeli boş yanıt döndürdü."
            )

        return response

    def complete_prompt(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        settings: GenerationSettings | None = None,
    ) -> str:
        """
        Basit prompt için chat mesajlarını otomatik oluşturur.
        """

        normalized_prompt = normalize_text(
            prompt,
            field_name="prompt",
        )

        messages: list[dict[str, str]] = []

        if system_prompt is not None:
            messages.append(
                {
                    "role": "system",
                    "content": normalize_text(
                        system_prompt,
                        field_name="system_prompt",
                    ),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": normalized_prompt,
            }
        )

        return self.complete(
            messages,
            settings=settings,
        )

    def close(self) -> None:
        """
        Yüklü modeli bellekten boşaltır.

        Birden fazla kez güvenle çağrılabilir.
        """

        with self._lifecycle_lock:
            if self._closed:
                return

            if self._model_loaded and self._model is not None:
                LOGGER.info(
                    "Foundry Local modeli bellekten boşaltılıyor: %s",
                    self.model_alias,
                )

                try:
                    self._model.unload()
                except Exception:
                    LOGGER.warning(
                        "Foundry Local modeli boşaltılırken hata oluştu.",
                        exc_info=True,
                    )

            self._chat_client = None
            self._model_loaded = False
            self._closed = True

            LOGGER.info("FoundryClient kapatıldı.")

    def __enter__(self) -> "FoundryClient":
        """Context manager giriş noktası."""

        self.ensure_ready()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        """Context manager çıkışında modeli boşaltır."""

        self.close()


def build_test_messages(question: str) -> list[dict[str, str]]:
    """Komut satırı testi için mesaj listesi oluşturur."""

    normalized_question = normalize_text(
        question,
        field_name="question",
    )

    return [
        {
            "role": "system",
            "content": (
                "Sen FinAi adlı Türkçe finansal okuryazarlık "
                "asistanısın. Cevabını sade, doğru ve en fazla üç kısa "
                "cümleyle yaz. Kişiye özel yatırım tavsiyesi verme."
            ),
        },
        {
            "role": "user",
            "content": normalized_question,
        },
    ]


def build_argument_parser() -> argparse.ArgumentParser:
    """Komut satırı argümanlarını tanımlar."""

    parser = argparse.ArgumentParser(
        description="FinAi Foundry Local istemcisini test eder."
    )

    parser.add_argument(
        "question",
        nargs="?",
        default="Enflasyon nedir?",
        help="Yerel modele yöneltilecek test sorusu.",
    )

    parser.add_argument(
        "--model",
        default=MODEL_ALIAS,
        help=f"Foundry Local model alias değeri. Varsayılan: {MODEL_ALIAS}",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Üretim temperature değeri. Varsayılan: {DEFAULT_TEMPERATURE}",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help=f"Üretim top_p değeri. Varsayılan: {DEFAULT_TOP_P}",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maksimum token sayısı. Varsayılan: {DEFAULT_MAX_TOKENS}",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="DEBUG loglarını gösterir.",
    )

    return parser


def main() -> int:
    """Komut satırı giriş noktası."""

    parser = build_argument_parser()
    args = parser.parse_args()

    configure_logging(verbose=args.verbose)

    try:
        settings = GenerationSettings(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )

        client = FoundryClient(
            model_alias=args.model,
            settings=settings,
        )

        messages = build_test_messages(args.question)

        print()
        print("=" * 72)
        print("FinAi — Foundry Local Testi")
        print("=" * 72)
        print(f"Model : {client.model_alias}")
        print(f"Soru  : {args.question}")
        print("-" * 72)
        print()

        response_parts: list[str] = []

        for text in client.stream(messages):
            print(text, end="", flush=True)
            response_parts.append(text)

        print()
        print()

        full_response = "".join(response_parts).strip()

        if not full_response:
            raise RuntimeError("Model boş yanıt döndürdü.")

        print("-" * 72)
        print("TEST BAŞARILI")
        print("=" * 72)

        client.close()

    except KeyboardInterrupt:
        LOGGER.warning("Foundry Local testi kullanıcı tarafından durduruldu.")
        return 130
    except Exception:
        LOGGER.exception("Foundry Local testi başarısız oldu.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())