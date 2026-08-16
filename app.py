"""
FinAi — Local Financial Literacy RAG  |  Streamlit UI

Run:
    streamlit run app.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import streamlit as st

from src.rag import FinAiRAG, RAGResponse


LOGGER = logging.getLogger("finai.app")

APP_TITLE    = "FinAi"
APP_ICON     = "💎"
MAX_QUESTION_LENGTH = 2_000

# ---------------------------------------------------------------------------
# Translations — add every UI string here (TR default, EN alternate)
# ---------------------------------------------------------------------------

TRANSLATIONS: dict[str, dict[str, str]] = {
    "tr": {
        # page / header
        "page_title":        "FinAi — Yerel Finansal Okuryazarlık RAG",
        "app_subtitle":      "Yerel Finansal Okuryazarlık Asistanı",
        "app_tagline":       "Kaynak temelli · Tamamen yerel · Yapay zeka destekli",
        "disclaimer":        (
            "FinAi yalnızca eğitim amaçlı genel bilgi sağlar. "
            "Kişiye özel yatırım, kredi, vergi veya hukuki tavsiye vermez."
        ),
        # sidebar
        "sidebar_caption":   "Yerel Finansal Okuryazarlık RAG",
        "lang_toggle":       "🌐 English",
        "system_status":     "Sistem Durumu",
        "rag_ready":         "RAG sistemi hazır",
        "collection_empty":  "Vektör koleksiyonu boş",
        "indexed_chunks":    "İndekslenmiş metin parçası",
        "local_model":       "Yerel model",
        "model_ready":       "Bellekte hazır",
        "model_loading":     "İlk soruda yüklenecek",
        "retrieval_settings":"Retrieval Ayarları",
        "max_sources":       "Maksimum kaynak sayısı",
        "max_sources_help":  "ChromaDB'den alınacak maksimum belge parçası sayısı.",
        "min_score":         "Minimum benzerlik skoru",
        "min_score_help":    "Bu değerin altında kalan retrieval sonuçları kullanılmaz.",
        "example_qs":        "Örnek Sorular",
        "clear_history":     "Sohbeti temizle",
        "re_index_note":     "Belgeleri değiştirdikten sonra terminalde `python -m src.ingest` komutunu yeniden çalıştırın.",
        # empty state
        "how_to_use":        "Nasıl kullanılır?",
        "step1_title":       "Soru Sor",
        "step1_desc":        "Finansal okuryazarlıkla ilgili açık ve kısa bir soru yazın.",
        "step2_title":       "Kaynaklar Taransın",
        "step2_desc":        "Sistem yerel ChromaDB koleksiyonunda en ilgili metinleri bulur.",
        "step3_title":       "Kaynaklı Cevap Al",
        "step3_desc":        "Foundry Local modeli yalnızca bulunan bağlama dayanarak cevap üretir.",
        "example_label":     "Örnek soru",
        "example_q":         "Bileşik faiz nedir?",
        # chat
        "input_placeholder": "Finansal okuryazarlık sorunuzu yazın...",
        "scanning":          "Yerel kaynaklar taranıyor...",
        "step_embed":        "Soru embedding vektörüne dönüştürülüyor.",
        "step_search":       "ChromaDB üzerinde ilgili kaynaklar aranıyor.",
        "step_generate":     "Foundry Local cevabı hazırlanıyor.",
        "answer_ready":      "Cevap hazır",
        # metrics
        "metric_sources":    "Kaynak",
        "metric_time":       "Üretim süresi",
        "metric_fallback":   "Güvenli yedek",
        "fallback_used":     "Kullanıldı",
        "fallback_not_used": "Kullanılmadı",
        "fallback_caption":  (
            "Yerel model çıktısı kalite kontrolünden geçmediği için cevap "
            "doğrudan bulunan kaynak metinlerinden oluşturuldu."
        ),
        "sec":               "sn",
        # sources expander
        "sources_expander":  "Kullanılan kaynaklar",
        "context_expander":  "Modele gönderilen RAG context'i",
        "similarity_score":  "Benzerlik skoru",
        "no_source_warn":    "Bu cevap için uygun kaynak bulunamadı.",
        "no_context":        "Context oluşturulmadı.",
        # errors
        "startup_error":     (
            "FinAi başlatılamadı. ChromaDB koleksiyonunun oluşturulduğunu "
            "ve Foundry Local kurulumunun çalıştığını kontrol edin."
        ),
        "empty_col_error":   (
            "ChromaDB koleksiyonunda belge bulunmuyor. Önce `documents` "
            "klasörüne TXT veya PDF ekleyin ve `python -m src.ingest` komutunu çalıştırın."
        ),
        "process_error":     "Soru işlenirken bir hata oluştu. Terminal loglarını kontrol edin.",
        "tech_error":        "Teknik hata ayrıntısı",
        "validation_empty":  "Lütfen bir soru yazın.",
        "validation_long":   "Soru en fazla {max} karakter olabilir.",
    },
    "en": {
        # page / header
        "page_title":        "FinAi — Local Financial Literacy RAG",
        "app_subtitle":      "Local Financial Literacy Assistant",
        "app_tagline":       "Source-grounded · Fully local · AI-powered",
        "disclaimer":        (
            "FinAi provides general educational information only. "
            "It does not give personalized investment, credit, tax, or legal advice."
        ),
        # sidebar
        "sidebar_caption":   "Local Financial Literacy RAG",
        "lang_toggle":       "🌐 Türkçe",
        "system_status":     "System Status",
        "rag_ready":         "RAG system ready",
        "collection_empty":  "Vector collection is empty",
        "indexed_chunks":    "Indexed text chunks",
        "local_model":       "Local model",
        "model_ready":       "Loaded in memory",
        "model_loading":     "Will load on first query",
        "retrieval_settings":"Retrieval Settings",
        "max_sources":       "Maximum source count",
        "max_sources_help":  "Maximum document chunks to fetch from ChromaDB.",
        "min_score":         "Minimum similarity score",
        "min_score_help":    "Retrieval results below this threshold are discarded.",
        "example_qs":        "Example Questions",
        "clear_history":     "Clear chat",
        "re_index_note":     "After changing documents, re-run `python -m src.ingest` in the terminal.",
        # empty state
        "how_to_use":        "How to use?",
        "step1_title":       "Ask a Question",
        "step1_desc":        "Write a clear, concise question about financial literacy.",
        "step2_title":       "Sources Are Scanned",
        "step2_desc":        "The system finds the most relevant texts in the local ChromaDB collection.",
        "step3_title":       "Get a Sourced Answer",
        "step3_desc":        "The Foundry Local model generates an answer based only on retrieved context.",
        "example_label":     "Example question",
        "example_q":         "What is compound interest?",
        # chat
        "input_placeholder": "Type your financial literacy question...",
        "scanning":          "Scanning local sources...",
        "step_embed":        "Converting question to embedding vector.",
        "step_search":       "Searching ChromaDB for relevant sources.",
        "step_generate":     "Preparing Foundry Local answer.",
        "answer_ready":      "Answer ready",
        # metrics
        "metric_sources":    "Sources",
        "metric_time":       "Generation time",
        "metric_fallback":   "Safe fallback",
        "fallback_used":     "Used",
        "fallback_not_used": "Not used",
        "fallback_caption":  (
            "The local model output failed quality control; "
            "the answer was built directly from retrieved source texts."
        ),
        "sec":               "s",
        # sources expander
        "sources_expander":  "Sources used",
        "context_expander":  "RAG context sent to model",
        "similarity_score":  "Similarity score",
        "no_source_warn":    "No suitable source found for this answer.",
        "no_context":        "No context was generated.",
        # errors
        "startup_error":     (
            "FinAi could not start. Check that the ChromaDB collection exists "
            "and Foundry Local is installed."
        ),
        "empty_col_error":   (
            "No documents in ChromaDB collection. Add TXT or PDF files to the "
            "`documents` folder and run `python -m src.ingest`."
        ),
        "process_error":     "An error occurred while processing the question. Check the terminal logs.",
        "tech_error":        "Technical error details",
        "validation_empty":  "Please enter a question.",
        "validation_long":   "Question must be at most {max} characters.",
    },
}

EXAMPLE_QUESTIONS_TR = (
    "Enflasyon nedir?",
    "Bileşik faiz nedir?",
    "Acil durum fonu neden önemlidir?",
    "Yatırım fonu nedir?",
    "Risk ve getiri arasındaki ilişki nedir?",
)

EXAMPLE_QUESTIONS_EN = (
    "What is inflation?",
    "What is compound interest?",
    "Why is an emergency fund important?",
    "What is a mutual fund?",
    "What is the relationship between risk and return?",
)


# ---------------------------------------------------------------------------
# Translation helper
# ---------------------------------------------------------------------------

def t(key: str) -> str:
    """Return the translated string for the active language."""
    lang = st.session_state.get("lang", "tr")
    return TRANSLATIONS[lang].get(key, key)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ChatRecord:
    """A single Q&A record stored in chat history."""
    question: str
    response: RAGResponse


# ---------------------------------------------------------------------------
# Page config  (called ONCE before any other st.* call)
# ---------------------------------------------------------------------------

def configure_page() -> None:
    lang = st.session_state.get("lang", "tr")
    st.set_page_config(
        page_title=TRANSLATIONS[lang]["page_title"],
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ---------------------------------------------------------------------------
# CSS — premium dark design
# ---------------------------------------------------------------------------

def apply_custom_styles() -> None:
    st.markdown(
        """
        <style>
        /* ── Google Font ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* ── Root tokens ── */
        :root {
            --bg:          #080d1a;
            --surface:     rgba(255,255,255,0.04);
            --surface-2:   rgba(255,255,255,0.07);
            --border:      rgba(255,255,255,0.08);
            --border-2:    rgba(255,255,255,0.14);
            --teal:        #00d4aa;
            --indigo:      #6366f1;
            --text:        #e8eaf0;
            --text-muted:  rgba(232,234,240,0.55);
            --radius:      16px;
            --radius-sm:   10px;
        }

        /* ── Base ── */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
            background-color: var(--bg) !important;
            color: var(--text) !important;
        }

        .main .block-container {
            max-width: 1140px;
            padding-top: 1.6rem;
            padding-bottom: 8rem;
        }

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d1526 0%, #080d1a 100%) !important;
            border-right: 1px solid var(--border) !important;
        }
        section[data-testid="stSidebar"] * {
            color: var(--text) !important;
        }

        /* ── Header card ── */
        .finai-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: linear-gradient(135deg,
                rgba(0,212,170,0.08) 0%,
                rgba(99,102,241,0.08) 100%);
            border: 1px solid var(--border-2);
            border-radius: var(--radius);
            padding: 1.5rem 2rem;
            margin-bottom: 1.4rem;
            backdrop-filter: blur(12px);
        }
        .finai-header-left { display: flex; flex-direction: column; gap: 0.3rem; }
        .finai-wordmark {
            font-size: 2.1rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            background: linear-gradient(90deg, var(--teal), var(--indigo));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1;
        }
        .finai-subtitle {
            font-size: 0.92rem;
            color: var(--text-muted);
            font-weight: 400;
            margin: 0;
        }
        .finai-tagline {
            font-size: 0.78rem;
            color: var(--text-muted);
            font-weight: 400;
            opacity: 0.7;
            margin: 0;
        }
        .finai-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.3rem 0.85rem;
            border-radius: 99px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            background: rgba(0,212,170,0.12);
            border: 1px solid rgba(0,212,170,0.25);
            color: var(--teal);
            align-self: flex-start;
            margin-top: 0.4rem;
        }

        /* ── Disclaimer banner ── */
        .finai-disclaimer {
            background: rgba(99,102,241,0.07);
            border: 1px solid rgba(99,102,241,0.2);
            border-radius: var(--radius-sm);
            padding: 0.75rem 1.1rem;
            font-size: 0.83rem;
            color: var(--text-muted);
            margin-bottom: 1.4rem;
        }

        /* ── Empty state steps ── */
        .step-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.4rem 1.2rem;
            transition: border-color 0.2s, transform 0.2s;
        }
        .step-card:hover {
            border-color: var(--border-2);
            transform: translateY(-2px);
        }
        .step-number {
            width: 32px; height: 32px;
            border-radius: 8px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.9rem;
            margin-bottom: 0.75rem;
        }
        .step-number-1 { background: rgba(0,212,170,0.15); color: var(--teal); }
        .step-number-2 { background: rgba(99,102,241,0.15); color: var(--indigo); }
        .step-number-3 { background: rgba(0,212,170,0.1); color: var(--teal); }
        .step-title { font-size: 1rem; font-weight: 600; margin: 0 0 0.4rem; }
        .step-desc  { font-size: 0.85rem; color: var(--text-muted); margin: 0; }
        .example-pill {
            display: inline-block;
            background: var(--surface-2);
            border: 1px solid var(--border-2);
            border-radius: 8px;
            padding: 0.6rem 1rem;
            font-size: 0.9rem;
            font-family: 'Courier New', monospace;
            color: var(--teal);
            margin-top: 1rem;
        }

        /* ── Chat messages ── */
        div[data-testid="stChatMessage"] {
            background: transparent !important;
            border-radius: var(--radius) !important;
        }
        div[data-testid="stChatMessage"][data-message-author-role="user"] {
            background: rgba(99,102,241,0.06) !important;
            border: 1px solid rgba(99,102,241,0.12) !important;
            padding: 0.4rem 0.8rem;
        }
        div[data-testid="stChatMessage"][data-message-author-role="assistant"] {
            background: rgba(0,212,170,0.04) !important;
            border: 1px solid rgba(0,212,170,0.1) !important;
            padding: 0.4rem 0.8rem;
        }

        /* ── Answer container ── */
        .answer-box {
            background: var(--surface);
            border: 1px solid var(--border-2);
            border-radius: var(--radius-sm);
            padding: 1rem 1.2rem;
            margin-bottom: 0.75rem;
            line-height: 1.7;
        }

        /* ── Metric chips ── */
        .metric-row {
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin-bottom: 0.75rem;
        }
        .metric-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.3rem 0.8rem;
            border-radius: 99px;
            font-size: 0.78rem;
            font-weight: 500;
            background: var(--surface-2);
            border: 1px solid var(--border);
            color: var(--text-muted);
        }
        .metric-chip b { color: var(--text); font-weight: 600; }
        .metric-chip-teal { border-color: rgba(0,212,170,0.3); }
        .metric-chip-teal b { color: var(--teal); }
        .metric-chip-fallback-yes { border-color: rgba(245,158,11,0.35);
            background: rgba(245,158,11,0.07); }
        .metric-chip-fallback-yes b { color: #f59e0b; }

        /* ── Source cards in expander ── */
        .source-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            padding: 0.9rem 1rem;
            margin-bottom: 0.6rem;
        }
        .source-label {
            font-size: 0.82rem;
            font-weight: 600;
            color: var(--teal);
            margin-bottom: 0.3rem;
        }
        .source-citation {
            font-size: 0.82rem;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }
        .source-score {
            display: inline-block;
            background: rgba(0,212,170,0.1);
            border: 1px solid rgba(0,212,170,0.2);
            color: var(--teal);
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.15rem 0.55rem;
            border-radius: 99px;
            margin-bottom: 0.5rem;
        }
        .source-text {
            font-size: 0.85rem;
            color: var(--text-muted);
            line-height: 1.6;
        }

        /* ── Expanders ── */
        details summary {
            font-size: 0.88rem !important;
            font-weight: 500 !important;
            color: var(--text-muted) !important;
        }
        details summary:hover { color: var(--text) !important; }

        /* ── Sliders ── */
        div[data-testid="stSlider"] > div > div > div {
            background: linear-gradient(90deg, var(--teal), var(--indigo)) !important;
        }
        div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
            background: var(--teal) !important;
            border-color: var(--teal) !important;
        }

        /* ── Buttons ── */
        div[data-testid="stSidebar"] button[kind="secondary"] {
            background: var(--surface) !important;
            border: 1px solid var(--border-2) !important;
            color: var(--text) !important;
            border-radius: var(--radius-sm) !important;
            font-size: 0.85rem !important;
            transition: background 0.15s, border-color 0.15s !important;
        }
        div[data-testid="stSidebar"] button[kind="secondary"]:hover {
            background: var(--surface-2) !important;
            border-color: rgba(0,212,170,0.4) !important;
        }

        /* ── Chat input ── */
        div[data-testid="stChatInput"] textarea {
            background: var(--surface-2) !important;
            border: 1px solid var(--border-2) !important;
            border-radius: var(--radius) !important;
            color: var(--text) !important;
            font-size: 0.95rem !important;
        }
        div[data-testid="stChatInput"] textarea::placeholder {
            color: var(--text-muted) !important;
        }
        div[data-testid="stChatInput"] button {
            background: linear-gradient(135deg, var(--teal), var(--indigo)) !important;
            border-radius: 12px !important;
            border: none !important;
        }

        /* ── Divider ── */
        hr { border-color: var(--border) !important; }

        /* ── Metric widget override ── */
        div[data-testid="stMetricValue"] {
            font-size: 1.6rem !important;
            font-weight: 700 !important;
            color: var(--text) !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.78rem !important;
            color: var(--text-muted) !important;
        }

        /* ── Status box ── */
        div[data-testid="stStatusWidget"] {
            background: var(--surface) !important;
            border: 1px solid var(--border-2) !important;
            border-radius: var(--radius-sm) !important;
        }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb {
            background: rgba(255,255,255,0.1);
            border-radius: 99px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# RAG service (cached — loaded once per process)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_rag_service() -> FinAiRAG:
    """Load FinAiRAG once for the lifetime of the Streamlit process."""
    LOGGER.info("FinAi: FinAiRAG service initialising.")
    return FinAiRAG()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def initialize_session_state() -> None:
    defaults: dict = {
        "chat_history":    [],
        "pending_question": None,
        "top_k":           4,
        "min_score":       0.55,
        "lang":            "tr",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Validation (unchanged logic)
# ---------------------------------------------------------------------------

def validate_question(question: str) -> str:
    if not isinstance(question, str):
        raise TypeError(t("validation_empty"))
    normalized = " ".join(question.split()).strip()
    if not normalized:
        raise ValueError(t("validation_empty"))
    if len(normalized) > MAX_QUESTION_LENGTH:
        raise ValueError(t("validation_long").format(max=MAX_QUESTION_LENGTH))
    return normalized


def clear_chat_history() -> None:
    st.session_state.chat_history    = []
    st.session_state.pending_question = None


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def render_header() -> None:
    st.markdown(
        f"""
        <div class="finai-header">
            <div class="finai-header-left">
                <div class="finai-wordmark">{APP_ICON} {APP_TITLE}</div>
                <p class="finai-subtitle">{t("app_subtitle")}</p>
                <p class="finai-tagline">{t("app_tagline")}</p>
            </div>
        </div>
        <div class="finai-disclaimer">ℹ️ &nbsp;{t("disclaimer")}</div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(rag: FinAiRAG) -> None:
    with st.sidebar:
        # ── Brand ──
        st.markdown(
            f"""
            <div style="padding:0.5rem 0 0.2rem;">
                <span style="font-size:1.5rem;font-weight:800;
                    background:linear-gradient(90deg,#00d4aa,#6366f1);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;">
                    {APP_ICON} {APP_TITLE}
                </span>
                <div style="font-size:0.75rem;opacity:0.5;margin-top:2px;">
                    {t("sidebar_caption")}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Language toggle ──
        if st.button(t("lang_toggle"), key="lang_toggle_btn", use_container_width=True):
            st.session_state.lang = "en" if st.session_state.lang == "tr" else "tr"
            st.rerun()

        st.divider()

        # ── System status ──
        st.markdown(
            f"<div style='font-size:0.8rem;font-weight:600;opacity:0.5;"
            f"text-transform:uppercase;letter-spacing:0.06em;"
            f"margin-bottom:0.5rem;'>{t('system_status')}</div>",
            unsafe_allow_html=True,
        )

        try:
            document_count = rag.retriever.document_count
        except Exception:
            LOGGER.exception("Could not read collection count.")
            document_count = 0

        if document_count > 0:
            st.success(t("rag_ready"))
        else:
            st.error(t("collection_empty"))

        st.metric(t("indexed_chunks"), document_count)

        model_status = (
            t("model_ready")
            if rag.foundry_client.is_ready
            else t("model_loading")
        )
        st.caption(f"{t('local_model')}: {model_status}")

        st.divider()

        # ── Retrieval settings ──
        st.markdown(
            f"<div style='font-size:0.8rem;font-weight:600;opacity:0.5;"
            f"text-transform:uppercase;letter-spacing:0.06em;"
            f"margin-bottom:0.5rem;'>{t('retrieval_settings')}</div>",
            unsafe_allow_html=True,
        )

        st.session_state.top_k = st.slider(
            t("max_sources"),
            min_value=1,
            max_value=10,
            value=int(st.session_state.top_k),
            step=1,
            help=t("max_sources_help"),
        )

        st.session_state.min_score = st.slider(
            t("min_score"),
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.min_score),
            step=0.05,
            help=t("min_score_help"),
        )

        st.divider()

        # ── Example questions ──
        st.markdown(
            f"<div style='font-size:0.8rem;font-weight:600;opacity:0.5;"
            f"text-transform:uppercase;letter-spacing:0.06em;"
            f"margin-bottom:0.5rem;'>{t('example_qs')}</div>",
            unsafe_allow_html=True,
        )

        example_list = (
            EXAMPLE_QUESTIONS_EN
            if st.session_state.lang == "en"
            else EXAMPLE_QUESTIONS_TR
        )

        for idx, q in enumerate(example_list, start=1):
            if st.button(q, key=f"example_q_{idx}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

        st.divider()

        if st.button(t("clear_history"), use_container_width=True):
            clear_chat_history()
            st.rerun()

        st.caption(t("re_index_note"))


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

def render_empty_state() -> None:
    st.markdown(
        f"<h3 style='font-weight:700;margin-bottom:1.2rem;'>"
        f"{t('how_to_use')}</h3>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="step-card">
                <div class="step-number step-number-1">1</div>
                <p class="step-title">{t("step1_title")}</p>
                <p class="step-desc">{t("step1_desc")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="step-card">
                <div class="step-number step-number-2">2</div>
                <p class="step-title">{t("step2_title")}</p>
                <p class="step-desc">{t("step2_desc")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="step-card">
                <div class="step-number step-number-3">3</div>
                <p class="step-title">{t("step3_title")}</p>
                <p class="step-desc">{t("step3_desc")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div style="margin-top:1.4rem;">
            <div style="font-size:0.78rem;font-weight:600;opacity:0.5;
                text-transform:uppercase;letter-spacing:0.06em;
                margin-bottom:0.5rem;">{t("example_label")}</div>
            <span class="example-pill">💬 &nbsp;{t("example_q")}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Metrics (rendered as custom HTML chips instead of st.metric)
# ---------------------------------------------------------------------------

def render_response_metrics(response: RAGResponse) -> None:
    fallback_class = (
        "metric-chip metric-chip-fallback-yes"
        if response.used_fallback
        else "metric-chip"
    )
    fallback_val = (
        t("fallback_used")
        if response.used_fallback
        else t("fallback_not_used")
    )

    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-chip metric-chip-teal">
                📄 {t("metric_sources")}: <b>{response.source_count}</b>
            </div>
            <div class="metric-chip">
                ⚡ {t("metric_time")}: <b>{response.generation_seconds:.2f} {t("sec")}</b>
            </div>
            <div class="{fallback_class}">
                🛡 {t("metric_fallback")}: <b>{fallback_val}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if response.used_fallback:
        st.caption(t("fallback_caption"))


# ---------------------------------------------------------------------------
# Source details
# ---------------------------------------------------------------------------

def render_source_details(response: RAGResponse) -> None:
    if not response.sources:
        st.warning(t("no_source_warn"))
        return

    with st.expander(
        f"📚 {t('sources_expander')} ({response.source_count})",
        expanded=False,
    ):
        for source in response.sources:
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-label">[{t("metric_sources")} {source.index}]</div>
                    <div class="source-citation">{source.citation}</div>
                    <span class="source-score">
                        {t("similarity_score")}: {source.score:.4f}
                    </span>
                    <div class="source-text">{source.text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander(f"🔍 {t('context_expander')}", expanded=False):
        st.code(
            response.context or t("no_context"),
            language="text",
        )


# ---------------------------------------------------------------------------
# Chat record
# ---------------------------------------------------------------------------

def render_chat_record(record: ChatRecord) -> None:
    with st.chat_message("user"):
        st.markdown(record.question)

    with st.chat_message("assistant", avatar=APP_ICON):
        st.markdown(
            f"<div class='answer-box'>{record.response.answer}</div>",
            unsafe_allow_html=True,
        )
        render_response_metrics(record.response)
        render_source_details(record.response)


def render_chat_history() -> None:
    for record in st.session_state.chat_history:
        render_chat_record(record)


# ---------------------------------------------------------------------------
# RAG logic (unchanged)
# ---------------------------------------------------------------------------

def generate_response(rag: FinAiRAG, question: str) -> RAGResponse:
    return rag.answer(
        question,
        use_llm=True,
        top_k=int(st.session_state.top_k),
        min_score=float(st.session_state.min_score),
    )


def process_and_store_question(rag: FinAiRAG, raw_question: str) -> None:
    """
    Validate, send to RAG, store in history.
    Does NOT render directly — page is re-run after storage so each
    record renders exactly once.
    """
    question = validate_question(raw_question)

    with st.status(t("scanning"), expanded=True) as status:
        st.write(t("step_embed"))
        st.write(t("step_search"))
        st.write(t("step_generate"))

        response = generate_response(rag=rag, question=question)

        status.update(label=t("answer_ready"), state="complete", expanded=False)

    st.session_state.chat_history.append(
        ChatRecord(question=question, response=response)
    )


def handle_pending_question(rag: FinAiRAG) -> None:
    pending = st.session_state.pending_question
    if not pending:
        return

    st.session_state.pending_question = None

    try:
        process_and_store_question(rag=rag, raw_question=pending)
    except Exception as exc:
        LOGGER.exception("Could not process example question.")
        st.error(t("process_error"))
        with st.expander(t("tech_error")):
            st.code(str(exc), language="text")
        return

    st.rerun()


def handle_chat_input(rag: FinAiRAG) -> None:
    user_question = st.chat_input(
        t("input_placeholder"),
        max_chars=MAX_QUESTION_LENGTH,
    )

    if not user_question:
        return

    try:
        process_and_store_question(rag=rag, raw_question=user_question)
    except (TypeError, ValueError) as exc:
        st.warning(str(exc))
        return
    except Exception as exc:
        LOGGER.exception("Streamlit RAG question could not be processed.")
        st.error(t("process_error"))
        with st.expander(t("tech_error")):
            st.code(str(exc), language="text")
        return

    st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Session state must be initialised before configure_page reads lang
    initialize_session_state()
    configure_page()
    apply_custom_styles()
    render_header()

    try:
        rag = load_rag_service()
    except Exception as exc:
        LOGGER.exception("FinAiRAG service could not start.")
        st.error(t("startup_error"))
        with st.expander(t("tech_error")):
            st.code(str(exc), language="text")
        st.stop()

    render_sidebar(rag)

    if not rag.is_ready:
        st.error(t("empty_col_error"))
        st.stop()

    handle_pending_question(rag)

    if st.session_state.chat_history:
        render_chat_history()
    else:
        render_empty_state()

    handle_chat_input(rag)


if __name__ == "__main__":
    main()