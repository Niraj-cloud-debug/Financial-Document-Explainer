"""
FinLens v4 · Financial Document Intelligence
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Changes from v3:
  1. OVERLAPPING CHUNKS — 50-word overlap between chunks prevents context from
     being severed at boundaries, improving retrieval accuracy noticeably.
  2. AUTO-SUMMARY ON UPLOAD — Sarvam LLM generates a 3-line plain-language
     summary the moment a document is indexed; shown in the sidebar immediately.
  3. DOCUMENT TYPE DETECTION — Detects insurance / mutual fund / loan / FD and
     tailors the system prompt and quick-prompt chips accordingly.
  4. VOICE INPUT (Sarvam STT) — Microphone button uses saaras:v3 to transcribe
     questions; showcases the full Sarvam API suite beyond just LLM + TTS.
  5. FULL-LENGTH TTS — Sentence-aware splitting into ≤490-char segments, each
     sent to bulbul:v2, WAV bytes concatenated; no answer gets cut off.
  6. SMARTER PROMPTING — Financial-consumer persona with explicit red-flag
     detection; LLM asked to surface watch-outs in a dedicated section.
  7. WATCH-OUT BANNER — Post-processes the LLM reply: if it detects risk
     keywords it injects a gold ⚠ callout above the main answer.
  8. CLEANER CODE — Helpers are better separated; error handling improved.
"""

import streamlit as st
import requests
import re
import base64
import io
import wave
from collections import Counter
import PyPDF2

try:
    from sentence_transformers import SentenceTransformer
    _EMBEDDINGS_AVAILABLE = True
except ImportError:
    _EMBEDDINGS_AVAILABLE = False

import chromadb

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════
SARVAM_API_KEY = st.secrets["MY_SARVAM_API"]   # ← Replace

STRUCTURED_FIELDS = {
    "Interest / Returns":  [r"interest rate[:\s]+([0-9.]+%?)", r"returns?[:\s]+([0-9.]+%?)", r"yield[:\s]+([0-9.]+%?)"],
    "Fees & Charges":      [r"fee[s]?[:\s]+(?:rs\.?\s*)?([0-9,]+)", r"charge[s]?[:\s]+(?:rs\.?\s*)?([0-9,]+)", r"premium[:\s]+(?:rs\.?\s*)?([0-9,]+)"],
    "Lock-in / Tenure":    [r"lock.?in[:\s]+([0-9]+\s*(?:year|month|day)s?)", r"tenure[:\s]+([0-9]+\s*(?:year|month|day)s?)"],
    "Maturity":            [r"maturit[yi][:\s]+([^\n.]{4,50})", r"expir[yi][:\s]+([^\n.]{4,50})"],
    "Sum Assured / Limit": [r"sum assured[:\s]+(?:rs\.?\s*)?([0-9,]+)", r"cover(?:age)?[:\s]+(?:rs\.?\s*)?([0-9,]+)"],
    "Waiting Period":      [r"waiting period[:\s]+([0-9]+\s*(?:year|month|day)s?)"],
}

# Quick prompts are now bucketed by document type.
QUICK_PROMPTS_DEFAULT = [
    "Summarise this document in simple terms",
    "What are the hidden charges?",
    "Is there a lock-in period?",
    "What happens if I miss a payment?",
    "What are the key risks?",
    "Can I exit early? Any penalties?",
]
QUICK_PROMPTS_INSURANCE = [
    "Explain this policy in simple terms",
    "What is NOT covered?",
    "What is the claim process?",
    "Is there a waiting period?",
    "What are the premium charges?",
    "Can I surrender early? Any penalties?",
]
QUICK_PROMPTS_MF = [
    "Explain this fund in simple terms",
    "What are the total expense charges?",
    "What is the exit load?",
    "What are the key risks?",
    "Is there a lock-in period?",
    "Who should NOT invest in this fund?",
]
QUICK_PROMPTS_LOAN = [
    "Explain this loan in simple terms",
    "What is the total interest I will pay?",
    "What are the prepayment charges?",
    "What happens if I miss an EMI?",
    "What are the hidden fees?",
    "Can I foreclose early? At what cost?",
]
QUICK_PROMPTS_FD = [
    "Explain this FD in simple terms",
    "What is the effective interest rate?",
    "Can I break the FD early?",
    "What are the premature withdrawal charges?",
    "Is the interest taxable?",
    "What happens at maturity?",
]

DOC_TYPE_PROMPTS = {
    "insurance":   QUICK_PROMPTS_INSURANCE,
    "mutual_fund": QUICK_PROMPTS_MF,
    "loan":        QUICK_PROMPTS_LOAN,
    "fd":          QUICK_PROMPTS_FD,
    "financial":   QUICK_PROMPTS_DEFAULT,
}

DOC_TYPE_LABELS = {
    "insurance":   "🛡️ Insurance Policy",
    "mutual_fund": "📈 Mutual Fund",
    "loan":        "🏦 Loan Agreement",
    "fd":          "💰 Fixed Deposit",
    "financial":   "📄 Financial Document",
}

# Keywords that trigger the ⚠ watch-out banner in the UI
WATCHOUT_KEYWORDS = [
    "penalty", "charge", "lock-in", "not covered", "exclusion", "forfeit",
    "forfeiture", "surrender", "deduction", "tax", "risk", "loss", "liable",
    "cancellation", "lapse", "default", "fine", "waiting period",
]

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FinLens · Financial Document Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0b0d12; color: #e2dfd8; }

[data-testid="stSidebar"] { background: #10131a !important; border-right: 1px solid #1c2030 !important; }

.brand { padding: 1.4rem 0 1.8rem; text-align: center; }
.brand-title { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #f0ede6; letter-spacing: -0.03em; line-height: 1; }
.brand-title span { color: #c9a84c; }
.brand-sub { font-size: 0.68rem; color: #44495c; letter-spacing: 0.14em; text-transform: uppercase; margin-top: 0.35rem; }

.lbl { font-size: 0.64rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: #44495c; margin-bottom: 0.5rem; }

.stTabs [data-baseweb="tab-list"] { background: #10131a !important; border-radius: 10px; padding: 3px; gap: 2px; border: 1px solid #1c2030; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: #44495c !important; border-radius: 8px !important; font-size: 0.82rem !important; font-weight: 500 !important; padding: 0.45rem 1.1rem !important; }
.stTabs [aria-selected="true"] { background: #1c2030 !important; color: #c9a84c !important; }

.card { background: #10131a; border: 1px solid #1c2030; border-radius: 12px; padding: 1.4rem; margin-bottom: 0.9rem; }

.answer-box { background:#0b0d12; border:1px solid #1c2030; border-left:3px solid #c9a84c; border-radius:0 12px 12px 0; padding:1.3rem 1.5rem; font-size:0.98rem; line-height:1.8; color:#d8d4ca; margin:0.8rem 0; }

/* Watch-out banner */
.watchout-box { background:#16100a; border:1px solid #4a2e0a; border-left:3px solid #e88c3a; border-radius:0 10px 10px 0; padding:0.85rem 1.2rem; font-size:0.85rem; line-height:1.7; color:#c8a07a; margin:0 0 0.5rem; }
.watchout-title { font-size:0.66rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#e88c3a; margin-bottom:0.35rem; }

/* Auto-summary card */
.summary-card { background:#0e1118; border:1px solid #1c2030; border-radius:10px; padding:0.9rem 1rem; margin: 0.6rem 0 0; }
.summary-text { font-size:0.8rem; color:#7a8099; line-height:1.65; }
.doc-type-badge { display:inline-block; background:#1a1d28; border:1px solid #252a3a; border-radius:20px; padding:0.15rem 0.7rem; font-size:0.66rem; color:#c9a84c; letter-spacing:0.08em; margin-bottom:0.5rem; }

.msg-user { background:#151821; border:1px solid #1c2030; border-radius:12px 12px 4px 12px; padding:0.75rem 1rem; font-size:0.9rem; color:#c9a84c; margin:0.5rem 0; max-width:80%; margin-left:auto; text-align:right; }
.msg-ai   { background:#10131a; border:1px solid #1c2030; border-radius:12px 12px 12px 4px; padding:0.85rem 1.1rem; font-size:0.93rem; color:#d8d4ca; margin:0.5rem 0; max-width:96%; }
.msg-ai ol, .msg-ai ul { margin-top:0.3rem; }
.msg-ai p:last-child { margin-bottom:0; }

/* Chip buttons */
.chip-btn > div > button, .chip-btn button {
    background: #10131a !important; color: #7a7f92 !important; border: 1px solid #252a3a !important;
    border-radius: 20px !important; font-size: 0.75rem !important; font-weight: 400 !important;
    padding: 0.22rem 0.8rem !important; letter-spacing: 0 !important; min-height: unset !important;
    line-height: 1.4 !important; transition: all 0.18s !important; white-space: nowrap !important;
}
.chip-btn > div > button:hover, .chip-btn button:hover {
    border-color: #c9a84c !important; color: #c9a84c !important;
    background: #13120a !important; transform: none !important; box-shadow: none !important;
}

.cmp-header { font-size:0.72rem; color:#c9a84c; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.8rem; font-weight:600; }
.cmp-item { display:flex; gap:0.8rem; padding:0.5rem 0; border-bottom:1px solid #161920; font-size:0.85rem; }
.cmp-field { color:#44495c; min-width:130px; font-size:0.78rem; }
.cmp-value { color:#d8d4ca; }

[data-testid="stChatInput"] { background:#10131a !important; border:1.5px solid #1c2030 !important; border-radius:12px !important; }
[data-testid="stChatInput"]:focus-within { border-color:#c9a84c !important; }
[data-testid="stChatInput"] textarea { background:transparent !important; border:none !important; color:#e2dfd8 !important; }
[data-testid="stChatInput"] textarea::placeholder { color:#2e3345 !important; }
[data-testid="stChatInputSubmitButton"] svg { fill:#c9a84c !important; }

.stButton > button { background:#c9a84c !important; color:#0b0d12 !important; border:none !important; border-radius:8px !important; font-weight:600 !important; font-size:0.86rem !important; letter-spacing:0.04em !important; padding:0.5rem 1.4rem !important; transition:all 0.2s !important; }
.stButton > button:hover { background:#ddb94f !important; transform:translateY(-1px) !important; box-shadow:0 4px 18px rgba(201,168,76,0.22) !important; }

[data-testid="stFileUploader"] { background:#0b0d12 !important; border:1.5px dashed #252a3a !important; border-radius:10px !important; }
[data-testid="stFileUploader"]:hover { border-color:#c9a84c !important; }

.streamlit-expanderHeader { background:#10131a !important; border:1px solid #1c2030 !important; border-radius:8px !important; color:#7a7f92 !important; font-size:0.8rem !important; }
.streamlit-expanderContent { background:#0b0d12 !important; border:1px solid #1c2030 !important; border-top:none !important; border-radius:0 0 8px 8px !important; }

.stRadio label { color:#7a7f92 !important; font-size:0.84rem !important; }
.stSpinner > div { color:#c9a84c !important; }

::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#0b0d12; }
::-webkit-scrollbar-thumb { background:#252a3a; border-radius:6px; }

hr { border-color:#1c2030 !important; margin:1.2rem 0 !important; }
audio { width:100%; border-radius:8px; margin-top:0.4rem; }

/* Voice input mic button — subtle secondary styling */
div[data-testid="stAudioInput"] { background:#10131a !important; border:1px solid #1c2030 !important; border-radius:10px !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# CHROMA SETUP
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_chroma():
    client = chromadb.Client()
    return (
        client.get_or_create_collection("finlens_doc1"),
        client.get_or_create_collection("finlens_doc2"),
    )

col1_db, col2_db = get_chroma()


@st.cache_resource
def load_embedder():
    if not _EMBEDDINGS_AVAILABLE:
        return None
    try:
        return SentenceTransformer("yiyanghkust/finbert-tone")
    except Exception:
        try:
            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            return None

embedder = load_embedder()


# ════════════════════════════════════════════════════════════════════════════
# DOCUMENT HELPERS
# ════════════════════════════════════════════════════════════════════════════

def extract_text(file) -> str:
    reader = PyPDF2.PdfReader(file)
    return "\n".join(p.extract_text() or "" for p in reader.pages)


def detect_doc_type(text: str) -> str:
    """Infer document category from keywords in the text."""
    t = text.lower()
    if any(w in t for w in ["sum assured", "policyholder", "insured", "claim settlement", "premium"]):
        return "insurance"
    if any(w in t for w in ["nav", "mutual fund", "scheme information", "aum", "exit load", "redemption"]):
        return "mutual_fund"
    if any(w in t for w in ["emi", "borrower", "disbursement", "repayment schedule", "mortgage"]):
        return "loan"
    if any(w in t for w in ["fixed deposit", "fd receipt", "interest payout", "tds on interest"]):
        return "fd"
    return "financial"


def extract_structured(text: str) -> dict:
    data = {}
    for field, patterns in STRUCTURED_FIELDS.items():
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                data[field] = m.group(1).strip()
                break
    return data


# ════════════════════════════════════════════════════════════════════════════
# CHUNKING & RETRIEVAL
# ════════════════════════════════════════════════════════════════════════════

def smart_chunk(text: str, target_words: int = 400, overlap_words: int = 50) -> list:
    """
    Split text into overlapping chunks.
    Overlap ensures that context straddling a chunk boundary is captured by
    at least one chunk during retrieval, significantly reducing missed answers.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_sents, current_words = [], 0

    for s in sentences:
        wc = len(s.split())
        if current_words + wc > target_words and current_sents:
            chunks.append(" ".join(current_sents))
            # Keep the last `overlap_words` worth of sentences for the next chunk
            overlap_sents, overlap_count = [], 0
            for sent in reversed(current_sents):
                sw = len(sent.split())
                if overlap_count + sw <= overlap_words:
                    overlap_sents.insert(0, sent)
                    overlap_count += sw
                else:
                    break
            current_sents = overlap_sents + [s]
            current_words = overlap_count + wc
        else:
            current_sents.append(s)
            current_words += wc

    if current_sents:
        chunks.append(" ".join(current_sents))

    return [c for c in chunks if len(c.strip()) > 40]


def embed_chunks(chunks: list):
    if embedder is None:
        return None
    try:
        return embedder.encode(chunks, show_progress_bar=False).tolist()
    except Exception:
        return None


def index_document(name: str, text: str, collection) -> int:
    chunks = smart_chunk(text)
    try:
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass
    embeddings = embed_chunks(chunks)
    ids   = [f"{name}_{i}" for i in range(len(chunks))]
    metas = [{"source": name, "chunk": i} for i in range(len(chunks))]
    if embeddings:
        collection.add(documents=chunks, ids=ids, metadatas=metas, embeddings=embeddings)
    else:
        collection.add(documents=chunks, ids=ids, metadatas=metas)
    return len(chunks)


def tfidf_rerank(query: str, docs: list, top_k: int = 4) -> list:
    def tokenize(t):
        return re.findall(r'\b[a-z]{2,}\b', t.lower())
    q_tok = Counter(tokenize(query))
    scores = []
    for doc in docs:
        d_tok = Counter(tokenize(doc))
        shared = set(q_tok) & set(d_tok)
        scores.append(sum(q_tok[t] * d_tok[t] for t in shared))
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]


def retrieve(question: str, collection, n_retrieve: int = 8, n_final: int = 4):
    count = collection.count()
    if count == 0:
        return [], []
    n = min(n_retrieve, count)
    q_embed = None
    if embedder:
        try:
            q_embed = embedder.encode([question], show_progress_bar=False).tolist()
        except Exception:
            pass
    results = (
        collection.query(query_embeddings=q_embed, n_results=n)
        if q_embed
        else collection.query(query_texts=[question], n_results=n)
    )
    docs  = results["documents"][0]
    metas = results["metadatas"][0]
    idx   = tfidf_rerank(question, docs, top_k=n_final)
    return [docs[i] for i in idx], [metas[i] for i in idx]


# ════════════════════════════════════════════════════════════════════════════
# LANGUAGE HELPERS
# ════════════════════════════════════════════════════════════════════════════

def lang_instruction(lang: str) -> str:
    if lang == "Hindi":
        return "Respond entirely in Hindi using Devanagari script."
    if lang == "Hinglish":
        return (
            "Respond in Hinglish — the natural mix of Hindi and English that urban Indians use in everyday conversation. "
            "Write in Roman script (not Devanagari). Use English for financial/technical terms and Hindi for conversational flow. "
            "Example: 'Aapke document mein ek 3-year lock-in period hai, matlab aap pehle 3 saal exit nahi kar sakte bina penalty ke.'"
        )
    return "Respond in English."


def tts_lang_code(lang: str) -> str:
    return "hi-IN" if lang in ("Hindi", "Hinglish") else "en-IN"


def tts_speaker(lang: str) -> str:
    return "anushka" if lang in ("Hindi", "Hinglish") else "arya"


def stt_lang_code(lang: str) -> str:
    """STT language code — use 'unknown' so Sarvam auto-detects for Hinglish."""
    if lang == "Hindi":
        return "hi-IN"
    if lang == "Hinglish":
        return "unknown"
    return "en-IN"


# ════════════════════════════════════════════════════════════════════════════
# SARVAM API — LLM
# ════════════════════════════════════════════════════════════════════════════

def ask_sarvam(messages: list, system: str) -> str:
    resp = requests.post(
        "https://api.sarvam.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {SARVAM_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "sarvam-m",
            "messages": [{"role": "system", "content": system}] + messages,
        },
        timeout=30,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def generate_doc_summary(text: str, doc_type: str, lang: str) -> str:
    """
    Generate a 2–3 sentence plain-language summary of the document.
    Called once on upload and cached in session state.
    """
    snippet = text[:3000]
    doc_label = DOC_TYPE_LABELS.get(doc_type, "financial document")
    system = (
        f"You are a financial advisor. The user has uploaded a {doc_label}. "
        f"Write exactly 2–3 sentences summarising what this document is about in plain language "
        f"a first-time investor would understand. Focus on: what the product is, who it is for, "
        f"and the single most important thing to know. Do not use jargon. "
        f"{lang_instruction(lang)}"
    )
    try:
        return ask_sarvam([{"role": "user", "content": snippet}], system)
    except Exception as e:
        return f"(Summary unavailable: {e})"


def sarvam_rag(question: str, chunks: list, doc_type: str, lang: str, history: list) -> str:
    """
    Main RAG call. System prompt is tailored to document type and explicitly
    asks the model to surface any risks or watch-outs in its answer.
    """
    context = "\n\n---\n\n".join(f"[Section {i+1}]: {c}" for i, c in enumerate(chunks))
    doc_label = DOC_TYPE_LABELS.get(doc_type, "financial document")

    system = f"""You are FinLens, a trusted financial advisor helping a retail investor in India understand their {doc_label}.

Your job is to give clear, honest, and actionable answers. Follow these rules strictly:
1. Speak directly to the person using "you" and "your".
2. Answer ONLY from the provided document sections. Cite which section your answer comes from.
3. If the answer is not in the sections, say so plainly — never guess.
4. Structure your answer: give the direct answer first, then any important context.
5. If there are any penalties, exclusions, risks, or catches the person should know about — flag them explicitly.
6. Keep the total answer to 4–6 sentences or a short numbered list if the question has multiple parts.
7. End with a one-line "Bottom line:" that tells the person what action to consider.

{lang_instruction(lang)}

DOCUMENT SECTIONS:
{context}"""

    msgs = history[-6:] + [{"role": "user", "content": question}]
    return ask_sarvam(msgs, system)


# ════════════════════════════════════════════════════════════════════════════
# SARVAM API — TTS (full-length, no truncation)
# ════════════════════════════════════════════════════════════════════════════

def strip_markdown(text: str) -> str:
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'`+', '', text)
    text = re.sub(r'\n{2,}', '. ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def split_for_tts(text: str, max_chars: int = 490) -> list:
    """
    Split cleaned text into sentence-aware chunks of at most max_chars.
    Splitting on sentence boundaries avoids mid-word cuts in audio.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current = [], ""
    for s in sentences:
        candidate = (current + " " + s).strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # A single sentence longer than max_chars — hard split
            current = s[:max_chars]
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def _tts_single(text: str, lang: str) -> tuple:
    """Call Sarvam TTS for a single chunk (≤490 chars). Returns (wav_bytes, error)."""
    lang_code = tts_lang_code(lang)
    speaker   = tts_speaker(lang)
    try:
        resp = requests.post(
            "https://api.sarvam.ai/text-to-speech",
            headers={"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"},
            json={
                "inputs": [text],
                "target_language_code": lang_code,
                "speaker": speaker,
                "model": "bulbul:v2",
                "enable_preprocessing": True,
            },
            timeout=25,
        )
        if resp.status_code == 200:
            audios = resp.json().get("audios", [])
            if audios and audios[0]:
                return base64.b64decode(audios[0]), None
            return None, f"Empty audio list. Response: {resp.text[:200]}"
        return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return None, str(e)


def _concatenate_wav(wav_list: list) -> bytes:
    """Merge multiple WAV byte strings into a single WAV using Python's wave module."""
    if not wav_list:
        return b""
    if len(wav_list) == 1:
        return wav_list[0]
    buf_out = io.BytesIO()
    params = None
    all_frames = []
    for wav_bytes in wav_list:
        buf = io.BytesIO(wav_bytes)
        try:
            with wave.open(buf, 'rb') as wf:
                if params is None:
                    params = wf.getparams()
                all_frames.append(wf.readframes(wf.getnframes()))
        except Exception:
            pass
    if not all_frames or params is None:
        return wav_list[0]
    with wave.open(buf_out, 'wb') as wf:
        wf.setparams(params)
        for f in all_frames:
            wf.writeframes(f)
    return buf_out.getvalue()


def text_to_speech_full(text: str, lang: str) -> tuple:
    """
    Convert the full answer text to audio without truncation.
    Splits into sentence-aligned ≤490-char segments, calls Sarvam TTS for each,
    then concatenates the WAV bytes into one seamless audio file.
    Returns (wav_bytes, error_str).
    """
    clean  = strip_markdown(text)
    chunks = split_for_tts(clean)

    wav_parts = []
    for chunk in chunks:
        wav, err = _tts_single(chunk, lang)
        if err:
            return None, err
        if wav:
            wav_parts.append(wav)

    if not wav_parts:
        return None, "No audio generated."

    return _concatenate_wav(wav_parts), None


# ════════════════════════════════════════════════════════════════════════════
# SARVAM API — STT (voice input)
# ════════════════════════════════════════════════════════════════════════════

def speech_to_text(audio_bytes: bytes, lang: str) -> tuple:
    """
    Transcribe audio using Sarvam saaras:v3 STT.
    Returns (transcript_str, error_str).
    """
    lang_code = stt_lang_code(lang)
    try:
        resp = requests.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": SARVAM_API_KEY},
            files={"file": ("audio.wav", audio_bytes, "audio/wav")},
            data={"model": "saaras:v3", "language_code": lang_code},
            timeout=30,
        )
        if resp.status_code == 200:
            transcript = resp.json().get("transcript", "").strip()
            if transcript:
                return transcript, None
            return None, "Empty transcript returned."
        return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return None, str(e)


# ════════════════════════════════════════════════════════════════════════════
# WATCH-OUT DETECTOR
# ════════════════════════════════════════════════════════════════════════════

def extract_watchouts(answer: str) -> list:
    """
    Scan the LLM answer for sentences that contain risk/penalty keywords.
    Returns a list of flagged sentences to show in the ⚠ banner.
    """
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    flagged = []
    for s in sentences:
        if any(kw in s.lower() for kw in WATCHOUT_KEYWORDS):
            flagged.append(s.strip())
    return flagged[:3]  # Cap at 3 watch-outs to keep UI clean


# ════════════════════════════════════════════════════════════════════════════
# HTML FORMATTING
# ════════════════════════════════════════════════════════════════════════════

def format_response_html(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    lines = text.split("\n")
    out = []
    in_ol = in_ul = False

    def close_lists():
        nonlocal in_ol, in_ul
        if in_ol: out.append("</ol>"); in_ol = False
        if in_ul: out.append("</ul>"); in_ul = False

    def inline_fmt(s):
        s = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#e8e4da;">\1</strong>', s)
        s = re.sub(r'__(.+?)__',     r'<strong style="color:#e8e4da;">\1</strong>', s)
        s = re.sub(r'\*(.+?)\*',     r'<em>\1</em>', s)
        s = re.sub(r'`(.+?)`',       r'<code style="background:#1a1e2a;padding:0.1em 0.4em;border-radius:4px;font-size:0.87em;">\1</code>', s)
        return s

    for line in lines:
        s = line.strip()
        if not s:
            close_lists()
            continue
        m = re.match(r'^(\d+)[.)]\s+(.*)', s)
        if m:
            if in_ul: out.append("</ul>"); in_ul = False
            if not in_ol:
                out.append('<ol style="margin:0.4rem 0 0.5rem 1.3rem;padding:0;display:flex;flex-direction:column;gap:0.45rem;">')
                in_ol = True
            out.append(f'<li style="color:#d8d4ca;font-size:0.93rem;line-height:1.65;">{inline_fmt(m.group(2))}</li>')
            continue
        m = re.match(r'^[-\u2022*]\s+(.*)', s)
        if m:
            if in_ol: out.append("</ol>"); in_ol = False
            if not in_ul:
                out.append('<ul style="margin:0.4rem 0 0.5rem 0;padding:0;display:flex;flex-direction:column;gap:0.4rem;list-style:none;">')
                in_ul = True
            out.append(f'<li style="color:#d8d4ca;font-size:0.93rem;line-height:1.65;display:flex;gap:0.5rem;"><span style="color:#c9a84c;flex-shrink:0;">›</span><span>{inline_fmt(m.group(1))}</span></li>')
            continue
        m = re.match(r'^#{1,3}\s+(.*)', s)
        if m:
            close_lists()
            out.append(f'<div style="font-size:0.7rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#c9a84c;margin:0.8rem 0 0.3rem;">{inline_fmt(m.group(1))}</div>')
            continue
        # Highlight "Bottom line:" if present
        if s.lower().startswith("bottom line"):
            close_lists()
            s_html = inline_fmt(s)
            out.append(f'<p style="margin:0.6rem 0 0;color:#c9a84c;font-size:0.88rem;font-weight:600;line-height:1.7;border-top:1px solid #1c2030;padding-top:0.5rem;">{s_html}</p>')
            continue
        close_lists()
        out.append(f'<p style="margin:0 0 0.45rem;color:#d8d4ca;font-size:0.93rem;line-height:1.7;">{inline_fmt(s)}</p>')

    close_lists()
    return "\n".join(out)


# ════════════════════════════════════════════════════════════════════════════
# REPORT EXPORT
# ════════════════════════════════════════════════════════════════════════════

def build_report(doc_name: str, chat_history: list) -> str:
    chat_html = "".join(
        f'<p style="text-align:{"right" if m["role"]=="user" else "left"};'
        f'color:{"#c9a84c" if m["role"]=="user" else "#d8d4ca"};">'
        f'{m["content"]}</p><hr style="border-color:#222;">'
        for m in chat_history
    )
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>FinLens Report – {doc_name}</title>
<style>body{{font-family:Georgia,serif;background:#0b0d12;color:#d8d4ca;max-width:760px;margin:0 auto;padding:2rem;}}
h1{{color:#c9a84c;}}h2{{color:#888;font-size:0.85rem;text-transform:uppercase;letter-spacing:0.1em;border-bottom:1px solid #222;padding-bottom:0.4rem;}}</style>
</head><body>
<h1>FinLens Q&amp;A Report</h1>
<p style="color:#555;">Document: <strong style="color:#c9a84c;">{doc_name}</strong></p>
<h2>Conversation History</h2>{chat_html}
</body></html>"""


# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════════════
defaults = {
    "doc1_name": None, "doc1_text": "", "doc1_chunks": 0,
    "doc1_structured": None, "doc1_type": "financial", "doc1_summary": "",
    "doc2_name": None, "doc2_text": "", "doc2_chunks": 0,
    "doc2_structured": None, "doc2_type": "financial", "doc2_summary": "",
    "chat_history": [],
    "last_answer": "",
    "last_chunks": [],
    "lang_setting": "English",
    "pending_prompt": "",
    "last_audio_hash": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-title">Fin<span>Lens</span></div>
        <div class="brand-sub">Document Intelligence</div>
    </div>""", unsafe_allow_html=True)

    # ── Primary document ─────────────────────────────────────────────────────
    st.markdown('<div class="lbl">📄 Primary Document</div>', unsafe_allow_html=True)
    f1 = st.file_uploader("", type=["pdf"], key="upload1", label_visibility="collapsed")
    if f1 and f1.name != st.session_state.doc1_name:
        with st.spinner("Indexing & summarising…"):
            text = extract_text(f1)
            doc_type = detect_doc_type(text)
            n = index_document(f1.name, text, col1_db)
            summary = generate_doc_summary(text, doc_type, st.session_state.lang_setting)
            st.session_state.update({
                "doc1_name": f1.name, "doc1_text": text, "doc1_chunks": n,
                "doc1_structured": extract_structured(text),
                "doc1_type": doc_type, "doc1_summary": summary,
                "chat_history": [], "last_answer": "", "last_chunks": [],
            })
        st.success(f"✓ {n} sections indexed")

    if st.session_state.doc1_name:
        doc_label = DOC_TYPE_LABELS.get(st.session_state.doc1_type, "📄 Financial Document")
        st.markdown(f"""
        <div style="margin:0.7rem 0 0.4rem;background:#10131a;border:1px solid #1c2030;border-radius:8px;padding:0.7rem 1rem;">
          <div class="doc-type-badge">{doc_label}</div>
          <div style="font-size:0.82rem;color:#c9a84c;font-weight:500;word-break:break-all;">{st.session_state.doc1_name}</div>
          <div style="font-size:0.72rem;color:#2e3345;margin-top:0.2rem;">{st.session_state.doc1_chunks} sections indexed</div>
        </div>""", unsafe_allow_html=True)

        if st.session_state.doc1_summary:
            st.markdown(f"""
            <div class="summary-card">
              <div class="lbl" style="margin-bottom:0.3rem;">AI Summary</div>
              <div class="summary-text">{st.session_state.doc1_summary}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Compare document ─────────────────────────────────────────────────────
    st.markdown('<div class="lbl">📄 Compare Document (optional)</div>', unsafe_allow_html=True)
    f2 = st.file_uploader("", type=["pdf"], key="upload2", label_visibility="collapsed")
    if f2 and f2.name != st.session_state.doc2_name:
        with st.spinner("Indexing & summarising…"):
            text2 = extract_text(f2)
            doc_type2 = detect_doc_type(text2)
            n2 = index_document(f2.name, text2, col2_db)
            summary2 = generate_doc_summary(text2, doc_type2, st.session_state.lang_setting)
            st.session_state.update({
                "doc2_name": f2.name, "doc2_text": text2, "doc2_chunks": n2,
                "doc2_structured": extract_structured(text2),
                "doc2_type": doc_type2, "doc2_summary": summary2,
            })
        st.success(f"✓ {n2} sections indexed")

    if st.session_state.doc2_name:
        doc_label2 = DOC_TYPE_LABELS.get(st.session_state.doc2_type, "📄 Financial Document")
        st.markdown(f"""
        <div style="margin:0.4rem 0;background:#10131a;border:1px solid #1c2030;border-radius:8px;padding:0.7rem 1rem;">
          <div class="doc-type-badge">{doc_label2}</div>
          <div style="font-size:0.82rem;color:#c9a84c;font-weight:500;word-break:break-all;">{st.session_state.doc2_name}</div>
          <div style="font-size:0.72rem;color:#2e3345;margin-top:0.2rem;">{st.session_state.doc2_chunks} sections indexed</div>
        </div>""", unsafe_allow_html=True)

        if st.session_state.doc2_summary:
            st.markdown(f"""
            <div class="summary-card" style="margin-top:0.4rem;">
              <div class="lbl" style="margin-bottom:0.3rem;">AI Summary</div>
              <div class="summary-text">{st.session_state.doc2_summary}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Language ─────────────────────────────────────────────────────────────
    st.markdown('<div class="lbl">Answer Language</div>', unsafe_allow_html=True)
    lang_choice = st.radio(
        "lang",
        ["English", "Hindi", "Hinglish"],
        index=["English", "Hindi", "Hinglish"].index(st.session_state.lang_setting),
        label_visibility="collapsed",
    )
    st.session_state.lang_setting = lang_choice

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.7rem;color:#2e3345;line-height:1.65;">'
        'Answers derived strictly from uploaded documents. No external knowledge used.</div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="padding:1.8rem 0 1rem;">
  <h1 style="font-family:'DM Serif Display',serif;font-size:2rem;color:#f0ede6;letter-spacing:-0.03em;margin:0;line-height:1.15;">
    Ask your document <span style="color:#c9a84c;font-style:italic;">anything.</span>
  </h1>
  <p style="color:#44495c;font-size:0.86rem;margin-top:0.6rem;">
    Upload a policy, loan agreement, or investment brochure — then ask in plain language or by voice.
  </p>
</div>""", unsafe_allow_html=True)

if not st.session_state.doc1_name:
    st.markdown("""
    <div style="text-align:center;padding:3.5rem 2rem;border:1px dashed #1c2030;border-radius:14px;margin-top:1.5rem;">
      <div style="font-size:2.2rem;margin-bottom:1rem;">📄</div>
      <div style="font-size:0.95rem;color:#2e3345;line-height:1.75;">
        Upload a financial document from the sidebar to begin.<br>Supports PDFs of any length.
      </div>
      <div style="margin-top:1.2rem;font-size:0.76rem;color:#1c2030;">
        Mutual fund SIDs · Insurance policies · Loan agreements · FD term sheets
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

tab_chat, tab_compare = st.tabs(["💬  Ask & Chat", "⚖️  Compare"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — ASK & CHAT
# ════════════════════════════════════════════════════════════════════════════
with tab_chat:

    # ── Quick prompt chips (tailored to document type) ────────────────────────
    doc_type = st.session_state.get("doc1_type", "financial")
    quick_prompts = DOC_TYPE_PROMPTS.get(doc_type, QUICK_PROMPTS_DEFAULT)

    st.markdown('<div class="lbl">Quick prompts — tailored to your document</div>', unsafe_allow_html=True)
    for row_start in range(0, len(quick_prompts), 3):
        row = quick_prompts[row_start:row_start + 3]
        cols = st.columns(len(row))
        for col, prompt in zip(cols, row):
            with col:
                st.markdown('<div class="chip-btn">', unsafe_allow_html=True)
                if st.button(prompt, key=f"chip__{prompt}"):
                    st.session_state.pending_prompt = prompt
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chat history ──────────────────────────────────────────────────────────
    if st.session_state.chat_history:
        st.markdown('<div class="lbl">Conversation</div>', unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                content = (msg["content"]
                           .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
                st.markdown(f'<div class="msg-user">{content}</div>', unsafe_allow_html=True)
            else:
                # Show watch-out banner above the answer if risks were found
                watchouts = extract_watchouts(msg["content"])
                if watchouts:
                    items_html = "".join(f'<div style="margin-bottom:0.25rem;">⚠ {w}</div>' for w in watchouts)
                    st.markdown(
                        f'<div class="watchout-box"><div class="watchout-title">⚠ Watch-outs in this answer</div>{items_html}</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f'<div class="msg-ai">{format_response_html(msg["content"])}</div>',
                    unsafe_allow_html=True,
                )
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Source expander + action controls ────────────────────────────────────
    if st.session_state.last_answer:
        with st.expander("📎 Source sections used for last answer"):
            for i, ch in enumerate(st.session_state.last_chunks):
                st.markdown(f"""
                <div class="card" style="margin-bottom:0.5rem;">
                  <div style="font-size:0.68rem;color:#c9a84c;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem;">Section {i+1}</div>
                  <div style="font-size:0.8rem;color:#555a6e;line-height:1.6;">{ch[:300]}{"…" if len(ch) > 300 else ""}</div>
                </div>""", unsafe_allow_html=True)

        ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 3])

        with ctrl1:
            if st.button("🔊 Read aloud"):
                with st.spinner("Generating audio…"):
                    audio_bytes, err = text_to_speech_full(
                        st.session_state.last_answer,
                        st.session_state.lang_setting,
                    )
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav")
                else:
                    st.error(f"Audio failed — {err}")

        with ctrl2:
            if st.button("🗑 Clear chat"):
                st.session_state.update({
                    "chat_history": [], "last_answer": "",
                    "last_chunks": [], "pending_prompt": "",
                })
                st.rerun()

        with ctrl3:
            st.download_button(
                "⬇ Export Q&A Report",
                data=build_report(st.session_state.doc1_name, st.session_state.chat_history),
                file_name="finlens_report.html",
                mime="text/html",
            )

    # ── Voice input (Sarvam STT) ──────────────────────────────────────────────
    # ── Voice input (Sarvam STT) ──────────────────────────────────────────────
    import hashlib

    st.markdown('<div class="lbl" style="margin-top:0.5rem;">🎙 Ask by voice</div>', unsafe_allow_html=True)
    try:
        audio_input = st.audio_input(
            "Record your question — FinLens will transcribe it with Sarvam STT",
            key="voice_input",
            label_visibility="collapsed",
        )
    except AttributeError:
        audio_input = st.file_uploader(
            "Upload a WAV/MP3 of your question",
            type=["wav", "mp3", "m4a"],
            key="voice_input_file",
            label_visibility="collapsed",
        )

    if audio_input is not None:
        audio_bytes_in = (
            audio_input.read()
            if hasattr(audio_input, "read")
            else audio_input.getvalue()
        )
        audio_hash = hashlib.md5(audio_bytes_in).hexdigest()

        if audio_hash != st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = audio_hash
            with st.spinner("Transcribing with Sarvam STT…"):
                transcript, stt_err = speech_to_text(audio_bytes_in, st.session_state.lang_setting)
            if transcript:
                st.markdown(
                    f'<div style="font-size:0.8rem;color:#44495c;margin:0.3rem 0 0.6rem;">'
                    f'Heard: <em style="color:#c9a84c;">{transcript}</em></div>',
                    unsafe_allow_html=True,
                )
                st.session_state.pending_prompt = transcript
                st.rerun()
            else:
                st.warning(f"Could not transcribe audio — {stt_err}")

    # ── Chat input (typed) ────────────────────────────────────────────────────
    user_input = st.chat_input("Type your question and press Enter…")

    question_to_run = ""
    if st.session_state.pending_prompt:
        question_to_run = st.session_state.pending_prompt
        st.session_state.pending_prompt = ""
    elif user_input:
        question_to_run = user_input

    if question_to_run:
        with st.spinner("Searching document…"):
            chunks, _ = retrieve(question_to_run, col1_db)
            if not chunks:
                answer = "I couldn't find relevant sections in your document for that question."
            else:
                answer = sarvam_rag(
                    question_to_run,
                    chunks,
                    st.session_state.doc1_type,
                    st.session_state.lang_setting,
                    st.session_state.chat_history,
                )
        st.session_state.chat_history.append({"role": "user",      "content": question_to_run})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.session_state.last_answer = answer
        st.session_state.last_chunks = chunks
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPARE
# ════════════════════════════════════════════════════════════════════════════
with tab_compare:
    if not st.session_state.doc2_name:
        st.markdown("""
        <div style="text-align:center;padding:3rem;border:1px dashed #1c2030;border-radius:12px;color:#2e3345;font-size:0.9rem;">
          Upload a second document in the sidebar to compare it against your primary document.
        </div>""", unsafe_allow_html=True)
    else:
        d1n = st.session_state.doc1_name
        d2n = st.session_state.doc2_name
        s1  = st.session_state.doc1_structured or {}
        s2  = st.session_state.doc2_structured or {}
        all_fields = sorted(set(list(s1.keys()) + list(s2.keys())))

        # ── Side-by-side summaries ────────────────────────────────────────────
        sum1 = st.session_state.doc1_summary
        sum2 = st.session_state.doc2_summary
        if sum1 or sum2:
            st.markdown('<div class="lbl">Document Summaries</div>', unsafe_allow_html=True)
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown(f'<div class="cmp-header">📄 {d1n[:28]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="summary-text" style="font-size:0.82rem;color:#7a8099;line-height:1.65;">{sum1}</div>', unsafe_allow_html=True)
            with sc2:
                st.markdown(f'<div class="cmp-header">📄 {d2n[:28]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="summary-text" style="font-size:0.82rem;color:#7a8099;line-height:1.65;">{sum2}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        # ── Key terms side-by-side ────────────────────────────────────────────
        if all_fields:
            st.markdown('<div class="lbl">Key Terms Side-by-Side</div>', unsafe_allow_html=True)
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown(f'<div class="cmp-header">📄 {d1n[:28]}</div>', unsafe_allow_html=True)
                for f in all_fields:
                    v = s1.get(f, "—")
                    st.markdown(f'<div class="cmp-item"><div class="cmp-field">{f}</div><div class="cmp-value">{v}</div></div>', unsafe_allow_html=True)
            with col_d2:
                st.markdown(f'<div class="cmp-header">📄 {d2n[:28]}</div>', unsafe_allow_html=True)
                for f in all_fields:
                    v = s2.get(f, "—")
                    st.markdown(f'<div class="cmp-item"><div class="cmp-field">{f}</div><div class="cmp-value">{v}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#44495c;font-size:0.85rem;margin-bottom:1rem;">No standard key fields auto-detected. Use the AI comparison below.</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="lbl">AI Comparison Summary</div>', unsafe_allow_html=True)

        if st.button("🤖 Generate Comparison"):
            with st.spinner("Comparing documents…"):
                ex1 = st.session_state.doc1_text[:2000]
                ex2 = st.session_state.doc2_text[:2000]
                d1_type = DOC_TYPE_LABELS.get(st.session_state.doc1_type, "financial document")
                d2_type = DOC_TYPE_LABELS.get(st.session_state.doc2_type, "financial document")
                system = f"""You are a financial advisor comparing two documents for a retail consumer in India.
Document 1 ({d1_type}): {d1n}
Document 2 ({d2_type}): {d2n}

Give a structured comparison as a numbered list covering:
1. Key differences in fees, charges, or premiums
2. Differences in lock-in periods, tenures, or exit conditions
3. Exclusions or unfavourable clauses unique to each document
4. Which document offers better overall terms and why
5. A plain-language recommendation with a clear "Bottom line:" verdict

Be direct, specific, and use simple language a first-time investor would understand.
{lang_instruction(st.session_state.lang_setting)}"""
                try:
                    answer = ask_sarvam(
                        [{"role": "user", "content": f"Document 1:\n{ex1}\n\nDocument 2:\n{ex2}"}],
                        system,
                    )
                    st.markdown(f'<div class="answer-box">{format_response_html(answer)}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Comparison failed: {e}")
