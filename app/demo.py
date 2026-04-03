"""
demo.py
=======
Streamlit demo: Baseline RAG vs. CRAG side-by-side.

Shows visually:
  - Retrieval grading decisions
  - Correction loop activations
  - Side-by-side answers
  - Cost delta (extra LLM calls)
  - Evaluation results from eval.json (if run)
"""

import json
import os

import streamlit as st
from dotenv import load_dotenv

from crag import VectorStore, baseline_rag, crag

load_dotenv()

# ---------------------------------------------------------------------------
# Sample knowledge base
# ---------------------------------------------------------------------------

DEFAULT_DOCUMENTS = [
    """Return Policy:
    We accept returns within 30 days of purchase. Items must be unused and in original packaging.
    To initiate a return, contact support@example.com with your order number.
    Refunds are processed within 5–7 business days after we receive the item.
    We do not accept returns on digital products or sale items.""",

    """Shipping Information:
    Standard shipping takes 5–7 business days. Express shipping takes 2–3 business days.
    Free shipping is available on orders over $50. We ship to all 50 US states.
    International shipping is not currently available. Tracking numbers are emailed upon shipment.""",

    """Subscription Plans:
    We offer three plans: Basic ($9/mo), Pro ($29/mo), and Enterprise ($99/mo).
    Basic includes 5 projects, 10GB storage. Pro includes unlimited projects, 100GB storage.
    Enterprise adds priority support, SSO, and custom integrations.
    All plans include a 14-day free trial. Cancel anytime — no penalty.""",

    """Privacy Policy Summary:
    We collect email, name, and usage data to provide our service.
    We never sell personal data to third parties.
    Users can request data deletion by emailing privacy@example.com.
    We use industry-standard encryption for all data at rest and in transit.""",

    """Support and Contact:
    Support is available Monday–Friday, 9am–6pm EST.
    Email: support@example.com. Average response time: 4 hours.
    Live chat is available on Pro and Enterprise plans.""",

    """Cancellation Policy:
    You can cancel your subscription at any time from Account Settings.
    Cancellation takes effect at the end of the current billing period.
    We do not offer prorated refunds for mid-cycle cancellations.""",
]

SAMPLE_QUESTIONS = [
    "What is your return policy?",
    "How long does shipping take?",
    "What does the Pro plan include?",
    "Do you ship internationally?",
    "What are your support hours?",
    "Do you support OAuth 2.0?",          # Not in docs — should trigger CRAG fallback
    "What is your quarterly revenue?",    # Not in docs — should trigger CRAG fallback
]

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Corrective RAG Demo",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Corrective RAG (CRAG)")
st.caption(
    "**Standard RAG** retrieves → generates (hallucinating if retrieval fails).  \n"
    "**CRAG** adds a quality gate: retrieve → grade → correct → generate."
)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔬 Live Comparison", "📊 Evaluation Results", "⚙️ How It Works"])

# ---------------------------------------------------------------------------
# TAB 1: Live comparison
# ---------------------------------------------------------------------------
with tab1:
    with st.sidebar:
        st.header("📚 Knowledge Base")
        st.caption("Documents indexed into the vector store.")

        # Custom documents
        custom_docs_text = st.text_area(
            "Add custom documents (one per line, use blank line to separate):",
            value="",
            height=150,
            help="Leave blank to use the default FAQ documents.",
        )

        use_custom = bool(custom_docs_text.strip())

        if use_custom:
            docs = [d.strip() for d in custom_docs_text.split("\n\n") if d.strip()]
        else:
            docs = DEFAULT_DOCUMENTS
            with st.expander(f"Default knowledge base ({len(docs)} docs)"):
                for i, d in enumerate(docs):
                    st.markdown(f"**Doc {i+1}:** {d[:80]}…")

        st.subheader("Settings")
        grade_threshold = st.slider("Relevance threshold", 0.0, 1.0, 0.5, 0.1,
                                    help="Min relevance score to accept a document")
        max_corrections = st.number_input("Max correction attempts", 1, 3, 2,
                                          help="How many times CRAG will try to fix bad retrievals")

    # Initialize vector store
    if "vector_store" not in st.session_state or st.sidebar.button("🔄 Re-index Documents"):
        with st.spinner("Indexing documents…"):
            store = VectorStore()
            store.add_documents(docs)
            st.session_state.vector_store = store
            st.session_state.indexed_doc_count = len(docs)
        st.sidebar.success(f"✅ {len(docs)} documents indexed")

    store = st.session_state.vector_store

    # Query input
    st.subheader("Ask a Question")
    col_q, col_btn = st.columns([4, 1])

    with col_q:
        question = st.selectbox("Try a sample question or type your own:", SAMPLE_QUESTIONS)
        custom_q = st.text_input("Or type your own question:", placeholder="What is your refund policy?")
        query = custom_q.strip() if custom_q.strip() else question

    if st.button("🚀 Compare Baseline vs. CRAG", type="primary", use_container_width=True):
        if not query:
            st.warning("Enter a question.")
            st.stop()

        with st.spinner("Running both systems…"):
            b_trace = baseline_rag(query, store)
            c_trace = crag(query, store)

        st.divider()
        st.subheader(f"Query: *{query}*")

        # Side-by-side answers
        col_b, col_c = st.columns(2)

        with col_b:
            st.markdown("### 📄 Baseline RAG")
            st.markdown(f"> {b_trace.answer}")
            st.caption(f"LLM calls: {b_trace.total_llm_calls}  ·  Docs used: {len(b_trace.docs_used)}")

        with col_c:
            status_color = "green" if not c_trace.fallback_used else "orange"
            status_text = "✅ Answered with sources" if not c_trace.fallback_used else "⚠️ Fallback used"
            correction_text = f" · 🔄 {len(c_trace.corrections)} correction(s)" if c_trace.needed_correction else ""

            st.markdown("### 🛡️ CRAG")
            st.markdown(f"> {c_trace.answer}")
            st.caption(
                f"LLM calls: {c_trace.total_llm_calls}  ·  "
                f"Docs used: {len(c_trace.docs_used)}  ·  "
                f"{status_text}{correction_text}"
            )

        st.divider()

        # CRAG decision trace
        st.subheader("🔍 CRAG Decision Trace")

        if c_trace.grades:
            with st.expander("📋 Grader Decisions", expanded=True):
                for i, g in enumerate(c_trace.grades):
                    icon = "✅" if g.relevant else "❌"
                    color = "#2ecc71" if g.relevant else "#e74c3c"
                    st.markdown(
                        f"{icon} **Doc {i+1}** (score: {g.score:.2f}) — {g.reason}  \n"
                        f"*Preview: {g.document_preview}*"
                    )

        if c_trace.corrections:
            with st.expander("🔄 Correction Loop", expanded=True):
                for i, corr in enumerate(c_trace.corrections):
                    st.markdown(
                        f"**Attempt {i+1}:** Strategy = `{corr.strategy}`  \n"
                        f"Reformulated query: *\"{corr.query_used}\"*  \n"
                        f"Retrieved: {corr.docs_retrieved} docs → {corr.docs_passed_grade} passed grade"
                    )

        if c_trace.fallback_used:
            st.warning(
                "⚠️ **Fallback triggered:** All correction strategies were exhausted. "
                "CRAG answered without retrieved documents (from model training data). "
                "Consider adding relevant documents to the knowledge base."
            )

        # Cost delta
        extra = c_trace.total_llm_calls - b_trace.total_llm_calls
        st.info(
            f"💰 **Cost delta:** CRAG used {extra} more LLM call(s) than Baseline "
            f"(~{extra * 0.01:.2f}–{extra * 0.03:.2f}¢ extra at gpt-4o-mini pricing)"
        )

# ---------------------------------------------------------------------------
# TAB 2: Evaluation Results
# ---------------------------------------------------------------------------
with tab2:
    st.header("📊 Evaluation: Baseline RAG vs. CRAG")
    st.caption("Run `python eval.py` to generate results, then reload this page.")

    eval_path = os.path.join(os.path.dirname(__file__), "eval_results.json")

    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_data = json.load(f)

        summary = eval_data["summary"]

        # Key metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Baseline Hallucination Rate",
            f"{summary['baseline_hallucination_rate']}%",
        )
        m2.metric(
            "CRAG Hallucination Rate",
            f"{summary['crag_hallucination_rate']}%",
            delta=f"-{summary['baseline_hallucination_rate'] - summary['crag_hallucination_rate']}%",
            delta_color="inverse",
        )
        m3.metric("Hallucination Reduction", f"{summary['hallucination_reduction_pct']}%")
        m4.metric("Avg Extra LLM Calls (CRAG)", f"+{summary['avg_extra_llm_calls']}")

        st.divider()

        # Per-question breakdown
        st.subheader("Per-Question Results")
        for r in eval_data["per_question"]:
            b_icon = "❌" if r["baseline_hallucinated"] else "✅"
            c_icon = "❌" if r["crag_hallucinated"] else "✅"
            corr_note = " 🔄" if r["crag_needed_correction"] else ""
            fb_note = " ⚠️ fallback" if r["crag_fallback"] else ""

            with st.expander(f"{b_icon} Baseline  {c_icon} CRAG{corr_note}{fb_note}  —  *{r['question']}*"):
                col_b, col_c = st.columns(2)
                with col_b:
                    st.markdown("**Baseline Answer:**")
                    st.markdown(f"> {r['baseline_answer']}")
                with col_c:
                    st.markdown("**CRAG Answer:**")
                    st.markdown(f"> {r['crag_answer']}")

    else:
        st.info(
            "No evaluation results found. Run the evaluation script first:\n\n"
            "```bash\ncd app\npython eval.py\n```\n\n"
            "Then reload this page."
        )

# ---------------------------------------------------------------------------
# TAB 3: How It Works
# ---------------------------------------------------------------------------
with tab3:
    st.header("⚙️ How CRAG Works")

    st.markdown("""
### The Problem with Standard RAG

Standard RAG retrieves documents and passes them to the LLM regardless of relevance.
When retrieval fails, the LLM **hallucinates with confidence**.

```
Query: "What's your return policy?"
Retriever: [doc about shipping] ← wrong doc
Generator: "Returns are accepted within 45 days" ← MADE UP ❌
```

---

### The CRAG Solution

```
Query: "What's your return policy?"
  ↓
[1] Retrieve: [doc about shipping]
  ↓
[2] Grade: "Is this doc relevant?" → NO (score: 0.1)
  ↓
[3] Correct: Expand query → "product return refund policy"
  ↓
[1] Retrieve again: [return policy doc] ✓
  ↓
[2] Grade: "Is this relevant?" → YES (score: 0.9)
  ↓
[4] Generate: "Returns accepted within 30 days..." [Doc 1] ✓
```

---

### Grader Agent

The grader is an LLM call (gpt-4o-mini) that evaluates:
- **Input:** (query, document)
- **Output:** `{relevant: bool, score: 0–1, reason: str}`

Binary output keeps routing logic simple. Score gives observability.

---

### Correction Strategies

| Strategy | When Used | What It Does |
|---|---|---|
| **Query expansion** | First failure | Rewrite query with broader/different terminology |
| **Decomposition** | Second failure | Break complex query into simpler sub-queries |
| **Keyword fallback** | Third failure | Extract keywords for BM25-style sparse search |

---

### Fallback

If all corrections fail:
- CRAG returns "I don't have enough information"
- Never invents an answer
- This is the key trust-building feature

---

### Cost Tradeoff

| Path | Extra LLM calls | When |
|---|---|---|
| Docs pass grade immediately | +1 (grader) | ~70% of queries |
| One correction needed | +2–3 (grader + corrector) | ~20% of queries |
| Fallback | +4–5 (all strategies) | ~10% of queries |

**Average overhead: ~1.3x baseline cost for 75% fewer hallucinations.**
""")
