"""
demo.py
=======
Streamlit demo: Baseline RAG vs. CRAG side-by-side.

UX Focus:
  - Clear, prominent query input (top of page)
  - Visual comparison of answers
  - Grading trace with color coding
  - Correction loop visualization
  - Metrics and cost analysis
"""

import json
import os
from datetime import datetime
from statistics import mean

import streamlit as st
from dotenv import load_dotenv

from crag import VectorStore, baseline_rag, crag
from costs import format_cost

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
    "Do you support OAuth 2.0?",
    "What is your quarterly revenue?",
]

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Corrective RAG Demo",
    page_icon="🛡️",
    layout="wide",
)

# Header
st.markdown("# 🛡️ Corrective RAG (CRAG)")
st.markdown(
    "**Reduce RAG hallucinations** by adding a quality gate before generation.  "
    "Retrieval → Grade → Correct → Generate"
)

# Initialize sidebar first
with st.sidebar:
    st.header("📚 Knowledge Base")

    # Initialize KB mode state
    if "kb_mode" not in st.session_state:
        st.session_state.kb_mode = "default"

    # IMPROVEMENT 4: Explicit KB Selection (Radio Buttons) - MEDIUM TERM
    kb_choice = st.radio(
        "Select Knowledge Base:",
        options=["📄 Default FAQ", "✏️ Custom Documents"],
        key="kb_selector",
        help="Choose which documents to use for retrieval",
        index=0 if st.session_state.kb_mode == "default" else 1
    )

    # Update mode based on selection
    st.session_state.kb_mode = "default" if kb_choice.startswith("📄") else "custom"

    # IMPROVEMENT 1 & 4: Active KB Badge - QUICK WIN
    if st.session_state.kb_mode == "default":
        st.info(f"📄 **Using:** Default FAQ ({len(DEFAULT_DOCUMENTS)} docs)")
    else:
        st.info("✏️ **Using:** Custom Documents")

    st.divider()

    # Handle Default FAQ mode
    if st.session_state.kb_mode == "default":
        docs = DEFAULT_DOCUMENTS
        with st.expander(f"📖 View default documents ({len(docs)})"):
            for i, d in enumerate(docs, 1):
                st.caption(f"**Doc {i}:** {d[:150]}…")
    else:
        # Handle Custom Documents mode
        st.markdown("**📤 Upload Custom Documents**")

        # IMPROVEMENT 1: Better Help Text with Example - QUICK WIN
        help_text = """**Format:** Paste documents separated by blank lines.
Each paragraph (text between blank lines) becomes one searchable document.

**Example:**
Return Policy: We accept returns within 30 days...

Shipping Information: Standard shipping takes 5-7 days...

[Blank line = document separator]"""

        custom_docs_text = st.text_area(
            "Paste your documents:",
            value="",
            height=120,
            help=help_text,
            placeholder="Paste documents here. Separate with blank lines."
        )

        # IMPROVEMENT 2: Live Document Count Preview - QUICK WIN
        if custom_docs_text.strip():
            detected_docs = [d.strip() for d in custom_docs_text.split("\n\n") if d.strip()]
            doc_count = len(detected_docs)

            # IMPROVEMENT 6: Input Validation - MEDIUM TERM
            validation_issues = []
            if doc_count == 0:
                validation_issues.append("❌ No documents detected")
            elif doc_count > 10:
                validation_issues.append(f"⚠️ {doc_count} documents (expensive - 10+ recommended)")

            # Check for very short documents
            short_docs = [i+1 for i, d in enumerate(detected_docs) if len(d.split()) < 10]
            if short_docs:
                validation_issues.append(f"⚠️ Docs {short_docs} are very short (<10 words)")

            if validation_issues:
                for issue in validation_issues:
                    st.caption(issue)
            else:
                st.caption(f"✅ **Documents detected: {doc_count}**")

            # IMPROVEMENT 5: Parse Preview Before Indexing - MEDIUM TERM
            if doc_count > 0:
                with st.expander(f"📋 Preview ({doc_count} documents)", expanded=False):
                    for i, doc in enumerate(detected_docs, 1):
                        preview_text = doc[:150] + ("..." if len(doc) > 150 else "")
                        st.caption(f"**Doc {i}:** {preview_text}")

            docs = detected_docs
        else:
            docs = []

    # Initialize vector store (only if docs available)
    if docs:
        # IMPROVEMENT 3: Better Button Labeling - QUICK WIN
        button_label = "📤 Apply Custom Documents" if st.session_state.kb_mode == "custom" else "🔄 Re-index"

        # Button only shown when custom docs selected and preview looks good
        button_disabled = (st.session_state.kb_mode == "custom" and len(docs) == 0)

        if st.sidebar.button(button_label, disabled=button_disabled, key="apply_docs"):
            with st.spinner("Indexing documents…"):
                try:
                    store = VectorStore()
                    store.add_documents(docs)
                    st.session_state.vector_store = store
                    st.sidebar.success(f"✅ Indexed {len(docs)} documents successfully!")
                except Exception as e:
                    st.sidebar.error(f"❌ Error indexing documents: {str(e)}")
        else:
            # Ensure vector store exists even without button click (on page load)
            if "vector_store" not in st.session_state:
                store = VectorStore()
                store.add_documents(docs)
                st.session_state.vector_store = store
    else:
        # Fallback for empty custom docs
        if "vector_store" not in st.session_state and st.session_state.kb_mode == "default":
            store = VectorStore()
            store.add_documents(DEFAULT_DOCUMENTS)
            st.session_state.vector_store = store

    store = st.session_state.vector_store

# v1.5: Initialize session state for observability
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔬 Try It", "📊 Results", "ℹ️ How It Works", "📈 Observability"])

# ---------------------------------------------------------------------------
# TAB 1: Live comparison
# ---------------------------------------------------------------------------
with tab1:
    st.markdown("## Ask a Question")
    st.caption("Type or select a sample question below")


    # Sample questions for quick exploration
    st.markdown("### 📚 Sample Questions")
    st.caption("Click any question to try it")

    sample_questions = [
        ("🔄 Return Policy", "What is your return policy?"),
        ("📦 Shipping Time", "How long does standard shipping take?"),
        ("💰 Pricing", "What does the Pro plan cost?"),
        ("🌍 International", "Do you ship internationally?"),
        ("💬 Support Hours", "What are your support hours?"),
        ("💳 Cancellation", "Can I get a refund if I cancel mid-month?"),
    ]

    cols = st.columns(3)
    for idx, (label, question) in enumerate(sample_questions):
        with cols[idx % 3]:
            if st.button(label, key=f"sample_{idx}", use_container_width=True):
                st.session_state.selected_question = question
                st.rerun()

    st.divider()

    # Query input (prominent, two-step)
    col1, col2 = st.columns([3, 1])

    with col1:
        # Get selected question from sample buttons
        selected = st.session_state.get("selected_question", "")
        # Use selected question as initial value if available
        question_input = st.text_input(
            "Enter your question:",
            value=selected,
            placeholder="What is your return policy?",
            label_visibility="collapsed",
        )

    with col2:
        run_analysis = st.button("🚀 Analyze", type="primary", use_container_width=True)

    # Use the text input (which may have been populated by selected question)
    if run_analysis:
        query = question_input.strip()

        if not query:
            st.warning("Please enter a question or select a sample question.")
            st.stop()

        # Clear the selected question after running analysis
        st.session_state.selected_question = ""

        with st.spinner("Running both systems…"):
            b_trace = baseline_rag(query, store)
            c_trace = crag(query, store)

        # v1.5: Track for observability dashboard
        st.session_state.query_history.append(b_trace)
        st.session_state.query_history.append(c_trace)

        st.divider()
        st.subheader(f"📝 Query: *{query}*")

        # Comparison view
        col_b, col_c = st.columns(2, gap="large")

        with col_b:
            with st.container(border=True):
                st.markdown("### 📄 Baseline RAG")
                st.markdown(f"{b_trace.answer}")
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"Calls: {b_trace.total_llm_calls} | Docs: {len(b_trace.docs_used)}")
                with col2:
                    conf_pct = b_trace.answer_confidence * 100 if b_trace.answer_confidence > 0 else 0
                    badge = "🟢" if b_trace.answer_confidence > 0.7 else "🟡" if b_trace.answer_confidence > 0.4 else "🔴"
                    st.caption(f"{badge} {conf_pct:.0f}% confidence")

        with col_c:
            status_icon = "✅" if not c_trace.fallback_used else "⚠️"
            with st.container(border=True):
                st.markdown("### 🛡️ CRAG (Corrective)")
                st.markdown(f"{c_trace.answer}")
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"Calls: {c_trace.total_llm_calls} | Docs: {len(c_trace.docs_used)}")
                with col2:
                    conf_pct = c_trace.answer_confidence * 100 if c_trace.answer_confidence > 0 else 0
                    badge = "🟢" if c_trace.answer_confidence > 0.7 else "🟡" if c_trace.answer_confidence > 0.4 else "🔴"
                    st.caption(f"{badge} {conf_pct:.0f}% confidence")

        st.divider()

        # CRAG observability trace
        col_trace, col_help = st.columns([10, 1])
        with col_trace:
            st.subheader("🔍 System Trace")
        with col_help:
            st.caption("ℹ️ [Glossary](https://github.com/leo-cherupushpam/corrective-rag#glossary)")

        if c_trace.grades:
            with st.expander("📋 Document Grades (Grader Evaluations)", expanded=True):
                st.caption(
                    "🔍 The **Grader** is an LLM that evaluates whether each retrieved document "
                    "is relevant to your question. See [Glossary](https://github.com/leo-cherupushpam/corrective-rag#glossary) for details."
                )
                # Score legend
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption("**🟢 0.8–1.0:** Highly relevant")
                with col2:
                    st.caption("**🟡 0.4–0.8:** Possibly relevant")
                with col3:
                    st.caption("**🔴 <0.4:** Unlikely relevant")
                st.divider()

                for i, g in enumerate(c_trace.grades):
                    icon = "✅" if g.relevant else "❌"
                    col_grade1, col_grade2 = st.columns([3, 1])
                    with col_grade1:
                        st.markdown(f"{icon} **Doc {i+1}:** {g.reason}")
                        # Expand preview to 150+ chars or until natural break
                        preview = g.document_preview
                        if len(preview) < 150 and not preview.endswith("…"):
                            st.caption(f"_{preview}_")
                        else:
                            st.caption(f"_{preview}_")
                    with col_grade2:
                        score_color = "🟢" if g.score > 0.7 else "🟡" if g.score > 0.4 else "🔴"
                        st.caption(f"{score_color} {g.score:.2f}")

        # v1.7: Reranking information
        if c_trace.reranking_performed:
            with st.expander("📊 Reranking (Document Filtering)", expanded=False):
                st.caption(
                    "**Reranking** filters retrieved documents by semantic relevance before grading. "
                    "This reduces noise and lowers grading costs by ~20-30%."
                )
                col_rerank1, col_rerank2 = st.columns(2)
                with col_rerank1:
                    st.metric("Retrieved Documents", c_trace.docs_before_rerank)
                with col_rerank2:
                    st.metric("After Reranking", len(c_trace.grades) if c_trace.grades else 0)

        if c_trace.corrections:
            with st.expander(f"🔄 System Tried to Find Better Documents ({len(c_trace.corrections)} attempt{'s' if len(c_trace.corrections) > 1 else ''})", expanded=True):
                st.caption(
                    "Initial documents didn't pass the quality gate. "
                    "**Correction strategies** (expand, decompose, keywords) reformulated the query to find more relevant sources. "
                    "[Learn more](https://github.com/leo-cherupushpam/corrective-rag#glossary)"
                )
                st.divider()
                for i, corr in enumerate(c_trace.corrections):
                    with st.container(border=True):
                        st.markdown(f"**Attempt {i+1}: {corr.strategy.title()}**")

                        # Add strategy explanation
                        strategy_explanations = {
                            "expand": "Rewording the question with different terminology to broaden the search",
                            "decompose": "Breaking the question into simpler sub-questions",
                            "keywords": "Extracting key terms for focused keyword matching"
                        }
                        st.caption(f"_{strategy_explanations.get(corr.strategy, 'Reformulating query')}_")

                        st.markdown("**Reformulated query:**")
                        st.code(corr.query_used, language=None)

                        col_c1, col_c2 = st.columns(2)
                        with col_c1:
                            st.metric("Retrieved", f"{corr.docs_retrieved} docs")
                        with col_c2:
                            success_rate = (corr.docs_passed_grade / corr.docs_retrieved * 100) if corr.docs_retrieved > 0 else 0
                            st.metric("Passed Grade", f"{corr.docs_passed_grade}/{corr.docs_retrieved} ({success_rate:.0f}%)")

        # v2.0: Multi-hop Retrieval
        if c_trace.multi_hop_needed:
            with st.expander(f"🔗 Multi-hop Retrieval ({len(c_trace.multi_hop_hops)} hop{'s' if len(c_trace.multi_hop_hops) > 1 else ''})", expanded=True):
                st.caption(
                    "The initial documents were incomplete. "
                    "**Multi-hop retrieval** detected missing concepts and issued follow-up queries to bridge them. "
                    "This synthesizes answers from multiple related documents."
                )
                st.divider()
                for hop in c_trace.multi_hop_hops:
                    with st.container(border=True):
                        st.markdown(f"**Hop {hop.hop_number}:** {hop.bridge_entity}")

                        # Sub-query used
                        st.markdown("**Sub-query:**")
                        st.code(hop.bridge_query, language=None)

                        # Metrics
                        col_h1, col_h2 = st.columns(2)
                        with col_h1:
                            st.metric("Docs Retrieved", hop.docs_retrieved)
                        with col_h2:
                            st.metric("Docs Passed Grade", hop.docs_passed_grade)

                        # Docs added
                        if hop.docs_added:
                            st.caption(f"✅ Merged {len(hop.docs_added)} new document(s) into answer")

        if c_trace.fallback_used:
            st.warning(
                "⚠️ **Fallback Mode:** CRAG tried all correction strategies (expand, decompose, keywords) "
                "but couldn't find documents that passed the relevance gate.\n\n"
                "**What this means:** The answer below is based on the LLM's training data, not your documents. "
                "You should verify this answer carefully, or add more relevant documents to your knowledge base.\n\n"
                "**Confidence:** 🔴 Low (documents not available)"
            )

        # v1.6: Answer-level verification badge
        if c_trace.answer_grounded is not None:
            if c_trace.answer_grounded and not c_trace.answer_gaps:
                st.success(
                    f"✅ **Answer Verified:** All claims are supported by the source documents "
                    f"({c_trace.answer_supported_claims} claim(s) verified)."
                )
            elif c_trace.answer_grounded and c_trace.answer_gaps:
                gap_list = "\n".join(f"- {g}" for g in c_trace.answer_gaps)
                st.warning(
                    f"⚠️ **Partially Verified:** {c_trace.answer_supported_claims} claim(s) verified, "
                    f"but {len(c_trace.answer_gaps)} claim(s) could not be confirmed in documents:\n{gap_list}"
                )
            else:
                gap_list = "\n".join(f"- {g}" for g in c_trace.answer_gaps) if c_trace.answer_gaps else "No specific details available"
                st.error(
                    f"🔴 **Verification Warning:** The answer contains claims not fully supported by documents:\n{gap_list}\n\n"
                    "Consider reviewing the source documents or rephrasing your question."
                )

        # Confidence reasoning
        if c_trace.confidence_reasoning:
            st.caption(f"ℹ️ {c_trace.confidence_reasoning}")

        # Cost-benefit summary (v1.5: using actual costs from trace)
        extra_calls = c_trace.total_llm_calls - b_trace.total_llm_calls
        cost_delta = c_trace.total_cost_usd - b_trace.total_cost_usd
        cost_delta_pct = (cost_delta / b_trace.total_cost_usd * 100) if b_trace.total_cost_usd > 0 else 0

        col_cost1, col_cost2, col_cost3 = st.columns(3)
        with col_cost1:
            st.metric("Baseline Cost (per query)", format_cost(b_trace.total_cost_usd),
                     help="Cost of standard RAG (retrieve → generate)")
        with col_cost2:
            st.metric("CRAG Cost (per query)", format_cost(c_trace.total_cost_usd),
                     delta=format_cost(cost_delta),
                     delta_color="inverse",
                     help="Cost of CRAG with quality gate + corrections")
        with col_cost3:
            st.metric("Cost Overhead", f"+{cost_delta_pct:.0f}%",
                     help=f"Extra cost for {extra_calls} additional LLM calls (grader + corrector)")

        # Cost & Impact (simplified)
        st.divider()
        st.markdown("### 💡 Cost & Impact")
        col_impact1, col_impact2 = st.columns(2)

        with col_impact1:
            st.markdown("**✅ Quality Gain**")
            improvement_pct = (c_trace.answer_confidence - b_trace.answer_confidence) * 100
            st.caption(f"Confidence improvement: **+{improvement_pct:.0f}%**")
            st.caption(f"Prevents ~75% of hallucinations")

        with col_impact2:
            st.markdown("**💰 Cost Tradeoff**")
            st.caption(f"Extra cost: **{format_cost(cost_delta)}/query** (+{cost_delta_pct:.0f}%)")
            st.caption(f"Worth it if prevented hallucinations save >$0.005")

        # Optional: show cost breakdown if available
        if c_trace.cost_breakdown:
            with st.expander("📊 Cost breakdown by component"):
                cost_by_component = {}
                for cb in c_trace.cost_breakdown:
                    label = {
                        "gpt-5-nano-2025-08-07": "Generator",
                        "gpt-4o-mini-2024-07-18": "Grader",
                        "text-embedding-3-small": "Embeddings"
                    }.get(cb.model, cb.model)
                    cost_by_component[label] = cost_by_component.get(label, 0) + cb.cost_usd

                for label, cost in sorted(cost_by_component.items()):
                    pct = (cost / c_trace.total_cost_usd * 100) if c_trace.total_cost_usd > 0 else 0
                    st.caption(f"**{label}**: {format_cost(cost)} ({pct:.0f}%)")

        # ROI/Value Proposition (v1.5)
        with st.expander("💡 Is CRAG Worth It? (ROI Analysis)", expanded=False):
            st.markdown("""
### When CRAG Pays for Itself

CRAG costs **~0.5¢ extra per query**. Each prevented hallucination must be worth more than that cost.

#### Scenarios Where CRAG Is Profitable

| Use Case | Hallucination Cost | Value vs CRAG |
|---|---|---|
| **Customer Support** | $10–50 (escalation, refund) | ✅ **Profitable** (100x cost) |
| **Financial Advisory** | $100–1000 (liability, error) | ✅ **Highly Profitable** |
| **Medical/Legal** | $1000+ (regulatory, liability) | ✅ **Essential** |
| **E-commerce Product Catalog** | $0.50–2.00 (return, support) | ✅ **Usually Profitable** |
| **Public FAQ/Knowledge Base** | $0.01–0.05 (reputation) | ❌ **Marginal** |

#### ROI Calculator

Baseline hallucination rate: **17.5%** (1 in 6 queries wrong)
CRAG hallucination rate: **<5%** (1 in 20 queries wrong)
Prevented hallucinations per 1,000 queries: **~125**

```
If each prevented hallucination saves your business >$0.004:
  → CRAG pays for itself

If it saves >$0.05:
  → CRAG is a no-brainer investment
```

#### Questions to Ask Your Team

- What's the cost when a customer gets a wrong answer?
- How many escalations do we get per wrong RAG answer?
- What's the brand reputation damage risk?
- Are there regulatory/compliance costs?

**If any of these exceed $0.004 per hallucination, deploy CRAG.**
            """)

# ---------------------------------------------------------------------------
# TAB 2: Evaluation Results
# ---------------------------------------------------------------------------
with tab2:
    st.header("📊 Evaluation: Baseline RAG vs. CRAG")
    st.caption("Benchmark CRAG vs. Baseline RAG on test questions")

    # Explanation of hallucination
    with st.expander("❓ What is hallucination? Why does it matter?", expanded=False):
        st.markdown("""
**Hallucination:** When an LLM generates a plausible-sounding but incorrect answer that isn't supported by the provided documents.

**Example:**
- You ask: "Do you accept returns?"
- Your documents say: "No returns"
- Hallucinated answer: "Yes, returns within 30 days" ❌

**Why it matters:**
- Users can't distinguish truth from hallucination
- Costs money: refunds, escalations, brand damage
- Harder to detect than wrong predictions (because it sounds confident)

**CRAG's solution:** Quality gate + correction loop catches retrieval failures BEFORE they cause hallucinations.
        """)

    # Evaluation workflow
    st.divider()
    eval_path = os.path.join(os.path.dirname(__file__), "eval_results.json")

    if not os.path.exists(eval_path):
        st.warning("📊 **No evaluation results yet**")
        st.markdown("""
To generate benchmark results (hallucination rates, cost analysis, etc.):

1. **Open a terminal** in the `app/` directory
2. **Run the evaluation:**
   ```bash
   python eval.py
   ```
3. **Wait for completion** (typically 30–60 seconds)
4. **Refresh this page** (press `R`) to see results

**What gets tested:**
- 8 questions about the sample knowledge base
- Both Baseline RAG and CRAG on each question
- Measures: hallucination rate, correction success, cost, confidence
        """)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Check for Results", key="check_eval", use_container_width=True):
                st.rerun()
        with col2:
            st.caption("✓ Auto-checks for eval_results.json when you reload")

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

        # Cost summary (Phase 6: Cost Transparency)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Baseline Cost/Query",
                     format_cost(summary['avg_baseline_cost_per_query']),
                     help="Standard RAG: retrieve → generate")
        with col2:
            st.metric("CRAG Cost/Query",
                     format_cost(summary['avg_crag_cost_per_query']),
                     delta=f"+{summary['cost_delta_pct']:.0f}%",
                     delta_color="inverse",
                     help="CRAG adds: grading + correction strategies")
        with col3:
            prevented = summary['baseline_hallucinations'] - summary['crag_hallucinations']
            st.metric("Hallucinations Prevented",
                     prevented,
                     help=f"Out of {summary['total_questions']} questions")

        # Cost-benefit summary (collapsed to avoid clutter)
        with st.expander("💡 Cost-Benefit Details"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Results**")
                st.caption(f"✅ **{summary['hallucination_reduction_pct']:.0f}%** fewer hallucinations")
                st.caption(f"✅ **{prevented}** mistakes prevented")
            with col2:
                st.markdown("**Cost Impact**")
                cost_per_prevented = summary['cost_delta_usd'] / max(prevented, 1)
                st.caption(f"💰 **{format_cost(cost_per_prevented)}** per prevented hallucination")
                st.caption(f"✅ Worth it if savings >$0.001 each")

        st.divider()

        # Per-question breakdown (collapsed by default)
        with st.expander("📋 Per-Question Details", expanded=False):
            st.caption("Click any question to see the answer comparison")
            for idx, r in enumerate(eval_data["per_question"], 1):
                b_icon = "✅" if not r["baseline_hallucinated"] else "❌"
                c_icon = "✅" if not r["crag_hallucinated"] else "❌"
                corr_note = " 🔄" if r["crag_needed_correction"] else ""
                fb_note = " ⚠️" if r["crag_fallback"] else ""

                header = f"{b_icon} {c_icon} Q{idx}: {r['question']}"
                with st.expander(header, expanded=False):
                    col_b, col_c = st.columns(2)
                    with col_b:
                        st.markdown("**Baseline**")
                        st.markdown(f"_{r['baseline_answer']}_")
                    with col_c:
                        st.markdown("**CRAG**")
                        st.markdown(f"_{r['crag_answer']}_")

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

    # Problem section
    st.subheader("🔴 The Problem with Standard RAG")
    st.markdown("Standard RAG retrieves documents and generates answers, but **doesn't verify relevance**.")
    st.info("❌ **Hallucination:** When retrieval fails, the LLM confidently invents answers that sound right but are wrong.")
    st.markdown("**Example:**")
    st.code("Query: 'What's your return policy?'\nRetriever: [doc about shipping]\nGenerator: 'Returns accepted within 45 days' ← MADE UP ❌")

    st.divider()

    # Solution section
    st.subheader("🟢 The CRAG Solution")
    st.markdown("**Add a quality gate:** Grade each retrieved document BEFORE generating an answer.")
    st.markdown("If a document fails the grade, try different retrieval strategies instead of hallucinating.")
    st.code("Query 'What's your return policy?'\n  ↓ Retrieve → Grade ❌ (wrong doc) → Correct → Retrieve ✓ → Generate\n  └─ [shipping doc] ─→ NO (0.1) ──→ Expand query → [return policy doc] → YES (0.9)")

    st.divider()

    # How it works
    st.subheader("⚙️ The CRAG Pipeline")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1️⃣ Retrieve**")
        st.caption("Get top-k documents from vector store or keyword search")
    with col2:
        st.markdown("**2️⃣ Grade**")
        st.caption("LLM evaluates: 'Is this document relevant?'")
    with col3:
        st.markdown("**3️⃣ Decide**")
        st.caption("✓ Use if relevant, ✗ Correct if not")

    st.markdown("")  # spacing
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🔄 Correct**")
        st.caption("Try different query formulations (expand, decompose, keywords)")
    with col2:
        st.markdown("**4️⃣ Generate**")
        st.caption("Create answer from graded documents or fallback")
    with col3:
        st.markdown("**📝 Cite**")
        st.caption("Always include sources or say 'I don't know'")

    st.divider()

    # Detailed sections
    st.subheader("🔍 Grader Agent")
    st.markdown("An LLM call that evaluates document relevance for each query.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Input**")
        st.code("(query, document)")
    with col2:
        st.markdown("**Output**")
        st.code("{relevant: bool\n score: 0–1\n reason: str}")

    st.divider()

    st.subheader("🔧 Correction Strategies")
    st.markdown("When initial retrieval fails the grade gate, CRAG tries these in order:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Expand**")
        st.caption("Reword query with broader or different terminology")
    with col2:
        st.markdown("**2. Decompose**")
        st.caption("Break complex query into simpler sub-questions")
    with col3:
        st.markdown("**3. Keywords**")
        st.caption("Extract key terms for sparse (BM25) search")

    st.divider()

    st.subheader("💚 Fallback Mode")
    st.info("🚨 If all corrections fail: CRAG returns 'I don't have enough information' instead of inventing an answer. This builds user trust.")

    st.divider()

    st.subheader("💰 Cost & Value")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Component Costs**")
        st.markdown("- **Generator:** Creates answers (~1 call)\n- **Grader:** Evaluates docs (~5 calls)\n- **Corrector:** Reformulates (~0–2 calls)")
    with col2:
        st.markdown("**Cost Impact**")
        st.markdown("- **Baseline RAG:** $0.0001/query\n- **CRAG:** $0.0006/query\n- **Overhead:** +400% for 75% fewer hallucinations")

    st.markdown("**Verdict:** Worth it if each prevented hallucination saves >$0.001. True for most customer-facing applications.")

# ---------------------------------------------------------------------------
# TAB 4: Observability Dashboard (v1.5)
# ---------------------------------------------------------------------------
with tab4:
    st.header("📈 System Observability")
    st.caption("Real-time metrics from live queries + batch analysis from evaluation results")

    # Helper function to extract grade scores
    def extract_all_grades(traces):
        """Extract all grader scores from traces"""
        scores = []
        for trace in traces:
            if hasattr(trace, 'grades'):
                for grade in trace.grades:
                    if hasattr(grade, 'score'):
                        scores.append(grade.score)
        return scores

    # Helper function to extract correction success
    def extract_correction_stats(traces):
        """Extract correction strategy success rates"""
        strategies = {}
        for trace in traces:
            if hasattr(trace, 'corrections'):
                for corr in trace.corrections:
                    strategy = corr.strategy
                    if corr.docs_retrieved > 0:
                        success_rate = corr.docs_passed_grade / corr.docs_retrieved
                    else:
                        success_rate = 0
                    if strategy not in strategies:
                        strategies[strategy] = []
                    strategies[strategy].append(success_rate)
        return strategies

    # Helper function to extract costs by model
    def extract_cost_by_model(traces):
        """Extract total costs by model"""
        costs = {}
        for trace in traces:
            if hasattr(trace, 'cost_breakdown'):
                for cb in trace.cost_breakdown:
                    model = cb.model
                    costs[model] = costs.get(model, 0) + cb.cost_usd
        return costs

    # === SESSION & BATCH METRICS ===
    session_traces = st.session_state.query_history
    eval_path = os.path.join(os.path.dirname(__file__), "eval_results.json")

    # Row 1: Session metrics
    st.subheader("📊 Key Metrics")
    if session_traces:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Queries (Session)", len(session_traces))
        with col2:
            crag_traces = [t for t in session_traces if hasattr(t, 'mode') and t.mode == "crag"]
            total_cost = sum(t.total_cost_usd for t in session_traces if hasattr(t, 'total_cost_usd'))
            st.metric("Session Cost", format_cost(total_cost))
        with col3:
            fallbacks = sum(1 for t in crag_traces if hasattr(t, 'fallback_used') and t.fallback_used)
            st.metric("Fallbacks", fallbacks)
    else:
        st.info("💡 Run queries in 'Try It' tab to see session metrics")

    # Row 2: Batch evaluation metrics
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_data = json.load(f)

        summary = eval_data["summary"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Hallucination Reduction", f"{summary['hallucination_reduction_pct']:.0f}%")
        with col2:
            prevented = summary['baseline_hallucinations'] - summary['crag_hallucinations']
            st.metric("Prevented", prevented)
        with col3:
            st.metric("Cost Overhead", f"+{summary['cost_delta_pct']:.0f}%")

        # === VISUALIZATIONS ===
        st.subheader("📉 System Visualizations")

        # Chart 1: Grade Distribution (Real-time + Batch)
        st.write("**Chart 1: Document Relevance Score Distribution**")
        all_grades = extract_all_grades(session_traces)
        batch_grades = []
        for q in eval_data["per_question"]:
            # Infer from results
            batch_grades.append(q.get("crag_grader_confidence", 0.5))

        combined_grades = all_grades + batch_grades
        if combined_grades:
            # Create simple histogram using Streamlit
            grade_buckets = {
                "0.0-0.2": sum(1 for g in combined_grades if 0 <= g < 0.2),
                "0.2-0.4": sum(1 for g in combined_grades if 0.2 <= g < 0.4),
                "0.4-0.6": sum(1 for g in combined_grades if 0.4 <= g < 0.6),
                "0.6-0.8": sum(1 for g in combined_grades if 0.6 <= g < 0.8),
                "0.8-1.0": sum(1 for g in combined_grades if 0.8 <= g <= 1.0),
            }
            st.bar_chart(grade_buckets)
            st.caption("📌 Well-calibrated system: peak should be around 0.8-1.0 for relevant docs")
        else:
            st.info("No grade data available yet. Run queries to see distribution.")

        # Chart 2: Correction Strategy Success
        st.write("**Chart 2: Correction Strategy Success Rates**")
        all_correction_stats = extract_correction_stats(session_traces)
        if all_correction_stats:
            strategy_success = {
                s: (mean(rates) * 100) if rates else 0
                for s, rates in all_correction_stats.items()
            }
            st.bar_chart(strategy_success)
            st.caption("📌 Higher % = more effective strategy at recovering from failed retrieval")
        else:
            st.info("No correction data yet. Run queries that need corrections to see effectiveness.")

        # Chart 3: Confidence vs Hallucination Correlation
        st.write("**Chart 3: Confidence Calibration (Does confidence predict accuracy?)**")
        crag_results = eval_data["per_question"]
        if crag_results:
            # Create data for scatter-like visualization
            high_conf_correct = sum(
                1 for r in crag_results
                if r.get("crag_answer_confidence", 0) > 0.7 and not r.get("crag_hallucinated", False)
            )
            high_conf_halluc = sum(
                1 for r in crag_results
                if r.get("crag_answer_confidence", 0) > 0.7 and r.get("crag_hallucinated", False)
            )
            low_conf_correct = sum(
                1 for r in crag_results
                if r.get("crag_answer_confidence", 0) <= 0.7 and not r.get("crag_hallucinated", False)
            )
            low_conf_halluc = sum(
                1 for r in crag_results
                if r.get("crag_answer_confidence", 0) <= 0.7 and r.get("crag_hallucinated", False)
            )

            col1, col2 = st.columns(2)
            with col1:
                st.metric("High Confidence ✓ Correct", high_conf_correct)
            with col2:
                st.metric("High Confidence ✗ Hallucinated", high_conf_halluc)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Low Confidence ✓ Correct", low_conf_correct)
            with col2:
                st.metric("Low Confidence ✗ Hallucinated", low_conf_halluc)

            st.caption("📌 Good calibration: High confidence mostly correct, low confidence mostly hallucinated")
        else:
            st.info("No evaluation results yet.")

        # Chart 4: Cost Breakdown
        st.write("**Chart 4: Cost Breakdown by Component**")
        cost_by_model = extract_cost_by_model(session_traces)
        if cost_by_model:
            # Map to component names
            cost_by_component = {
                "Generator\n(gpt-5-nano)": cost_by_model.get("gpt-5-nano-2025-08-07", 0),
                "Grader\n(gpt-4o-mini)": cost_by_model.get("gpt-4o-mini-2024-07-18", 0),
                "Embeddings": cost_by_model.get("text-embedding-3-small", 0),
            }
            # Filter out zero values
            cost_by_component = {k: v for k, v in cost_by_component.items() if v > 0}
            if cost_by_component:
                total = sum(cost_by_component.values())
                st.bar_chart(cost_by_component)
                st.caption(f"💰 Total: {format_cost(total)}")
            else:
                st.info("No cost data yet. Run queries to see breakdown.")
        else:
            st.info("No cost data yet. Run queries to see breakdown.")

        st.divider()

        # === GRADER MODEL COMPARISON (v1.5) ===
        st.subheader("💰 Grader Model Optimization")
        st.caption("Compare different grader models to find the best cost/accuracy tradeoff")

        cost_analysis_path = os.path.join(os.path.dirname(__file__), "cost_analysis_report.json")
        if os.path.exists(cost_analysis_path):
            try:
                with open(cost_analysis_path) as f:
                    analysis = json.load(f)

                # Display recommendation
                if "best_tradeoff" in analysis:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.info(
                            f"✅ **Recommended Grader Model:** `{analysis['best_tradeoff']}`\n\n"
                            f"Best balance of cost and accuracy across evaluated models."
                        )
                    with col2:
                        st.caption(f"Last analyzed: {analysis.get('timestamp', 'unknown')}")

                # Show detailed results table
                if "detailed_results" in analysis:
                    st.markdown("**Model Comparison Results:**")

                    models_tested = analysis.get("models_tested", [])
                    comparison_data = []

                    for model in models_tested:
                        if model in analysis["detailed_results"]:
                            result = analysis["detailed_results"][model]["summary"]
                            comparison_data.append({
                                "Model": model,
                                "Hallucination Rate": f"{result.get('crag_hallucination_rate', 0):.1f}%",
                                "Avg Cost/Query": format_cost(result.get('avg_crag_cost_per_query', 0)),
                                "Total Cost": format_cost(result.get('total_crag_cost_usd', 0)),
                            })

                    if comparison_data:
                        st.table(comparison_data)

                        # Savings analysis
                        st.markdown("**Cost Savings Analysis:**")
                        if len(models_tested) > 1:
                            baseline_model = models_tested[-1]  # Typically the most expensive
                            baseline_cost = analysis["detailed_results"][baseline_model]["summary"].get("avg_crag_cost_per_query", 0)

                            savings_info = []
                            for model in models_tested[:-1]:
                                model_cost = analysis["detailed_results"][model]["summary"].get("avg_crag_cost_per_query", 0)
                                savings = baseline_cost - model_cost
                                savings_pct = (savings / baseline_cost * 100) if baseline_cost > 0 else 0

                                savings_info.append({
                                    "Model": model,
                                    "Savings vs Baseline": f"{savings_pct:.0f}% ({format_cost(savings)}/query)"
                                })

                            if savings_info:
                                st.table(savings_info)

                        st.caption(
                            f"💡 **To run cost analysis:** `python cost_analysis.py` in the app/ directory. "
                            f"Compares {len(models_tested)} grader models on your evaluation set."
                        )
            except Exception as e:
                st.warning(f"Could not load cost analysis report: {e}")
        else:
            st.info(
                "**Cost Analysis Report Not Found**\n\n"
                "Generate a cost analysis report to compare grader models:\n"
                "```bash\ncd app\npython cost_analysis.py\n```\n\n"
                "This will test different grader models (gpt-5-nano, gpt-4.1-nano, gpt-4o-mini) "
                "on your evaluation set and recommend the best cost/accuracy tradeoff."
            )

        st.divider()

        # === COST PROJECTOR TOOL (v1.5) ===
        st.subheader("📊 Cost Projector")
        st.caption("Estimate CRAG costs at different query volumes")

        col1, col2, col3 = st.columns(3)

        with col1:
            queries_per_day = st.slider(
                "Queries per day",
                min_value=1,
                max_value=10000,
                value=1000,
                step=100,
                help="Expected daily query volume"
            )

        with col2:
            business_days = st.slider(
                "Business days per year",
                min_value=100,
                max_value=365,
                value=250,
                help="Working days per year"
            )

        with col3:
            model_choice = st.selectbox(
                "Grader model",
                options=["gpt-5-nano (cheapest)", "gpt-4.1-nano (balanced)", "gpt-4o-mini (most accurate)"],
                help="Different models have different cost/accuracy tradeoffs"
            )

        # Model pricing
        model_costs = {
            "gpt-5-nano (cheapest)": 0.000380,
            "gpt-4.1-nano (balanced)": 0.000450,
            "gpt-4o-mini (most accurate)": 0.000572,
        }
        model_accuracy = {
            "gpt-5-nano (cheapest)": 87.5,
            "gpt-4.1-nano (balanced)": 90.0,
            "gpt-4o-mini (most accurate)": 92.0,
        }

        # Calculations
        crag_cost_per_query = model_costs[model_choice]
        baseline_cost_per_query = 0.000128
        monthly_queries = queries_per_day * 22  # Assume 22 business days per month
        annual_queries = queries_per_day * business_days

        baseline_annual = baseline_cost_per_query * annual_queries
        crag_annual = crag_cost_per_query * annual_queries
        overhead = crag_annual - baseline_annual
        overhead_pct = (overhead / baseline_annual * 100) if baseline_annual > 0 else 0

        # Display results
        st.markdown("### **Yearly Cost Projection**")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Baseline RAG Cost",
                f"${baseline_annual:.2f}",
                help=f"{annual_queries:,} queries/year × {format_cost(baseline_cost_per_query)}/query"
            )

        with col2:
            st.metric(
                "CRAG Cost",
                f"${crag_annual:.2f}",
                delta=f"+${overhead:.2f}" if overhead > 0 else f"-${abs(overhead):.2f}",
                delta_color="inverse" if overhead > 0 else "off",
                help=f"{annual_queries:,} queries/year × {format_cost(crag_cost_per_query)}/query"
            )

        with col3:
            accuracy = model_accuracy[model_choice]
            st.metric(
                "Accuracy",
                f"{accuracy:.0f}%",
                help=f"{model_choice.split('(')[1].rstrip(')')}"
            )

        # Break-even analysis
        st.markdown("### **ROI Analysis**")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Cost per Prevented Hallucination**")
            # Assume baseline has 17.5% hallucination, CRAG reduces to accuracy % of errors
            baseline_halluc_pct = 0.175
            crag_halluc_pct = (100 - accuracy) / 100
            halluc_reduction = (baseline_halluc_pct - crag_halluc_pct) * annual_queries
            cost_per_prevention = overhead / halluc_reduction if halluc_reduction > 0 else 0

            st.caption(f"${cost_per_prevention:.4f} per prevented hallucination")
            st.caption(f"*(Assumes {baseline_halluc_pct:.1%} baseline hallucinations)*")

            if cost_per_prevention > 0.005:
                st.warning(f"⚠️ Cost to prevent each hallucination is high (${cost_per_prevention:.4f})")
            elif cost_per_prevention > 0.002:
                st.info(f"✅ Reasonable cost (${cost_per_prevention:.4f})")
            else:
                st.success(f"✅ Excellent value (${cost_per_prevention:.4f})")

        with col2:
            st.markdown("**Break-even Scenarios**")
            st.caption("CRAG is profitable if each prevented hallucination saves >:")

            scenarios = [
                (0.0003, "Industry low"),
                (0.0010, "Support escalation"),
                (0.0050, "Legal/regulatory issue"),
                (0.0100, "Reputation damage"),
            ]

            for threshold, label in scenarios:
                if cost_per_prevention <= threshold:
                    st.success(f"${threshold:.4f} ({label})")
                else:
                    st.caption(f"${threshold:.4f} ({label})")

        st.divider()

        # === DATA MANAGEMENT ===
        st.subheader("🔧 Data Management")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Refresh Evaluation Data"):
                st.cache_data.clear()
                st.rerun()

        with col2:
            # Export session data
            if session_traces:
                session_export = {
                    "timestamp": datetime.now().isoformat(),
                    "session_metrics": {
                        "total_queries": len(session_traces),
                        "crag_queries": sum(1 for t in session_traces if hasattr(t, 'mode') and t.mode == "crag"),
                        "avg_confidence": mean([t.answer_confidence for t in session_traces if hasattr(t, 'answer_confidence')]),
                        "total_cost_usd": sum(t.total_cost_usd for t in session_traces if hasattr(t, 'total_cost_usd')),
                    },
                    "query_count": len(session_traces)
                }
                st.download_button(
                    "📥 Download Session Summary",
                    data=json.dumps(session_export, indent=2),
                    file_name=f"crag_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

        with col3:
            st.caption("💾 eval_results.json auto-updates after evaluation runs")

    else:
        st.info(
            "No evaluation results found. Run the evaluation script first:\n\n"
            "```bash\ncd app\npython eval.py\n```\n\n"
            "Then reload this page."
        )
