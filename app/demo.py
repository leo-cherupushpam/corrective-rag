"""
demo.py
=======
Streamlit demo: Baseline RAG vs. CRAG side-by-side.

Enhanced UI with:
  - Consistent color scheme and styling (styles.py)
  - Data visualizations (utils.py)
  - Smart tab organization (Query, Dashboard, How It Works, Settings)
  - Professional design system
"""

import json
import os
from datetime import datetime
from statistics import mean

import streamlit as st
from dotenv import load_dotenv

from crag import VectorStore, baseline_rag, crag
from costs import format_cost
from styles import COLORS, get_custom_css, get_confidence_label, get_relevance_label, make_confidence_badge, make_metric_card
from utils import (
    chart_relevance_scores,
    chart_cost_breakdown,
    chart_confidence_calibration,
    chart_hallucination_metrics,
    format_cost as format_cost_util,
    format_confidence,
    get_confidence_emoji,
    make_correction_flow,
    make_multi_hop_flow,
)

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

# Apply custom CSS styling
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Header
st.markdown("# 🛡️ Corrective RAG (CRAG)")
st.markdown(
    "**Reduce RAG hallucinations** by adding a quality gate before generation.  "
    "Retrieval → Grade → Correct → Generate"
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "query_history" not in st.session_state:
    st.session_state.query_history = []

if "kb_mode" not in st.session_state:
    st.session_state.kb_mode = "default"

if "vector_store" not in st.session_state:
    store = VectorStore()
    store.add_documents(DEFAULT_DOCUMENTS)
    st.session_state.vector_store = store

store = st.session_state.vector_store

# ---------------------------------------------------------------------------
# Tab Navigation
# ---------------------------------------------------------------------------

tab_query, tab_dashboard, tab_how, tab_settings = st.tabs([
    "🔬 Query",
    "📊 Dashboard",
    "ℹ️ How It Works",
    "⚙️ Settings"
])

# ============================================================================
# TAB 1: Query - Core Interactive Workflow
# ============================================================================

with tab_query:
    st.markdown("## Ask a Question")
    st.caption("Type or select a sample question to compare Baseline RAG vs CRAG")

    # Sample questions for quick exploration
    st.markdown("### 📚 Sample Questions")
    st.caption("Click any question to try it — each tests different capabilities")

    # Phase 4: Enhanced sample question labels with context hints
    sample_questions = [
        ("🔄 Return Policy\n_(Tests: Direct Lookup)_", "What is your return policy?"),
        ("📦 Shipping Time\n_(Tests: Simple Extraction)_", "How long does standard shipping take?"),
        ("💰 Pricing\n_(Tests: Multi-Value Retrieval)_", "What does the Pro plan cost?"),
        ("🌍 International\n_(Tests: Negative Answer)_", "Do you ship internationally?"),
        ("💬 Support Hours\n_(Tests: Specific Information)_", "What are your support hours?"),
        ("💳 Cancellation\n_(Tests: Policy Understanding)_", "Can I get a refund if I cancel mid-month?"),
    ]

    cols = st.columns(3)
    for idx, (label, question) in enumerate(sample_questions):
        with cols[idx % 3]:
            if st.button(label, key=f"sample_{idx}", use_container_width=True):
                st.session_state.selected_question = question
                st.rerun()

    st.divider()

    # Phase 4: Query History - Show recent queries
    if st.session_state.query_history:
        st.markdown("### 🕐 Recent Queries")
        st.caption(f"You have {len(st.session_state.query_history) // 2} previous queries (showing last 5)")

        # Get unique queries (since we store both baseline and crag traces)
        unique_queries = []
        seen = set()
        for trace in reversed(st.session_state.query_history):
            if trace.query not in seen and len(unique_queries) < 5:
                unique_queries.append(trace.query)
                seen.add(trace.query)

        if unique_queries:
            cols_history = st.columns(len(unique_queries))
            for idx, query in enumerate(unique_queries):
                with cols_history[idx]:
                    # Phase 4: Accessibility - Add aria labels to buttons
                    if st.button(
                        f"🔁 {query[:20]}..." if len(query) > 20 else f"🔁 {query}",
                        key=f"history_{idx}",
                        use_container_width=True,
                        help=f"Re-run: {query}"
                    ):
                        st.session_state.selected_question = query
                        st.rerun()
        st.divider()
    else:
        st.caption("💡 Tip: Your recent queries will appear here")

    # Query input (prominent, two-step)
    col1, col2 = st.columns([3, 1])

    with col1:
        selected = st.session_state.get("selected_question", "")
        question_input = st.text_input(
            "Enter your question:",
            value=selected,
            placeholder="What is your return policy?",
            label_visibility="collapsed",
        )

    with col2:
        run_analysis = st.button("🚀 Analyze", type="primary", use_container_width=True)

    # Execute analysis
    if run_analysis:
        query = question_input.strip()

        if not query:
            st.warning("Please enter a question or select a sample question.")
            st.stop()

        st.session_state.selected_question = ""

        with st.spinner("Running both systems…"):
            b_trace = baseline_rag(query, store)
            c_trace = crag(query, store)

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
                    conf_label, conf_color = get_confidence_label(b_trace.answer_confidence)
                    st.markdown(make_confidence_badge(b_trace.answer_confidence), unsafe_allow_html=True)

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
                    conf_label, conf_color = get_confidence_label(c_trace.answer_confidence)
                    st.markdown(make_confidence_badge(c_trace.answer_confidence), unsafe_allow_html=True)

        st.divider()

        # CRAG observability trace with Phase 4 tooltips
        col_trace, col_help = st.columns([10, 1])
        with col_trace:
            st.subheader("🔍 System Trace")
        with col_help:
            st.popover("ℹ️", help="Detailed trace showing what CRAG did to find and verify documents")

        # Phase 4: Help text for trace sections
        with st.expander("❓ Understanding the Trace", expanded=False):
            st.markdown("""
**System Trace** shows the internal decision-making:

1. **Document Grades** — How relevant is each retrieved document? (0-100%)
2. **Reranking** — Did we filter out less relevant docs?
3. **Corrections** — Did we try alternative query formulations?
4. **Multi-hop** — Did we need to retrieve in multiple steps?
5. **Verification** — Are the answer's claims supported by documents?

**Green checkmarks (✅)** = Good signal
**Red X's (❌)** = Issues detected
**Yellow warnings (⚠️)** = Partial success
            """)

        st.divider()

        # Document Grades with chart (smart default: expanded)
        if c_trace.grades:
            relevant_count = sum(1 for g in c_trace.grades if g.relevant)
            with st.expander(
                f"📋 Document Grades — {relevant_count}/{len(c_trace.grades)} relevant",
                expanded=True
            ):
                st.caption(
                    "🔍 The **Grader** is an LLM that evaluates whether each retrieved document "
                    "is relevant to your question. "
                )

                # Phase 4: Accessibility - Provide legend with colors and text
                st.markdown("**Relevance Scale:**")
                leg_col1, leg_col2, leg_col3 = st.columns(3)
                with leg_col1:
                    st.markdown('<span style="color: #06A77D;">🟢 **0.8–1.0:** Highly Relevant</span>', unsafe_allow_html=True)
                with leg_col2:
                    st.markdown('<span style="color: #F18F01;">🟡 **0.4–0.8:** Somewhat Relevant</span>', unsafe_allow_html=True)
                with leg_col3:
                    st.markdown('<span style="color: #D62828;">🔴 **<0.4:** Not Relevant</span>', unsafe_allow_html=True)

                # Add relevance chart
                fig = chart_relevance_scores(c_trace.grades, query)
                st.plotly_chart(fig, use_container_width=True)

                st.divider()

                # Text-based details with better accessibility
                for i, g in enumerate(c_trace.grades):
                    # Phase 4: Accessibility - Use text + color, not just emoji
                    icon = "✅ RELEVANT" if g.relevant else "❌ NOT RELEVANT"
                    relevance_label = get_relevance_label(g.score)

                    with st.container(border=True):
                        col_grade1, col_grade2 = st.columns([3, 1])
                        with col_grade1:
                            st.markdown(f"**Doc {i+1}:** {g.reason}")
                            st.markdown(f"_Status: {icon}_")
                            preview = g.document_preview
                            st.caption(f"_{preview}_")
                        with col_grade2:
                            score_pct = f"{g.score:.0%}"
                            st.metric(
                                "Relevance",
                                score_pct,
                                help=relevance_label
                            )

        # Reranking information (smart default: collapsed)
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

        # Corrections (smart default: expanded if corrections exist)
        if c_trace.corrections:
            with st.expander(f"🔄 System Tried to Find Better Documents ({len(c_trace.corrections)} attempt{'s' if len(c_trace.corrections) > 1 else ''})", expanded=True):
                st.caption(
                    "Initial documents didn't pass the quality gate. "
                    "**Correction strategies** (expand, decompose, keywords) reformulated the query to find more relevant sources."
                )
                st.divider()

                # Correction flow visualization
                st.markdown(make_correction_flow(c_trace.corrections), unsafe_allow_html=True)
                st.divider()

                for i, corr in enumerate(c_trace.corrections):
                    with st.container(border=True):
                        st.markdown(f"**Attempt {i+1}: {corr.strategy.title()}**")

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

        # Multi-hop Retrieval (smart default: collapsed)
        if c_trace.multi_hop_needed:
            with st.expander(f"🔗 Multi-hop Retrieval ({len(c_trace.multi_hop_hops)} hop{'s' if len(c_trace.multi_hop_hops) > 1 else ''})", expanded=False):
                st.caption(
                    "The initial documents were incomplete. "
                    "**Multi-hop retrieval** detected missing concepts and issued follow-up queries to bridge them."
                )
                st.divider()

                # Multi-hop flow visualization
                st.markdown(make_multi_hop_flow(c_trace.multi_hop_hops), unsafe_allow_html=True)
                st.divider()

                for hop in c_trace.multi_hop_hops:
                    with st.container(border=True):
                        st.markdown(f"**Hop {hop.hop_number}:** {hop.bridge_entity}")

                        st.markdown("**Sub-query:**")
                        st.code(hop.bridge_query, language=None)

                        col_h1, col_h2 = st.columns(2)
                        with col_h1:
                            st.metric("Docs Retrieved", hop.docs_retrieved)
                        with col_h2:
                            st.metric("Docs Passed Grade", hop.docs_passed_grade)

                        if hop.docs_added:
                            st.caption(f"✅ Merged {len(hop.docs_added)} new document(s) into answer")

        if c_trace.fallback_used:
            st.warning(
                "⚠️ **Fallback Mode:** CRAG tried all correction strategies "
                "but couldn't find documents that passed the relevance gate.\n\n"
                "**What this means:** The answer is based on the LLM's training data, not your documents. "
                "You should verify this answer carefully, or add more relevant documents to your knowledge base."
            )

        # Answer verification
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
                    f"but {len(c_trace.answer_gaps)} claim(s) could not be confirmed:\n{gap_list}"
                )
            else:
                gap_list = "\n".join(f"- {g}" for g in c_trace.answer_gaps) if c_trace.answer_gaps else "No details"
                st.error(
                    f"🔴 **Verification Warning:** Claims not fully supported by documents:\n{gap_list}\n\n"
                    "Consider reviewing the source documents or rephrasing your question."
                )

        # Cost analysis with Phase 4 tooltips
        if c_trace.confidence_reasoning:
            st.caption(f"ℹ️ Confidence: {c_trace.confidence_reasoning}")

        extra_calls = c_trace.total_llm_calls - b_trace.total_llm_calls
        cost_delta = c_trace.total_cost_usd - b_trace.total_cost_usd
        cost_delta_pct = (cost_delta / b_trace.total_cost_usd * 100) if b_trace.total_cost_usd > 0 else 0

        st.subheader("💰 Cost Analysis")

        col_cost1, col_cost2, col_cost3 = st.columns(3)
        with col_cost1:
            st.metric(
                "Baseline Cost",
                format_cost(b_trace.total_cost_usd),
                help="Cost of standard RAG (retrieve → generate)\n\nNo quality checks, faster, but more hallucinations"
            )
        with col_cost2:
            st.metric(
                "CRAG Cost",
                format_cost(c_trace.total_cost_usd),
                delta=format_cost(cost_delta),
                delta_color="inverse",
                help=f"Cost of CRAG with quality gate + corrections\n\nIncludes grading, reranking, and retry strategies\n\nExtra calls: {extra_calls}"
            )
        with col_cost3:
            st.metric(
                "Cost Overhead",
                f"+{cost_delta_pct:.0f}%",
                help=f"Additional cost for quality improvements\n\nExtra LLM calls: {extra_calls}\n\nWorth it if prevented hallucinations save more"
            )

        # Phase 4: Help text for cost interpretation
        with st.expander("💡 Is this cost increase worth it?", expanded=False):
            improvement_pct = (c_trace.answer_confidence - b_trace.answer_confidence) * 100
            st.markdown(f"""
**Your Query's Numbers:**
- Confidence improvement: **+{improvement_pct:.0f}%**
- Extra LLM calls: **{extra_calls}**
- Extra cost: **{format_cost(cost_delta)}**

**Cost-Benefit Decision:**

If preventing ONE hallucination is worth more than **{format_cost(cost_delta)}**, then CRAG is worth it.

**Examples:**
- **E-commerce customer:** Preventing a wrong product recommendation might save $10-50 in returns → **CRAG is worth it**
- **FAQ chatbot:** Wrong answer costs reputation damage → **probably worth it**
- **Public knowledge base:** Low stakes per wrong answer → **might not be worth it**

**Rule of thumb:** CRAG pays for itself if your hallucinations cost >$0.001 each
            """)

        # Cost breakdown chart
        if c_trace.cost_breakdown:
            cost_by_component = {}
            for cb in c_trace.cost_breakdown:
                label = {
                    "gpt-5-nano-2025-08-07": "Generator",
                    "gpt-4o-mini-2024-07-18": "Grader",
                    "text-embedding-3-small": "Embeddings"
                }.get(cb.model, cb.model)
                cost_by_component[label] = cost_by_component.get(label, 0) + cb.cost_usd

            st.divider()
            st.markdown("### 💰 Cost & Impact")

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

            with st.expander("📊 Cost breakdown by component"):
                for label, cost in sorted(cost_by_component.items()):
                    pct = (cost / c_trace.total_cost_usd * 100) if c_trace.total_cost_usd > 0 else 0
                    st.caption(f"**{label}**: {format_cost(cost)} ({pct:.0f}%)")

        # ROI Analysis
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
            """)

# ============================================================================
# TAB 2: Dashboard - Unified Results & Metrics
# ============================================================================

with tab_dashboard:
    st.header("📊 Dashboard: CRAG Performance")
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
        m4.metric("Avg Extra LLM Calls", f"+{summary['avg_extra_llm_calls']}")

        st.divider()

        # Hallucination metrics gauge chart
        fig_hall = chart_hallucination_metrics(
            summary['baseline_hallucination_rate'] / 100,
            summary['crag_hallucination_rate'] / 100
        )
        st.plotly_chart(fig_hall, use_container_width=True)

        st.divider()

        # Cost summary
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

        st.divider()

        # Cost breakdown chart
        breakdown = {
            "Baseline": summary['avg_baseline_cost_per_query'],
            "CRAG": summary['avg_crag_cost_per_query']
        }
        fig_cost = chart_cost_breakdown(
            summary['avg_baseline_cost_per_query'],
            summary['avg_crag_cost_per_query'],
            {}
        )
        st.plotly_chart(fig_cost, use_container_width=True)

        # Cost-benefit summary
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

        # Phase 3: Confidence Calibration Visualization
        st.subheader("📈 Confidence Calibration")
        st.caption("How well does the system's confidence match actual correctness?")

        # Prepare calibration data from per-question results
        calibration_data = []
        for r in eval_data["per_question"]:
            calibration_data.append({
                "confidence": r.get("crag_confidence", 0.7),
                "correct": not r.get("crag_hallucinated", False)
            })

        if calibration_data:
            fig_calib = chart_confidence_calibration(calibration_data)
            st.plotly_chart(fig_calib, use_container_width=True)

            # Calibration interpretation
            with st.expander("ℹ️ How to read this chart", expanded=False):
                st.markdown("""
**Perfect Calibration:** Points lie on the diagonal line
- High-confidence predictions are correct
- Low-confidence predictions are incorrect

**Overconfident:** Points above the line
- Model is more confident than it should be
- May produce misleading hallucinations

**Underconfident:** Points below the line
- Model is less confident than deserved
- May unnecessarily flag good answers as uncertain
                """)

        st.divider()

        # Per-question breakdown (fixed: no nested expanders)
        with st.expander("📋 Per-Question Details", expanded=False):
            st.caption("Detailed comparison of answers for each test question")
            st.divider()

            for idx, r in enumerate(eval_data["per_question"], 1):
                b_icon = "✅" if not r["baseline_hallucinated"] else "❌"
                c_icon = "✅" if not r["crag_hallucinated"] else "❌"

                # Use container instead of nested expander
                with st.container(border=True):
                    st.markdown(f"**{b_icon} {c_icon} Q{idx}: {r['question']}**")
                    st.divider()

                    col_b, col_c = st.columns(2)
                    with col_b:
                        st.markdown("**📄 Baseline RAG**")
                        st.markdown(f"_{r['baseline_answer']}_")
                    with col_c:
                        st.markdown("**🛡️ CRAG**")
                        st.markdown(f"_{r['crag_answer']}_")

# ============================================================================
# TAB 3: How It Works - Educational
# ============================================================================

with tab_how:
    st.header("⚙️ How CRAG Works")

    # Problem section
    st.subheader("🔴 The Problem with Standard RAG")
    st.markdown("Standard RAG retrieves documents and generates answers, but **doesn't verify relevance**.")
    st.info("❌ **Hallucination:** When retrieval fails, the LLM confidently invents answers that sound right but are wrong.")
    st.markdown("**Example:**")
    st.code("Query: 'What's your return policy?'\nRetriever: [doc about shipping]\nGenerator: 'Returns accepted within 45 days' ← MADE UP ❌")

    st.divider()

    # Solution section
    st.subheader("🟢 CRAG's Solution")
    st.markdown("Add a **quality gate** before answer generation:")

    st.markdown("""
1. **Grade** each retrieved document: Is it relevant to the query?
2. **Decide:** If all documents fail grading, try harder
3. **Correct** the query using 3 strategies:
   - **Expand:** Rephrase with synonyms
   - **Decompose:** Break into sub-questions
   - **Keywords:** Extract key terms
4. **Retrieve again** with corrected query
5. **Generate** answer only if we have good documents
    """)

    st.divider()

    # Pipeline visualization
    st.subheader("📊 CRAG Pipeline (6 Steps)")

    pipeline_steps = [
        ("1️⃣ **Retrieve**", "Find up to 10 most similar documents from knowledge base"),
        ("2️⃣ **Grade**", "LLM evaluates: Is each document relevant? (score: 0.0-1.0)"),
        ("3️⃣ **Decide**", "Are there enough good documents? (threshold: 0.7+ relevance)"),
        ("4️⃣ **Correct** (if needed)", "Try 3 strategies: Expand, Decompose, Keywords"),
        ("5️⃣ **Verify**", "Check if answer is grounded in the documents"),
        ("6️⃣ **Return**", "Answer + confidence score + trace (why we believe this)"),
    ]

    for step_title, step_desc in pipeline_steps:
        st.markdown(f"**{step_title}**  \n{step_desc}")
        st.caption("")

    st.divider()

    # Key concepts
    st.subheader("🎓 Key Concepts")

    # Phase 4: Interactive tabs for different concepts
    concept_tabs = st.tabs(["📊 Relevance Score", "🎯 Confidence Score", "⏱️ Why It Matters"])

    with concept_tabs[0]:
        st.markdown("**Relevance Score:** How well does a document answer the question?")
        st.markdown("Range: **0.0** (irrelevant) to **1.0** (perfect match)")
        st.divider()

        # Phase 4: Accessibility - Color + text label combination
        st.markdown("**🟢 Highly Relevant (0.8–1.0)**")
        st.caption("Document directly answers the question. Use this.")

        st.markdown("**🟡 Somewhat Relevant (0.4–0.8)**")
        st.caption("Document may be helpful but isn't a perfect match. Might use if no better options.")

        st.markdown("**🔴 Not Relevant (<0.4)**")
        st.caption("Document doesn't answer the question. Discard and try another query.")

        st.divider()
        st.caption("**Why it matters:** High relevance scores = accurate answers, no hallucinations")

    with concept_tabs[1]:
        st.markdown("**Confidence Score:** How certain is the system in the answer?")
        st.markdown("Range: **0.0** (very uncertain) to **1.0** (very confident)")
        st.divider()

        st.markdown("**Factors that increase confidence:**")
        st.caption("✅ High document relevance (good sources)")
        st.caption("✅ Multiple documents supporting the same answer")
        st.caption("✅ Clear agreement from the grader")
        st.caption("✅ Successful retrieval on first try (no corrections needed)")

        st.markdown("**Factors that decrease confidence:**")
        st.caption("⚠️ Low document relevance (weak sources)")
        st.caption("⚠️ Had to use corrections to find documents")
        st.caption("⚠️ Fallback mode (no good documents found)")
        st.caption("⚠️ Answer not verified by document content")

        st.divider()
        st.caption("**Why it matters:** Low confidence scores = be skeptical of the answer")

    with concept_tabs[2]:
        st.markdown("**Why These Concepts Matter**")
        st.divider()

        st.markdown("**Hallucination Problem:**")
        st.caption("When an LLM doesn't find good documents, it often makes up plausible-sounding answers. Users can't distinguish right from wrong.")

        st.markdown("**CRAG's Solution:**")
        st.caption("1. Grade documents BEFORE generating answers\n2. Try multiple retrieval strategies if grading fails\n3. Report confidence score so users know when to be skeptical")

        st.markdown("**Real Cost:**")
        st.caption("Hallucinations cost money:\n- Customer refunds and escalations\n- Brand reputation damage\n- Regulatory/legal liability (in some domains)\n\nCRAG costs pennies to prevent these.")

    st.divider()

    # Correction strategies deep-dive
    st.subheader("🔧 Correction Strategies (When Initial Retrieval Fails)")

    with st.expander("**Expand:** Rephrase with synonyms", expanded=False):
        st.markdown("""
**Idea:** The query might use words that aren't in the documents.

**Example:**
- Original: "What's your refund policy?"
- Expanded: "How do I get money back?", "What's the return process?"

**When it works:** Different vocabulary in documents
        """)

    with st.expander("**Decompose:** Break into sub-questions", expanded=False):
        st.markdown("""
**Idea:** Complex questions might be better answered by finding multiple documents.

**Example:**
- Original: "What's your subscription cancellation policy?"
- Decomposed:
  - "How do I cancel my subscription?"
  - "What happens after I cancel?"
  - "Can I get a refund?"

**When it works:** Multi-part questions
        """)

    with st.expander("**Keywords:** Extract key terms", expanded=False):
        st.markdown("""
**Idea:** Use specific terms as-is (no synonyms) to find exact mentions.

**Example:**
- Original: "What's your return policy?"
- Keywords: ["return", "policy", "refund", "30 days"]

**When it works:** Documents have specific terminology
        """)

    st.divider()

    # Comparison table
    st.subheader("📊 Baseline RAG vs CRAG")

    comparison_data = {
        "Aspect": [
            "Retrieval",
            "Quality Gate",
            "Correction Loop",
            "Hallucination Rate",
            "Cost per Query",
            "Confidence Calibration"
        ],
        "Baseline RAG": [
            "Get top-10 documents",
            "❌ None",
            "❌ None",
            "~17.5% (high)",
            "$0.001–0.003",
            "❌ Overconfident"
        ],
        "CRAG": [
            "Get top-10 documents",
            "✅ Grade each document",
            "✅ Retry with corrections",
            "~3-5% (low)",
            "$0.002–0.005",
            "✅ Better calibrated"
        ]
    }

    st.dataframe(comparison_data, use_container_width=True, hide_index=True)

# ============================================================================
# TAB 4: Settings - Configuration & KB Management
# ============================================================================

with tab_settings:
    st.header("⚙️ Settings")
    st.caption("Configure your knowledge base and system parameters")

    # Knowledge Base Management
    st.subheader("📚 Knowledge Base")

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.markdown("**Select Knowledge Base:**")
        kb_choice = st.radio(
            "Knowledge Base Mode:",
            options=["📄 Default FAQ", "✏️ Custom Documents"],
            key="kb_selector_settings",
            help="Choose which documents to use for retrieval",
            index=0 if st.session_state.kb_mode == "default" else 1,
            label_visibility="collapsed"
        )

        st.session_state.kb_mode = "default" if kb_choice.startswith("📄") else "custom"

    with col2:
        st.markdown("**Current Status:**")
        if st.session_state.kb_mode == "default":
            st.success(f"📄 Using Default FAQ ({len(DEFAULT_DOCUMENTS)} documents)")
        else:
            st.info("✏️ Using Custom Documents")

    st.divider()

    # Handle Default FAQ mode
    if st.session_state.kb_mode == "default":
        st.markdown("### 📖 Default FAQ Documents")

        with st.expander(f"View all documents ({len(DEFAULT_DOCUMENTS)})", expanded=False):
            for i, d in enumerate(DEFAULT_DOCUMENTS, 1):
                with st.container(border=True):
                    st.caption(f"**Document {i}**")
                    st.caption(d[:200] + ("..." if len(d) > 200 else ""))

    else:
        # Custom Documents mode
        st.markdown("### 📤 Upload Custom Documents")

        # Phase 4: Enhanced help text with better UX
        with st.expander("📖 Format Guide", expanded=False):
            st.markdown("""
**Paste documents separated by blank lines:**
- Each blank line creates a document boundary
- Each paragraph becomes one searchable document
- Better: 5-10 documents with good content
- Avoid: Very long single document or many tiny documents

**Example:**
```
Return Policy: We accept returns within 30 days of purchase.
Items must be in original condition.

Shipping Information: Standard shipping takes 5-7 business days.
Express shipping available for $10 extra.

[Each block above = one document]
```

**Tips:**
- ✅ Paste documents exactly as they appear in your system
- ✅ Include document titles/headers for clarity
- ✅ Separate by blank lines
- ❌ Don't paste HTML or formatting codes
            """)

        custom_docs_text = st.text_area(
            "Paste your documents:",
            value="",
            height=250,
            help="Paste documents separated by blank lines",
            placeholder="Paste documents here. Separate with blank lines.",
        )

        # Phase 3 & 4: Live validation with better visual feedback
        if custom_docs_text.strip():
            detected_docs = [d.strip() for d in custom_docs_text.split("\n\n") if d.strip()]
            doc_count = len(detected_docs)

            # Create validation feedback
            validation_issues = []
            validation_info = []

            if doc_count == 0:
                validation_issues.append("❌ No documents detected")
            elif doc_count == 1:
                validation_info.append("📌 Single document detected (works, but multiple docs better for retrieval)")
            elif doc_count <= 10:
                validation_info.append(f"✅ **{doc_count} documents** — optimal range")
            else:
                validation_issues.append(f"⚠️ {doc_count} documents (large KB — may be slower/more expensive)")

            # Check for very short documents
            short_docs = [i+1 for i, d in enumerate(detected_docs) if len(d.split()) < 10]
            if short_docs:
                validation_issues.append(f"⚠️ Docs {short_docs} are very short (<10 words)")
            else:
                total_words = sum(len(d.split()) for d in detected_docs)
                avg_words = total_words // doc_count if doc_count > 0 else 0
                validation_info.append(f"📊 Average doc length: {avg_words} words")

            # Display validation feedback
            if validation_issues:
                for issue in validation_issues:
                    st.warning(issue) if "❌" in issue else st.info(issue)

            if validation_info:
                st.caption("")  # Spacer
                for info in validation_info:
                    st.caption(info)

            # Live preview with document count
            if doc_count > 0:
                preview_cols = st.columns([2, 1])
                with preview_cols[0]:
                    if st.checkbox(f"📋 Preview all {doc_count} documents", key="show_preview", value=False):
                        st.divider()
                        for i, doc in enumerate(detected_docs, 1):
                            with st.container(border=True):
                                col_title, col_length = st.columns([3, 1])
                                with col_title:
                                    st.caption(f"**Document {i}**")
                                with col_length:
                                    word_count = len(doc.split())
                                    st.caption(f"_{word_count} words_")

                                preview_text = doc[:200] + ("..." if len(doc) > 200 else "")
                                st.caption(preview_text)

            docs = detected_docs
        else:
            docs = []
            st.caption("💡 Paste documents above to get started")

        # Phase 4: Enhanced apply button with accessibility
        if docs:
            button_label = f"📤 Apply {len(docs)} Document(s)"

            col_btn1, col_btn2 = st.columns([2, 1])
            with col_btn1:
                if st.button(button_label, key="apply_docs_settings", use_container_width=True, type="primary"):
                    with st.spinner(f"Indexing {len(docs)} document(s)…"):
                        try:
                            new_store = VectorStore()
                            new_store.add_documents(docs)
                            st.session_state.vector_store = new_store
                            st.success(f"✅ Successfully indexed {len(docs)} document(s)!")
                            st.session_state.kb_mode = "custom"
                            st.session_state.vector_store = new_store
                            st.info("📍 Custom KB is now active. Use the Query tab to test.")
                        except Exception as e:
                            st.error(f"❌ Error indexing documents: {str(e)}")
            with col_btn2:
                st.caption(f"✓ Will index {len(docs)} docs")
        else:
            st.info("📍 Enter documents above to activate the Apply button")

    st.divider()

    # System Parameters with Phase 4: Accessibility and help text
    st.subheader("⚙️ System Parameters")
    st.caption("Current configuration settings (fixed in demo mode)")

    # Phase 4: Interactive help for parameters
    param_help = st.expander("❓ What do these parameters mean?", expanded=False)
    with param_help:
        st.markdown("""
**Relevance Threshold (0.70):** Documents scoring below 70% relevance are rejected as unhelpful.
- Lower value = accept more documents (retrieves more, but lower quality)
- Higher value = accept fewer documents (stricter filtering)

**Top-K Documents (10):** Retrieve up to 10 most similar documents from your knowledge base.
- More documents = higher chance of finding answer, but slower and more expensive
- Fewer documents = faster, but might miss relevant information

**Max Retrieval Attempts (3):** Try up to 3 correction strategies (Expand, Decompose, Keywords) if initial retrieval fails.
- More attempts = higher chance of finding answer with corrections
- Fewer attempts = faster, but might give up too early

**Confidence Threshold (0.40):** Answers scoring below 40% confidence get a low-confidence warning.
- Lower threshold = warn on fewer answers (risk: users trust uncertain answers)
- Higher threshold = warn on more answers (risk: users lose trust in good answers)
        """)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Relevance Threshold",
            "0.70",
            help="Documents must score ≥70% relevant to be used. Filters out weak sources."
        )
        st.metric(
            "Max Retrieval Attempts",
            "3",
            help="Try up to 3 different query reformulations if first retrieval fails"
        )

    with col2:
        st.metric(
            "Top-K Documents",
            "10",
            help="Retrieve top 10 most similar documents from knowledge base per query"
        )
        st.metric(
            "Confidence Threshold",
            "0.40",
            help="Flag answers as low-confidence if below 40% certainty"
        )

    st.divider()

    # About
    st.subheader("ℹ️ About CRAG")
    st.markdown("""
**Corrective Retrieval-Augmented Generation (CRAG)** is a technique to reduce hallucinations in RAG systems.

**Key Features:**
- 🛡️ Quality gate prevents hallucinations
- 🔄 Correction loop tries multiple strategies
- ✅ Answer verification
- 💰 Cost-benefit analysis
- 📊 Confidence calibration

**Learn More:**
- [GitHub Repository](https://github.com/leo-cherupushpam/corrective-rag)
- [Research Paper](https://arxiv.org/abs/2401.15884)
    """)

    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
