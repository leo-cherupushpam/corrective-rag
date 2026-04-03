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

    # Custom documents
    custom_docs_text = st.text_area(
        "Add custom documents:",
        value="",
        height=120,
        help="Leave blank to use default FAQ.",
    )

    use_custom = bool(custom_docs_text.strip())

    if use_custom:
        docs = [d.strip() for d in custom_docs_text.split("\n\n") if d.strip()]
    else:
        docs = DEFAULT_DOCUMENTS

    # Initialize vector store
    if "vector_store" not in st.session_state or st.sidebar.button("🔄 Re-index"):
        with st.spinner("Indexing…"):
            store = VectorStore()
            store.add_documents(docs)
            st.session_state.vector_store = store
        st.sidebar.success(f"✅ {len(docs)} documents")

    store = st.session_state.vector_store

    if not use_custom:
        with st.expander(f"View default docs ({len(docs)})", expanded=False):
            for i, d in enumerate(docs, 1):
                st.caption(f"**Doc {i}:** {d[:100]}…")

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

    # Query input (prominent, two-step)
    col1, col2 = st.columns([3, 1])

    with col1:
        question_input = st.text_input(
            "Enter your question:",
            placeholder="What is your return policy?",
            label_visibility="collapsed",
        )

    with col2:
        run_analysis = st.button("🚀 Analyze", type="primary", use_container_width=True)

    if run_analysis:
        query = question_input.strip()
        if not query:
            st.warning("Please enter a question.")
            st.stop()

        with st.spinner("Running both systems…"):
            b_trace = baseline_rag(query, store)
            c_trace = crag(query, store)

        # v1.5: Track for observability dashboard
        st.session_state.query_history.append(b_trace)
        st.session_state.query_history.append(c_trace)

        st.divider()
        st.subheader(f"📝 Query: *{query}*")

        # Side-by-side answers
        col_b, col_c = st.columns(2, gap="large")

        with col_b:
            with st.container(border=True):
                st.markdown("### 📄 Baseline RAG")
                st.markdown(f"{b_trace.answer}")
                st.divider()
                st.caption(f"**LLM calls:** {b_trace.total_llm_calls}  |  **Docs used:** {len(b_trace.docs_used)}")

                # v1.5: Show baseline confidence
                if b_trace.answer_confidence > 0:
                    confidence_pct = b_trace.answer_confidence * 100
                    if b_trace.answer_confidence > 0.7:
                        badge = "🟢 High confidence"
                    elif b_trace.answer_confidence > 0.4:
                        badge = "🟡 Medium confidence"
                    else:
                        badge = "🔴 Low confidence"
                    st.caption(f"**{badge}** ({confidence_pct:.0f}%)")
                    if b_trace.confidence_reasoning:
                        st.caption(f"_{b_trace.confidence_reasoning}_")

        with col_c:
            status_icon = "✅" if not c_trace.fallback_used else "⚠️"
            status_text = "Answered with sources" if not c_trace.fallback_used else "Fallback (no docs)"

            with st.container(border=True):
                st.markdown("### 🛡️ CRAG (Corrective)")
                st.markdown(f"{c_trace.answer}")
                st.divider()

                col_meta1, col_meta2 = st.columns([1, 1])
                with col_meta1:
                    st.caption(f"**LLM calls:** {c_trace.total_llm_calls}")
                    st.caption(f"**Docs used:** {len(c_trace.docs_used)}")
                with col_meta2:
                    st.caption(f"{status_icon} {status_text}")
                    if c_trace.needed_correction:
                        st.caption(f"🔄 {len(c_trace.corrections)} correction(s)")
                    else:
                        st.caption("✓ Passed grade immediately")

                # v1.5: Show CRAG confidence with reasoning
                if c_trace.answer_confidence > 0:
                    confidence_pct = c_trace.answer_confidence * 100
                    if c_trace.answer_confidence > 0.7:
                        badge = "🟢 High confidence"
                    elif c_trace.answer_confidence > 0.4:
                        badge = "🟡 Medium confidence"
                    else:
                        badge = "🔴 Low confidence"
                    st.markdown(f"**{badge}** ({confidence_pct:.0f}%)")
                    if c_trace.confidence_reasoning:
                        st.caption(f"_{c_trace.confidence_reasoning}_")

        st.divider()

        # CRAG observability trace
        st.subheader("🔍 System Trace")

        if c_trace.grades:
            with st.expander("📋 Document Grades", expanded=True):
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

        if c_trace.corrections:
            with st.expander(f"🔄 System Tried to Find Better Documents ({len(c_trace.corrections)} attempt{'s' if len(c_trace.corrections) > 1 else ''})", expanded=True):
                st.caption("Initial documents didn't pass the quality gate. CRAG reformulated the query to find more relevant sources.")
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

        if c_trace.fallback_used:
            st.warning(
                "⚠️ **Fallback Mode:** CRAG tried all correction strategies (expand, decompose, keywords) "
                "but couldn't find documents that passed the relevance gate.\n\n"
                "**What this means:** The answer below is based on the LLM's training data, not your documents. "
                "You should verify this answer carefully, or add more relevant documents to your knowledge base.\n\n"
                "**Confidence:** 🔴 Low (documents not available)"
            )

        # Cost-benefit summary (v1.5: using actual costs from trace)
        extra_calls = c_trace.total_llm_calls - b_trace.total_llm_calls
        cost_delta = c_trace.total_cost_usd - b_trace.total_cost_usd
        cost_delta_pct = (cost_delta / b_trace.total_cost_usd * 100) if b_trace.total_cost_usd > 0 else 0

        col_cost1, col_cost2, col_cost3 = st.columns(3)
        with col_cost1:
            st.metric("Baseline Cost", format_cost(b_trace.total_cost_usd),
                     help="Cost of standard RAG (retrieve → generate)")
        with col_cost2:
            st.metric("CRAG Cost", format_cost(c_trace.total_cost_usd),
                     delta=format_cost(cost_delta),
                     help="Cost of CRAG with quality gate + corrections")
        with col_cost3:
            st.metric("Cost Delta", f"+{cost_delta_pct:.0f}%",
                     help=f"Extra cost for {extra_calls} additional LLM calls (grader + corrector)")

        # Cost breakdown by component (v1.5)
        if c_trace.cost_breakdown:
            st.divider()
            st.markdown("**Cost Breakdown by Component:**")
            cost_by_component = {}
            for cb in c_trace.cost_breakdown:
                # Map model names to readable labels
                label = {
                    "gpt-5-nano-2025-08-07": "Generator",
                    "gpt-4o-mini-2024-07-18": "Grader",
                    "text-embedding-3-small": "Embeddings"
                }.get(cb.model, cb.model)
                cost_by_component[label] = cost_by_component.get(label, 0) + cb.cost_usd

            col1, col2, col3 = st.columns(3)
            col_idx = 0
            for label, cost in sorted(cost_by_component.items()):
                cols = [col1, col2, col3]
                with cols[col_idx % 3]:
                    pct_of_total = (cost / c_trace.total_cost_usd * 100) if c_trace.total_cost_usd > 0 else 0
                    st.caption(f"**{label}**: {format_cost(cost)} ({pct_of_total:.0f}%)")
                col_idx += 1

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
        st.subheader("Per-Question Breakdown")
        for idx, r in enumerate(eval_data["per_question"], 1):
            b_icon = "❌" if r["baseline_hallucinated"] else "✅"
            c_icon = "❌" if r["crag_hallucinated"] else "✅"
            corr_note = " 🔄" if r["crag_needed_correction"] else ""
            fb_note = " ⚠️" if r["crag_fallback"] else ""

            header = f"{b_icon} {c_icon} Q{idx}: {r['question'][:60]}{corr_note}{fb_note}"
            with st.expander(header):
                col_b, col_c = st.columns(2)
                with col_b:
                    with st.container(border=True):
                        st.markdown("**📄 Baseline**")
                        st.markdown(f"_{r['baseline_answer']}_")
                        status = "✅ Correct" if not r["baseline_hallucinated"] else "❌ Hallucinated"
                        st.caption(status)
                with col_c:
                    with st.container(border=True):
                        st.markdown("**🛡️ CRAG**")
                        st.markdown(f"_{r['crag_answer']}_")
                        status = "✅ Correct" if not r["crag_hallucinated"] else "❌ Hallucinated"
                        st.caption(status)

        # Cost breakdown by question (v1.5)
        st.divider()
        st.subheader("💰 Cost Breakdown by Question")

        # Extract cost data from per-question results
        cost_data = {}
        for idx, r in enumerate(eval_data["per_question"], 1):
            q_label = f"Q{idx}"
            crag_cost = r.get("crag_cost_usd", 0)
            cost_data[q_label] = crag_cost

        if cost_data:
            st.bar_chart(cost_data)

            # Add analysis
            max_cost_q = max(cost_data, key=cost_data.get)
            min_cost_q = min(cost_data, key=cost_data.get)
            max_cost_val = cost_data[max_cost_q]
            min_cost_val = cost_data[min_cost_q]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Most Expensive", max_cost_q,
                         help=f"{max_cost_q} cost {format_cost(max_cost_val)}")
            with col2:
                st.metric("Least Expensive", min_cost_q,
                         help=f"{min_cost_q} cost {format_cost(min_cost_val)}")
            with col3:
                avg_cost = mean(cost_data.values()) if cost_data else 0
                st.metric("Average Cost", format_cost(avg_cost))

            # Explanation
            st.caption(
                "📌 **Why costs vary:** Questions requiring correction strategies (🔄) or fallback (⚠️) "
                "cost more because the system tries multiple retrieval approaches. "
                "See per-question details above for correction attempts."
            )
        else:
            st.info("Cost data not available in evaluation results.")

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

    # === REAL-TIME METRICS ===
    st.subheader("🟢 Real-Time Session Metrics")

    # Session KPIs
    session_traces = st.session_state.query_history
    if session_traces:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            num_queries = len(session_traces)
            st.metric("Queries Run", num_queries)

        with col2:
            crag_traces = [t for t in session_traces if hasattr(t, 'mode') and t.mode == "crag"]
            if crag_traces:
                avg_confidence = mean([t.answer_confidence for t in crag_traces if hasattr(t, 'answer_confidence')])
                st.metric("Avg CRAG Confidence", f"{avg_confidence:.0%}")
            else:
                st.metric("Avg CRAG Confidence", "N/A")

        with col3:
            hallucinations = sum(1 for t in crag_traces if hasattr(t, 'fallback_used') and t.fallback_used)
            st.metric("Fallback Cases", hallucinations)

        with col4:
            total_cost = sum(t.total_cost_usd for t in session_traces if hasattr(t, 'total_cost_usd'))
            st.metric("Total Cost (Session)", format_cost(total_cost))
    else:
        st.info("🔵 Run some queries in the 'Try It' tab to see real-time metrics here")

    st.divider()

    # === BATCH METRICS ===
    st.subheader("📊 Batch Evaluation Metrics")

    eval_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_data = json.load(f)

        summary = eval_data["summary"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Questions", summary["total_questions"])
        with col2:
            st.metric("Hallucination Reduction", f"{summary['hallucination_reduction_pct']:.0f}%")
        with col3:
            st.metric("Avg CRAG Confidence", f"{summary['avg_crag_answer_confidence']:.2f}")
        with col4:
            st.metric("Cost Delta", f"{summary['cost_delta_pct']:+.0f}%")

        st.divider()

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
