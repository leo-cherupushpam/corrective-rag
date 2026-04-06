# PRD: Corrective RAG (CRAG) System
**Version:** 1.5
**Status:** Production Beta
**Author:** Leo Cherupushpam (AI Product Manager)
**Last Updated:** 2026-04-03

---

## 1. Problem Statement

> "RAG systems hallucinate when retrievers fail to find relevant documents. Users get confident-sounding wrong answers instead of admitting uncertainty."

### Root Problems

| Problem | Evidence | Severity |
|---|---|---|
| Retrieval failure causes hallucination | Irrelevant docs passed to LLM = LLM "makes up" answer | Critical |
| No quality gate on retrieval | System never checks: "are these docs actually relevant?" | High |
| Users can't distinguish good answers from hallucinations | Both look equally confident | High |
| One-shot retrieval wastes attempts | Single query may miss good docs; no refinement loop | Medium |

### Why This Matters for AI Product Managers

RAG is everywhere (customer support, knowledge bases, legal docs). But standard RAG fails silently. As a PM, you need to know:
- How do you measure if your RAG is working?
- When does retrieval fail, how do you recover?
- How do you build trust with users when some answers are hallucinations?

---

## 2. User Personas

### Primary Persona: "Cautious Carla" (Enterprise)
- **Role:** Knowledge manager at a Fortune 500 company
- **JTBD:** "Deploy RAG on our KB without embarrassing hallucinations"
- **Pain Points:**
  - Legal/compliance team rejects RAG because answers sometimes invent citations
  - Can't afford one hallucination per 10 answers (trust broken)
  - Needs confidence scores, not just answers
- **Tech comfort:** High — understands API integration

### Secondary Persona: "Startup Sam"
- **Role:** Co-founder building AI-powered chat support
- **JTBD:** "Use RAG without spending on human fact-checkers"
- **Pain Points:**
  - Limited budget for LLM calls; can't afford multiple retrievals per query
  - Customers complaining: "Your bot made up that feature"
  - No monitoring — don't know which questions cause hallucinations
- **Tech comfort:** High — will integrate anything if it improves answers

---

## 3. Solution: Corrective RAG (CRAG)

A retrieval quality gate that catches bad retrievals and corrects them before generation.

### The Pattern

```
User Query
    ↓
[1] Retrieve documents (embedding-based)
    ↓
[2] Grade: Are these docs relevant? (LLM-as-judge)
    ├─ YES → [4] Generate answer (use retrieved docs)
    └─ NO  → [3] Correct: Reformulate query or try keyword search
                  ↓
              Re-retrieve with new strategy
                  ↓
              [2] Grade again
                  ├─ YES → [4] Generate
                  └─ NO  → [5] Fallback: "I don't have docs, answering from training data"
    ↓
[4] Answer with source citations
```

### Key Innovations

1. **Grader Agent** — LLM evaluates: "Is retrieved doc relevant to query?" (binary or score)
2. **Correction Loop** — Reformulate query if grade fails (e.g., expand keywords, decompose complex query)
3. **Fallback Strategy** — If retrieval exhausted, either:
   - Return "I don't know" (safest)
   - Answer without sources + explicit disclaimer (transparent)
4. **Observability** — Log every grade decision → understand where RAG fails

### What It Is NOT
- Not a full reranker (though could integrate one)
- Not fixing hallucinations after they happen
- Not replacing human fact-checking (but reduces false positives)

---

## 4. Desired Outcome

> **"Reduce hallucination rate from 15–20% (baseline RAG) to <5% while maintaining answer utility."**

Users should be able to trust RAG answers because the system filters out low-confidence retrievals before they cause hallucinations.

### Outcome vs. Output

| Output (what we build) | Outcome (what we measure) |
|---|---|
| Retrieval grader component | Precision: does grader correctly identify relevant docs? |
| Query reformulation loop | Recall: can we recover from failed retrieval? |
| Confidence scores | User trust: do users rely on confidence scores? |
| Observability dashboards | System health: what % of queries need correction? |
| Integration examples | Adoption: how many teams use CRAG vs. baseline RAG? |

### Success Metrics

| Metric | Baseline (Standard RAG) | Target (CRAG) | How Measured |
|---|---|---|---|
| Hallucination rate | 15–20% | <5% | Manual eval of 100 Q&A pairs |
| Grader accuracy (identifies relevant docs) | — | >90% precision | Held-out eval set |
| Correction loop success rate | N/A | >60% (fix rate on failed retrievals) | Logs: % of re-retrievals that pass grade |
| Answer coverage (% of queries answered) | ~95% | >90% | Queries returning "I don't know" |
| User confidence (NPS proxy) | — | >50 | Survey: "Do you trust this answer?" |

---

## 5. Architecture

### Core Components

1. **Retriever** (vector store)
   - Existing: BM25 or embedding-based (e.g., OpenAI embeddings + FAISS)
   - No changes needed for CRAG baseline

2. **Grader Agent** (LLM-as-judge)
   - Input: (user_query, retrieved_docs)
   - Output: {"relevant": bool, "score": 0–1, "reason": str}
   - Model: GPT-4o (accurate judgments) or smaller model for cost (Claude Haiku)

3. **Correction Module**
   - If grade = NO, choose strategy:
     - **Query expansion:** Add synonyms, decompose complex Q
     - **Keyword fallback:** Switch to sparse retrieval (BM25) if dense failed
     - **Vector re-embed:** Try different embedding query
   - Max 2–3 correction attempts (avoid infinite loops)

4. **Generator** (LLM answer synthesis)
   - Input: (user_query, graded_docs)
   - Output: answer + source citations + confidence
   - Constraint: "Cite only from provided docs; if missing info, say so"

5. **Observability**
   - Log every decision: query → retrieval → grade → correction → answer
   - Metrics: grade distribution, correction success rate, hallucination rate

### Why This Architecture

- **Minimal changes to existing RAG:** Grader is additive, doesn't replace retriever
- **Cost-effective:** One extra LLM call (grader) is cheaper than repeated re-retrievals
- **Debuggable:** Logs show exactly where/why hallucinations happen
- **Extensible:** Can swap grader model, add different correction strategies

---

## 6. AI-Specific Design Decisions

### 6.1 Grader Model: GPT-4o vs. Haiku
- **Decision:** Start with Haiku for cost, validate with GPT-4o
- **Why:** Grading is a simpler task than generation; smaller model likely sufficient
- **Tradeoff:** Haiku may miss subtle relevance; GPT-4o is safer but 3x cost
- **Revisit trigger:** If Haiku grade accuracy <85%, switch to GPT-4o

### 6.2 Correction Loop Depth
- **Decision:** Max 2 correction attempts per query
- **Why:** Prevents infinite loops; diminishing returns after 2 tries
- **Tradeoff:** Some recoverable failures missed if we stop early
- **Revisit trigger:** If >30% of corrections succeed at attempt 2, allow 3

### 6.3 Grading Threshold
- **Decision:** Binary (relevant/not relevant), not continuous score
- **Why:** Simpler decision logic, easier to interpret
- **Tradeoff:** Loses nuance (somewhat relevant, mostly relevant, etc.)
- **Revisit trigger:** If edge cases accumulate (borderline relevance), move to scoring

---

## 7. Success Roadmap

### v1.0 — Proof of Concept ✅ COMPLETE
- [x] Implement grader agent (simple yes/no relevance)
- [x] Implement correction loop (query expansion + BM25 fallback)
- [x] Baseline RAG (retriever + generator, no grading)
- [x] Manual evaluation: 100 Q&A pairs, compare baseline vs. CRAG
- [x] Document: where does each approach fail?

### v1.5 — Production Hardening ✅ COMPLETE (2026-04-03)
- [x] Cost optimization (Haiku vs. GPT-4o tradeoff analysis)
- [x] Observability dashboard (grade distribution, correction success rate)
- [x] Confidence scores in output (not just answer, but "how sure?")
- [x] Integration example: customer support chatbot
- [x] Cost analysis feature with real-time cost tracking
- [x] Knowledge Base sidebar improvements (radio buttons, validation, preview)
- [x] Comprehensive UI/UX refresh (all 4 tabs restructured for clarity)
- [x] Sample questions for easier exploration
- [x] Grading trace clarity (expanded previews, score explanations)
- [x] Visual hierarchy improvements (Info boxes, callouts, better spacing)

**v1.5 Highlights:**
- Cost information consolidated from 5+ locations to single clear section
- Information overload reduced: Tab 4 metrics reduced from 8 to 6 cards
- Visual hierarchy improved with Streamlit components (subheaders, info boxes)
- Tab 3 restructured with column-based pipeline visualization
- Sample question buttons for faster demo exploration
- Real-time cost tracking and confidence score calculations

### v2.0 — Advanced (In Progress)
- [ ] Continuous reranking (integrate Cohere/BGE reranker)
- [ ] Multi-hop retrieval (for complex questions needing 3+ docs)
- [ ] Fine-tuned grader (on domain-specific labeled data)
- [ ] A/B testing framework (CRAG vs. standard RAG head-to-head)
- [ ] Advanced metrics dashboard (confidence calibration, strategy effectiveness analysis)

---

## 8. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Grader is wrong (says relevant docs irrelevant) | Medium | High | Validate grader on eval set; allow human override |
| Correction loop doesn't improve retrieval | Medium | Medium | If >3 corrections fail, fast-fail to "I don't know" |
| Cost spike from multiple retrievals | Low | Medium | Monitor cost/query; set max correction budget |
| Hallucination still happens (grader passes bad docs) | Medium | High | Combine grader + answer-level fact-checking (v2.0) |

---

## 9. How We'll Know This Works

### Day 1: System works
- CRAG runs without errors
- Grader outputs reasonable judgments
- Correction loop triggers on bad retrievals

### Week 1: Baseline comparison
- Evaluate CRAG vs. standard RAG on 100 test Q&As
- Measure: hallucination rate, precision, recall
- Identify which types of queries benefit most from CRAG

### Month 1: Production readiness
- Document cost per query (grader + correction loops)
- Build observability dashboard
- Gather feedback from 2–3 internal teams

### Success: >50% hallucination reduction
- CRAG achieves <5% hallucination rate
- Cost per query still < baseline RAG + human fact-checking
- Teams actively using CRAG in production

---

## 10. Open Questions

1. What's the right grading threshold? (Binary vs. confidence score)
2. How many correction attempts before we give up?
3. Should we combine CRAG with answer-level fact-checking?
4. What domains/question types benefit most from CRAG?
5. Can fine-tuning a small grader model match GPT-4o at 10x lower cost?

---

## 11. v1.5 Implementation Summary (April 2026)

### Feature: Cost Analysis Surfacing
**Goal:** Help users understand and justify the cost-value tradeoff of CRAG

**8 Improvements Implemented:**
1. Real-time cost tracking with per-query breakdown (generator, grader, embeddings)
2. Cost comparison visualization (baseline vs CRAG side-by-side)
3. Consolidated cost information (removed 5+ redundant cost sections)
4. Cost-benefit metrics (prevented hallucinations per dollar spent)
5. ROI calculator in Observability tab
6. Cost projection tool (yearly extrapolation based on query volume)
7. Model optimization tool (compare grader models: Haiku vs GPT-4o-mini)
8. Clear cost explainer text (why CRAG costs more, when it's worth it)

**Impact:** Users can now make informed cost decisions with transparent, real-time data

### Feature: Knowledge Base Sidebar Improvements
**Goal:** Make KB management explicit, safe, and discoverable

**Improvements Implemented:**
1. Radio button selection (replaces implicit "empty = default" behavior)
2. Active KB badge showing which KB is currently in use
3. Live document count preview (updates as user types)
4. Parse preview expander (shows how documents will be split before indexing)
5. Input validation (prevents empty KB, warns on 10+ documents)
6. Better button labeling ("Apply Custom Documents" instead of "Re-index")
7. Format example in help text (users know what's expected)
8. Clearer error messages (validation feedback)

**Impact:** New users can confidently manage knowledge bases without confusion

### Feature: UI/UX Comprehensive Refresh
**Goal:** Reduce visual clutter and improve information hierarchy

**Tab 1: Try It Simplification**
- Consolidated cost sections: 3 separate cost explainers → 1 "Cost & Impact" section
- Removed cost-benefit summary expander (moved to Tab 4)
- Simplified results display: removed duplicate confidence badges
- Better visual distinction between Baseline and CRAG columns
- System Trace kept focused (Document Grades and Corrections expandable)
- Removed ~50 lines of redundant cost content

**Tab 2: Results Reorganization**
- Reorganized metrics into Quality and Efficiency groups
- Collapsed per-question breakdown by default (expanders for curiosity)
- Moved explanations to bottom of page (less prominent)
- Removed individual cost breakdown per question
- Cleaner empty state messaging

**Tab 3: How It Works Polish**
- Added visual hierarchy with proper Streamlit components
- Info boxes for key concepts (quality gate, correction strategies, cost model)
- Column-based pipeline visualization for CRAG process
- Better spacing and readability improvements
- More visual breaks (no long walls of text)
- Consistent formatting and callout styles

**Tab 4: Observability Streamlining**
- Reduced metric cards from 8 to 6 (consolidated redundant metrics)
- Organized by topic rather than data type
- Removed excessive dividers (tab structure handles organization)
- Added conditional rendering (show metrics only if data exists)
- Consistent chart styling and captions

**Overall Impact:**
- 333 lines of code removed (reduced clutter)
- Visual hierarchy dramatically improved
- Information redundancy eliminated
- Easier user navigation and understanding

### Feature: Demo Usability Improvements
**Implemented:**
1. Sample question buttons (6 common questions for easy exploration)
2. Automatic text population on button click (no manual typing)
3. Proper session state management (selections persist across reruns)

**Impact:** New users can immediately explore the system without writing queries

### Metrics Achieved
- Cost analysis: All 8 data points exposed to user (vs 0 before)
- UI consistency: All 4 tabs now follow unified visual language
- Information reduction: ~15% of code removed, clarity improved
- User journey: Can complete demo in 2 minutes (vs 5+ minutes previously)

---

## 12. Context: Why This Project Matters

This project demonstrates:
- **Understanding RAG limitations** (hallucination root causes)
- **Systems thinking** (quality gates, feedback loops, fallbacks)
- **Pragmatic engineering** (cost tradeoffs, observability, iteration)
- **Product rigor** (baselines, metrics, v1 → v2 progression)

Unlike many "AI projects," CRAG solves a real production problem: how do you deploy RAG systems safely?
