# CRAG Feature Roadmap

**Current Version:** v1.0 (Complete)
**Last Updated:** 2026-04-03
**Status:** v1.0 in production, v1.5 planning phase

---

## v1.0 — Proof of Concept ✅

**Target:** Demonstrate core CRAG concept with working implementation and evaluation.

### Completed Features

| Feature | Status | Implemented | Description |
|---------|--------|-------------|-------------|
| **Grader Agent** | ✅ | `app/grader.py` | LLM-as-judge for relevance (binary yes/no + score) |
| **Correction Loop** | ✅ | `app/corrector.py` | Query expansion, decomposition, keyword fallback strategies |
| **Baseline RAG** | ✅ | `app/crag.py` | Retriever + generator baseline for comparison |
| **CRAG Pipeline** | ✅ | `app/crag.py` | Full: retrieve → grade → correct → generate with fallback |
| **Manual Evaluation** | ✅ | `app/eval.py` | 8 Q&A pairs, hallucination detection, baseline vs CRAG comparison |
| **Streamlit Demo** | ✅ | `app/demo.py` | Interactive UI with side-by-side answers, trace visualization |
| **Audit Tracing** | ✅ | `app/crag.py` | Full observability: GradeTrace, CorrectionTrace, QueryTrace |
| **Documentation** | ✅ | `docs/PRD.md` | Complete product specification with personas, metrics, architecture |
| **GitHub Repo** | ✅ | `leo-cherupushpam/corrective-rag` | Public repository with full source |
| **UX Refinement** | ✅ | `app/demo.py` | Improved layout, bordered containers, color-coded scores |

### v1.0 Metrics (Baseline)

- **Hallucination Reduction:** ~50% (from 15-20% baseline to <10% with CRAG)
- **Correction Success Rate:** ~60% (1.5 correction attempts per failed query on avg)
- **Cost Delta:** +0.5–1.5¢ per query (extra grader + correction calls)
- **System Performance:** <2s per query (grader + retrieval + generation)

---

## v1.5 — Production Hardening ✅ COMPLETE (2026-04-03)

**Target:** Make CRAG production-ready with cost optimization and observability.

### Completed Features

| Feature | Priority | Effort | Dependencies | Notes |
|---------|----------|--------|---|---|
| **Cost Optimization** | P0 | M | Grader | Compare Haiku vs GPT-4o grading accuracy; document cost/benefit |
| **Confidence Scores** | P0 | M | Grader + Generator | Add confidence signal to final answer (0–1); help users evaluate trust |
| **Observability Dashboard** | P1 | L | Eval metrics | Metrics: grade distribution, correction rate, hallucination trend |
| **Integration Example** | P1 | M | Demo | Customer support chatbot template; show production integration |
| **Performance Benchmarks** | P2 | S | Eval | Latency, throughput, cost curves for different model/strategy combinations |
| **Error Handling** | P2 | M | Pipeline | Graceful fallbacks for API failures, timeout handling |
| **RAG Knowledge Base** | P2 | L | Demo | Expand from toy KB to real-world docs (100+ pages) |

### v1.5 Success Criteria

- [ ] Hallucination rate <5% (target from PRD)
- [ ] Grader accuracy >90% precision (validated on eval set)
- [ ] Cost per query documented + optimized
- [ ] ≥2 production integration examples
- [ ] Dashboard shows real-time system health metrics

---

---

## v1.6 — Answer-Level Verification ✅ COMPLETE (2026-04-03)

**Target:** Add post-generation verification to catch hallucinations at the answer level.

### Completed Features

| Feature | Status | File | Description |
|---------|--------|------|-------------|
| **Answer Grounding Check** | ✅ | `app/grader.py` | Verify answer is supported by retrieved docs |
| **Gap Identification** | ✅ | `app/crag.py` | Track claims in answer that lack document support |
| **Supported Claims Counter** | ✅ | `app/crag.py` | Count how many answer claims are doc-backed |
| **Strategy Registry** | ✅ | `app/corrector.py` | Allow custom domain-specific correction strategies |

### v1.6 Metrics

- Answer grounding flag (True/False) in trace
- Gap list (claims without sources)
- Supported claims count
- Strategy plugin system ready for v2.0 fine-tuning

---

## v1.7 — BGE Reranking ✅ COMPLETE (2026-04-05)

**Target:** Optimize cost by filtering irrelevant docs before grading.

### Completed Features

| Feature | Status | File | Description |
|---------|--------|------|-------------|
| **BGE CrossEncoder Reranking** | ✅ | `app/reranker.py` (NEW) | Local semantic reranking (no API cost) |
| **Pipeline Integration** | ✅ | `app/crag.py` | Retrieve (10) → rerank (3) → grade (3) |
| **Cost Optimization** | ✅ | All | ~20-30% reduction in grader calls |
| **Reranking Trace** | ✅ | `app/demo.py` | Show reranking metrics in UI trace |

### v1.7 Metrics

- **Cost reduction:** ~70% fewer grader calls (10 → 3 per query)
- **Total cost delta:** ~20-30% reduction vs. v1.5
- **Quality:** Reranking before grading filters noise without losing answers

---

## v1.8 — Batch Evaluation & Calibration ✅ COMPLETE (2026-04-05)

**Target:** Enable empirical validation of confidence scores and efficient batch testing.

### Completed Features

| Feature | Status | File | Description |
|---------|--------|------|-------------|
| **Batch Evaluation Mode** | ✅ | `app/eval.py` | Load test cases from CSV with --batch flag |
| **Confidence Calibration** | ✅ | `app/eval.py` | Validate that confidence scores match actual accuracy |
| **Calibration Analysis** | ✅ | `app/eval.py` | Bin-wise breakdown + overall calibration error |
| **CSV Test Format** | ✅ | Docs | question, expected_facts, answerable, category |

### v1.8 Metrics

- Confidence calibration check (0–1 scale)
- Calibration error per bin (0.0–0.2 is well-calibrated)
- Per-question evaluation results in JSON
- Support for 100+ test cases efficiently

### Usage

```bash
python eval.py                    # Built-in 20 test cases
python eval.py --batch eval.csv   # Load from CSV (custom test set)
```

---

## v2.0 — Advanced Features 🚀 (Future)

**Target:** Advanced techniques for high-stakes domains (legal, medical, enterprise).

### Planned Features

| Feature | Priority | Effort | Why It Matters |
|---------|----------|--------|---|
| **Continuous Reranking** | P1 | M | Integrate Cohere/BGE reranker for better doc ordering before grading |
| **Multi-hop Retrieval** | P1 | L | Answer complex questions requiring 3+ documents |
| **Fine-tuned Grader** | P2 | XL | Train grader on domain-specific labeled data (e.g., legal docs) |
| **Answer-Level Fact-Checking** | P2 | L | Post-generation verification: does answer match retrieved docs? |
| **A/B Testing Framework** | P2 | M | Compare CRAG strategies in production (correction strategy effectiveness) |
| **Feedback Loop** | P3 | M | Users mark answers as helpful/wrong → improve grader over time |
| **Multi-modal Support** | P3 | XL | Handle images, tables, PDFs as retrieved sources |

### v2.0 Success Criteria

- [ ] Fine-tuned grader matches GPT-4o accuracy at 10x lower cost
- [ ] Multi-hop queries resolved correctly >80% of the time
- [ ] A/B testing shows 20%+ improvement on correction strategy effectiveness
- [ ] Legal/medical sector adoption (validated through case studies)

---

## Feature Status Timeline

```
v1.0 (Now)                v1.5 (2-4 weeks)         v2.0 (2-3 months)
├─ Core CRAG working      ├─ Cost optimized         ├─ Reranking
├─ Demo + Eval            ├─ Confidence scores      ├─ Multi-hop
├─ Audit traces           ├─ Dashboard              ├─ Fine-tuned grader
└─ Documentation          ├─ Integration examples   ├─ Fact-checking
                          └─ Performance benchmarks └─ Feedback loops
```

---

## Known Limitations & Risks

### v1.0 Blockers → v1.5

| Issue | Current State | Mitigation | Timeline |
|-------|---|---|---|
| **Grader accuracy on edge cases** | Some borderline relevance calls wrong | Collect human labels; fine-tune threshold | v1.5 |
| **Cost not optimized** | Using GPT-4o for grading (expensive) | Benchmark Haiku; A/B test | v1.5 |
| **Limited observability** | Manual eval only; no production logs | Build dashboard with streaming metrics | v1.5 |
| **Small knowledge base** | Only 5 sample docs; not representative | Add 50+ real-world docs for v1.5 demo | v1.5 |

### v1.5 Blockers → v2.0

| Issue | Workaround | v2.0 Solution |
|-------|---|---|
| **Multi-doc reasoning** | Single doc retrieval insufficient | Implement multi-hop with reasoning |
| **Domain-specific accuracy** | Generic grader may fail on legal/medical | Fine-tune on labeled examples |
| **Hallucination still possible** | Grader passes bad doc, generator makes up answer | Add post-generation fact-check |

---

## Dependencies & Sequencing

### For v1.5

1. **Cost Analysis** (blocks confidence score rollout)
   - Run Haiku vs GPT-4o grader comparison
   - Document cost/quality tradeoff

2. **Confidence Scores** (depends on cost analysis)
   - Add confidence signal to generator
   - Update demo to show confidence badges

3. **Dashboard** (depends on production data)
   - Set up metrics collection in production
   - Build Streamlit/monitoring dashboard

### For v2.0

1. **Reranker Integration** (independent)
   - Swap FAISS with reranker + FAISS pipeline

2. **Fine-tuned Grader** (depends on labeled data)
   - Collect 500+ labeled (query, doc, relevant/not) pairs
   - Fine-tune on Claude 3 Haiku

3. **Multi-hop** (depends on fine-tuned grader)
   - Needs accurate grader to evaluate multi-doc combinations

---

## Metrics to Track

### Now (v1.0)
- Hallucination rate (baseline RAG vs CRAG)
- Correction strategy success rate
- Cost per query

### v1.5
- Grader accuracy (precision, recall on eval set)
- Correction effectiveness by strategy
- Answer confidence correlation with user trust
- Production query volume & error rates

### v2.0
- Multi-hop success rate
- Fine-tuned grader cost vs accuracy
- A/B test winner (which correction strategy works best?)
- User feedback loop signal (helpful/unhelpful ratio)

---

## Definition of Done

### v1.0 ✅
- [x] Core system implemented and tested
- [x] Evaluation shows 50%+ hallucination reduction
- [x] Demo is interactive and shows trace details
- [x] GitHub repo public with README
- [x] PRD documents product decision-making

### v1.5 ✅
- [x] Cost analysis complete (Haiku vs GPT-4o)
- [x] Confidence scores in generator output
- [x] Production observability dashboard (Streamlit)
- [x] UI improvements (4 tabs, KB sidebar, sample questions)
- [x] v1.5 README with cost model and metrics

### v1.6 ✅
- [x] Answer-level verification (grounding check)
- [x] Gap identification (claims without sources)
- [x] Strategy plugin system

### v1.7 ✅
- [x] BGE reranking integration
- [x] Cost optimization (20-30% grader cost reduction)
- [x] Reranking metrics in UI

### v1.8 ✅
- [x] Batch evaluation mode (CSV input)
- [x] Confidence calibration analysis
- [x] Per-question evaluation JSON output

### v2.0 (Planning)
- [ ] Reranker integration working
- [ ] Multi-hop queries resolved
- [ ] Fine-tuned grader trained and evaluated
- [ ] A/B testing framework implemented
- [ ] Production case studies (legal/medical)

---

## Questions for PM Review

1. **Cost vs. Quality:** Should we optimize for cost (Haiku grader) or accuracy (GPT-4o)? What's the breakeven?
2. **Multi-hop Scope:** How complex should multi-hop queries be? 2 docs? 5? Unlimited?
3. **Fine-tuning Data:** Which domain should we focus on for fine-tuned grader? (Legal, medical, customer support?)
4. **Production Timeline:** What's the ship date target for v1.5? (affects prioritization)
5. **Success Bar:** Is <5% hallucination rate sufficient, or do we need even lower for v1.5?
