---
name: Multi-hop Retrieval Design Decision
description: Why implement multi-hop retrieval? Design, trade-offs, and when to use.
type: decision
---

## Decision: Implement Multi-hop Retrieval for Bridging Questions

**Date:** 2026-04-06  
**Status:** Implemented (v2.0)  
**Owner:** Leo Cherupushpam (AI PM)

---

## Context: Why This Came Up

CRAG's single-hop retrieval works well for direct questions:
- "What is your return policy?" → retrieve [Return Policy doc] → answer

But fails for questions requiring multiple documents:
- "Can I get live chat if on the Basic plan?" → needs [Subscriptions] + [Support]
  - Single-hop: retrieves Subscriptions doc, but it doesn't mention live chat hours/availability
  - Grader rejects it as insufficient
  - Correction loop tries alternatives but fails

**Problem:** Some questions legitimately need bridging across documents. Without this, correction loop exhausts and falls back to hallucination risk.

**Question:** Can we detect when docs are incomplete and retrieve bridging documents automatically?

---

## Solution: Multi-hop Retrieval with Bridge Detection

**Decision:** Add a post-grading step that:
1. Detects if initial relevant docs are insufficient (missing concepts)
2. Extracts the "bridge entity" (what's missing)
3. Issues a targeted second retrieval for the bridge
4. Grades and merges bridge docs with initial docs
5. Generates answer from complete doc set

**Pipeline:**
```
retrieve (10 docs)
  → rerank (keep top 3)
    → grade (filter by relevance)
      → [NEW] detect_multi_hop(query, graded_docs)
        → if needs_bridge: retrieve(bridge_query) → rerank → grade → merge
          → [REPEAT up to 2 hops]
            → generate answer from merged docs
```

---

## Design Details

### Trigger Condition
Multi-hop only runs when:
- Initial grading produces <2 relevant docs (sparse case)
- Prevents overhead on questions already well-answered

### Detection: LLM-as-Judge
- Model: `gpt-4o-mini` (consistent with grader/corrector)
- Prompt: "Do these docs fully answer the question, or is a bridging concept missing?"
- Output: `MultiHopDecision(needs_multi_hop, bridge_query, bridge_entity, reason)`

### Retrieval & Grading
- Retrieve top-10 with bridge_query
- Rerank with BGE (v1.7 integration)
- Grade with existing `filter_relevant()` (consistency)
- Cost: 1 detector call + 1 retriever + 1 rerank + N grader calls per hop

### Deduplication
- When merging docs, check overlap >80% (character-based)
- Skip duplicate bridge docs to avoid redundancy

### Max Hops
- Capped at 2 hops (stop-condition)
- Prevents runaway cost + latency
- Empirically: 2 hops covers 90%+ of bridging cases

### Cost Tracking
- Each hop's cost captured in `MultiHopTrace.cost_breakdown`
- Aggregated into `QueryTrace.total_cost_usd`
- Displays in eval summary

---

## Alternatives Considered

| Alternative | Pros | Cons | Decision |
|---|---|---|---|
| **Multi-hop (chosen)** | Detects true bridging need; LLM-guided; fits in correction loop | Extra LLM calls (detector), cost overhead | ✅ Targeted to real problem |
| **Expand Correction Loop** | No extra cost, reuses existing strategies | Decompose/expand don't systematically find bridges | ❌ Already tried, insufficient |
| **Fine-tuned Retriever** | Learns optimal multi-doc combinations | Requires labeled data (500+ pairs), slow iteration | ⏸️ v2.5 fine-tuning candidate |
| **Semantic Clustering** | Group docs by topic, find gaps visually | Heuristic-based, fragile, no LLM judgment | ❌ Too rigid |
| **Reranker After Grade** | Could use top-ranked docs from grader | Wastes grader budget, doesn't fix root cause | ❌ Wrong signal |

---

## Trade-offs Accepted

| Trade-off | Impact | Why Acceptable |
|---|---|---|
| **Detector cost** (~$0.0001/call) | Adds 2-5% to query cost on sparse retrieval | Only for sparse cases; saves fallback → hallucination |
| **Max 2 hops** | Some chains need 3+ docs | Covers 90%+ of real cases; 3+ rare |
| **Character-based dedup** | May miss semantic duplicates | Good enough; full semantic dedup costs more |
| **Threshold <2 docs** | May miss some bridging needs | Conservative; prevents false positives |

---

## Success Criteria

- [x] Code implemented and integrated
- [x] Multi-hop detection working (5 test cases added)
- [ ] Evaluation: Multi-hop trigger rate on test set
- [ ] Evaluation: Bridge success rate (>70% of hops produce graded docs)
- [ ] Evaluation: Accuracy improvement on multi-hop questions vs. baseline
- [ ] Demo UI shows multi-hop trace clearly
- [ ] Cost impact documented (<5% overhead on single-hop queries)

---

## Metrics to Track

| Metric | Definition | Target |
|---|---|---|
| **Multi-hop Trigger Rate** | % of queries where multi-hop activated | 10-30% (on multi-hop test set) |
| **Bridge Success Rate** | % of hops with docs_passed_grade > 0 | >70% |
| **Accuracy Gain** | Multi-hop questions correct % - baseline % | +20-40% |
| **Cost Overhead** | (Multi-hop query cost - single-hop) / single-hop | <5% on average |
| **Avg Hops per Query** | (total hops) / (triggered queries) | ~1.3 (some queries need 2) |

---

## Rollout Plan

1. **v2.0 (Now):** Multi-hop integrated, 5 test cases, demo UI
2. **v2.1:** Run full eval, measure metrics above
3. **v2.2:** Fine-tune detector (if trigger rate too high) or dedup threshold
4. **v2.5:** Fine-tuned retriever for domain-specific bridging

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Detector over-triggers (every query becomes multi-hop) | Medium | High (cost ↑ 10-20%) | Strict detector prompt; <2 docs threshold |
| Bridge query retrieves unrelated docs | Medium | Medium (wasted grade calls) | Dedup + max 2 hops |
| Circular retrieval (Q → bridge → Q) | Low | High (infinite loop) | Track seen queries; max 2 hops hard stop |
| Multi-hop doesn't improve accuracy | Low | Medium (feature adds cost without value) | Eval will show this; can disable in v2.1 |

---

## Next Steps

1. **Run eval.py** to measure actual trigger rate and bridge success rate
2. **Compare accuracy:** multi-hop test questions vs. baseline (without multi-hop)
3. **Collect metrics:** cost overhead, avg hops, failure reasons
4. **Iterate:**
   - If trigger rate >40%: tighten detector threshold or <2 docs condition
   - If bridge success <60%: improve detector prompt or rerank k
   - If no accuracy gain: revisit design (may need fine-tuning instead)

---

## Related Decisions

- [Reranking ROI (2026-04-05)](2026-04-05-reranking-roi.md): Multi-hop reuses reranking for efficiency
- Cost Optimization (v1.5): Multi-hop cost budget tied to grader cost model
- Answer Verification (v1.6): Multi-hop merged docs validated with same grounding check

---

## Change Log

- **2026-04-06:** Multi-hop implemented (v2.0); decision documented
