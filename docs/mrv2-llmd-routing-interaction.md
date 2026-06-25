# When the Engine Gets Faster, the Router Gets Smarter: MRV2 Through the Lens of llm-d Routing

**llm-d Engineering Blog** · *June 2026*

---

The vLLM project recently shipped [Model Runner V2](https://vllm.ai/blog/2026-03-24-mrv2) — a ground-up rewrite of its execution core that eliminates host CPU bottlenecks by moving input preparation to GPU-native Triton kernels and enabling zero-sync async pipelining. The headline number: **56% higher throughput** on a small model with fast hardware.

But vLLM pods don't serve traffic in isolation. In a production llm-d deployment, an **Endpoint Picker (EPP)** sits in front of those pods and makes per-request routing decisions based on real-time signals: KV cache pressure, queue depth, prefix cache locality. The router's entire scoring model was tuned against V1's performance characteristics.

What happens when you drop a fundamentally faster execution engine underneath a routing layer that was calibrated for a slower one?

The answer is not "everything gets 56% faster." It's more interesting than that.

---

## The Feedback Loop You Didn't Know You Had

The llm-d optimized-baseline deploys a weighted scoring pipeline to pick the best pod for each request:

```yaml
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: queue-scorer           # weight: 2
  - pluginRef: kv-cache-utilization-scorer  # weight: 2
  - pluginRef: prefix-cache-scorer    # weight: 3
  - pluginRef: no-hit-lru-scorer      # weight: 2
```

For an endpoint $e$, the final routing score is:

$$S(e) = 2 \cdot Q(e) + 2 \cdot K(e) + 3 \cdot P(e) + 2 \cdot L(e)$$

where $Q$ is inverse queue depth, $K$ is inverse KV cache utilization, $P$ is prefix cache match ratio, and $L$ is the no-hit LRU score. Each scorer returns a value in $[0.0, 1.0]$.

Every one of these signals is *downstream* of how fast the model runner executes. The model runner's speed determines how quickly requests drain from the queue, how rapidly KV cache blocks are freed after completion, and how long prefix cache entries remain resident before eviction pressure forces them out. In other words:

> **The model runner's execution speed is not a constant the router observes — it's a variable that reshapes the entire scoring landscape.**

---

## Signal by Signal: How MRV2 Changes the Game

### 1. Queue Depth ($Q$): Faster Drain, Flatter Scores

The `queue-scorer` scores each endpoint inversely to its `vllm:waiting_requests_total` metric. Under V1, a pod processing a batch at moderate speed accumulates a visible queue at moderate load — the scorer has meaningful signal to differentiate pods.

MRV2's zero-sync pipelining changes the dynamics:

```
V1 Step Cadence                    MRV2 Step Cadence
─────────────────                  ─────────────────
CPU: ████████░░░░░░░               CPU: ████████░░░░░░░
GPU: ░░░░░░░░████████              GPU: ░░░████████████████
     ▲               ▲                  ▲                  ▲
     t₀           t₀+20ms               t₀              t₀+12ms
     (step latency ~20ms)                (step latency ~12ms)
```

With MRV2, the GPU starts step $N+1$ immediately after step $N$ — no idle gap waiting for CPU scheduling. Requests drain faster, queues stay shorter. At moderate load, most pods may report `waiting_requests = 0` simultaneously.

**The routing implication:** The queue scorer becomes less discriminating at moderate concurrency. Pods that V1 would have differentiated by queue depth of 3 vs. 0 now both report 0. The scorer's effective weight *decreases* — ceding influence to the other scorers. This is not a problem; it means the system has headroom. But it means the *prefix cache scorer* (weight 3) becomes even more dominant in the final score at operating points where V1 would have leaned on queue differentiation.

**At high load**, MRV2's faster drain rate means the system reaches higher QPS before queues begin to build. The queue scorer's discriminating power kicks in at a higher throughput tier — exactly where you need it most.

### 2. KV Cache Utilization ($K$): Faster Turnover, Lower Steady-State

The `kv-cache-utilization-scorer` polls `vllm:kv_cache_usage_perc` from each pod. Under V1, the steady-state KV cache utilization at a given QPS reflects the balance between new block allocations (from incoming requests) and block deallocations (from completed requests).

MRV2 doesn't change how blocks are allocated — that's scheduler work. But it accelerates how quickly requests *complete*, which means blocks are freed sooner. The effect:

| QPS | V1 Avg KV Usage | MRV2 Avg KV Usage | Delta |
|-----|-----------------|-------------------|-------|
| Low | ~30% | ~20% | KV scorer has less signal; all pods look "fine" |
| Medium | ~65% | ~50% | Scorer differentiates, but threshold for concern is pushed higher |
| High | ~90% | ~75% | **This is the sweet spot** — MRV2 keeps the fleet in the scorer's useful range instead of saturating it |

The critical insight: **MRV2 extends the load range over which the KV cache scorer provides useful gradient.** Under V1, at high QPS, most pods are above 85% and the scorer can only distinguish between "bad" and "worse." Under MRV2, the same QPS produces a spread of 60-80%, giving the scorer room to steer traffic away from pods that happen to be serving longer-sequence requests.

### 3. Prefix Cache Match ($P$): Longer Residency, Better Hits

This is where MRV2 and the router create a genuine **positive feedback loop**.

The `prefix-cache-scorer` (weight 3, the highest in the default profile) estimates how much of a request's prompt prefix is already cached on each pod. The [approximate implementation](https://github.com/llm-d/llm-d-router/tree/main/pkg/epp/framework/plugins/requestcontrol/dataproducer/approximateprefix) maintains an in-memory LRU index in the EPP. But the *actual* cache on the vLLM pod is subject to eviction — when KV cache memory pressure is high, vLLM evicts cached prefix blocks to make room for active requests.

Under V1 at high load, prefix blocks are evicted aggressively because KV cache utilization is high. The EPP's LRU index says "pod 3 has your prefix" but pod 3 evicted it 2 seconds ago under memory pressure. The router makes a stale decision; the request pays full prefill cost anyway. The approximate scorer's precision *degrades under load* — exactly when you need it most.

MRV2 changes this dynamic by lowering steady-state KV cache utilization at the same QPS:

```
V1 at 40 QPS:
  KV usage ~90% --> aggressive eviction --> prefix residency ~ 2s
  Router prefix score accuracy: LOW (stale index)

MRV2 at 40 QPS:
  KV usage ~75% --> mild eviction --> prefix residency ~ 8s
  Router prefix score accuracy: HIGH (index reflects reality)
```

**The compounding effect:** When the prefix scorer is accurate, requests land on pods that *actually* have the prefix cached. Those pods skip prefill, finish faster, free KV blocks sooner, which keeps utilization low, which keeps prefixes resident longer, which makes the next routing decision more accurate. It's a virtuous cycle that V1's higher steady-state KV pressure interrupted.

### 4. No-Hit LRU ($L$): Cold Requests Get Better Pods

The `no-hit-lru-scorer` activates only for requests with zero prefix cache hits — "cold" requests that need full prefill. It steers these to the least-recently-used pod to spread the compute-heavy prefill workload evenly.

Under V1, cold requests are expensive and slow, occupying a pod's GPU for the full prefill duration. Under MRV2, prefill duration is the same (MRV2 optimizes per-step overhead, not attention computation itself), but the *decode phase* that follows is faster. This means a pod that just handled a cold prefill recovers to "available" status sooner, reducing the chance that the no-hit scorer sends the *next* cold request to an already-burdened pod.

---

## The Latency Predictor: Where MRV2 Has the Deepest Impact

The optimized-baseline uses heuristic scorers. But llm-d also ships a [Latency Predictor](https://github.com/llm-d/llm-d-latency-predictor) — an XGBoost model trained on live traffic that predicts per-pod TTFT and TPOT, letting the router make SLO-aware placement decisions.

The predictor's TPOT model uses these features:

| Feature | What It Captures |
|---------|------------------|
| KV Cache Usage % | Memory pressure during decode |
| Input Length | Attention cost |
| Queue Depth | Queue contention leaking into decode batching |
| Running Requests | Active decode batch size |
| Tokens Generated | Output tokens produced so far |

MRV2 fundamentally changes the *learned mapping* from these features to TPOT. Under V1, the relationship between `running_requests` and TPOT includes a CPU overhead component that scales with batch size (more requests = more Python-side metadata assembly). Under MRV2, that CPU component is near-zero — the relationship between batch size and TPOT is *purer*, driven almost entirely by GPU compute and memory bandwidth.

**Consequence for the predictor:**

1. **Lower TPOT variance.** V1's CPU jitter (Python GC pauses, NUMA effects on metadata assembly) added noise to TPOT measurements. MRV2's GPU-native path produces more consistent step times. The XGBoost model achieves lower prediction error because the underlying signal is less noisy.

2. **The model retrains faster.** The predictor uses stratified bucketing across KV utilization and prefix hit rate. With MRV2, the feature space is more compact (lower KV utilization at the same QPS means fewer active high-utilization buckets), so the model converges with fewer samples.

3. **SLO headroom increases.** If the predictor estimates TPOT = 45ms for a pod under V1, MRV2 might produce TPOT = 38ms for the same pod state. The `latency-scorer` using `headroomSelectionStrategy: least` now has more pods in the "meets SLO" tier, enabling tighter bin-packing before violating latency budgets.

---

## The Stacking Effect: Quantifying the Compound Gain

The existing llm-d optimized-baseline benchmarks show these results on 16x H100 (8 pods, TP=2) running Qwen3-32B with V1:

| Metric | k8s Service (RR) | llm-d Optimized | Improvement |
|---|---|---|---|
| Output tok/s | 5,722 | 13,163 | **+130%** |
| TTFT mean | 58.10s | 0.156s | **-99.7%** |
| TTFT p90 | 107.43s | 0.206s | **-99.8%** |

The 130% throughput gain comes entirely from *routing* — sending requests to the right pod instead of round-robin. Now consider what MRV2 adds on top:

```
Gain Source                  Mechanism                         Where It Hits
──────────────────────────── ──────────────────────────────── ────────────────
MRV2 raw execution speed     Zero-sync pipelining,            TPOT, throughput
                              GPU-native input prep

MRV2 x Queue scorer          Queues drain faster -->           Higher QPS before
                              scorer differentiates later        queue saturation

MRV2 x KV cache scorer       Lower steady-state KV usage -->  Better load spreading
                              scorer has gradient longer         at high utilization

MRV2 x Prefix cache scorer   Lower eviction pressure -->      Higher cache hit rate -->
                              longer prefix residency            less redundant prefill

MRV2 x Latency predictor     Cleaner TPOT signal -->          Tighter SLO bin-packing,
                              lower prediction error             fewer violations
```

These are not additive gains. They are **multiplicative**. Faster execution makes routing decisions more accurate, and more accurate routing decisions let the faster engine run at higher utilization without degradation. The system reaches a new equilibrium that neither component could achieve alone.

### A Conservative Estimate

Using the published benchmarks as anchors:

- llm-d routing over V1: **+130% throughput** vs. round-robin (measured, 16x H100)
- MRV2 over V1: **+56% throughput** at the execution layer (measured, 1x GB200, small model)
- MRV2 TPOT improvement on large models with speculative decoding: **-6.3%** (measured, 4x GB200)

For a production workload (Qwen3-32B on 16x H100), the MRV2 execution gain will be more modest than 56% — the model is larger, so CPU overhead is a smaller fraction of step time. A realistic estimate for MRV2's raw throughput gain on a 32B model is **8-15%**. But the compounding effects through the router — better prefix cache accuracy, extended KV scorer range, flatter queue profiles — could push the total improvement to **15-25%** on top of the already-optimized baseline.

---

## Retuning the Weights

If you enable MRV2 and observe that your P99 TPOT improved but your cache hit rate didn't change, your scorer weights may need adjustment. The default weights were tuned against V1's performance characteristics:

```yaml
# Default weights (tuned for V1)
- pluginRef: queue-scorer              # weight: 2
- pluginRef: kv-cache-utilization-scorer  # weight: 2
- pluginRef: prefix-cache-scorer       # weight: 3
- pluginRef: no-hit-lru-scorer         # weight: 2
```

Under MRV2, the queue scorer is less discriminating at moderate load. You may want to shift weight toward the prefix cache scorer or add the latency scorer:

```yaml
# Suggested starting point for MRV2 (validate with your workload)
- pluginRef: queue-scorer              # weight: 1
- pluginRef: kv-cache-utilization-scorer  # weight: 2
- pluginRef: prefix-cache-scorer       # weight: 4
- pluginRef: no-hit-lru-scorer         # weight: 2
```

Or, if you're running the [latency predictor](https://github.com/llm-d/llm-d-latency-predictor), let it learn the new MRV2 dynamics directly — it will retrain on the lower TPOT values automatically. The predictor's stratified bucketing means it adapts to the shifted feature distributions within a few hundred completed requests.

---

## How to Test This Yourself

### Step 1: Deploy the optimized-baseline with V1 (current default)

Follow the [optimized-baseline guide](https://github.com/llm-d/llm-d/tree/main/guides/optimized-baseline) to deploy the full stack. Run the benchmark profile to establish your V1 baseline:

```bash
llmdbenchmark \
    --spec guides/optimized-baseline \
    run \
    --endpoint-url "${ENDPOINT_URL}" \
    --gateway-class "${GATEWAY_CLASS}" \
    --model "Qwen/Qwen3-32B" \
    --namespace "${NAMESPACE}" \
    --harness inference-perf \
    --workload guide_optimized-baseline_1.yaml \
    --analyze
```

### Step 2: Enable MRV2 on the model server pods

Add the environment variable to your vLLM deployment patch:

```yaml
env:
  - name: VLLM_USE_V2_MODEL_RUNNER
    value: "1"
```

Redeploy and wait for pods to pass readiness probes:

```bash
kubectl apply -n ${NAMESPACE} -k \
  ${REPO_ROOT}/guides/optimized-baseline/modelserver/gpu/vllm/base/

kubectl rollout status deployment/optimized-baseline-nvidia-gpu-vllm-decode \
  -n ${NAMESPACE} --timeout=600s
```

Verify MRV2 is active:

```bash
kubectl logs -l llm-d.ai/role=decode -n ${NAMESPACE} | grep -i "MRV2\|model_runner_v2"
```

### Step 3: Re-run the same benchmark

```bash
llmdbenchmark \
    --spec guides/optimized-baseline \
    run \
    --endpoint-url "${ENDPOINT_URL}" \
    --gateway-class "${GATEWAY_CLASS}" \
    --model "Qwen/Qwen3-32B" \
    --namespace "${NAMESPACE}" \
    --harness inference-perf \
    --workload guide_optimized-baseline_1.yaml \
    --analyze
```

### Step 4: Compare the numbers

Key metrics to watch across the load ladder:

| Metric | What MRV2 Should Improve | Where to Look |
|---|---|---|
| Output tok/s | Higher at every QPS tier | `llmdbenchmark` report |
| TPOT mean/p90 | Lower, especially at high load | `vllm:time_per_output_token_seconds` |
| KV cache utilization | Lower at same QPS | `vllm:kv_cache_usage_perc` |
| Queue depth | Lower at same QPS | `vllm:num_requests_waiting` |
| Prefix cache hit rate | Higher at high load | `llm_d_epp_prefix_indexer_hit_ratio` |

The most telling comparison is the QPS at which TTFT begins to degrade. Under V1, the optimized-baseline benchmarks show TTFT staying flat until ~15 QPS and then spiking at ~20 QPS for round-robin (the router pushes this to 40+ QPS). With MRV2, that inflection point should shift even higher — the system absorbs more load before the routing signals indicate saturation.

---

## Prometheus Queries for the A/B Comparison

Track the interaction effects with these queries:

```promql
# P99 TPOT — the primary MRV2 signal
histogram_quantile(0.99,
  rate(vllm:time_per_output_token_seconds_bucket[5m])
)

# KV cache utilization spread across pods — should be lower and tighter with MRV2
stddev(vllm:kv_cache_usage_perc)

# Prefix cache hit ratio — should improve at high load with MRV2
histogram_quantile(0.50,
  rate(llm_d_epp_prefix_indexer_hit_ratio_bucket[5m])
)

# Queue depth variance — should flatten with MRV2
stddev(llm_d_epp_per_endpoint_queue_size)

# Scheduling latency — should remain unchanged (router overhead is independent)
histogram_quantile(0.99,
  rate(llm_d_epp_scheduler_e2e_duration_seconds_bucket[5m])
)
```

---

## Conclusion

MRV2 is not just a model runner upgrade. When deployed behind llm-d's scoring pipeline, it reshapes the information landscape that every routing decision is made against. Queues drain faster, KV caches run cooler, prefix entries live longer, and latency predictions become more precise.

The optimized-baseline's 130% throughput gain over round-robin was achieved against V1's execution characteristics. MRV2 raises the floor that the router is optimizing above — and makes the router's own decisions more effective in the process. The two improvements don't just stack; they compound.

The existing scorer weights were tuned for V1. If you're enabling MRV2, run the benchmark, check whether the queue scorer still differentiates at your operating point, and consider shifting weight toward prefix affinity or adopting the latency predictor to let the system learn the new dynamics directly.

---

*This analysis is based on the [vLLM MRV2 design document](https://docs.vllm.ai/en/v0.22.1/design/model_runner_v2/), the [llm-d router architecture](https://github.com/llm-d/llm-d-router), and the [optimized-baseline benchmark results](https://github.com/llm-d/llm-d/tree/main/guides/optimized-baseline). The compound gain estimates are analytical projections — run the benchmark on your own cluster to measure the actual interaction effect.*