"""
==============================================================================
 DISTRIBUTED AI SYSTEMS USING PySpark RDDs AND INTELLIGENT AGENTS
==============================================================================
Author  : Generated for Academic / Project Submission
Engine  : Apache Spark (PySpark) - RDD API
Purpose : A single, runnable reference implementation covering FIVE
          distributed AI system designs:

    1. Hybrid AI Recommendation Engine (Content-Based + Collaborative)
    2. Smart Disaster Detection System (Multi-source data fusion)
    3. AI-Based Fraud Detection Framework (Distributed anomaly detection)
    4. Autonomous Supply Chain Optimization System (Decentralized agents)
    5. Personalized Learning Analytics Platform (Adaptive learning)

Each system is implemented as an independent module using PySpark's RDD
API (map, filter, reduceByKey, groupByKey, join, aggregate, etc.) to
demonstrate distributed, large-scale data processing. Every module also
implements a lightweight "IntelligentAgent" class that consumes the
distributed computation's output and makes adaptive, real-time decisions
(recommendations, alerts, trades, routing, or content adjustments).

The script generates synthetic datasets so that it runs end-to-end with
no external dependencies beyond `pyspark`, `matplotlib`, `numpy` and
`pandas`. At the end, it produces a single consolidated dashboard image:

    output_results.png

Run:
    python hybrid_ai_pyspark_systems.py
==============================================================================
"""

import random
import math
import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark import SparkConf, SparkContext

random.seed(42)
np.random.seed(42)

# ------------------------------------------------------------------------
# SPARK CONTEXT INITIALIZATION
# ------------------------------------------------------------------------
conf = SparkConf().setAppName("HybridDistributedAISystems").setMaster("local[*]")
sc = SparkContext.getOrCreate(conf=conf)
sc.setLogLevel("ERROR")

RESULTS = {}   # collects summary metrics from every module for the dashboard


def banner(title):
    print("\n" + "=" * 78)
    print(f" {title}")
    print("=" * 78)


# ==========================================================================
# INTELLIGENT AGENT BASE CLASS
# ==========================================================================
class IntelligentAgent:
    """
    A minimal reusable "intelligent agent" abstraction shared by all five
    systems. Each concrete agent:
      - observes distributed (Spark-computed) state,
      - scores/ranks/evaluates it against a policy,
      - adapts its policy weights using simple online (real-time) feedback,
      - emits an action (recommendation, alert, trade, route, hint).
    This mirrors how, in a production architecture, agents would sit at the
    edge of a Spark Streaming / structured-streaming pipeline and consume
    micro-batch RDDs to act in near real time.
    """

    def __init__(self, name, learning_rate=0.1):
        self.name = name
        self.learning_rate = learning_rate
        self.weight = 0.5  # adaptive confidence weight, tuned via feedback

    def adapt(self, feedback_signal):
        """Online update rule (simple gradient-style nudge)."""
        self.weight += self.learning_rate * (feedback_signal - self.weight)
        self.weight = min(max(self.weight, 0.0), 1.0)
        return self.weight


# ==========================================================================
# 1. HYBRID AI RECOMMENDATION ENGINE (Content-Based + Collaborative)
# ==========================================================================
def hybrid_recommendation_engine():
    banner("1. HYBRID AI RECOMMENDATION ENGINE  (PySpark RDDs)")

    n_users, n_items = 60, 25
    genres = ["Action", "Drama", "Comedy", "SciFi", "Romance"]

    # --- synthetic distributed datasets -----------------------------------
    # (user_id, item_id, rating)
    ratings_raw = [
        (u, i, random.randint(1, 5))
        for u in range(n_users)
        for i in random.sample(range(n_items), k=random.randint(3, 8))
    ]
    # (item_id, genre_vector) -> content profile
    item_profiles = {i: random.choice(genres) for i in range(n_items)}

    ratings_rdd = sc.parallelize(ratings_raw)                     # distributed ingestion
    items_rdd = sc.parallelize(list(item_profiles.items()))       # distributed item metadata

    # ---- COLLABORATIVE FILTERING (item-item co-rating via RDDs) ----------
    # Map to (user, (item, rating)) then group by user -> compute item co-occurrence
    user_items = ratings_rdd.map(lambda x: (x[0], (x[1], x[2]))).groupByKey() \
                             .mapValues(list)

    def item_pairs(items):
        pairs = []
        for i in range(len(items)):
            for j in range(len(items)):
                if i != j:
                    (item_a, r_a), (item_b, r_b) = items[i], items[j]
                    pairs.append(((item_a, item_b), r_a * r_b))
        return pairs

    item_similarity = user_items.flatMap(lambda kv: item_pairs(kv[1])) \
                                 .reduceByKey(lambda a, b: a + b) \
                                 .sortBy(lambda kv: -kv[1])

    top_cf_pairs = item_similarity.take(5)

    # ---- CONTENT-BASED FILTERING (genre match score via RDDs) ------------
    def content_score(item_genre, target_genre):
        return 1.0 if item_genre == target_genre else 0.2

    target_user_genre = "SciFi"
    content_scores = items_rdd.map(
        lambda kv: (kv[0], content_score(kv[1], target_user_genre))
    )

    # ---- HYBRID FUSION: weighted blend of CF strength + content score ----
    cf_strength = item_similarity.map(lambda kv: (kv[0][0], kv[1])) \
                                  .reduceByKey(lambda a, b: a + b)

    def normalize(rdd):
        vals = rdd.map(lambda kv: kv[1]).collect()
        mx = max(vals) if vals else 1
        return rdd.mapValues(lambda v: v / mx if mx else 0)

    cf_norm = normalize(cf_strength)
    hybrid = cf_norm.join(content_scores) \
                     .mapValues(lambda cv: 0.6 * cv[0] + 0.4 * cv[1]) \
                     .sortBy(lambda kv: -kv[1])

    top_recommendations = hybrid.take(5)

    # ---- INTELLIGENT AGENT: adapts blend weight from real-time feedback --
    rec_agent = IntelligentAgent("RecommenderAgent")
    simulated_click_feedback = [1, 0, 1, 1, 0, 1, 1, 0]  # user engagement stream
    for f in simulated_click_feedback:
        rec_agent.adapt(f)

    print(f"Top hybrid recommendations (item_id, score): {top_recommendations}")
    print(f"Agent adapted confidence weight after real-time feedback: {rec_agent.weight:.3f}")

    RESULTS["recommendation"] = {
        "items": [f"Item {i}" for i, _ in top_recommendations],
        "scores": [round(s, 3) for _, s in top_recommendations],
        "agent_weight": rec_agent.weight,
    }


# ==========================================================================
# 2. SMART DISASTER DETECTION SYSTEM (Multi-source fusion)
# ==========================================================================
def disaster_detection_system():
    banner("2. SMART DISASTER DETECTION SYSTEM  (PySpark RDDs)")

    zones = [f"Zone-{z}" for z in range(1, 11)]

    # simulated multi-source streams
    sensor_stream = [(z, round(random.uniform(0, 10), 2)) for z in zones for _ in range(5)]   # water level / seismic reading
    satellite_stream = [(z, round(random.uniform(0, 1), 2)) for z in zones for _ in range(3)]  # anomaly index from imagery
    social_media_stream = [(z, random.randint(0, 50)) for z in zones for _ in range(4)]        # disaster-keyword mentions

    sensor_rdd = sc.parallelize(sensor_stream)
    satellite_rdd = sc.parallelize(satellite_stream)
    social_rdd = sc.parallelize(social_media_stream)

    # ---- distributed aggregation per zone ---------------------------------
    avg_sensor = sensor_rdd.mapValues(lambda v: (v, 1)) \
                            .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
                            .mapValues(lambda v: v[0] / v[1])

    avg_satellite = satellite_rdd.mapValues(lambda v: (v, 1)) \
                                  .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
                                  .mapValues(lambda v: v[0] / v[1])

    total_social = social_rdd.reduceByKey(lambda a, b: a + b)

    # ---- DATA FUSION: weighted risk index combining all 3 sources --------
    fused = avg_sensor.join(avg_satellite).join(total_social) \
        .mapValues(lambda v: (
            0.5 * (v[0][0] / 10) +      # normalized sensor severity
            0.35 * v[0][1] +            # satellite anomaly index (0-1)
            0.15 * min(v[1] / 50, 1)    # normalized social signal
        )) \
        .sortBy(lambda kv: -kv[1])

    risk_index = fused.collect()

    # ---- INTELLIGENT AGENTS: coordinate early-warning decisions ----------
    class DisasterAgent(IntelligentAgent):
        def evaluate(self, risk_score):
            threshold = 0.5 + (0.1 * (1 - self.weight))  # adapts sensitivity
            if risk_score >= threshold:
                return "CRITICAL - Evacuate & Dispatch Response Teams"
            elif risk_score >= threshold * 0.6:
                return "WATCH - Increase Monitoring"
            return "NORMAL"

    coordinator = DisasterAgent("DisasterCoordinatorAgent")
    # feedback loop: false-alarm rate from last cycle nudges sensitivity
    for fb in [0.9, 0.8, 0.85]:
        coordinator.adapt(fb)

    alerts = [(zone, round(score, 3), coordinator.evaluate(score)) for zone, score in risk_index]

    print("Zone risk index (fused sensor + satellite + social signals):")
    for z, s, status in alerts[:6]:
        print(f"   {z:10s} risk={s:.3f}  -> {status}")

    RESULTS["disaster"] = {
        "zones": [z for z, _, _ in alerts],
        "risk": [s for _, s, _ in alerts],
        "critical_count": sum(1 for *_, st in alerts if st.startswith("CRITICAL")),
    }


# ==========================================================================
# 3. AI-BASED FRAUD DETECTION FRAMEWORK
# ==========================================================================
def fraud_detection_framework():
    banner("3. AI-BASED FRAUD DETECTION FRAMEWORK  (PySpark RDDs)")

    n_tx = 4000
    n_accounts = 200

    transactions = []
    for tx_id in range(n_tx):
        acc = random.randint(0, n_accounts - 1)
        amount = round(np.random.lognormal(mean=4, sigma=1.1), 2)
        hour = random.randint(0, 23)
        # inject a small fraction of anomalous high-value / odd-hour transactions
        is_seed_fraud = random.random() < 0.03
        if is_seed_fraud:
            amount *= random.uniform(8, 20)
            hour = random.choice([1, 2, 3, 4])
        transactions.append((tx_id, acc, amount, hour))

    tx_rdd = sc.parallelize(transactions)

    # ---- distributed per-account statistics (mean & std via RDDs) --------
    acc_amounts = tx_rdd.map(lambda t: (t[1], t[2]))
    acc_stats = acc_amounts.aggregateByKey(
        (0.0, 0.0, 0),  # (sum, sum_sq, count)
        lambda agg, v: (agg[0] + v, agg[1] + v * v, agg[2] + 1),
        lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2])
    ).mapValues(lambda v: (
        v[0] / v[2],
        math.sqrt(max(v[1] / v[2] - (v[0] / v[2]) ** 2, 0))
    )).collectAsMap()

    acc_stats_bc = sc.broadcast(acc_stats)

    # ---- DISTRIBUTED ANOMALY DETECTION: z-score + odd-hour rule ----------
    def score_tx(t):
        tx_id, acc, amount, hour = t
        mean, std = acc_stats_bc.value[acc]
        z = (amount - mean) / std if std > 0 else 0
        odd_hour_flag = 1 if hour in (1, 2, 3, 4) else 0
        anomaly_score = 0.75 * min(abs(z) / 5, 1) + 0.25 * odd_hour_flag
        return (tx_id, acc, amount, hour, round(anomaly_score, 3))

    scored_rdd = tx_rdd.map(score_tx)
    suspicious_rdd = scored_rdd.filter(lambda t: t[4] >= 0.5).sortBy(lambda t: -t[4])
    suspicious = suspicious_rdd.take(10)
    total_flagged = suspicious_rdd.count()

    # ---- INTELLIGENT AGENTS: collaborative alerting + accuracy feedback --
    class FraudAgent(IntelligentAgent):
        def decide(self, score):
            cutoff = 0.5 + 0.15 * (1 - self.weight)
            return "ALERT" if score >= cutoff else "MONITOR"

    detector_agent = FraudAgent("FraudDetectorAgent")
    reviewer_agent = FraudAgent("HumanReviewFeedbackAgent")
    # simulated analyst feedback: 1 = confirmed fraud, 0 = false positive
    analyst_feedback = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    for fb in analyst_feedback:
        detector_agent.adapt(fb)
        reviewer_agent.adapt(fb)

    precision_estimate = sum(analyst_feedback) / len(analyst_feedback)

    print(f"Transactions processed: {n_tx} | Flagged suspicious: {total_flagged}")
    print(f"Detector agent adapted confidence weight: {detector_agent.weight:.3f}")
    print(f"Estimated precision from analyst feedback loop: {precision_estimate:.2%}")
    print("Top suspicious transactions (tx_id, acc, amount, hour, score):")
    for t in suspicious[:5]:
        print("  ", t)

    RESULTS["fraud"] = {
        "scores": [t[4] for t in scored_rdd.takeSample(False, 300, seed=1)],
        "flagged": total_flagged,
        "total": n_tx,
        "precision": precision_estimate,
    }


# ==========================================================================
# 4. AUTONOMOUS SUPPLY CHAIN OPTIMIZATION SYSTEM
# ==========================================================================
def supply_chain_optimization():
    banner("4. AUTONOMOUS SUPPLY CHAIN OPTIMIZATION SYSTEM  (PySpark RDDs)")

    warehouses = [f"WH-{i}" for i in range(1, 6)]
    products = [f"P-{i}" for i in range(1, 16)]

    inventory = [(w, p, random.randint(0, 500)) for w in warehouses for p in products]
    demand_forecast = [(w, p, random.randint(50, 400)) for w in warehouses for p in products]
    logistics_cost = {w: round(random.uniform(2.0, 9.0), 2) for w in warehouses}  # cost per unit-km

    inv_rdd = sc.parallelize(inventory).map(lambda x: ((x[0], x[1]), x[2]))
    dem_rdd = sc.parallelize(demand_forecast).map(lambda x: ((x[0], x[1]), x[2]))

    # ---- distributed join: inventory vs forecasted demand -----------------
    gap_rdd = inv_rdd.join(dem_rdd).mapValues(lambda v: v[1] - v[0])  # positive => shortage
    shortages = gap_rdd.filter(lambda kv: kv[1] > 0)
    surplus = gap_rdd.filter(lambda kv: kv[1] < 0)

    # ---- RDD aggregation: total shortage per product across warehouses ---
    shortage_by_product = shortages.map(lambda kv: (kv[0][1], kv[1])) \
                                    .reduceByKey(lambda a, b: a + b) \
                                    .sortBy(lambda kv: -kv[1])

    surplus_by_warehouse = surplus.map(lambda kv: (kv[0][0], -kv[1])) \
                                   .reduceByKey(lambda a, b: a + b)

    # ---- decentralized agent decision: transfer surplus -> shortage ------
    class WarehouseAgent(IntelligentAgent):
        def propose_transfer(self, product, units_needed, cost_per_unit):
            # agent balances cost efficiency vs urgency (adaptive weight)
            urgency = min(units_needed / 300, 1)
            confidence = self.weight * (1 - cost_per_unit / 10) + (1 - self.weight) * urgency
            return confidence

    agents = {w: WarehouseAgent(f"Agent-{w}") for w in warehouses}
    # feedback: on-time delivery rate per warehouse from last cycle
    delivery_feedback = {w: random.uniform(0.6, 0.95) for w in warehouses}
    for w, agent in agents.items():
        agent.adapt(delivery_feedback[w])

    decisions = []
    top_shortages = shortage_by_product.take(5)
    for product, need in top_shortages:
        best_source = max(surplus_by_warehouse.collect(), key=lambda kv: kv[1], default=None)
        if best_source:
            wh, avail = best_source
            conf = agents[wh].propose_transfer(product, need, logistics_cost[wh])
            decisions.append((product, wh, min(need, avail), round(conf, 3)))

    total_shortage_units = shortages.map(lambda kv: kv[1]).sum()
    total_surplus_units = -surplus.map(lambda kv: kv[1]).sum()

    print(f"Total shortage units across network: {total_shortage_units}")
    print(f"Total surplus units across network:  {total_surplus_units}")
    print("Top decentralized transfer decisions (product, source_wh, units, agent_confidence):")
    for d in decisions:
        print("  ", d)

    RESULTS["supply_chain"] = {
        "products": [p for p, _ in top_shortages],
        "shortage_units": [u for _, u in top_shortages],
        "total_shortage": total_shortage_units,
        "total_surplus": total_surplus_units,
    }


# ==========================================================================
# 5. PERSONALIZED LEARNING ANALYTICS PLATFORM
# ==========================================================================
def learning_analytics_platform():
    banner("5. PERSONALIZED LEARNING ANALYTICS PLATFORM  (PySpark RDDs)")

    n_students = 500
    topics = ["Algebra", "Geometry", "Probability", "Calculus", "Statistics"]

    interactions = []
    for sid in range(n_students):
        for _ in range(random.randint(5, 15)):
            topic = random.choice(topics)
            score = round(np.clip(np.random.normal(65, 20), 0, 100), 1)
            time_spent = round(random.uniform(2, 40), 1)  # minutes
            interactions.append((sid, topic, score, time_spent))

    inter_rdd = sc.parallelize(interactions)

    # ---- distributed per-student performance aggregation ------------------
    student_perf = inter_rdd.map(lambda x: (x[0], (x[2], 1))) \
                             .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
                             .mapValues(lambda v: v[0] / v[1])

    # ---- distributed per-topic difficulty (avg score, avg time) ----------
    topic_stats = inter_rdd.map(lambda x: (x[1], (x[2], x[3], 1))) \
                            .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2])) \
                            .mapValues(lambda v: (round(v[0] / v[2], 1), round(v[1] / v[2], 1)))

    topic_stats_collected = dict(topic_stats.collect())

    # ---- adaptive content-delivery agent per student ----------------------
    class TutorAgent(IntelligentAgent):
        def recommend_next(self, avg_score):
            difficulty_bias = self.weight  # higher weight -> push harder content sooner
            if avg_score < 50:
                return "Remedial content + guided practice"
            elif avg_score < 75:
                return "Standard-pace content" if difficulty_bias < 0.6 else "Slightly advanced content"
            else:
                return "Advanced / enrichment content"

    tutor = TutorAgent("AdaptiveTutorAgent")
    engagement_feedback = [1, 1, 0, 1, 1, 1, 0, 1]  # completed lessons vs dropped
    for fb in engagement_feedback:
        tutor.adapt(fb)

    sample_students = student_perf.take(6)
    personalized_plan = [(sid, round(avg, 1), tutor.recommend_next(avg)) for sid, avg in sample_students]

    at_risk = student_perf.filter(lambda kv: kv[1] < 50).count()
    class_avg = student_perf.map(lambda kv: kv[1]).mean()

    print(f"Class-wide average performance: {class_avg:.2f}")
    print(f"At-risk students (avg score < 50): {at_risk} / {n_students}")
    print("Topic difficulty (avg_score, avg_time_min):", topic_stats_collected)
    print("Sample personalized learning plan (student_id, avg_score, recommendation):")
    for p in personalized_plan:
        print("  ", p)

    RESULTS["learning"] = {
        "topics": list(topic_stats_collected.keys()),
        "avg_scores": [v[0] for v in topic_stats_collected.values()],
        "class_avg": class_avg,
        "at_risk": at_risk,
        "n_students": n_students,
    }


# ==========================================================================
# DASHBOARD: consolidated output visualization (output_results.png)
# ==========================================================================
def build_dashboard(out_path="output_results.png"):
    banner("BUILDING CONSOLIDATED OUTPUT DASHBOARD")

    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    fig.suptitle("Distributed AI Systems on PySpark RDDs — Consolidated Output Dashboard",
                 fontsize=16, fontweight="bold")

    # 1. Recommendation engine
    ax = axes[0, 0]
    r = RESULTS["recommendation"]
    ax.barh(r["items"], r["scores"], color="#4C72B0")
    ax.set_title("1. Hybrid Recommendation Scores")
    ax.set_xlabel("Hybrid Score")
    ax.invert_yaxis()

    # 2. Disaster detection
    ax = axes[0, 1]
    d = RESULTS["disaster"]
    colors = ["#C44E52" if s >= 0.5 else "#55A868" for s in d["risk"]]
    ax.bar(d["zones"], d["risk"], color=colors)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Alert threshold")
    ax.set_title(f"2. Disaster Risk Index by Zone ({d['critical_count']} critical)")
    ax.set_ylabel("Fused Risk Score")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)

    # 3. Fraud detection
    ax = axes[0, 2]
    f = RESULTS["fraud"]
    ax.hist(f["scores"], bins=20, color="#8172B2", edgecolor="white")
    ax.axvline(0.5, color="red", linestyle="--", label="Alert cutoff")
    ax.set_title(f"3. Fraud Anomaly Score Distribution\n({f['flagged']}/{f['total']} flagged, "
                 f"~{f['precision']:.0%} precision)")
    ax.set_xlabel("Anomaly Score")
    ax.legend(fontsize=8)

    # 4. Supply chain
    ax = axes[1, 0]
    s = RESULTS["supply_chain"]
    ax.bar(s["products"], s["shortage_units"], color="#CCB974")
    ax.set_title(f"4. Top Product Shortages\n(Network shortage={s['total_shortage']}, "
                 f"surplus={s['total_surplus']})")
    ax.set_ylabel("Shortage Units")
    ax.tick_params(axis="x", rotation=30)

    # 5. Learning analytics
    ax = axes[1, 1]
    l = RESULTS["learning"]
    ax.bar(l["topics"], l["avg_scores"], color="#64B5CD")
    ax.axhline(l["class_avg"], color="black", linestyle="--", label=f"Class avg={l['class_avg']:.1f}")
    ax.set_title(f"5. Topic Avg Scores\n({l['at_risk']}/{l['n_students']} students at risk)")
    ax.set_ylabel("Avg Score")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)

    # 6. Agent adaptive-weight summary panel
    ax = axes[1, 2]
    ax.axis("off")
    summary_text = (
        "INTELLIGENT AGENT ADAPTIVE WEIGHTS\n"
        "(after real-time feedback loops)\n\n"
        f"Recommender Agent      : {RESULTS['recommendation']['agent_weight']:.3f}\n"
        f"Disaster Coordinator    : adaptive threshold agent\n"
        f"Fraud Detector Agent    : precision-tuned\n"
        f"Supply-Chain Agents     : decentralized, per-warehouse\n"
        f"Adaptive Tutor Agent    : engagement-tuned\n\n"
        "All five systems share the same distributed-computing\n"
        "pattern: Spark RDD transformations (map / filter / \n"
        "reduceByKey / join / aggregateByKey) perform large-scale\n"
        "batch/stream processing, while lightweight intelligent\n"
        "agents consume the reduced state to make fast, adaptive,\n"
        "real-time decisions."
    )
    ax.text(0.02, 0.98, summary_text, va="top", ha="left", fontsize=10.5, family="monospace")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150)
    print(f"Dashboard saved to: {out_path}")


# ==========================================================================
# MAIN DRIVER
# ==========================================================================
if __name__ == "__main__":
    hybrid_recommendation_engine()
    disaster_detection_system()
    fraud_detection_framework()
    supply_chain_optimization()
    learning_analytics_platform()
    build_dashboard("output_results.png")

    with open("results_summary.json", "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print("\nAll five distributed AI systems executed successfully.")
    print("Summary metrics written to results_summary.json")

    sc.stop()
