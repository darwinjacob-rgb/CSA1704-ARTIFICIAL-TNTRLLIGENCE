# Distributed AI Systems using PySpark RDDs and Intelligent Agents

This project implements and documents **five distributed AI system designs**, all built on
Apache Spark's **RDD (Resilient Distributed Dataset)** API, each paired with a lightweight
**intelligent agent** layer for real-time adaptive decision-making.

| # | System | Domain Focus |
|---|--------|---------------|
| 1 | Hybrid AI Recommendation Engine | Content-based + Collaborative filtering fusion |
| 2 | Smart Disaster Detection System | Multi-source (sensor / satellite / social) data fusion |
| 3 | AI-Based Fraud Detection Framework | Distributed anomaly detection on transactions |
| 4 | Autonomous Supply Chain Optimization System | Decentralized, multi-agent inventory/logistics decisions |
| 5 | Personalized Learning Analytics Platform | Adaptive, scalable learner analytics |

## 📁 Files in this Deliverable

| File | Description |
|------|--------------|
| `Problem_Statement.pdf` | Formal problem statements, background, objectives, and constraints for all 5 systems. |
| `Solution_Design.pdf` | Architecture, RDD transformation strategy, and intelligent-agent design for all 5 systems. |
| `hybrid_ai_pyspark_systems.py` | **Complete, runnable** PySpark RDD implementation of all 5 systems (single file). |
| `output_results.png` | Consolidated dashboard image visualizing the output of every system. |
| `Report.pdf` | Full project report: abstract, methodology, results, discussion, conclusion. |
| `README.md` | This file. |
| `results_summary.json` | Raw JSON metrics produced by the script (used to build the report/dashboard). |

## ⚙️ Requirements

- Python 3.9+
- Java (JDK 11 or newer) — required by Spark
- Packages:
  ```bash
  pip install pyspark matplotlib numpy pandas --break-system-packages
  ```

## ▶️ How to Run

```bash
python hybrid_ai_pyspark_systems.py
```

This will:
1. Start a local Spark session (`local[*]`, all available cores).
2. Generate synthetic datasets for each of the 5 systems.
3. Run distributed RDD pipelines (`map`, `filter`, `groupByKey`, `reduceByKey`,
   `aggregateByKey`, `join`, `sortBy`, `broadcast`) for each domain.
4. Run each system's `IntelligentAgent` subclass to simulate real-time adaptive decisions
   from feedback (clicks, false-alarm rates, analyst review, delivery performance, engagement).
5. Print a readable summary for every system to the console.
6. Save a consolidated 6-panel dashboard to `output_results.png`.
7. Save all summary metrics to `results_summary.json`.

Expected console output ends with:
```
All five distributed AI systems executed successfully.
Summary metrics written to results_summary.json
```

## 🧠 Design Pattern (shared across all 5 systems)

```
Raw Data ──▶ RDD Ingestion ──▶ RDD Transformations ──▶ Reduced / Fused State
                                (map, groupByKey,
                                 reduceByKey, join,
                                 aggregateByKey)
                                                              │
                                                              ▼
                                              Intelligent Agent(s)
                                        (adaptive weight, real-time feedback,
                                         threshold tuning, decision output)
                                                              │
                                                              ▼
                                          Action: recommendation / alert /
                                          trade flag / transfer / content plan
```

- **Distributed processing (PySpark RDDs)** handles the large-scale, computationally heavy
  aggregation and fusion of data — this is the part that must scale horizontally.
- **Intelligent agents** are small, fast, interpretable decision units that consume the
  reduced Spark output and adapt continuously using a simple online-learning update rule:

  ```
  weight(t+1) = weight(t) + learning_rate * (feedback(t) - weight(t))
  ```

## 📊 Module Overview

1. **Recommendation Engine** — item-item collaborative filtering via `groupByKey` +
   `flatMap` + `reduceByKey`, fused with content-based genre scoring via `join`.
2. **Disaster Detection** — three-source RDD streams aggregated per zone and fused into a
   weighted risk index; a coordinator agent triggers CRITICAL / WATCH / NORMAL alerts.
3. **Fraud Detection** — per-account statistics via `aggregateByKey`, broadcast to workers,
   z-score + rule-based anomaly scoring, ranked with `filter` + `sortBy`.
4. **Supply Chain Optimization** — inventory vs. demand gap analysis via `join`, decentralized
   per-warehouse agents proposing cost/urgency-balanced transfers.
5. **Learning Analytics** — per-student and per-topic aggregation via `reduceByKey`, at-risk
   detection via `filter`, adaptive tutor agent for personalized content recommendations.

## 📝 Notes

- All datasets in this deliverable are **synthetic** (randomly generated with a fixed seed)
  so the script is fully self-contained and reproducible without external data sources.
- The code is written against PySpark's RDD API specifically (as required by the assignment)
  rather than the DataFrame/SQL API, to explicitly demonstrate `map`, `filter`, `groupByKey`,
  `reduceByKey`, `aggregateByKey`, `join`, `broadcast`, and `sortBy`.
- For production use, replace synthetic data generation with real ingestion (Kafka, HDFS, S3,
  JDBC) and consider Spark Structured Streaming for the continuously-arriving sources
  (disaster sensors, transactions, learner interactions).
