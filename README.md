# Real-Time Personalization, Search & Ranking Intelligence Engine

A production-oriented recommendation and ranking platform that combines **deep retrieval, hybrid search, learning-to-rank, real-time behavioral features, session personalization, category affinity, and diversity optimization**.

The system is designed around a realistic recommendation architecture:

```text
                         ┌──────────────────────┐
                         │    User / Client     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      FastAPI API     │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    │                                │
                    ▼                                ▼
             Redis Online Features             Request Context
                    │                                │
                    └───────────────┬────────────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ Candidate Retrieval  │
                         └──────────┬───────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 │                  │                  │
                 ▼                  ▼                  ▼
          Two-Tower + HNSW       BM25             Behavioral /
          Semantic Retrieval     Lexical           Retrieval Signals
                 │                  │                  │
                 └──────────┬───────┴──────────────────┘
                            ▼
                     RRF Candidate Fusion
                            │
                            ▼
                  LightGBM LambdaRank
                            │
                            ▼
              Session + Category Personalization
                            │
                            ▼
                    MMR Diversification
                            │
                            ▼
                         Top-N
                            │
                            ▼
                       FastAPI Response

Streaming Path:

User Events → Kafka → Feature Consumer → Redis
                                      │
                                      └──► Online Features

Observability:

FastAPI → Prometheus
ML Experiments → MLflow
```

---

## 1. Problem

Large recommendation systems must solve several problems simultaneously:

* retrieve relevant candidates from a large item universe
* incorporate historical user behavior
* react to recent user activity
* combine multiple retrieval strategies
* rank candidates using behavioral signals
* personalize recommendations by session and category
* avoid repetitive recommendations
* maintain low serving latency
* support real-time feature updates
* monitor the production API

A single recommendation model is insufficient for these requirements.

This project therefore implements a **multi-stage retrieval and ranking architecture** rather than treating recommendation as a single prediction problem.

---

# 2. System Architecture

## Offline ML Pipeline

```text
RetailRocket Events
        │
        ▼
Data Validation
        │
        ▼
Temporal Train / Validation / Test Split
        │
        ├──────────────► Popularity Baseline
        │
        ├──────────────► Implicit Collaborative Filtering
        │
        ├──────────────► Two-Tower Retrieval
        │                       │
        │                       ▼
        │                  FAISS HNSW
        │
        ├──────────────► BM25 Attribute Retrieval
        │
        └──────────────► Category Representation
                                │
                                ▼
                         Temporal SVD Embedding

Candidate Retrieval
        │
        ▼
Candidate Fusion / RRF
        │
        ▼
Ranking Feature Generation
        │
        ▼
LightGBM LambdaRank
        │
        ▼
Offline Evaluation
```

## Online Serving Pipeline

```text
Request
  │
  ├── visitorid
  ├── k
  └── optional query/context
  │
  ▼
FastAPI
  │
  ▼
Online Feature Store ──► Redis
  │
  ├── recent items
  ├── event counts
  ├── behavior score
  └── last event timestamp
  │
  ▼
Candidate Generation
  │
  ├── Two-Tower / HNSW
  └── BM25 + RRF
  │
  ▼
LightGBM Ranker
  │
  ▼
Personalization
  │
  ├── Category affinity
  └── Session affinity
  │
  ▼
MMR Diversification
  │
  ▼
Top-N Recommendations
```

---

# 3. Dataset

The project uses the **RetailRocket recommender dataset**.

Approximate scale:

| Component                     |     Scale |
| ----------------------------- | --------: |
| Events                        |    ~2.76M |
| Users                         |     ~1.4M |
| Items                         |     ~235K |
| Two-Tower users               | 1,123,765 |
| Two-Tower items               |   212,915 |
| Retrieval embedding dimension |        64 |

Interaction weights:

```text
view         → 1
addtocart    → 3
transaction  → 10
```

A strict temporal split was used:

```text
Train
  ↓
Validation
  ↓
Test
```

The test period is strictly later than the training period.

This is important because recommendation evaluation can easily become invalid if future interactions leak into candidate generation or user features.

---

# 4. Retrieval System

## Two-Tower Retrieval

A two-tower neural architecture learns separate representations for users and items.

```text
User Features ──► User Tower ──► 64D User Embedding
                                      │
                                      │ similarity
                                      ▼
Item Features ──► Item Tower ──► 64D Item Embedding
```

Training uses implicit interaction data with sampled negative items.

Configuration:

```text
Framework:       PyTorch
Embedding dim:   64
Loss:             BCEWithLogitsLoss
Optimizer:        Adam
Batch size:       4096
Epochs:           3
Negative samples: 2 / positive
```

The trained item embeddings are indexed using FAISS.

---

# 5. FAISS Retrieval

Two FAISS configurations were evaluated.

### Exact Search

```text
IndexFlatIP
```

Measured approximately:

```text
~1.25 ms/query
```

### Approximate Search

```text
HNSW
M = 32
efConstruction = 200
efSearch = 128
```

Measured approximately:

```text
~0.084 ms/query
```

The measured HNSW neighbor agreement at `efSearch=128` was approximately:

```text
97.66%
```

This benchmark measures retrieval-neighbor agreement, not end-to-end recommendation quality.

---

# 6. Hybrid Retrieval

The system combines semantic retrieval with lexical/attribute retrieval.

## Semantic Retrieval

Two-Tower embeddings provide the primary semantic retrieval mechanism.

## BM25 Retrieval

The dataset does not contain rich natural-language product descriptions, so BM25 is used as an **attribute/category-oriented lexical retrieval layer**.

Indexed metadata includes product properties and category information.

## Reciprocal Rank Fusion

Candidate lists can be combined using Reciprocal Rank Fusion:

```text
RRF score = Σ 1 / (k + rank)
```

This allows candidates supported by different retrieval strategies to enter the ranking stage without requiring their raw scores to be directly comparable.

---

# 7. Learning-to-Rank

Candidate features are generated from user, item, and user-item interaction history.

The ranking model is:

```text
LightGBM LambdaRank
```

Configuration:

```text
n_estimators       = 300
learning_rate      = 0.05
num_leaves         = 31
min_child_samples  = 50
subsample           = 0.8
colsample_bytree    = 0.8
reg_alpha           = 0.1
reg_lambda          = 1
```

Ranking features include:

* retrieval score
* retrieval rank
* user interaction count
* unique items interacted with
* total interaction weight
* transaction count
* item interaction count
* unique users per item
* item interaction weight
* item transaction count
* user-item interaction count
* user-item interaction weight
* user-item transaction count
* user-item recency

---

# 8. Session Personalization

Recent user interactions are converted into a session affinity signal.

The system compares candidate item embeddings against recently interacted item embeddings.

The resulting session affinity captures short-term intent that may differ from long-term user behavior.

```text
Recent User Items
        │
        ▼
Item Embeddings
        │
        ▼
Candidate Similarity
        │
        ▼
Session Affinity
```

---

# 9. Category-Aware Personalization

Because the dataset contains category metadata but limited natural-language product information, category representation is used as an additional personalization signal.

Category paths are converted into a sparse item-category matrix and reduced using SVD.

Temporal construction prevents future metadata from entering the representation used for evaluation.

Configuration:

```text
Category embedding dimension: 32
Temporal vocabulary:           1,488 categories
Item universe:                 212,915
```

The selected category personalization weight was determined through offline validation.

The final production scoring structure is:

```text
personalized_score =
    ltr_score
    + 4.0 × category_affinity
    + 0.15 × session_affinity
```

---

# 10. Diversity Optimization

The ranking stage can produce highly similar recommendations.

Maximum Marginal Relevance (MMR) is therefore applied after personalization to balance:

```text
Relevance
    +
Diversity
```

This reduces redundant recommendations while preserving high-scoring candidates.

---

# 11. Offline Evaluation

Evaluation was performed using a temporal test period rather than random splitting.

The evaluation pipeline:

```text
Historical interactions
        │
        ▼
Candidate Retrieval
        │
        ▼
Seen-item filtering
        │
        ▼
Test-period positives
        │
        ▼
Ranking
        │
        ▼
Recall@10
HitRate@10
NDCG@10
```

### Stage 5 Category Personalization

The selected configuration produced:

| Metric     |      Score |
| ---------- | ---------: |
| Recall@10  | **0.4475** |
| HitRate@10 | **0.4785** |
| NDCG@10    | **0.2676** |

Compared with the existing LightGBM ranking baseline:

```text
Recall@10:  +0.2067
HitRate@10: +0.1935
NDCG@10:    +0.1185
```

The evaluation contained a relatively small number of positive test labels, so these measurements should be interpreted as offline experimental results rather than production-level statistical guarantees.

---

# 12. Retrieval Baselines

Initial retrieval experiments established reference points.

| Retriever     | Recall@10 | HitRate@10 |  NDCG@10 |
| ------------- | --------: | ---------: | -------: |
| Popularity    |  0.006824 |   0.007708 | 0.003548 |
| Content-based |  0.001875 |   0.002892 | 0.000942 |
| Two-Tower     |  0.002123 |   0.004143 | 0.001246 |

These baseline results demonstrate why the project uses a **multi-stage ranking architecture** rather than relying on a single retrieval model.

---

# 13. Real-Time Feature Pipeline

The online feature system uses Kafka and Redis.

```text
Interaction Event
       │
       ▼
     Kafka
       │
       ▼
Feature Consumer
       │
       ▼
     Redis
       │
       ├── recent_items
       ├── event_counts
       ├── behavior_score
       └── last_event_ts
```

Kafka topic:

```text
interaction-events
```

The feature consumer updates Redis immediately after receiving events.

This allows recommendation requests to incorporate recent behavior without rebuilding the offline feature dataset.

---

# 14. API

The recommendation service is implemented with FastAPI.

Primary endpoints:

```text
GET  /health
POST /recommend
GET  /metrics
```

Example request:

```json
{
  "visitorid": 1,
  "k": 10
}
```

The recommendation response exposes ranking and personalization signals, including:

```text
itemid
ltr_score
mmr_score
category_affinity
session_affinity
personalized_score
```

---

# 15. Observability

Prometheus instrumentation tracks:

```text
recommendation_requests_total
recommendation_latency_seconds
recommendation_results_count
```

Metrics are exposed through:

```text
/metrics
```

Prometheus continuously scrapes the API.

This enables monitoring of:

* request volume
* HTTP status distribution
* latency
* recommendation result counts

---

# 16. MLflow

MLflow is used to track Two-Tower training experiments.

Tracked information includes:

```text
training parameters
epoch losses
training sample count
model artifact
user mapping
item mapping
```

Tracking backend:

```text
SQLite
```

This provides reproducibility for model-training experiments.

---

# 17. Docker Deployment

The complete application stack is containerized.

```text
┌─────────────────────────────────────────────┐
│              Docker Compose                 │
│                                             │
│  ┌────────────┐      ┌───────────────┐     │
│  │  FastAPI   │─────►│     Redis     │     │
│  └─────┬──────┘      └───────────────┘     │
│        │                                    │
│        │             ┌───────────────┐      │
│        └────────────►│     Kafka     │      │
│                      └───────────────┘      │
│                                             │
│  ┌──────────────────┐                       │
│  │ Feature Consumer │                       │
│  └──────────────────┘                       │
│                                             │
│  ┌──────────────────┐                       │
│  │   Prometheus     │                       │
│  └──────────────────┘                       │
└─────────────────────────────────────────────┘
```

Services:

```text
api
feature-consumer
kafka
redis
prometheus
```

The API image contains the trained retrieval, ranking, and category artifacts required for serving.

---

# 18. CI/CD

GitHub Actions validates the repository on pushes and pull requests.

The CI pipeline performs:

```text
Checkout
   ↓
Python 3.12 setup
   ↓
Dependency installation
   ↓
Python compilation
   ↓
Test suite
   ↓
Docker Compose validation
   ↓
API image build
```

This provides automated validation before changes are merged.

---

# 19. Serving Performance

A 100-request concurrent benchmark was performed against the Dockerized API.

Configuration:

```text
Requests:    100
Concurrency: 10
```

Results:

| Metric              |          Result |
| ------------------- | --------------: |
| Successful requests |   **100 / 100** |
| Errors              |           **0** |
| Throughput          | **75.23 req/s** |
| p50                 |   **128.81 ms** |
| p95                 |   **170.75 ms** |
| p99                 |   **206.15 ms** |
| Maximum             |   **229.15 ms** |

The benchmark demonstrates successful concurrent serving under the tested local environment.

These numbers are environment-specific and should not be interpreted as a cloud-production SLA.

---

# 20. Engineering Decisions

### Temporal evaluation instead of random splitting

Random recommendation splits can leak future behavior into training or candidate construction.

The project uses chronological boundaries to better represent real deployment conditions.

### Multi-stage retrieval and ranking

Retrieval and ranking have different computational requirements.

The architecture therefore separates:

```text
Candidate Generation
        ↓
Ranking
        ↓
Personalization
        ↓
Diversification
```

### HNSW instead of exhaustive retrieval

Exact similarity search provides a useful correctness baseline, while HNSW provides a lower-latency approximate retrieval path.

### Redis for online state

Recent behavioral features change continuously and therefore should not require offline feature regeneration for every interaction.

### Category representation instead of conventional text embeddings

The source dataset has limited natural-language product information. Category and attribute signals therefore provide a more defensible representation than pretending the data contains rich product descriptions.

### MLflow for experiment tracking

Model parameters, losses, and artifacts are tracked independently from source code to improve reproducibility.

---

# 21. Limitations

The system is a portfolio-scale production-oriented implementation rather than a deployed commercial recommendation platform.

Important limitations include:

* RetailRocket is an implicit-feedback dataset.
* The dataset does not provide rich product descriptions.
* Offline evaluation has a limited number of positive test interactions.
* No claim of statistical significance is made for the reported offline improvements.
* Serving benchmarks were performed in a local Docker environment.
* Kafka is configured as a single-node development deployment.
* Redis is configured as a single-node deployment.
* MLflow uses SQLite for local experiment tracking.
* The system does not claim online A/B-test results because it has not been deployed to real users.

---

# 22. Technology Stack

### Machine Learning

```text
Python
PyTorch
LightGBM
scikit-learn
SciPy
NumPy
Pandas
```

### Retrieval

```text
FAISS
Two-Tower Retrieval
HNSW
BM25
Reciprocal Rank Fusion
```

### Personalization

```text
Session Affinity
Category Affinity
MMR Diversification
```

### Data / Streaming

```text
Kafka
Redis
```

### Serving

```text
FastAPI
Uvicorn
```

### MLOps / Observability

```text
MLflow
Prometheus
GitHub Actions
Docker
Docker Compose
```

---

# 23. Repository Structure

```text
personalization-ranking-engine/
│
├── data/
│   ├── processed/
│   └── features/
│
├── models/
│   ├── two_tower.pt
│   ├── two_tower_items_hnsw.faiss
│   ├── lightgbm_ranker.txt
│   ├── popularity.csv
│   ├── bm25_index.pkl
│   └── item_category_embedding_temporal.npy
│
├── src/
│   ├── api/
│   │   ├── app.py
│   │   └── metrics.py
│   │
│   ├── evaluation/
│   │
│   ├── features/
│   │
│   ├── models/
│   │   └── two_tower.py
│   │
│   ├── pipeline/
│   │   └── recommend.py
│   │
│   ├── retrieval/
│   │   └── bm25.py
│   │
│   └── streaming/
│       ├── producer.py
│       ├── consumer.py
│       ├── feature_consumer.py
│       └── online_features.py
│
├── infra/
│   └── prometheus/
│       └── prometheus.yml
│
├── tests/
│
├── Dockerfile
├── compose.yaml
├── requirements.txt
└── .github/
    └── workflows/
        └── ci.yml
```

---

# 24. End-to-End Data Flow

The complete system can be summarized as:

```text
                 USER INTERACTION
                       │
                       ▼
                    Kafka
                       │
                       ▼
              Online Feature Consumer
                       │
                       ▼
                    Redis
                       │
                       │
                       ▼
                  FastAPI API
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   User / Session State       Query / Context
          │                         │
          └────────────┬────────────┘
                       ▼
                Candidate Retrieval
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        Two-Tower/HNSW          BM25
             │                   │
             └─────────┬─────────┘
                       ▼
                     RRF
                       │
                       ▼
              LightGBM LambdaRank
                       │
                       ▼
             Personalization Layer
                       │
              ┌────────┴────────┐
              ▼                 ▼
       Category Affinity   Session Affinity
              │                 │
              └────────┬────────┘
                       ▼
                  MMR Diversity
                       │
                       ▼
                    Top-N
                       │
                       ▼
                 API Response
```

---

# 25. Project Outcome

This project demonstrates the implementation of a complete recommendation system beyond a standalone ML model.

The system covers:

```text
Data
 ↓
Temporal Evaluation
 ↓
Retrieval
 ↓
Approximate Nearest Neighbor Search
 ↓
Hybrid Search
 ↓
Learning-to-Rank
 ↓
Real-Time Features
 ↓
Personalization
 ↓
Diversification
 ↓
API Serving
 ↓
Observability
 ↓
Experiment Tracking
 ↓
Containerization
 ↓
CI/CD
```

The primary engineering objective was to build a recommendation platform where **retrieval quality, ranking quality, real-time behavior, serving performance, and operational infrastructure are treated as one system rather than isolated ML experiments**.
