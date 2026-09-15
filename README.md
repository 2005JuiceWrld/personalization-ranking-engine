\# Real-Time Personalization, Search \& Ranking Intelligence Engine



An end-to-end recommendation and ranking system designed around a production-style retrieval → ranking → diversification architecture.



The system combines implicit-feedback modeling, Two-Tower retrieval, approximate nearest-neighbor search, LightGBM learning-to-rank, MMR-based diversification, cold-start handling, and FastAPI serving.



\## System Architecture



```text

User / Visitor

&#x20;     │

&#x20;     ▼

Two-Tower User Embedding

&#x20;     │

&#x20;     ▼

HNSW Approximate Nearest Neighbor Search

&#x20;     │

&#x20;     │  Top-100 candidates

&#x20;     ▼

Feature Engineering

&#x20;     │

&#x20;     ▼

LightGBM LambdaRank

&#x20;     │

&#x20;     ▼

MMR Diversification

&#x20;     │

&#x20;     ▼

Top-N Recommendations

&#x20;     │

&#x20;     ▼

FastAPI

```



For unknown users, the system bypasses personalized retrieval and uses a popularity-based fallback.



\## Problem



A recommendation system must balance several competing requirements:



\* retrieve relevant items from a large catalog

\* personalize results from sparse implicit feedback

\* rank candidates using contextual features

\* keep retrieval latency low

\* avoid overly repetitive recommendations

\* handle users with no historical interactions

\* expose the model through a production-style API



This project implements those components as separate stages rather than treating recommendation as a single model.



\## Dataset



The system was developed using the \*\*RetailRocket e-commerce dataset\*\*.



Original event statistics:



\* 2.76M+ events

\* 1.4M+ users

\* 235K+ items

\* event types: view, add-to-cart, transaction



Interaction weights:



```text

view         = 1

add-to-cart  = 3

transaction  = 10

```



The event data was cleaned and split chronologically:



```text

80% Train

10% Validation

10% Test

```



This preserves temporal ordering and avoids randomly mixing future interactions into training data.



\## Retrieval Models



The project implements multiple recommendation approaches for comparison.



\### Popularity



A weighted interaction-frequency baseline.



\### Content-Based



Builds user preferences from product category information.



\### Collaborative Filtering



Uses implicit user-item interactions with truncated SVD representations.



\### Two-Tower Retrieval



A neural retrieval model learns separate user and item embeddings.



```text

User features ──► User Tower ──► User Embedding

&#x20;                                     │

&#x20;                                     ▼

&#x20;                                 Similarity

&#x20;                                     ▲

&#x20;                                     │

Item features ──► Item Tower ──► Item Embedding

```



The Two-Tower model uses 64-dimensional embeddings and is trained with implicit positive interactions and sampled negatives.



\## ANN Retrieval



Two ANN implementations were evaluated:



\* FAISS exact inner-product search

\* FAISS HNSW approximate search



HNSW configuration:



```text

M = 32

efConstruction = 200

efSearch = 128

```



Measured local benchmark:



```text

Exact search:

\~1.254 ms/query



HNSW:

\~0.084 ms/query



Top-10 neighbor agreement:

97.66%

```



This corresponds to approximately \*\*14.9× lower measured retrieval latency\*\* in the benchmark.



> The 97.66% figure is ANN neighbor recall against exact search. It is not the recommendation-system Recall@10 metric.



\## Learning-to-Rank



Retrieved candidates are passed to a LightGBM LambdaRank model.



Features include:



\### User features



\* interaction count

\* unique items

\* total interaction weight

\* transaction count



\### Item features



\* interaction count

\* unique users

\* total interaction weight

\* transaction count



\### User-item features



\* historical interactions

\* interaction weight

\* transaction count

\* recency



\### Retrieval features



\* retrieval score

\* retrieval rank



The ranking model is trained separately from the final temporal test evaluation.



\## Diversification



After ranking, MMR-style diversification is applied to reduce repetitive results.



The pipeline therefore separates:



```text

Relevance

&#x20;  ↓

Ranking

&#x20;  ↓

Diversification

```



The API exposes the resulting normalized relevance component as `mmr\_score`.



\## Evaluation



Baseline evaluation on the held-out recommendation task:



| Model                   | Recall@10 | HitRate@10 |  NDCG@10 |

| ----------------------- | --------: | ---------: | -------: |

| Popularity              |  0.006824 |   0.007708 | 0.003548 |

| Content-Based           |  0.001875 |   0.002892 | 0.000942 |

| Collaborative Filtering |  0.001686 |   0.004056 | 0.001285 |

| Two-Tower               |  0.002123 |   0.004143 | 0.001246 |



The sparse dataset makes personalization difficult. Popularity remains a strong baseline, while Two-Tower retrieval improves over content-based retrieval on the reported Recall@10 and HitRate@10 metrics.



\### Held-Out Temporal Ranking Evaluation



Using a separate temporal test set:



```text

Retrieval only



Recall@10   0.247133

HitRate@10  0.279570

NDCG@10     0.125327

```



With HNSW retrieval + LightGBM reranking:



```text

Recall@10   0.240860

HitRate@10  0.284946

NDCG@10     0.149167

```



The reranker therefore improved NDCG while slightly reducing Recall@10 on the evaluated subset.



\*\*Important limitation:\*\* the final ranking evaluation used a relatively small processed evaluation subset, so these results should be interpreted as an engineering validation rather than a statistically definitive production result.



An earlier experiment trained and evaluated on the same validation labels and produced artificially high metrics. That experiment was identified as label leakage and is \*\*not used as a project result\*\*.



\## End-to-End Performance



Local CPU benchmark:



```text

Queries:       100

Average/query: \~4.882 ms

Throughput:    \~204.83 QPS

```



This is a local benchmark of the in-process pipeline and should not be interpreted as production capacity.



The main optimized serving path is:



```text

Two-Tower

&#x20;   ↓

HNSW

&#x20;   ↓

100 candidates

&#x20;   ↓

Feature lookup

&#x20;   ↓

LightGBM

&#x20;   ↓

MMR

&#x20;   ↓

Top-N

```



\## API



FastAPI exposes:



```text

GET  /health

POST /recommend

GET  /docs

```



Example request:



```json

{

&#x20; "visitorid": 1,

&#x20; "k": 10

}

```



Example response:



```json

{

&#x20; "visitorid": 1,

&#x20; "recommendations": \[

&#x20;   {

&#x20;     "itemid": 344071,

&#x20;     "ltr\_score": 1.192949891090393,

&#x20;     "mmr\_score": 1.0

&#x20;   }

&#x20; ]

}

```



The API validates:



```text

1 <= k <= 100

```



Unknown users are handled through the popularity fallback.



\## Docker Deployment



The inference service is containerized using:



```text

Python 3.12

FastAPI

Uvicorn

PyTorch

FAISS

LightGBM

```



The Docker image contains only the artifacts required for inference.



Build:



```bash

docker build -t personalization-ranking-engine:1.0 .

```



Run:



```bash

docker run -d \\

&#x20; --name personalization-ranking-engine \\

&#x20; -p 8000:8000 \\

&#x20; personalization-ranking-engine:1.0

```



Health check:



```bash

curl http://127.0.0.1:8000/health

```



The container was validated against the local API and produced identical recommendation IDs, ordering, LTR scores, and MMR scores for the same request.



\## Project Structure



```text

personalization-ranking-engine/

│

├── src/

│   ├── api/

│   │   └── app.py

│   │

│   ├── data/

│   │   ├── prepare\_data.py

│   │   └── prepare\_items.py

│   │

│   ├── evaluation/

│   │   ├── metrics.py

│   │   ├── evaluate\_popularity.py

│   │   ├── evaluate\_content\_based.py

│   │   ├── evaluate\_cf.py

│   │   ├── evaluate\_two\_tower.py

│   │   └── evaluate\_ltr\_test.py

│   │

│   ├── features/

│   │   └── ranking\_features.py

│   │

│   ├── models/

│   │   ├── popularity.py

│   │   ├── content\_based.py

│   │   ├── collaborative\_filtering.py

│   │   ├── two\_tower.py

│   │   └── two\_tower\_recommender.py

│   │

│   ├── pipeline/

│   │   ├── recommend.py

│   │   ├── benchmark\_pipeline.py

│   │   └── profile\_pipeline.py

│   │

│   ├── ranking/

│   │   ├── build\_ranking\_dataset.py

│   │   ├── build\_ltr\_test\_dataset.py

│   │   └── train\_ltr.py

│   │

│   └── retrieval/

│       ├── build\_faiss\_index.py

│       ├── build\_hnsw\_index.py

│       ├── benchmark\_hnsw.py

│       └── diversification.py

│

├── tests/

│   └── test\_api.py

│

├── Dockerfile

├── .dockerignore

├── .gitignore

├── requirements.txt

└── README.md

```



\## Testing



FastAPI integration tests cover:



\* health endpoint

\* known-user recommendations

\* unknown-user cold start

\* invalid `k` below minimum

\* invalid `k` above maximum

\* maximum `k=100`



Current result:



```text

6 passed

```



\## Engineering Decisions



\### Why Two-Stage Retrieval + Ranking?



Running a complex ranking model over the entire catalog is expensive.



Instead:



```text

Large catalog

&#x20;   ↓

Fast retrieval

&#x20;   ↓

Small candidate set

&#x20;   ↓

Expensive ranking

```



This allows more sophisticated ranking while keeping inference practical.



\### Why HNSW?



Exact nearest-neighbor search provides a useful correctness baseline, while HNSW provides a latency/recall tradeoff suitable for large-scale retrieval.



\### Why a Popularity Fallback?



New or anonymous users may have no historical representation. A deterministic popularity model provides a safe fallback instead of returning an error or attempting meaningless personalization.



\### Why Temporal Evaluation?



Recommendation systems operate in time. A chronological train/validation/test split provides a more realistic evaluation than randomly mixing future interactions into training data.



\## Limitations



The RetailRocket dataset is highly sparse and provides limited product metadata.



Consequently:



\* popularity is difficult to beat consistently

\* personalization metrics are relatively low

\* category-based content features are limited

\* ranking evaluation has a small final processed subset

\* local latency benchmarks are not production capacity measurements



The system is therefore presented as an \*\*end-to-end ML engineering system and architecture demonstration\*\*, not as a claim of production recommendation quality.



\## Technologies



```text

Python

PyTorch

NumPy

Pandas

Scikit-learn

SciPy

FAISS

HNSW

LightGBM

FastAPI

Uvicorn

Docker

```



\## Core Takeaway



This project demonstrates a complete recommendation serving architecture rather than a single recommendation model:



```text

Implicit Feedback

&#x20;      ↓

Candidate Retrieval

&#x20;      ↓

Two-Tower Embeddings

&#x20;      ↓

HNSW ANN Search

&#x20;      ↓

Feature Engineering

&#x20;      ↓

LightGBM Learning-to-Rank

&#x20;      ↓

MMR Diversification

&#x20;      ↓

Cold-Start Fallback

&#x20;      ↓

FastAPI

&#x20;      ↓

Docker

```



The emphasis is on \*\*retrieval architecture, ranking, evaluation discipline, latency engineering, serving, and reproducibility\*\*.



