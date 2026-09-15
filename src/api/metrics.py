from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST


RECOMMENDATION_REQUESTS = Counter(
    "recommendation_requests_total",
    "Total recommendation API requests",
    ["endpoint", "status"],
)

RECOMMENDATION_LATENCY = Histogram(
    "recommendation_latency_seconds",
    "Recommendation API latency in seconds",
    ["endpoint"],
)

RECOMMENDATION_RESULTS = Histogram(
    "recommendation_results_count",
    "Number of recommendations returned per request",
    buckets=(1, 5, 10, 20, 50, 100),
)


def metrics_response():
    return generate_latest(), CONTENT_TYPE_LATEST
