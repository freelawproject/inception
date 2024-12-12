from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "inception_requests_total",
    "Total number of embedding requests",
    ["endpoint"],
)

PROCESSING_TIME = Histogram(
    "inception_processing_seconds",
    "Time spent processing embedding requests",
    ["endpoint"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")),
)

ERROR_COUNT = Counter(
    "inception_errors_total",
    "Total number of errors",
    ["endpoint", "error_type"],
)

CHUNK_COUNT = Counter(
    "inception_chunks_total",
    "Total number of text chunks processed",
    ["endpoint"],
)

MODEL_LOAD_TIME = Histogram(
    "inception_model_load_seconds",
    "Time spent loading the model",
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, float("inf")),
)
