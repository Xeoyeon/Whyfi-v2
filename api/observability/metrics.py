import logging
import time
from collections import Counter

logger = logging.getLogger(__name__)


class IngestionMetrics:
    def __init__(self):
        self.started_at = time.monotonic()
        self.success_terms = 0
        self.failed_terms = 0
        self.failure_reasons = Counter()
        self.uploaded_chunks = 0
        self.slow_fetch_count = 0
        self.timeout_count = 0
        self.batch_upload_seconds = 0.0
        self.embedding_seconds = 0.0
        self.qdrant_upload_seconds = 0.0

    @property
    def elapsed_seconds(self):
        return time.monotonic() - self.started_at

    def record_success(self, slow_fetch=False):
        self.success_terms += 1
        if slow_fetch:
            self.slow_fetch_count += 1

    def record_failure(self, reason):
        normalized_reason = reason or "Unknown error"
        self.failed_terms += 1
        self.failure_reasons[normalized_reason] += 1

        if "timeout" in normalized_reason.lower():
            self.timeout_count += 1

    def record_uploaded_chunks(self, chunk_count):
        self.uploaded_chunks += chunk_count

    def record_batch_upload_time(self, elapsed_seconds):
        self.batch_upload_seconds += elapsed_seconds

    def record_embedding_time(self, elapsed_seconds):
        self.embedding_seconds += elapsed_seconds

    def record_qdrant_upload_time(self, elapsed_seconds):
        self.qdrant_upload_seconds += max(elapsed_seconds, 0.0)

    def log_summary(self):
        elapsed = self.elapsed_seconds
        elapsed_minutes = elapsed / 60 if elapsed else 0
        terms_per_min = self.success_terms / elapsed_minutes if elapsed_minutes else 0
        chunks_per_min = self.uploaded_chunks / elapsed_minutes if elapsed_minutes else 0
        failure_summary = (
            ", ".join(f"{reason}={count}" for reason, count in self.failure_reasons.most_common())
            or "none"
        )

        logger.info(
            "[metrics] total_elapsed=%.1fs success_terms=%s failed_terms=%s "
            "terms_per_min=%.2f chunks_per_min=%.2f uploaded_chunks=%s "
            "slow_fetch_count=%s timeout_count=%s batch_upload_time=%.1fs "
            "embedding_time=%.1fs qdrant_upload_time=%.1fs "
            "failure_reasons={%s}",
            elapsed,
            self.success_terms,
            self.failed_terms,
            terms_per_min,
            chunks_per_min,
            self.uploaded_chunks,
            self.slow_fetch_count,
            self.timeout_count,
            self.batch_upload_seconds,
            self.embedding_seconds,
            self.qdrant_upload_seconds,
            failure_summary,
        )


class TimedEmbeddings:
    def __init__(self, embeddings, metrics):
        self.embeddings = embeddings
        self.metrics = metrics

    def embed_documents(self, texts):
        started_at = time.monotonic()
        try:
            return self.embeddings.embed_documents(texts)
        finally:
            self.metrics.record_embedding_time(time.monotonic() - started_at)

    def embed_query(self, text):
        started_at = time.monotonic()
        try:
            return self.embeddings.embed_query(text)
        finally:
            self.metrics.record_embedding_time(time.monotonic() - started_at)

    def __getattr__(self, name):
        return getattr(self.embeddings, name)
