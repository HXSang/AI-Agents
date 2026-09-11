"""Health insight extract + process pipeline (independent of chat agent)."""

__all__ = [
    "HealthDataExtractor",
    "HealthInsightProcessor",
]


def __getattr__(name: str):
    if name == "HealthDataExtractor":
        from insights.health.extractor import HealthDataExtractor

        return HealthDataExtractor
    if name == "HealthInsightProcessor":
        from insights.health.health_processor.processor import HealthInsightProcessor

        return HealthInsightProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
