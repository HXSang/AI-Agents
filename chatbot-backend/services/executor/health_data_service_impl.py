from typing import Any, Dict, List, Optional

from clients.redis_client import RedisClient
from models.models import HealthDataPayload, HealthMetricEntry
from services.health_data_service import HealthDataService
from utils.logger import logger


class HealthDataServiceImpl(HealthDataService):
    """Implementation of health data service with Redis cache"""

    def __init__(self):
        self.redis_client = RedisClient()
        self._cache_ttl = 86400  # 24 hours — health data must stay fresh;
        # raw payloads pushed by mobile clients are kept short enough that
        # stale rows (today's steps/sleep not yet re-uploaded) self-heal
        # within a day instead of pinning for 30 days.

    def _deduplicate_metrics_by_date(
        self, metric_list: Optional[List[HealthMetricEntry]]
    ) -> Optional[List[HealthMetricEntry]]:
        """
        Remove duplicate entries from a metric list based on 'date' field.
        If multiple entries have the same date, keep the first one (or the one with latest timestamp if dates are identical).

        Optimized for performance using dict for O(1) lookup.

        Args:
            metric_list: List of HealthMetricEntry objects

        Returns:
            Deduplicated list of HealthMetricEntry objects, or None if input is None/empty
        """
        if not metric_list:
            return None

        # Use dict with date as key for O(1) lookup and deduplication
        # Keep the first occurrence of each date (or can be modified to keep latest)
        seen_dates: Dict[str, HealthMetricEntry] = {}

        for entry in metric_list:
            if entry and entry.date:
                # If date already seen, skip (keep first occurrence)
                # Alternative: keep latest by comparing timestamps if needed
                if entry.date not in seen_dates:
                    seen_dates[entry.date] = entry

        # Convert back to list, preserving order (first occurrence kept)
        deduplicated = list(seen_dates.values())

        if len(deduplicated) < len(metric_list):
            logger.debug(
                f"Deduplicated metrics: {len(metric_list)} -> {len(deduplicated)} entries (removed {len(metric_list) - len(deduplicated)} duplicates)"
            )

        return deduplicated if deduplicated else None

    def _deduplicate_metrics_dict(self, metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deduplicate all metric lists in a metrics dictionary by date.

        Args:
            metrics_dict: Dictionary with metric types as keys and lists of HealthMetricEntry as values

        Returns:
            Dictionary with deduplicated metric lists
        """
        if not metrics_dict:
            return {}

        deduplicated_metrics = {}

        for metric_type, metric_list in metrics_dict.items():
            if metric_list is not None:
                # Convert to HealthMetricEntry objects if they're dicts
                if metric_list and len(metric_list) > 0:
                    if isinstance(metric_list[0], dict):
                        entries = [HealthMetricEntry(**entry) for entry in metric_list]
                    elif isinstance(metric_list[0], HealthMetricEntry):
                        entries = metric_list
                    else:
                        entries = []
                else:
                    entries = []

                # Deduplicate by date
                deduplicated = self._deduplicate_metrics_by_date(entries)
                if deduplicated:
                    # Convert back to dict format for storage
                    deduplicated_metrics[metric_type] = [
                        (
                            entry.model_dump()
                            if isinstance(entry, HealthMetricEntry)
                            else entry
                        )
                        for entry in deduplicated
                    ]
            else:
                # Keep None values as is
                deduplicated_metrics[metric_type] = None

        return deduplicated_metrics

    async def create_health_data(self, payload: HealthDataPayload) -> bool:
        """
        Create or update health data for a user in Redis cache
        - If no existing data: create new
        - If existing data: extend metrics lists (don't replace), update goals/consents

        Args:
            payload: The health data payload

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create cache key with format: health_data:user_id
            cache_key = f"health_data:{payload.user_id}"

            # Get existing data if available
            existing_data = await self.get_health_data(payload.user_id)

            # Convert payload to dict
            payload_dict = payload.model_dump()
            new_metrics = payload_dict.get("metrics", {})

            if existing_data is None:
                # No existing data - deduplicate new payload only once
                if new_metrics:
                    payload_dict["metrics"] = self._deduplicate_metrics_dict(
                        new_metrics
                    )
                data_dict = payload_dict
                logger.info(f"Creating new health data for user_id: {payload.user_id}")
            else:
                # Existing data found - merge/extend metrics first, then deduplicate once
                existing_dict = existing_data.model_dump()
                existing_metrics = existing_dict.get("metrics", {})

                # Merge metrics: combine new + existing first, then deduplicate once
                merged_metrics = {}
                # Get all metric types from HealthMetrics
                metric_types = [
                    "steps",
                    "heart_rate",
                    "blood_pressure",
                    "weight",
                    "sleep",
                    "calories",
                    "distance",
                    "flights_climbed",
                    "active_energy",
                    "resting_heart_rate",
                    "blood_oxygen",
                    "body_temperature",
                ]

                for metric_type in metric_types:
                    existing_list = existing_metrics.get(metric_type)
                    new_list = new_metrics.get(metric_type)

                    if new_list is not None:
                        # New payload has this metric
                        if existing_list:
                            # Combine lists: new first (newest), then existing
                            combined_list = new_list + existing_list
                            merged_metrics[metric_type] = combined_list
                        else:
                            # No existing data for this metric - use new data
                            merged_metrics[metric_type] = new_list
                    elif existing_list is not None:
                        # Keep existing data if new payload doesn't have this metric
                        merged_metrics[metric_type] = existing_list

                # Deduplicate all merged metrics once after combining everything
                if merged_metrics:
                    merged_metrics = self._deduplicate_metrics_dict(merged_metrics)
                    logger.info(
                        f"Merged and deduplicated metrics for user_id: {payload.user_id}"
                    )

                # Update goals and consents (replace if provided, otherwise keep existing)
                # Use new goals/consents if provided, otherwise keep existing
                merged_goals = (
                    payload_dict.get("goals")
                    if payload_dict.get("goals")
                    else existing_dict.get("goals", {})
                )
                merged_consents = (
                    payload_dict.get("consents")
                    if payload_dict.get("consents")
                    else existing_dict.get("consents", {})
                )

                # Reconstruct HealthDataPayload to ensure proper structure
                merged_data = {
                    "user_id": payload.user_id,
                    "metrics": merged_metrics,
                    "goals": merged_goals,
                    "consents": merged_consents,
                }
                data_dict = merged_data
                logger.info(
                    f"Merged health data for user_id: {payload.user_id} (extended metrics, updated goals/consents)"
                )

            # Store in Redis with JSON serialization
            await self.redis_client.set_data(
                cache_key, data_dict, expire=self._cache_ttl
            )

            logger.info(
                f"Successfully stored health data for user_id: {payload.user_id}"
            )
            return True

        except Exception as e:
            logger.error(
                f"Error storing health data for user_id {payload.user_id}: {str(e)}"
            )
            return False

    async def get_health_data(
        self, user_id: str, force_fresh: bool = False
    ) -> Optional[HealthDataPayload]:
        try:
            if force_fresh:
                logger.info(
                    f"Bypassing health_data cache for user_id: {user_id}"
                )
                return None

            # Create cache key with format: health_data:user_id
            cache_key = f"health_data:{user_id}"

            # Get data from Redis
            data_dict = await self.redis_client.get_data(cache_key)

            if data_dict is None:
                logger.info(f"No health data found for user_id: {user_id}")
                return None

            # Create HealthDataPayload from dictionary
            payload = HealthDataPayload(**data_dict)

            logger.info(f"Successfully retrieved health data for user_id: {user_id}")
            return payload

        except Exception as e:
            logger.error(
                f"Error retrieving health data for user_id {user_id}: {str(e)}"
            )
            return None
