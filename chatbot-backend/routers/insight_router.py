"""Insight Router for productivity analysis"""

import asyncio
import httpx
import json
import os
import sys
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from insights.helpers.empty_value_cleanser import (
    DEFAULT_KEEP_TOP_LEVEL_LIST_KEYS,
    DISPLAY_DROP_EMPTY_LIST_KEYS as _DISPLAY_DROP_EMPTY_LIST_KEYS,
    clean_payload,
    drop_empty_lists,
)
from models.models import (
    HealthInsightRequest,
    HealthInsightResponse,
    OverallInsightRequest,
    OverallInsightResponse,
    ProductivityInsightRequest,
    ProductivityInsightResponse,
)
from services.service_factory import ServiceFactory
from utils.logger import logger

router = APIRouter(prefix="/insight", tags=["insight"])


def _strip_zoneinfo(obj):
    """Recursively strip ZoneInfo, datetime, date for JSON serialization."""
    import datetime as dt
    from zoneinfo import ZoneInfo

    if isinstance(obj, dict):
        return {k: _strip_zoneinfo(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_strip_zoneinfo(item) for item in obj]
    if isinstance(obj, dt.datetime):
        return obj.isoformat()
    if isinstance(obj, dt.date):
        return obj.isoformat()
    if isinstance(obj, ZoneInfo):
        return str(obj)
    return obj

# Allow imports of the scratch/ package from the demo html.
_SCRATCH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scratch"
)
if _SCRATCH_DIR not in sys.path:
    sys.path.insert(0, _SCRATCH_DIR)


@router.get("/mock-raw-data")
async def get_mock_raw_data(scenario: str = Query(...)):
    """Return the raw_data dict for a given mock scenario name.

    Used by ``insight_demo.html`` so the frontend can preview and inject
    deterministic test data without any real user or API key.
    """
    try:
        import importlib.util as _ilu
        import pathlib as _pl

        _path = _pl.Path(_SCRATCH_DIR) / "mock_raw_data.py"
        _spec = _ilu.spec_from_file_location("mock_raw_data", _path)
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        raw = _mod.get_raw_data(scenario)
    except KeyError as ke:
        raise HTTPException(status_code=404, detail=str(ke))
    except Exception as e:
        logger.exception(f"Failed to load mock scenario {scenario}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    serialized = json.dumps(raw, ensure_ascii=False, default=str)
    return {
        "status": "success",
        "scenario": scenario,
        "raw_data": _strip_zoneinfo(raw),
        "size_bytes": len(serialized),
    }


async def _resolve_user_timezone(
    user_id: Optional[str],
    payload_timezone: Optional[str],
    insight_service: Any,
) -> str:
    return await insight_service._data_collector.resolve_timezone(
        user_id=user_id,
        payload_timezone=payload_timezone,
    )


@router.post("/preview_raw_data")
async def preview_raw_data(payload: dict = Body(...)):
    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    insight_service = ServiceFactory.get_insight_service()
    timezone = await _resolve_user_timezone(
        user_id=user_id,
        payload_timezone=payload.get("timezone"),
        insight_service=insight_service,
    )
    provider_name = payload.get("provider_name") or ""
    language = payload.get("language") or "vi-VN"
    # Default health — never use domain=all (that pulls finance).
    domain = (payload.get("domain") or "health").strip().lower()
    if domain not in ("health", "productivity", "overall"):
        domain = "health"

    try:
        insight_service = ServiceFactory.get_insight_service()
        from insights.health.extractor import HealthDataExtractor

        extractor = HealthDataExtractor(
            data_collector=insight_service._data_collector,
            external_api_service=insight_service.external_api_service,
        )
        raw_data = await extractor.extract(
            user_id=user_id,
            timezone=timezone,
            provider_name=provider_name,
            force_update=False,
            domain=domain,
        )

        serialized = json.dumps(raw_data, ensure_ascii=False, default=str)
        raw_data = drop_empty_lists(
            clean_payload(
                raw_data,
                keep_top_level_list_keys=DEFAULT_KEEP_TOP_LEVEL_LIST_KEYS,
            ),
            keys=_DISPLAY_DROP_EMPTY_LIST_KEYS,
        )
        user_profile_data = raw_data.get("user_profile") or {}
        calendar_events = raw_data.get("calendar_events") or []
        calendar_metrics = raw_data.get("calendar_metrics") or {}
        health_params = raw_data.get("health_params") or {}
        today_health_stats = raw_data.get("today_health_stats") or {}
        api_fetches = raw_data.get("_extract_api_fetches") or []

        return {
            "status": "success",
            "user_id": user_id,
            "timezone": timezone,
            "language": language,
            "domain": domain,
            "raw_data": _strip_zoneinfo(raw_data),
            "size_bytes": len(serialized),
            "api_fetches": api_fetches,
            "field_counts": {
                "user_profile_keys": len(user_profile_data) if isinstance(user_profile_data, dict) else 0,
                "calendar_events": len(calendar_events) if isinstance(calendar_events, list) else 0,
                "calendar_metrics_keys": len(calendar_metrics) if isinstance(calendar_metrics, dict) else 0,
                "today_health_stats_keys": (
                    len(today_health_stats) if isinstance(today_health_stats, dict) else 0
                ),
                "health_params_keys": len(health_params) if isinstance(health_params, dict) else 0,
                "api_fetches": len(api_fetches) if isinstance(api_fetches, list) else 0,
                "top_level_keys": len(raw_data),
                "has_finance": any(
                    k in raw_data
                    for k in (
                        "finance_summary",
                        "finance_goals",
                        "finance_bills",
                        "finance_logs",
                        "finance_budgets",
                    )
                ),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"preview_raw_data failed for user_id {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"preview_raw_data failed: {str(e)}",
        )


def _extract_prose(insight: Any) -> Optional[str]:
    """Pull prose string from service output."""
    if insight is None:
        return None
    if isinstance(insight, dict):
        nested = insight.get("insight")
        if isinstance(nested, str) and nested.strip():
            return nested
        # Already-shaped overall payload {great_job, need_attention, opportunity}
        for key in ("need_attention", "great_job", "opportunity"):
            value = insight.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None
    if isinstance(insight, str):
        try:
            parsed = json.loads(insight)
        except json.JSONDecodeError:
            return insight
        if isinstance(parsed, dict):
            return _extract_prose(parsed)
        if parsed is None:
            return None
        return parsed if isinstance(parsed, str) else str(parsed)
    return str(insight)


def _wrap_domain_insight(domain: str, prose: Optional[str]) -> Optional[str]:
    """Wrap prose into the per-domain JSON blob expected by FE (health/productivity)."""
    if prose is None:
        return None
    if domain in ("health", "productivity"):
        payload = {"point1": prose, "point2": ""}
    else:
        return prose
    return json.dumps(payload, ensure_ascii=False)


def _wrap_overall_insight(prose: Optional[str]) -> Optional[Dict[str, Optional[str]]]:
    if prose is None:
        return None
    return {
        "great_job": "",
        "need_attention": prose,
        "opportunity": "",
    }


@router.post("/productivity", response_model=ProductivityInsightResponse)
async def analyze_productivity(payload: ProductivityInsightRequest):
    """Analyze user productivity based on today's calendar events."""
    try:
        logger.info(f"📊 Productivity analysis request for user_id: {payload.user_id}")
        endpoint_start = time.perf_counter()
        logger.info(
            f"⏱️ [TIMING][productivity][start] user_id={payload.user_id} "
            f"force_update={payload.force_update} lang={payload.language}"
        )

        insight_service = ServiceFactory.get_insight_service()
        result = await insight_service.analyze_productivity_insight(
            user_id=payload.user_id,
            timezone=payload.timezone,
            provider_name=payload.provider_name or "",
            language=payload.language or "en-US",
            force_update=payload.force_update,
        )
        logger.info(
            f"⏱️ [TIMING][productivity][end] user_id={payload.user_id} "
            f"total_elapsed={time.perf_counter() - endpoint_start:.3f}s"
        )
        logger.info(f"🔍 Productivity analysis result: {result}")
        if result["status"] == "error":
            logger.error(
                f"❌ Productivity analysis failed for user_id: {payload.user_id}, error: {result.get('error')}"
            )
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Failed to analyze productivity"),
            )

        return ProductivityInsightResponse(
            status=result["status"],
            user_id=result["user_id"],
            insight=_wrap_domain_insight(
                "productivity", _extract_prose(result["insight"])
            ),
            error=result.get("error"),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"❌ Error in productivity analysis endpoint for user_id {payload.user_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze productivity: {str(e)}",
        )


@router.post("/overall_insight", response_model=OverallInsightResponse)
async def analyze_overall_insight(payload: OverallInsightRequest):
    """Analyze overall insights (combining cross-domain signals)."""
    try:
        logger.info(
            f"📊 Overall insight analysis request for user_id: {payload.user_id}"
        )
        endpoint_start = time.perf_counter()
        logger.info(
            f"⏱️ [TIMING][overall][start] user_id={payload.user_id} "
            f"force_update={payload.force_update} lang={payload.language}"
        )

        insight_service = ServiceFactory.get_insight_service()
        result = await insight_service.analyze_overall_insight(
            user_id=payload.user_id,
            timezone=payload.timezone,
            provider_name=payload.provider_name or "",
            language=payload.language or "en-US",
            force_update=payload.force_update,
        )
        logger.info(
            f"⏱️ [TIMING][overall][end] user_id={payload.user_id} "
            f"total_elapsed={time.perf_counter() - endpoint_start:.3f}s"
        )
        if result["status"] == "error":
            logger.error(
                f"❌ Overall insight analysis failed for user_id: {payload.user_id}, error: {result.get('error')}"
            )
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Failed to analyze overall insight"),
            )

        return OverallInsightResponse(
            status=result["status"],
            user_id=result["user_id"],
            insight=_wrap_overall_insight(_extract_prose(result["insight"])),
            error=result.get("error"),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"❌ Error in overall insight analysis endpoint for user_id {payload.user_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze overall insight: {str(e)}",
        )


@router.post("/health_insight", response_model=HealthInsightResponse)
async def analyze_health_insight(payload: HealthInsightRequest):
    """Analyze user health insight in exactly two points: summary + suggestion."""
    try:
        logger.info(
            f"📊 Health insight analysis request for user_id: {payload.user_id}"
        )
        endpoint_start = time.perf_counter()
        logger.info(
            f"⏱️ [TIMING][health][start] user_id={payload.user_id} "
            f"force_update={payload.force_update} lang={payload.language}"
        )

        insight_service = ServiceFactory.get_insight_service()
        result = await insight_service.analyze_health_insight(
            user_id=payload.user_id,
            timezone=payload.timezone,
            provider_name=payload.provider_name or "",
            language=payload.language or "en-US",
            force_update=payload.force_update,
        )
        logger.info(
            f"⏱️ [TIMING][health][end] user_id={payload.user_id} "
            f"total_elapsed={time.perf_counter() - endpoint_start:.3f}s"
        )
        if result["status"] == "error":
            logger.error(
                f"❌ Health insight analysis failed for user_id: {payload.user_id}, error: {result.get('error')}"
            )
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Failed to analyze health insight"),
            )

        return HealthInsightResponse(
            status=result["status"],
            user_id=result["user_id"],
            insight=_wrap_domain_insight("health", _extract_prose(result["insight"])),
            error=result.get("error"),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"❌ Error in health insight analysis endpoint for user_id {payload.user_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze health insight: {str(e)}",
        )


@router.post("/debug")
async def debug_insight_pipeline(payload: dict = Body(...)):
    user_id = payload.get("user_id")
    raw_data_override = payload.get("raw_data")
    if not user_id and not raw_data_override:
        raise HTTPException(
            status_code=400,
            detail="Either user_id or raw_data is required",
        )

    insight_service = ServiceFactory.get_insight_service()
    payload_tz = payload.get("timezone")
    if user_id:
        timezone = await _resolve_user_timezone(
            user_id=user_id,
            payload_timezone=payload_tz,
            insight_service=insight_service,
        )
    else:
        override_profile = (payload.get("raw_data") or {}).get("user_profile") or {}
        override_tz = (
            override_profile.get("timezone")
            or override_profile.get("time_data_timezone")
        )
        timezone = await _resolve_user_timezone(
            user_id=None,
            payload_timezone=override_tz or payload_tz,
            insight_service=insight_service,
        )
    provider_name = payload.get("provider_name") or ""
    language = payload.get("language") or "vi-VN"
    domains = payload.get("domains") or ["health", "productivity", "overall"]
    return_raw_data = payload.get("return_raw_data", False)

    try:
        if user_id and hasattr(insight_service, "invalidate_user_narrative_cache"):
            try:
                await insight_service.invalidate_user_narrative_cache(user_id)
            except Exception as _e:
                logger.warning(
                    f"⚠️ /insight/debug could not invalidate narrative cache "
                    f"for {user_id}: {_e}"
                )

        from insights.health.extractor import HealthDataExtractor
        from modules.data_collector import strip_finance_from_raw_data
        from datetime import datetime

        from zoneinfo import ZoneInfo
        tz = ZoneInfo(timezone)
        current_dt = datetime.now(tz)

        valid_domains = [
            d for d in domains if d in ("health", "productivity", "overall")
        ]
        if not valid_domains:
            valid_domains = ["health"]

        # Never use domain=all (pulls finance). Prefer single domain; else overall
        # (covers health+productivity topics without finance).
        if len(valid_domains) == 1:
            extract_domain = valid_domains[0]
        elif "overall" in valid_domains:
            extract_domain = "overall"
        else:
            extract_domain = "overall"

        if raw_data_override:
            raw_data = dict(raw_data_override)
            user_profile_data = raw_data.get("user_profile") or {}
            if not user_id:
                user_id = (
                    user_profile_data.get("user_id")
                    if isinstance(user_profile_data, dict)
                    else None
                ) or "mock-user"
            raw_data["timezone"] = timezone
            raw_data["current_time"] = current_dt.isoformat()
            raw_data["collect_domain"] = extract_domain
            # Mock/pasted payloads must not carry finance into daily insight debug
            strip_finance_from_raw_data(raw_data)
        else:
            extractor = HealthDataExtractor(
                data_collector=insight_service._data_collector,
                external_api_service=insight_service.external_api_service,
            )
            raw_data = await extractor.extract(
                user_id=user_id,
                timezone=timezone,
                provider_name=provider_name,
                force_update=False,
                domain=extract_domain,
            )

        # Hard guarantee: no finance in daily insight debug extract
        strip_finance_from_raw_data(raw_data)

        result: dict[str, Any] = {
            "status": "success",
            "user_id": user_id,
            "timezone": timezone,
            "language": language,
            "domains": {},
            "extract_domain": extract_domain,
            "mode": "raw_data" if raw_data_override else "user_id",
        }

        if return_raw_data and not raw_data_override:
            serialized = json.dumps(raw_data, ensure_ascii=False, default=str)
            cleaned_raw_data = drop_empty_lists(
                clean_payload(
                    raw_data,
                    keep_top_level_list_keys=DEFAULT_KEEP_TOP_LEVEL_LIST_KEYS,
                ),
                keys=_DISPLAY_DROP_EMPTY_LIST_KEYS,
            )
            result["raw_data"] = _strip_zoneinfo(cleaned_raw_data)
            result["raw_data_size_bytes"] = len(serialized)

        for domain in valid_domains:
            logger.info(
                f"🐛 [debug_insight_pipeline] running domain={domain} user={user_id}"
            )
            domain_result = await insight_service.debug_pipeline(
                raw_data=raw_data,
                domain=domain,
                language=language,
                raw_data_override=bool(raw_data_override),
            )
            result["domains"][domain] = domain_result

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"debug_insight_pipeline failed for user_id {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"debug_insight_pipeline failed: {str(e)}",
        )


async def _fetch_external_api(
    client: httpx.AsyncClient,
    base_url: str,
    headers: dict,
    topic: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
) -> tuple:
    """Call a single external API endpoint. Returns (topic, path, data, error)."""
    try:
        url = f"{base_url}{path}"
        response = await client.get(url, headers=headers, params=params)
        if response.status_code == 200:
            return (topic, path, response.json(), None)
        if response.status_code == 404 and topic == "mood_latest":
            # Fallback: use /api/moods and take first item
            fallback_url = f"{base_url}/api/moods"
            fb_params = dict(params or {})
            fb_params["page"] = 1
            fb_params["size"] = 1
            fb_resp = await client.get(fallback_url, headers=headers, params=fb_params)
            if fb_resp.status_code == 200:
                fb_data = fb_resp.json()
                items = None
                if isinstance(fb_data, list):
                    items = fb_data
                elif isinstance(fb_data, dict):
                    items = fb_data.get("data") or fb_data.get("items") or fb_data.get("moods") or fb_data.get("results")
                if items:
                    return (topic, "/api/moods", items[0], None)
                return (topic, "/api/moods", None, "No mood data")
            return (topic, path, None, f"HTTP 404 (fallback failed)")
        return (topic, path, None, f"HTTP {response.status_code}")
    except Exception as e:
        return (topic, path, None, str(e))


def _build_raw_api_tasks(
    client: httpx.AsyncClient,
    base_url: str,
    headers: dict,
    user_id: str,
    resolved_tz: str,
    topics: set,
    provider_name: Optional[str] = None,
) -> list:
    """Build list of (topic, path, params) tuples for raw API responses."""
    from datetime import date, timedelta
    from modules.data_collector import (
        TOPIC_HEALTH, TOPIC_MOOD, TOPIC_CALENDAR, TOPIC_REMINDERS, TOPIC_BALANCE, TOPIC_PRODUCTIVITY
    )

    today = date.today()
    week_start = today - timedelta(days=today.weekday())
    today_str = today.isoformat()

    tasks = []

    # User profile
    tasks.append(("user_profile", f"/api/onboarding/users/{user_id}/profiles", None))

    # Health summaries (current week)
    if TOPIC_HEALTH in topics:
        tasks.append((
            "health_summaries",
            "/api/health/summaries",
            {"userId": user_id, "timezone": resolved_tz, "startDate": week_start.isoformat(), "endDate": today.isoformat()},
        ))

    # Mood latest
    if TOPIC_MOOD in topics:
        tasks.append((
            "mood_latest",
            "/api/moods/latest",
            {"userId": user_id, "timezone": resolved_tz},
        ))

    # Calendar events (today)
    if TOPIC_CALENDAR in topics:
        events_params = {"userId": user_id, "timezone": resolved_tz, "startDateTime": f"{today_str}T00:00:00", "endDateTime": f"{today_str}T23:59:59"}
        if provider_name:
            events_params["providerName"] = provider_name
        tasks.append(("calendar_events", "/api/calendar/events", events_params))

    # Reminders (today)
    if TOPIC_REMINDERS in topics:
        reminders_params = {"userId": user_id, "timezone": resolved_tz, "startDate": today_str, "endDate": today_str, "page": 1, "size": 50, "incompleteOnly": False}
        if provider_name:
            reminders_params["providerName"] = provider_name
        tasks.append(("reminders", "/api/calendar/reminders", reminders_params))

    # Balance scores (last 30 days)
    if TOPIC_BALANCE in topics:
        month_start = (today - timedelta(days=30)).isoformat()
        tasks.append((
            "balance_scores",
            "/api/balance",
            {"userId": user_id, "timezone": resolved_tz, "startDate": month_start, "endDate": today_str},
        ))

    # Daily snapshots (last 30 days)
    if TOPIC_PRODUCTIVITY in topics:
        month_start = (today - timedelta(days=30)).isoformat()
        tasks.append((
            "daily_snapshots",
            "/api/pipeline/daily-user-snapshots",
            {"userId": user_id, "startDate": month_start, "endDate": today_str, "page": 1, "size": 100},
        ))

    return tasks


@router.post("/raw_api_responses")
async def get_raw_api_responses(payload: dict = Body(...)):
    """Call each upstream API endpoint directly and return raw responses.

    This endpoint bypasses all processing/cleaning and returns the raw
    responses from each API endpoint for debugging purposes.
    """
    from modules.data_collector import resolve_collect_topics

    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    timezone = payload.get("timezone") or "UTC"
    domain = (payload.get("domain") or "health").strip().lower()
    provider_name = payload.get("provider_name") or ""
    if domain not in ("health", "productivity", "overall"):
        domain = "health"

    try:
        insight_service = ServiceFactory.get_insight_service()

        # Resolve timezone
        resolved_tz = await _resolve_user_timezone(
            user_id=user_id,
            payload_timezone=timezone,
            insight_service=insight_service,
        )

        # Get external API service
        external_api = insight_service.external_api_service
        if not external_api:
            raise HTTPException(status_code=500, detail="External API service not available")

        # Authenticate
        if not await external_api._ensure_authenticated():
            raise HTTPException(status_code=401, detail="Failed to authenticate with external API")

        base_url = external_api.base_url
        headers = external_api._get_headers()

        # Determine which topics to call
        topics = resolve_collect_topics(domain=domain)
        task_specs = _build_raw_api_tasks(
            client=None,  # not used directly; kept for signature compatibility
            base_url=base_url,
            headers=headers,
            user_id=user_id,
            resolved_tz=resolved_tz,
            topics=topics,
            provider_name=provider_name or None,
        )

        # Execute all API calls concurrently
        async with httpx.AsyncClient(timeout=30.0) as client:
            api_results = await asyncio.gather(
                *[_fetch_external_api(client, base_url, headers, topic, path, params)
                  for topic, path, params in task_specs],
                return_exceptions=True,
            )

        # Process results
        successful = []
        failed = []
        for result in api_results:
            if isinstance(result, Exception):
                failed.append({"error": str(result)})
                continue
            topic, path, data, error = result
            entry = {"topic": topic, "path": path, "success": error is None, "data": data}
            if error:
                entry["error"] = error
                failed.append(entry)
            else:
                successful.append(entry)

        return {
            "status": "success",
            "user_id": user_id,
            "timezone": resolved_tz,
            "domain": domain,
            "total_apis": len(task_specs),
            "successful": len(successful),
            "failed": len(failed),
            "apis": successful + failed,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"raw_api_responses failed for user_id {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"raw_api_responses failed: {str(e)}",
        )
