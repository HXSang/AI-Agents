import asyncio
import json
from datetime import date, datetime, time
from typing import Any, Dict, List, Optional, Union
from zoneinfo import ZoneInfo

from clients.rabbitmq_client import RabbitMQClient
from clients.redis_client import RedisClient
from fastapi import APIRouter, Request
from models.models import HealthCheck
from utils.logger import logger


def _strip_non_serializable(obj: Any) -> Any:
    """Remove non-JSON-serializable objects (ZoneInfo, date, datetime, time)
    from a nested dict/list so it can be JSON-serialized for the debugger response."""
    if isinstance(obj, dict):
        return {k: _strip_non_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_strip_non_serializable(item) for item in obj]
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, time):
        return obj.isoformat()
    if isinstance(obj, ZoneInfo):
        return str(obj)
    return obj


def _pydantic_dump_with_zoneinfo(model) -> dict:
    """Dump a Pydantic model to a JSON-safe dict, handling ZoneInfo."""
    raw = model.model_dump(mode="python")
    return _strip_non_serializable(raw)

router = APIRouter(tags=["health"])


@router.get("/", response_model=Dict[str, str])
async def root(request: Request):
    """Root endpoint"""
    # Custom access logging
    logger.info(
        f'{request.client.host} - "{request.method} {request.url.path} HTTP/1.1" 200 OK'
    )

    return {
        "message": "Chatbot Backend Service",
        "version": "1.0.0",
        "status": "running",
    }


@router.get("/health", response_model=HealthCheck)
async def health_check(request: Request):
    """Health check endpoint"""
    try:
        logger.log_api_request("GET", "/health")

        services = {}
        overall_status = "healthy"
        status_code = 200

        # Check RabbitMQ with timeout
        try:
            rabbitmq_client = RabbitMQClient()
            rabbitmq_healthy = await asyncio.wait_for(
                rabbitmq_client.health_check(), timeout=2.0
            )
            services["rabbitmq"] = "healthy" if rabbitmq_healthy else "unhealthy"
            if not rabbitmq_healthy:
                overall_status = "degraded"
        except asyncio.TimeoutError:
            services["rabbitmq"] = "unavailable"
            logger.warning("RabbitMQ health check timed out after 2 seconds")
        except Exception as e:
            services["rabbitmq"] = "unavailable"
            logger.warning(f"RabbitMQ health check failed: {str(e)}")

        # Check Redis with timeout
        try:
            redis_client = RedisClient()
            redis_healthy = await asyncio.wait_for(redis_client.ping(), timeout=2.0)
            services["redis"] = "healthy" if redis_healthy else "unhealthy"
            if not redis_healthy:
                overall_status = "degraded"
        except asyncio.TimeoutError:
            services["redis"] = "unavailable"
            logger.warning("Redis health check timed out after 2 seconds")
        except Exception as e:
            services["redis"] = "unavailable"
            logger.warning(f"Redis health check failed: {str(e)}")

        # If all services are unavailable, mark as unhealthy
        if all(status == "unavailable" for status in services.values()):
            overall_status = "unhealthy"
            status_code = 500

        # Custom access logging
        logger.info(
            f'{request.client.host} - "{request.method} {request.url.path} HTTP/1.1" {status_code} OK'
        )

        logger.log_api_response("GET", "/health", status_code)

        return HealthCheck(status=overall_status, services=services)

    except Exception as e:
        logger.error(f"Error in health_check: {str(e)}")
        # Custom access logging for error
        logger.info(
            f'{request.client.host} - "{request.method} {request.url.path} HTTP/1.1" 500 Internal Server Error'
        )
        return HealthCheck(status="unhealthy", services={"error": str(e)})


from fastapi import HTTPException
from pydantic import BaseModel


class DebugFetchRealDataRequest(BaseModel):
    user_id: str
    timezone: Optional[str] = None
    provider_name: str = ""
    language: str = "vi-VN"
    force_update: bool = False


@router.post("/debug_fetch_real_data")
async def debug_fetch_real_data(payload: DebugFetchRealDataRequest):
    """
    Debug endpoint: fetch real raw_data via DataCollector + PrepareX.
    Returns phase1 (ExtractedSignals) + phase1b (User Context Narrative).
    Dùng cho pipeline_debugger.html.
    """
    try:
        from services.service_factory import ServiceFactory

        service = ServiceFactory.get_insight_service()
        timezone = await service._data_collector.resolve_timezone(
            user_id=payload.user_id,
            payload_timezone=payload.timezone,
        )

        # Health extract → processor (week/month period aggregates)
        signals = await service._collect_and_process_signals(
            user_id=payload.user_id,
            timezone=timezone,
            provider_name=payload.provider_name,
            force_update=payload.force_update,
            domain="all",
        )

        user_narrative = await service._build_user_context_narrative(
            context=signals,
            user_id=payload.user_id,
            language=payload.language,
        )

        # health_params / time_data / calendar_metrics are injected into raw_data
        # by _collect_and_process before calling DataProcessor.extract_signals.
        # We retrieve them from the processed signals context.
        raw_data = getattr(signals, "raw_data", None) or {}

        return {
            "status": "success",
            "raw_data": _strip_non_serializable(raw_data),
            "phase1_extracted_signals": _pydantic_dump_with_zoneinfo(signals),
            "phase1b_user_narrative": user_narrative,
            "user_profile": signals.user_profile,
        }
    except Exception as e:
        logger.error(f"debug_fetch_real_data failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/random_insight")
async def random_insight_debug(
    user_id: str,
    domain: str = "health",
    timezone: Optional[str] = None,
    language: str = "vi-VN",
):
    """
    Debug endpoint: pick one random insight from Q&A pool (no auth required).
    Use this instead of /insight/random for pipeline_debugger.html.
    """
    try:
        from services.service_factory import ServiceFactory

        service = ServiceFactory.get_insight_service()
        resolved_tz = await service._data_collector.resolve_timezone(
            user_id=user_id,
            payload_timezone=timezone,
        )
        result = await service.get_random_insight(
            user_id=user_id,
            domain=domain,
            timezone=resolved_tz,
            language=language,
        )
        return result
    except Exception as e:
        logger.error(f"random_insight_debug failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


