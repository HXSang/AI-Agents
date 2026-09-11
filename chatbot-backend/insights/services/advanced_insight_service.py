import asyncio
import json
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from agents.llm_helper import llm_response_text
from agents.llm_manager import LLMManager
from clients.redis_client import RedisClient
from insights.health.extractor import HealthDataExtractor
from insights.health.health_processor.processor import HealthInsightProcessor
from insights.health.health_processor.context.processor import ContextSignalBuilder
from insights.helpers.insight_validator import _strip_mixed_language_tokens
from insights.helpers.insight_validator import validate_insight_against_health_params
from insights.helpers.question_gate import get_active_questions
from insights.helpers.question_gate import post_filter_insights
from insights.helpers.question_gate import score_insight_relevance
from insights.helpers.empty_value_cleanser import (
    DEFAULT_KEEP_TOP_LEVEL_LIST_KEYS,
    DISPLAY_DROP_EMPTY_LIST_KEYS as _DISPLAY_DROP_EMPTY_LIST_KEYS,
    clean_payload,
    drop_empty_lists,
)
from insights.prompts.context_narrative_templates import build_3part_narrative_parts
from insights.prompts.qa_prompts import DOMAIN_GROUP_REGISTRY
from insights.prompts.qa_prompts import build_sylo_qa_prompts
from insights.prompts.qa_prompts import parse_qa_response
from insights.schemas.insight_item import InsightItem
from insights.schemas.insight_pool import InsightPool as InsightPoolSchema
from insights.schemas.processed_context import ExtractedSignals
from insights.services.base import InsightService
from insights.services.insight_picker import _save_pool
from insights.services.insight_picker import clear_insight_history
from insights.services.insight_picker import pick_random_insight
from insights.services.insight_picker import pool_exists
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from modules.data_collector import DataCollector
from services.external_api_service import IExternalAPIService
from utils.logger import logger


def _zoneinfo_serializer(value: Any) -> Any:
    """Custom serializer for ZoneInfo, datetime, date — used in model_dump(mode="python")."""
    if isinstance(value, ZoneInfo):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value


def _strip_zoneinfo(obj: Any) -> Any:
    """Recursively strip ZoneInfo, date, datetime objects for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _strip_zoneinfo(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_strip_zoneinfo(item) for item in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, ZoneInfo):
        return str(obj)
    return obj


class AdvancedInsightService(InsightService):

    def __init__(self):
        llm_manager = LLMManager(provider="bedrock")
        self.llm = llm_manager.create_model(temperature=0.3)
        self.redis_client = RedisClient()
        self.external_api_service: Optional[IExternalAPIService] = None

        self._data_collector = DataCollector()
        self._user_locks: dict[str, asyncio.Lock] = {}
        logger.info("AdvancedInsightService (V2) initialized")

    def set_external_api_service(self, external_api_service: IExternalAPIService):
        self.external_api_service = external_api_service
        self._data_collector.external_api_service = external_api_service
        
    def _get_user_lock(self, user_id: str) -> asyncio.Lock:
        if user_id not in self._user_locks:
            self._user_locks[user_id] = asyncio.Lock()
        return self._user_locks[user_id]

    async def invalidate_qa_insight_cache(self, user_id: str) -> int:
        deleted = 0
        try:
            pattern = f"qa_insight:{user_id}:*"
            matching_keys = self.redis_client.redis.keys(pattern)
            if matching_keys:
                deleted += self.redis_client.redis.delete(*matching_keys)
                logger.info(
                    f"Invalidated {deleted} qa_insight cache(s) for user {user_id} "
                    f"(pattern: {pattern})"
                )
        except Exception as e:
            logger.warning(
                f"Failed to invalidate qa_insight cache for user {user_id}: {e}"
            )

        # Also clear signals cache so next request rebuilds with fresh data
        deleted += await self.invalidate_extracted_signals_cache(user_id)
        return int(deleted or 0)

    async def invalidate_extracted_signals_cache(self, user_id: str) -> int:
        deleted = 0
        try:
            pattern = f"extracted_signals:pipeline_v3:{user_id}:*"
            matching_keys = self.redis_client.redis.keys(pattern)
            if matching_keys:
                deleted = self.redis_client.redis.delete(*matching_keys)
                logger.info(
                    f"Invalidated {deleted} extracted_signals cache(s) for user "
                    f"{user_id} (pattern: {pattern})"
                )
            return int(deleted or 0)
        except Exception as e:
            logger.warning(
                f"Failed to invalidate extracted_signals cache for user "
                f"{user_id}: {e}"
            )
            return 0

    async def _collect_and_process_signals(
        self,
        user_id: str,
        timezone: Optional[str],
        provider_name: str,
        force_update: bool = False,
        domain: str = "health",
    ) -> ExtractedSignals:
        """Extract → process pipeline for insight domains.

        Fetch topics are filtered by ``domain`` (health / productivity / overall).
        """
        extractor = HealthDataExtractor(
            data_collector=self._data_collector,
            external_api_service=self.external_api_service,
        )
        raw_data = await extractor.extract(
            user_id=user_id,
            timezone=timezone,
            provider_name=provider_name,
            force_update=force_update,
            domain=domain,
        )
        return HealthInsightProcessor.process(raw_data)

    async def _get_or_create_extracted_signals(
        self,
        user_id: str,
        timezone: Optional[str],
        provider_name: str,
        force_update: bool,
        domain: str = "health",
    ) -> ExtractedSignals:
        sig_start = time.perf_counter()
        valid_domain = (
            domain if domain in ("health", "productivity", "overall") else "health"
        )
        sig_tag = f"user={user_id} domain={valid_domain}"
        logger.info(
            f"[TIMING][signals][start] {sig_tag} force_update={force_update}"
        )

        # Domain-scoped cache so health-only fetch is not reused as productivity
        cache_key = f"extracted_signals:pipeline_v3:{user_id}:{valid_domain}"
        cache_start = time.perf_counter()
        if not force_update:
            cached = await self.redis_client.get_data(cache_key)
            cache_elapsed = time.perf_counter() - cache_start
            if cached and "context" in cached:
                try:
                    result = ExtractedSignals.model_validate_json(cached["context"])
                    # Time-sensitive context (sleep_window / work_hours / weekend)
                    # must not ride a stale 30-minute health-signals cache.
                    if valid_domain in ("health", "overall"):
                        ContextSignalBuilder.refresh(result)
                    logger.info(
                        f"[TIMING][signals][cache_hit] {sig_tag} "
                        f"check_elapsed={cache_elapsed:.3f}s "
                        f"total_elapsed={time.perf_counter() - sig_start:.3f}s"
                    )
                    return result
                except Exception as e:
                    logger.warning(
                        f"Failed to parse cached ExtractedSignals for {user_id}: {e}"
                    )

        # force_update or cache miss → regenerate + clear Q&A pool + seen history
        if force_update:
            await clear_insight_history(user_id, self.redis_client)
            await self.invalidate_user_narrative_cache(user_id)
            logger.info(f"Cleared Q&A insight pool (force_update) for {user_id}")
        logger.info(
            f"[TIMING][signals][cache_miss] {sig_tag} "
            f"check_elapsed={time.perf_counter() - cache_start:.3f}s"
        )

        collect_start = time.perf_counter()
        ctx = await self._collect_and_process_signals(
            user_id, timezone, provider_name, force_update, domain=valid_domain
        )
        _log_phase_v3(
            "collect_and_process",
            sig_start,
            sig_tag,
            f"phase_elapsed={time.perf_counter() - collect_start:.3f}s",
        )

        # ZoneInfo in ctx.raw_data handled by custom_encoder below
        try:
            await self.redis_client.set_data(
                cache_key,
                {"context": json.dumps(ctx.model_dump(mode="python"), default=_zoneinfo_serializer)},
                expire=300,
            )
        except Exception as e:
            logger.warning(f"Failed to cache ExtractedSignals for {user_id}: {e}")
        logger.info(
            f"[TIMING][signals][done] {sig_tag} "
            f"total_elapsed={time.perf_counter() - sig_start:.3f}s"
        )
        return ctx

    async def get_qa_single_insight(
        self,
        user_id: str,
        domain: str = "health",
        timezone: Optional[str] = None,
        provider_name: str = "",
        language: str = "en-US",
        force_update: bool = False,
    ) -> Dict[str, Any]:
        method_start = time.perf_counter()
        user_language = language or "en-US"
        valid_domain = (
            domain if domain in ("health", "productivity", "overall") else "health"
        )
        user_tag = f"user={user_id} domain={valid_domain}"

        logger.info(
            f"[TIMING][qa_single][start] {user_tag} "
            f"lang={user_language} force_update={force_update}"
        )

        try:
            lock = self._get_user_lock(user_id)
            async with lock:
                # ── Step 0: Cache check ───────────────────────────────────────────
                s0_start = time.perf_counter()
                cache_key = f"qa_insight:{user_id}:{valid_domain}:{user_language}"
                if not force_update:
                    cached = await self.redis_client.get_data(cache_key)
                    logger.info(
                        f"[TIMING][qa_single][s0_cache_check] {user_tag} "
                        f"elapsed={time.perf_counter() - s0_start:.3f}s"
                    )
                    if cached and "insight" in cached:
                        cached_insight = cached.get("insight")
                        is_valid_cached = False
                        if isinstance(cached_insight, dict):
                            prose = cached_insight.get("insight")
                            is_valid_cached = isinstance(prose, str) and bool(
                                prose.strip()
                            )
                        elif isinstance(cached_insight, str):
                            is_valid_cached = bool(cached_insight.strip())

                        if is_valid_cached:
                            logger.info(
                                f"[TIMING][qa_single][cache_hit] {user_tag} "
                                f"elapsed={time.perf_counter() - method_start:.3f}s"
                            )
                            return {
                                "status": "success",
                                "user_id": user_id,
                                "insight": cached_insight,
                                "error": None,
                            }

                        logger.warning(
                            f"[TIMING][qa_single][cache_stale] {user_tag} "
                            "invalid cached payload; bypassing cache"
                        )

                # ── Layer 1–2: Extract → Signals (cached) ─────────────────────
                s1_start = time.perf_counter()
                ctx = await self._get_or_create_extracted_signals(
                    user_id,
                    timezone,
                    provider_name,
                    force_update=force_update,
                    domain=valid_domain,
                )
                logger.info(
                    f"[TIMING][qa_single][s1_signals] {user_tag} "
                    f"elapsed={time.perf_counter() - s1_start:.3f}s"
                )

                # ── Layers 3–4: Prompt (narrative inline) → LLM ───────────────
                async def run_single_domain(
                    d: str,
                    *,
                    domain_ctx: Optional[ExtractedSignals] = None,
                ) -> Optional[dict]:
                    """Assemble prompt + one Sylo Q&A call → final prose."""
                    step_start = time.perf_counter()
                    step_tag = f"{user_tag} domain={d}"
                    local_ctx = domain_ctx or ctx

                    s2_start = time.perf_counter()
                    assembled = self._assemble_sylo_prompt(
                        local_ctx, d, user_language, user_id=user_id
                    )
                    if not assembled:
                        logger.info(f"📭 No overview prompt for {d}")
                        return None
                    chosen_group = assembled["group"]
                    chosen_q = assembled["question"]
                    logger.info(
                        f"[TIMING][qa_single][{d}][s2_prompt] "
                        f"{step_tag} group={chosen_group.get('group_id')} "
                        f"q={chosen_q.get('id')} "
                        f"elapsed={time.perf_counter() - s2_start:.3f}s"
                    )

                    s3_start = time.perf_counter()
                    try:
                        response = await self.llm.ainvoke(assembled["messages"])
                        raw = (llm_response_text(response) or "").strip()
                        parsed = parse_qa_response(raw)
                    except Exception as e:
                        logger.warning(f"Q&A LLM failed for {d}/{chosen_q['id']}: {e}")
                        parsed = []

                    if not parsed:
                        return None

                    qa_result = parsed[0]
                    prose = (qa_result.get("prose") or "").strip()
                    if not prose:
                        logger.warning(f"[{d}] empty Sylo insight for {step_tag}")
                        return None
                    logger.info(
                        f"[TIMING][qa_single][{d}][s3_llm] "
                        f"{step_tag} elapsed={time.perf_counter() - s3_start:.3f}s"
                    )

                    self._log_hallucination_guard(
                        prose,
                        local_ctx,
                        user_language,
                        step_tag=step_tag,
                        domain=d,
                        question_id=chosen_q.get("id", "?"),
                    )

                    # Language consistency: sanitize prose in place.
                    if user_language and user_language.lower().startswith("vi"):
                        cleaned, replacements = _strip_mixed_language_tokens(
                            prose, user_language
                        )
                        if replacements:
                            logger.info(
                                f"[LANG_CONSISTENCY][single] sanitized "
                                f"{len(replacements)} token(s) domain={d}"
                            )
                            prose = cleaned
                            qa_result["prose"] = cleaned

                    logger.info(
                        f"[TIMING][qa_single][{d}][total] "
                        f"{step_tag} total={time.perf_counter() - step_start:.3f}s"
                    )
                    logger.info(
                        f"QA single: {d} → group={chosen_group['group_id']} "
                        f"→ question={chosen_q['id']}"
                    )
                    return {
                        "insight": prose,
                        "question_id": chosen_q["id"],
                        "question_text": chosen_q["text"],
                        "domain": d,
                        "action_family": qa_result.get("action_family", ""),
                        "tone": qa_result.get("tone", ""),
                        "confidence": qa_result.get("confidence", 0.7),
                    }

                primary_insight = await run_single_domain(valid_domain)

                if primary_insight:
                    try:
                        await self.redis_client.set_data(
                            cache_key,
                            {"insight": primary_insight},
                            expire=300,
                        )
                    except Exception as e:
                        logger.warning(
                            f"[TIMING][qa_single][cache_write_failed] {user_tag}: {e}"
                        )
                    logger.info(
                        f"[TIMING][qa_single][done] {user_tag} "
                        f"total={time.perf_counter() - method_start:.3f}s"
                    )

                    return {
                        "status": "success",
                        "user_id": user_id,
                        "insight": primary_insight,
                        "error": None,
                    }
                else:
                    return {
                        "status": "error",
                        "user_id": user_id,
                        "insight": None,
                        "error": f"No {valid_domain} insight generated",
                    }

        except Exception as e:
            logger.exception(f"get_qa_single_insight error for {user_id}: {e}")
            return {
                "status": "error",
                "user_id": user_id,
                "insight": None,
                "error": str(e),
            }

    async def get_random_insight(
        self,
        user_id: str,
        domain: str = "health",
        timezone: Optional[str] = None,
        provider_name: str = "",
        language: str = "en-US",
    ) -> Dict[str, Any]:

        try:
            user_language = language or "en-US"
            valid_domain = (
                domain if domain in ("health", "productivity", "overall") else "health"
            )
            rand_start = time.perf_counter()
            rand_tag = f"user={user_id} domain={valid_domain}"
            logger.info(
                f"[TIMING][random_v3][start] {rand_tag} lang={user_language}"
            )

            # 1. Ensure signals exist
            ctx = await self._get_or_create_extracted_signals(
                user_id,
                timezone,
                provider_name,
                force_update=False,
                domain=valid_domain,
            )
            _log_phase_v3("random_signals", rand_start, rand_tag)

            # 2. Ensure narrative exists
            narrative = await self._build_user_context_narrative(
                ctx, user_id, language=user_language, domain=valid_domain
            )
            _log_phase_v3("random_narrative", rand_start, rand_tag)

            # 3. Ensure pool exists
            pool_check_start = time.perf_counter()
            pool_exists_now = await pool_exists(user_id, self.redis_client)
            pool_check_elapsed = time.perf_counter() - pool_check_start
            if not pool_exists_now:
                logger.info(f"📦 No pool found for {user_id} — building Q&A pool...")
                pool_build_start = time.perf_counter()
                await self._build_qa_insight_pool(
                    user_id=user_id,
                    ctx=ctx,
                    user_narrative=narrative,
                    language=user_language,
                )
                _log_phase_v3(
                    "random_pool_build",
                    rand_start,
                    rand_tag,
                    f"check_elapsed={pool_check_elapsed:.3f}s "
                    f"build_elapsed={time.perf_counter() - pool_build_start:.3f}s",
                )
            else:
                _log_phase_v3(
                    "random_pool_hit",
                    rand_start,
                    rand_tag,
                    f"check_elapsed={pool_check_elapsed:.3f}s",
                )

            # 4. Pick one insight
            pick_start = time.perf_counter()
            chosen = await pick_random_insight(
                user_id=user_id,
                domain=valid_domain,
                redis=self.redis_client,
            )
            _log_phase_v3(
                "random_pick",
                rand_start,
                rand_tag,
                f"elapsed={time.perf_counter() - pick_start:.3f}s",
            )

            if not chosen:
                logger.info(
                    f"[TIMING][random_v3][no_insight] {rand_tag} "
                    f"total_elapsed={time.perf_counter() - rand_start:.3f}s"
                )
                return {
                    "status": "error",
                    "user_id": user_id,
                    "insight": None,
                    "error": f"No {valid_domain} insights available",
                }

            # 5. Pool items already bake the Sylo title+insight into current_state
            final_prose = (chosen.current_state or "").strip()

            # 6. Return in standard format
            result = {
                "status": "success",
                "user_id": user_id,
                "insight": {
                    "point1": final_prose,
                    "point2": "",
                    "insight_id": chosen.insight_id,
                    "question_id": chosen.question_id,
                    "domain": chosen.domain,
                    "action_family": chosen.action_family,
                    "tone": chosen.tone,
                    "confidence": chosen.confidence,
                    # Raw fields from InsightItem (useful for debugging / structured display)
                    "current_state": chosen.current_state,
                    "cause": chosen.cause,
                    "action": chosen.action,
                    "evidence": chosen.evidence,
                    "expected_outcome": chosen.expected_outcome,
                },
                "error": None,
            }

            logger.info(
                f"[TIMING][random_v3][done] {rand_tag} "
                f"total_elapsed={time.perf_counter() - rand_start:.3f}s"
            )
            return result

        except Exception as e:
            logger.exception(f"get_random_insight error for {user_id}: {e}")
            try:
                logger.error(
                    f"[TIMING][random_v3][error] user={user_id} "
                    f"error={str(e)[:120]}"
                )
            except Exception:
                pass
            return {
                "status": "error",
                "user_id": user_id,
                "insight": None,
                "error": str(e),
            }

    # ──────────────────────────────────────────────────────────────────────────
    # Public interface — mirrors InsightService / InsightServiceImpl signatures
    # ──────────────────────────────────────────────────────────────────────────

    async def _build_user_context_narrative(
        self,
        context: ExtractedSignals,
        user_id: str,
        language: str = "en",
        domain: str = "overall",
    ) -> str:
        """Template digest used inside prompt assembly (not a separate pipeline phase)."""
        return self._context_narrative(context, user_id, language, domain)

    def _context_narrative(
        self,
        context: ExtractedSignals,
        user_id: str,
        language: str = "en",
        domain: str = "overall",
    ) -> str:
        t_start = time.perf_counter()
        bg_text, current_text, future_text = build_3part_narrative_parts(
            context, language, domain=domain
        )
        if not getattr(context, "user_profile", None):
            logger.warning(
                f"⚠️ Narrative built without identity profile for {user_id}"
            )
        narrative = f"{bg_text}\n\n{current_text}\n\n{future_text}"
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            f"Context narrative built for {user_id} in {elapsed_ms:.1f}ms"
        )
        return narrative

    def _assemble_sylo_prompt(
        self,
        ctx: ExtractedSignals,
        domain: str,
        language: str,
        *,
        user_id: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Layer 3: narrative + fixed overview → LLM messages.

        Returns ``{narrative, group, question, messages}`` or None.
        """
        narrative = self._context_narrative(
            ctx, user_id or "unknown", language, domain=domain
        )
        time_phase = ctx.meta.time_phase if ctx.meta else "afternoon"
        meta_dict = ctx.meta.model_dump() if ctx.meta else {}
        # Strip ZoneInfo before JSON serialization
        if ctx.raw_data is not None:
            ctx.raw_data = _strip_zoneinfo(ctx.raw_data)
        bundle = build_sylo_qa_prompts(
            domain,
            context_json=json.dumps(ctx.model_dump(mode="python"), default=_zoneinfo_serializer),
            narrative=narrative,
            language=language,
            time_phase=time_phase,
            active_hours_end_time=meta_dict.get("active_hours_end_time"),
        )
        if not bundle:
            return None
        return {
            "narrative": narrative,
            "group": bundle["group"],
            "question": bundle["question"],
            "messages": [
                SystemMessage(content=bundle["system"]),
                HumanMessage(content=bundle["user"]),
            ],
        }

    def _log_hallucination_guard(
        self,
        prose: str,
        ctx: ExtractedSignals,
        language: str,
        *,
        step_tag: str,
        domain: str,
        question_id: str,
    ) -> None:
        try:
            hp = (ctx.raw_data or {}).get("health_params") or {}
            meta_dict = ctx.meta.model_dump() if ctx.meta else None
            if meta_dict:
                hr_staleness = meta_dict.get("hr_data_staleness_days")
                if hr_staleness is not None and hr_staleness >= 1:
                    hp = dict(hp)
                    hp["resting_heart_rate"] = None
                    hp["latest_heart_rate"] = None
                    hp["hrv_score"] = None
                sleep_staleness = meta_dict.get("sleep_data_staleness_days")
                if sleep_staleness is not None and sleep_staleness >= 1:
                    hp = hp if isinstance(hp, dict) else dict(hp)
                    hp["sleep_lastnight"] = None
                    hp["sleep_quality_score"] = None
                steps_staleness = meta_dict.get("steps_data_staleness_days")
                if steps_staleness is not None and steps_staleness >= 1:
                    hp = hp if isinstance(hp, dict) else dict(hp)
                    hp["steps_today"] = None
            pp = (
                ctx.productivity_signals.model_dump()
                if ctx.productivity_signals
                else None
            )
            ci = (
                ctx.calendar_intelligence.model_dump()
                if ctx.calendar_intelligence
                else None
            )
            validation = validate_insight_against_health_params(
                prose,
                hp,
                meta=meta_dict,
                language=language,
                productivity_params=pp,
                calendar_intelligence=ci,
            )
            if not validation["ok"]:
                logger.warning(
                    f"[HALLUCINATION_GUARD] {step_tag} "
                    f"domain={domain} q={question_id} "
                    f"suspicious_values={validation['suspicious'][:3]} "
                    f"date_violations={validation.get('date_violations', [])[:3]} "
                    f"language_violations={validation.get('language_violations', [])[:3]} "
                    f"duration_violations="
                    f"{validation.get('pattern_duration_violations', [])[:3]} "
                    f"contradiction_violations="
                    f"{validation.get('contradiction_violations', [])[:3]} "
                    f"advice={validation['advice']}"
                )
        except Exception as _ve:
            logger.debug(f"insight_validator skipped: {_ve}")

    # ─────────────────────────────────────────────────────────────────
    # Language consistency enforcement (4 layers)
    # ─────────────────────────────────────────────────────────────────
    # Goal: NO mixed-language insight. If the user language is vi-VN, every
    # user-facing prose field must be 100% Vietnamese (calendar event titles
    # in double quotes are the only allowed English exception).
    #
    # Layer 1 — Prompt rule added in core_persona.py.
    # Layer 2 — Post-generation validator runs on every item.
    # Layer 3 — Hard sanitizer replaces any remaining English tokens we have a
    #           Vietnamese mapping for.
    # Layer 4 — If still not clean, downgrade confidence to 0.0 so downstream
    #           filter drops the item (better to skip than ship mixed text).

    _LANG_ENFORCE_FIELDS = (
        "current_state",
        "evidence",
        "cause",
        "action",
        "expected_outcome",
    )

    def _enforce_language_consistency(
        self,
        insight: "InsightItem",
        language: str,
        raw_item: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mutate an InsightItem in place to remove any English leak.

        Only runs when language starts with "vi" (Vietnamese). For English or
        other languages, this is a no-op — the prompt already enforced it.

        Per user-facing field (current_state, evidence, cause, action,
        expected_outcome):
          1. Compose full prose from these fields.
          2. Run language consistency check.
          3. If violations exist, run the hard sanitizer immediately.
          4. Re-check; if still dirty, downgrade confidence to 0.0 to drop.
        """
        if not language or not language.lower().startswith("vi"):
            return

        fields = {
            "current_state": insight.current_state or "",
            "evidence": insight.evidence or "",
            "cause": insight.cause or "",
            "action": insight.action or "",
            "expected_outcome": insight.expected_outcome or "",
        }
        full_prose = "\n".join(v for v in fields.values() if v).strip()
        if not full_prose:
            return

        from insights.helpers.insight_validator import (
            _check_language_consistency,
        )

        violations = _check_language_consistency(full_prose, language)
        if not violations:
            return  # Already clean.

        sample = [v["token"] for v in violations[:8]]
        logger.warning(
            f"[LANG_CONSISTENCY] user-language={language} "
            f"domain={insight.domain} q={insight.question_id} "
            f"violations={len(violations)} sample={sample}"
        )

        # Layer 3 — hard sanitizer.
        total_replacements = 0
        for key in self._LANG_ENFORCE_FIELDS:
            original = fields[key]
            if not original:
                continue
            cleaned, replacements = _strip_mixed_language_tokens(original, language)
            if replacements:
                total_replacements += len(replacements)
                setattr(insight, key, cleaned)
                fields[key] = cleaned

        # Re-check after sanitization.
        full_prose = "\n".join(v for v in fields.values() if v).strip()
        remaining = (
            _check_language_consistency(full_prose, language) if full_prose else []
        )

        if total_replacements:
            logger.info(
                f"[LANG_CONSISTENCY] sanitized {total_replacements} token(s) "
                f"domain={insight.domain} q={insight.question_id}"
            )

        if remaining:
            # Layer 4 — still dirty. Downgrade to drop.
            logger.warning(
                f"[LANG_CONSISTENCY] {len(remaining)} violation(s) remain after "
                f"sanitization. Downgrading confidence to 0.0 to drop item. "
                f"remaining={[v['token'] for v in remaining[:5]]} "
                f"domain={insight.domain} q={insight.question_id}"
            )
            insight.confidence = 0.0

    async def invalidate_user_narrative_cache(self, user_id: str) -> int:
        """Best-effort cleanup of legacy narrative-bg Redis keys (no longer written)."""
        try:
            pattern = f"v2_narrative_bg:{user_id}:*"
            matching = self.redis_client.redis.keys(pattern)
            if not matching:
                return 0
            deleted = self.redis_client.redis.delete(*matching)
            logger.info(
                f"🧹 Invalidated {deleted} orphan narrative-bg cache(s) for {user_id}"
            )
            return deleted
        except Exception as e:
            logger.warning(
                f"⚠️ invalidate_user_narrative_cache failed for {user_id}: {e}"
            )
            return 0

    async def _build_qa_insight_pool(
        self,
        user_id: str,
        ctx: ExtractedSignals,
        user_narrative: str,
        language: str = "en-US",
    ) -> InsightPoolSchema:

        time_phase = ctx.meta.time_phase if ctx.meta else "afternoon"
        meta_dict = ctx.meta.model_dump() if ctx.meta else {}
        context_json = json.dumps(ctx.model_dump(mode="python"), default=_zoneinfo_serializer)

        all_insights_by_domain: dict[str, list[InsightItem]] = {
            "health": [],
            "productivity": [],
            "overall": [],
        }

        for domain, groups in DOMAIN_GROUP_REGISTRY.items():
            # Align pool questions with Sylo overview IDs in qa_prompts groups.
            active_qs = get_active_questions(domain, time_phase, meta_dict)
            active_q_ids = {q.id for q in active_qs}
            q_id_to_text = {
                q["id"]: q.get("text", "")
                for g in groups
                for q in g.get("questions", [])
            }

            async def run_group(group: dict, d: str = domain):
                active_in_group = [
                    q for q in group["questions"] if q["id"] in active_q_ids
                ]
                if not active_in_group:
                    active_in_group = list(group.get("questions") or [])
                if not active_in_group:
                    return []

                bundle = build_sylo_qa_prompts(
                    d,
                    context_json=context_json,
                    narrative=user_narrative,
                    language=language,
                    time_phase=time_phase,
                    active_hours_end_time=meta_dict.get("active_hours_end_time"),
                )
                if not bundle:
                    return []
                # Prefer active subset when filtering (pool path)
                user_prompt = bundle["user"]
                if active_in_group != (group.get("questions") or []):
                    from insights.prompts.qa_prompts import _USER_PROMPT_BUILDERS

                    user_fn = _USER_PROMPT_BUILDERS.get(d)
                    if user_fn:
                        user_prompt = user_fn(
                            context_json=context_json,
                            narrative=user_narrative,
                            group={**group, "questions": active_in_group},
                            language=language,
                        )

                messages = [
                    SystemMessage(content=bundle["system"]),
                    HumanMessage(content=user_prompt),
                ]
                try:
                    response = await self.llm.ainvoke(messages)
                    raw = (llm_response_text(response) or "").strip()
                    parsed = parse_qa_response(raw)
                    for item in parsed:
                        qid = item.get("question_id", "")
                        item["question_text"] = q_id_to_text.get(qid, "")
                    return parsed
                except Exception as e:
                    logger.warning(f"Q&A LLM failed for {d}/{group['group_id']}: {e}")
                    return []

            group_tasks = [run_group(g) for g in groups]
            group_results = await asyncio.gather(*group_tasks)

            domain_insights: list[InsightItem] = []
            flat_items: list[dict] = []
            for group_result in group_results:
                flat_items.extend(group_result)

            for item in flat_items:
                try:
                    current_state = (
                        item.get("prose")
                        or item.get("current_state", "")
                        or ""
                    )
                    insight = InsightItem(
                        insight_id=uuid.uuid4().hex,
                        domain=domain,
                        question_id=item.get("question_id", ""),
                        question_text=item.get("question_text", ""),
                        metrics=item.get("metrics", []),
                        current_state=current_state,
                        cause=item.get("cause", ""),
                        action=item.get("action", "") or "",
                        expected_outcome=item.get("expected_outcome", "") or "",
                        evidence=item.get("evidence", "") or "",
                        action_family=item.get("action_family", "observation")
                        or "observation",
                        tone=item.get("tone", "informative") or "informative",
                        confidence=float(item.get("confidence", 0.7)),
                        created_at=datetime.now(timezone.utc),
                    )
                    # ── Language consistency enforcement (4 layers) ──
                    self._enforce_language_consistency(insight, language, item)
                    domain_insights.append(insight)
                except Exception as e:
                    logger.warning(f"Failed to parse Q&A item: {e}")

            filtered = post_filter_insights(domain_insights, time_phase)
            filtered.sort(
                key=lambda x: score_insight_relevance(x, time_phase), reverse=True
            )
            all_insights_by_domain[domain] = filtered

            logger.info(
                f"Q&A {domain}: {len(domain_insights)} raw → {len(filtered)} filtered"
            )

        # ── Build and save InsightPool ──────────────────────────────────────
        now = datetime.now(timezone.utc)
        pool = InsightPoolSchema(
            user_id=user_id,
            generated_at=now,
            pool_expire_at=now,  # placeholder; override below
            health_insights=all_insights_by_domain["health"],
            productivity_insights=all_insights_by_domain["productivity"],
            overall_insights=all_insights_by_domain["overall"],
        )
        # Set expire_at 30 minutes from now
        pool.pool_expire_at = now + timedelta(seconds=1800)

        await _save_pool(self.redis_client, user_id, pool)
        logger.info(
            f"💾 InsightPool saved for {user_id}: "
            f"h={len(pool.health_insights)} "
            f"p={len(pool.productivity_insights)} "
            f"o={len(pool.overall_insights)}"
        )
        return pool

    async def analyze_health_insight(
        self,
        user_id: str,
        timezone: Optional[str] = None,
        provider_name: Optional[str] = "",
        language: Optional[str] = "en-US",
        force_update: bool = False,
    ) -> Dict[str, Any]:
        return await self.get_qa_single_insight(
            user_id=user_id,
            domain="health",
            timezone=timezone,
            provider_name=provider_name,
            language=language or "en-US",
            force_update=force_update,
        )

    async def analyze_productivity(
        self,
        user_id: str,
        timezone: Optional[str] = None,
        provider_name: Optional[str] = "",
        language: Optional[str] = "en-US",
        force_update: bool = False,
    ) -> Dict[str, Any]:
        return await self.get_qa_single_insight(
            user_id=user_id,
            domain="productivity",
            timezone=timezone,
            provider_name=provider_name,
            language=language or "en-US",
            force_update=force_update,
        )

    async def analyze_productivity_insight(
        self,
        user_id: str,
        timezone: Optional[str] = None,
        provider_name: Optional[str] = "",
        language: Optional[str] = "en-US",
        force_update: bool = False,
    ) -> Dict[str, Any]:
        return await self.analyze_productivity(
            user_id=user_id,
            timezone=timezone,
            provider_name=provider_name,
            language=language,
            force_update=force_update,
        )

    async def analyze_overall_insight(
        self,
        user_id: str,
        timezone: Optional[str] = None,
        provider_name: Optional[str] = "",
        language: Optional[str] = "en-US",
        force_update: bool = False,
    ) -> Dict[str, Any]:
        # Return Q&A payload as-is; router wraps prose into
        # {great_job, need_attention, opportunity}. Do NOT pre-format here —
        # that drops the nested "insight" prose when the router calls _extract_prose.
        return await self.get_qa_single_insight(
            user_id=user_id,
            domain="overall",
            timezone=timezone,
            provider_name=provider_name,
            language=language or "en-US",
            force_update=force_update,
        )

    async def debug_pipeline(
        self,
        raw_data: dict,
        domain: str,
        language: str,
        raw_data_override: bool = False,
    ) -> dict:
        """Mirror production insight steps for the demo UI.

        Four layers (same as production):
        1. Extract  2. Signals  3. Prompt (narrative + fixed overview)  4. LLM.

        Legacy s3–s9 / phase* keys are filled as aliases for older demo clients.
        """
        try:
            from modules.data_collector import resolve_collect_topics
            from modules.data_collector import build_extract_api_fetches
            from modules.data_collector import TOPIC_SNAPSHOTS

            logger.info(f"Running debug_pipeline domain={domain}")
            response: dict[str, Any] = {
                "status": "success",
                "error": None,
                "domain": domain,
            }
            timings: dict[str, float] = {}
            pipeline_start = time.perf_counter()

            def _mark(key: str, started: float) -> float:
                elapsed = time.perf_counter() - started
                timings[key] = round(elapsed, 4)
                logger.info(
                    f"[TIMING][debug][{domain}][{key}] elapsed={elapsed:.3f}s"
                )
                return time.perf_counter()

            # Strip ZoneInfo recursively from raw_data before any serialization
            raw_data = _strip_zoneinfo(raw_data)

            # ── 1 · Extract (debug: raw_data is extract output) ───────────────
            t = time.perf_counter()
            topics = sorted(resolve_collect_topics(domain=domain))
            # Mirror HealthDataExtractor: short snapshots skipped (extended fetch instead)
            topics = [t for t in topics if t != TOPIC_SNAPSHOTS]
            # Capture raw_data BEFORE projection so s1_extract shows canonical names
            # from canonicalize_extracted_data (extractor's last step).
            extract_raw = _strip_zoneinfo(
                json.loads(json.dumps(raw_data, ensure_ascii=False, default=str))
            )
            extract_raw = clean_payload(
                extract_raw,
                keep_top_level_list_keys=DEFAULT_KEEP_TOP_LEVEL_LIST_KEYS,
            )
            extract_raw = drop_empty_lists(
                extract_raw, keys=_DISPLAY_DROP_EMPTY_LIST_KEYS
            )
            # Mirror extract normalize before processor (mock/pasted payloads).
            # Only run when raw_data is bypassing HealthDataExtractor (pasted/mock
            # payload) — when extractor.extract() already ran, the data is
            # canonicalized and re-running keep_map wipes balance / other fields
            # whose keys have been renamed to canonical_*** forms.
            if domain == "health" and raw_data_override:
                from insights.health.extractor import (
                    project_health_extract_for_processor,
                )

                project_health_extract_for_processor(raw_data)
            snaps = raw_data.get("historical_snapshots")
            events = raw_data.get("calendar_events")
            health_stats = raw_data.get("today_health_stats")
            health_params = raw_data.get("health_params")
            mood = raw_data.get("latest_mood")
            balance_today = raw_data.get("balance_score")
            balance_week = raw_data.get("balance_scores_30d")
            # Always rebuild after projection so HTML api_fetches.response
            # reflects field_keep_map (never reuse stale pre-strip payloads).
            api_fetches = build_extract_api_fetches(
                raw_data,
                domain=domain,
                user_id=str(raw_data.get("user_id") or ""),
            )
            api_fetches = _strip_zoneinfo(api_fetches)
            raw_data["_extract_api_fetches"] = api_fetches
            # Avoid nesting api_fetches twice inside the merged raw_data blob
            if isinstance(extract_raw, dict):
                extract_raw = {
                    k: v
                    for k, v in extract_raw.items()
                    if k != "_extract_api_fetches"
                }
            from insights.health.field_keep_maps import FIELD_KEEP_MAP

            response["s1_extract"] = {
                "domain": domain,
                "collect_topics": topics,
                "raw_keys": sorted(str(k) for k in raw_data.keys()),
                "raw_data": extract_raw,
                "api_fetches": api_fetches,
                "field_keep_map": FIELD_KEEP_MAP,
                "fields": {
                    "health_params": {
                        "present": "health_params" in raw_data,
                        "count": (
                            len(health_params)
                            if isinstance(health_params, dict)
                            else 0
                        ),
                    },
                    "historical_snapshots": {
                        "present": "historical_snapshots" in raw_data,
                        "count": len(snaps) if isinstance(snaps, list) else 0,
                    },
                    "today_health_stats": {
                        "present": "today_health_stats" in raw_data,
                        "count": (
                            len(health_stats)
                            if isinstance(health_stats, dict)
                            else 0
                        ),
                    },
                    "balance_score": {
                        "present": balance_today is not None,
                        "count": (
                            len(balance_today)
                            if isinstance(balance_today, dict)
                            else (1 if balance_today is not None else 0)
                        ),
                    },
                    "balance_scores_30d": {
                        "present": "balance_scores_30d" in raw_data,
                        "count": (
                            len(balance_week) if isinstance(balance_week, list) else 0
                        ),
                    },
                    "calendar_events": {
                        "present": "calendar_events" in raw_data,
                        "count": len(events) if isinstance(events, list) else 0,
                    },
                    "mood": {
                        "present": mood is not None,
                        "count": 1 if mood is not None else 0,
                    },
                    "time_data": {
                        "present": "time_data" in raw_data,
                        "count": (
                            len(raw_data["time_data"])
                            if isinstance(raw_data.get("time_data"), dict)
                            else 0
                        ),
                    },
                },
                "has_health_params": "health_params" in raw_data,
                "has_historical_snapshots": "historical_snapshots" in raw_data,
                "has_today_health_stats": "today_health_stats" in raw_data,
                "has_balance_score": balance_today is not None,
                "has_calendar_events": "calendar_events" in raw_data,
                "has_mood": mood is not None,
                "note": (
                    "api_fetches lists each upstream GET used by HealthDataExtractor. "
                    "Health: TWO GET /api/health/summaries "
                    "(today_health_stats + health_params). "
                    "After extract, field_keep_map projects payloads: "
                    "balance → date/timezone/healthScore; "
                    "calendar/event → summary/completed/startTime/endTime/eventType; "
                    "health/summaries → ENERGY|HR|SLEEP|STEPS without id."
                ),
            }
            t = _mark("s1_extract", t)

            # ── 2 · Signals ──────────────────────────────────────────────────
            extracted_signals = HealthInsightProcessor.process(raw_data)
            processor_dump = _strip_zoneinfo(
                extracted_signals.model_dump(mode="python")
            )
            processor_dump = clean_payload(
                processor_dump,
                keep_top_level_list_keys=DEFAULT_KEEP_TOP_LEVEL_LIST_KEYS,
            )
            processor_dump = drop_empty_lists(
                processor_dump, keys=_DISPLAY_DROP_EMPTY_LIST_KEYS
            )
            response["s2_signals"] = processor_dump
            response["s2_processor"] = processor_dump  # legacy alias
            t = _mark("s2_signals", t)
            timings["s2_processor"] = timings["s2_signals"]

            user_id: str = str(raw_data.get("user_id") or "debug_user")

            # ── 3 · Prompt (narrative + fixed overview, not separate phases) ─
            assembled = self._assemble_sylo_prompt(
                extracted_signals, domain, language, user_id=user_id
            )
            if not assembled:
                timings["total"] = round(time.perf_counter() - pipeline_start, 4)
                response["timings"] = timings
                return {
                    "status": "error",
                    "error": "No overview prompt available",
                    "timings": timings,
                }

            chosen_group = assembled["group"]
            chosen_q = assembled["question"]
            user_narrative = assembled["narrative"]
            group_info = {
                "group_id": chosen_group.get("group_id"),
                "focus": chosen_group.get("focus"),
                "group_name": chosen_group.get("group_name"),
                "domain": domain,
                "selection": "fixed_overview",
            }
            response["s3_prompt"] = {
                "narrative": user_narrative,
                "group": group_info,
                "question": chosen_q,
            }
            t = _mark("s3_prompt", t)

            # ── 4 · LLM (parse folds compose → prose) ────────────────────────
            answer_raw = await self.llm.ainvoke(assembled["messages"])
            parsed = parse_qa_response((llm_response_text(answer_raw) or "").strip())
            t = _mark("s4_llm", t)
            if not parsed:
                timings["total"] = round(time.perf_counter() - pipeline_start, 4)
                response["timings"] = timings
                return {
                    "status": "error",
                    "error": "Failed to parse Q&A response",
                    "timings": timings,
                }
            answer = parsed[0]
            final_prose = (answer.get("prose") or "").strip()
            if not final_prose:
                timings["total"] = round(time.perf_counter() - pipeline_start, 4)
                response["timings"] = timings
                return {
                    "status": "error",
                    "error": f"Empty {domain} Sylo insight",
                    "timings": timings,
                }
            response["s4_llm"] = {**answer, "prose": final_prose}

            timings["total"] = round(time.perf_counter() - pipeline_start, 4)
            response["timings"] = timings
            logger.info(
                f"[TIMING][debug][{domain}][total] elapsed={timings['total']:.3f}s "
                f"breakdown={timings}"
            )

            # Legacy keys (older FE / logs) — map 4-layer → old 9-step shape
            response["s3_narrative"] = user_narrative
            response["s4_group"] = group_info
            response["s5_question"] = chosen_q
            response["s6_qa"] = answer
            skip_note = {
                "skipped": True,
                "reason": (
                    f"{domain.capitalize()} Sylo Q&A: title + insight (80–180 words); "
                    "no advice/render/polish"
                ),
            }
            response["s7_advice"] = {**skip_note, "action": "", "expected_outcome": ""}
            response["s8_render"] = final_prose
            response["s9_polish"] = final_prose
            timings["s3_narrative"] = timings.get("s3_prompt", 0.0)
            timings["s4_group"] = 0.0
            timings["s5_question"] = 0.0
            timings["s6_qa"] = timings.get("s4_llm", 0.0)
            timings["s7_advice"] = 0.0
            timings["s8_render"] = 0.0
            timings["s9_polish"] = 0.0

            response["phase0_raw_data"] = _strip_zoneinfo(
                drop_empty_lists(
                    clean_payload(
                        raw_data,
                        keep_top_level_list_keys=DEFAULT_KEEP_TOP_LEVEL_LIST_KEYS,
                    ),
                    keys=_DISPLAY_DROP_EMPTY_LIST_KEYS,
                )
            )
            response["phase1_signals"] = processor_dump
            response["phase1_extracted_signals"] = processor_dump
            response["phase1b_narrative"] = user_narrative
            response["phase1b_user_narrative"] = user_narrative
            response["phase2_group"] = group_info
            response["qa_selected_group"] = group_info
            response["phase3_question"] = chosen_q
            response["qa_selected_question"] = chosen_q
            response["phase4_qa_answer"] = answer
            response["qa_answer"] = answer
            response["phase4b_advice"] = response["s7_advice"]
            response["advice"] = response["s7_advice"]
            response["phase5_render"] = final_prose
            response["phase5_rendered_insight"] = final_prose
            response["phase6_polish"] = final_prose
            response["phase6_polished_insight"] = final_prose

            return response

        except Exception as e:
            logger.exception(f"Debug pipeline error: {e}")
            return {"status": "error", "error": str(e)}


def _log_phase_v3(
    phase: str, method_start: float, user_tag: str, extra: str = ""
) -> None:
    """Helper to log a single phase with elapsed time for V3 pipeline."""
    elapsed = time.perf_counter() - method_start
    logger.info(
        f"[TIMING][v3][{phase}] {user_tag} "
        f"elapsed={elapsed:.3f}s" + (f" {extra}" if extra else "")
    )