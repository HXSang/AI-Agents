# Insight Pipeline — Workflow Reference


## 1. Entry Points

3 main endpoints in `routers/insight_router.py` + 1 debug endpoint:

| Endpoint | Request schema | Response schema | Service call |
|---|---|---|---|
| `POST /insight/productivity` | `ProductivityInsightRequest` | `ProductivityInsightResponse` | `analyze_productivity()` → `get_qa_single_insight(domain="productivity")` |
| `POST /insight/health_insight` | `HealthInsightRequest` | `HealthInsightResponse` | `analyze_health_insight()` → `get_qa_single_insight(domain="health")` |
| `POST /insight/overall_insight` | `OverallInsightRequest` | `OverallInsightResponse` | `analyze_overall_insight()` → `get_qa_single_insight(domain="overall")` |
| `POST /insight/debug` | — | full pipeline dump | `debug_pipeline()` |

**The first 3 endpoints all call `get_qa_single_insight()` — a single unified Q&A flow.**

## 2. Cache Inventory

| Key (format) | TTL | Producer | Consumer | Notes |
|---|---|---|---|---|
| `qa_insight:{uid}:{domain}:{lang}` | 300s (5 min) | `get_qa_single_insight` | itself | Primary cache for a single insight |
| `extracted_signals:pipeline_v3:{uid}:{domain}` | 300s | `_get_or_create_extracted_signals` | multiple consumers | **Domain-scoped** signals + context; shared |
| `insight_pool:v2:{uid}` | **no TTL** (30 min logic in pool) | `_build_qa_insight_pool` | `pick_random_insight` | Pool is built on demand for random mode |
| `insight_seen:v2:{uid}` | no TTL | `pick_random_insight` | itself | Tracks previously shown insights to avoid repeats |
| `v2_narrative_bg:{uid}:{lang}:{date}:{fp}` | **disabled** (no cache) | — | — | Narrative is built inline in the prompt; no separate cache |

**Key difference**: the `extracted_signals` cache key changed from `v2` → `pipeline_v3` and is now **domain-scoped** (health/productivity/overall). Narrative is no longer cached separately.

## 3. Detailed Flow: `get_qa_single_insight()`

```
Request → Cache check (qa_insight:{uid}:{domain}:{lang})
           ├─ HIT → return immediately
           └─ MISS:
               ├─ Step 1: Signals (domain-scoped, cached 5 min)
               ├─ Step 2: Prompt assembly (narrative inline — no cache)
               ├─ Step 3: Single LLM call → Sylo Q&A answer
               ├─ Step 4: Hallucination guard + language enforcement
               ├─ Cache 5 min + return
               └─ NO background pre-cache (removed)
```

### Step 1: Signals — `_get_or_create_extracted_signals()`

- **Cache key**: `extracted_signals:pipeline_v3:{uid}:{domain}` (domain-scoped)
- **On cache miss** → calls `_collect_and_process_signals()`:
  - `HealthDataExtractor.extract()`: collects raw data filtered by domain
    - `DataCollector._collect_all_data()`: 12+ parallel API calls
    - `PrepareCalendarData`, `PrepareTimeData`, `PrepareHealthData`
    - Extended snapshots (30 days) for period aggregates
  - `HealthInsightProcessor.process()`:
    - `DataProcessor.extract_signals()`: 5 focus blocks → `ExtractedSignals`
    - `compute_health_period_aggregates()`: today vs. week/month/baseline
    - `_sync_rhr_baseline()`: aligns RHR with the period baseline
    - `_sync_energy_from_period()`: fills energy from period aggregates
    - `_mirror_five_focuses()`: keeps steps ↔ activity and heart_rate ↔ cardio_stress aligned
    - `_apply_health_sanity_guard()`: flags extreme values
    - `ContextSignalBuilder.attach()`: attaches deterministic context signals
- **Output**: an `ExtractedSignals` object with `meta`, `health_signals` (5 focuses), `calendar_intelligence`, `goal_signals`, `behavioral_patterns`, `historical_trends`, `user_profile`, `productivity_signals`, `balance_snapshot`, `finance_signals`, ...

### Context Signals (Processor-attached)

4 deterministic signals in `health_signals.context_signals`:

| Signal | Status | Priority | Trigger |
|---|---|---|---|
| `sleep_window` | active/inactive | **critical** | Current time is within the bedtime window |
| `recent_workout` | active/inactive | high | A workout ≥30 minutes occurred within the last 12 hours |
| `work_hours` | active/inactive | medium | Current time is within active working hours |
| `weekend` | active/inactive | low | Saturday or Sunday |

### Step 2: Prompt Assembly — `_assemble_sylo_prompt()`

1. Build the narrative inline: `_context_narrative()` → `build_3part_narrative_parts()`:
   - **Part 1 (Background)**: sleep, steps, HR/mood/energy, health_score, period_aggregates, historical trends, behavioral patterns, productivity signals, finance signals, user profile
   - **Part 2 (Current)**: time_phase, active context signals, calendar intelligence (events, reminders, free windows), steps today, goals, balance scores
   - **Part 3 (Future)**: next event, upcoming events, free windows, deadlines, active hours end
2. Load the fixed overview question from `DOMAIN_GROUP_REGISTRY`:
   - Health: `h_group_overview` → `h_overview_infer`
   - Productivity: `p_group_overview` → `p_overview_infer`
   - Overall: `o_group_overview` → `o_overview_infer`
3. Build messages: system prompt (persona + rules + guardrails) + user prompt (narrative + context_json + question)

### Step 3: Single LLM Call — Sylo Q&A

- **1 LLM call** (system prompt + user prompt with the narrative inline)
- The system prompt contains:
  - Core persona + language enforcement
  - Domain rules (health/productivity/overall)
  - Insight guardrails (NOT_DO constraints)
  - Full UX rules (sections A–L): vocabulary rules, data integrity, schedule timing, pattern duration, causal linking, mood rules, etc.
  - Canonical field glossary
  - 5-step process: Collect Evidence → Find Story → Multi-layer Reasoning → Emotional Writing → Output
- Output: structured JSON `{"insights": [{...}]}`
- `parse_qa_response()` extracts the answer
- `compose_sylo_prose()` renders: `title + "\n\n" + (insight/current_state)`

### Step 4: Hallucination Guard + Language Enforcement

**Hallucination Guard** (`validate_insight_against_health_params`):
- Checks for figures not present in the source data
- Checks for stale data mislabeled as "today"
- Checks pattern duration violations (e.g. "7 days" when `chronic_overload_days < 7`)
- Checks calendar contradictions
- Logs a warning if violations are found

**Language Consistency (4 layers)**:
- Layer 1: Prompt rule in `core_persona.py`
- Layer 2: Post-generation validator
- Layer 3: Hard sanitizer — replaces English tokens with a Vietnamese mapping
- Layer 4: If still dirty → downgrade confidence → 0.0 → drop the item

## 4. Five-Focus Health Signal Architecture

Each focus block has: `summary → signals → anomalies → overall`

| Focus | Processor | Key signals | Output schema |
|---|---|---|---|
| **sleep** | `sleep/processor.py` | sleep_duration, sleep_efficiency, sleep_consistency, deep_sleep, rem_sleep, sleep_debt, sleep_quality, bedtime_timing, sleep_trend, recovery | `SleepSignal` |
| **heart_rate** | `heart_rate/processor.py` | resting_heart_rate, hrv, heart_rate_range, heart_rate_stability, recovery_state, recovery_trend, baseline_comparison | `HeartRateSignal` |
| **steps** | `steps/processor.py` | activity_volume, goal_achievement, activity_consistency, step_trend, baseline_comparison, activity_pattern | `StepsSignal` |
| **energy** | `energy/processor.py` | activity_volume, energy_consistency, energy_trend, baseline_comparison | `EnergySignal` |
| **mood** | `mood/processor.py` | snapshot, baseline, trend, pattern, stability, frequency, change, correlation | `MoodSignal` |

**Also**: `HealthScoreSignal` (composite daily healthScore signal engine) with a `HealthScoreFiredSignal` list.

## 5. Period Aggregates

Computed via `compute_health_period_aggregates()` — today vs. week/month/baseline:

| Metric | today | week_avg | month_avg | baseline_avg |
|---|---|---|---|---|
| steps | ✅ | ✅ | ✅ | ✅ |
| sleep_hours | ✅ | ✅ | ✅ | ✅ |
| resting_heart_rate | ✅ | ✅ | ✅ | ✅ |
| active_minutes | ✅ | ✅ | ✅ | ✅ |
| total_workout_min | ✅ | ✅ | ✅ | ✅ |
| calories_burned | ✅ | ✅ | ✅ | ✅ |

Energy mirrors: `energy_month_resting_total`, `energy_total_active_today`, `energy_current_month_avg`, `energy_week_avg`, `energy_prev_month_avg`, `energy_week_vs_prior_month_pct`

## 6. LLM Call Count

| Step | Phase | LLM calls |
|---|---|---|
| 1 | Signals (extract + process) | 0 |
| 2 | Narrative + prompt assembly | 0 |
| 3 | Q&A single call | **1** |
| **Total per domain** | | **1** |

**Fully removed**: the separate group-selection call, the separate render call, the separate polish call, and the background pre-cache for 2 domains.

## 7. time_phase Resolution

`compute_time_phase()` (`helpers/day_profile.py`):

- `time_to_bedtime <= 0` → `bedtime`
- `time_to_bedtime <= wind_down_buf` (60 min default) → `wind_down`
- Hour-based fallback:
  - `< 5` → `late_night`
  - `< 12` → `morning`
  - `< active_end_hour` (default 18.0) → `afternoon`
  - `< 22` → `evening`
  - `>= 22` → `late_night`

**Possible values**: `wind_down, bedtime, morning, evening, after_work, afternoon, late_night, Unknown` (8 values)

## 8. Response Format

```python
# Single Insight (primary)
{
    "status": "success" | "error",
    "user_id": str,
    "insight": {
        "insight": prose,           # Final rendered paragraph
        "question_id": str,         # e.g. "h_overview_infer"
        "question_text": str,       # Question text
        "domain": str,              # health | productivity | overall
        "action_family": str,
        "tone": str,                # encouraging | urgent | gentle | celebratory | informative
        "confidence": float,        # 0.5–1.0
    },
    "error": Optional[str]
}
```

## 9. Failure Modes

| Scenario | Behavior |
|---|---|
| LLM timeout/error | Returns `{"status": "error", "insight": None, "error": ...}` |
| Redis down | Full recompute (1 LLM call + extraction calls) |
| Stale cache payload | Bypasses the cache, recomputes |
| Stale health data (≥1 day) | Hallucination guard logs a warning; the LLM is hedged via prompt rules |
| Language violation | Layer 3 sanitizer → re-check → Layer 4 downgrade confidence → drop |

## 10. Debug Pipeline

`POST /insight/debug` → `debug_pipeline()`:

4 layers mirroring production:
1. **Extract**: raw_data → canonicalize → field_keep_map projection
2. **Signals**: `HealthInsightProcessor.process()` → `ExtractedSignals`
3. **Prompt**: `_assemble_sylo_prompt()` → narrative + overview question + messages
4. **LLM**: single Sylo Q&A call → parsed answer

Output contains: `s1_extract`, `s2_signals`, `s3_prompt` (narrative + group + question), `s4_llm` (parsed answer), `timings`, plus legacy aliases (`s3_narrative`, `s6_qa`, `s8_render`, `phase*_keys`).

## 11. Legacy Code Removed

| Item | Notes |
|---|---|
| `_generate_and_cache_all_insights()` | 6-round multi-domain pipeline (350+ lines) — fully removed |
| `_get_or_generate_insight()` | Old round-based entry point |
| `insight_service_impl.py` | Legacy service, no longer imported |
| Background pre-cache for 2 domains | Removed — no more background sequential calls |
| Separate render/polish calls | Merged into the single Sylo Q&A output |

## 12. Data Sources

### API Endpoints Called

| Topic | Endpoint | Output |
|---|---|---|
| health | `GET /api/health/summaries` (today_health_stats) | Energy/HR/Sleep/Steps summary |
| health | `GET /api/health/summaries` (health_params via range) | Flat health params dict |
| balance | `GET /api/balance/score` | Today's balance scores |
| balance | `GET /api/balance` (30d range) | Historical balance scores |
| calendar | `GET /api/calendar/event` | Today's events |
| calendar | `GET /api/calendar/reminders` | Today's reminders |
| calendar | `GET /api/calendar/work-hours` | Work hours config |
| moods | `GET /api/moods/latest` | Latest mood |
| moods | `GET /api/moods` | 7-day mood history |
| snapshots | `GET /api/daily-user-snapshots` (30d) | Historical daily data |
| productivity | `GET /api/productivity/summaries` (7d) | Productivity 7-day summaries |
| productivity | `GET /api/productivity/summary` | Today's productivity |
| user | `GET /api/onboarding/users/profiles` | User profile + goals |

## 13. Key File Map

```
insights/
├── services/
│   ├── advanced_insight_service.py    # Main service (get_qa_single_insight, get_random_insight)
│   ├── insight_picker.py              # Pool management + random pick
│   └── base.py                        # Base interface
├── prompts/
│   ├── qa_prompts.py                  # System prompts + user prompt builders + DOMAIN_GROUP_REGISTRY
│   ├── core_persona.py                # Language enforcement
│   ├── domain_rules.py                # Per-domain rules
│   ├── insight_guardrails.py          # NOT_DO constraints
│   └── context_narrative_templates.py # 3-part narrative builder
├── helpers/
│   ├── question_gate.py               # Question registry, post-filter, relevance scoring
│   ├── insight_validator.py           # Hallucination guard + language sanitizer
│   └── day_profile.py                 # time_phase computation
├── health/
│   ├── extractor.py                   # HealthDataExtractor: collects raw data
│   ├── health_processor/
│   │   ├── processor.py               # HealthInsightProcessor: signals pipeline
│   │   ├── context/processor.py       # ContextSignalBuilder (sleep_window, weekend, work_hours, recent_workout)
│   │   ├── sleep/processor.py         # SleepSignal processor
│   │   ├── heart_rate/processor.py    # HeartRateSignal processor
│   │   ├── steps/processor.py         # StepsSignal processor
│   │   ├── energy/processor.py        # EnergySignal processor
│   │   ├── mood/processor.py          # MoodSignal processor
│   │   ├── health_score/processor.py  # HealthScoreSignal processor
│   │   └── period_metrics.py          # compute_health_period_aggregates
│   └── field_keep_maps.py             # Field projection map
├── schemas/
│   ├── processed_context.py           # ExtractedSignals + all signal block schemas
│   ├── insight_pool.py                # InsightPool schema
│   └── insight_item.py                # InsightItem schema
└── processors/
    ├── data_processor.py               # DataProcessor.extract_signals (typed blocks)
    ├── calendar_intelligence_processor.py
    ├── productivity_signal_processor.py
    ├── reminder_signal_processor.py
    ├── goal_progress_processor.py
    ├── historical_trends_processor.py
    ├── behavioral_pattern_processor.py
    ├── balance_snapshot_processor.py
    ├── work_hours_processor.py
    └── finance_signal_processor.py
```
