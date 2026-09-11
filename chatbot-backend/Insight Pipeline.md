# Insight Pipeline — Workflow Reference
> **Stack**: FastAPI → `AdvancedInsightService` (Bedrock LLM via `LLMManager`) → Redis cache → `ExternalAPIServiceImpl` (BE pipeline).

---

## 1. Entry Points

3 endpoint chính trong `routers/insight_router.py` + 1 endpoint debug + 1 cho random mode:

| Endpoint | Request schema | Response schema | Service call |
|---|---|---|---|
| `POST /insight/productivity` | `ProductivityInsightRequest` | `ProductivityInsightResponse` | `analyze_productivity()` → `get_qa_single_insight(domain="productivity")` |
| `POST /insight/health_insight` | `HealthInsightRequest` | `HealthInsightResponse` | `analyze_health_insight()` → `get_qa_single_insight(domain="health")` |
| `POST /insight/overall_insight` | `OverallInsightRequest` | `OverallInsightResponse` | `analyze_overall_insight()` → `get_qa_single_insight(domain="overall")` |
| `POST /insight/random_insight` | — | — | `get_random_insight()` → `_build_qa_insight_pool` + `pick_random_insight` |
| `POST /insight/debug` | — | full pipeline dump | `debug_pipeline()` |

**3 endpoint đầu đều gọi `get_qa_single_insight()` — luồng Q&A duy nhất.**

---

## 2. Cache Inventory

| Key (format) | TTL | Producer | Consumer | Notes |
|---|---|---|---|---|
| `qa_insight:{uid}:{domain}:{lang}` | 300s (5 min) | `get_qa_single_insight` | chính nó | Primary cache cho single insight |
| `extracted_signals:pipeline_v3:{uid}:{domain}` | 300s | `_get_or_create_extracted_signals` | nhiều nơi | **Domain-scoped** signals + context; dùng chung |
| `insight_pool:v2:{uid}` | **no TTL** (30 min logic in pool) | `_build_qa_insight_pool` | `pick_random_insight` | Pool được build khi cần cho random mode |
| `insight_seen:v2:{uid}` | no TTL | `pick_random_insight` | chính nó | Tracking insight đã shown để tránh lặp |
| `v2_narrative_bg:{uid}:{lang}:{date}:{fp}` | **disabled** (no cache) | — | — | Narrative được build inline trong prompt, không cache riêng |

**Điểm khác biệt quan trọng**: `extracted_signals` cache key đã thay đổi từ `v2` → `pipeline_v3` và **có domain scope** (health/productivity/overall). Narrative không còn cache riêng.

---

## 3. Luồng chi tiết: `get_qa_single_insight()`

```
Request → Cache check (qa_insight:{uid}:{domain}:{lang})
           ├─ HIT → return immediately
           └─ MISS:
               ├─ Step 1: Signals (domain-scoped, cached 5 min)
               ├─ Step 2: Prompt assembly (narrative inline — no cache)
               ├─ Step 3: LLM single call → Sylo Q&A answer
               ├─ Step 4: Hallucination guard + language enforcement
               ├─ Cache 5min + return
               └─ NO background pre-cache (đã bỏ)
```

### Step 1: Signals — `_get_or_create_extracted_signals()`

- **Cache key**: `extracted_signals:pipeline_v3:{uid}:{domain}` (domain-scoped)
- **Cache miss** → gọi `_collect_and_process_signals()`:
  - `HealthDataExtractor.extract()`: collect raw data filtered by domain
    - `DataCollector._collect_all_data()`: 12+ parallel API calls
    - `PrepareCalendarData`, `PrepareTimeData`, `PrepareHealthData`
    - Extended snapshots (30 days) cho period aggregates
  - `HealthInsightProcessor.process()`:
    - `DataProcessor.extract_signals()`: 5 focus blocks → `ExtractedSignals`
    - `compute_health_period_aggregates()`: today vs week/month/baseline
    - `_sync_rhr_baseline()`: align RHR với period baseline
    - `_sync_energy_from_period()`: fill energy từ period aggregates
    - `_mirror_five_focuses()`: keep steps ↔ activity, heart_rate ↔ cardio_stress aligned
    - `_apply_health_sanity_guard()`: flag extreme values
    - `ContextSignalBuilder.attach()`: attach deterministic context signals
- **Output**: `ExtractedSignals` object với `meta`, `health_signals` (5 focuses), `calendar_intelligence`, `goal_signals`, `behavioral_patterns`, `historical_trends`, `user_profile`, `productivity_signals`, `balance_snapshot`, `finance_signals`, ...

### Context Signals (Processor-attached)

4 deterministic signals trong `health_signals.context_signals`:

| Signal | Status | Priority | Trigger |
|---|---|---|---|
| `sleep_window` | active/inactive | **critical** | Current time trong bedtime window |
| `recent_workout` | active/inactive | high | Workout ≥30 phút trong 12h qua |
| `work_hours` | active/inactive | medium | Current time trong active hours |
| `weekend` | active/inactive | low | Thứ 7 / Chủ nhật |

### Step 2: Prompt Assembly — `_assemble_sylo_prompt()`

1. Build narrative inline: `_context_narrative()` → `build_3part_narrative_parts()`:
   - **Part 1 (Background)**: sleep, steps, HR/mood/energy, health_score, period_aggregates, historical trends, behavioral patterns, productivity signals, finance signals, user profile
   - **Part 2 (Current)**: time_phase, active context signals, calendar intelligence (events, reminders, free windows), steps today, goals, balance scores
   - **Part 3 (Future)**: next event, upcoming, free windows, deadlines, active hours end
2. Load fixed overview question từ `DOMAIN_GROUP_REGISTRY`:
   - Health: `h_group_overview` → `h_overview_infer`
   - Productivity: `p_group_overview` → `p_overview_infer`
   - Overall: `o_group_overview` → `o_overview_infer`
3. Build messages: system prompt (persona + rules + guardrails) + user prompt (narrative + context_json + question)

### Step 3: LLM Single Call — Sylo Q&A

- **1 LLM call** (system prompt + user prompt với narrative inline)
- System prompt chứa:
  - Core persona + language enforcement
  - Domain rules (health/productivity/overall)
  - Insight guardrails (NOT_DO constraints)
  - Full UX rules (A–L sections): vocabulary rules, data integrity, schedule timing, pattern duration, causal linking, mood rules, etc.
  - Canonical field glossary
  - 5-step process: Collect Evidence → Find Story → Multi-layer Reasoning → Emotional Writing → Output
- Output: structured JSON `{"insights": [{...}]}`
- `parse_qa_response()` extracts answer
- `compose_sylo_prose()` renders: `title + "\n\n" + (insight/current_state)`

### Step 4: Hallucination Guard + Language Enforcement

**Hallucination Guard** (`validate_insight_against_health_params`):
- Check số liệu không có trong source data
- Check stale data được gắn là "today"
- Check pattern duration violations (e.g. "7 days" khi `chronic_overload_days < 7`)
- Check calendar contradictions
- Log warning nếu violations tìm thấy

**Language Consistency (4 layers)**:
- Layer 1: Prompt rule trong `core_persona.py`
- Layer 2: Post-generation validator
- Layer 3: Hard sanitizer — replace English tokens bằng Vietnamese mapping
- Layer 4: Nếu still dirty → downgrade confidence → 0.0 → drop item

---

## 4. Five-Focus Health Signal Architecture

Mỗi focus block có: `summary → signals → anomalies → overall`

| Focus | Processor | Key signals | Output schema |
|---|---|---|---|
| **sleep** | `sleep/processor.py` | sleep_duration, sleep_efficiency, sleep_consistency, deep_sleep, rem_sleep, sleep_debt, sleep_quality, bedtime_timing, sleep_trend, recovery | `SleepSignal` |
| **heart_rate** | `heart_rate/processor.py` | resting_heart_rate, hrv, heart_rate_range, heart_rate_stability, recovery_state, recovery_trend, baseline_comparison | `HeartRateSignal` |
| **steps** | `steps/processor.py` | activity_volume, goal_achievement, activity_consistency, step_trend, baseline_comparison, activity_pattern | `StepsSignal` |
| **energy** | `energy/processor.py` | activity_volume, energy_consistency, energy_trend, baseline_comparison | `EnergySignal` |
| **mood** | `mood/processor.py` | snapshot, baseline, trend, pattern, stability, frequency, change, correlation | `MoodSignal` |

**Ngoài ra**: `HealthScoreSignal` (composite daily healthScore signal engine) với `HealthScoreFiredSignal` list.

---

## 5. Period Aggregates

Tính từ `compute_health_period_aggregates()` — today vs week/month/baseline:

| Metric | today | week_avg | month_avg | baseline_avg |
|---|---|---|---|---|
| steps | ✅ | ✅ | ✅ | ✅ |
| sleep_hours | ✅ | ✅ | ✅ | ✅ |
| resting_heart_rate | ✅ | ✅ | ✅ | ✅ |
| active_minutes | ✅ | ✅ | ✅ | ✅ |
| total_workout_min | ✅ | ✅ | ✅ | ✅ |
| calories_burned | ✅ | ✅ | ✅ | ✅ |

Energy mirrors: `energy_month_resting_total`, `energy_total_active_today`, `energy_current_month_avg`, `energy_week_avg`, `energy_prev_month_avg`, `energy_week_vs_prior_month_pct`

---

## 6. LLM Call Count

| Step | Phase | LLM calls |
|---|---|---|
| 1 | Signals (extract + process) | 0 |
| 2 | Narrative + prompt assembly | 0 |
| 3 | Q&A single call | **1** |
| **Total per domain** | | **1** |

**Đã bỏ hoàn toàn**: group selection call riêng, render call riêng, polish call riêng, background pre-cache 2 domains.

---

## 7. time_phase Resolution

`compute_time_phase()` (`helpers/day_profile.py`):

- `time_to_bedtime <= 0` → `bedtime`
- `time_to_bedtime <= wind_down_buf` (60m default) → `wind_down`
- Hour-based fallback:
  - `< 5` → `late_night`
  - `< 12` → `morning`
  - `< active_end_hour` (default 18.0) → `afternoon`
  - `< 22` → `evening`
  - `>= 22` → `late_night`

**Possible values**: `wind_down, bedtime, morning, evening, after_work, afternoon, late_night, Unknown` (8 values)

---

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

# Random Insight (pool mode)
{
    "status": "success" | "error",
    "user_id": str,
    "insight": {
        "point1": prose,
        "point2": "",
        "insight_id": str,
        "question_id": str,
        "domain": str,
        "action_family": str,
        "tone": str,
        "confidence": float,
        "current_state": str,
        "cause": str,
        "action": str,
        "evidence": str,
        "expected_outcome": str,
    },
    "error": Optional[str]
}
```

---

## 9. Random Insight Mode

Luồng `_build_qa_insight_pool` + `pick_random_insight`:

1. Build signals (cached, domain-scoped)
2. Build narrative (inline)
3. Check pool exists → build if not
4. Build pool: run ALL groups × ALL domains qua LLM
   - `get_active_questions()`: returns overview question per domain
   - Run group → LLM Q&A → parse → `InsightItem`
   - Language consistency enforcement (4 layers)
   - Post-filter: confidence ≥ 0.3, deduplicate by domain+question+action_family
   - Sort by `score_insight_relevance()` (confidence-based)
5. Pick random: from unseen candidates → mark seen → return

Pool cache: `insight_pool:v2:{uid}` (no TTL, checked on read)
Seen tracking: `insight_seen:v2:{uid}` (reset khi pool exhausted)

---

## 10. Failure Modes

| Scenario | Behavior |
|---|---|
| LLM timeout/error | Return `{"status": "error", "insight": None, "error": ...}` |
| Redis down | Full recompute (1 LLM + extract calls) |
| Cache stale payload | Bypass cache, recompute |
| Stale health data (≥1 day) | Hallucination guard logs warning; LLM được hedge qua prompt rules |
| Language violation | Layer 3 sanitizer → re-check → Layer 4 downgrade confidence → drop |

---

## 11. Debug Pipeline

`POST /insight/debug` → `debug_pipeline()`:

4 layers tương tự production:
1. **Extract**: raw_data → canonicalize → field_keep_map projection
2. **Signals**: `HealthInsightProcessor.process()` → `ExtractedSignals`
3. **Prompt**: `_assemble_sylo_prompt()` → narrative + overview question + messages
4. **LLM**: single Sylo Q&A call → parsed answer

Output chứa: `s1_extract`, `s2_signals`, `s3_prompt` (narrative + group + question), `s4_llm` (parsed answer), `timings`, plus legacy aliases (`s3_narrative`, `s6_qa`, `s8_render`, `phase*_keys`).

---

## 12. Legacy Code Đã Xóa

| Item | Notes |
|---|---|
| `_generate_and_cache_all_insights()` | 6-round multi-domain pipeline (350+ lines) — đã xóa hoàn toàn |
| `_get_or_generate_insight()` | Old round-based entry point |
| `insight_service_impl.py` | Legacy service, không còn import |
| Background pre-cache 2 domains | Đã bỏ — không còn background sequential calls |
| Separate render/polish calls | Đã gộp vào single Sylo Q&A output |

---

## 13. Data Sources

### API Endpoints Called

| Topic | Endpoint | Output |
|---|---|---|
| health | `GET /api/health/summaries` (today_health_stats) | Energy/HR/Sleep/Steps summary |
| health | `GET /api/health/summaries` (health_params via range) | Flat health params dict |
| balance | `GET /api/balance/score` | Today's balance scores |
| balance | `GET /api/balance` (30d range) | Balance scores historical |
| calendar | `GET /api/calendar/event` | Today's events |
| calendar | `GET /api/calendar/reminders` | Today's reminders |
| calendar | `GET /api/calendar/work-hours` | Work hours config |
| moods | `GET /api/moods/latest` | Latest mood |
| moods | `GET /api/moods` | 7-day mood history |
| snapshots | `GET /api/daily-user-snapshots` (30d) | Historical daily data |
| productivity | `GET /api/productivity/summaries` (7d) | Productivity 7-day summaries |
| productivity | `GET /api/productivity/summary` | Today's productivity |
| user | `GET /api/onboarding/users/profiles` | User profile + goals |

---

## 14. Key File Map

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
│   ├── extractor.py                   # HealthDataExtractor: collect raw data
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
