# Insight Pipeline

## Technical Architecture and Workflow Reference

> **System Stack:** FastAPI → `AdvancedInsightService` → Bedrock LLM
> (via `LLMManager`) → Redis → `ExternalAPIServiceImpl` → Backend APIs

------------------------------------------------------------------------

# 1. Overview

The Insight Pipeline is responsible for generating personalized insights
across three primary domains:

-   **Health**
-   **Productivity**
-   **Overall**

The production pipeline follows a deterministic processing flow before
invoking the LLM:

``` text
Request
   │
   ▼
API Router
   │
   ▼
AdvancedInsightService
   │
   ├── Redis Cache Check
   │
   ├── Signal Extraction & Processing
   │
   ├── Context & Narrative Construction
   │
   ├── Prompt Assembly
   │
   ├── Single LLM Invocation
   │
   └── Validation & Response Formatting
   │
   ▼
Final Insight Response
```

The current architecture is designed around a **single LLM call per
domain request**. All data collection, signal processing, context
construction, and validation logic are performed deterministically
outside the LLM.

------------------------------------------------------------------------

# 2. System Architecture

| Component | Responsibility |
|---|---|
| FastAPI Router | Receives requests and exposes Insight API endpoints |
| `AdvancedInsightService` | Orchestrates the complete insight generation workflow |
| `HealthDataExtractor` | Collects and normalizes raw data from backend APIs |
| `HealthInsightProcessor` | Converts raw data into structured signals |
| `ContextSignalBuilder` | Attaches deterministic real-time context signals |
| Prompt Layer | Builds system and user prompts |
| `LLMManager` | Manages Bedrock LLM invocation |
| Redis | Stores short-lived insight and signal caches |
| Insight Validator | Validates hallucinations and language consistency |
| `ExternalAPIServiceImpl` | Provides access to backend services |

------------------------------------------------------------------------

# 3. API Entry Points

The pipeline exposes three primary insight endpoints and one debugging
endpoint.

  -------------------------------------------------------------------------------------------------------------------------------------------------
  Endpoint                          Request Schema                 Response Schema                 Service Flow
  --------------------------------- ------------------------------ ------------------------------- ------------------------------------------------
  `POST /insight/productivity`      `ProductivityInsightRequest`   `ProductivityInsightResponse`   `analyze_productivity()` →
                                                                                                   `get_qa_single_insight(domain="productivity")`

  `POST /insight/health_insight`    `HealthInsightRequest`         `HealthInsightResponse`         `analyze_health_insight()` →
                                                                                                   `get_qa_single_insight(domain="health")`

  `POST /insight/overall_insight`   `OverallInsightRequest`        `OverallInsightResponse`        `analyze_overall_insight()` →
                                                                                                   `get_qa_single_insight(domain="overall")`

  `POST /insight/debug`             ---                            Full pipeline diagnostics       `debug_pipeline()`
  -------------------------------------------------------------------------------------------------------------------------------------------------

### Unified Insight Generation Flow

The three production endpoints use the same underlying orchestration
method:

``` python
get_qa_single_insight()
```

The requested domain determines:

-   The data scope
-   The domain rules
-   The overview question
-   The prompt context
-   The generated insight

------------------------------------------------------------------------

# 4. Cache Architecture

Redis is used to reduce repeated data processing and unnecessary LLM
invocations.

## 4.1 Cache Inventory

  --------------------------------------------------------------------------------------------------------------------------------------------------
  Cache Key                                        TTL            Producer                               Consumer                    Purpose
  ------------------------------------------------ -------------- -------------------------------------- --------------------------- ---------------
  `qa_insight:{uid}:{domain}:{lang}`               300 seconds    `get_qa_single_insight()`              `get_qa_single_insight()`   Primary cache
                                                                                                                                     for generated
                                                                                                                                     single insights

  `extracted_signals:pipeline_v3:{uid}:{domain}`   300 seconds    `_get_or_create_extracted_signals()`   Multiple pipeline           Domain-scoped
                                                                                                         components                  processed
                                                                                                                                     signals and
                                                                                                                                     context

  `insight_pool:v2:{uid}`                          No Redis TTL   `_build_qa_insight_pool()`             `pick_random_insight()`     Stores
                                                                                                                                     generated
                                                                                                                                     insights for
                                                                                                                                     random
                                                                                                                                     selection

  `insight_seen:v2:{uid}`                          No Redis TTL   `pick_random_insight()`                `pick_random_insight()`     Tracks
                                                                                                                                     previously
                                                                                                                                     displayed
                                                                                                                                     insights

  `v2_narrative_bg:{uid}:{lang}:{date}:{fp}`       Disabled       ---                                    ---                         Legacy
                                                                                                                                     narrative
                                                                                                                                     cache; no
                                                                                                                                     longer used
  --------------------------------------------------------------------------------------------------------------------------------------------------

## 4.2 Domain-Scoped Signal Cache

The current signal cache uses:

``` text
extracted_signals:pipeline_v3:{uid}:{domain}
```

The cache is explicitly scoped by domain:

``` text
health
productivity
overall
```

This prevents context generated for one domain from being incorrectly
reused by another domain.

The previous cache version:

``` text
extracted_signals:v2:{uid}
```

has been replaced by the `pipeline_v3` domain-scoped implementation.

## 4.3 Narrative Caching

Narratives are no longer cached independently.

Instead, the narrative is generated inline during prompt construction:

``` text
ExtractedSignals
      │
      ▼
Context Narrative Builder
      │
      ▼
Prompt Assembly
      │
      ▼
Single LLM Call
```

This avoids maintaining a separate narrative cache and ensures that the
generated narrative remains synchronized with the current signal
context.

------------------------------------------------------------------------

# 5. Single Insight Execution Flow

The primary production method is:

``` python
get_qa_single_insight()
```

The execution flow is:

``` text
Request
   │
   ▼
Check Insight Cache
   │
   ├── Cache HIT
   │      │
   │      └── Return Cached Insight
   │
   └── Cache MISS
          │
          ▼
     Extract or Load Signals
          │
          ▼
     Build Context Narrative
          │
          ▼
     Assemble Prompt
          │
          ▼
     Single LLM Invocation
          │
          ▼
     Parse LLM Response
          │
          ▼
     Validate Output
          │
          ▼
     Cache Result
          │
          ▼
     Return Insight
```

### Execution Characteristics

-   Signal processing: deterministic
-   Narrative construction: deterministic
-   Prompt construction: deterministic
-   LLM calls: **1**
-   Background pre-generation: **disabled**
-   Separate render calls: **removed**
-   Separate polish calls: **removed**

------------------------------------------------------------------------

# 6. Signal Extraction and Processing

## 6.1 Signal Pipeline

Signals are obtained through:

``` python
_get_or_create_extracted_signals()
```

The process begins with a Redis lookup.

``` text
Check Signal Cache
       │
       ├── HIT
       │      │
       │      └── Return ExtractedSignals
       │
       └── MISS
              │
              ▼
       _collect_and_process_signals()
              │
              ▼
       HealthDataExtractor.extract()
              │
              ▼
       Raw Data Collection
              │
              ▼
       HealthInsightProcessor.process()
              │
              ▼
       ExtractedSignals
              │
              ▼
       Store in Redis
```

## 6.2 Raw Data Collection

`HealthDataExtractor.extract()` coordinates raw data collection.

The underlying collector:

``` python
DataCollector._collect_all_data()
```

performs **12+ API calls in parallel**.

The extraction layer also prepares:

-   Calendar data
-   Time data
-   Health data
-   User profile data
-   Productivity data
-   Historical snapshots
-   Extended 30-day snapshots for period analysis

Primary preparation components include:

``` text
PrepareCalendarData
PrepareTimeData
PrepareHealthData
```

------------------------------------------------------------------------

# 7. Signal Processing

Raw data is transformed into structured signals through:

``` python
HealthInsightProcessor.process()
```

The processing pipeline includes:

``` text
Raw Data
   │
   ▼
DataProcessor.extract_signals()
   │
   ▼
Five-Focus Health Signals
   │
   ├── Sleep
   ├── Heart Rate
   ├── Steps
   ├── Energy
   └── Mood
   │
   ▼
Period Aggregation
   │
   ▼
Signal Synchronization
   │
   ▼
Sanity Validation
   │
   ▼
Context Signal Attachment
   │
   ▼
ExtractedSignals
```

## 7.1 Processing Stages

### Signal Extraction

``` python
DataProcessor.extract_signals()
```

Extracts typed signal blocks and produces the core `ExtractedSignals`
structure.

### Period Aggregation

``` python
compute_health_period_aggregates()
```

Computes comparisons across:

``` text
Today
Week
Month
Baseline
```

### Resting Heart Rate Synchronization

``` python
_sync_rhr_baseline()
```

Aligns resting heart rate data with the appropriate period baseline.

### Energy Synchronization

``` python
_sync_energy_from_period()
```

Populates energy-related values from period aggregates.

### Focus Mirroring

``` python
_mirror_five_focuses()
```

Maintains consistency between related signal groups:

``` text
Steps ↔ Activity

Heart Rate ↔ Cardio Stress
```

### Health Sanity Guard

``` python
_apply_health_sanity_guard()
```

Detects and flags extreme or potentially invalid health values.

### Context Attachment

``` python
ContextSignalBuilder.attach()
```

Adds deterministic context signals based on the current user state and
time.

------------------------------------------------------------------------

# 8. ExtractedSignals Structure

The final output of the processing pipeline is an `ExtractedSignals`
object.

The structure may include:

``` text
ExtractedSignals
│
├── meta
│
├── health_signals
│   ├── sleep
│   ├── heart_rate
│   ├── steps
│   ├── energy
│   ├── mood
│   └── context_signals
│
├── calendar_intelligence
├── goal_signals
├── behavioral_patterns
├── historical_trends
├── user_profile
├── productivity_signals
├── balance_snapshot
└── finance_signals
```

This object serves as the canonical structured context used during
prompt generation.

------------------------------------------------------------------------

# 9. Context Signals

Context signals are deterministic signals attached by:

``` python
ContextSignalBuilder
```

They are stored under:

``` text
health_signals.context_signals
```

## 9.1 Context Signal Definitions

  ------------------------------------------------------------------------
  Signal             Status            Priority          Activation
                                                         Condition
  ------------------ ----------------- ----------------- -----------------
  `sleep_window`     active / inactive **Critical**      Current time
                                                         falls within the
                                                         user's bedtime
                                                         window

  `recent_workout`   active / inactive High              A workout lasting
                                                         ≥30 minutes
                                                         occurred within
                                                         the previous 12
                                                         hours

  `work_hours`       active / inactive Medium            Current time
                                                         falls within
                                                         configured active
                                                         working hours

  `weekend`          active / inactive Low               Current day is
                                                         Saturday or
                                                         Sunday
  ------------------------------------------------------------------------

These signals provide deterministic real-time context to the LLM.

------------------------------------------------------------------------

# 10. Prompt Assembly

Prompt generation is handled by:

``` python
_assemble_sylo_prompt()
```

The prompt assembly process consists of three stages:

``` text
ExtractedSignals
       │
       ▼
Context Narrative Construction
       │
       ▼
Overview Question Resolution
       │
       ▼
System + User Prompt Assembly
```

------------------------------------------------------------------------

# 11. Context Narrative

The context narrative is generated inline through:

``` python
_context_narrative()
```

which internally uses:

``` python
build_3part_narrative_parts()
```

The narrative is divided into three temporal perspectives.

## 11.1 Part 1 --- Background

The background section may include:

-   Sleep metrics
-   Step activity
-   Heart rate
-   Mood
-   Energy
-   Health score
-   Period aggregates
-   Historical trends
-   Behavioral patterns
-   Productivity signals
-   Finance signals
-   User profile

The purpose of this section is to provide historical and behavioral
context.

## 11.2 Part 2 --- Current State

The current-state section may include:

-   `time_phase`
-   Active context signals
-   Calendar intelligence
-   Events
-   Reminders
-   Available time windows
-   Today's step count
-   Goal progress
-   Balance scores

The purpose of this section is to describe the user's immediate
situation.

## 11.3 Part 3 --- Future Context

The future section may include:

-   Next calendar event
-   Upcoming events
-   Available time windows
-   Deadlines
-   Active hours end time

The purpose of this section is to provide forward-looking context.

------------------------------------------------------------------------

# 12. Domain Question Resolution

Each domain uses a predefined overview question from:

``` python
DOMAIN_GROUP_REGISTRY
```

  Domain         Group                Question
  -------------- -------------------- --------------------
  Health         `h_group_overview`   `h_overview_infer`
  Productivity   `p_group_overview`   `p_overview_infer`
  Overall        `o_group_overview`   `o_overview_infer`

The selected question guides the LLM toward the appropriate insight
scope.

------------------------------------------------------------------------

# 13. LLM Generation

The pipeline performs a single LLM invocation.

``` text
System Prompt
      +
User Prompt
      │
      ▼
Bedrock LLM
      │
      ▼
Structured JSON Response
      │
      ▼
Response Parser
      │
      ▼
Insight Object
```

## 13.1 LLM Call Count

  Pipeline Stage                   LLM Calls
  ------------------------------ -----------
  Signal extraction                        0
  Signal processing                        0
  Narrative construction                   0
  Prompt assembly                          0
  Insight generation                   **1**
  Validation                               0
  **Total per domain request**         **1**

------------------------------------------------------------------------

# 14. System Prompt Responsibilities

The system prompt defines the behavioral and output constraints for
Sylo.

It includes:

### Persona

Defines the core assistant identity and communication style.

### Language Enforcement

Defines the required output language and language consistency rules.

### Domain Rules

Provides domain-specific behavior for:

``` text
Health
Productivity
Overall
```

### Insight Guardrails

Defines prohibited behavior through `NOT_DO` constraints.

### UX Rules

The prompt includes detailed UX rules covering:

-   Vocabulary usage
-   Data integrity
-   Schedule awareness
-   Timing interpretation
-   Pattern duration
-   Causal relationships
-   Mood interpretation
-   Context relevance

### Canonical Field Glossary

Provides definitions and interpretation guidance for structured data
fields.

### Five-Step Reasoning Process

The generation process follows:

``` text
1. Collect Evidence
2. Find the Story
3. Multi-layer Reasoning
4. Emotional Writing
5. Output
```

------------------------------------------------------------------------

# 15. LLM Response Processing

The LLM returns a structured response:

``` json
{
  "insights": [
    {
      "...": "..."
    }
  ]
}
```

The response is processed by:

``` python
parse_qa_response()
```

The final insight prose is generated through:

``` python
compose_sylo_prose()
```

Conceptually:

``` python
title + "\n\n" + (insight or current_state)
```

------------------------------------------------------------------------

# 16. Output Validation

The generated insight passes through multiple validation layers before
being returned to the client.

``` text
LLM Output
    │
    ▼
Hallucination Validation
    │
    ▼
Language Validation
    │
    ▼
Language Sanitization
    │
    ▼
Final Consistency Check
    │
    ▼
Validated Insight
```

------------------------------------------------------------------------

# 17. Hallucination Guard

The primary validation function is:

``` python
validate_insight_against_health_params()
```

The validator checks for several categories of inconsistencies.

## 17.1 Unsupported Numerical Values

Detects numerical claims that are not supported by the source data.

Example:

``` text
Source data:
Steps = 5,000

Generated insight:
"You walked 8,000 steps today."
```

This would be flagged as unsupported.

## 17.2 Stale Data Misrepresentation

Detects historical or stale data incorrectly described as current data.

Example:

``` text
Data timestamp:
Yesterday

Generated statement:
"Today, your heart rate..."
```

This may trigger a validation warning.

## 17.3 Pattern Duration Violations

Detects claims that imply unsupported durations.

Example:

``` text
chronic_overload_days = 3
```

Invalid claim:

``` text
"You have experienced chronic overload for seven days."
```

## 17.4 Calendar Contradictions

Checks whether generated statements contradict known calendar data.

------------------------------------------------------------------------

# 18. Language Consistency Architecture

Language enforcement is implemented as a four-layer system.

## Layer 1 --- Prompt-Level Enforcement

Language requirements are defined in:

``` text
core_persona.py
```

The LLM is instructed to generate output in the required language.

## Layer 2 --- Post-Generation Validation

Generated content is checked for language violations after LLM
generation.

## Layer 3 --- Hard Sanitization

Detected English tokens are replaced using a predefined Vietnamese
mapping.

## Layer 4 --- Final Enforcement

If the output remains invalid:

``` text
Confidence → 0.0

Insight → Dropped
```

This prevents language-inconsistent output from being returned.

------------------------------------------------------------------------

# 19. Five-Focus Health Signal Architecture

Health analysis is organized into five primary focus areas.

Each focus generally follows:

``` text
Summary
   │
   ▼
Signals
   │
   ▼
Anomalies
   │
   ▼
Overall Assessment
```

## 19.1 Sleep

**Processor:**

``` text
health_processor/sleep/processor.py
```

Key signals:

-   `sleep_duration`
-   `sleep_efficiency`
-   `sleep_consistency`
-   `deep_sleep`
-   `rem_sleep`
-   `sleep_debt`
-   `sleep_quality`
-   `bedtime_timing`
-   `sleep_trend`
-   `recovery`

Output:

``` python
SleepSignal
```

## 19.2 Heart Rate

**Processor:**

``` text
health_processor/heart_rate/processor.py
```

Key signals:

-   `resting_heart_rate`
-   `hrv`
-   `heart_rate_range`
-   `heart_rate_stability`
-   `recovery_state`
-   `recovery_trend`
-   `baseline_comparison`

Output:

``` python
HeartRateSignal
```

## 19.3 Steps

**Processor:**

``` text
health_processor/steps/processor.py
```

Key signals:

-   `activity_volume`
-   `goal_achievement`
-   `activity_consistency`
-   `step_trend`
-   `baseline_comparison`
-   `activity_pattern`

Output:

``` python
StepsSignal
```

## 19.4 Energy

**Processor:**

``` text
health_processor/energy/processor.py
```

Key signals:

-   `activity_volume`
-   `energy_consistency`
-   `energy_trend`
-   `baseline_comparison`

Output:

``` python
EnergySignal
```

## 19.5 Mood

**Processor:**

``` text
health_processor/mood/processor.py
```

Key signals:

-   `snapshot`
-   `baseline`
-   `trend`
-   `pattern`
-   `stability`
-   `frequency`
-   `change`
-   `correlation`

Output:

``` python
MoodSignal
```

------------------------------------------------------------------------

# 20. Health Score Signal Engine

The system also includes:

``` python
HealthScoreSignal
```

This component represents a composite daily health score signal engine.

The engine produces a collection of:

``` python
HealthScoreFiredSignal
```

Each fired signal represents a specific factor contributing to the
overall health score interpretation.

------------------------------------------------------------------------

# 21. Period Aggregates

Period aggregates are calculated through:

``` python
compute_health_period_aggregates()
```

The system compares current values against multiple time horizons.

  Metric                  Today   Week Average   Month Average   Baseline Average
  ----------------------- ------- -------------- --------------- ------------------
  Steps                   ✓       ✓              ✓               ✓
  Sleep Hours             ✓       ✓              ✓               ✓
  Resting Heart Rate      ✓       ✓              ✓               ✓
  Active Minutes          ✓       ✓              ✓               ✓
  Total Workout Minutes   ✓       ✓              ✓               ✓
  Calories Burned         ✓       ✓              ✓               ✓

## 21.1 Energy Period Metrics

Energy-related period metrics include:

``` text
energy_month_resting_total

energy_total_active_today

energy_current_month_avg

energy_week_avg

energy_prev_month_avg

energy_week_vs_prior_month_pct
```

These values provide additional context for activity and energy trend
analysis.

------------------------------------------------------------------------

# 22. `time_phase` Resolution

The user's current temporal context is determined by:

``` python
compute_time_phase()
```

located in:

``` text
helpers/day_profile.py
```

## 22.1 Bedtime Logic

``` python
if time_to_bedtime <= 0:
    return "bedtime"
```

## 22.2 Wind-Down Logic

``` python
if time_to_bedtime <= wind_down_buf:
    return "wind_down"
```

The default buffer is:

``` text
60 minutes
```

## 22.3 Hour-Based Fallback

If bedtime rules do not apply:

``` text
Hour < 5
    → late_night

Hour < 12
    → morning

Hour < active_end_hour
    → afternoon

Hour < 22
    → evening

Hour ≥ 22
    → late_night
```

The default value for:

``` text
active_end_hour = 18.0
```

## 22.4 Possible Values

``` text
wind_down
bedtime
morning
evening
after_work
afternoon
late_night
Unknown
```

A total of **eight possible values** are supported.

------------------------------------------------------------------------

# 23. Response Models

## Single Insight Response

``` python
{
    "status": "success" | "error",
    "user_id": str,
    "insight": {
        "insight": prose,
        "question_id": str,
        "question_text": str,
        "domain": str,
        "action_family": str,
        "tone": str,
        "confidence": float,
    },
    "error": Optional[str]
}
```

### Field Definitions

  Field               Description
  ------------------- ------------------------------------------
  `status`            Request execution status
  `user_id`           User identifier
  `insight.insight`   Final rendered insight
  `question_id`       Identifier of the source question
  `question_text`     Question used for generation
  `domain`            Insight domain
  `action_family`     Classification of the recommended action
  `tone`              Communication tone
  `confidence`        Confidence score
  `error`             Error details when applicable

Supported tones include:

``` text
encouraging
urgent
gentle
celebratory
informative
```

------------------------------------------------------------------------

# 24. Failure Handling

The pipeline supports several failure scenarios.

  -----------------------------------------------------------------------
  Scenario                            Behavior
  ----------------------------------- -----------------------------------
  LLM timeout or error                Returns an error response

  Redis unavailable                   Performs a full recomputation

  Stale cache payload                 Bypasses the cache and recomputes

  Stale health data                   Logs validation warnings and
                                      applies prompt-level hedging

  Language violation                  Sanitizes, revalidates, downgrades
                                      confidence, or drops the insight
  -----------------------------------------------------------------------

## 24.1 LLM Failure

When the LLM invocation fails:

``` python
{
    "status": "error",
    "insight": None,
    "error": "..."
}
```

is returned.

## 24.2 Redis Failure

Redis is treated as an optimization layer rather than a hard dependency.

If Redis is unavailable:

``` text
Cache unavailable
      │
      ▼
Recompute Signals
      │
      ▼
Build Prompt
      │
      ▼
Invoke LLM
      │
      ▼
Return Response
```

The pipeline continues operating without cached data.

## 24.3 Stale Data

Health data older than one day may trigger warnings.

The system:

1.  Detects potential stale data.
2.  Logs validation warnings.
3.  Applies prompt rules that encourage hedged language.

The LLM should avoid presenting stale information as immediate real-time
data.

------------------------------------------------------------------------

# 25. Debug Pipeline

The debugging endpoint is:

``` text
POST /insight/debug
```

The underlying method is:

``` python
debug_pipeline()
```

The debug pipeline mirrors the production architecture.

## Layer 1 --- Extract

``` text
Raw Data
   │
   ▼
Canonicalization
   │
   ▼
field_keep_map Projection
```

Output:

``` text
s1_extract
```

## Layer 2 --- Signals

``` text
HealthInsightProcessor.process()
            │
            ▼
     ExtractedSignals
```

Output:

``` text
s2_signals
```

## Layer 3 --- Prompt

``` text
ExtractedSignals
       │
       ▼
_assemble_sylo_prompt()
       │
       ├── Narrative
       ├── Domain Group
       ├── Overview Question
       └── Messages
```

Output:

``` text
s3_prompt
```

## Layer 4 --- LLM

``` text
Prompt
   │
   ▼
Single Sylo Q&A Call
   │
   ▼
Parsed Answer
```

Output:

``` text
s4_llm
```

## Debug Output

The debug response includes:

``` text
s1_extract

s2_signals

s3_prompt

s4_llm

timings
```

It also exposes legacy compatibility aliases:

``` text
s3_narrative

s6_qa

s8_render

phase*_keys
```

------------------------------------------------------------------------

# 26. Backend Data Sources

The pipeline aggregates data from multiple backend services.

## 26.1 Health Data

  Endpoint                      Purpose
  ----------------------------- --------------------------------------
  `GET /api/health/summaries`   Today's health summary
  `GET /api/health/summaries`   Health parameters using a date range

Health data includes:

-   Energy
-   Heart rate
-   Sleep
-   Steps

## 26.2 Balance Data

  Endpoint                   Purpose
  -------------------------- ---------------------------------------------
  `GET /api/balance/score`   Current balance scores
  `GET /api/balance`         Historical balance data over a 30-day range

## 26.3 Calendar Data

  Endpoint                         Purpose
  -------------------------------- ------------------------------
  `GET /api/calendar/event`        Today's calendar events
  `GET /api/calendar/reminders`    Today's reminders
  `GET /api/calendar/work-hours`   User work-hour configuration

## 26.4 Mood Data

  Endpoint                  Purpose
  ------------------------- ------------------------
  `GET /api/moods/latest`   Latest recorded mood
  `GET /api/moods`          Seven-day mood history

## 26.5 Historical Snapshots

  Endpoint                          Purpose
  --------------------------------- ------------------------------------
  `GET /api/daily-user-snapshots`   Historical daily data over 30 days

## 26.6 Productivity Data

  Endpoint                            Purpose
  ----------------------------------- ----------------------------------
  `GET /api/productivity/summaries`   Seven-day productivity summaries
  `GET /api/productivity/summary`     Current-day productivity

## 26.7 User Profile Data

  --------------------------------------------------------------------------
  Endpoint                               Purpose
  -------------------------------------- -----------------------------------
  `GET /api/onboarding/users/profiles`   User profile and configured goals

  --------------------------------------------------------------------------

------------------------------------------------------------------------

# 27. Key File Map

``` text
insights/
│
├── services/
│   ├── advanced_insight_service.py
│   │   # Main orchestration service
│   │   # get_qa_single_insight()
│   │   # get_random_insight()
│   │
│   ├── insight_picker.py
│   │   # Insight pool management
│   │   # Random insight selection
│   │
│   └── base.py
│       # Base service interface
│
├── prompts/
│   ├── qa_prompts.py
│   │   # System prompts
│   │   # User prompt builders
│   │   # DOMAIN_GROUP_REGISTRY
│   │
│   ├── core_persona.py
│   │   # Core persona
│   │   # Language enforcement
│   │
│   ├── domain_rules.py
│   │   # Health rules
│   │   # Productivity rules
│   │   # Overall rules
│   │
│   ├── insight_guardrails.py
│   │   # NOT_DO constraints
│   │
│   └── context_narrative_templates.py
│       # Three-part context narrative builder
│
├── helpers/
│   ├── question_gate.py
│   │   # Question registry
│   │   # Post-filtering
│   │   # Relevance scoring
│   │
│   ├── insight_validator.py
│   │   # Hallucination validation
│   │   # Language sanitization
│   │
│   └── day_profile.py
│       # time_phase computation
│
├── health/
│   ├── extractor.py
│   │   # HealthDataExtractor
│   │   # Raw data collection
│   │
│   ├── health_processor/
│   │   ├── processor.py
│   │   │   # HealthInsightProcessor
│   │   │   # Main signal processing pipeline
│   │   │
│   │   ├── context/
│   │   │   └── processor.py
│   │   │       # ContextSignalBuilder
│   │   │       # sleep_window
│   │   │       # weekend
│   │   │       # work_hours
│   │   │       # recent_workout
│   │   │
│   │   ├── sleep/processor.py
│   │   ├── heart_rate/processor.py
│   │   ├── steps/processor.py
│   │   ├── energy/processor.py
│   │   ├── mood/processor.py
│   │   ├── health_score/processor.py
│   │   └── period_metrics.py
│   │
│   └── field_keep_maps.py
│       # Field projection rules
│
├── schemas/
│   ├── processed_context.py
│   │   # ExtractedSignals
│   │   # Signal block schemas
│   │
│   ├── insight_pool.py
│   │   # InsightPool schema
│   │
│   └── insight_item.py
│       # InsightItem schema
│
└── processors/
    ├── data_processor.py
    │   # DataProcessor.extract_signals()
    │   # Typed signal extraction
    │
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

------------------------------------------------------------------------

# 28. Architectural Characteristics

The current Insight Pipeline follows several key architectural
principles.

## Single LLM Invocation

Each production insight request performs:

``` text
1 LLM Call
```

All other processing is deterministic.

## Signal-First Architecture

The LLM does not directly process raw backend responses.

Instead:

``` text
Backend APIs
      │
      ▼
Raw Data Extraction
      │
      ▼
Signal Processing
      │
      ▼
Structured Signals
      │
      ▼
Context Narrative
      │
      ▼
LLM
```

This reduces the amount of raw data exposed to the LLM and provides a
more structured reasoning context.

## Domain Isolation

Signal caching is scoped by:

``` text
User
+
Domain
```

This prevents cross-domain signal reuse.

## Cache as an Optimization Layer

Redis improves latency and reduces repeated LLM calls but is not
required for pipeline correctness.

## Deterministic Context

Time-sensitive context signals are generated deterministically rather
than inferred by the LLM.

Examples include:

-   Sleep window
-   Recent workout
-   Work hours
-   Weekend status

## Multi-Layer Validation

Generated insights are validated for:

-   Unsupported numerical claims
-   Stale data
-   Invalid pattern durations
-   Calendar contradictions
-   Language consistency

------------------------------------------------------------------------

# 29. End-to-End Request Lifecycle

The complete production lifecycle can be summarized as follows:

``` text
Client Request
      │
      ▼
FastAPI Router
      │
      ▼
Domain-Specific Endpoint
      │
      ▼
AdvancedInsightService
      │
      ▼
Check Insight Cache
      │
      ├──────────── Cache HIT ────────────► Return Cached Insight
      │
      ▼
Signal Cache Check
      │
      ├──────────── Cache HIT ────────────┐
      │                                   │
      └──── Cache MISS                    │
                 │                        │
                 ▼                        │
          Parallel API Collection         │
                 │                        │
                 ▼                        │
          Signal Processing               │
                 │                        │
                 ▼                        │
          Context Signal Attachment       │
                 │                        │
                 └────────────────────────┘
                              │
                              ▼
                    ExtractedSignals
                              │
                              ▼
                  Build Context Narrative
                              │
                              ▼
                   Resolve Domain Question
                              │
                              ▼
                      Assemble Prompt
                              │
                              ▼
                       Bedrock LLM
                              │
                              ▼
                    Parse LLM Response
                              │
                              ▼
                  Hallucination Validation
                              │
                              ▼
                    Language Enforcement
                              │
                              ▼
                     Format Final Insight
                              │
                              ▼
                      Store in Cache
                              │
                              ▼
                       Return Response
```

------------------------------------------------------------------------

# 30. Summary

The Insight Pipeline is a **signal-driven, single-call LLM
architecture** designed to generate personalized insights across Health,
Productivity, and Overall domains.

Its core design principles are:

-   **Deterministic data processing before LLM invocation**
-   **Structured signal extraction**
-   **Domain-scoped caching**
-   **Single LLM call per request**
-   **Inline context narrative generation**
-   **Deterministic real-time context signals**
-   **Multi-layer hallucination protection**
-   **Four-layer language enforcement**
-   **Redis as an optimization layer rather than a hard dependency**

The architecture separates **data collection**, **signal reasoning**,
**context construction**, **LLM generation**, and **output validation**,
making the pipeline easier to debug, optimize, and extend.
