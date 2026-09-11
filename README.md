# AI Agents

An intelligent AI chatbot platform consisting of two core subsystems:

| Subsystem | Description | Location |
|---|---|---|
| **Chatbot Agent** | Multi-action conversational agent with real-time web search | `app/` |
| **Insight Pipeline** | Personalized health & productivity insights generation | `chatbot-backend/` |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
┌─────────────────────┐               ┌─────────────────────────┐
│   Chatbot Agent     │               │   Insight Pipeline       │
│   (app/)            │               │   (chatbot-backend/)     │
│                     │               │                         │
│ • Multi-Action      │               │ • Q&A Single Insight    │
│ • Web Search        │               │ • Random Insight Pool    │
│                     │               │ • Five-Focus Signals    │
└─────────────────────┘               └─────────────────────────┘
```

---

## 1. Chatbot Agent (`app/`)

### Overview

A LangGraph-based conversational agent that executes user requests through a unified tool ecosystem.

**Stack**: LangGraph → `HybridWorkflow` → LangChain ReAct Agent → Tool Executors → External APIs

### Core Features

#### Multi-Action System
Execute multiple tasks from a single user message with sequential confirmation.

```
User: "Schedule a meeting at 3pm and remind me to prepare slides"

→ Extract actions: [create_event, create_reminder]
→ Show plan confirmation
→ User confirms → Execute each action sequentially
→ After each action: Continue / Cancel buttons
→ Summarize all results
```

**Key Capabilities**:
- Intent analysis → separate actions for each distinct task
- Plan confirmation before execution
- Continue/Cancel quick replies after each action
- Redis-backed queue persistence (30min TTL)

#### Search Tool
Real-time web search via Tavily API with intelligent routing.

**Query Types**:
| Type | Use Case | Examples |
|---|---|---|
| `volatile` | Time-sensitive data | Gold prices, exchange rates, news |
| `stable` | Slowly changing data | Company profiles, definitions, laws |

**Features**:
- Source priority (Government → Research → Local → Social)
- Date integrity validation (reject stale data)
- Multi-language support (Vietnamese, English, etc.)

### Workflow Node Graph

```
check_quick_reply
    │
    ▼
check_pending_action ──→ detect_language
    │                        │
    ▼                        ▼
classify_intent ←──────────────────┐
    │                              │
    ▼                              │
extract_date_expressions ──────────┘ (parallel fan-out)
    │
    ▼
merge_preprocess (barrier)
    │
    ├── is_multi_action = True?
    │   │
    │   ▼
    │   extract_multi_actions → plan → execute → summarize
    │
    └── Normal flow
        │
        ├── pending_action → clear_pending / update_agent
        │
        └── execute_agent → validate_output → generate_response
```

### Middleware Chain

| Middleware | Purpose |
|---|---|
| `RealValueContextMiddleware` | Capture real entity IDs from tool responses |
| `AIToolValidationMiddleware` | Validate tool arguments before execution |
| `AIOutputValidationMiddleware` | Validate LLM output format & language |
| `ReasoningContentFilterMiddleware` | Strip internal reasoning from response |
| `ToolChoicesValidationMiddleware` | Ensure selected tool is in available list |
| `ToolBoundaryMiddleware` | Prevent tool→tool calls |
| `ToolProgressMiddleware` | Emit per-tool progress labels |
| `RetryModelCallMiddleware` | Retry on transient LLM failures |

### Quick Links

- **Detailed Documentation**: [`app/docs/multi-action-search-workflow.md`](app/docs/multi-action-search-workflow.md)
- **Main Workflow**: [`app/ai_agents/hybrid_workflow.py`](app/ai_agents/hybrid_workflow.py)
- **Multi-Action Core**: [`app/ai_agents/multi_action/`](app/ai_agents/multi_action/)
- **Search Tool**: [`app/ai_agents/tools/search_tools.py`](app/ai_agents/tools/search_tools.py)

---

## 2. Insight Pipeline (`chatbot-backend/`)

### Overview

A health & productivity insight generation system using AWS Bedrock LLM with multi-layered signal processing.

**Stack**: FastAPI → `AdvancedInsightService` → Bedrock LLM → Redis cache → External APIs

### Core Features

#### Q&A Single Insight (Primary)
Generate one personalized insight per request with minimal LLM calls.

```
Request → Cache check → Extract Signals (5 focuses) → Build narrative → LLM call → Guard → Return
                                                                                    (1 call total)
```

**Signal Architecture** (5 Focuses):
| Focus | Signals |
|---|---|
| **Sleep** | duration, efficiency, consistency, deep_sleep, rem_sleep, sleep_debt |
| **Heart Rate** | resting_hr, hrv, stability, recovery_state, baseline_comparison |
| **Steps** | activity_volume, goal_achievement, consistency, trend |
| **Energy** | activity_volume, energy_consistency, energy_trend |
| **Mood** | snapshot, baseline, trend, stability, frequency |

#### Random Insight Mode
Build an insight pool across all domains, pick randomly with deduplication.

**Flow**:
1. Extract signals (cached per domain)
2. Build narrative
3. Run ALL question groups × ALL domains via LLM
4. Filter by confidence (≥0.3), deduplicate
5. Pick random from unseen candidates

#### Hallucination Guard (4 Layers)
| Layer | Mechanism |
|---|---|
| 1 | Prompt rules in system prompt |
| 2 | Post-generation validator |
| 3 | Hard sanitizer (replace English tokens) |
| 4 | Confidence downgrade → drop if dirty |

#### Period Aggregates
Compare today vs week/month/baseline for:
- Steps, sleep_hours, resting_heart_rate
- Active minutes, workout minutes, calories burned

### Response Format

```python
# Single Insight
{
    "status": "success",
    "insight": {
        "insight": "Your energy levels have been consistently...",
        "question_id": "h_overview_infer",
        "domain": "health",
        "tone": "encouraging",
        "confidence": 0.85
    }
}

# Random Insight
{
    "status": "success",
    "insight": {
        "point1": "...",
        "domain": "health",
        "current_state": "...",
        "cause": "...",
        "action": "...",
        "evidence": "..."
    }
}
```

### API Endpoints

| Endpoint | Description |
|---|---|
| `POST /insight/productivity` | Get productivity insight |
| `POST /insight/health_insight` | Get health insight |
| `POST /insight/overall_insight` | Get overall insight |
| `POST /insight/random_insight` | Get random insight from pool |
| `POST /insight/debug` | Full pipeline debug dump |

### Cache Inventory

| Key | TTL | Description |
|---|---|---|
| `qa_insight:{uid}:{domain}:{lang}` | 300s | Primary insight cache |
| `extracted_signals:pipeline_v3:{uid}:{domain}` | 300s | Domain-scoped signals |
| `insight_pool:v2:{uid}` | — | Random insight pool |
| `insight_seen:v2:{uid}` | — | Seen tracking |

### Quick Links

- **Detailed Documentation**: [`chatbot-backend/docs/workflow-v2.md`](chatbot-backend/docs/workflow-v2.md)
- **Main Service**: [`chatbot-backend/insights/services/advanced_insight_service.py`](chatbot-backend/insights/services/advanced_insight_service.py)
- **Signal Processors**: [`chatbot-backend/insights/health/health_processor/`](chatbot-backend/insights/health/health_processor/)
- **Prompts**: [`chatbot-backend/insights/prompts/`](chatbot-backend/insights/prompts/)

---

## 3. Project Structure

```
AI-Agents/
├── app/                                # Chatbot Agent
│   ├── main.py                         # FastAPI entrypoint
│   ├── README.md
│   └── ai_agents/
│       ├── hybrid_workflow.py          # Main LangGraph orchestration
│       ├── multi_action/               # Multi-action system
│       │   ├── multi_action_extractor.py
│       │   ├── multi_action_prompt.py
│       │   ├── multi_action_store.py
│       │   ├── suggestion_to_actions.py
│       │   └── draft_edit.py
│       ├── tools/                      # Tool ecosystem
│       │   ├── search_tools.py
│       │   ├── validate/               # Argument & pending-action validation
│       │   │   ├── pending_action_manager.py
│       │   │   └── prompts/
│       │   │       └── search_prompt.py
│       │   └── executor/
│       │       ├── web_search_executor.py
│       │       └── search_providers/
│       │           ├── base_provider.py
│       │           ├── tavily_provider.py
│       │           ├── aws_kendra_provider.py
│       │           └── google_provider.py
│       └── middleware/
│           └── ai_middleware.py        # Middleware chain (8 middlewares)
│
├── chatbot-backend/                    # Insight Pipeline
│   ├── main.py                         # FastAPI entrypoint
│   ├── README.md
│   ├── requirements.txt
│   ├── routers/                        # HTTP route handlers
│   │   ├── insight_router.py
│   │   └── health_router.py
│   ├── agents/                         # LLM orchestration & auditing
│   │   ├── llm_manager.py
│   │   ├── llm_helper.py
│   │   ├── prompt.py
│   │   ├── prompt_coherence_auditor.py
│   │   └── insight_coherence_auditor.py
│   ├── clients/                        # External service clients
│   │   ├── redis_client.py
│   │   └── rabbitmq_client.py
│   ├── models/                         # Pydantic data models
│   ├── modules/
│   │   └── data_collector.py
│   ├── services/                       # Domain services & data prep
│   │   ├── service_factory.py
│   │   ├── insight_service.py
│   │   ├── health_data_service.py
│   │   ├── init_session_service.py
│   │   ├── external_api_service.py
│   │   └── executor/
│   │       ├── cache_helpers.py
│   │       ├── constant.py
│   │       ├── prepare_calendar_data.py
│   │       ├── prepare_health_data.py
│   │       └── prepare_time_data.py
│   └── insights/                       # Insight generation core
│       ├── data_processor.py
│       ├── insight_config.py
│       ├── rule_guards.yaml
│       ├── services/                   # Insight services
│       │   ├── advanced_insight_service.py   # Q&A + Random insight orchestrator
│       │   ├── base.py
│       │   ├── insight_picker.py
│       │   ├── financial_base.py
│       │   ├── financial_executor.py
│       │   ├── monthly_base.py
│       │   └── monthly_executor.py
│       ├── prompts/                    # LLM prompts & guardrails
│       │   ├── core_persona.py
│       │   ├── qa_prompts.py
│       │   ├── domain_rules.py
│       │   ├── context_narrative_templates.py
│       │   └── insight_guardrails.py
│       ├── schemas/                    # Pydantic schemas
│       │   ├── insight_pool.py
│       │   ├── insight_item.py
│       │   └── processed_context.py
│       ├── helpers/                    # Validation & utility helpers
│       │   ├── insight_validator.py
│       │   ├── question_gate.py
│       │   ├── empty_value_cleanser.py
│       │   └── day_profile.py
│       ├── processors/                 # Productivity signal processors
│       │   ├── productivity_signal_processor.py
│       │   ├── productivity_calendar_processor.py
│       │   ├── productivity_reminder_processor.py
│       │   ├── balance_snapshot_processor.py
│       │   ├── goal_progress_processor.py
│       │   ├── work_hours_processor.py
│       │   ├── finance_signal_processor.py
│       │   ├── historical_trends_processor.py
│       │   ├── calendar_intelligence_processor.py
│       │   ├── behavioral_pattern_processor.py
│       │   └── reminder_signal_processor.py
│       └── health/                     # Health signal extraction
│           ├── extractor.py
│           ├── field_keep_maps.py
│           ├── canonical_field_mapping.py
│           └── health_processor/       # Per-focus signal processors
│               ├── processor.py
│               ├── signal_processor.py
│               ├── period_metrics.py
│               ├── common/
│               ├── context/
│               ├── sleep/
│               ├── heart_rate/         # Includes legacy_cardio.py
│               ├── steps/
│               ├── energy/
│               ├── mood/
│               └── health_score/
│
└── chatbot-frontend/                   # User interface (separate repo)
```

---

## 4. Technology Stack

### Chatbot Agent
| Component | Technology |
|---|---|
| Orchestration | LangGraph |
| Agent | LangChain ReAct |
| LLM | AWS Bedrock |
| Cache | Redis |
| Search | Tavily API |
| Language | Python 3.11+ |

### Insight Pipeline
| Component | Technology |
|---|---|
| API | FastAPI |
| LLM | AWS Bedrock |
| Cache | Redis |
| Data Processing | Pydantic |
| Language | Python 3.11+ |

