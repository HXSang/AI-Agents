# Chatbot Agent — Workflow Reference
> **Stack**: LangGraph → `HybridWorkflow` (LangChain ReAct Agent + LLM) → Tool Executors → External APIs (Tavily Search, Calendar, Health, Finance, etc.)

---

## 1. Overview

The chatbot agent system in `app/` comprises **2 core features**:

| Feature | Description | Entry point |
|---|---|---|
| **Multi-Action** | Execute multiple actions from a single prompt | `HybridWorkflow` → `_extract_multi_actions_node` |
| **Search Tool** | Real-time web search via Tavily API | `web_search` tool → `WebSearchToolExecutor` |

---

## 2. Multi-Action System

### 2.1 Overview

Multi-action allows users to request **multiple tasks in one message** (e.g., "schedule a meeting at 3pm and remind me to prepare slides"). The system will:

1. Analyze intent → extract a list of actions
2. Show **plan confirmation** (action list)
3. User confirms → execute each action sequentially
4. After each action → ask Continue/Cancel
5. All done → summarize results

### 2.2 Detailed Flow

```
User Message
    │
    ▼
_classify_intent_node() ──→ is_multi_action = True?
    │
    ▼
_extract_multi_actions_node()
    │
    ├── extract_operations_prompt() ──→ LLM extract JSON {operations[]}
    │   ├── action_id: "action_1", "action_2", ...
    │   ├── tool_name: exact tool name
    │   └── arguments: {param: value}
    │
    ▼
plan_multi_actions() ──→ LLM generate plan confirmation message
    │
    ▼
[User confirms via Continue quick_reply]
    │
    ▼
execute_single_action() ──→ Execute each action sequentially
    │
    ├── _run_agent() ──→ LangChain ReAct agent
    │   ├── Middleware chain (validation, handoff, retry)
    │   └── Tool execution
    │
    ├── Save result to complete_action_queue
    │
    ├── If more actions → generate_connective_tissue() → ask Continue/Cancel
    │
    └── If queue empty → summarize_multi_actions() → summarize results
```

### 2.3 Quick Reply Buttons

After each action, the system displays 2 buttons:

| Button | Value | Behavior |
|---|---|---|
| ✅ Continue | `MULTI_ACTION_CONTINUE` | Execute next action |
| ❌ Cancel | `MULTI_ACTION_CANCEL` | Cancel entire queue |

### 2.4 Action Execution States

| State | Description | Behavior |
|---|---|---|
| `success` | Action completed | Pop from queue, add to completed |
| `need_confirmation` | Awaiting user confirmation | **Pause** queue, show preview |
| `cancelled` | User cancelled action | Skip action, move to next |
| `error` | Execution error | Report error, stop queue |

### 2.5 Redis Cache Keys (Multi-Action)

| Key | TTL | Description |
|---|---|---|
| `user:{uid}:session:{sid}:multi_action:draft` | 1800s | Draft queue awaiting user confirmation |
| `user:{uid}:session:{sid}:multi_action:pending` | 1800s | Queue currently awaiting execution |
| `user:{uid}:session:{sid}:multi_action:completed` | 1800s | Completed actions |

### 2.6 Key Files

```
app/ai_agents/
├── multi_action/
│   ├── multi_action_extractor.py     # Core: extract_multi_actions, execute_single_action
│   ├── multi_action_prompt.py       # Prompts: plan, summary, connective tissue
│   ├── multi_action_store.py        # Redis CRUD for queues
│   └── draft_edit.py                # Free-text edit flow
├── hybrid_workflow.py                # LangGraph state machine orchestration
└── middleware/ai_middleware.py      # Tool validation middleware chain
```

### 2.7 Validation Rules (extract_operations_prompt)

| Rule | Description |
|---|---|
| R1 | Each distinct intent → separate action |
| R2 | `tool_name` must match exact tool name |
| R3 | "rưỡi" = :30 (7r = 07:30) |
| R4 | Don't guess time if user only said "morning"/"afternoon" |
| R5 | Recurring events → `tool_name: "none"` (unsupported) |
| R6 | Suggest actions → `web_search`, not "none" |
| R7 | Self-contained content → "none" |

---

## 3. Search Tool System

### 3.1 Overview

The search tool allows the agent to perform real-time web searches to answer questions about:

- 💰 Gold prices, exchange rates, stocks (volatile)
- 📰 News, events (volatile)
- 🏢 Company info, person profiles (stable)
- 📋 Guides, definitions, laws (stable)

### 3.2 Detailed Flow

```
User Message (search request)
    │
    ▼
HybridWorkflow._execute_agent_node()
    │
    ▼
create_agent() with tools = [...] + web_search
    │
    ▼
Agent decides to call web_search tool
    │
    ▼
web_search(search_query, query_type)
    │
    ├── query_type = "volatile" → time-sensitive (prices, news)
    └── query_type = "stable" → slowly changing (people, facts)
    │
    ▼
WebSearchToolExecutor.execute_search()
    │
    ├── TavilyProvider.search(query, query_type)
    │   ├── volatile → time_range="day" → fallback "week"
    │   └── stable → time_range=None (full scan)
    │
    ├── _format_raw_results() → clean context string
    │
    ├── build_search_synthesis_prompt() → LLM extraction prompt
    │
    └── LLM synthesize → validated answer
    │
    ▼
get_agent_directive() → main agent render response
```

### 3.3 Dynamic Routing (Query Type)

| Query Type | When to Use | Tavily time_range | Date in Query |
|---|---|---|---|
| `volatile` | Prices, exchange rates, news, weather | day → week | ❌ Do NOT append |
| `stable` | People, companies, definitions | None (full scan) | ❌ Do NOT append |

### 3.4 Source Priority (Tavily Results)

| Tier | Source | Examples |
|---|---|---|
| **Tier 1** | Government & official | .gov, .org, central banks |
| **Tier 2** | Established research & media | Peer-reviewed, major news |
| **Tier 3** | Local reputable sources | Local retailers, finance sites |
| **Tier 4** | Social media (fallback only) | YouTube, Facebook |

### 3.5 Date Integrity Check (Volatile Data)

```
Today: 2026-09-11

Source Date    → Classification
─────────────────────────────────
2026-09-11    → ✅ Valid (today's data)
2026-09-10    → ⚠️ Case B (yesterday - acceptable with label)
2026-09-09    → ❌ Case A (stale - 2+ days old)
2025-09-11    → ❌ Case A (different year)
```

### 3.6 Redis Cache Keys (Search)

The search tool **does not cache** search results because:
- Volatile data → cache goes stale quickly
- Each query is unique → low repeat potential
- Tavily already has internal rate limiting

### 3.7 Key Files

```
app/ai_agents/tools/
├── search_tools.py                        # Tool definition
└── executor/
    ├── web_search_executor.py             # Orchestration logic
    └── search_providers/
        ├── base_provider.py               # Abstract interface
        ├── tavily_provider.py            # Tavily API implementation
        ├── google_provider.py             # Google Search (unused)
        ├── aws_kendra_provider.py        # AWS Kendra (unused)
        └── __init__.py
app/ai_agents/tools/validate/prompts/
└── search_prompt.py                      # Query formulation + synthesis prompts
```

### 3.8 Search Tool Tooltip (Agent Receives)

```
"Search the web for real-time or current information.

ROUTING PRINCIPLE — call this tool whenever answering requires
knowing what actually happened or currently exists in the external world.

QUERY TYPE CLASSIFICATION:
- "volatile" (Tier 1): market prices, exchange rates, gold/crypto/stock prices,
  today's news, weather — data that changes daily. DO NOT append dates.
- "stable" (Tier 2/3): people's roles, company profiles, historical facts,
  definitions, laws — slow-changing data. MUST NOT append dates."
```

### 3.9 Search Integration with Multi-Action

The search tool can be extracted as an action in the multi-action flow:

```
User: "Check gold price then remind me to buy at 3pm"

extract_operations() → [
  {
    "action_id": "action_1",
    "tool_name": "web_search",
    "arguments": {
      "search_query": "gold price today",
      "query_type": "volatile"
    }
  },
  {
    "action_id": "action_2",
    "tool_name": "create_reminder",
    "arguments": {
      "title": "Buy gold",
      "due_date": "2026-09-11T15:00"
    }
  }
]
```

---

## 4. Hybrid Workflow (Orchestration)

### 4.1 State Definition

```python
class HybridState(TypedDict):
    # Input
    user_message: str
    chat_history: List[Dict[str, str]]
    user_id: str
    session_id: str

    # Language detection
    language_code: str
    language_name: str

    # Intent classification
    intent: Optional[Dict[str, Any]]

    # Multi-action
    action_queue: Optional[List[Dict[str, Any]]]
    complete_action_queue: Optional[List[Dict[str, Any]]]
    multi_action_mode: bool

    # Output
    response: str
    quick_reply_buttons: Optional[List[Dict[str, str]]]
    error: Optional[str]
```

### 4.2 Node Graph

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
    │   extract_multi_actions
    │   │
    │   ├── plan_multi_actions() → show plan
    │   │
    │   └── execute_single_action() → loop
    │
    └── Normal flow
        │
        ├── pending_action → clear_pending / update_agent
        │
        └── execute_agent → validate_output
```

### 4.3 Middleware Chain (Agent Execution)

| Middleware | Purpose |
|---|---|
| `RealValueContextMiddleware` | Capture real entity IDs from tool responses |
| `AIToolValidationMiddleware` | Validate tool arguments before execution |
| `AIOutputValidationMiddleware` | Validate LLM output format & language |
| `ReasoningContentFilterMiddleware` | Strip internal reasoning from response |
| `ToolChoicesValidationMiddleware` | Ensure selected tool is in available list |
| `ToolBoundaryMiddleware` | Prevent tool→tool calls (single hop only) |
| `ToolProgressMiddleware` | Emit per-tool progress labels |
| `RetryModelCallMiddleware` | Retry on transient LLM failures |

---

## 5. Constants & Configuration

### 5.1 Quick Reply Values

```python
class QuickReply:
    MULTI_ACTION_CONTINUE = "multi_action_continue"
    MULTI_ACTION_CANCEL = "multi_action_cancel"
    LABEL_KEY_CONFIRM = "confirm"
    LABEL_KEY_CANCEL = "cancel"
```

### 5.2 Tool Status

```python
class ToolStatus:
    SUCCESS = "success"
    ERROR = "error"
    NEED_CONFIRMATION = "need_confirmation"
    MISSING = "missing"
    SUGGEST = "suggest"
    EDITING = "editing"
    OVERLAPPING = "overlapping"
```

### 5.3 Configuration

```python
# app/config.py
class Settings:
    search_max_results: int = 10          # Tavily max results per query
    response_cache_ttl: int = 300         # Response cache TTL in seconds
```

---

## 6. Integration Points

```
Multi-action queue with web_search
    │
    ▼ execute_single_action()
    │   └── web_search(search_query, query_type)
    │       └── TavilyProvider.search()
    │
    ▼ Result → connective_tissue → next action
```

---

## 7. Key Prompt Templates

### 7.1 Multi-Action Extract Prompt

Key rules in `extract_operations_prompt()`:
- Every distinct intent → separate action
- tool_name MUST match exact tool name
- SEARCH RULE 1: General research → NO date
- SEARCH RULE 2: Time-sensitive → MUST append current date
- VIETNAMESE TIME: 'rưỡi' = :30, 'kém' = before hour
- RECURRING → tool_name: "none" (unsupported)

### 7.2 Plan Confirmation Prompt

Generates friendly bulleted list with:
- Future tense ("I will...")
- Emoji per bullet
- One confirm question ("Shall we proceed?")

### 7.3 Connective Tissue Prompt

Generates transition between actions:
- React to current result
- Flow naturally to next action
- One specific question about next step

### 7.4 Search Query Formulation Prompt

Key rules:
- volatile → NO date appended, generic terms ("today")
- stable → NO date, full scan
- Follow-up with citations → NEW search required
- Location: local → append location, global → no location

---

## 8. Failure Modes

| Scenario | Behavior |
|---|---|
| LLM extract fails | Return `multi_action_mode: False`, single action fallback |
| Redis down | Queue operations fail → user sees error |
| Tavily no results | Return "No reliable information found" |
| Stale volatile data | Case B: show with date label |
| Wrong topic/volatile stale | Case A: return no results |
| Tool not found in multi-action | Fallback to full tool list |
| Pending confirmation timeout | Queue stays in Redis (1800s TTL) |

---

## 9. Example Flows

### 9.1 Simple Multi-Action

```
User: "Schedule a meeting with An at 3pm and remind me to prepare slides"

LLM Extract → [
  {action_id: "action_1", tool_name: "create_event_by_name", arguments: {...}},
  {action_id: "action_2", tool_name: "create_reminder", arguments: {...}}
]

Plan → "📅 I'll schedule a meeting with An at 15:00
       ⏰ Then remind you to prepare slides
       Shall we proceed?"

User clicks Continue
  → execute_single_action(action_1) → calendar → success
  → "Meeting created! Now let me set up your slide prep reminder?"

User clicks Continue
  → execute_single_action(action_2) → reminder → success

Summary → "✅ Done! Meeting with An at 15:00 and reminder are all set."
```

### 9.2 Search + Multi-Action

```
User: "Check SJC gold price today, if under 100 million remind me to buy"

LLM Extract → [
  {action_id: "action_1", tool_name: "web_search", arguments: {
    search_query: "SJC gold price today", query_type: "volatile"
  }}
]

Plan → "🔍 I'll check today's SJC gold price"

User clicks Continue
  → web_search() → Tavily → returns "SJC gold price: 98.5 million/tael"
  → Check condition: 98.5 < 100 → TRUE
  → create_reminder → due_date = today 17:00

Summary → "✅ SJC gold is at 98.5 million — below 100 million! Reminder created."
```

### 9.3 Cancel Mid-Flow

```
User clicks Cancel after action_1 completes

→ process_multi_action_cancel()
  → Pop action_2 from queue
  → action_2.status = "cancelled"
  → build_cancel_response() → "Reminder cancelled. Your meeting is still set!"
```
