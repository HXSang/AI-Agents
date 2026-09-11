# Chatbot Agent — Workflow Reference
> **Stack**: LangGraph → `HybridWorkflow` (LangChain ReAct Agent + LLM) → Tool Executors → External APIs (Tavily Search, Calendar, Health, Finance, etc.)

---

## 1. Overview

Hệ thống chatbot agent trong `app/` bao gồm **2 feature chính**:

| Feature | Mô tả | Entry point |
|---|---|---|
| **Multi-Action** | Thực thi nhiều hành động liên tiếp từ 1 câu prompt | `HybridWorkflow` → `_extract_multi_actions_node` |
| **Search Tool** | Web search thời gian thực qua Tavily API | `web_search` tool → `WebSearchToolExecutor` |

---

## 2. Multi-Action System

### 2.1 Tổng quan

Multi-action cho phép user yêu cầu **nhiều tác vụ trong 1 tin nhắn** (ví dụ: "tạo lịch họp 3h chiều và nhắc tôi chuẩn bị slide"). Hệ thống sẽ:

1. Phân tích intent → trích xuất danh sách actions
2. Hiển thị **plan confirmation** (danh sách actions)
3. User xác nhận → thực thi tuần tự từng action
4. Mỗi action xong → hỏi Continue/Cancel
5. Tất cả xong → tổng hợp kết quả

### 2.2 Luồng chi tiết

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
execute_single_action() ──→ Thực thi từng action
    │
    ├── _run_agent() ──→ LangChain ReAct agent
    │   ├── Middleware chain (validation, handoff, retry)
    │   └── Tool execution
    │
    ├── Lưu result vào complete_action_queue
    │
    ├── Nếu còn action → generate_connective_tissue() → hỏi Continue/Cancel
    │
    └── Nếu queue empty → summarize_multi_actions() → tổng hợp kết quả
```

### 2.3 Quick Reply Buttons

Sau mỗi action, hệ thống hiển thị 2 buttons:

| Button | Value | Hành động |
|---|---|---|
| ✅ Continue | `MULTI_ACTION_CONTINUE` | Thực thi action tiếp theo |
| ❌ Cancel | `MULTI_ACTION_CANCEL` | Hủy toàn bộ queue |

### 2.4 Action Execution States

| State | Mô tả | Behavior |
|---|---|---|
| `success` | Action hoàn thành | Pop khỏi queue, thêm vào completed |
| `need_confirmation` | Chờ user xác nhận | **Tạm dừng** queue, hiển thị preview |
| `cancelled` | User hủy action | Skip action, chuyển sang action tiếp theo |
| `error` | Lỗi execution | Thông báo lỗi, dừng queue |

### 2.5 Redis Cache Keys (Multi-Action)

| Key | TTL | Mô tả |
|---|---|---|
| `user:{uid}:session:{sid}:multi_action:draft` | 1800s | Draft queue chờ user confirm |
| `user:{uid}:session:{sid}:multi_action:pending` | 1800s | Queue đang chờ thực thi |
| `user:{uid}:session:{sid}:multi_action:completed` | 1800s | Actions đã hoàn thành |

### 2.6 Key Files

```
app/ai_agents/
├── multi_action/
│   ├── multi_action_extractor.py     # Core: extract_multi_actions, execute_single_action
│   ├── multi_action_prompt.py        # Prompts: plan, summary, connective tissue
│   ├── multi_action_store.py         # Redis CRUD cho queues
│   ├── suggestion_to_actions.py      # Convert smart_suggestions → action queue
│   └── draft_edit.py                 # Free-text edit flow
├── hybrid_workflow.py                 # LangGraph state machine orchestration
└── middleware/ai_middleware.py       # Tool validation middleware chain
```

### 2.7 Smart Suggestions → Multi-Action

Khi user dùng `smart_suggestions` (plan mode), hệ thống tự động convert suggestions thành action queue:

```
smart_suggestions (plan mode)
    │
    ▼
build_action_queue_from_suggestions()
    │
    ├── create_event → batch vào 1 create_event_by_name(events=[...])
    ├── update_event → update_event_by_name
    ├── delete_event → delete_event_by_name
    ├── set_health_goal → set_health_goal
    ├── create_finance_log → create_finance_logs
    └── create_reminder → create_reminder
```

### 2.8 Validation Rules (extract_operations_prompt)

| Rule | Mô tả |
|---|---|
| R1 | Mỗi intent → 1 action riêng |
| R2 | `tool_name` phải match đúng tên tool |
| R3 | "rưỡi" = :30 (7r = 07:30) |
| R4 | Không guess giờ nếu user chỉ nói "sáng"/"chiều" |
| R5 | Recurring events → `tool_name: "none"` (unsupported) |
| R6 | Suggest actions → `web_search`, không phải "none" |
| R7 | Self-contained content → "none" |

---

## 3. Search Tool System

### 3.1 Tổng quan

Search tool cho phép agent tìm kiếm web thời gian thực để trả lời các câu hỏi về:

- 💰 Giá vàng, tỷ giá, chứng khoán (volatile)
- 📰 Tin tức, sự kiện (volatile)
- 🏢 Thông tin công ty, nhân vật (stable)
- 📋 Hướng dẫn, định nghĩa, luật (stable)

### 3.2 Luồng chi tiết

```
User Message (yêu cầu search)
    │
    ▼
HybridWorkflow._execute_agent_node()
    │
    ▼
create_agent() với tools = [...] + web_search
    │
    ▼
Agent decides to call web_search tool
    │
    ▼
web_search(search_query, query_type)
    │
    ├── query_type = "volatile" → thời gian nhạy cảm (prices, news)
    └── query_type = "stable" → ít thay đổi (people, facts)
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

| Query Type | Khi nào | Tavily time_range | Date trong query |
|---|---|---|---|
| `volatile` | Giá, tỷ giá, tin tức, thời tiết | day → week | ❌ Không append |
| `stable` | Người, công ty, định nghĩa | None (full scan) | ❌ Không append |

### 3.4 Source Priority (Tavily Results)

| Tier | Nguồn | Ví dụ |
|---|---|---|
| **Tier 1** | Government & official | .gov, .org, ngân hàng trung ương |
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

Search tool **không cache** kết quả search vì:
- Dữ liệu volatile → cache stale nhanh
- Mỗi query unique → ít repeat potential
- Tavily đã có internal rate limiting

### 3.7 Key Files

```
app/ai_agents/tools/
├── search_tools.py                        # Tool definition
└── executor/
    ├── web_search_executor.py            # Orchestration logic
    └── search_providers/
        ├── base_provider.py              # Abstract interface
        ├── tavily_provider.py            # Tavily API implementation
        ├── google_provider.py            # Google Search (unused)
        ├── aws_kendra_provider.py        # AWS Kendra (unused)
        └── __init__.py
app/ai_agents/tools/validate/prompts/
└── search_prompt.py                      # Query formulation + synthesis prompts
```

### 3.8 Search Tool Tooltip (Agent nhận)

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

Search tool có thể được extract như một action trong multi-action flow:

```
User: "Check gold price then remind me to buy at 3pm"

extract_operations() → [
  {
    "action_id": "action_1",
    "tool_name": "web_search",
    "arguments": {
      "search_query": "giá vàng hôm nay",
      "query_type": "volatile"
    }
  },
  {
    "action_id": "action_2",
    "tool_name": "create_reminder",
    "arguments": {
      "title": "Mua vàng",
      "due_date": "2026-09-11T15:00"
    }
  }
]
```

---

## 4. Tool Ecosystem

### 4.1 All Available Tools

| Category | Tools |
|---|---|
| **Calendar** | `create_event_by_name`, `update_event_by_name`, `delete_event_by_name`, `sync_event_by_name` |
| **Reminder** | `create_reminder`, `update_reminder`, `delete_reminder`, `mark_off_reminder` |
| **Health** | `set_health_goal`, `get_health_goals`, `get_health_metrics` |
| **Finance** | `create_finance_logs`, `get_finance_summary` |
| **Balance** | `get_balance_scores` |
| **Productivity** | `get_productivity_summary`, `get_tasks` |
| **Company Info** | `get_company_info` |
| **Profile** | `get_user_profile`, `get_goals`, `get_calendar_preferences` |
| **Search** | `web_search` |
| **Smart Suggestions** | `get_smart_suggestions` |

### 4.2 Tool Creation Pattern

```python
def create_search_tools(
    user_id: str,
    llm: Any,
    session_id: Optional[str] = None,
    user_data: Optional[Dict[str, Any]] = None,
    timezone: Optional[str] = None,
    language: Optional[str] = None,
) -> List[BaseTool]:
    # 1. Initialize executor
    executor = WebSearchToolExecutor(
        user_id=user_id, llm=llm, session_id=session_id, today_str=today_str
    )
    
    # 2. Build tool description
    tool_description = create_query_formulation_prompt(...)
    
    # 3. Define tool with @tool decorator
    @tool(description=tool_description)
    async def web_search(search_query: str, query_type: str) -> str:
        return await executor.execute_search(search_query, user_lang, query_type)
    
    return [web_search]
```

---

## 5. Hybrid Workflow (Orchestration)

### 5.1 State Definition

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
    
    # Smart suggestions
    smart_suggestions_data: Optional[Dict[str, Any]]
    
    # Output
    response: str
    quick_reply_buttons: Optional[List[Dict[str, str]]]
    error: Optional[str]
```

### 5.2 Node Graph

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

### 5.3 Middleware Chain (Agent Execution)

| Middleware | Purpose |
|---|---|
| `RealValueContextMiddleware` | Capture real entity IDs from tool responses |
| `AIToolValidationMiddleware` | Validate tool arguments before execution |
| `AIOutputValidationMiddleware` | Validate LLM output format & language |
| `ReasoningContentFilterMiddleware` | Strip internal reasoning from response |
| `ToolChoicesValidationMiddleware` | Ensure selected tool is in available list |
| `ToolBoundaryMiddleware` | Prevent tool→tool calls (single hop only) |
| `SuggestionHandoffMiddleware` | Capture smart_suggestions plan mode output |
| `ToolProgressMiddleware` | Emit per-tool progress labels |
| `RetryModelCallMiddleware` | Retry on transient LLM failures |

---

## 6. Constants & Configuration

### 6.1 Quick Reply Values

```python
class QuickReply:
    MULTI_ACTION_CONTINUE = "multi_action_continue"
    MULTI_ACTION_CANCEL = "multi_action_cancel"
    LABEL_KEY_CONFIRM = "confirm"
    LABEL_KEY_CANCEL = "cancel"
```

### 6.2 Tool Status

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

### 6.3 Configuration

```python
# app/config.py
class Settings:
    search_max_results: int = 10          # Tavily max results per query
    response_cache_ttl: int = 300          # Response cache TTL in seconds
```

---

## 7. Integration Points

### 7.1 Multi-Action ↔ Smart Suggestions

```
smart_suggestions (plan mode)
    │
    ▼ SuggestionHandoffMiddleware captures suggestion_envelope
    │
    ▼ _route_after_agent() detects mode="plan"
    │
    ▼ extract_multi_actions_node()
    │   └── build_suggestion_handoff() → action queue
    │
    ▼ execute_single_action() with nested suggestions
```

### 7.2 Search ↔ Multi-Action

```
Multi-action queue với web_search
    │
    ▼ execute_single_action()
    │   └── web_search(search_query, query_type)
    │       └── TavilyProvider.search()
    │
    ▼ Result → connective_tissue → next action
```

---

## 8. Key Prompt Templates

### 8.1 Multi-Action Extract Prompt

Key rules trong `extract_operations_prompt()`:
- Every distinct intent → separate action
- tool_name MUST match exact tool name
- SEARCH RULE 1: General research → NO date
- SEARCH RULE 2: Time-sensitive → MUST append current date
- VIETNAMESE TIME: 'rưỡi' = :30, 'kém' = before hour
- RECURRING → tool_name: "none" (unsupported)

### 8.2 Plan Confirmation Prompt

Generates friendly bulleted list with:
- Future tense ("Mình sẽ...")
- Emoji per bullet
- One confirm question ("Mình tiến hành nhé?")

### 8.3 Connective Tissue Prompt

Generates transition between actions:
- React to current result
- Flow naturally to next action
- One specific question about next step

### 8.4 Search Query Formulation Prompt

Key rules:
- volatile → NO date appended, generic terms ("hôm nay")
- stable → NO date, full scan
- Follow-up with citations → NEW search required
- Location: local → append location, global → no location

---

## 9. Failure Modes

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

## 10. Example Flows

### 10.1 Simple Multi-Action

```
User: "Tạo lịch họp với An lúc 3h chiều và nhắc tôi chuẩn bị slide"

LLM Extract → [
  {action_id: "action_1", tool_name: "create_event_by_name", arguments: {...}},
  {action_id: "action_2", tool_name: "create_reminder", arguments: {...}}
]

Plan → "📅 Mình sẽ tạo lịch họp với An lúc 15:00
       ⏰ Sau đó nhắc bạn chuẩn bị slide
       Tiến hành nhé?"

User clicks Continue
  → execute_single_action(action_1) → calendar → success
  → "Lịch họp đã tạo! Giờ mình nhắc bạn chuẩn bị slide nhé?"
  
User clicks Continue
  → execute_single_action(action_2) → reminder → success

Summary → "✅ Hoàn thành! Lịch họp với An lúc 15:00 và reminder đã sẵn sàng."
```

### 10.2 Search + Multi-Action

```
User: "Check giá vàng SJC hôm nay, nếu dưới 100 triệu thì nhắc tôi mua"

LLM Extract → [
  {action_id: "action_1", tool_name: "web_search", arguments: {
    search_query: "giá vàng SJC hôm nay", query_type: "volatile"
  }}
]

Plan → "🔍 Mình sẽ kiểm tra giá vàng SJC hôm nay"

User clicks Continue
  → web_search() → Tavily → returns "Giá vàng SJC: 98.5 triệu/lượng"
  → Check condition: 98.5 < 100 → TRUE
  → create_reminder → due_date = today 17:00
  
Summary → "✅ Giá vàng SJC hiện 98.5 triệu — dưới 100 triệu! Đã tạo reminder cho bạn."
```

### 10.3 Cancel Mid-Flow

```
User clicks Cancel after action_1 completes

→ process_multi_action_cancel()
  → Pop action_2 from queue
  → action_2.status = "cancelled"
  → build_cancel_response() → "Đã hủy tạo reminder. Lịch họp vẫn giữ nguyên nhé!"
```
