# Insight Pipeline V3 — Workflow Reference

> **Status**: Luồng hiện tại là **Q&A Single Insight** (`get_qa_single_insight`).
> **Source root**: `/Users/sanghx/test/AI-Agents/chatbot-backend`
> **Stack**: FastAPI → `AdvancedInsightService` (Bedrock LLM via `LLMManager`) → Redis cache → `ExternalAPIServiceImpl` (BE pipeline).

---

## 1. Entry Points

3 endpoints trong `routers/insight_router.py`:

| Endpoint | Request schema | Response schema | Service call |
|---|---|---|---|
| `POST /insight/productivity` | `ProductivityInsightRequest` | `ProductivityInsightResponse` | `analyze_productivity()` → `get_qa_single_insight(domain="productivity")` |
| `POST /insight/overall_insight` | `OverallInsightRequest` | `OverallInsightResponse` | `analyze_overall_insight()` → `get_qa_single_insight(domain="overall")` |
| `POST /insight/health_insight` | `HealthInsightRequest` | `HealthInsightResponse` | `analyze_health_insight()` → `get_qa_single_insight(domain="health")` |

**Tất cả 3 endpoint đều gọi `get_qa_single_insight()` — luồng Q&A duy nhất.**

---

## 2. Cache Inventory

| Key (format) | TTL | Producer | Consumer | Notes |
|---|---|---|---|---|
| `qa_insight:{uid}:{domain}:{lang}` | 300s (5 min) | `get_qa_single_insight` | chính nó | Primary cache cho single insight |
| `extracted_signals:v2:{uid}` | 300s | `_get_or_create_extracted_signals` | nhiều nơi | Signals + context, dùng chung (read-only; no producer — always rebuilds) |
| `insight_pool:v2:{uid}` | 1800s (30 min) | `_build_qa_insight_pool` | **không đọc** | Pool được build nhưng không dùng trong luồng mới |
| `qa_asked:{uid}` | 1800s (30 min) | `qa_asked_add` | `qa_asked_get` | 30-min dedup: câu hỏi đã hỏi |
| `kw_pool:v2:{uid}` | 300s | `_get_or_create_keyword_pool` | `_select_keyword_for_domain` | Keyword pool |
| `kw_used:v2:{uid}:{domain}` | 300s | `_select_keyword_for_domain` | chính nó | Rotation tracking |
| `insight_actions_history:{uid}` | 21600s (6 hours) | `_save_action_history` | `get_recent_context` | Banned actions — matches `_WINDOW_HOURS=6` |
| `v2_narrative_bg:{uid}:{lang}:{date}:{fp}` | **disabled** (no cache) | — | — | Previously cached until end-of-day, removed to keep narrative in sync with mid-day calendar/health/mood changes |

---

## 3. Luồng chi tiết: `get_qa_single_insight()`

```
Request → Cache check (qa_insight:{uid}:{domain}:{lang})
           ├─ HIT → return immediately
           └─ MISS:
               ├─ Step 1: Signals (always fresh — no cache)
               ├─ Step 2: Narrative (always fresh — no cache)
               ├─ Step 3: LLM picks group
               ├─ Step 4: Pick 1 question (30-min dedup)
               ├─ Step 5: LLM answers question (structured)
               ├─ Step 6: Render to prose (with current_time)
               ├─ Step 7: Polish
               ├─ Cache 5min + return
               └─ Background: pre-cache 2 remaining domains
```

### Step 1: Signals — `_get_or_create_extracted_signals()`

- **Cache check**: `extracted_signals:v2:{uid}`
- **Cache miss** → gọi `_collect_and_process()`:
  - `_collect_all_data()`: 12 parallel API calls (historical + today)
  - `PrepareCalendarData`, `PrepareTimeData`, `PrepareHealthData`
  - `DataProcessor.extract_signals()`: 8 blocks → `ExtractedSignals`
- **Output**: `ExtractedSignals` object với `meta.current_time`, `meta.time_phase`, ...

### Step 2: Narrative — `_build_user_context_narrative()`

- Template-based (no LLM), 10 sections
- **Not cached** — always rebuilt from current `ExtractedSignals`. The previous
  `v2_narrative_bg` cache (end-of-day TTL) was removed because user feedback
  showed stale narrative drifting from real-time calendar/health/mood changes.

### Step 3: Group Selection — LLM

- `build_group_selection_prompt()`: LLM chọn best group dựa trên narrative
- Group descriptions matched với user context
- **1 LLM call**

### Step 4: Question Pick (30-min dedup)

- Get `qa_asked:{uid}` từ Redis
- Filter questions in chosen group (skip đã hỏi trong 30 phút)
- Random pick 1 question từ remaining
- **0 LLM call**

### Step 5: Q&A Answer — LLM

- `system_builders[domain]()`: system prompt với constraints
- `prompt_builders[domain]()`: user prompt với narrative + question
- **1 LLM call**
- Output: structured `{current_state, evidence, cause, action, expected_outcome, confidence}`

### Step 6: Render to Prose — LLM

- `build_render_prompt()`:
  - `current_time_anchor`: thời gian hiện tại được inject vào prompt
  - `time_phase_directive`: late_night/bedtime → sleep only, evening → wind_down
  - `logic_context_block`: narrative_hook, causal_chain, insight_structure
- **1 LLM call**
- Output: 1 prose paragraph

### Step 7: Polish — LLM

- Simple language polishing
- **1 LLM call**

### Background Pre-cache

- Sau khi return, chạy 2 remaining domains sequential
- Mỗi domain: signals (cache hit) → narrative (cache hit) → Q&A → render → polish → cache
- **6 LLM calls** (2 domains × 3 LLM calls)

---

## 4. LLM Call Count

| Step | Phase | LLM calls |
|---|---|---|
| 1 | Signals | 0 |
| 2 | Narrative | 0 |
| 3 | Group selection | 1 |
| 4 | Question pick | 0 |
| 5 | Q&A answer | 1 |
| 6 | Render | 1 |
| 7 | Polish | 1 |
| **Total per domain** | | **4** |
| Background pre-cache | 2 domains × 3 | 6 |
| **Total per request + background** | | **10** |

---

## 5. time_phase Resolution

`compute_time_phase(group, time_data, health_params)` (`insights/helpers/day_profile.py`):

- 5-group map: V1 groups → phases
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

## 6. current_time Anchor

`meta.current_time` được set từ `PrepareTimeData`:

```python
current_dt = datetime.now(tz)  # Tz from request timezone
current_time_iso = current_dt.isoformat()
```

**Inject vào render prompt**:
```
CURRENT TIME: 13:15 (Thursday, July 09, 2026).
ALL action timings MUST be anchored to this exact moment.
If the user is free NOW, schedule action as 'immediate'.
Never suggest past times or generic 'morning/afternoon' actions.
```

---

## 7. Response Format

```python
# Productivity / Health
ProductivityInsightResponse:
  status: "success" | "error"
  user_id: str
  insight: {"point1": prose, "point2": ""}
  error: Optional[str]

# Overall
OverallInsightResponse:
  status: "success" | "error"
  user_id: str
  insight: {"great_job": "", "need_attention": prose, "opportunity": ""}
  error: Optional[str]
```

---

## 8. Failure Modes

- **LLM timeout**: Return fallback dict/empty string → `{"point1": "Please try again.", "point2": ""}`
- **Redis down**: Full recompute (10 LLM + 12 API calls)
- **No questions available**: Return `None` (edge case)
- **Q&A LLM fails**: Absorbed per-group → skip to next group

---

## 9. Dead Code (luồng cũ đã xóa)

| Item | Notes |
|---|---|
| `_get_or_generate_insight()` | Đã xóa - từng là entry point cho round-based pipeline |
| `_generate_and_cache_all_insights()` | Đã xóa - 6-round multi-domain pipeline (350+ lines) |
| `insight_service_impl.py` | Legacy service, không còn được import |

---

## 10. Luồng cũ (đã xóa) — để tham khảo

### Round-based pipeline (đã xóa)

Luồng cũ `_generate_and_cache_all_insights()` chạy 6 rounds:

| Round | Actions | LLM calls |
|---|---|---|
| 1 | health plan | 1 |
| 2 | prod plan + health draft | 2 |
| 3 | overall plan + prod draft + health render | 3 |
| 4 | overall draft + prod render + health polish | 3 |
| 5 | overall render + prod polish + audit | 2 |
| 6 | overall polish | 0 |
| **Total** | | **11** (chưa tính background) |

**Đã xóa hoàn toàn** — không còn được gọi từ bất kỳ đâu.
