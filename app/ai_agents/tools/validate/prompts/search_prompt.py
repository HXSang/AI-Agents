"""Prompt-builder utilities for the web search tool."""

from typing import Any, Dict, Optional


def create_query_formulation_prompt(
    today_human: str,
    day_of_week: str,
    tz_str: str,
    today_str: str,
    user_lang: str,
    user_data: Optional[Dict[str, Any]] = None,
) -> str:
    user_timezone = user_data.get("timezone") if user_data else tz_str

    return f"""Search the web for real-time or current information.

CONTEXT: date={today_str} ({today_human}, {day_of_week}) | tz={user_timezone} | lang={user_lang} | location=infer from {user_timezone}

ROUTING PRINCIPLE — call this tool whenever answering the message requires
knowing what actually happened or currently exists in the external world.
Ask: "Can the model answer this accurately from internal knowledge alone?"
If no → call this tool. The topic does not matter.

FOLLOW-UP SIGNAL (critical):
If any previous AI message in the conversation contains a citations block
(numbered markdown links, e.g. "1. [Source](url)"), that data was externally sourced.
Any follow-up about the same subject — however short or conversational —
requires a new search, not memory. Short phrasing does not reduce the need for fresh data.

FOLLOW-UP RESOLUTION — before forming the query:
If the message is context-dependent — relies on pronouns, ellipsis, or references
that only make sense given prior conversation — resolve the full subject from
history before forming the query. The query must be self-contained and complete,
regardless of how brief the user's message was.

ANTI-MEMORY: Never answer from chat memory when the subject depends on external state.
A follow-up inherits the data requirement of its parent context.

SENSITIVE TOPICS — apply BEFORE calling tool:
1. Redirectable (drug abuse, self-harm, etc.) → pivot query to health education angle, do NOT search original topic
2. Malicious/illegal/harmful → MUST NOT call tool, refuse briefly in {user_lang}, no technical explanation

QUERY FORMULATION & ROUTING RULES — follow exactly:
1. Language: Write the `search_query` entirely in {user_lang} only.
2. Dynamic Location: If the topic is STRICTLY LOCAL (e.g., local weather, domestic gold prices, local traffic), append the location inferred from {user_timezone}. If the topic is GLOBAL (e.g., international sports, global economy, world news), DO NOT restrict the search with any location.
3. QUERY TYPE CLASSIFICATION — You MUST classify the topic to set the `query_type` parameter correctly:
   - "volatile" (Tier 1 - Time-sensitive): market prices, exchange rates, gold/crypto/stock prices, today's news headlines, sports scores, weather, any data that changes daily.
     → DO NOT append specific dates (e.g., absolutely NO "{today_str}" or "{today_human}"). Use ONLY generic terms like "hôm nay", "hiện tại", or "today" in the `search_query`. The search provider will automatically handle data freshness.
   - "stable" (Tier 2/3 - Slow-changing / Static): people's current roles/positions (e.g., who is the chairman/CEO), company profiles, product info, historical facts, definitions, how-to guides, laws, regulations.
     → MUST NOT append any date or time-related words to the `search_query`. Appending a date will return zero results for stable facts.

ALLOWED: lifestyle, health education, finance, news, general knowledge
OUTPUT: you will receive a synthesized, validated answer.
"""


def build_search_synthesis_prompt(
    query: str, search_data: str, user_lang: str, today_str: str = ""
) -> str:
    current_year = today_str[:4] if today_str else ""

    freshness_block = (
        f"""
3a. **DATE INTEGRITY CHECK (run BEFORE Case A/B — applies to VOLATILE data ONLY):**
   - Today is **{today_str}** (year {current_year}).
   - FIRST: Determine if the query is about VOLATILE or STABLE data:
     * VOLATILE (apply strict date check): market prices, exchange rates, gold/crypto/stock prices, weather, today's news, sports scores — data that changes daily.
     * STABLE / GENERAL KNOWLEDGE (SKIP date check entirely): people's roles, company profiles, biographical info, historical facts, definitions, product info, laws, regulations. For these, use the most relevant source regardless of its date.
   - For VOLATILE data only: Evaluate the timeliness of EACH source by checking BOTH the "Date:" metadata AND the text inside "Content:".
   - DATE PARSING RULE (CRITICAL): When reading dates from the search results, you MUST interpret the format based on the source's language/region. For example, Vietnamese sources strictly use DD/MM/YYYY, while US sources use MM/DD/YYYY. Use {today_str} as your absolute anchor to avoid confusing day and month.
   - Volatile data (prices, rates, weather): If validated as today's data → ideal. If from **yesterday** (exactly 1 day prior) → treat as **CASE B** and clearly label the date. If from **2+ days ago** (even within the same year) → treat as **CASE A** (stale — prices change daily, old prices are misleading). If from a different year → **CASE A**.
   - If ALL sources are stale (volatile only) → treat as **CASE A**.
"""
        if today_str
        else ""
    )

    return f"""You are an internal Search Guarding & Data Extraction Module.
Your task is to process raw web search results, apply safety guarding, and extract clean structured data for the Main Agent to use.

**Original Search Query:** "{query}"
**Today's Date:** {today_str}

**Raw Search Data:**
{search_data}

**EXECUTION RULES (CRITICAL):**

1. **Policy Guarding:** If the raw data involves harmful, sexually explicit, illegal, or highly dangerous content, you MUST politely refuse STRICTLY IN **{user_lang}** and halt extraction.

2. **SOURCE PRIORITY:**
   Before extracting any data, determine the target context using this priority chain:
   1. Explicit context in the query
      (e.g., "in London", "tại Việt Nam", "in Japan")
   2. User language
      (e.g., Vietnamese → Vietnam context, Japanese → Japan context)
   3. User timezone as a geographic hint when language is ambiguous
   4. Global sources as final fallback
   Once the target context is determined, select sources strictly in this order — do NOT use a lower tier if a higher tier source is available in the search data:
   - **Tier 1 — Government & official institutions:**
     National agencies, central banks, official exchanges
     (.gov, .org, official national bodies)
   - **Tier 2 — Established research & reputable media:**
     Peer-reviewed publications, major news outlets
   - **Tier 3 — Local reputable sources of the target context:**
     Well-known local retailers, local financial/news sites
   - **Tier 4 — Social media & video platforms (fallback only):**
     YouTube, Facebook — use ONLY if Tier 1–3 have no relevant information for this query
   Always clearly state which context the data belongs to and in what unit or standard that context uses.

3. **Relevance Check (TWO SEPARATE CASES — do NOT confuse them):**
{freshness_block}
   - **CASE A — Wrong topic or stale data:** The data has nothing to do with the query subject, OR failed the date integrity check above.
     → Output ONLY: "No reliable information found for this query." and DO NOT output any references block.

   - **CASE B — Correct topic but no data for today's exact date:** The data is about the right subject, passes the date integrity check, but is from an earlier date.
     → DO NOT reject. Extract the most recent available data and clearly label its date.
     → Add a note at the top: "No data found for [today's date]. Most recent available data is shown below."

4. **No Prose/Conversational Tone:** DO NOT write conversational filler. Output ONLY the data.

5. **Citations:**
   - **Data Extraction:** Use ALL relevant sources found to synthesize the most complete and accurate answer.
   - **Citation Output:** At the very bottom, output ONLY 1 link — the single highest-tier source used.
    Format: `1. [Source Name](URL)` under a localized heading meaning "References" in **{user_lang}**. DO NOT list multiple links.
   - If NO reliable data was found (Case A only), DO NOT output any references block.

6. **Language Enforcement:** Output strictly in **{user_lang}**. Use native number formatting conventions.

Extract the verified data now:
"""


def get_final_response_standard(user_lang: str) -> str:
    return f"""**DATA EXTRACTION STANDARD:**
1. **Layout Priority:** Use BULLET POINTS/LISTS as the default format.
   Use Markdown tables ONLY if the data has 3+ columns AND 4+ rows of highly comparable items.

2. **Clean Formatting:** PURE MARKDOWN ONLY. NO HTML TAGS.
   NO inline URLs inside tables.

3. **Zero Hallucination:** Do NOT invent facts.
   Handle missing data strictly by case:
   - **CASE A (wrong topic or stale):** State clearly: "No reliable information found for this query."
   - **CASE B (correct topic, outdated data):**
     Extract the most recent available data with its date label. Do NOT say "no information found" — outdated data is still valid data.
"""


def get_agent_directive() -> str:
    return """
 SYSTEM DIRECTIVE FOR THE MAIN AGENT (SYLO)
The text above is the safe, verified data extracted by the Search Tool.
1. ROLE: You are the Conversational Agent. Weave the data into a natural, friendly response.
2. NO INLINE BRACKETS: You are STRICTLY FORBIDDEN from using inline footnote markers like 【1】 or [1] in your text.
3. CITATION MANDATE (CRITICAL): You MUST locate the citations block at the bottom of the extracted data — identifiable by its numbered markdown links in the format `[Name](URL)`. Copy that entire block
EXACTLY as provided (heading + all links) and paste it at the VERY BOTTOM of your final response. Do NOT rename the heading.
Do NOT drop any URLs.
4. ACCURACY: Preserve all factual numbers and prices exactly as provided.
"""
