max_chars: int = 100
target_language = str 

HEALTH_INSIGHT_PROMPT_TEMPLATE = """Analyze available health data and extract key metrics with values and units. 
Compare them with standard reference ranges and the user's personal baseline. 
Identify 7/14/30-day trends, significant changes, and anomalies.
Cross-analyze metrics to infer possible relationships and explain the reasoning without diagnosing. 
COMPARISON RULES:
- Use 7-day, 14-day, or 30-day comparisons whenever sufficient historical data is available.
- If a requested comparison period is unavailable, fall back to the nearest meaningful comparison (such as the previous 3 days or the most recent available day).
- If no suitable comparison data exists, omit the comparison entirely instead of stating that comparison data is unavailable.
- Never infer or invent historical values, trends, or percentage changes.
- Only describe trends that are directly supported by the available data.
Return the top 3–5 insight (Each insight should implicitly include an observation, supporting evidence, trend, interpretation, and recommendation, but do not explicitly output or repeat the labels "Observation", "Evidence", "Trend", "Interpretation", or "Recommendation" (or their translations in {target_language}).
Write each insight as a natural, fluent sentence or paragraph. Recommendations should use natural advisory language appropriate to {target_language}, such as equivalents of "should", "could", "may", "consider", or "it is recommended to", rather than imperative commands.)
Each insight must internally include:
- an observation,
- supporting evidence,
- the trend,
- an interpretation,
- and a recommendation.
OUTPUT FORMAT:
- Select the 3–5 most important insights.
- Merge them into one coherent summary instead of listing them separately.
- Connect related ideas naturally using appropriate transition words in {target_language}.
- The final output must be a single paragraph that reads like a concise narrative rather than multiple independent insights.
- Do not use headings, labels, bullet points, numbering, or line breaks.
- The entire paragraph should be approximately {max_chars} words long.
LANGUAGE:
- Write the response entirely in {target_language}.
- Translate all field names, labels, and values into {target_language}.
- Do not mix languages."""

PRODUCTIVITY_INSIGHT_PROMPT_TEMPLATE = """Analyze available productivity data and extract key metrics with values and units. 
Compare them with standard reference ranges and the user's personal baseline. 
Identify 7/14/30-day trends, significant changes, and anomalies.
Cross-analyze metrics to infer possible relationships and explain the reasoning without diagnosing. 
COMPARISON RULES:
- Use 7-day, 14-day, or 30-day comparisons whenever sufficient historical data is available.
- If a requested comparison period is unavailable, fall back to the nearest meaningful comparison (such as the previous 3 days or the most recent available day).
- If no suitable comparison data exists, omit the comparison entirely instead of stating that comparison data is unavailable.
- Never infer or invent historical values, trends, or percentage changes.
- Only describe trends that are directly supported by the available data.
Return the top 3–5 insight (Each insight should implicitly include an observation, supporting evidence, trend, interpretation, and recommendation, but do not explicitly output or repeat the labels "Observation", "Evidence", "Trend", "Interpretation", or "Recommendation" (or their translations in {target_language}).
Write each insight as a natural, fluent sentence or paragraph. Recommendations should use natural advisory language appropriate to {target_language}, such as equivalents of "should", "could", "may", "consider", or "it is recommended to", rather than imperative commands.)
Each insight must internally include:
- an observation,
- supporting evidence,
- the trend,
- an interpretation,
- and a recommendation.
OUTPUT FORMAT:
- Select the 3–5 most important insights.
- Merge them into one coherent summary instead of listing them separately.
- Connect related ideas naturally using appropriate transition words in {target_language}.
- The final output must be a single paragraph that reads like a concise narrative rather than multiple independent insights.
- Do not use headings, labels, bullet points, numbering, or line breaks.
- The entire paragraph should be approximately {max_chars} words long.
LANGUAGE:
- Write the response entirely in {target_language}.
- Translate all field names, labels, and values into {target_language}.
- Do not mix languages."""

OVERALL_INSIGHT_PROMPT_TEMPLATE = """\
Analyze available productivity, health, financial data and extract key metrics with values and units
Compare them with standard reference ranges and the user's personal baseline. 
Identify 7/14/30-day trends, significant changes, and anomalies.
Cross-analyze metrics to infer possible relationships and explain the reasoning without diagnosing. 
COMPARISON RULES:
- Use 7-day, 14-day, or 30-day comparisons whenever sufficient historical data is available.
- If a requested comparison period is unavailable, fall back to the nearest meaningful comparison (such as the previous 3 days or the most recent available day).
- If no suitable comparison data exists, omit the comparison entirely instead of stating that comparison data is unavailable.
- Never infer or invent historical values, trends, or percentage changes.
- Only describe trends that are directly supported by the available data.
Return the top 3–5 insight (Each insight should implicitly include an observation, supporting evidence, trend, interpretation, and recommendation, but do not explicitly output or repeat the labels "Observation", "Evidence", "Trend", "Interpretation", or "Recommendation" (or their translations in {target_language}).
Write each insight as a natural, fluent sentence or paragraph. Recommendations should use natural advisory language appropriate to {target_language}, such as equivalents of "should", "could", "may", "consider", or "it is recommended to", rather than imperative commands.)
Each insight must internally include:
- an observation,
- supporting evidence,
- the trend,
- an interpretation,
- and a recommendation.
Write it as one long sentence of approximately {max_chars} character.
LANGUAGE:
- Write the response entirely in {target_language}.
- Translate all field names, labels, and values into {target_language}.
- Do not mix languages."""