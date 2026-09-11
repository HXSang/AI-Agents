from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def _parse_dt(raw: Any) -> Optional[datetime]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _minutes_between(start: datetime, end: datetime) -> int:
    return max(int((end - start).total_seconds() // 60), 0)


def _safe_rate(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return (num / den) * 100.0


def _is_meeting(event: Dict[str, Any]) -> bool:
    event_type = str(event.get("eventType") or "").lower()
    participants = event.get("participants")
    p_count = len(participants) if isinstance(participants, list) else 0
    return event_type == "meeting" or p_count >= 2


def _calc_free_blocks(intervals: List[Tuple[datetime, datetime]]) -> List[int]:
    if len(intervals) <= 1:
        return []
    gaps: List[int] = []
    ordered = sorted(intervals, key=lambda x: x[0])
    for i in range(len(ordered) - 1):
        cur_end = ordered[i][1]
        nxt_start = ordered[i + 1][0]
        gap = _minutes_between(cur_end, nxt_start)
        if gap > 0:
            gaps.append(gap)
    return gaps


class ProductivityCalendarProcessor:
    """Doc-aligned schedule-quality processor."""

    @staticmethod
    def process(
        *,
        calendar_events: List[Dict[str, Any]],
        productivity_summary: Dict[str, Any],
        calendar_work_hours: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        rows = [e for e in (calendar_events or []) if isinstance(e, dict)]
        intervals: List[Tuple[datetime, datetime, Dict[str, Any]]] = []
        for e in rows:
            s = _parse_dt(e.get("startTime"))
            t = _parse_dt(e.get("endTime"))
            if not s or not t or t <= s:
                continue
            intervals.append((s, t, e))
        intervals.sort(key=lambda x: x[0])

        meeting_intervals = [x for x in intervals if _is_meeting(x[2])]
        meeting_count = len(meeting_intervals)
        meeting_hours = round(sum(_minutes_between(s, e) for s, e, _ in meeting_intervals) / 60.0, 2)

        # Workload / utilization
        occupied_hours = 0.0
        total_hours = 0.0
        for wh in calendar_work_hours or []:
            if not isinstance(wh, dict):
                continue
            occupied_hours += float(wh.get("totalHours") or wh.get("scheduledHours") or 0.0)
            total_hours += float(wh.get("targetHours") or 8.0)
        if occupied_hours <= 0 and intervals:
            occupied_hours = round(sum(_minutes_between(s, e) for s, e, _ in intervals) / 60.0, 2)
            total_hours = 8.0
        utilization = round(_safe_rate(occupied_hours, total_hours or 8.0), 1)

        # Meeting quality
        long_meetings = [
            {
                "title": ev.get("summary") or ev.get("title") or "(untitled)",
                "category": (str(ev.get("category")).strip().upper() if ev.get("category") else None),
                "eventType": (str(ev.get("eventType") or ev.get("event_type") or "").strip().lower() or None),
                "durationMinutes": _minutes_between(s, e),
            }
            for s, e, ev in meeting_intervals
            if _minutes_between(s, e) > 120
        ]

        chains = 0
        gaps_minutes: List[int] = []
        for i in range(len(meeting_intervals) - 1):
            cur_end = meeting_intervals[i][1]
            nxt_start = meeting_intervals[i + 1][0]
            gap = _minutes_between(cur_end, nxt_start)
            gaps_minutes.append(gap)
            if gap < 10:
                chains += 1
        avg_gap = round(sum(gaps_minutes) / len(gaps_minutes), 1) if gaps_minutes else None

        # Meeting overlaps: start of a later meeting before end of an earlier one
        overlaps: List[Dict[str, Any]] = []
        for i in range(len(meeting_intervals)):
            s1, e1, ev1 = meeting_intervals[i]
            for j in range(i + 1, len(meeting_intervals)):
                s2, e2, ev2 = meeting_intervals[j]
                if s2 >= e1:
                    break
                overlap_end = min(e1, e2)
                overlap_mins = _minutes_between(s2, overlap_end)
                if overlap_mins <= 0:
                    continue
                overlaps.append(
                    {
                        "a": ev1.get("summary") or ev1.get("title") or "(untitled)",
                        "aCategory": (
                            str(ev1.get("category")).strip().upper()
                            if ev1.get("category")
                            else None
                        ),
                        "b": ev2.get("summary") or ev2.get("title") or "(untitled)",
                        "bCategory": (
                            str(ev2.get("category")).strip().upper()
                            if ev2.get("category")
                            else None
                        ),
                        "overlapMinutes": overlap_mins,
                    }
                )

        # Structure
        free_blocks = _calc_free_blocks([(s, e) for s, e, _ in intervals])
        fragment_count = len(free_blocks)
        focus_blocks = [g for g in free_blocks if g >= 90]
        recovery_windows = [g for g in free_blocks if 15 <= g <= 30]

        # Lunch break availability (11:30-13:30 local day anchor)
        lunch_available = False
        if intervals:
            base = intervals[0][0]
            lunch_start = base.replace(hour=11, minute=30, second=0, microsecond=0)
            lunch_end = base.replace(hour=13, minute=30, second=0, microsecond=0)
            blocks = sorted([(s, e) for s, e, _ in intervals], key=lambda x: x[0])
            cursor = lunch_start
            for s, e in blocks:
                if e <= lunch_start or s >= lunch_end:
                    continue
                if s > cursor and _minutes_between(cursor, s) >= 30:
                    lunch_available = True
                    break
                cursor = max(cursor, e)
            if not lunch_available and cursor < lunch_end and _minutes_between(cursor, lunch_end) >= 30:
                lunch_available = True

        # Execution metrics from productivity summary
        commitments_due = float(productivity_summary.get("commitmentsDue") or 0)
        commitments_done = float(productivity_summary.get("commitmentsCompleted") or 0)
        commitment_rate = round(_safe_rate(commitments_done, commitments_due), 1)

        completed_meetings = float(productivity_summary.get("completedMeetingSoFar") or 0)
        meeting_rate = round(_safe_rate(completed_meetings, meeting_count), 1)

        productivity_score = {
            "productivityScorePercent": productivity_summary.get("productivityScorePercent"),
            "overallPercent": productivity_summary.get("overallPercent"),
            "focusOutputPercent": productivity_summary.get("focusOutputPercent"),
            "meetingPercent": productivity_summary.get("meetingPercent"),
        }

        anomalies: List[Dict[str, Any]] = []
        if long_meetings:
            anomalies.append({"type": "long_meeting", "count": len(long_meetings)})
        if overlaps:
            anomalies.append(
                {
                    "type": "meeting_overlap",
                    "count": len(overlaps),
                    "pairs": overlaps[:5],
                }
            )
        if chains >= 2:
            anomalies.append({"type": "back_to_back_meetings", "count": chains})
        if not intervals:
            anomalies.append({"type": "empty_workday"})
        if utilization >= 95:
            anomalies.append({"type": "overloaded_workday", "utilization": utilization})
        if not focus_blocks and intervals:
            anomalies.append({"type": "no_focus_block"})
        if not lunch_available and intervals:
            anomalies.append({"type": "no_lunch_break"})

        summary = {
            "eventCount": len(intervals),
            "meetingCount": meeting_count,
            "meetingHours": meeting_hours,
            "occupiedHours": occupied_hours,
            "utilization": utilization,
        }

        signals: Dict[str, Dict[str, Any]] = {
            "meetingLoad": {
                "status": "high" if meeting_hours >= 5 else "moderate" if meeting_hours >= 2 else "light",
                "metrics": {"meetingCount": meeting_count, "meetingHours": meeting_hours},
                "comparison": {},
                "trend": {"direction": "stable"},
                "evidence": {
                    "meetings": [
                        {
                            "title": ev.get("summary") or ev.get("title"),
                            "category": (
                                str(ev.get("category")).strip().upper()
                                if ev.get("category")
                                else None
                            ),
                            "eventType": (
                                str(ev.get("eventType") or ev.get("event_type") or "").strip().lower()
                                or None
                            ),
                        }
                        for _, _, ev in meeting_intervals[:5]
                    ]
                },
            },
            "calendarUtilization": {
                "status": "high" if utilization >= 85 else "moderate" if utilization >= 60 else "light",
                "metrics": {"occupiedHours": round(occupied_hours, 2), "utilization": utilization},
                "comparison": {"targetHours": round(total_hours or 8.0, 2)},
                "trend": {"direction": "stable"},
                "evidence": {},
            },
            "longMeetings": {
                "status": "attention" if long_meetings else "good",
                "metrics": {"count": len(long_meetings)},
                "comparison": {},
                "trend": {"direction": "stable"},
                "evidence": {"meetings": long_meetings[:3]},
            },
            "backToBackMeetings": {
                "status": "high" if chains >= 3 else "moderate" if chains >= 1 else "low",
                "metrics": {"chains": chains},
                "comparison": {},
                "trend": {"direction": "stable"},
                "evidence": {"gaps": gaps_minutes[:6]},
            },
            "meetingOverlap": {
                "status": "attention" if overlaps else "good",
                "metrics": {"count": len(overlaps)},
                "comparison": {},
                "trend": {"direction": "stable"},
                "evidence": {"pairs": overlaps[:5]},
            },
            "meetingDensity": {
                "status": "healthy" if (avg_gap or 999) >= 45 else "dense",
                "metrics": {"averageGapMinutes": avg_gap},
                "comparison": {},
                "trend": {"direction": "stable"},
                "evidence": {},
            },
            "scheduleFragmentation": {
                "status": "high" if fragment_count >= 5 else "moderate" if fragment_count >= 3 else "low",
                "metrics": {"fragmentCount": fragment_count},
                "comparison": {},
                "trend": {"direction": "stable"},
                "evidence": {"freeBlocksMinutes": free_blocks[:8]},
            },
            "focusBlocks": {
                "status": "good" if len(focus_blocks) >= 2 else "limited" if len(focus_blocks) == 1 else "none",
                "metrics": {
                    "focusBlocks": len(focus_blocks),
                    "longestMinutes": max(focus_blocks) if focus_blocks else 0,
                },
                "comparison": {},
                "trend": {"direction": "stable"},
                "evidence": {"blocksMinutes": focus_blocks[:6]},
            },
            "recoveryWindows": {
                "status": "good" if len(recovery_windows) >= 2 else "limited",
                "metrics": {"availableBreaks": len(recovery_windows)},
                "comparison": {},
                "trend": {"direction": "stable"},
                "evidence": {"windowsMinutes": recovery_windows[:6]},
            },
            "lunchBreakAvailability": {
                "status": "available" if lunch_available else "missing",
                "metrics": {"available": lunch_available},
                "comparison": {},
                "trend": {"direction": "stable"},
                "evidence": {},
            },
            "commitmentCompletion": {
                "status": "good" if commitment_rate >= 75 else "attention" if commitment_rate < 50 else "moderate",
                "metrics": {
                    "completionRate": commitment_rate,
                    "completed": int(commitments_done),
                    "due": int(commitments_due),
                },
                "comparison": {},
                "trend": {"direction": "stable"},
                "evidence": {},
            },
            "meetingCompletion": {
                "status": "good" if meeting_rate >= 75 else "pending",
                "metrics": {"completionRate": meeting_rate},
                "comparison": {},
                "trend": {"direction": "stable"},
                "evidence": {},
            },
            "productivityScore": {
                "status": "ok",
                "metrics": productivity_score,
                "comparison": {},
                "trend": {"direction": "stable"},
                "evidence": {},
            },
            "anomalies": {
                "status": "attention" if anomalies else "normal",
                "metrics": {"count": len(anomalies)},
                "comparison": {},
                "trend": {"direction": "stable"},
                "evidence": {"items": anomalies},
            },
        }

        penalty = (
            min(len(anomalies), 5) * 7
            + (10 if utilization >= 95 else 0)
            + (8 if not focus_blocks and intervals else 0)
            + (12 if overlaps else 0)
        )
        score = max(0.0, min(100.0, 100.0 - penalty))
        overall = {
            "status": "good" if score >= 80 else "attention" if score < 60 else "moderate",
            "score": round(score, 1),
            "confidence": 0.9 if intervals else 0.6,
        }

        return {
            "summary": summary,
            "signals": signals,
            "anomalies": anomalies,
            "overall": overall,
        }
