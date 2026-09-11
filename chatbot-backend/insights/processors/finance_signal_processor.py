import logging
from typing import Any, Dict, Optional

from insights.schemas.processed_context import FinanceSignalsBlock

logger = logging.getLogger(__name__)


class FinanceSignalProcessor:
    @staticmethod
    def process(raw_data: Dict[str, Any]) -> Optional[FinanceSignalsBlock]:
        finance_summary = raw_data.get("finance_summary") or {}
        finance_goals = raw_data.get("finance_goals") or []
        finance_bills = raw_data.get("finance_bills") or []

        if not finance_summary and not finance_goals and not finance_bills:
            return None

        income = float(finance_summary.get("total_income", 0))
        expense = float(finance_summary.get("total_expense", 0))
        budget = float(finance_summary.get("budget_limit", 0))

        budget_util = (expense / budget) if budget > 0 else None
        cashflow = income - expense
        cashflow_status = (
            "positive" if cashflow > 0 else ("negative" if cashflow < 0 else "neutral")
        )

        on_track = sum(1 for g in finance_goals if g.get("status") == "ON_TRACK")
        behind = sum(1 for g in finance_goals if g.get("status") == "BEHIND")

        bills_due = sum(
            1
            for b in finance_bills
            if b.get("status") == "DUE_SOON" or b.get("status") == "OVERDUE"
        )

        return FinanceSignalsBlock(
            budget_utilization=budget_util,
            bills_due_today=bills_due,
            cashflow_status=cashflow_status,
            active_goals_on_track=on_track,
            active_goals_behind=behind,
            critical_alerts=[],
        )
