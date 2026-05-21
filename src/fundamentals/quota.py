"""
src/fundamentals/quota.py — Alpha Vantage Quota Manager
=========================================================
Tracks how many Alpha Vantage API requests have been used today
and provides smart quota-aware fetching to avoid hitting the 25/day limit.

NOTE: Commodity prices now use Yahoo Finance (free, unlimited).
      This quota manager only applies to: news, overview, income, balance.
"""

import json
import logging
from datetime import date
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATA_RAW_DIR, get_shared_dir

logger = logging.getLogger(__name__)

# Free tier daily limit
DAILY_LIMIT = 25

# Cost per feature per ticker (commodities removed — now uses Yahoo Finance)
COST = {
    "news":     1,   # NEWS_SENTIMENT endpoint
    "overview": 1,   # OVERVIEW endpoint
    "income":   1,   # INCOME_STATEMENT endpoint
    "balance":  1,   # BALANCE_SHEET endpoint
}

# Quota file stored in shared/quota/ subfolder
# Path: data/raw/shared/quota/av_quota.json
QUOTA_FILE = get_shared_dir("quota") / "av_quota.json"


class QuotaManager:
    """
    Tracks daily Alpha Vantage API usage by persisting counts to a JSON file.
    Resets automatically when a new calendar day begins.
    """

    def __init__(self):
        self._load()

    def _load(self):
        """Load quota data from disk. Reset if it's a new day."""
        today = str(date.today())
        if QUOTA_FILE.exists():
            try:
                data = json.loads(QUOTA_FILE.read_text())
                if data.get("date") == today:
                    self.used  = data.get("used", 0)
                    self.today = today
                    return
            except Exception:
                pass
        # New day or corrupt file — reset
        self.used  = 0
        self.today = today
        self._save()

    def _save(self):
        """Persist current quota usage to disk."""
        try:
            QUOTA_FILE.write_text(json.dumps({"date": self.today, "used": self.used}))
        except Exception:
            pass

    @property
    def remaining(self) -> int:
        return max(0, DAILY_LIMIT - self.used)

    def record(self, n: int = 1):
        """Record that n requests were used."""
        self.used = min(self.used + n, DAILY_LIMIT)
        self._save()

    def can_afford(self, n: int) -> bool:
        return self.remaining >= n

    def status_line(self) -> str:
        """
        Returns a coloured progress bar showing today's quota usage.

        Examples:
          🟢 Alpha Vantage: 4/25 used  [███░░░░░░░░░░░░░░░░░] 21 remaining  (resets midnight UTC)
          🟡 Alpha Vantage: 18/25 used [██████████████░░░░░░] 7 remaining   (resets midnight UTC)
          🔴 Alpha Vantage: 25/25 used [████████████████████] 0 remaining   (resets midnight UTC)
        """
        pct         = self.used / DAILY_LIMIT
        bar_filled  = int(pct * 20)
        bar         = "█" * bar_filled + "░" * (20 - bar_filled)
        remaining   = self.remaining

        if remaining >= 15:
            icon = "🟢"
        elif remaining >= 5:
            icon = "🟡"
        else:
            icon = "🔴"

        note = " ← refill at midnight UTC" if remaining <= 3 else "  (resets midnight UTC)"
        return (
            f"  {icon} Alpha Vantage: {self.used}/{DAILY_LIMIT} used "
            f"[{bar}] {remaining} remaining{note}"
        )

    def what_can_afford(self, n_tickers: int, intent: str) -> list:
        """
        Given remaining quota and number of tickers, return which
        features can be fetched without exceeding the daily limit.

        Priority order (most valuable first):
          1. overview  — P/E, ROE, ROA, D/E, analyst targets
          2. income    — revenue, profit trends
          3. news      — sentiment scores
          4. balance   — balance sheet
        """
        # Map intent to desired features
        intent_features = {
            "news":         ["news"],
            "fundamentals": ["overview", "income", "balance"],
            "income":       ["income", "overview"],
            "balance":      ["balance", "overview"],
            "full":         ["overview", "income", "news", "balance"],
            "analyse":      ["overview", "news"],
        }
        wanted = intent_features.get(intent, ["overview", "income", "news", "balance"])

        affordable = []
        budget = self.remaining

        for feature in wanted:
            cost = COST[feature] * n_tickers
            if budget >= cost:
                affordable.append(feature)
                budget -= cost
            else:
                logger.warning(
                    f"  ⚠️  Skipping '{feature}' — needs {cost} request(s) "
                    f"but only {budget} remaining today"
                )

        return affordable