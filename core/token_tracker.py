"""
Token Tracker - Token Counting and Cost Calculation
Tracks API usage and calculates costs for Claude models

Phase 4 Enhancement: Hard Limit Checking
- Integration with CostLogModel for persistent cost tracking
- Daily cost aggregation from database
- Hard limit safety checks with automatic shutdown trigger
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging


logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Supported Claude model types"""

    HAIKU_3_5 = "haiku-3.5"
    SONNET_3_5 = "sonnet-3.5"


@dataclass
class ModelPricing:
    """Pricing information for a specific model (per million tokens)"""

    model_name: str
    input_price_per_million: float  # USD per million input tokens
    output_price_per_million: float  # USD per million output tokens

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost for given token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Total cost in USD
        """
        input_cost = (input_tokens / 1_000_000) * self.input_price_per_million
        output_cost = (output_tokens / 1_000_000) * self.output_price_per_million
        return input_cost + output_cost


# Hardcoded pricing for Claude models (as of specification)
MODEL_PRICING: Dict[str, ModelPricing] = {
    ModelType.HAIKU_3_5: ModelPricing(
        model_name="Claude Haiku 3.5",
        input_price_per_million=1.00,
        output_price_per_million=5.00
    ),
    ModelType.SONNET_3_5: ModelPricing(
        model_name="Claude Sonnet 3.5",
        input_price_per_million=3.00,
        output_price_per_million=15.00
    )
}


@dataclass
class UsageRecord:
    """Record of a single API usage event"""

    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    metadata: Dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens used in this record"""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "metadata": self.metadata
        }


@dataclass
class UsageSummary:
    """Summary of token usage and costs"""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    record_count: int = 0
    by_model: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens across all usage"""
        return self.total_input_tokens + self.total_output_tokens

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "record_count": self.record_count,
            "by_model": self.by_model
        }


class TokenTracker:
    """
    Tracks token usage and calculates costs for Claude API calls.

    This class maintains a history of all API calls and provides
    methods for cost calculation and usage analysis.

    Phase 4 Enhancement: Now includes hard limit checking and database integration.
    """

    def __init__(self, db_session: Optional[Any] = None):
        """
        Initialize the token tracker.

        Args:
            db_session: Optional async database session for cost log queries
        """
        self.usage_records: List[UsageRecord] = []
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cost_usd: float = 0.0

        # Database session for persistent cost tracking
        self.db_session = db_session

        logger.info("TokenTracker initialized")

    def track_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        metadata: Optional[Dict] = None
    ) -> float:
        """
        Track API usage and calculate cost.

        Args:
            model: Model identifier (e.g., 'haiku-3.5', 'sonnet-3.5')
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
            metadata: Optional metadata to attach to this usage record

        Returns:
            Cost in USD for this usage

        Raises:
            ValueError: If model is not supported or token counts are invalid
        """
        # Validate inputs
        if model not in MODEL_PRICING:
            raise ValueError(
                f"Unsupported model: {model}. "
                f"Supported models: {list(MODEL_PRICING.keys())}"
            )

        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts must be non-negative")

        # Get pricing for this model
        pricing = MODEL_PRICING[model]

        # Calculate cost
        cost = pricing.calculate_cost(input_tokens, output_tokens)

        # Create usage record
        record = UsageRecord(
            timestamp=datetime.utcnow(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            metadata=metadata or {}
        )

        # Store record and update totals
        self.usage_records.append(record)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost

        logger.info(
            f"Tracked usage: {model} | "
            f"Input: {input_tokens} | Output: {output_tokens} | "
            f"Cost: ${cost:.6f}"
        )

        return cost

    def get_summary(self) -> UsageSummary:
        """
        Get a summary of all tracked usage.

        Returns:
            UsageSummary with aggregated statistics
        """
        summary = UsageSummary(
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            total_cost_usd=self.total_cost_usd,
            record_count=len(self.usage_records)
        )

        # Aggregate by model
        for record in self.usage_records:
            if record.model not in summary.by_model:
                summary.by_model[record.model] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "call_count": 0
                }

            model_stats = summary.by_model[record.model]
            model_stats["input_tokens"] += record.input_tokens
            model_stats["output_tokens"] += record.output_tokens
            model_stats["total_tokens"] += record.total_tokens
            model_stats["cost_usd"] += record.cost_usd
            model_stats["call_count"] += 1

        return summary

    def get_usage_records(
        self,
        model: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[UsageRecord]:
        """
        Get usage records, optionally filtered by model.

        Args:
            model: Optional model filter
            limit: Optional limit on number of records returned (most recent first)

        Returns:
            List of UsageRecord objects
        """
        records = self.usage_records

        if model:
            records = [r for r in records if r.model == model]

        # Sort by timestamp (most recent first)
        records = sorted(records, key=lambda r: r.timestamp, reverse=True)

        if limit:
            records = records[:limit]

        return records

    def calculate_estimated_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate estimated cost without tracking it.

        Useful for cost estimation before making API calls.

        Args:
            model: Model identifier
            input_tokens: Estimated input tokens
            output_tokens: Estimated output tokens

        Returns:
            Estimated cost in USD

        Raises:
            ValueError: If model is not supported
        """
        if model not in MODEL_PRICING:
            raise ValueError(
                f"Unsupported model: {model}. "
                f"Supported models: {list(MODEL_PRICING.keys())}"
            )

        pricing = MODEL_PRICING[model]
        return pricing.calculate_cost(input_tokens, output_tokens)

    def reset(self) -> None:
        """Reset all tracked usage (use with caution)"""
        self.usage_records.clear()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        logger.warning("TokenTracker reset - all usage history cleared")

    def export_records(self) -> List[Dict]:
        """
        Export all usage records as dictionaries.

        Returns:
            List of usage record dictionaries
        """
        return [record.to_dict() for record in self.usage_records]

    def get_cost_breakdown(self) -> Dict[str, Dict[str, float]]:
        """
        Get detailed cost breakdown by model.

        Returns:
            Dictionary with cost breakdown per model
        """
        summary = self.get_summary()
        return summary.by_model

    async def get_daily_cost(self, date: Optional[datetime] = None) -> float:
        """
        Get total cost for a specific day from the database.

        This queries the CostLogModel to get the actual cumulative cost
        for budget tracking and hard limit checks.

        Args:
            date: Date to query (defaults to today)

        Returns:
            Total cost in USD for the specified day

        Raises:
            RuntimeError: If database session is not configured
        """
        if not self.db_session:
            logger.warning(
                "No database session configured. "
                "Falling back to in-memory tracker (may be inaccurate)."
            )
            return self.total_cost_usd

        # Import here to avoid circular dependencies
        try:
            from sqlalchemy import select, func
            from infrastructure.database import CostLogModel
        except ImportError as e:
            logger.error(f"Failed to import database models: {e}")
            return self.total_cost_usd

        # Use today if no date specified
        if date is None:
            date = datetime.utcnow()

        # Get start and end of day
        start_of_day = datetime(date.year, date.month, date.day, 0, 0, 0)
        end_of_day = datetime(date.year, date.month, date.day, 23, 59, 59)

        try:
            # Query database for total cost in the date range
            query = select(func.sum(CostLogModel.cost_usd)).where(
                CostLogModel.timestamp >= start_of_day,
                CostLogModel.timestamp <= end_of_day
            )

            result = await self.db_session.execute(query)
            total_cost = result.scalar()

            # Handle None result (no records for this day)
            if total_cost is None:
                total_cost = 0.0

            logger.info(
                f"Daily cost for {date.strftime('%Y-%m-%d')}: ${total_cost:.4f}"
            )

            return float(total_cost)

        except Exception as e:
            logger.error(f"Failed to query daily cost from database: {e}")
            # Fallback to in-memory tracker
            return self.total_cost_usd

    async def check_hard_limit(self, daily_limit: float) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if current daily cost exceeds the hard limit.

        This is a critical safety check. If the limit is exceeded,
        the system should trigger an emergency shutdown.

        Args:
            daily_limit: Maximum allowed daily cost in USD

        Returns:
            Tuple of (limit_exceeded: bool, details: dict)
            - limit_exceeded: True if limit is exceeded
            - details: Dictionary with cost information and limit status

        Example:
            exceeded, details = await tracker.check_hard_limit(15.00)
            if exceeded:
                # Trigger emergency shutdown
                await emergency_shutdown.stop_all_agents(
                    reason=ShutdownReason.BUDGET_EXCEEDED,
                    details=details
                )
        """
        # Get actual daily cost from database
        daily_cost = await self.get_daily_cost()

        # Calculate remaining budget
        remaining_budget = daily_limit - daily_cost

        # Check if limit exceeded
        limit_exceeded = daily_cost >= daily_limit

        # Calculate utilization percentage
        utilization_pct = (daily_cost / daily_limit * 100) if daily_limit > 0 else 0

        details = {
            "daily_cost_usd": round(daily_cost, 4),
            "daily_limit_usd": round(daily_limit, 4),
            "remaining_budget_usd": round(remaining_budget, 4),
            "utilization_percent": round(utilization_pct, 2),
            "limit_exceeded": limit_exceeded,
            "checked_at": datetime.utcnow().isoformat(),
            "date": datetime.utcnow().strftime("%Y-%m-%d")
        }

        if limit_exceeded:
            logger.critical(
                f"🚨 HARD LIMIT EXCEEDED! 🚨\n"
                f"Daily Cost: ${daily_cost:.4f}\n"
                f"Daily Limit: ${daily_limit:.4f}\n"
                f"Overage: ${daily_cost - daily_limit:.4f}\n"
                f"Emergency shutdown should be triggered immediately!"
            )
        elif utilization_pct >= 90:
            logger.warning(
                f"⚠️  Budget utilization at {utilization_pct:.1f}%\n"
                f"Daily Cost: ${daily_cost:.4f} / ${daily_limit:.4f}\n"
                f"Remaining: ${remaining_budget:.4f}"
            )
        elif utilization_pct >= 75:
            logger.info(
                f"Budget utilization at {utilization_pct:.1f}% "
                f"(${daily_cost:.4f} / ${daily_limit:.4f})"
            )

        return limit_exceeded, details

    async def get_cost_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get cost summary for the last N days from database.

        Args:
            days: Number of days to include in summary

        Returns:
            Dictionary with cost statistics by day
        """
        if not self.db_session:
            logger.warning("No database session configured")
            return {"error": "Database not configured"}

        try:
            from sqlalchemy import select, func
            from infrastructure.database import CostLogModel
        except ImportError as e:
            logger.error(f"Failed to import database models: {e}")
            return {"error": str(e)}

        summary = {
            "days": days,
            "daily_costs": [],
            "total_cost": 0.0,
            "average_daily_cost": 0.0
        }

        # Get costs for each day
        for day_offset in range(days):
            date = datetime.utcnow() - timedelta(days=day_offset)
            daily_cost = await self.get_daily_cost(date)

            summary["daily_costs"].append({
                "date": date.strftime("%Y-%m-%d"),
                "cost_usd": round(daily_cost, 4)
            })

            summary["total_cost"] += daily_cost

        # Calculate average
        if days > 0:
            summary["average_daily_cost"] = round(summary["total_cost"] / days, 4)

        summary["total_cost"] = round(summary["total_cost"], 4)

        return summary

    def set_db_session(self, db_session: Any) -> None:
        """
        Set or update the database session.

        This allows the tracker to be initialized without a session
        and have one added later.

        Args:
            db_session: Async database session
        """
        self.db_session = db_session
        logger.info("Database session configured for TokenTracker")

    def __repr__(self) -> str:
        db_status = "DB-enabled" if self.db_session else "memory-only"
        return (
            f"TokenTracker("
            f"records={len(self.usage_records)}, "
            f"total_tokens={self.total_input_tokens + self.total_output_tokens}, "
            f"total_cost=${self.total_cost_usd:.6f}, "
            f"mode={db_status})"
        )
