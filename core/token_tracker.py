"""
Token Tracker - Token Counting and Cost Calculation
Tracks API usage and calculates costs for Claude models
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
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
    """

    def __init__(self):
        """Initialize the token tracker"""
        self.usage_records: List[UsageRecord] = []
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cost_usd: float = 0.0

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

    def __repr__(self) -> str:
        return (
            f"TokenTracker("
            f"records={len(self.usage_records)}, "
            f"total_tokens={self.total_input_tokens + self.total_output_tokens}, "
            f"total_cost=${self.total_cost_usd:.6f})"
        )
