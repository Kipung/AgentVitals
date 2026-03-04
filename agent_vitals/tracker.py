from datetime import datetime, timezone
from typing import Dict, Optional
from .tokenizer import count_tokens
from .pricing import get_context_window, get_price


class VitalsTracker:
    """
    Tracks the vitals of an AI agent session, including token usage,
    time, interaction count, and cost.
    """

    def __init__(
        self,
        model_name: str = "unknown",
        encoding_name: str = "cl100k_base",
        context_window_tokens: Optional[int] = None,
    ):
        self.model_name = model_name
        self.encoding_name = encoding_name
        self.start_time = datetime.now(timezone.utc)
        self.interactions = []
        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.last_interaction_time = self.start_time
        if context_window_tokens is None:
            context_window_tokens = get_context_window(model_name)
        if context_window_tokens is not None and context_window_tokens <= 0:
            raise ValueError("context_window_tokens must be positive")
        self.context_window_tokens = context_window_tokens

    def add_interaction(self, source: str, text: str):
        """
        Adds a new interaction (e.g., from a user or agent) to the tracker.

        Args:
            source: Who sent the text ("user", "agent", etc.).
            text: The content of the interaction.
        """
        timestamp = datetime.now(timezone.utc)
        tokens = count_tokens(text, self.encoding_name)

        response_time_seconds = None
        # Treat 'user' as input and 'agent' as output for pricing
        token_type = "input" if source.lower() == "user" else "output"

        if token_type == "output":
            response_time_seconds = (
                timestamp - self.last_interaction_time
            ).total_seconds()

        interaction = {
            "source": source,
            "text": text,
            "tokens": tokens,
            "token_type": token_type,
            "timestamp": timestamp,
            "response_time_seconds": response_time_seconds,
        }

        self.interactions.append(interaction)
        self.total_tokens += tokens
        if token_type == "input":
            self.input_tokens += tokens
        else:
            self.output_tokens += tokens
        self.last_interaction_time = timestamp

    def get_cost(self) -> float:
        """Calculates the estimated cost of the session."""
        total_cost = 0.0
        for interaction in self.interactions:
            tokens = interaction["tokens"]
            token_type = interaction["token_type"]
            price_per_million = get_price(self.model_name, token_type)
            cost = (tokens / 1_000_000) * price_per_million
            total_cost += cost
        return total_cost

    def get_context_usage(self) -> Dict[str, Optional[float]]:
        """
        Return context window usage values.

        If the model context window is unknown, values are None.
        """
        if self.context_window_tokens is None:
            return {
                "context_window_tokens": None,
                "context_remaining_tokens": None,
                "context_utilization_pct": None,
                "is_context_window_exceeded": None,
            }

        used_tokens = self.total_tokens
        remaining_tokens = max(self.context_window_tokens - used_tokens, 0)
        utilization_pct = (used_tokens / self.context_window_tokens) * 100
        is_exceeded = used_tokens > self.context_window_tokens

        return {
            "context_window_tokens": self.context_window_tokens,
            "context_remaining_tokens": remaining_tokens,
            "context_utilization_pct": utilization_pct,
            "is_context_window_exceeded": is_exceeded,
        }

    def get_tokens_per_minute(self) -> float:
        """Return total token throughput for the current session."""
        duration_seconds = (
            datetime.now(timezone.utc) - self.start_time
        ).total_seconds()
        if duration_seconds <= 0:
            return 0.0
        return self.total_tokens / (duration_seconds / 60)

    def get_summary(self) -> dict:
        """
        Returns a summary of the current session vitals.
        """
        duration_seconds = (
            datetime.now(timezone.utc) - self.start_time
        ).total_seconds()

        agent_interactions = [
            i
            for i in self.interactions
            if i["token_type"] == "output" and i["response_time_seconds"] is not None
        ]

        total_response_time = sum(
            i["response_time_seconds"] for i in agent_interactions
        )
        average_response_time = (
            total_response_time / len(agent_interactions) if agent_interactions else 0
        )
        context_usage = self.get_context_usage()

        return {
            "model_name": self.model_name,
            "start_time": self.start_time.isoformat(),
            "duration_seconds": round(duration_seconds, 2),
            "interaction_count": len(self.interactions),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": f"${self.get_cost():.6f}",
            "average_agent_response_time_seconds": round(average_response_time, 2),
            "tokens_per_minute": round(self.get_tokens_per_minute(), 2),
            "context_window_tokens": context_usage["context_window_tokens"],
            "context_remaining_tokens": context_usage["context_remaining_tokens"],
            "context_utilization_pct": (
                round(context_usage["context_utilization_pct"], 2)
                if context_usage["context_utilization_pct"] is not None
                else None
            ),
            "is_context_window_exceeded": context_usage["is_context_window_exceeded"],
        }

    def __repr__(self) -> str:
        return (
            f"VitalsTracker(model='{self.model_name}', "
            f"tokens={self.total_tokens}, "
            f"context_window={self.context_window_tokens}, "
            f"interactions={len(self.interactions)})"
        )


if __name__ == "__main__":
    # Example Usage
    tracker = VitalsTracker(model_name="gpt-4")

    # Simulate a user prompt
    user_prompt = "Hello, Agent! Please write me a short story."
    tracker.add_interaction(source="user", text=user_prompt)
    print(f"After user prompt: {tracker.get_summary()}")

    # Simulate an agent response
    agent_response = "Once upon a time, in a land of code..."
    tracker.add_interaction(source="agent", text=agent_response)
    print(f"After agent response: {tracker.get_summary()}")

    print(f"\nFinal state: {tracker}")
