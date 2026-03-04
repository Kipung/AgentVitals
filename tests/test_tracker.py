import time
from agent_vitals.tracker import VitalsTracker
from agent_vitals.tokenizer import count_tokens


def test_tracker_initialization():
    """Tests that the tracker initializes correctly."""
    tracker = VitalsTracker(model_name="gpt-4")
    assert tracker.model_name == "gpt-4"
    assert tracker.total_tokens == 0
    assert tracker.input_tokens == 0
    assert tracker.output_tokens == 0
    assert len(tracker.interactions) == 0


def test_add_interaction_and_token_count():
    """Tests that interactions are added and tokens are counted."""
    tracker = VitalsTracker()
    user_text = "Hello world"
    agent_text = "Hi there!"

    tracker.add_interaction(source="user", text=user_text)
    assert tracker.total_tokens == count_tokens(user_text)
    assert tracker.input_tokens == count_tokens(user_text)
    assert len(tracker.interactions) == 1

    tracker.add_interaction(source="agent", text=agent_text)
    assert tracker.total_tokens == count_tokens(user_text) + count_tokens(agent_text)
    assert tracker.output_tokens == count_tokens(agent_text)
    assert len(tracker.interactions) == 2
    assert tracker.interactions[1]["source"] == "agent"


def test_cost_calculation():
    """Tests the cost calculation logic by using the real tokenizer."""
    tracker = VitalsTracker(model_name="gpt-3.5-turbo")

    # Cost factors for gpt-3.5-turbo
    input_price_per_mil = 0.50
    output_price_per_mil = 1.50

    # Calculate cost for input
    input_text = "a " * 500
    input_tokens = count_tokens(input_text)
    tracker.add_interaction(source="user", text=input_text)
    expected_input_cost = (input_tokens / 1_000_000) * input_price_per_mil
    assert tracker.get_cost() == expected_input_cost
    
    # Calculate cost for output and total
    output_text = "a " * 1000
    output_tokens = count_tokens(output_text)
    tracker.add_interaction(source="agent", text=output_text)
    expected_output_cost = (output_tokens / 1_000_000) * output_price_per_mil
    total_expected_cost = expected_input_cost + expected_output_cost
    assert tracker.get_cost() == total_expected_cost


def test_response_time_calculation():
    """Tests that response time is only calculated for the agent."""
    tracker = VitalsTracker()

    tracker.add_interaction(source="user", text="First prompt")
    # No response time for user
    assert tracker.interactions[0]["response_time_seconds"] is None

    time.sleep(0.1)  # Simulate agent thinking time

    tracker.add_interaction(source="agent", text="First response")
    # Response time should be calculated for agent
    assert tracker.interactions[1]["response_time_seconds"] is not None
    assert tracker.interactions[1]["response_time_seconds"] >= 0.1
    
    summary = tracker.get_summary()
    assert summary["average_agent_response_time_seconds"] >= 0.1


def test_summary_output():
    """Tests the structure and types of the summary dictionary."""
    tracker = VitalsTracker()
    tracker.add_interaction(source="user", text="Hello")
    summary = tracker.get_summary()

    assert isinstance(summary, dict)
    assert "total_tokens" in summary
    assert "estimated_cost_usd" in summary
    assert isinstance(summary["estimated_cost_usd"], str)
    assert summary["estimated_cost_usd"].startswith("$")
    assert "context_window_tokens" in summary
    assert "context_remaining_tokens" in summary
    assert "context_utilization_pct" in summary
    assert "tokens_per_minute" in summary


def test_context_window_tracking_and_overflow():
    """Tests context window usage metrics and overflow handling."""
    tracker = VitalsTracker(model_name="gpt-4", context_window_tokens=10)
    tracker.add_interaction(source="user", text="a " * 100)

    summary = tracker.get_summary()
    assert summary["context_window_tokens"] == 10
    assert summary["context_remaining_tokens"] == 0
    assert summary["is_context_window_exceeded"] is True
    assert summary["context_utilization_pct"] > 100


def test_default_model_context_window_lookup():
    """Tests model-based context window lookup when not overridden."""
    tracker = VitalsTracker(model_name="gpt-4-turbo")
    tracker.add_interaction(source="user", text="hello")

    summary = tracker.get_summary()
    assert summary["context_window_tokens"] == 128_000
    assert summary["context_remaining_tokens"] < 128_000
    assert summary["is_context_window_exceeded"] is False
