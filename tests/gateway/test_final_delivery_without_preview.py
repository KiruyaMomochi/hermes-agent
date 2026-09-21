"""Final delivery confirmation without preview flag."""


class _MockConsumer:
    """Minimal consumer exposing the durable-delivery predicate."""

    def __init__(self, *, final_sent=False, delivered_text=None):
        self.final_response_sent = final_sent
        self._delivered = delivered_text

    def has_durably_delivered_text(self, text):
        return text == self._delivered


def test_has_durable_delivered_text_checked_without_preview():
    """The durable-delivery predicate is checked even when previewed=False."""
    from gateway.run import GatewayRunner

    consumer = _MockConsumer(final_sent=False, delivered_text="Already sent")
    # previewed=False, but durable delivery should still be consulted
    result = GatewayRunner._run_agent_stream_confirmed_final_delivery(
        consumer, "Already sent", previewed=False
    )
    assert result is True


def test_has_durable_delivered_text_returns_false_on_mismatch():
    """The durable predicate correctly identifies undelivered finals."""
    from gateway.run import GatewayRunner

    consumer = _MockConsumer(final_sent=False, delivered_text="Different text")
    result = GatewayRunner._run_agent_stream_confirmed_final_delivery(
        consumer, "New final", previewed=False
    )
    assert result is False
