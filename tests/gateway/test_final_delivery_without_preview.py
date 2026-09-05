"""Final delivery confirmation without preview flag."""


class _MockConsumer:
    """Minimal consumer for testing has_delivered_text fallback."""

    def __init__(self, *, final_sent=False, delivered_text=None):
        self.final_response_sent = final_sent
        self._delivered = delivered_text

    def has_delivered_text(self, text):
        return text == self._delivered


def test_has_delivered_text_checked_without_preview():
    """has_delivered_text is called even when previewed=False."""
    from gateway.run import GatewayRunner

    consumer = _MockConsumer(final_sent=False, delivered_text="Already sent")
    # previewed=False, but has_delivered_text should still be consulted
    result = GatewayRunner._run_agent_stream_confirmed_final_delivery(
        consumer, "Already sent", previewed=False
    )
    assert result is True


def test_has_delivered_text_returns_false_on_mismatch():
    """has_delivered_text correctly identifies undelivered finals."""
    from gateway.run import GatewayRunner

    consumer = _MockConsumer(final_sent=False, delivered_text="Different text")
    result = GatewayRunner._run_agent_stream_confirmed_final_delivery(
        consumer, "New final", previewed=False
    )
    assert result is False
