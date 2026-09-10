"""Regression: the consumer-declared final + interim-send contract
(live findings, 2026-08-16 canary — the duplicate-final class).

Three invariants:

1. ``finish(final_text=...)`` makes the finalize payload the AUTHORITATIVE
   completed final_response — post-stream augmentation (file-mutation
   verifier footer) rides the seal/final edit instead of arriving via a
   separate corrective send (live finding #11).

2. Interim sends (commentary) from the consumer carry ``_interim_send``
   metadata, and the relay adapter's seal-interception ignores them — a
   mid-turn commentary must never seal the live native stream (which
   orphaned the true final into a plain-send duplicate).

3. The queued-follow-up lane reconciles an unconfirmed final by EDITING the
   consumer's delivered message in place, not by plain-sending a duplicate.
"""

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig


def _make_draft_adapter():
    """BasePlatformAdapter subclass with stream-is-the-message drafts."""
    from gateway.platforms.base import BasePlatformAdapter, SendResult

    A = type("StreamIsMsgAdapter", (BasePlatformAdapter,), {"MAX_MESSAGE_LENGTH": 39000})
    A.__abstractmethods__ = frozenset()
    a = A.__new__(A)
    a._typing_paused = set()
    a._fatal_error_message = None
    a.draft_stream_is_message = True
    a.draft_calls = []
    a.send_calls = []
    a.edit_calls = []

    def _supports(chat_type=None, metadata=None):
        return True
    a.supports_draft_streaming = _supports

    async def _send_draft(*, chat_id, draft_id, content, metadata=None):
        a.draft_calls.append({"draft_id": draft_id, "content": content})
        return SendResult(success=True, message_id=None)
    a.send_draft = _send_draft

    async def _send(chat_id, content, reply_to=None, metadata=None, **kw):
        a.send_calls.append({"content": content, "metadata": dict(metadata or {})})
        return SendResult(success=True, message_id="sealed_ts_1")
    a.send = _send

    async def _edit(chat_id, message_id, content, **kw):
        a.edit_calls.append({"message_id": message_id, "content": content})
        return SendResult(success=True, message_id=message_id)
    a.edit_message = _edit
    return a


class TestConsumerDeclaredFinal:
    def test_structured_reasoning_delta_has_a_dedicated_sink(self):
        adapter = _make_draft_adapter()
        sc = GatewayStreamConsumer(adapter, "D1")

        sc.on_reasoning_delta("checking ")
        sc.on_reasoning_delta("the answer\n---\nstill reasoning")

        assert sc.reasoning_text == "checking the answer\n---\nstill reasoning"
        assert sc._accumulated == "", "raw reasoning must not enter the answer stream"

        sc.on_delta(None)
        sc.on_reasoning_delta("final reasoning")
        assert sc.reasoning_text == "final reasoning", "only the last reasoning block is displayed"

    def test_turn_callback_wiring_registers_reasoning_sink(self, monkeypatch):
        from gateway.run_turn_runner import TurnRunner

        consumer = SimpleNamespace(parts=[], on_reasoning_delta=lambda text: consumer.parts.append(text))
        ctx = SimpleNamespace(
            stream_consumer_holder=[consumer], progress_callback=None,
            _voice_ack_guild=[None], _native_slack_task_cards=False,
            native_tool_start_callback=None, voice_ack_callback=None,
            native_tool_complete_callback=None, _hooks_ref=SimpleNamespace(loaded_hooks=[]),
            _step_callback_sync=None, _status_callback_sync=None, _event_callback_sync=None,
            user_config={}, _status_adapter=None, session_key="s1", _thinking_enabled=False,
            agent_holder=[None], tools_holder=[None], source=SimpleNamespace(),
            process_task_id=None, process_baseline=None, _interrupt_depth=0,
        )
        runner = SimpleNamespace(
            _service_tier=None, _consume_pending_turn_sidecar_notes=lambda _key: [],
        )
        turn_runner = TurnRunner(cast(Any, runner), cast(Any, ctx))
        monkeypatch.setattr(turn_runner, "_attach_session_title_callback", lambda *_args, **_kwargs: None)
        agent = SimpleNamespace(request_overrides={}, tools=[])

        turn_runner._wire_turn_agent_callbacks(
            agent, {"request_overrides": {}}, None, lambda text: None, None, False,
        )
        agent.reasoning_callback("structured")

        assert consumer.parts == ["structured"]

    @pytest.mark.asyncio
    async def test_finish_final_text_rides_the_final_send(self):
        """The footer-bearing final_response must BE the finalize payload —
        one message, no separate corrective send needed."""
        adapter = _make_draft_adapter()
        cfg = StreamConsumerConfig(
            transport="auto", chat_type="dm",
            edit_interval=0.01, buffer_threshold=1, cursor="",
        )
        sc = GatewayStreamConsumer(adapter, "D1", cfg)

        task = asyncio.create_task(sc.run())
        sc.on_delta("streamed answer body")
        await asyncio.sleep(0.06)
        # Turn completes; turn_finalizer appended the verifier footer.
        final_with_footer = (
            "streamed answer body\n\n"
            "⚠️ File-mutation verifier: 1 file(s) were NOT modified this turn."
        )
        sc.finish(final_with_footer)
        await task

        # The turn-final send carried the COMPLETE footer-bearing final.
        assert adapter.send_calls, "expected a turn-final send"
        assert adapter.send_calls[-1]["content"] == final_with_footer
        # And the recorded payload reconciles → gateway suppression is safe.
        assert sc.delivered_final_matches(final_with_footer) is True

    @pytest.mark.asyncio
    async def test_finish_bare_keeps_legacy_behavior(self):
        adapter = _make_draft_adapter()
        cfg = StreamConsumerConfig(
            transport="auto", chat_type="dm",
            edit_interval=0.01, buffer_threshold=1, cursor="",
        )
        sc = GatewayStreamConsumer(adapter, "D1", cfg)
        task = asyncio.create_task(sc.run())
        sc.on_delta("plain answer")
        await asyncio.sleep(0.06)
        sc.finish()
        await task
        assert adapter.send_calls[-1]["content"] == "plain answer"

    @pytest.mark.parametrize(
        ("streamed_reasoning", "fallback_reasoning", "expected_reasoning"),
        [
            ("structured delta", "stored fallback", "structured delta"),
            ("", "stored fallback", "stored fallback"),
        ],
    )
    def test_turn_runner_decorates_stream_final_from_sink_or_fallback(
        self, streamed_reasoning, fallback_reasoning, expected_reasoning,
    ):
        from gateway.run_turn_runner import TurnRunner

        finished = []
        consumer = SimpleNamespace(
            reasoning_text=streamed_reasoning,
            finish=lambda text=None: finished.append(text),
        )
        ctx = SimpleNamespace(result_holder=[None], source=SimpleNamespace(platform="telegram"))

        class _Runner:
            @staticmethod
            def _hmwa_prepend_reasoning(result, response, source, intentional_silence):
                assert source is ctx.source
                assert intentional_silence is False
                return f"REASONING[{result['last_reasoning']}]\n{response}"

        turn_runner = TurnRunner(cast(Any, _Runner()), cast(Any, ctx))
        result = {
            "final_response": "final answer", "last_reasoning": fallback_reasoning,
            "completed": True,
        }

        displayed = turn_runner._finish_stream_consumer(result, [], consumer)

        assert displayed == f"REASONING[{expected_reasoning}]\nfinal answer"
        assert finished == [displayed]
        assert result["final_response"] == "final answer", "canonical final must remain unchanged"


class TestInterimSendContract:
    @pytest.mark.asyncio
    async def test_commentary_is_marked_interim(self):
        adapter = _make_draft_adapter()
        cfg = StreamConsumerConfig(
            transport="auto", chat_type="dm",
            edit_interval=0.01, buffer_threshold=1, cursor="",
        )
        sc = GatewayStreamConsumer(adapter, "D1", cfg)
        ok = await sc._send_commentary("Using the browser tool…")
        assert ok is True
        assert adapter.send_calls[-1]["metadata"].get("_interim_send") is True

    @pytest.mark.asyncio
    async def test_relay_adapter_interim_send_does_not_seal(self):
        """An armed open draft must survive an interim send untouched."""
        from tests.gateway.relay.test_relay_live_cards import _connected_adapter

        adapter, _ = _connected_adapter(
            supported_ops=("send", "edit", "typing", "draft"),
        )

        class _T:
            def __init__(self):
                self.ops = []
            async def send_outbound(self, payload, platform=None):
                self.ops.append(dict(payload))
                return {"success": True, "message_id": "111.222"}

        t = _T()
        adapter._transport = t
        md = {"thread_ts": "1700.42"}
        await adapter.send_draft("C1", 5, "streaming...", metadata=md)
        key = adapter._draft_key("C1", md)
        assert adapter._open_draft_by_chat.get(key) == 5

        # Interim commentary while the stream is open: must NOT seal.
        res = await adapter.send(
            "C1", "delegation dispatched, continuing…",
            metadata={**md, "_interim_send": True},
        )
        assert res.success
        assert adapter._open_draft_by_chat.get(key) == 5, "interim send sealed the stream"
        sent_ops = [o for o in t.ops if o["op"] == "send"]
        assert len(sent_ops) == 1
        # Marker never leaks to the wire.
        assert "_interim_send" not in (sent_ops[0].get("metadata") or {})

        # The true turn-final still seals.
        final = await adapter.send("C1", "streaming... done.", metadata=md)
        assert final.success
        assert key not in adapter._open_draft_by_chat
        seals = [o for o in t.ops if o["op"] == "draft" and o.get("final")]
        assert len(seals) == 1

    @pytest.mark.asyncio
    async def test_send_for_platform_interim_does_not_seal(self):
        """The delivery-resolver egress door honors the interim contract too.

        send_for_platform is the second seal-interception site (finding #7);
        an interim send routed through it must neither seal the open stream
        nor leak the gateway-internal marker onto the wire.
        """
        from tests.gateway.relay.test_relay_live_cards import _connected_adapter

        adapter, _ = _connected_adapter(
            supported_ops=("send", "edit", "typing", "draft"),
        )

        class _T:
            def __init__(self):
                self.ops = []
                # fronts_platform reads the handshake identity set off the
                # transport; advertise slack so send_for_platform proceeds.
                self._identities = [("slack", "U-bot")]
            async def send_outbound(self, payload, platform=None):
                self.ops.append(dict(payload))
                return {"success": True, "message_id": "111.333"}

        t = _T()
        adapter._transport = t
        md = {"thread_ts": "1700.77"}
        await adapter.send_draft("C2", 9, "streaming...", metadata=md)
        key = adapter._draft_key("C2", md)
        assert adapter._open_draft_by_chat.get(key) == 9

        res = await adapter.send_for_platform(
            "slack", "C2", "interim status", metadata={**md, "_interim_send": True},
        )
        assert res.success
        assert adapter._open_draft_by_chat.get(key) == 9, "interim send_for_platform sealed the stream"
        sent_ops = [o for o in t.ops if o["op"] == "send"]
        assert len(sent_ops) == 1
        assert "_interim_send" not in (sent_ops[0].get("metadata") or {})


class TestFinalAdoptionGuards:
    @pytest.mark.asyncio
    async def test_no_stream_turn_does_not_adopt_final(self):
        """finish(final_text) on a turn that never streamed must not move
        delivery ownership into the consumer — the gateway's normal final
        send path owns those turns (non-streaming models, tool-only turns)."""
        adapter = _make_draft_adapter()
        cfg = StreamConsumerConfig(
            transport="auto", chat_type="dm",
            edit_interval=0.01, buffer_threshold=1, cursor="",
        )
        sc = GatewayStreamConsumer(adapter, "D1", cfg)
        task = asyncio.create_task(sc.run())
        # No on_delta at all — straight to completion with a payload.
        sc.finish("the final answer from a non-streaming turn")
        await task
        assert adapter.send_calls == [], "no-stream turn must not deliver via the consumer"
        assert adapter.draft_calls == []


class TestQueuedLaneReconcile:
    @pytest.mark.asyncio
    async def test_decorated_stream_delivery_suppresses_canonical_resend(self):
        """The consumer ACKs the decorated payload while canonical text remains separate."""
        from gateway.run import GatewayRunner

        decorated = "💭 **Reasoning:**\n```\nchecking\n```\n\nanswer"
        seen = []
        consumer = SimpleNamespace(
            final_response_sent=True,
            final_content_delivered=True,
            delivered_final_matches=lambda text: seen.append(text) or text == decorated,
        )
        response = {
            "final_response": decorated,
            "_canonical_final_response": "answer",
            "_streamed_delivery_response": decorated,
        }
        turn_ctx = SimpleNamespace(
            stream_consumer_holder=[consumer], source=SimpleNamespace(chat_id="D1"),
            session_key="session-1",
        )

        await GatewayRunner._run_agent_mark_streamed_delivery(
            object.__new__(GatewayRunner), response, cast(Any, turn_ctx),
        )

        assert response["already_sent"] is True
        assert seen == [decorated, decorated]

    @pytest.mark.asyncio
    async def test_reasoning_display_does_not_change_canonical_match(self):
        """Gateway presentation must not be used to reconcile the streamed final."""
        from gateway.run import GatewayRunner

        seen = []
        consumer = SimpleNamespace(
            final_response_sent=True,
            final_content_delivered=True,
            delivered_final_matches=lambda text: seen.append(text) or True,
        )
        response = {
            "final_response": "> 💭 **Reasoning:**\n> checking\n\nanswer",
            "_canonical_final_response": "answer",
        }
        turn_ctx = SimpleNamespace(
            stream_consumer_holder=[consumer], source=SimpleNamespace(chat_id="D1"),
            session_key="session-1",
        )
        await GatewayRunner._run_agent_mark_streamed_delivery(
            object.__new__(GatewayRunner), response, turn_ctx,
        )
        assert response["already_sent"] is True
        assert seen == ["answer", "answer"]

    @pytest.mark.asyncio
    async def test_split_canonical_delivery_suppresses_corrective_combined_send(self):
        """Split chunks reconcile against the canonical final, not its display wrapper."""
        from gateway.run import GatewayRunner

        seen = []
        consumer = SimpleNamespace(
            final_response_sent=True, final_content_delivered=True,
            _turn_split_delivery=True,
            delivered_final_matches=lambda text: seen.append(text) or True,
        )
        response = {
            "final_response": "> 💭 **Reasoning:**\n> checking\n\nlong split answer",
            "_canonical_final_response": "long split answer",
        }
        turn_ctx = SimpleNamespace(
            stream_consumer_holder=[consumer], source=SimpleNamespace(chat_id="D1"),
            session_key="session-1",
        )
        await GatewayRunner._run_agent_mark_streamed_delivery(
            object.__new__(GatewayRunner), response, turn_ctx,
        )
        assert response["already_sent"] is True
        assert seen == ["long split answer", "long split answer"]

    @pytest.mark.asyncio
    async def test_queued_first_response_edits_in_place(self):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        adapter = _make_draft_adapter()
        sc = SimpleNamespace(message_id="sealed_ts_9", _turn_split_delivery=False)
        source = SimpleNamespace(chat_id="D1")
        await GatewayRunner._deliver_queued_first_response(
            runner,
            "the complete final with footer",
            source=source,
            adapter=adapter,
            metadata=None,
            text_already_delivered=False,
            deliver_media=False,
            stream_consumer=sc,
        )
        # Edited the sealed message; did NOT plain-send a duplicate.
        assert adapter.edit_calls == [
            {"message_id": "sealed_ts_9", "content": "the complete final with footer"}
        ]
        assert adapter.send_calls == []

    @pytest.mark.asyncio
    async def test_queued_first_response_falls_back_without_message(self):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        adapter = _make_draft_adapter()
        sc = SimpleNamespace(message_id=None, _turn_split_delivery=False)
        source = SimpleNamespace(chat_id="D1")
        await GatewayRunner._deliver_queued_first_response(
            runner,
            "final text",
            source=source,
            adapter=adapter,
            metadata=None,
            text_already_delivered=False,
            deliver_media=False,
            stream_consumer=sc,
        )
        assert adapter.edit_calls == []
        assert len(adapter.send_calls) == 1
