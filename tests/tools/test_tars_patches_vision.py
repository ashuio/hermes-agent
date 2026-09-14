"""Tests for the TARS patches: native vision for tool-result and MCP images.

Covers the four patches:
  P1  agent/error_classifier.py    — Hyper Charm rejection pattern
  P2  agent/vision_message_prep.py — promote tool images to a user message
  P3  agent/turn_recovery.py       — promote-first wiring (covered indirectly)
  P4  tools/mcp_tool_content.py + tools/mcp_tool_handlers.py — MCP image parts

These are downstream patches, not upstream code. If this file fails after a
`hermes update`, the patches were lost — run
/root/.hermes/scripts/hermes-patch-drift-check.sh
"""
from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import pytest


def _png_b64() -> str:
    """Minimal valid 1x1 PNG, base64-encoded (matches the existing image tests)."""
    return base64.b64encode(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        )
    ).decode("ascii")


# ── P1: classifier pattern ────────────────────────────────────────────────

class TestClassifierPattern:
    def test_hypercharm_rejection_is_recognised(self):
        """Hyper Charm's empty-type-name 400 must match the multimodal pattern list.

        Without this the recovery path never fires and the client sees a
        misleading replayed 402 from the last key instead of the real error.
        """
        from agent.error_classifier import _MULTIMODAL_TOOL_CONTENT_PATTERNS as P
        msg = '{"error":{"message":"unsupported input item type: ","type":"invalid_request_error"}}'
        assert any(p in msg.lower() for p in P), (
            "Hyper Charm's 'unsupported input item type' is not in "
            "_MULTIMODAL_TOOL_CONTENT_PATTERNS — P1 is missing"
        )


# ── P2: promote tool images to a user message ─────────────────────────────

class TestPromoteImageParts:
    def _agent(self):
        from agent.vision_message_prep import VisionMessagePrepMixin

        class _A(VisionMessagePrepMixin):
            provider = "custom:bifrost"
            model = "Hyper Charm/deepseek-v4.1-flash"

        return _A()

    def _tool_msg_with_image(self):
        return {
            "role": "tool",
            "tool_call_id": "c1",
            "content": [
                {"type": "text", "text": "Screenshot captured."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        }

    def test_moves_image_to_a_user_message(self):
        agent = self._agent()
        msgs = [{"role": "user", "content": "shot?"}, self._tool_msg_with_image()]

        assert agent._try_promote_image_parts_to_user_message(msgs) is True

        # The tool message keeps its text and loses the image.
        tool_msg = next(m for m in msgs if m.get("role") == "tool")
        assert isinstance(tool_msg["content"], str)
        assert "Screenshot captured." in tool_msg["content"]
        assert "image_url" not in json.dumps(tool_msg)

        # A user message now carries the image.
        user_msgs = [m for m in msgs if m.get("role") == "user"]
        assert len(user_msgs) == 2
        promoted = user_msgs[-1]["content"]
        assert any(p.get("type") == "image_url" for p in promoted)

    def test_returns_false_when_no_images(self):
        agent = self._agent()
        msgs = [{"role": "tool", "tool_call_id": "c1", "content": "plain text"}]
        assert agent._try_promote_image_parts_to_user_message(msgs) is False
        assert len(msgs) == 1  # nothing appended

    def test_returns_false_for_non_list_input(self):
        assert self._agent()._try_promote_image_parts_to_user_message(None) is False  # type: ignore[arg-type]

    def test_strip_still_discards(self):
        """The original strip path must remain available as the fallback."""
        agent = self._agent()
        msgs = [self._tool_msg_with_image()]
        assert agent._try_strip_image_parts_from_tool_messages(msgs) is True
        assert "image_url" not in json.dumps(msgs)
        assert len(msgs) == 1  # strip does NOT append a user message


# ── P4: MCP image parts ───────────────────────────────────────────────────

class TestMcpImagePart:
    def test_builds_image_url_part(self):
        from tools.mcp_tool_content import _mcp_image_part

        block = SimpleNamespace(data=_png_b64(), mimeType="image/png")
        part = _mcp_image_part(block)
        assert part is not None
        assert part["type"] == "image_url"
        assert part["image_url"]["url"].startswith("data:image/png;base64,")

    def test_none_for_non_image_mime(self):
        from tools.mcp_tool_content import _mcp_image_part

        block = SimpleNamespace(data=_png_b64(), mimeType="application/pdf")
        assert _mcp_image_part(block) is None

    def test_none_when_no_data(self):
        from tools.mcp_tool_content import _mcp_image_part

        assert _mcp_image_part(SimpleNamespace(mimeType="image/png")) is None


class TestRenderContentBlocksReturnsImages:
    def test_returns_three_tuple_with_image_parts(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_tool_handlers import _render_content_blocks

        result = SimpleNamespace(content=[
            SimpleNamespace(type="text", text="here is the shot"),
            SimpleNamespace(type="image", data=_png_b64(), mimeType="image/png"),
        ])
        text, usable, images = _render_content_blocks(result, "test-server")

        assert "here is the shot" in text
        assert usable >= 1
        assert len(images) == 1
        assert images[0]["type"] == "image_url"

    def test_no_images_yields_empty_list(self):
        from tools.mcp_tool_handlers import _render_content_blocks

        result = SimpleNamespace(content=[SimpleNamespace(type="text", text="just text")])
        text, usable, images = _render_content_blocks(result, "test-server")
        assert images == []


class TestRenderCallToolResultEnvelope:
    def test_returns_multimodal_envelope_when_images_present(self, tmp_path, monkeypatch):
        """The registry only accepts a dict with _multimodal=True and a content list."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_tool_handlers import _render_call_tool_result

        result = SimpleNamespace(
            isError=False,
            content=[
                SimpleNamespace(type="text", text="shot taken"),
                SimpleNamespace(type="image", data=_png_b64(), mimeType="image/png"),
            ],
        )
        out = _render_call_tool_result(result, "test-server")

        assert isinstance(out, dict), f"expected envelope dict, got {type(out).__name__}"
        assert out["_multimodal"] is True
        assert isinstance(out["content"], list)
        assert any(p.get("type") == "image_url" for p in out["content"])
        assert out["text_summary"]

    def test_still_returns_json_string_without_images(self):
        from tools.mcp_tool_handlers import _render_call_tool_result

        result = SimpleNamespace(
            isError=False, content=[SimpleNamespace(type="text", text="no images here")]
        )
        out = _render_call_tool_result(result, "test-server")
        assert isinstance(out, str)
        assert json.loads(out)["result"] == "no images here"

    def test_envelope_passes_registry_normalisation(self, tmp_path, monkeypatch):
        """End-to-end contract: the registry must accept what the handler returns."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_tool_handlers import _render_call_tool_result
        from tools.registry import ToolRegistry

        result = SimpleNamespace(
            isError=False,
            content=[
                SimpleNamespace(type="text", text="shot"),
                SimpleNamespace(type="image", data=_png_b64(), mimeType="image/png"),
            ],
        )
        envelope = _render_call_tool_result(result, "test-server")
        normalised = ToolRegistry._normalize_handler_result("mcp_test_shot", envelope)
        assert normalised is envelope, (
            "registry rejected the envelope — it would become a tool_result_contract error"
        )
