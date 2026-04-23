"""
Tests for LLMAdapter tool-calling logic.
These tests do NOT require a running llama-server — the HTTP call is mocked.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.adapters.llm_adapter import LLMAdapter
from src.config import AppSettings
from src.models import CommandDefinition, ConditionalCommand


@pytest.fixture
def command_defs() -> list[CommandDefinition]:
    return [
        CommandDefinition(name="call",   spellings=["call", "cold"],      n_slots=1, slots_type=["users"]),
        CommandDefinition(name="photo",  spellings=["photo", "picture"],  n_slots=0, slots_type=[]),
        CommandDefinition(name="vitals", spellings=["vitals"],            n_slots=0, slots_type=[]),
    ]


@pytest.fixture
def adapter(command_defs) -> LLMAdapter:
    settings = AppSettings()
    return LLMAdapter(settings, command_defs)


# ---------------------------------------------------------------------------
# Helpers: build fake /v1/chat/completions responses
# ---------------------------------------------------------------------------

def _exec_response(*commands: dict) -> dict:
    """Fake execute_commands tool call with the given command dicts."""
    return {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "execute_commands",
                        "arguments": json.dumps({"commands": list(commands)}),
                    },
                }],
            },
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _tool_response(name: str, args: dict) -> dict:
    """Fake response for a single named tool call (conditional, get_weather, etc.)."""
    return {
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }],
            },
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _text_response(content: str) -> dict:
    """Fake response where Qwen replies with plain text (no tool calls)."""
    return {
        "choices": [{
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": content, "tool_calls": None},
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8},
    }


# ---------------------------------------------------------------------------
# Tool-building tests (no HTTP needed)
# ---------------------------------------------------------------------------

class TestBuildTools:
    def test_execute_commands_tool_present(self, adapter):
        names = {t["function"]["name"] for t in adapter._tools}
        assert "execute_commands" in names

    def test_no_individual_command_tools(self, adapter):
        names = {t["function"]["name"] for t in adapter._tools}
        assert "photo" not in names
        assert "call" not in names

    def test_conditional_tool_present(self, adapter):
        names = {t["function"]["name"] for t in adapter._tools}
        assert "conditional" in names

    def test_data_tools_present(self, adapter):
        names = {t["function"]["name"] for t in adapter._tools}
        assert "get_weather" in names

    def test_execute_commands_enum_matches_commands(self, adapter):
        exec_tool = next(t for t in adapter._tools if t["function"]["name"] == "execute_commands")
        enum_vals = (
            exec_tool["function"]["parameters"]
            ["properties"]["commands"]["items"]
            ["properties"]["command"]["enum"]
        )
        assert set(enum_vals) == {"call", "photo", "vitals"}

    def test_conditional_tool_enum_matches_commands(self, adapter):
        cond_tool = next(t for t in adapter._tools if t["function"]["name"] == "conditional")
        enum_vals = (
            cond_tool["function"]["parameters"]
            ["properties"]["then_commands"]["items"]
            ["properties"]["command"]["enum"]
        )
        assert set(enum_vals) == {"call", "photo", "vitals"}


# ---------------------------------------------------------------------------
# extract_commands tests (HTTP mocked via _chat_completion)
# ---------------------------------------------------------------------------

class TestExtractCommands:
    @pytest.mark.asyncio
    async def test_single_robot_command(self, adapter):
        with patch.object(adapter, "_chat_completion", new_callable=AsyncMock) as m:
            m.return_value = _exec_response({"command": "photo"})
            batch = await adapter.extract_commands("take a photo")
        assert not batch.is_conversation
        assert len(batch.commands) == 1
        assert batch.commands[0].command == "photo"

    @pytest.mark.asyncio
    async def test_robot_command_with_slot(self, adapter):
        with patch.object(adapter, "_chat_completion", new_callable=AsyncMock) as m:
            m.return_value = _exec_response({"command": "call", "data": "John"})
            batch = await adapter.extract_commands("call John")
        assert not batch.is_conversation
        assert batch.commands[0].command == "call"
        assert batch.commands[0].data == "John"
        assert batch.commands[0].data_type == "users"

    @pytest.mark.asyncio
    async def test_multiple_commands_in_one_call(self, adapter):
        with patch.object(adapter, "_chat_completion", new_callable=AsyncMock) as m:
            m.return_value = _exec_response(
                {"command": "photo"},
                {"command": "vitals"},
            )
            batch = await adapter.extract_commands("take a photo and check vitals")
        assert not batch.is_conversation
        assert len(batch.commands) == 2
        assert {c.command for c in batch.commands} == {"photo", "vitals"}

    @pytest.mark.asyncio
    async def test_text_tool_call_fallback(self, adapter):
        """Qwen produces execute_commands as plain text — must be parsed as a real command."""
        with patch.object(adapter, "_chat_completion", new_callable=AsyncMock) as m:
            m.return_value = _text_response('execute_commands([{command:"call", data:"Jake"}])')
            batch = await adapter.extract_commands("call Jake")
        assert not batch.is_conversation
        assert len(batch.commands) == 1
        assert batch.commands[0].command == "call"
        assert batch.commands[0].data == "Jake"

    @pytest.mark.asyncio
    async def test_goto_and_call_in_one_call(self, adapter):
        """Regression: 'go to home and call Jake' must produce both commands."""
        # Add goto to a fresh adapter for this test
        defs = [
            CommandDefinition(name="goto", spellings=["go to", "goto"], n_slots=1, slots_type=["locations"]),
            CommandDefinition(name="call", spellings=["call"],           n_slots=1, slots_type=["users"]),
        ]
        local_adapter = LLMAdapter(AppSettings(), defs)
        with patch.object(local_adapter, "_chat_completion", new_callable=AsyncMock) as m:
            m.return_value = _exec_response(
                {"command": "goto", "data": "home"},
                {"command": "call", "data": "Jake"},
            )
            batch = await local_adapter.extract_commands("go to home and call Jake")
        assert not batch.is_conversation
        assert len(batch.commands) == 2
        cmds = {c.command: c.data for c in batch.commands}
        assert cmds["goto"] == "home"
        assert cmds["call"] == "Jake"

    @pytest.mark.asyncio
    async def test_unknown_command_in_execute_skipped(self, adapter):
        with patch.object(adapter, "_chat_completion", new_callable=AsyncMock) as m:
            m.return_value = _exec_response(
                {"command": "fly_to_moon"},
                {"command": "photo"},
            )
            batch = await adapter.extract_commands("do something and take a photo")
        assert not batch.is_conversation
        assert len(batch.commands) == 1
        assert batch.commands[0].command == "photo"

    @pytest.mark.asyncio
    async def test_conversational_response(self, adapter):
        with patch.object(adapter, "_chat_completion", new_callable=AsyncMock) as m:
            m.return_value = _text_response("I can take photos, make calls, and more!")
            batch = await adapter.extract_commands("what can you do?")
        assert batch.is_conversation
        assert batch.conversation_response

    @pytest.mark.asyncio
    async def test_weather_tool_two_rounds(self, adapter):
        """get_weather call → result fed back → Qwen produces spoken answer."""
        with patch.object(adapter, "_chat_completion", new_callable=AsyncMock) as mock_chat, \
             patch.object(adapter, "_execute_data_tool", new_callable=AsyncMock) as mock_tool:
            mock_chat.side_effect = [
                _tool_response("get_weather", {"location": "Paris"}),
                _text_response("It's sunny and 20°C in Paris."),
            ]
            mock_tool.return_value = "Sunny, +20°C"
            batch = await adapter.extract_commands("what's the weather in Paris?")
        assert batch.is_conversation
        assert batch.conversation_response

    @pytest.mark.asyncio
    async def test_llm_failure_returns_graceful_error(self, adapter):
        with patch.object(adapter, "_chat_completion", new_callable=AsyncMock) as m:
            m.return_value = None
            batch = await adapter.extract_commands("take vitals")
        assert batch.is_conversation
        assert batch.conversation_response


# ---------------------------------------------------------------------------
# Conditional command tests
# ---------------------------------------------------------------------------

class TestConditionalCommands:
    @pytest.mark.asyncio
    async def test_conditional_then_only(self, adapter):
        with patch.object(adapter, "_chat_completion", new_callable=AsyncMock) as m:
            m.return_value = _tool_response("conditional", {
                "condition": "person detected",
                "then_commands": [{"command": "photo"}],
            })
            batch = await adapter.extract_commands("if someone is there take a photo")
        assert not batch.is_conversation
        cmd = batch.commands[0]
        assert isinstance(cmd, ConditionalCommand)
        assert cmd.condition == "person detected"
        assert cmd.then_commands[0].command == "photo"
        assert cmd.else_commands is None

    @pytest.mark.asyncio
    async def test_conditional_with_else(self, adapter):
        with patch.object(adapter, "_chat_completion", new_callable=AsyncMock) as m:
            m.return_value = _tool_response("conditional", {
                "condition": "person detected",
                "then_commands": [{"command": "photo"}],
                "else_commands": [{"command": "vitals"}],
            })
            batch = await adapter.extract_commands("if someone is there take a photo otherwise check vitals")
        cmd = batch.commands[0]
        assert isinstance(cmd, ConditionalCommand)
        assert cmd.else_commands[0].command == "vitals"

    @pytest.mark.asyncio
    async def test_conditional_with_slot_in_then(self, adapter):
        with patch.object(adapter, "_chat_completion", new_callable=AsyncMock) as m:
            m.return_value = _tool_response("conditional", {
                "condition": "person detected",
                "then_commands": [{"command": "call", "data": "John"}],
            })
            batch = await adapter.extract_commands("if someone is there call John")
        cmd = batch.commands[0]
        assert isinstance(cmd, ConditionalCommand)
        assert cmd.then_commands[0].data == "John"
        assert cmd.then_commands[0].data_type == "users"

    @pytest.mark.asyncio
    async def test_execute_and_conditional_together(self, adapter):
        """'take a photo and if someone is there call John' → execute_commands + conditional."""
        with patch.object(adapter, "_chat_completion", new_callable=AsyncMock) as m:
            # Qwen calls both tools at once
            m.return_value = {
                "choices": [{
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_0",
                                "type": "function",
                                "function": {
                                    "name": "execute_commands",
                                    "arguments": json.dumps({"commands": [{"command": "photo"}]}),
                                },
                            },
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "conditional",
                                    "arguments": json.dumps({
                                        "condition": "person detected",
                                        "then_commands": [{"command": "call", "data": "John"}],
                                    }),
                                },
                            },
                        ],
                    },
                }],
                "usage": {},
            }
            batch = await adapter.extract_commands("take a photo and if someone is there call John")
        assert not batch.is_conversation
        assert len(batch.commands) == 2
        assert batch.commands[0].command == "photo"
        assert isinstance(batch.commands[1], ConditionalCommand)


# ---------------------------------------------------------------------------
# _parse_conditional_tool unit tests
# ---------------------------------------------------------------------------

class TestParseConditionalTool:
    def test_valid_conditional(self, adapter):
        result = adapter._parse_conditional_tool({
            "condition": "obstacle ahead",
            "then_commands": [{"command": "photo"}],
        })
        assert result is not None
        assert result.condition == "obstacle ahead"

    def test_missing_condition_returns_none(self, adapter):
        assert adapter._parse_conditional_tool({"then_commands": [{"command": "photo"}]}) is None

    def test_empty_then_returns_none(self, adapter):
        assert adapter._parse_conditional_tool({"condition": "x", "then_commands": []}) is None

    def test_unknown_sub_command_skipped(self, adapter):
        result = adapter._parse_conditional_tool({
            "condition": "x",
            "then_commands": [{"command": "fly_to_moon"}, {"command": "photo"}],
        })
        assert result is not None
        assert len(result.then_commands) == 1
        assert result.then_commands[0].command == "photo"
