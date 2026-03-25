"""
Tests for the LLM adapter's response parsing logic.
These tests do NOT require a running llama-server.
"""

import json
import pytest
from src.adapters.llm_adapter import LLMAdapter
from src.config import AppSettings
from src.models import CommandDefinition, SimpleCommand, ConditionalCommand


@pytest.fixture
def command_defs() -> list[CommandDefinition]:
    return [
        CommandDefinition(name="call", spellings=["call"], sequence=["WAKEWORD", "call"], n_slots=1, slots_type=["users"]),
        CommandDefinition(name="photo", spellings=["photo"], sequence=["WAKEWORD", "photo"], n_slots=0, slots_type=[]),
        CommandDefinition(name="vitals", spellings=["vitals"], sequence=["WAKEWORD", "vitals"], n_slots=0, slots_type=[]),
    ]


@pytest.fixture
def adapter(command_defs) -> LLMAdapter:
    settings = AppSettings()
    return LLMAdapter(settings, command_defs)


class TestParseResponse:
    def test_single_command(self, adapter):
        content = json.dumps({
            "commands": [{"command": "call", "data_type": "users", "data": "John", "status": "pending"}],
            "is_conversation": False,
            "conversation_response": None,
        })
        result = adapter._parse_response(content)
        assert len(result.commands) == 1
        assert result.commands[0].command == "call"
        assert result.commands[0].data == "John"
        assert result.is_conversation is False

    def test_multiple_commands(self, adapter):
        content = json.dumps({
            "commands": [
                {"command": "photo", "data_type": "", "data": "", "status": "pending"},
                {"command": "vitals", "data_type": "", "data": "", "status": "pending"},
            ],
            "is_conversation": False,
            "conversation_response": None,
        })
        result = adapter._parse_response(content)
        assert len(result.commands) == 2
        assert result.commands[0].command == "photo"
        assert result.commands[1].command == "vitals"

    def test_conditional_command(self, adapter):
        content = json.dumps({
            "commands": [{
                "command": "conditional",
                "condition": "person detected",
                "then_commands": [{"command": "photo", "data_type": "", "data": "", "status": "pending"}],
                "else_commands": None,
                "status": "pending",
            }],
            "is_conversation": False,
            "conversation_response": None,
        })
        result = adapter._parse_response(content)
        assert len(result.commands) == 1
        assert isinstance(result.commands[0], ConditionalCommand)
        assert result.commands[0].condition == "person detected"
        assert len(result.commands[0].then_commands) == 1

    def test_conversation_fallback(self, adapter):
        content = json.dumps({
            "commands": [],
            "is_conversation": True,
            "conversation_response": "I don't understand.",
        })
        result = adapter._parse_response(content)
        assert result.is_conversation is True
        assert result.conversation_response == "I don't understand."
        assert len(result.commands) == 0

    def test_invalid_json_returns_conversation(self, adapter):
        result = adapter._parse_response("not json at all {{{")
        assert result.is_conversation is True

    def test_unknown_command_filtered(self, adapter):
        content = json.dumps({
            "commands": [
                {"command": "fly_to_moon", "data_type": "", "data": "", "status": "pending"},
                {"command": "photo", "data_type": "", "data": "", "status": "pending"},
            ],
            "is_conversation": False,
            "conversation_response": None,
        })
        result = adapter._parse_response(content)
        assert len(result.commands) == 1
        assert result.commands[0].command == "photo"


class TestParseResponseEdgeCases:
    def test_empty_commands_becomes_conversation(self, adapter):
        content = json.dumps({
            "commands": [],
            "is_conversation": False,
            "conversation_response": None,
        })
        result = adapter._parse_response(content)
        assert result.is_conversation is True

    def test_conditional_with_else(self, adapter):
        content = json.dumps({
            "commands": [{
                "command": "conditional",
                "condition": "obstacle ahead",
                "then_commands": [{"command": "photo", "data_type": "", "data": "", "status": "pending"}],
                "else_commands": [{"command": "vitals", "data_type": "", "data": "", "status": "pending"}],
                "status": "pending",
            }],
            "is_conversation": False,
            "conversation_response": None,
        })
        result = adapter._parse_response(content)
        cond = result.commands[0]
        assert isinstance(cond, ConditionalCommand)
        assert len(cond.else_commands) == 1
        assert cond.else_commands[0].command == "vitals"
