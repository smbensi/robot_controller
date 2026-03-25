"""
Domain models for robot commands.
These models define the contract between the LLM output and the robot actuators.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class CommandStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"


class CommandDefinition(BaseModel):
    """Schema for a single command definition loaded from commands.json."""

    name: str
    spellings: List[str]
    sequence: List[str]
    n_slots: int
    slots_type: List[str]


class SimpleCommand(BaseModel):
    """A single robot command extracted by the LLM."""

    command: str
    data_type: str = ""
    data: str = ""
    status: CommandStatus = CommandStatus.PENDING

    # Resolved entity IDs keyed by slot_type (filled post-LLM by EntityResolver).
    # e.g. {"users": "64f3a...", "locations": "64f3b..."}
    # Excluded from JSON serialisation — the pipeline unpacks them into the output dict.
    resolved_ids: dict[str, str] = Field(default_factory=dict, exclude=True)


class ConditionalCommand(BaseModel):
    """A conditional command: execute then_commands if condition is met."""

    command: str = "conditional"
    condition: str
    then_commands: List[SimpleCommand]
    else_commands: Optional[List[SimpleCommand]] = None
    status: CommandStatus = CommandStatus.PENDING


class CommandBatch(BaseModel):
    """The top-level response from the LLM pipeline."""

    commands: List[SimpleCommand | ConditionalCommand] = Field(default_factory=list)
    is_conversation: bool = False
    conversation_response: Optional[str] = None


class ParsedMessage(BaseModel):
    """Incoming MQTT message wrapper."""

    transaction_id: str
    robot_id: str
    text: str
    timestamp: float


class CommandResponse(BaseModel):
    """Outgoing MQTT message wrapper."""

    transaction_id: str
    robot_id: str
    commands: List[dict]
    is_conversation: bool = False
    conversation_response: Optional[str] = None
