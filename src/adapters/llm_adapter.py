"""
LLM Adapter: interfaces with llama-server's OpenAI-compatible API.
Handles prompt construction, GBNF grammar loading, and response parsing.

The system prompt is loaded from an external template file (data/system_prompt.txt)
and injected with command definitions from commands.json at startup.
Edit the template to change the LLM's behavior without modifying code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import httpx

from src.config import AppSettings
from src.models import CommandBatch, CommandDefinition, SimpleCommand, ConditionalCommand
from src.utils import get_logger

logger = get_logger(__name__)


class LLMAdapter:
    """Async adapter for llama-server with grammar-constrained decoding."""

    def __init__(self, settings: AppSettings, command_defs: List[CommandDefinition]) -> None:
        self._settings = settings.llm
        self._command_defs = command_defs
        self._grammar = self._load_file(settings.llm.grammar_path, "gbnf_grammar")
        self._system_prompt = self._build_system_prompt(settings.llm.system_prompt_path)
        self._client = httpx.AsyncClient(
            base_url=self._settings.base_url,
            timeout=httpx.Timeout(self._settings.timeout),
        )

    # ------------------------------------------------------------------ #
    #  Prompt & file loading                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_file(path: str, label: str) -> str | None:
        """Load a text file from disk. Returns None if not found."""
        file_path = Path(path)
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            logger.info(f"{label}_loaded", path=str(file_path), size=len(content))
            return content
        logger.warning(f"{label}_not_found", path=str(file_path))
        return None

    def _build_system_prompt(self, template_path: str) -> str:
        """
        Build the system prompt by:
          1. Loading the external template from data/system_prompt.txt
          2. Generating the commands block from commands.json definitions
          3. Replacing the {{COMMANDS_BLOCK}} placeholder in the template

        To customize the LLM's behavior, edit data/system_prompt.txt directly.
        The placeholder {{COMMANDS_BLOCK}} is auto-populated — do not remove it.
        """
        # --- Generate the commands block from definitions ---
        cmd_lines = []
        for cmd in self._command_defs:
            slots_info = ""
            if cmd.n_slots > 0:
                slots_info = (
                    f" | Requires {cmd.n_slots} slot(s) of type: "
                    f"{', '.join(cmd.slots_type)}"
                )
            spellings = ", ".join(f'"{s}"' for s in cmd.spellings)
            cmd_lines.append(
                f'  - "{cmd.name}": triggered by [{spellings}]{slots_info}'
            )

        commands_block = "\n".join(cmd_lines)

        # --- Load the template ---
        template = self._load_file(template_path, "system_prompt_template")

        if template is None:
            logger.error(
                "system_prompt_template_missing",
                path=template_path,
                hint="Create data/system_prompt.txt with {{COMMANDS_BLOCK}} placeholder.",
            )
            # Hard fallback: minimal functional prompt
            return (
                "You are a robot command parser. Extract commands from text as JSON.\n"
                f"Available commands:\n{commands_block}\n"
                "Respond with valid JSON only."
            )

        # --- Inject the commands block ---
        prompt = template.replace("{{COMMANDS_BLOCK}}", commands_block)

        logger.info(
            "system_prompt_built",
            template_path=template_path,
            n_commands=len(self._command_defs),
            prompt_length=len(prompt),
        )
        return prompt

    # ------------------------------------------------------------------ #
    #  Inference                                                          #
    # ------------------------------------------------------------------ #

    async def extract_commands(self, text: str) -> CommandBatch:
        """Send text to llama-server and parse the structured response."""
        logger.info("llm_inference_start", text_length=len(text))

        # Build the full prompt using Qwen2.5 ChatML format
        payload: dict = {
            "prompt": (
                f"<|im_start|>system\n{self._system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            ),
            "temperature": self._settings.temperature,
            "n_predict": self._settings.max_tokens,
            "stop": ["<|im_end|>", "<|endoftext|>"],
            "stream": False,
        }

        if self._grammar:
            payload["grammar"] = self._grammar

        try:
            response = await self._client.post("/completion", json=payload)
            response.raise_for_status()
            result = response.json()
            content = result.get("content", "").strip()

            logger.info(
                "llm_inference_complete",
                tokens_predicted=result.get("tokens_predicted", 0),
                tokens_evaluated=result.get("tokens_evaluated", 0),
            )

            return self._parse_response(content)

        except httpx.TimeoutException:
            logger.error("llm_timeout", timeout=self._settings.timeout)
            return CommandBatch(
                is_conversation=True,
                conversation_response="I'm sorry, I took too long to process. Please try again.",
            )
        except httpx.HTTPStatusError as exc:
            logger.error("llm_http_error", status=exc.response.status_code)
            return CommandBatch(
                is_conversation=True,
                conversation_response="I'm having trouble right now. Please try again.",
            )
        except Exception as exc:
            logger.error("llm_unexpected_error", error=str(exc))
            return CommandBatch(
                is_conversation=True,
                conversation_response="An unexpected error occurred. Please try again.",
            )

    # ------------------------------------------------------------------ #
    #  Response parsing                                                   #
    # ------------------------------------------------------------------ #

    def _parse_response(self, content: str) -> CommandBatch:
        """Parse LLM JSON response into a validated CommandBatch."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("llm_invalid_json", content=content[:200])
            return CommandBatch(
                is_conversation=True,
                conversation_response="I couldn't understand that. Could you rephrase?",
            )

        commands = []
        for cmd_data in data.get("commands", []):
            if cmd_data.get("command") == "conditional":
                then_cmds = [
                    SimpleCommand(**tc) for tc in cmd_data.get("then_commands", [])
                ]
                else_cmds = (
                    [SimpleCommand(**ec) for ec in cmd_data["else_commands"]]
                    if cmd_data.get("else_commands")
                    else None
                )
                commands.append(
                    ConditionalCommand(
                        condition=cmd_data.get("condition", ""),
                        then_commands=then_cmds,
                        else_commands=else_cmds,
                    )
                )
            else:
                commands.append(SimpleCommand(**cmd_data))

        # Validate: only keep commands that exist in our definitions
        valid_names = {d.name for d in self._command_defs} | {"conditional"}
        validated = []
        for cmd in commands:
            if isinstance(cmd, ConditionalCommand):
                cmd.then_commands = [
                    c for c in cmd.then_commands if c.command in valid_names
                ]
                if cmd.else_commands:
                    cmd.else_commands = [
                        c for c in cmd.else_commands if c.command in valid_names
                    ]
                if cmd.then_commands:
                    validated.append(cmd)
            elif cmd.command in valid_names:
                validated.append(cmd)
            else:
                logger.warning("llm_unknown_command_filtered", command=cmd.command)

        is_conversation = data.get("is_conversation", False) or len(validated) == 0
        return CommandBatch(
            commands=validated,
            is_conversation=is_conversation,
            conversation_response=data.get("conversation_response"),
        )

    # ------------------------------------------------------------------ #
    #  Health & lifecycle                                                 #
    # ------------------------------------------------------------------ #

    async def health_check(self) -> bool:
        """Check if llama-server is responding."""
        try:
            resp = await self._client.get("/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()
