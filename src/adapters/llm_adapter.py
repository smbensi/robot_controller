"""
LLM Adapter: interfaces with llama-server's OpenAI-compatible API.
Uses /v1/chat/completions with Qwen2.5 tool calling.

Tool design — three tools total:
  - execute_commands: single tool that takes an ARRAY of robot commands.
    Using one tool for all commands (instead of one tool per command) avoids
    the reliability problem where Qwen only calls one tool out of many.
  - conditional: for if/then/else robot logic.
  - <data tools>: fetch real-time data (weather, etc.); result is fed back to
    Qwen so it can speak a natural-language answer.

To add a new data tool:
  1. Add its JSON schema to _DATA_TOOLS.
  2. Add a handler branch in _execute_data_tool().
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, List

import httpx

from src.config import AppSettings
from src.models import CommandBatch, CommandDefinition, ConditionalCommand, SimpleCommand
from src.utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data tool definitions (non-robot tools that fetch real-time information)
# ---------------------------------------------------------------------------

_DATA_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather conditions for a city or location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or location, e.g. 'New York' or 'London'.",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current local date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression and return the result. Use for any arithmetic or math question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g. '2 + 2', 'sqrt(144)', '15 % 4'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": (
                "Get the latest news headlines. "
                "Supports any search term: country ('Israel'), topic ('health'), keyword ('earthquake'), etc. "
                "Leave query empty for general top headlines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Optional search term, e.g. 'Israel', 'health', 'technology'. Empty for top headlines.",
                    }
                },
                "required": [],
            },
        },
    },
]

_MAX_TOOL_ROUNDS = 5  # Prevent infinite tool-call loops


class LLMAdapter:
    """Async adapter for llama-server using OpenAI-compatible tool calling."""

    def __init__(self, settings: AppSettings, command_defs: List[CommandDefinition]) -> None:
        self._settings = settings.llm
        self._command_defs = command_defs
        self._robot_command_names = {d.name for d in command_defs}
        self._cmd_lookup = {d.name: d for d in command_defs}
        self._tools = self._build_tools()
        self._system_prompt = self._build_system_prompt(settings.llm.system_prompt_path)
        self._client = httpx.AsyncClient(
            base_url=self._settings.base_url,
            timeout=httpx.Timeout(self._settings.timeout),
        )
        # Conversation history — only used for conversational turns, not robot commands.
        # Safe without locking: max_concurrent_llm=1 serialises all LLM calls.
        self._history: list[dict] = []
        self._max_history_msgs: int = settings.llm.max_history_pairs * 2

    # ------------------------------------------------------------------ #
    #  Tool & prompt construction                                          #
    # ------------------------------------------------------------------ #

    def _build_tools(self) -> list[dict]:
        """
        Build the tools list:
          - execute_commands: one tool, array of commands (avoids multi-tool reliability issues)
          - conditional: for if/then/else logic
          - data tools (weather, etc.)
        """
        cmd_names = sorted(self._robot_command_names)

        # Inline command descriptions so Qwen knows spellings and slot requirements
        cmd_desc_lines = []
        for cmd in self._command_defs:
            slot_note = f" (slot: {', '.join(cmd.slots_type)})" if cmd.slots_type else ""
            cmd_desc_lines.append(
                f'  "{cmd.name}": triggered by {", ".join(cmd.spellings)}{slot_note}'
            )
        cmd_descriptions = "\n".join(cmd_desc_lines)

        # Shared sub-command item schema (reused in execute_commands and conditional)
        sub_cmd_schema: dict = {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": cmd_names,
                    "description": "Robot command name.",
                },
                "data": {
                    "type": "string",
                    "description": "Slot value if the command requires one (e.g. person name, location).",
                },
            },
            "required": ["command"],
        }

        tools: list[dict] = [
            # ---- execute_commands ----------------------------------------
            {
                "type": "function",
                "function": {
                    "name": "execute_commands",
                    "description": (
                        "Execute one or more robot commands. "
                        "Include ALL commands the user requested in a single call — never split them. "
                        "If the input matches a command spelling, call this tool immediately — do NOT ask for clarification.\n"
                        f"Available commands:\n{cmd_descriptions}"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "commands": {
                                "type": "array",
                                "description": "All commands to execute, in the order mentioned.",
                                "items": sub_cmd_schema,
                                "minItems": 1,
                            }
                        },
                        "required": ["commands"],
                    },
                },
            },
            # ---- conditional ---------------------------------------------
            {
                "type": "function",
                "function": {
                    "name": "conditional",
                    "description": (
                        "Queue commands that run only when a condition is met. "
                        "Use when the user says 'if', 'when', 'only if', 'otherwise', etc."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "condition": {
                                "type": "string",
                                "description": "Clear description of the trigger condition.",
                            },
                            "then_commands": {
                                "type": "array",
                                "description": "Commands to run when the condition is true.",
                                "items": sub_cmd_schema,
                            },
                            "else_commands": {
                                "type": "array",
                                "description": "Commands to run when the condition is false (omit if not needed).",
                                "items": sub_cmd_schema,
                            },
                        },
                        "required": ["condition", "then_commands"],
                    },
                },
            },
        ]

        tools.extend(_DATA_TOOLS)
        logger.info(
            "tools_built",
            n_robot_commands=len(self._command_defs),
            n_data_tools=len(_DATA_TOOLS),
        )
        return tools

    def _build_system_prompt(self, template_path: str) -> str:
        """Load the system prompt template and inject the commands block."""
        cmd_lines = []
        for cmd in self._command_defs:
            slots_info = ""
            if cmd.n_slots > 0:
                slots_info = (
                    f" | requires slot of type: {', '.join(cmd.slots_type)}"
                )
            spellings = ", ".join(f'"{s}"' for s in cmd.spellings)
            cmd_lines.append(f'  - "{cmd.name}": [{spellings}]{slots_info}')

        commands_block = "\n".join(cmd_lines)

        file_path = Path(template_path)
        if not file_path.exists():
            logger.warning("system_prompt_template_not_found", path=template_path)
            return (
                "You are a robot assistant. Use the available tools to execute "
                "robot commands or fetch information for the user.\n"
                f"Available commands:\n{commands_block}"
            )

        template = file_path.read_text(encoding="utf-8")
        prompt = template.replace("{{COMMANDS_BLOCK}}", commands_block)
        logger.info(
            "system_prompt_built",
            template_path=template_path,
            n_commands=len(self._command_defs),
            prompt_length=len(prompt),
        )
        return prompt

    # ------------------------------------------------------------------ #
    #  Inference                                                           #
    # ------------------------------------------------------------------ #

    async def extract_commands(self, text: str) -> CommandBatch:
        """
        Command detection only — always runs WITHOUT history so conversational
        context never biases command recognition.
        For conversational turns the pipeline calls stream_reply() separately,
        which handles history injection and streaming.
        Robot commands never touch history.
        """
        return await self._inference_loop(text, inject_history=False)

    async def stream_reply(self, text: str) -> AsyncGenerator[str, None]:
        """
        Stream a conversational reply to `text`, injecting history for context.
        Yields text chunks as they arrive from the LLM.
        Saves the full exchange to history when the stream is done.
        """
        messages = [
            {"role": "system", "content": self._system_prompt},
            *self._history,
            {"role": "user", "content": text},
        ]
        full_reply = ""
        async for chunk in self._stream_chat_completion(messages):
            full_reply += chunk
            yield chunk

        # Save to history after the stream completes
        if full_reply:
            self._history.append({"role": "user", "content": text})
            self._history.append({"role": "assistant", "content": full_reply})
            if len(self._history) > self._max_history_msgs:
                del self._history[:len(self._history) - self._max_history_msgs]

    async def _inference_loop(self, text: str, inject_history: bool) -> CommandBatch:
        """Run the tool-calling loop and return a CommandBatch."""
        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt},
            *(self._history if inject_history else []),
            {"role": "user", "content": text},
        ]
        pending_robot_cmds: list[SimpleCommand | ConditionalCommand] = []
        _robot_tool_names = {"execute_commands", "conditional"}
        had_data_tool = False  # True when at least one data tool was executed

        for _round in range(_MAX_TOOL_ROUNDS):
            response = await self._chat_completion(messages)
            if response is None:
                break

            choice = response["choices"][0]
            assistant_msg = choice["message"]
            tool_calls = assistant_msg.get("tool_calls") or []

            if not tool_calls:
                content = (assistant_msg.get("content") or "").strip()
                if pending_robot_cmds:
                    return CommandBatch(commands=pending_robot_cmds, is_conversation=False)
                # Fallback: Qwen sometimes produces tool calls as plain text instead
                # of structured tool calls (e.g. `execute_commands([{command:"call"…}])`).
                # Try to parse and execute them before falling back to conversation.
                fallback = self._try_parse_text_tool_call(content)
                if fallback:
                    logger.warning("text_tool_call_fallback", content_preview=content[:80])
                    return CommandBatch(commands=fallback, is_conversation=False)
                return CommandBatch(
                    is_conversation=True,
                    conversation_response=content or "I didn't understand that. Could you rephrase?",
                    tool_response=had_data_tool,
                )

            messages.append(assistant_msg)

            for tc in tool_calls:
                name = tc["function"]["name"]
                raw_args = tc["function"]["arguments"]
                try:
                    args = raw_args if isinstance(raw_args, dict) else json.loads(raw_args)
                except (json.JSONDecodeError, TypeError):
                    args = {}

                if name == "execute_commands":
                    items = args.get("commands", [])
                    if not items:
                        # Model signalled execute_commands but generated empty arguments —
                        # common with quantized models. Fall back to spelling match.
                        items = self._spelling_match(text)
                        if items:
                            logger.warning("execute_commands_empty_args_spelling_fallback",
                                           matched=[i["command"] for i in items])
                    for item in items:
                        cmd_name = item.get("command", "")
                        if cmd_name not in self._robot_command_names:
                            logger.warning("execute_commands_unknown", command=cmd_name)
                            continue
                        cmd_def = self._cmd_lookup[cmd_name]
                        pending_robot_cmds.append(
                            SimpleCommand(
                                command=cmd_name,
                                data_type=cmd_def.slots_type[0] if cmd_def.slots_type else "",
                                data=item.get("data", ""),
                            )
                        )
                        logger.info("robot_command_queued", command=cmd_name, data=item.get("data", ""))
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": '{"status": "queued"}'})

                elif name == "conditional":
                    cond_cmd = self._parse_conditional_tool(args)
                    if cond_cmd:
                        pending_robot_cmds.append(cond_cmd)
                        logger.info(
                            "conditional_command_queued",
                            condition=cond_cmd.condition,
                            n_then=len(cond_cmd.then_commands),
                            n_else=len(cond_cmd.else_commands) if cond_cmd.else_commands else 0,
                        )
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": '{"status": "queued"}'})

                else:
                    result = await self._execute_data_tool(name, args)
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                    logger.info("data_tool_executed", tool=name, result_length=len(result))
                    had_data_tool = True

            if all(tc["function"]["name"] in _robot_tool_names for tc in tool_calls):
                return CommandBatch(commands=pending_robot_cmds, is_conversation=False)

        logger.error("llm_tool_loop_exhausted", rounds=_MAX_TOOL_ROUNDS)
        if pending_robot_cmds:
            return CommandBatch(commands=pending_robot_cmds, is_conversation=False)
        return CommandBatch(
            is_conversation=True,
            conversation_response="I'm having trouble right now. Please try again.",
        )

    # ------------------------------------------------------------------ #
    #  HTTP call                                                           #
    # ------------------------------------------------------------------ #

    async def _chat_completion(self, messages: list[dict]) -> dict | None:
        """POST to /v1/chat/completions and return the parsed JSON response."""
        payload = {
            "model": self._settings.model,
            "messages": messages,
            "tools": self._tools,
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_tokens,
        }
        try:
            response = await self._client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            result = response.json()
            usage = result.get("usage", {})
            choice = result.get("choices", [{}])[0].get("message", {})
            logger.debug(
                "llm_raw_response",
                content=choice.get("content"),
                tool_calls=choice.get("tool_calls"),
            )
            logger.info(
                "llm_inference_complete",
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            )
            return result
        except httpx.TimeoutException:
            logger.error("llm_timeout", timeout=self._settings.timeout)
            return None
        except httpx.HTTPStatusError as exc:
            logger.error("llm_http_error", status=exc.response.status_code)
            return None
        except Exception as exc:
            logger.error("llm_unexpected_error", error=str(exc))
            return None

    async def _stream_chat_completion(self, messages: list[dict]) -> AsyncGenerator[str, None]:
        """POST to /v1/chat/completions with stream=True and yield text chunks."""
        payload = {
            "model": self._settings.model,
            "messages": messages,
            "tools": self._tools,
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_tokens,
            "stream": True,
        }
        try:
            async with self._client.stream("POST", "/v1/chat/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        content = data["choices"][0]["delta"].get("content") or ""
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        except httpx.TimeoutException:
            logger.error("llm_stream_timeout")
        except httpx.HTTPStatusError as exc:
            logger.error("llm_stream_http_error", status=exc.response.status_code)
        except Exception as exc:
            logger.error("llm_stream_unexpected_error", error=str(exc))

    # ------------------------------------------------------------------ #
    #  Text tool-call fallback parser                                      #
    # ------------------------------------------------------------------ #

    def _spelling_match(self, text: str) -> list[dict]:
        """Match user text against command spellings; returns items in execute_commands format."""
        text_lower = text.lower()
        for cmd_def in self._command_defs:
            for spelling in cmd_def.spellings:
                if spelling.lower() in text_lower:
                    return [{"command": cmd_def.name, "data": ""}]
        return []

    def _try_parse_text_tool_call(
        self, content: str
    ) -> list[SimpleCommand | ConditionalCommand] | None:
        """
        Qwen/MLC occasionally produces tool calls as plain text instead of structured
        tool calls. Two known formats:
            execute_commands([{command:"call", data:"Jake"}])
            call data="Jake"
        Returns None if the content doesn't match or parsing fails.
        """
        import re

        # Format 1: execute_commands([{command:"...", data:"..."}])
        match = re.search(
            r"execute_commands\s*\(\s*\[(.+?)\]\s*\)",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            raw = "[" + match.group(1) + "]"
            raw = re.sub(r'(?<=[{,\s])(\w+)(?=\s*:)', r'"\1"', raw)
            raw = raw.replace("'", '"')
            try:
                items: list[dict] = json.loads(raw)
            except json.JSONDecodeError:
                return None
            cmds: list[SimpleCommand | ConditionalCommand] = []
            for item in items:
                cmd_name = item.get("command", "")
                if cmd_name not in self._robot_command_names:
                    logger.warning("text_fallback_unknown_command", command=cmd_name)
                    continue
                cmd_def = self._cmd_lookup[cmd_name]
                cmds.append(
                    SimpleCommand(
                        command=cmd_name,
                        data_type=cmd_def.slots_type[0] if cmd_def.slots_type else "",
                        data=item.get("data", ""),
                    )
                )
            return cmds or None

        # Format 2: <command_name> [data="<value>"]  — produced by MLC
        m = re.match(r'^(\w+)(?:\s+data="([^"]*)")?$', content.strip())
        if m:
            cmd_name = m.group(1)
            if cmd_name in self._robot_command_names:
                cmd_def = self._cmd_lookup[cmd_name]
                return [
                    SimpleCommand(
                        command=cmd_name,
                        data_type=cmd_def.slots_type[0] if cmd_def.slots_type else "",
                        data=m.group(2) or "",
                    )
                ]

        return None

    # ------------------------------------------------------------------ #
    #  Conditional command parsing                                         #
    # ------------------------------------------------------------------ #

    def _parse_conditional_tool(self, args: dict) -> ConditionalCommand | None:
        """Convert the 'conditional' tool call arguments into a ConditionalCommand."""
        condition = (args.get("condition") or "").strip()
        if not condition:
            logger.warning("conditional_missing_condition")
            return None

        def _parse_sub_cmds(raw: list) -> list[SimpleCommand]:
            result = []
            for item in raw:
                name = item.get("command", "")
                if name not in self._robot_command_names:
                    logger.warning("conditional_unknown_sub_command", command=name)
                    continue
                cmd_def = self._cmd_lookup[name]
                result.append(
                    SimpleCommand(
                        command=name,
                        data_type=cmd_def.slots_type[0] if cmd_def.slots_type else "",
                        data=item.get("data", ""),
                    )
                )
            return result

        then_cmds = _parse_sub_cmds(args.get("then_commands") or [])
        if not then_cmds:
            logger.warning("conditional_empty_then_commands")
            return None

        else_raw = args.get("else_commands") or []
        else_cmds = _parse_sub_cmds(else_raw) if else_raw else None

        return ConditionalCommand(
            condition=condition,
            then_commands=then_cmds,
            else_commands=else_cmds,
        )

    # ------------------------------------------------------------------ #
    #  Data tool execution                                                 #
    # ------------------------------------------------------------------ #

    async def _execute_data_tool(self, name: str, args: dict) -> str:
        """Execute a data tool and return the result as a plain string."""
        if name == "get_weather":
            return await self._get_weather(args.get("location", ""))
        if name == "get_time":
            return self._get_time()
        if name == "calculate":
            return self._calculate(args.get("expression", ""))
        if name == "get_news":
            return await self._get_news(args.get("query", ""))
        logger.warning("unknown_data_tool", tool=name)
        return f"Tool '{name}' is not implemented."

    @staticmethod
    def _calculate(expression: str) -> str:
        """Safely evaluate a math expression using ast — no exec/eval on arbitrary code."""
        import ast
        import math
        import operator

        _OPERATORS = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }
        _FUNCTIONS = {
            "abs": abs, "round": round,
            "sqrt": math.sqrt, "ceil": math.ceil, "floor": math.floor,
            "log": math.log, "log2": math.log2, "log10": math.log10,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "asin": math.asin, "acos": math.acos, "atan": math.atan,
            "exp": math.exp, "factorial": math.factorial,
        }
        _CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}

        def _eval(node: ast.expr):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.Name) and node.id in _CONSTANTS:
                return _CONSTANTS[node.id]
            if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
                return _OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
                return _OPERATORS[type(node.op)](_eval(node.operand))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS:
                return _FUNCTIONS[node.func.id](*(_eval(a) for a in node.args))
            raise ValueError(f"Unsupported expression: {ast.dump(node)}")

        if not expression:
            return "No expression provided."
        try:
            tree = ast.parse(expression.strip(), mode="eval")
            result = _eval(tree.body)
            # Format: drop .0 for whole numbers
            if isinstance(result, float) and result.is_integer():
                return f"{expression} = {int(result)}"
            return f"{expression} = {result}"
        except ZeroDivisionError:
            return "Division by zero."
        except Exception as exc:
            return f"Could not evaluate '{expression}': {exc}"

    @staticmethod
    def _get_time() -> str:
        """Return the current local date and time as a plain string."""
        return datetime.now().strftime("%A, %B %d %Y, %H:%M")

    @staticmethod
    async def _get_news(query: str = "") -> str:
        """Fetch headlines from Google News RSS (free, no API key, supports any search term)."""
        import xml.etree.ElementTree as ET
        from urllib.parse import quote

        if query:
            url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en&gl=US&ceid=US:en"
        else:
            url = "https://news.google.com/rss?hl=en&gl=US&ceid=US:en"

        try:
            async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()

            root = ET.fromstring(resp.text)
            items = (root.find("channel") or root).findall("item")[:5]
            headlines = []
            for item in items:
                title = item.findtext("title", "").strip()
                # Google News appends " - Source Name"; strip it for cleaner output
                if " - " in title:
                    title = title.rsplit(" - ", 1)[0].strip()
                if title:
                    headlines.append(title)

            if not headlines:
                return "No headlines available right now."

            label = f"news about '{query}'" if query else "top headlines"
            return f"Latest {label}:\n" + "\n".join(f"- {h}" for h in headlines)

        except Exception as exc:
            logger.warning("news_fetch_error", query=query, error=str(exc))
            return "News is currently unavailable."

    @staticmethod
    async def _get_weather(location: str) -> str:
        """Fetch current weather from wttr.in (no API key required)."""
        if not location:
            return "No location provided."
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"https://wttr.in/{location}",
                    params={"format": "%C, %t"},
                    headers={"Accept": "text/plain"},
                )
                resp.raise_for_status()
                return resp.text.strip()
        except Exception as exc:
            logger.warning("weather_fetch_error", location=location, error=str(exc))
            return f"Weather data for '{location}' is currently unavailable."

    # ------------------------------------------------------------------ #
    #  Health & lifecycle                                                  #
    # ------------------------------------------------------------------ #

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self._history.clear()
        logger.info("history_cleared")

    async def health_check(self) -> bool:
        """Check if llama-server is responding."""
        try:
            resp = await self._client.get("/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()
