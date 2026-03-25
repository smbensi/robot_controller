# Robot Controller — Source Code Documentation

## Overview

The robot controller is an async Python service that converts free-form spoken or typed text into structured robot commands. It sits between a speech-to-text (STT) system and robot actuators, using a local LLM to interpret natural language.

```
STT / Text Input
      │
      ▼ MQTT  robot/{id}/input/text
┌─────────────────────────────────┐
│       Robot Controller          │
│                                 │
│  MQTT ──► Pipeline ──► MQTT     │
│              │                  │
│           LLM (llama-server)    │
│           MongoDB (entities)    │
└─────────────────────────────────┘
      │
      ▼ MQTT  robot/{id}/commands/parsed/{tid}
Robot Actuators
```

---

## Directory Structure

```
robot-controller/
├── src/
│   ├── main.py                   # Entry point — wires everything together
│   ├── config/
│   │   └── settings.py           # Pydantic-settings config classes
│   ├── models/
│   │   └── commands.py           # Domain models (commands, responses)
│   ├── adapters/
│   │   ├── llm_adapter.py        # llama-server HTTP client + response parser
│   │   ├── mqtt_adapter.py       # MQTT pub/sub (aiomqtt)
│   │   └── entity_resolver.py    # MongoDB name → ID resolver
│   ├── services/
│   │   └── command_pipeline.py   # Main orchestration logic
│   └── utils/
│       └── logging.py            # Structured logging (structlog)
├── data/
│   ├── commands.json             # Command definitions (source of truth)
│   └── system_prompt.txt         # LLM system prompt template
├── grammars/
│   └── commands.gbnf             # GBNF grammar — constrains LLM output format
├── scripts/
│   ├── chat.py                   # Interactive CLI test (no MQTT/MongoDB needed)
│   ├── start_llama_server.sh     # Jetson startup script
│   ├── download_model.sh         # Model download helper
│   └── test_mqtt_client.py       # MQTT publish/subscribe test tool
├── tests/
│   └── test_llm_parsing.py       # Unit tests (no servers required)
├── docs/
│   ├── architecture.md           # This file
│   └── llama-server-mac.md       # How to run llama-server on macOS
├── .env.example                  # Config reference — copy to .env
├── pyproject.toml                # Project metadata, pytest + ruff config
└── requirements.txt              # Python dependencies
```

---

## Data Flow

### Full pipeline (production)

```
1. STT publishes:
   Topic:   robot/robot-01/input/text
   Payload: {"transaction_id": "abc123", "text": "call John"}

2. MQTTAdapter receives the message, parses it into a ParsedMessage,
   and calls pipeline.enqueue()

3. CommandPipeline dequeues the message (backpressure via asyncio.Queue)

4. LLMAdapter.extract_commands("call John") is called:
   - Builds a ChatML prompt with the system prompt + user text
   - Sends it to llama-server /completion with GBNF grammar
   - Parses the JSON response into a CommandBatch

5. EntityResolver resolves "John" → MongoDB ObjectId
   (only for commands that have slots, e.g. "call")

6. CommandPipeline publishes:
   Topic:   robot/robot-01/commands/parsed/abc123
   Payload: {"commands": [{"command": "call", "data_type": "users",
             "data": "John", "user_id": "64f3a...", "status": "pending"}]}
```

### Interactive test (development, no MQTT/MongoDB)

```
scripts/chat.py
    └── LLMAdapter.extract_commands(text)
              └── llama-server /completion
```

---

## Module Reference

### `src/main.py` — Entry Point

Startup sequence:
1. Load `AppSettings` from environment / `.env` file
2. Load command definitions from `data/commands.json`
3. Create `EntityResolver`, `LLMAdapter`, `MQTTAdapter`
4. Connect to MongoDB — warns and continues if unavailable
5. Check llama-server health — warns and continues if unavailable
6. Build `CommandPipeline`
7. Register `SIGTERM` / `SIGINT` handlers for graceful shutdown
8. Run two concurrent async tasks:
   - `pipeline.run()` — drains the processing queue
   - `mqtt.listen()` — receives incoming messages and enqueues them
9. On shutdown signal: cancel tasks, close HTTP and DB clients

---

### `src/config/settings.py` — Configuration

All settings are read from environment variables or a `.env` file in the project root. Nested settings groups have their own env prefix.

| Class | Env prefix | Key settings |
|-------|-----------|--------------|
| `AppSettings` | `APP_` | `robot_id`, `log_level`, `log_format`, `commands_file` |
| `MQTTSettings` | `MQTT_` | `host`, `port`, `username`, `password`, `qos` |
| `MongoSettings` | `MONGO_` | `uri`, `database`, `users_collection` |
| `LLMSettings` | `LLM_` | `base_url`, `timeout`, `temperature`, `max_tokens` |

**Important:** Use `127.0.0.1` not `localhost` for `LLM_BASE_URL` on macOS — Python's asyncio resolves `localhost` to IPv6 (`::1`) but llama-server only binds to IPv4.

Copy `.env.example` to `.env` to configure:
```bash
cp .env.example .env
```

---

### `src/models/commands.py` — Domain Models

All data structures used across the pipeline:

```
CommandDefinition          — loaded from commands.json at startup
  name: str                  e.g. "call"
  spellings: List[str]       e.g. ["call", "cold", "ring"]
  n_slots: int               number of data slots (0 or 1)
  slots_type: List[str]      e.g. ["users"]

SimpleCommand              — a single extracted command
  command: str               e.g. "vitals"
  data_type: str             e.g. "users" (empty if no slot)
  data: str                  e.g. "John" (empty if no slot)
  status: CommandStatus      always "pending" from LLM
  resolved_user_id: str      filled by EntityResolver (excluded from JSON)
  resolved_location_id: str  filled by EntityResolver (excluded from JSON)

ConditionalCommand         — if/then/else logic
  condition: str             e.g. "person detected in field of view"
  then_commands: List[SimpleCommand]
  else_commands: Optional[List[SimpleCommand]]

CommandBatch               — LLM output (internal)
  commands: List[SimpleCommand | ConditionalCommand]
  is_conversation: bool      True when no command was detected
  conversation_response: str the robot's natural language reply

ParsedMessage              — MQTT input wrapper
  transaction_id: str
  robot_id: str
  text: str
  timestamp: float

CommandResponse            — MQTT output wrapper
  transaction_id: str
  robot_id: str
  commands: List[dict]
```

---

### `src/adapters/llm_adapter.py` — LLM Adapter

Communicates with llama-server over HTTP.

**Startup:**
1. Loads the GBNF grammar from `grammars/commands.gbnf`
2. Loads the system prompt template from `data/system_prompt.txt`
3. Injects command definitions into the `{{COMMANDS_BLOCK}}` placeholder
4. Creates a persistent `httpx.AsyncClient` pointed at llama-server

**`extract_commands(text) → CommandBatch`:**
1. Wraps text in Qwen2.5 ChatML format:
   ```
   <|im_start|>system
   {system_prompt}<|im_end|>
   <|im_start|>user
   {text}<|im_end|>
   <|im_start|>assistant
   ```
2. Sends to `/completion` with grammar, temperature (0.1), max_tokens (1024)
3. Parses the JSON response into `CommandBatch`
4. Filters out commands not present in `commands.json`

**`_parse_response(content) → CommandBatch`:**
- Deserialises the LLM's JSON output
- Validates all command names against the loaded definitions
- Forces `is_conversation=True` if no valid commands were extracted
- Unknown commands are silently dropped (logged as warnings)

**Error handling:**
- Timeout → conversation response asking to retry
- HTTP error → conversation response
- Any exception → conversation response (never crashes the pipeline)

---

### `src/adapters/mqtt_adapter.py` — MQTT Adapter

Uses `aiomqtt` (async wrapper around paho-mqtt).

**Topics:**

| Direction | Topic | Purpose |
|-----------|-------|---------|
| Subscribe | `robot/{id}/input/text` | Incoming text from STT |
| Publish | `robot/{id}/commands/parsed/{tid}` | Structured commands |
| Publish | `robot/{id}/chat/response/{tid}` | Conversational replies |
| Publish | `robot/{id}/status` | Online/offline (retained) |

**`listen(on_message)` (runs forever):**
- Connects to broker, publishes `{"status": "online"}` (retained)
- Subscribes to input topic
- For each message: parses JSON or raw text → `ParsedMessage` → calls `on_message()`
- On disconnect: waits 5 seconds, reconnects automatically

**Last Will Testament:**
If the client disconnects unexpectedly, the broker automatically publishes `{"status": "offline"}` to the status topic (retained). This lets other services detect when the controller goes down.

**Message format (input):**
```json
{"transaction_id": "abc123", "text": "call John"}
```
If `transaction_id` is absent, a UUID is generated automatically.
If the payload is not JSON, the raw string is used as `text`.

---

### `src/adapters/entity_resolver.py` — Entity Resolver

Resolves human-readable names from commands (e.g. "John") to MongoDB document IDs.

**`resolve_by_type(slot_type, name) → Optional[str]`:**
- `slot_type` comes from `CommandDefinition.slots_type` (e.g. `"users"`)
- Does a case-insensitive regex match in the appropriate MongoDB collection
- Returns the document's `_id` as a string, or `None` if not found
- Results are cached in a `TTLCache` (default: 500 entries, 300s TTL)

**MongoDB not available:**
The app starts and runs normally. Commands that need entity resolution (e.g. `call`) are published without a `user_id` field. The robot actuator must handle the missing ID gracefully.

---

### `src/services/command_pipeline.py` — Command Pipeline

Orchestrates the full flow from received text to published response.

**Queue and concurrency:**
- `asyncio.Queue(maxsize=50)` provides backpressure — if the queue is full, the caller receives a "busy" chat response instead of blocking
- `asyncio.Semaphore(1)` serialises LLM calls — the GPU handles one request at a time (configurable via `APP_MAX_CONCURRENT_LLM`)

**`_process(message)` flow:**
```
ParsedMessage
    │
    ▼ (semaphore)
LLMAdapter.extract_commands()
    │
    ├── is_conversation? ──► publish_chat()
    │
    └── commands list
            │
            ▼
    _resolve_entities()   ← MongoDB lookup per slot
            │
            ▼
    CommandResponse
            │
            ▼
    mqtt.publish_commands()
```

**Error handling:** Any exception in `_process` is caught, logged, and a chat error response is published. The queue loop continues regardless.

---

### `src/utils/logging.py` — Structured Logging

Uses `structlog` for structured log output.

- `APP_LOG_FORMAT=json` — JSON lines (production / log aggregation)
- `APP_LOG_FORMAT=console` — coloured human-readable output (development)
- `APP_LOG_LEVEL=DEBUG/INFO/WARNING/ERROR`

Log events use keyword arguments instead of format strings:
```python
logger.info("llm_inference_complete", tokens_predicted=45, tokens_evaluated=1371)
```

---

## Command Definitions (`data/commands.json`)

This file is the **single source of truth** for available commands. It is loaded at startup and used to:
1. Build the LLM system prompt (tell the model what commands exist)
2. Validate LLM responses (filter out hallucinated commands)
3. Drive entity resolution (know which commands need slot data)

```json
{
  "commands": [
    {
      "name": "call",
      "spellings": ["call", "called", "cold", "ring", "phone"],
      "sequence": ["WAKEWORD", "call"],
      "n_slots": 1,
      "slots_type": ["users"]
    },
    {
      "name": "vitals",
      "spellings": ["take vitals", "vitals"],
      "sequence": ["WAKEWORD", "vitals"],
      "n_slots": 0,
      "slots_type": []
    }
  ]
}
```

**Fields:**
- `name` — canonical command name used in the JSON output
- `spellings` — phrases that should trigger this command (used in system prompt)
- `n_slots` — number of data parameters needed (0 = no data, 1 = one value)
- `slots_type` — MongoDB collection to resolve the slot against (`"users"`, `"locations"`)
- `sequence` — legacy field from the speech-recognition era (not used by the LLM pipeline)

**To add a new command**, add an entry here. The GBNF grammar (`grammars/commands.gbnf`) must also be updated to include the new command name in the `cmd-name` rule.

---

## Grammar-Constrained Decoding (`grammars/commands.gbnf`)

The GBNF grammar is sent to llama-server with every request. llama-server applies it at the **token level during decoding** — the model can only produce tokens that are valid continuations of the grammar. This means:

- The output is **always valid JSON** — no need to handle malformed responses
- The `command` field is **always one of the defined names** — no hallucination
- The `status` field is **always `"pending"`** — no unexpected values

When you add a command to `commands.json`, also add its name to the `cmd-name` rule:

```gbnf
# Before
cmd-name ::= "\"call\"" | "\"vitals\"" | "\"camera\"" | "\"photo\"" | "\"dictation\""

# After adding "navigate"
cmd-name ::= "\"call\"" | "\"vitals\"" | "\"camera\"" | "\"photo\"" | "\"dictation\"" | "\"navigate\""
```

---

## System Prompt (`data/system_prompt.txt`)

The system prompt instructs the LLM on how to interpret input. It is a template with a `{{COMMANDS_BLOCK}}` placeholder that is replaced at startup with the command definitions from `commands.json`.

**Key behaviours defined in the prompt:**
- **Two modes:** COMMAND MODE (extract JSON) vs CONVERSATION MODE (reply naturally)
- **Spelling tolerance:** maps speech recognition errors (e.g. "cold" → "call")
- **Multi-command:** "take a photo then call John" → two commands
- **Conditional commands:** "if someone is there, take a photo" → conditional structure
- **Slot extraction:** extracts the data value for commands that need it

Edit `data/system_prompt.txt` to change LLM behaviour without touching the code.

---

## Running the System

### Development (no MQTT/MongoDB)

```bash
# Start llama-server (see docs/llama-server-mac.md)
llama-server --model ~/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  --host 0.0.0.0 --port 8080 -c 4096 --log-disable &

# Interactive test
uv run python scripts/chat.py
```

### Production (full stack)

```bash
# Required services
mosquitto &                          # MQTT broker
mongod --dbpath /data/db &           # MongoDB

# Start controller
cp .env.example .env                 # configure once
uv run python -m src.main
```

### Tests (no servers required)

```bash
uv run pytest tests/ -v
```

---

## Configuration Reference

```bash
# .env — copy from .env.example

APP_ROBOT_ID=robot-01                  # Robot identifier (used in MQTT topics)
APP_LOG_LEVEL=INFO                     # DEBUG, INFO, WARNING, ERROR
APP_LOG_FORMAT=console                 # console (dev) or json (prod)
APP_COMMANDS_FILE=data/commands.json   # Path to command definitions
APP_QUEUE_MAXSIZE=50                   # Max queued messages before backpressure
APP_MAX_CONCURRENT_LLM=1              # Parallel LLM calls (1 = serialised)

LLM_BASE_URL=http://127.0.0.1:8080    # llama-server address (use 127.0.0.1, not localhost)
LLM_TIMEOUT=30.0                       # Seconds before giving up on LLM
LLM_TEMPERATURE=0.1                    # Low = deterministic, high = creative
LLM_MAX_TOKENS=1024                    # Max tokens in LLM response

MQTT_HOST=localhost
MQTT_PORT=1883
MQTT_USERNAME=                         # Leave empty if no auth
MQTT_PASSWORD=

MONGO_URI=mongodb://localhost:27017    # Optional — only for entity resolution
MONGO_DATABASE=robot
```

---

## Extending the System

### Add a new command

1. **`data/commands.json`** — add the command entry
2. **`grammars/commands.gbnf`** — add the command name to `cmd-name`
3. Restart the controller — the system prompt regenerates automatically

### Change LLM behaviour

Edit `data/system_prompt.txt`. No code changes needed.

### Add a new entity type (e.g. locations)

1. Add `"slots_type": ["locations"]` to the command in `commands.json`
2. MongoDB collection name must match (`MONGO_LOCATIONS_COLLECTION=locations`)
3. The `EntityResolver.resolve_by_type` already handles `"locations"` via `resolve_location()`
