# Running llama-server on macOS

## Prerequisites
- macOS (Apple Silicon or Intel)
- Homebrew — install at https://brew.sh if needed
- ~6 GB free disk space for the model

---

## 1. Install llama.cpp

```bash
brew install llama.cpp
```

Homebrew builds with Metal GPU acceleration automatically on Apple Silicon.
Verify the binary is available:

```bash
which llama-server
llama-server --version
```

---

## 2. Download the model

The Q4_K_M variant is split into two files (~4.7 GB total):

```bash
mkdir -p ~/models

curl -L --progress-bar \
  -o ~/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"

curl -L --progress-bar \
  -o ~/models/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf \
  "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"
```

Verify:
```bash
ls -lh ~/models/qwen2.5-7b-instruct-q4_k_m-*.gguf
# Part 1: ~3.2 GB  Part 2: ~1.6 GB
```

---

## 3. Start llama-server

Pass only the **first part** — llama-server automatically finds and loads the rest:

Run in a **separate terminal** (it stays in the foreground):

```bash
llama-server \
  --model ~/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -c 4096 \
  --log-disable
```

Or run in the background:

```bash
llama-server \
  --model ~/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -c 4096 \
  --log-disable &
```

Key flags:
| Flag | Meaning |
|------|---------|
| `--host 0.0.0.0` | Listen on all interfaces (use `127.0.0.1` to restrict to localhost) |
| `--port 8080` | Port (matches `LLM_BASE_URL` default in `.env`) |
| `-c 4096` | Context window size — keep at 4096 for speed |
| `--log-disable` | Suppress verbose token logs (remove to debug) |

> **Apple Silicon note:** Metal acceleration is used automatically — no extra flags needed.
> **Intel Mac note:** Add `-t 8` (or however many cores you have) for CPU threading.

---

## 4. Verify it's running

```bash
curl http://127.0.0.1:8080/health
# Expected: {"status":"ok"}
```

---

## 5. Test the pipeline interactively

From the project root:

```bash
uv run python scripts/chat.py
```

Type any free text and the controller will return the matching command from `data/commands.json`.

---

## Differences from the Jetson setup

| | Mac | Jetson |
|--|-----|--------|
| GPU backend | Metal (automatic) | CUDA via `--ngl 99` |
| Power mode | n/a | `sudo nvpmodel -m 0` |
| Clock lock | n/a | `sudo jetson_clocks` |
| Memory mapping | default (fine) | `--no-mmap` (required) |
| Flash attention | not needed | `-fa` |

The `scripts/start_llama_server.sh` is Jetson-specific. Use the command in step 3 above on Mac.
