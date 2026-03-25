# Robot Command Controller

**LLM-powered voice/text command extraction for robotics on NVIDIA Jetson Orin NX 16GB (JetPack 5.1.x)**

## Architecture

```
STT Module ──► MQTT ──► Command Controller ──► MQTT ──► Robot Actuators
                              │
                    ┌─────────┼─────────┐
                    │         │         │
              llama-server  MongoDB  Entity Cache
              (Qwen2.5-7B)  (users,  (TTLCache)
               Q4_K_M GGUF  locations)
```

## Requirements

- NVIDIA Jetson Orin NX 16GB with JetPack 5.1.x (L4T R35.x, CUDA 11.4)
- Python 3.8+ (system Python on JP5.1 is 3.8)
- MongoDB 4.4+ (ARM64 build)
- Mosquitto MQTT Broker
- llama.cpp (pinned to compatible commit for CUDA 11.4)

## JetPack 5.1 Specific Notes

⚠️ **No Super Mode** — JetPack 5.1 maxes out at MAXN (25W for NX 16GB), GPU capped at 918 MHz.
⚠️ **CUDA 11.4** — Recent llama.cpp master may produce garbled output. Use the pinned version or jetson-containers JP5 branch.
⚠️ **Ubuntu 20.04** — Python 3.8 is system default. Use a venv with Python 3.10 if needed.

## Quick Start

```bash
# 1. System setup (run once)
sudo bash scripts/setup_jetson_jp5.sh

# 2. Start llama-server
bash scripts/start_llama_server.sh

# 3. Start the controller
python3 -m src.main
```

## Configuration

All configuration via environment variables or `.env` file. See `src/config/settings.py`.

## MQTT Topics

| Topic | Direction | Description |
|-------|-----------|-------------|
| `robot/{id}/input/text` | IN | Raw text from STT |
| `robot/{id}/commands/parsed` | OUT | Parsed command(s) JSON |
| `robot/{id}/chat/response` | OUT | Conversational response (no command) |
| `robot/{id}/status` | OUT (retained) | Controller health status |

## License

Proprietary
