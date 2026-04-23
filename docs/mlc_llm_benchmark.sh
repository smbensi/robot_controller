#!/usr/bin/env bash
# =============================================================================
# MLC-LLM Setup & Benchmark Script for Jetson Orin NX (JetPack 6.1/6.2)
# Model: Qwen2.5-7B-Instruct-q4f16_1-MLC
# =============================================================================
# Usage:
#   chmod +x mlc_llm_setup_benchmark.sh
#   ./mlc_llm_setup_benchmark.sh [COMMAND]
#
# Commands:
#   install       Install jetson-containers and pull MLC image
#   optimize      Set max clocks and power mode
#   serve         Start MLC-LLM server (interactive mode, default)
#   serve-local   Start MLC-LLM server (local mode, smaller KV cache)
#   serve-server  Start MLC-LLM server (server mode, max throughput)
#   benchmark     Run all benchmark configurations and compare
#   status        Show system status (GPU, memory, clocks, server health)
#   stop          Stop the running MLC-LLM container
#   help          Show this help message
#
# Examples:
#   ./mlc_llm_setup_benchmark.sh install
#   ./mlc_llm_setup_benchmark.sh optimize && ./mlc_llm_setup_benchmark.sh serve
#   ./mlc_llm_setup_benchmark.sh benchmark
# =============================================================================
 
set -euo pipefail
 
# =============================================================================
# CONFIGURATION — edit these to match your setup
# =============================================================================
MODEL_NAME="mlc-ai/Qwen2.5-7B-Instruct-q4f16_1-MLC"
MODEL_DIR="/data/models/mlc_llm"
CACHE_DIR="/mnt/nvme/cache"           # NVMe recommended; fallback: /data/cache
SERVER_PORT=9000
CONTAINER_NAME="mlc_llm_server"
HF_TOKEN="${HUGGINGFACE_TOKEN:-}"     # set via env or edit here: HF_TOKEN="hf_..."
MAX_TOKENS_BENCHMARK=512
BENCHMARK_PROMPT="Explain how a CPU works in detail"
WARMUP_RUNS=1
BENCHMARK_RUNS=3
 
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color
 
# =============================================================================
# HELPERS
# =============================================================================
log()     { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC}   $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERR]${NC}  $*"; }
header()  { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════${NC}"; \
            echo -e "${BOLD}${CYAN}  $*${NC}"; \
            echo -e "${BOLD}${CYAN}══════════════════════════════════════════${NC}"; }
 
check_root() {
    if [[ $EUID -ne 0 ]]; then
        warn "Some operations need sudo — you may be prompted for your password."
    fi
}
 
require_cmd() {
    command -v "$1" &>/dev/null || { error "Required command '$1' not found."; exit 1; }
}
 
server_running() {
    curl -sf "http://localhost:${SERVER_PORT}/v1/models" &>/dev/null
}
 
get_model_lib() {
    # Find the compiled .so file
    find "${MODEL_DIR}" -name "aarch64-cu126-sm87.so" 2>/dev/null | head -1
}
 
get_model_path() {
    find "${MODEL_DIR}" -name "mlc-chat-config.json" 2>/dev/null \
        | xargs -I{} dirname {} | head -1
}
 
# =============================================================================
# COMMAND: install
# =============================================================================
cmd_install() {
    header "Installing jetson-containers + MLC-LLM"
 
    require_cmd git
    require_cmd docker
 
    # 1. Clone jetson-containers
    if [[ ! -d "$HOME/jetson-containers" ]]; then
        log "Cloning jetson-containers..."
        git clone --depth=1 https://github.com/dusty-nv/jetson-containers \
            "$HOME/jetson-containers"
        success "Cloned jetson-containers"
    else
        log "jetson-containers already present — pulling latest..."
        git -C "$HOME/jetson-containers" pull --ff-only || \
            warn "Could not pull; using existing version."
    fi
 
    # 2. Install jetson-containers tooling
    log "Installing jetson-containers requirements..."
    pip3 install -q -r "$HOME/jetson-containers/requirements.txt"
    bash "$HOME/jetson-containers/install.sh" || true
    success "jetson-containers tooling installed"
 
    # 3. Ensure Docker NVIDIA runtime is default
    if ! docker info 2>/dev/null | grep -q "nvidia"; then
        warn "NVIDIA Docker runtime may not be configured as default."
        warn "Run: sudo nvidia-ctk runtime configure --runtime=docker --set-as-default"
    else
        success "NVIDIA Docker runtime is active"
    fi
 
    # 4. Create directories
    mkdir -p "${MODEL_DIR}" "${CACHE_DIR}"
    success "Created model dir: ${MODEL_DIR}"
    success "Created cache dir: ${CACHE_DIR}"
 
    # 5. Pull the MLC image
    log "Pulling MLC-LLM container image (this may take a while)..."
    MLC_IMAGE=$(cd "$HOME/jetson-containers" && ./autotag mlc 2>/dev/null || \
                echo "dustynv/mlc:0.20.0-r36.4.0")
    docker pull "${MLC_IMAGE}"
    success "MLC image ready: ${MLC_IMAGE}"
 
    # 6. Install tmux for persistent sessions
    if ! command -v tmux &>/dev/null; then
        log "Installing tmux..."
        sudo apt-get install -y tmux
    fi
    success "tmux installed"
 
    echo ""
    success "Installation complete!"
    echo -e "  Next steps:"
    echo -e "  1. ${CYAN}./mlc_llm_setup_benchmark.sh optimize${NC}  — maximize clocks"
    echo -e "  2. ${CYAN}./mlc_llm_setup_benchmark.sh serve${NC}     — start the server"
    echo -e "  3. ${CYAN}./mlc_llm_setup_benchmark.sh benchmark${NC} — run benchmarks"
}
 
# =============================================================================
# COMMAND: optimize
# =============================================================================
cmd_optimize() {
    header "Optimizing System Performance"
 
    # Power mode
    log "Setting MAXN power mode..."
    sudo nvpmodel -m 0
    success "Power mode set to MAXN (mode 0)"
 
    # Lock clocks
    log "Locking all clocks to maximum..."
    sudo jetson_clocks
    success "jetson_clocks applied"
 
    # Force performance governor on all CPU cores
    log "Setting CPU governor to performance on all cores..."
    sudo bash -c 'for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo performance > "$cpu"
    done'
    success "CPU governor set to performance on all $(nproc) cores"
 
    # Compact memory
    log "Compacting memory..."
    sudo bash -c 'echo 1 > /proc/sys/vm/compact_memory' || true
 
    # Verify
    echo ""
    log "Verification:"
    sudo jetson_clocks --show | grep -E "cpu0|GPU|EMC"
 
    # Make persistent with systemd
    log "Making clock settings persistent across reboots..."
    sudo tee /etc/systemd/system/jetson-performance.service > /dev/null << 'EOF'
[Unit]
Description=Maximize Jetson clocks for AI performance
After=multi-user.target
 
[Service]
Type=oneshot
ExecStart=/usr/sbin/nvpmodel -m 0
ExecStart=/usr/bin/jetson_clocks
ExecStart=/bin/bash -c "for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > $cpu; done"
RemainAfterExit=yes
 
[Install]
WantedBy=multi-user.target
EOF
 
    sudo systemctl daemon-reload
    sudo systemctl enable jetson-performance.service
    sudo systemctl start  jetson-performance.service
    success "Persistent performance service enabled"
 
    echo ""
    success "System fully optimized for maximum LLM throughput"
}
 
# =============================================================================
# COMMAND: serve (with mode selection)
# =============================================================================
_start_server() {
    local mode="$1"
    local context_size="$2"
    local description="$3"
 
    header "Starting MLC-LLM Server — ${description}"
 
    # Stop existing container if running
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        log "Stopping existing container..."
        docker stop "${CONTAINER_NAME}" 2>/dev/null || true
        sleep 2
    fi
 
    # Find compiled model library
    MODEL_PATH=$(get_model_path || true)
    MODEL_LIB=$(get_model_lib || true)
 
    # Determine if model needs to be downloaded (first run)
    if [[ -z "${MODEL_LIB}" ]]; then
        warn "No compiled .so found — model will compile on first run (~10 min)."
        warn "This is a one-time operation."
        # Use sudonim serve for first run (handles download + compilation)
        _serve_with_sudonim "${mode}" "${context_size}"
        return
    fi
 
    success "Found compiled model: ${MODEL_LIB}"
 
    # Build overrides string
    local overrides="tensor_parallel_shards=1;prefill_chunk_size=2048"
    if [[ "${context_size}" -gt 0 ]]; then
        overrides="${overrides};context_window_size=${context_size}"
    fi
 
    # HF token env
    local hf_env=""
    if [[ -n "${HF_TOKEN}" ]]; then
        hf_env="-e HUGGINGFACE_TOKEN=${HF_TOKEN}"
    fi
 
    log "Mode:          ${mode}"
    log "Context size:  ${context_size} tokens"
    log "Port:          ${SERVER_PORT}"
    log "Model lib:     ${MODEL_LIB}"
 
    # Get image tag
    local MLC_IMAGE
    MLC_IMAGE=$(cd "$HOME/jetson-containers" 2>/dev/null && ./autotag mlc 2>/dev/null || \
                echo "dustynv/mlc:0.20.0-r36.4.0")
 
    # Launch
    docker run \
        --runtime nvidia \
        --rm \
        --detach \
        --network host \
        --shm-size=8g \
        --name "${CONTAINER_NAME}" \
        --volume "${MODEL_DIR}:/data/models/mlc_llm" \
        --volume "${CACHE_DIR}:/root/.cache" \
        --volume /tmp/argus_socket:/tmp/argus_socket \
        --volume /etc/enctune.conf:/etc/enctune.conf \
        --volume /etc/nv_tegra_release:/etc/nv_tegra_release \
        --volume /tmp/nv_jetson_model:/tmp/nv_jetson_model \
        --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
        ${hf_env} \
        "${MLC_IMAGE}" \
        mlc_llm serve \
            "${MODEL_PATH}" \
            --model-lib "${MODEL_LIB}" \
            --mode "${mode}" \
            --overrides "${overrides}" \
            --host 0.0.0.0 \
            --port "${SERVER_PORT}" \
            --device cuda
 
    # Wait for server to be ready
    log "Waiting for server to start..."
    local retries=0
    until server_running || [[ $retries -ge 30 ]]; do
        sleep 2
        retries=$((retries + 1))
        echo -n "."
    done
    echo ""
 
    if server_running; then
        success "Server is running on http://localhost:${SERVER_PORT}"
        echo -e "  ${CYAN}Test it:${NC}  curl http://localhost:${SERVER_PORT}/v1/models"
        echo -e "  ${CYAN}Logs:${NC}     docker logs -f ${CONTAINER_NAME}"
        echo -e "  ${CYAN}Stop:${NC}     ./mlc_llm_setup_benchmark.sh stop"
    else
        error "Server did not start in time. Check logs:"
        echo "  docker logs ${CONTAINER_NAME}"
    fi
}
 
_serve_with_sudonim() {
    local mode="$1"
    local context_size="$2"
 
    local MLC_IMAGE
    MLC_IMAGE=$(cd "$HOME/jetson-containers" 2>/dev/null && ./autotag mlc 2>/dev/null || \
                echo "dustynv/mlc:0.20.0-r36.4.0")
 
    local hf_env=""
    if [[ -n "${HF_TOKEN}" ]]; then
        hf_env="-e HUGGINGFACE_TOKEN=${HF_TOKEN}"
    fi
 
    log "Using sudonim serve (handles first-run compilation automatically)"
 
    docker run \
        --runtime nvidia \
        --rm \
        --detach \
        --network host \
        --shm-size=8g \
        --name "${CONTAINER_NAME}" \
        --volume "${MODEL_DIR}:/data/models/mlc_llm" \
        --volume "${CACHE_DIR}:/root/.cache" \
        --volume /tmp/argus_socket:/tmp/argus_socket \
        --volume /etc/enctune.conf:/etc/enctune.conf \
        --volume /etc/nv_tegra_release:/etc/nv_tegra_release \
        --volume /tmp/nv_jetson_model:/tmp/nv_jetson_model \
        --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
        ${hf_env} \
        "${MLC_IMAGE}" \
        sudonim serve \
            --model "${MODEL_NAME}" \
            --quantization q4f16_1 \
            --max-batch-size 1 \
            --prefill-chunk 2048 \
            --host 0.0.0.0 \
            --port "${SERVER_PORT}"
 
    warn "Compilation in progress — this takes 10-15 minutes on first run."
    warn "Monitor progress with: docker logs -f ${CONTAINER_NAME}"
    warn "Run '${0} benchmark' after the server is ready."
}
 
cmd_serve()        { _start_server "interactive" 32768 "Interactive (1 request, 32K ctx)"; }
cmd_serve_local()  { _start_server "local"       4096  "Local (4 requests, 4K ctx)"; }
cmd_serve_server() { _start_server "server"      8192  "Server (80 requests, 8K ctx)"; }
 
# =============================================================================
# COMMAND: stop
# =============================================================================
cmd_stop() {
    header "Stopping MLC-LLM Server"
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        docker stop "${CONTAINER_NAME}"
        success "Container stopped"
    else
        warn "No running container named '${CONTAINER_NAME}' found"
    fi
}
 
# =============================================================================
# COMMAND: status
# =============================================================================
cmd_status() {
    header "System Status"
 
    # JetPack version
    echo -e "${BOLD}JetPack / L4T:${NC}"
    cat /etc/nv_tegra_release 2>/dev/null | head -1 || echo "  Not found"
 
    # Power mode
    echo -e "\n${BOLD}Power Mode:${NC}"
    sudo nvpmodel -q 2>/dev/null || echo "  nvpmodel not available"
 
    # Clock speeds
    echo -e "\n${BOLD}Clock Speeds:${NC}"
    sudo jetson_clocks --show 2>/dev/null | \
        grep -E "cpu0|GPU|EMC" | sed 's/^/  /'
 
    # Memory
    echo -e "\n${BOLD}Memory:${NC}"
    free -h | grep -E "Mem|Swap" | sed 's/^/  /'
 
    # GPU utilization (one sample)
    echo -e "\n${BOLD}GPU & Thermals (1 sample):${NC}"
    timeout 2 tegrastats 2>/dev/null | head -1 | \
        grep -oP 'RAM \d+/\d+MB|GR3D_FREQ \d+%|gpu@[\d.]+C' | \
        tr '\n' '  '
    echo ""
 
    # Docker containers
    echo -e "\n${BOLD}Running Containers:${NC}"
    docker ps --format "  {{.Names}}\t{{.Image}}\t{{.Status}}" 2>/dev/null || \
        echo "  Docker not accessible"
 
    # Server health
    echo -e "\n${BOLD}MLC Server (port ${SERVER_PORT}):${NC}"
    if server_running; then
        success "Online"
        curl -sf "http://localhost:${SERVER_PORT}/v1/models" | \
            python3 -c "import sys,json; m=json.load(sys.stdin)['data']; \
            [print(f'  Model: {x[\"id\"]}') for x in m]" 2>/dev/null || true
    else
        warn "Offline — run: ./mlc_llm_setup_benchmark.sh serve"
    fi
 
    # Compiled model library
    echo -e "\n${BOLD}Compiled Model Library:${NC}"
    local lib
    lib=$(get_model_lib || true)
    if [[ -n "${lib}" ]]; then
        local size
        size=$(du -sh "${lib}" | cut -f1)
        success "${lib} (${size})"
    else
        warn "Not found — server will compile on first run"
    fi
}
 
# =============================================================================
# COMMAND: benchmark
# =============================================================================
cmd_benchmark() {
    header "MLC-LLM Benchmark Suite"
 
    require_cmd python3
    require_cmd curl
 
    # Check server is up
    if ! server_running; then
        error "Server is not running. Start it first:"
        echo "  ./mlc_llm_setup_benchmark.sh serve"
        exit 1
    fi
 
    # Write the Python benchmark runner inline
    python3 << PYEOF
import time
import requests
import json
import sys
import statistics
 
SERVER   = "http://localhost:${SERVER_PORT}"
MODEL    = None   # auto-detect
PROMPT   = """${BENCHMARK_PROMPT}"""
WARMUP   = ${WARMUP_RUNS}
RUNS     = ${BENCHMARK_RUNS}
MAX_TOK  = ${MAX_TOKENS_BENCHMARK}
 
# ── helpers ────────────────────────────────────────────────────────────────
BOLD  = "\033[1m"
CYAN  = "\033[0;36m"
GREEN = "\033[0;32m"
YELL  = "\033[1;33m"
RED   = "\033[0;31m"
NC    = "\033[0m"
 
def hdr(title):
    print(f"\n{BOLD}{CYAN}{'─'*50}{NC}")
    print(f"{BOLD}{CYAN}  {title}{NC}")
    print(f"{BOLD}{CYAN}{'─'*50}{NC}")
 
def detect_model():
    r = requests.get(f"{SERVER}/v1/models", timeout=5)
    return r.json()["data"][0]["id"]
 
def run_completion(messages, max_tokens, stream=False):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": 0.0,   # deterministic for fair benchmarking
    }
    return requests.post(f"{SERVER}/v1/chat/completions",
                         json=payload, stream=stream, timeout=120)
 
def benchmark_standard(label, max_tokens):
    """Non-streaming benchmark — measures total wall time."""
    msgs = [{"role": "user", "content": PROMPT}]
    speeds = []
 
    print(f"\n  [{label}] Warming up ({WARMUP} run(s))...")
    for _ in range(WARMUP):
        run_completion(msgs, max_tokens)
 
    print(f"  [{label}] Benchmarking ({RUNS} run(s))...")
    for i in range(RUNS):
        t0 = time.time()
        r  = run_completion(msgs, max_tokens)
        elapsed = time.time() - t0
        data = r.json()
        toks = data["usage"]["completion_tokens"]
        spd  = toks / elapsed
        speeds.append(spd)
        reason = data["choices"][0]["finish_reason"]
        print(f"    Run {i+1}: {elapsed:.2f}s | {toks} tokens | "
              f"{spd:.1f} tok/s | stop={reason}")
 
    return speeds
 
def benchmark_streaming(label):
    """Streaming benchmark — separates prefill from decode."""
    msgs = [{"role": "user", "content": PROMPT}]
    results = []
 
    print(f"\n  [{label}] Streaming benchmark ({RUNS} run(s))...")
    for i in range(RUNS):
        t_start         = time.time()
        first_token_t   = None
        token_count     = 0
 
        with run_completion(msgs, MAX_TOK, stream=True) as resp:
            for line in resp.iter_lines():
                if not line or line == b"data: [DONE]":
                    continue
                raw = line.decode().removeprefix("data: ")
                try:
                    chunk = json.loads(raw)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta and first_token_t is None:
                        first_token_t = time.time()
                    if delta:
                        token_count += 1
                except Exception:
                    pass
 
        t_end        = time.time()
        prefill_ms   = (first_token_t - t_start) * 1000 if first_token_t else 0
        decode_s     = t_end - first_token_t if first_token_t else 0
        decode_spd   = token_count / decode_s if decode_s > 0 else 0
        total_spd    = token_count / (t_end - t_start) if (t_end - t_start) > 0 else 0
 
        results.append({
            "prefill_ms": prefill_ms,
            "decode_tok_s": decode_spd,
            "total_tok_s": total_spd,
            "tokens": token_count,
        })
        print(f"    Run {i+1}: prefill={prefill_ms:.0f}ms | "
              f"decode={decode_spd:.1f} tok/s | total={total_spd:.1f} tok/s | "
              f"tokens={token_count}")
 
    return results
 
def benchmark_context_lengths():
    """Test throughput across different prompt sizes."""
    short_prompt  = "What is 2+2?"
    medium_prompt = PROMPT
    long_prompt   = PROMPT * 4   # simulate long context
 
    results = {}
    for name, prompt, max_t in [
        ("Short  (< 10 tok input)",    short_prompt,  128),
        ("Medium (~27 tok input)",     medium_prompt, MAX_TOK),
        ("Long   (~100 tok input)",    long_prompt,   MAX_TOK),
    ]:
        msgs = [{"role": "user", "content": prompt}]
        t0  = time.time()
        r   = run_completion(msgs, max_t)
        dur = time.time() - t0
        d   = r.json()
        toks = d["usage"]["completion_tokens"]
        inp  = d["usage"]["prompt_tokens"]
        spd  = toks / dur
        results[name] = spd
        print(f"    {name}: input={inp} tok | output={toks} tok | {spd:.1f} tok/s")
 
    return results
 
def print_summary(label, speeds):
    if not speeds:
        return
    avg = statistics.mean(speeds)
    mn  = min(speeds)
    mx  = max(speeds)
    std = statistics.stdev(speeds) if len(speeds) > 1 else 0
    color = GREEN if avg >= 20 else YELL if avg >= 14 else RED
    print(f"\n  {BOLD}Result [{label}]:{NC}")
    print(f"    Average:  {color}{avg:.1f} tok/s{NC}")
    print(f"    Min/Max:  {mn:.1f} / {mx:.1f} tok/s")
    print(f"    Std dev:  ±{std:.1f} tok/s")
    return avg
 
# ── main ───────────────────────────────────────────────────────────────────
print(f"\n{BOLD}{'='*54}{NC}")
print(f"{BOLD}  MLC-LLM Benchmark Suite — Jetson Orin NX{NC}")
print(f"{BOLD}{'='*54}{NC}")
 
MODEL = detect_model()
print(f"\n  Model:   {CYAN}{MODEL}{NC}")
print(f"  Server:  {SERVER}")
print(f"  Prompt:  \"{PROMPT[:60]}...\"")
print(f"  Runs:    {WARMUP} warmup + {RUNS} measured")
 
# ── 1. Standard benchmark (full 512 tokens) ───────────────────────────────
hdr("1. Standard Throughput (512 tokens)")
speeds_512 = benchmark_standard("512 tok", MAX_TOK)
avg_512 = print_summary("512 tokens", speeds_512)
 
# ── 2. Short response benchmark (128 tokens) ──────────────────────────────
hdr("2. Short Response Throughput (128 tokens)")
speeds_128 = benchmark_standard("128 tok", 128)
avg_128 = print_summary("128 tokens", speeds_128)
 
# ── 3. Streaming — prefill vs decode separation ───────────────────────────
hdr("3. Streaming — Prefill vs Decode Breakdown")
stream_results = benchmark_streaming("streaming")
if stream_results:
    avg_prefill = statistics.mean(r["prefill_ms"]  for r in stream_results)
    avg_decode  = statistics.mean(r["decode_tok_s"] for r in stream_results)
    avg_total   = statistics.mean(r["total_tok_s"]  for r in stream_results)
    print(f"\n  {BOLD}Streaming Summary:{NC}")
    print(f"    Time to first token (prefill): {avg_prefill:.0f} ms")
    color = GREEN if avg_decode >= 20 else YELL if avg_decode >= 14 else RED
    print(f"    Decode speed:                  {color}{avg_decode:.1f} tok/s{NC}")
    print(f"    End-to-end speed:              {avg_total:.1f} tok/s")
 
# ── 4. Context length sensitivity ─────────────────────────────────────────
hdr("4. Context Length Sensitivity")
ctx_results = benchmark_context_lengths()
 
# ── 5. Concurrency test (2 parallel requests) ─────────────────────────────
hdr("5. Concurrency (2 simultaneous requests)")
import threading
 
concurrent_speeds = []
def single_req():
    msgs = [{"role": "user", "content": PROMPT}]
    t0 = time.time()
    r  = run_completion(msgs, 128)
    elapsed = time.time() - t0
    try:
        toks = r.json()["usage"]["completion_tokens"]
        concurrent_speeds.append(toks / elapsed)
    except Exception:
        concurrent_speeds.append(0)
 
threads = [threading.Thread(target=single_req) for _ in range(2)]
wall_start = time.time()
for t in threads: t.start()
for t in threads: t.join()
wall_total = time.time() - wall_start
 
print(f"\n    2 parallel requests completed in {wall_total:.2f}s")
if concurrent_speeds:
    print(f"    Per-request avg: {statistics.mean(concurrent_speeds):.1f} tok/s each")
    print(f"    Combined throughput: {sum(concurrent_speeds):.1f} tok/s total")
 
# ── Final Comparison Table ─────────────────────────────────────────────────
print(f"\n{BOLD}{'='*54}{NC}")
print(f"{BOLD}  BENCHMARK SUMMARY{NC}")
print(f"{BOLD}{'='*54}{NC}")
 
rows = [
    ("Scenario",                     "Speed",      "vs llama.cpp baseline"),
    ("─"*30,                         "─"*12,       "─"*22),
    ("llama.cpp Q4_K_M (baseline)",  "~5-10 tok/s","1.0x (reference)"),
    (f"MLC 512 tok output",
     f"{avg_512:.1f} tok/s" if avg_512 else "n/a",
     f"{avg_512/7.5:.1f}x faster" if avg_512 else "n/a"),
    (f"MLC 128 tok output",
     f"{avg_128:.1f} tok/s" if avg_128 else "n/a",
     f"{avg_128/7.5:.1f}x faster" if avg_128 else "n/a"),
    ("MLC decode only (streaming)",
     f"{avg_decode:.1f} tok/s" if stream_results else "n/a",
     f"{avg_decode/7.5:.1f}x faster" if stream_results else "n/a"),
    ("JetPack 6.2 target (reference)","~23.5 tok/s","~3x faster than llama.cpp"),
]
 
for r in rows:
    print(f"  {r[0]:<32} {r[1]:<14} {r[2]}")
 
print(f"\n{BOLD}Interpretation:{NC}")
if avg_512:
    if avg_512 >= 20:
        print(f"  {GREEN}Excellent!{NC} You're at or near the JetPack 6.1 ceiling.")
    elif avg_512 >= 14:
        print(f"  {YELL}Good.{NC} This is expected for JetPack 6.1 on Orin NX 16GB.")
        print(f"  Upgrading to JetPack 6.2 + Super Mode could reach ~23 tok/s.")
    else:
        print(f"  {RED}Below expected.{NC} Check:")
        print(f"    - sudo nvpmodel -m 0 && sudo jetson_clocks")
        print(f"    - No competing GPU processes (docker ps / ps aux)")
 
print(f"\n  Results saved to: benchmark_results.json")
with open("benchmark_results.json", "w") as f:
    json.dump({
        "model": MODEL,
        "avg_512_tok_s": avg_512,
        "avg_128_tok_s": avg_128,
        "streaming_prefill_ms": avg_prefill if stream_results else None,
        "streaming_decode_tok_s": avg_decode if stream_results else None,
        "context_sensitivity": ctx_results,
    }, f, indent=2)
PYEOF
}
 
# =============================================================================
# COMMAND: help
# =============================================================================
cmd_help() {
    echo -e "${BOLD}MLC-LLM Setup & Benchmark — Jetson Orin NX${NC}"
    echo ""
    echo -e "${CYAN}Usage:${NC}  $0 <command>"
    echo ""
    echo -e "${CYAN}Commands:${NC}"
    echo "  install        Install jetson-containers and pull MLC image"
    echo "  optimize       Set MAXN power mode + lock all clocks (run first!)"
    echo "  serve          Start server — interactive mode (1 req, 32K ctx)"
    echo "  serve-local    Start server — local mode (4 req, 4K ctx, less memory)"
    echo "  serve-server   Start server — server mode (80 req, max throughput)"
    echo "  benchmark      Run full benchmark suite and compare configurations"
    echo "  status         Show system health: GPU, clocks, memory, server"
    echo "  stop           Stop the MLC-LLM container"
    echo "  help           Show this help"
    echo ""
    echo -e "${CYAN}Recommended workflow (first time):${NC}"
    echo "  1.  $0 install"
    echo "  2.  $0 optimize"
    echo "  3.  $0 serve          # waits ~10 min first run (compilation)"
    echo "  4.  $0 benchmark"
    echo ""
    echo -e "${CYAN}Environment variables:${NC}"
    echo "  HUGGINGFACE_TOKEN    HuggingFace token for model download"
    echo ""
    echo -e "${CYAN}Configuration (edit top of script):${NC}"
    echo "  MODEL_NAME           HuggingFace model ID"
    echo "  MODEL_DIR            Where model weights are stored"
    echo "  CACHE_DIR            Cache directory (use NVMe!)"
    echo "  SERVER_PORT          REST API port (default: 9000)"
    echo "  MAX_TOKENS_BENCHMARK Tokens to generate in benchmark (default: 512)"
}
 
# =============================================================================
# ENTRYPOINT
# =============================================================================
check_root
 
COMMAND="${1:-help}"
case "${COMMAND}" in
    install)       cmd_install ;;
    optimize)      cmd_optimize ;;
    serve)         cmd_serve ;;
    serve-local)   cmd_serve_local ;;
    serve-server)  cmd_serve_server ;;
    benchmark)     cmd_benchmark ;;
    status)        cmd_status ;;
    stop)          cmd_stop ;;
    help|--help|-h) cmd_help ;;
    *)
        error "Unknown command: ${COMMAND}"
        echo ""
        cmd_help
        exit 1
        ;;
esac