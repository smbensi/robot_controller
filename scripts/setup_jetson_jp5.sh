#!/usr/bin/env bash
# =============================================================================
# Jetson Orin NX 16GB — JetPack 5.1.x System Setup
# Run once after fresh flash. Requires sudo.
# =============================================================================
set -euo pipefail

echo "=========================================="
echo " Robot Controller — Jetson JP5.1 Setup"
echo "=========================================="

# ---------- 1. Power mode: MAXN (mode 0 on NX 16GB) ----------
echo "[1/7] Setting power mode to MAXN..."
sudo nvpmodel -m 0
sudo jetson_clocks
echo "  → MAXN enabled, clocks locked."

# ---------- 2. Disable desktop (headless) ----------
echo "[2/7] Switching to headless mode..."
sudo systemctl set-default multi-user.target
echo "  → Desktop disabled (reboot to take effect, saves ~800 MB RAM)."

# ---------- 3. Swap on NVMe (if available) ----------
echo "[3/7] Configuring NVMe swap..."
SWAP_FILE="/mnt/nvme/16GB.swap"
if [ -b /dev/nvme0n1 ]; then
    if [ ! -d /mnt/nvme ]; then
        sudo mkdir -p /mnt/nvme
        # If not already mounted, try mounting the first partition
        if ! mountpoint -q /mnt/nvme; then
            NVME_PART=$(lsblk -ln -o NAME /dev/nvme0n1 | tail -1)
            sudo mount "/dev/${NVME_PART}" /mnt/nvme 2>/dev/null || echo "  → NVMe partition not auto-mountable. Mount manually."
        fi
    fi
    if mountpoint -q /mnt/nvme && [ ! -f "$SWAP_FILE" ]; then
        sudo fallocate -l 16G "$SWAP_FILE"
        sudo chmod 600 "$SWAP_FILE"
        sudo mkswap "$SWAP_FILE"
        sudo swapon "$SWAP_FILE"
        echo "${SWAP_FILE} none swap sw 0 0" | sudo tee -a /etc/fstab > /dev/null
        echo "  → 16 GB NVMe swap created."
    else
        echo "  → NVMe swap already exists or not mountable."
    fi

    # Disable ZRAM (compresses into RAM — counterproductive for LLM)
    sudo systemctl disable nvzramconfig 2>/dev/null || true
    echo "  → ZRAM disabled."
else
    echo "  → No NVMe detected. Using default swap. Consider adding NVMe for better performance."
fi

# ---------- 4. Install system dependencies ----------
echo "[4/7] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    mosquitto mosquitto-clients \
    python3-pip python3-venv \
    build-essential cmake git \
    libcurl4-openssl-dev

# ---------- 5. MongoDB (ARM64 community server 4.4) ----------
echo "[5/7] Installing MongoDB 4.4..."
if ! command -v mongod &> /dev/null; then
    # MongoDB 4.4 is the last version with official ARM64 Ubuntu 20.04 support
    wget -qO - https://www.mongodb.org/static/pgp/server-4.4.asc | sudo apt-key add -
    echo "deb [ arch=arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/4.4 multiverse" | \
        sudo tee /etc/apt/sources.list.d/mongodb-org-4.4.list
    sudo apt-get update -qq
    sudo apt-get install -y -qq mongodb-org

    # Cap WiredTiger cache to 1 GB (critical for 16 GB shared memory)
    if ! grep -q "wiredTigerCacheSizeGB" /etc/mongod.conf; then
        sudo sed -i '/^storage:/a\  wiredTiger:\n    engineConfig:\n      cacheSizeGB: 1' /etc/mongod.conf
    fi
    sudo systemctl enable mongod
    sudo systemctl start mongod
    echo "  → MongoDB 4.4 installed, cache capped at 1 GB."
else
    echo "  → MongoDB already installed."
fi

# ---------- 6. Configure Mosquitto ----------
echo "[6/7] Configuring Mosquitto..."
MOSQUITTO_CONF="/etc/mosquitto/conf.d/robot.conf"
if [ ! -f "$MOSQUITTO_CONF" ]; then
    cat <<'EOF' | sudo tee "$MOSQUITTO_CONF" > /dev/null
# Robot Controller MQTT config
listener 1883
allow_anonymous true
max_queued_messages 1000
max_inflight_messages 20
EOF
    sudo systemctl restart mosquitto
    echo "  → Mosquitto configured."
else
    echo "  → Mosquitto config already exists."
fi

# ---------- 7. Build llama.cpp for CUDA 11.4 (JetPack 5.1) ----------
echo "[7/7] Building llama.cpp for CUDA 11.4..."
LLAMA_DIR="$HOME/llama.cpp"
if [ ! -d "$LLAMA_DIR" ]; then
    cd "$HOME"
    git clone https://github.com/ggml-org/llama.cpp.git
    cd llama.cpp

    # Pin to a known-good commit for CUDA 11.4 / compute_87
    # Recent master may have issues with Orin on JP5.1
    # Adjust this tag/commit as needed after testing
    git checkout b4568

    mkdir -p build && cd build
    cmake .. \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="87" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA_F16=ON
    cmake --build . --config Release -j$(nproc)
    echo "  → llama.cpp built successfully."
else
    echo "  → llama.cpp already cloned."
fi

echo ""
echo "=========================================="
echo " Setup complete!"
echo " Next steps:"
echo "  1. Reboot for headless mode: sudo reboot"
echo "  2. Download model: see scripts/download_model.sh"
echo "  3. Start llama-server: bash scripts/start_llama_server.sh"
echo "  4. Start controller: python3 -m src.main"
echo "=========================================="
