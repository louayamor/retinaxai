#!/usr/bin/env bash
set -euo pipefail

# Run this ONCE on the e2-micro VM to install ChromaDB as a systemd service.
# Usage: scp infra/infra/chromadb/*.service e2-micro:~ && ssh e2-micro sudo ./setup-chromadb.sh

CHROMA_DIR="/data/chroma"
CHROMA_USER="chroma"
VENV_DIR="/opt/chroma/venv"

echo "[1/4] Creating chroma user..."
id -u "$CHROMA_USER" &>/dev/null || sudo useradd --system --no-create-home "$CHROMA_USER"

echo "[2/4] Creating data directory..."
sudo mkdir -p "$CHROMA_DIR"
sudo chown "$CHROMA_USER":"$CHROMA_USER" "$CHROMA_DIR"

echo "[3/4] Installing ChromaDB..."
sudo mkdir -p /opt/chroma
sudo python3 -m venv "$VENV_DIR"
sudo "$VENV_DIR/bin/pip" install --no-cache-dir chromadb>=0.5.0
sudo chown -R "$CHROMA_USER":"$CHROMA_USER" /opt/chroma

echo "[4/4] Installing systemd service..."
sudo cp chromadb.service /etc/systemd/system/chromadb.service
sudo systemctl daemon-reload
sudo systemctl enable chromadb
sudo systemctl start chromadb
sudo systemctl status chromadb --no-pager

echo "Done. ChromaDB listening on 0.0.0.0:8000"
