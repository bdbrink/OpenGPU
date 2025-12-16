#!/usr/bin/env bash
set -e

MODE="${1:-serve}"

echo "🚀 KubeTrainer starting in mode: ${MODE}"
echo "📂 DATA_DIR: ${DATA_DIR:-/data}"

case "${MODE}" in
  train)
    echo "🔥 Running training job"
    exec python3 infra_learning.py
    ;;
  serve)
    echo "🧠 Starting inference service"
    exec python3 interact.py
    ;;
  *)
    echo "❌ Unknown mode: ${MODE}"
    echo "Valid modes: train | serve"
    exit 1
    ;;
esac
