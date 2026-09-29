#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Agent Output Comparator — Startup Script (Mac / Linux)
# ─────────────────────────────────────────────────────────────

set -e

echo ""
echo "  ⚡  Agent Output Comparator"
echo "  ─────────────────────────────────────"

# 1. Install Python dependencies
echo "  Installing Python dependencies..."
pip install -r requirements.txt --quiet --break-system-packages 2>/dev/null \
  || pip install -r requirements.txt --quiet

# 2. Start the Flask backend in background
echo "  Starting backend on http://localhost:5050 ..."
python3 server.py &
BACKEND_PID=$!

# 3. Wait a moment then open the frontend
sleep 1
echo "  Opening frontend in browser..."

if command -v open &>/dev/null; then
  open http://localhost:5050      # macOS
elif command -v xdg-open &>/dev/null; then
  xdg-open http://localhost:5050  # Linux
else
  echo "  → Open http://localhost:5050 in your browser."
fi

echo ""
echo "  ✅ Running! Press Ctrl+C to stop."
echo ""

# Keep alive until user quits
trap "echo '  Stopping…'; kill $BACKEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM
wait $BACKEND_PID
