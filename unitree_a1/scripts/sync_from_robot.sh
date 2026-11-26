#!/usr/bin/env bash
# rsync logs from Unitree A1 motion computer back to local machine
# Usage: ./scripts/sync_from_robot.sh [robot_ip]

set -e

ROBOT_IP="${1:-192.168.123.161}"
ROBOT_USER="unitree"
ROBOT_PATH="~/unitree_a1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Syncing logs from A1 Motion Computer ==="

mkdir -p "$PROJECT_ROOT/logs"

rsync -avz --progress \
    "$ROBOT_USER@$ROBOT_IP:$ROBOT_PATH/logs/" "$PROJECT_ROOT/logs/"

echo "=== Logs sync complete! ==="
echo "Check new logs in $PROJECT_ROOT/logs/"
