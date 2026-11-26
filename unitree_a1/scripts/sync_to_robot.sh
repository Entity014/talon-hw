#!/usr/bin/env bash
# rsync local code to Unitree A1 motion computer
# Usage: ./scripts/sync_to_robot.sh [robot_ip]

set -e

ROBOT_IP="${1:-192.168.123.161}"
ROBOT_USER="unitree"
ROBOT_PATH="~/unitree_a1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Syncing to A1 Motion Computer ($ROBOT_USER@$ROBOT_IP:$ROBOT_PATH) ==="

# Check connectivity first
if ! ping -c 1 -W 2 "$ROBOT_IP" > /dev/null 2>&1; then
    echo "ERROR: Cannot ping $ROBOT_IP. Are you connected to the robot network?"
    exit 1
fi

# Rsync with excludes (don't overwrite build artifacts or logs on robot)
rsync -avz --no-g --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.pytest_cache' \
    --exclude 'build/' \
    --exclude 'logs/*.npz' \
    --exclude '*.mp4' \
    --exclude '*.pcap' \
    "$PROJECT_ROOT/" "$ROBOT_USER@$ROBOT_IP:$ROBOT_PATH/"

echo "=== Sync complete! ==="
echo ""
echo "Next steps on robot:"
echo "  ssh $ROBOT_USER@$ROBOT_IP"
echo "  cd $ROBOT_PATH"
echo "  cmake --build build -j2  # (if C++ changed)"
echo "  sudo python3 scripts/run.py --policy policies/oursA/policy.npz --soft-start 2.0 --w 0.8 0.1 0.1 --dq-noise 1.5 --tau-limit 33 --flip-imu-x"
