#!/bin/bash
# Show or stop Unitree's sport controller on the A1 motion computer, so low-level commands have no competitor.
#
#     scripts/sport_mode.sh status
#     sudo scripts/sport_mode.sh stop
#
# keep_program_alive.sh starts A1_sport_2 within 0.2 s of A1_sport_1 dying, so the supervisor goes first. Nothing
# on disk changes: reboot to get sport mode back (autostart runs start.sh).

SUPERVISOR='keep_program_alive\.sh'
STARTER='keep_program_alive/start\.sh'
SPORT='bin/A1_sport_[12]$'
IFACE="${IFACE:-enp1s0}"
MCU=192.168.123.10

pids() {
  pgrep -f "$1"
}

show() {
  local p
  p=$(pids "$1")
  if [ -n "$p" ]; then
    # shellcheck disable=SC2086
    ps -o pid= -o ppid= -o cls= -o rtprio= -o etime= -o cmd= -p $(echo $p | tr ' ' ',')
  else
    echo "(none)"
  fi
}

traffic() {
  if [ "$(id -u)" -ne 0 ] || ! command -v tcpdump >/dev/null; then
    echo "traffic: skipped (needs root and tcpdump)"
    return
  fi
  local out in
  out=$(timeout 2 tcpdump -nn -i "$IFACE" -q "udp and dst host $MCU" 2>/dev/null | wc -l)
  in=$(timeout 2 tcpdump -nn -i "$IFACE" -q "udp and src host $MCU" 2>/dev/null | wc -l)
  echo "traffic over 2 s windows: to MCU $out packets, from MCU $in packets"
}

status() {
  echo "--- supervisor"
  show "$SUPERVISOR"
  echo "--- sport controller"
  show "$SPORT"
  traffic
}

# Waits up to $2 tenths of a second for no process to match $1.
wait_gone() {
  local i
  for ((i = 0; i < $2; i++)); do
    [ -z "$(pids "$1")" ] && return 0
    sleep 0.1
  done
  [ -z "$(pids "$1")" ]
}

stop() {
  if [ -z "$(pids "$SPORT")" ] && [ -z "$(pids "$SUPERVISOR")" ]; then
    echo "sport controller and supervisor are already stopped"
    status
    return 0
  fi

  echo "The motors may lose their commanded torque. The robot must be hanging, or lying on the ground, never standing."
  read -r -p "Type SAFE to continue: " answer
  if [ "$answer" != "SAFE" ]; then
    echo "not stopped"
    return 1
  fi

  # start.sh first: during its boot sleep it would otherwise still launch the supervisor.
  # shellcheck disable=SC2046
  [ -n "$(pids "$STARTER")" ] && kill -TERM $(pids "$STARTER")
  # shellcheck disable=SC2046
  [ -n "$(pids "$SUPERVISOR")" ] && kill -TERM $(pids "$SUPERVISOR")
  if ! wait_gone "$SUPERVISOR" 20; then
    echo "ABORT: the supervisor is still running, sport controller left untouched. Is this running with sudo?"
    status
    return 1
  fi

  # shellcheck disable=SC2046
  [ -n "$(pids "$SPORT")" ] && kill -TERM $(pids "$SPORT")
  if ! wait_gone "$SPORT" 30; then
    echo "FAIL: the sport controller ignored SIGTERM. Supervisor is stopped. Not escalating; reboot to recover."
    status
    return 1
  fi

  # A supervisor that survived would have started the other controller by now.
  sleep 1
  if [ -n "$(pids "$SPORT")" ] || [ -n "$(pids "$SUPERVISOR")" ]; then
    echo "FAIL: something restarted"
    status
    return 1
  fi
  echo "stopped. Reboot to restore sport mode."
  status
}

case "$1" in
  status) status ;;
  stop) stop ;;
  *)
    echo "usage: $0 status|stop"
    exit 2
    ;;
esac
