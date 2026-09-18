#!/bin/bash
# Run the brain's trader as a macOS background service (launchd) that starts at login,
# restarts if it stops, keeps the Mac awake while running, and logs to .brain/trade.log.
#
#   scripts/macos-service.sh install [--book main] [--top 100] [--interval 15m]
#   scripts/macos-service.sh status
#   scripts/macos-service.sh log          # follow the log
#   scripts/macos-service.sh uninstall
set -euo pipefail

LABEL="com.bigbrain.trade"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$PROJECT/.venv/bin/python"
LOG="$PROJECT/.brain/trade.log"

cmd="${1:-status}"; shift || true

case "$cmd" in
  install)
    [ -x "$PYTHON" ] || { echo "no virtualenv at $PROJECT/.venv; run: python3 -m venv .venv && source .venv/bin/activate && pip install -e ."; exit 1; }
    mkdir -p "$PROJECT/.brain" "$HOME/Library/LaunchAgents"
    ARGS=""
    for a in "$@"; do ARGS="$ARGS<string>$a</string>"; done
    cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>/usr/bin/caffeinate</string><string>-s</string>
    <string>$PYTHON</string><string>-m</string><string>bigbrain.cli</string><string>trade</string>$ARGS
  </array>
  <key>WorkingDirectory</key><string>$PROJECT</string>
  <key>EnvironmentVariables</key><dict><key>PYTHONUNBUFFERED</key><string>1</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
</dict></plist>
PLIST
    launchctl unload "$PLIST" 2>/dev/null || true
    launchctl load "$PLIST"
    echo "installed and started: $LABEL"
    echo "  log:       $LOG   (scripts/macos-service.sh log)"
    echo "  dashboard: bigbrain dashboard   (in any terminal)"
    echo "  note: caffeinate keeps the Mac from sleeping while the trader runs; closing the lid still sleeps a MacBook unless it is plugged in with an external display, or you disable lid sleep."
    ;;
  uninstall)
    launchctl unload "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    echo "removed $LABEL (the book and the brain are untouched)"
    ;;
  status)
    if launchctl list | grep -q "$LABEL"; then
      echo "$LABEL is loaded:"; launchctl list | grep "$LABEL"
      [ -f "$LOG" ] && { echo; tail -n 5 "$LOG"; }
    else
      echo "$LABEL is not installed"
    fi
    ;;
  log)
    tail -f "$LOG"
    ;;
  *)
    echo "usage: $0 install|status|log|uninstall"; exit 1;;
esac
