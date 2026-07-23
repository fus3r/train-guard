#!/usr/bin/env bash
# train-guard for Apple Silicon Macs (M1/M2/M3/M4).
#
# Supervises one long running job and pauses it when the battery policy says it
# should not run. It uses SIGSTOP/SIGCONT to freeze and resume the job, and
# `taskpolicy` to request background scheduling when the pack is warm. macOS
# chooses the actual cores; taskpolicy does not guarantee efficiency-core use. It reads
# power source, charge %, and battery pack temperature without sudo. macOS already
# handles CPU/SoC thermal safety.
#
# Heat and battery cycling are what age a laptop battery; this protects both.
# Thresholds + rationale: see README.md. Policy: config.env (reread live).
#
# USAGE
#   train-guard run [--name N] [--restart-on-login] [--cwd DIR] -- <command...>   launch + supervise a job
#   train-guard attach --match "<pattern>" [--name N] [--restart-on-login --start "<cmd>"]   supervise a running job
#   train-guard attach --pid <PID> [--name N]
#   train-guard status                        power/battery/temp/health + guards + reboot handling
#   train-guard list                          active guards
#   train-guard stop <NAME> [--kill]          stop supervising (unfreeze); --kill ends the job
#   train-guard config                        show the policy file
#   train-guard restart-persisted             restart/reattach configured jobs at login
#   train-guard install-agent | uninstall-agent   install/remove the login restart agent
#   train-guard unpersist <NAME>              forget a persisted job
set -u

TG_HOME="${TRAIN_GUARD_HOME:-$HOME/.train-guard}"
CONFIG="$TG_HOME/config.env"
RUNDIR="$TG_HOME/run"
LOGDIR="$TG_HOME/logs"
PERSIST="$TG_HOME/persist"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)/$(basename "${BASH_SOURCE[0]}")"  # finds its own path
mkdir -p "$RUNDIR" "$LOGDIR" "$PERSIST" 2>/dev/null || true

# Config: defaults first, then config.env overrides.
load_config(){
  POLL=20
  RUN_ON_BATTERY=false; BATTERY_FLOOR_PCT=30; BATTERY_BAND=gentle; AC_BAND=full
  TEMP_GENTLE_C=38; TEMP_PAUSE_C=42; TEMP_RESUME_C=36
  CHARGE_COOL_UNTIL_PCT=80; TEMP_CHARGE_GENTLE_C=35
  unset TEMP_ECORE_C TEMP_CHARGE_ECORE_C
  # shellcheck disable=SC1090
  [ -f "$CONFIG" ] && . "$CONFIG"
  # v0.1 compatibility: old names overstated what taskpolicy can guarantee.
  [ -n "${TEMP_ECORE_C:-}" ] && TEMP_GENTLE_C="$TEMP_ECORE_C"
  [ -n "${TEMP_CHARGE_ECORE_C:-}" ] && TEMP_CHARGE_GENTLE_C="$TEMP_CHARGE_ECORE_C"
}

# Power and battery readings.
power_source(){ pmset -g batt 2>/dev/null | grep -q "AC Power" && echo AC || echo Battery; }
is_charging(){ ioreg -rn AppleSmartBattery -w0 2>/dev/null | grep -q '"IsCharging" = Yes' && echo yes || echo no; }
batt_pct(){ local p; p=$(pmset -g batt 2>/dev/null | grep -oE '[0-9]+%' | head -1 | tr -d '%'); echo "${p:-100}"; }
batt_temp_c(){ local t; t=$(ioreg -rn AppleSmartBattery -w0 2>/dev/null | grep -oE '"Temperature" = [0-9]+' | grep -oE '[0-9]+$'); [ -n "$t" ] && echo $((t/100)) || echo 0; }

# Process helpers.
descendants(){ # echo the given pids + all their descendants (one ps snapshot, BFS)
  local snap frontier="$*" acc="$*" next parent kids
  snap=$(ps -Ao pid=,ppid= 2>/dev/null)
  while [ -n "${frontier// /}" ]; do
    next=""
    for parent in $frontier; do
      kids=$(echo "$snap" | awk -v P="$parent" '$2==P{print $1}')
      next="$next $kids"
    done
    frontier="$next"; acc="$acc $next"
  done
  echo $acc | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -un | tr '\n' ' '
}
not_self(){ local p; for p in "$@"; do ps -o command= -p "$p" 2>/dev/null | grep -q "train-guard" || printf '%s ' "$p"; done; }

apply(){ # $1 = run|gentle|stop ; remaining args = pids
  local d="$1"; shift; local p
  for p in "$@"; do
    case "$d" in
      stop)  kill -STOP "$p" 2>/dev/null ;;
      gentle) kill -CONT "$p" 2>/dev/null; taskpolicy -b -p "$p" >/dev/null 2>&1 ;;
      run)   kill -CONT "$p" 2>/dev/null; taskpolicy -B -p "$p" >/dev/null 2>&1 ;;
    esac
  done
}

# Policy decision. Sets DECISION and SIG; COOLING provides thermal hysteresis.
COOLING=0
decide(){
  local src pct temp chg ac_band batt_band
  src=$(power_source); pct=$(batt_pct); temp=$(batt_temp_c); chg=$(is_charging)
  SIG="power=$src batt=${pct}% temp=${temp}C charging=$chg"
  case "$AC_BAND" in gentle|ecore) ac_band=gentle;; *) ac_band=run;; esac
  case "$BATTERY_BAND" in full|run) batt_band=run;; *) batt_band=gentle;; esac

  # thermal cooldown (highest priority, with hysteresis)
  if [ "$COOLING" = 1 ]; then
    if [ "$temp" -le "$TEMP_RESUME_C" ]; then COOLING=0; else DECISION=stop; return; fi
  fi
  if [ "$temp" -ge "$TEMP_PAUSE_C" ]; then COOLING=1; DECISION=stop; return; fi

  if [ "$src" = Battery ]; then
    if [ "$RUN_ON_BATTERY" != "true" ]; then DECISION=stop; return; fi
    if [ "$pct" -le "$BATTERY_FLOOR_PCT" ]; then DECISION=stop; return; fi
    DECISION="$batt_band"; return
  fi
  # on AC
  if [ "$temp" -ge "$TEMP_GENTLE_C" ]; then DECISION=gentle; return; fi
  if [ "$chg" = yes ] && [ "$pct" -lt "$CHARGE_COOL_UNTIL_PCT" ] && [ "$temp" -ge "$TEMP_CHARGE_GENTLE_C" ]; then DECISION=gentle; return; fi
  DECISION="$ac_band"
}

glog(){ echo "[guard] $(date '+%F %T') $*" >> "$GUARDLOG"; }
_meta_put(){ printf '%s=%q\n' "$1" "$2"; }   # quotes spaces so the file sources back cleanly

# Supervisor. Internal command: train-guard __supervise <metafile>.
cmd_supervise(){
  local meta="$1"; MODE=""; NAME=""; JOBPID=""; PATTERN=""; GUARDLOG=""
  # shellcheck disable=SC1090
  . "$meta"; GUARDLOG="${GUARDLOG:-$LOGDIR/$NAME.guard.log}"
  local roots pids lastsig="" miss=0 how p st
  glog "START name=$NAME mode=$MODE${PATTERN:+ pattern=\"$PATTERN\"}${JOBPID:+ jobpid=$JOBPID}"
  while true; do
    load_config
    if [ -f "$RUNDIR/$NAME.stop" ]; then
      how=$(cat "$RUNDIR/$NAME.stop" 2>/dev/null)
      if [ "$MODE" = run ]; then roots="${JOBPID:-}"; else roots=$(pgrep -f "$PATTERN" 2>/dev/null | tr '\n' ' '); fi
      pids=$(not_self $(descendants $roots))
      apply run $pids
      if [ "$how" = kill ]; then for p in $pids; do kill -TERM "$p" 2>/dev/null; done; glog "STOP --kill: terminated job"; else glog "STOP: unfroze job, detaching"; fi
      rm -f "$RUNDIR/$NAME.stop" "$RUNDIR/$NAME.meta" "$RUNDIR/$NAME.gpid"
      exit 0
    fi
    if [ "$MODE" = run ]; then
      if kill -0 "${JOBPID:-0}" 2>/dev/null; then roots="$JOBPID"; else glog "job exited; guard done"; rm -f "$RUNDIR/$NAME.meta" "$RUNDIR/$NAME.gpid"; exit 0; fi
    else
      roots=$(pgrep -f "$PATTERN" 2>/dev/null | tr '\n' ' ')
      if [ -z "${roots// /}" ]; then miss=$((miss+1)); [ $((miss%15)) = 1 ] && glog "no process matches pattern yet (waiting)"; sleep "$POLL"; continue; fi
      miss=0
    fi
    pids=$(not_self $(descendants $roots))
    decide
    apply "$DECISION" $pids
    if [ "$DECISION | $SIG" != "$lastsig" ]; then glog "-> $DECISION   ($SIG)"; lastsig="$DECISION | $SIG"; fi
    sleep "$POLL"
  done
}

# CLI commands.
_policy_line(){ printf "AC=%s  battery=%s" "$AC_BAND" "$([ "$RUN_ON_BATTERY" = true ] && echo "$BATTERY_BAND(floor ${BATTERY_FLOOR_PCT}%)" || echo pause)"; }

cmd_run(){
  local NAME="" PERSISTF="" CWD=""
  while [ $# -gt 0 ]; do case "$1" in
    --name) NAME="$2"; shift 2;;
    --restart-on-login|--persist) PERSISTF=1; shift;;
    --cwd) CWD="$2"; shift 2;;
    --) shift; break;;
    *) break;;
  esac; done
  [ $# -ge 1 ] || { echo "usage: train-guard run [--name NAME] [--restart-on-login] [--cwd DIR] -- <command...>" >&2; return 2; }
  load_config
  NAME="${NAME:-job-$(date +%H%M%S)}"
  local meta="$RUNDIR/$NAME.meta" LOG="$LOGDIR/$NAME.log" GLOG="$LOGDIR/$NAME.guard.log"
  if [ -f "$meta" ] && kill -0 "$(cat "$RUNDIR/$NAME.gpid" 2>/dev/null)" 2>/dev/null; then
    echo "a guard named '$NAME' is already active; pick another --name or stop it" >&2; return 1; fi
  if [ -n "$PERSISTF" ]; then
    { _meta_put PMODE run; _meta_put PNAME "$NAME"; _meta_put PCWD "${CWD:-$PWD}"; } > "$PERSIST/$NAME.job"
    printf '%s\n' "$@" > "$PERSIST/$NAME.argv"
  fi
  echo "[train-guard] launching '$NAME': $*"
  ( [ -n "$CWD" ] && cd "$CWD" 2>/dev/null; exec nohup "$@" >"$LOG" 2>&1 ) &
  local jobpid=$!
  { _meta_put MODE run; _meta_put NAME "$NAME"; _meta_put JOBPID "$jobpid"; _meta_put LOG "$LOG"; _meta_put GUARDLOG "$GLOG"; } > "$meta"
  nohup "$SELF" __supervise "$meta" >/dev/null 2>&1 & disown
  echo "$!" > "$RUNDIR/$NAME.gpid"
  echo "[train-guard] job pid=$jobpid  guard pid=$(cat "$RUNDIR/$NAME.gpid")  out=$LOG${PERSISTF:+  (restarts at next login)}"
  echo "[train-guard] policy: $(_policy_line)   |   train-guard status"
}

cmd_attach(){
  local NAME="" PATTERN="" PID="" PERSISTF="" CWD="" START=""
  while [ $# -gt 0 ]; do case "$1" in
    --name) NAME="$2"; shift 2;;
    --match) PATTERN="$2"; shift 2;;
    --pid) PID="$2"; shift 2;;
    --restart-on-login|--persist) PERSISTF=1; shift;;
    --cwd) CWD="$2"; shift 2;;
    --start) START="$2"; shift 2;;
    *) echo "attach: unexpected arg '$1'" >&2; return 2;;
  esac; done
  [ -n "$PATTERN" ] || [ -n "$PID" ] || { echo "usage: train-guard attach --match \"<pattern>\" [--name NAME] [--restart-on-login --start \"<cmd>\"]   (or --pid PID)" >&2; return 2; }
  if [ -n "$PID" ] && [ -n "$PERSISTF" ]; then
    echo "attach: a PID cannot survive a reboot; use --match with --restart-on-login and optionally --start" >&2
    return 2
  fi
  load_config
  NAME="${NAME:-attach-$(date +%H%M%S)}"
  local meta="$RUNDIR/$NAME.meta" GLOG="$LOGDIR/$NAME.guard.log"
  if [ -f "$meta" ] && kill -0 "$(cat "$RUNDIR/$NAME.gpid" 2>/dev/null)" 2>/dev/null; then
    echo "a guard named '$NAME' is already active" >&2; return 1; fi
  if [ -n "$PERSISTF" ] && [ -n "$PATTERN" ]; then
    { _meta_put PMODE attach; _meta_put PNAME "$NAME"; _meta_put PCWD "${CWD:-$PWD}"; _meta_put PPATTERN "$PATTERN"; _meta_put PSTART "${START:-}"; } > "$PERSIST/$NAME.job"
  fi
  if [ -n "$PID" ]; then
    { _meta_put MODE run; _meta_put NAME "$NAME"; _meta_put JOBPID "$PID"; _meta_put GUARDLOG "$GLOG"; } > "$meta"
  else
    { _meta_put MODE attach; _meta_put NAME "$NAME"; _meta_put PATTERN "$PATTERN"; _meta_put GUARDLOG "$GLOG"; } > "$meta"
  fi
  nohup "$SELF" __supervise "$meta" >/dev/null 2>&1 & disown
  echo "$!" > "$RUNDIR/$NAME.gpid"
  echo "[train-guard] attached '$NAME' (guard pid $(cat "$RUNDIR/$NAME.gpid")) to ${PATTERN:+pattern \"$PATTERN\"}${PID:+pid $PID}${PERSISTF:+  (restart/reattach configured for next login)}"
  echo "[train-guard] policy: $(_policy_line)   |   train-guard status"
}

cmd_status(){
  load_config
  local src pct temp chg
  src=$(power_source); pct=$(batt_pct); temp=$(batt_temp_c); chg=$(is_charging)
  echo "power / battery"
  pmset -g batt | sed -n '1,2p' | sed 's/^/  /'
  printf "  pack temp: %s°C    charging: %s    source: %s\n" "$temp" "$chg" "$src"
  echo
  echo "battery health"
  system_profiler SPPowerDataType 2>/dev/null | grep -iE 'Cycle Count|Condition|Maximum Capacity' | sed 's/^[[:space:]]*/  /'
  echo
  echo "policy (config.env)"
  printf "  %s\n" "$(_policy_line)"
  printf "  thermal: gentle >= %s°C  pause >= %s°C  resume <= %s°C   charge cool: <%s%% and >= %s°C\n" \
         "$TEMP_GENTLE_C" "$TEMP_PAUSE_C" "$TEMP_RESUME_C" "$CHARGE_COOL_UNTIL_PCT" "$TEMP_CHARGE_GENTLE_C"
  echo
  echo "active guards"
  local found=0 m gp alive roots st MODE NAME JOBPID PATTERN GUARDLOG
  shopt -s nullglob
  for m in "$RUNDIR"/*.meta; do
    found=1; MODE=""; NAME=""; JOBPID=""; PATTERN=""; GUARDLOG=""
    # shellcheck disable=SC1090
    . "$m"
    gp=$(cat "$RUNDIR/$NAME.gpid" 2>/dev/null); alive="dead"; kill -0 "${gp:-0}" 2>/dev/null && alive="running"
    echo "  $NAME [$MODE]  guard=$alive (pid ${gp:-?})"
    if [ "$MODE" = run ]; then roots="${JOBPID:-}"; else roots=$(pgrep -f "$PATTERN" 2>/dev/null | tr '\n' ' '); fi
    if [ -n "${roots// /}" ]; then
      st=$(ps -o stat= -p "${roots%% *}" 2>/dev/null)
      case "$st" in T*) echo "    worker: PAUSED / frozen [$st]";; "") echo "    worker: (gone)";; *) echo "    worker: running [$st]  pids: $roots";; esac
    else echo "    worker: none right now"; fi
    [ -f "${GUARDLOG:-/nonexistent}" ] && tail -n 1 "$GUARDLOG" 2>/dev/null | sed 's/^/    last: /'
  done
  [ "$found" = 0 ] && echo "  (none)"
  echo
  echo "restart after reboot (starts a new process; RAM state cannot survive a reboot)"
  if [ -f "$HOME/Library/LaunchAgents/com.trainguard.resume.plist" ]; then echo "  login agent: INSTALLED"; else echo "  login agent: not installed  ->  train-guard install-agent"; fi
  local pj pc=0; for pj in "$PERSIST"/*.job; do pc=$((pc+1)); echo "  restart spec: $(basename "$pj" .job)"; done
  [ "$pc" = 0 ] && echo "  restart specs: none (add --restart-on-login to run/attach)"
  return 0
}

cmd_list(){ shopt -s nullglob; local m n=0; for m in "$RUNDIR"/*.meta; do n=1; basename "$m" .meta; done; [ "$n" = 0 ] && echo "(no active guards)"; return 0; }

cmd_stop(){
  local NAME="" KILL=""
  while [ $# -gt 0 ]; do case "$1" in --kill) KILL=kill; shift;; *) NAME="$1"; shift;; esac; done
  [ -n "$NAME" ] || { echo "usage: train-guard stop <NAME> [--kill]" >&2; return 2; }
  [ -f "$RUNDIR/$NAME.meta" ] || { echo "no active guard named '$NAME' (train-guard list)" >&2; return 1; }
  echo "${KILL:-soft}" > "$RUNDIR/$NAME.stop"
  rm -f "$PERSIST/$NAME.job" "$PERSIST/$NAME.argv"
  echo "[train-guard] stop requested for '$NAME'${KILL:+ (--kill the job)}; unfreezes & detaches within one poll. (login restart removed)"
}

cmd_config(){ echo "policy: $CONFIG"; echo; cat "$CONFIG" 2>/dev/null; }

cmd_restart_persisted(){   # recreates configured jobs; it cannot restore pre-reboot RAM state
  shopt -s nullglob
  local j PMODE PNAME PCWD PPATTERN PSTART line; local -a args inv
  for j in "$PERSIST"/*.job; do
    PMODE=""; PNAME=""; PCWD=""; PPATTERN=""; PSTART=""
    # shellcheck disable=SC1090
    . "$j"; [ -n "$PNAME" ] || continue
    if [ -f "$RUNDIR/$PNAME.meta" ] && kill -0 "$(cat "$RUNDIR/$PNAME.gpid" 2>/dev/null)" 2>/dev/null; then
      echo "[restart] '$PNAME' already active; skipping"; continue; fi
    if [ "$PMODE" = run ]; then
      args=(); [ -f "$PERSIST/$PNAME.argv" ] && while IFS= read -r line; do args+=("$line"); done < "$PERSIST/$PNAME.argv"
      [ ${#args[@]} -gt 0 ] || { echo "[restart] '$PNAME' missing argv; skipping"; continue; }
      inv=(--name "$PNAME" --restart-on-login); [ -n "$PCWD" ] && inv+=(--cwd "$PCWD")
      cmd_run "${inv[@]}" -- "${args[@]}"
    else
      if [ -n "$PSTART" ] && ! pgrep -f "$PPATTERN" >/dev/null 2>&1; then
        ( [ -n "$PCWD" ] && cd "$PCWD" 2>/dev/null; nohup bash -c "$PSTART" >>"$LOGDIR/$PNAME.start.log" 2>&1 & )
        echo "[restart] started job for '$PNAME': $PSTART"
      fi
      inv=(--name "$PNAME" --restart-on-login --match "$PPATTERN"); [ -n "$PCWD" ] && inv+=(--cwd "$PCWD"); [ -n "$PSTART" ] && inv+=(--start "$PSTART")
      cmd_attach "${inv[@]}"
    fi
  done
  echo "[restart] done (commands were restarted or reattached; no RAM state was restored)"
}

cmd_unpersist(){ [ -n "${1:-}" ] || { echo "usage: train-guard unpersist <NAME>" >&2; return 2; }; rm -f "$PERSIST/$1.job" "$PERSIST/$1.argv" && echo "[train-guard] '$1' will no longer restart or reattach at login"; }

cmd_install_agent(){
  case "$(uname -s)" in Darwin) ;; *) echo "install-agent currently supports macOS (launchd) only. On Linux use the Python package's systemd user service." >&2; return 2;; esac
  local plist="$HOME/Library/LaunchAgents/com.trainguard.resume.plist"
  mkdir -p "$HOME/Library/LaunchAgents"
  cat > "$plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.trainguard.resume</string>
  <key>ProgramArguments</key>
  <array><string>/bin/bash</string><string>$SELF</string><string>restart-persisted</string></array>
  <key>RunAtLoad</key><true/>
  <key>ProcessType</key><string>Background</string>
  <key>StandardOutPath</key><string>$LOGDIR/restart.log</string>
  <key>StandardErrorPath</key><string>$LOGDIR/restart.log</string>
</dict>
</plist>
PLIST
  launchctl unload "$plist" 2>/dev/null
  if launchctl load -w "$plist" 2>/dev/null; then
    echo "[train-guard] login agent installed; configured commands restart after reboot/login."
  else
    echo "[train-guard] wrote $plist; load it manually: launchctl load -w \"$plist\""
  fi
  echo "  plist: $plist"
}

cmd_uninstall_agent(){
  local plist="$HOME/Library/LaunchAgents/com.trainguard.resume.plist"
  launchctl unload "$plist" 2>/dev/null
  rm -f "$plist" && echo "[train-guard] login agent removed." || echo "[train-guard] no agent installed."
}

cmd="${1:-status}"; [ $# -gt 0 ] && shift
case "$cmd" in
  run)             cmd_run "$@" ;;
  attach)          cmd_attach "$@" ;;
  status|"")       cmd_status "$@" ;;
  list)            cmd_list "$@" ;;
  stop)            cmd_stop "$@" ;;
  config)          cmd_config "$@" ;;
  restart-persisted|resume) cmd_restart_persisted "$@" ;;
  unpersist)       cmd_unpersist "$@" ;;
  install-agent)   cmd_install_agent "$@" ;;
  uninstall-agent) cmd_uninstall_agent "$@" ;;
  __supervise)     cmd_supervise "$@" ;;
  -h|--help|help)  sed -n '2,36p' "$SELF" ;;
  *) echo "unknown command: $cmd  (run|attach|status|stop|list|config|restart-persisted|install-agent)" >&2; exit 2 ;;
esac
