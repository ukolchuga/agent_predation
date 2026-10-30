#!/usr/bin/env bash
# Start predation.launch on the server as a CPU-capped background job (survives logout).
# All arguments go to predation.launch unchanged. Push the code first: git push svoi master
#
#   predation/scripts/train_on_server.sh --seeds 0-4 --iterations 200 --B 512
#   predation/scripts/train_on_server.sh predation/configs/<other>.yaml --seeds 0-9 --workers 4
#
# Environment: HOST (default svoi), CPU_QUOTA (default 350% = 3.5 of 4 cores), UNIT (default ba-train)
set -eu
HOST=${HOST:-svoi}
CPU_QUOTA=${CPU_QUOTA:-350%}
UNIT=${UNIT:-ba-train}
ARGS=$(printf ' %q' "$@")

ssh -o ClearAllForwardings=yes "$HOST" \
  "mkdir -p ~/ba/logs && systemd-run --user --unit=$UNIT --collect -p CPUQuota=$CPU_QUOTA \
   bash -c 'cd ~/ba && .venv/bin/python -m predation.launch$ARGS > logs/$UNIT.log 2>&1'"

cat <<EOF
started on $HOST as unit $UNIT (CPUQuota=$CPU_QUOTA)
  progress:   ssh $HOST 'cat ~/ba/logs/$UNIT.log'
  running?:   ssh $HOST 'systemctl --user is-active $UNIT'
  stop:       ssh $HOST 'systemctl --user stop $UNIT'
  live plots: ssh -N -L 6006:localhost:6006 $HOST   then http://localhost:6006
EOF
