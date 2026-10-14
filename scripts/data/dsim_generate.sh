#!/usr/bin/env bash
# Generate D-sim routes with the Windows HoloOcean Python, $JOBS at a time (8 GB of VRAM holds three instances; GPU
# throughput is shared, so more instances do not finish sooner). Routes are taken one by one from the queue file
# runs/dsim/queue.txt (append lines to add work while it runs); each goes to data/sim/<route>_s01/raw, and an existing
# raw/meta.json is skipped. Workers exit when the queue is empty.
# Usage: JOBS=3 bash scripts/data/dsim_generate.sh [<route> ...]   (arguments are appended to the queue first)
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p runs/dsim
Q=runs/dsim/queue.txt
touch $Q
printf '%s\n' "$@" | grep -v '^$' >> $Q
pop() { flock $Q.lock bash -c "r=\$(head -n 1 $Q); sed -i 1d $Q; echo \$r"; }
worker() {
  while r=$(pop) && [ -n "$r" ]; do
    out=data/sim/${r}_s01
    if [ -e "$out/raw/meta.json" ]; then echo "$(date -Is) $r exists" >> runs/dsim/generate.log; continue; fi
    rm -rf "$out/raw"; mkdir -p "$out"
    python.exe team/sim/holoocean_generate.py configs/sim/$r.json $out/raw > runs/dsim/gen_${r}_s01.log 2>&1
    echo "$(date -Is) $r rc=$? $(tail -n 1 runs/dsim/gen_${r}_s01.log)" >> runs/dsim/generate.log
  done
}
for i in $(seq "${JOBS:-3}"); do worker & sleep 15; done
wait
