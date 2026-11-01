#!/usr/bin/env bash
# ncu on the GDN decode kernel (27B shapes, 48 layers' states, real clocks): median duration,
# DRAM bytes read/written per launch and achieved DRAM GB/s. Usage: ncu_gdn.sh <n rows> <q variant>
set -euo pipefail
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY=${PY:-python3}
OUT=$(mktemp)
ncu --clock-control none --kernel-name regex:gdn --launch-skip 96 --launch-count 16 \
    --metrics gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum --csv \
    "$PY" "$Q/bench/profile/ncu_gdn.py" "$1" "$2" "${3:-1}" "${4:-1}" 2>/dev/null > "$OUT"
"$PY" - "$OUT" "$1" "$2" "${3:-1}" "${4:-1}" <<'PYEOF'
import csv, io, sys, statistics as st
txt = open(sys.argv[1]).read(); txt = txt[txt.index('"ID"'):]
m = {}
for r in csv.DictReader(io.StringIO(txt)):
    try:
        v = float(r["Metric Value"].replace(",", ""))
    except ValueError:
        continue
    scale = {"ns": 1e-9, "us": 1e-6, "usecond": 1e-6, "msecond": 1e-3, "nsecond": 1e-9, "byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}.get(r["Metric Unit"], 1)
    m.setdefault(r["Metric Name"], []).append(v * scale)
t = st.median(m["gpu__time_duration.sum"]); rd = st.median(m["dram__bytes_op_read.sum"]); wr = st.median(m["dram__bytes_op_write.sum"])
print(f"gdn n={sys.argv[2]} q={sys.argv[3]} early={sys.argv[4]} phases={sys.argv[5]}: {t*1e6:.2f} us, DRAM read {rd/1e6:.2f} MB + write {wr/1e6:.2f} MB, {(rd+wr)/t/1e9:.0f} GB/s")
PYEOF
rm -f "$OUT"
