"""keep_awake_run.py --out LOG -- CMD ARGS...  : run CMD with stdout+stderr to LOG while asking Windows not to sleep.
Uses SetThreadExecutionState (per-process request, ends when this process exits; no system setting is changed)."""
import os
import subprocess
import sys

i = sys.argv.index("--")
out = sys.argv[sys.argv.index("--out") + 1]
if os.name == "nt":
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
with open(out, "a", encoding="utf-8") as f:
    rc = subprocess.call(sys.argv[i + 1:], stdout=f, stderr=subprocess.STDOUT)
sys.exit(rc)
