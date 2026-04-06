#!/usr/bin/env python3
"""Measure RAM during SDXL generation — captures MemAvailable before/during/after each phase."""
import subprocess, time, re, json
from pathlib import Path

ADB = str(Path(__file__).resolve().parent.parent / "adb.exe")

def memavail():
    r = subprocess.run([ADB, "shell", "grep MemAvailable /proc/meminfo"],
                       capture_output=True, text=True, timeout=5)
    m = re.search(r'(\d+)', r.stdout)
    return int(m.group(1)) if m else 0  # kB

def memtotal():
    r = subprocess.run([ADB, "shell", "grep MemTotal /proc/meminfo"],
                       capture_output=True, text=True, timeout=5)
    m = re.search(r'(\d+)', r.stdout)
    return int(m.group(1)) if m else 0

samples = []
def snap(label):
    kb = memavail()
    samples.append({"label": label, "avail_mb": kb // 1024, "time": time.time()})
    print(f"  RAM [{label}]: {kb//1024} MB available")

if __name__ == "__main__":
    total_kb = memtotal()
    print(f"Total RAM: {total_kb // 1024} MB")
    snap("baseline")

    # Run generate.py as subprocess, poll RAM in parallel
    import threading, sys

    stop_poll = threading.Event()
    poll_log = []

    def poll_ram():
        while not stop_poll.is_set():
            try:
                kb = memavail()
                poll_log.append({"t": time.time(), "avail_mb": kb // 1024})
            except:
                pass
            time.sleep(1.5)

    t = threading.Thread(target=poll_ram, daemon=True)
    t.start()

    snap("before_generate")
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).parent / "generate.py"),
         "1girl, anime, cherry blossoms, masterpiece, best quality",
         "--seed", "42", "--name", "ram_test"],
        cwd=str(Path(__file__).parent.parent)
    )
    proc.wait()
    snap("after_generate")

    stop_poll.set()
    t.join(timeout=3)

    # Report
    if poll_log:
        min_mb = min(p["avail_mb"] for p in poll_log)
        max_mb = max(p["avail_mb"] for p in poll_log)
        used_peak_mb = (total_kb // 1024) - min_mb
        print(f"\n{'='*50}")
        print(f"Total RAM:       {total_kb // 1024} MB")
        print(f"Baseline avail:  {samples[0]['avail_mb']} MB")
        print(f"Min avail:       {min_mb} MB (peak pressure)")
        print(f"Max avail:       {max_mb} MB")
        print(f"Peak used:       {used_peak_mb} MB")
        print(f"Delta baseline:  {samples[0]['avail_mb'] - min_mb} MB consumed by pipeline")

        out = Path(__file__).parent / "outputs" / "ram_profile.json"
        out.write_text(json.dumps({
            "total_mb": total_kb // 1024,
            "baseline_avail_mb": samples[0]["avail_mb"],
            "min_avail_mb": min_mb,
            "peak_used_mb": used_peak_mb,
            "delta_mb": samples[0]["avail_mb"] - min_mb,
            "poll_samples": poll_log[:20],  # first 20 points
        }, indent=2))
        print(f"Saved: {out}")
