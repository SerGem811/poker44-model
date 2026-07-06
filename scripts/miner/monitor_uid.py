"""
Continuous monitor for Poker44 UID 123 — checks metagraph every 5 min,
logs score changes, and counts live scoring queries from PM2 logs.
"""
import sys, os, time, subprocess, re
from pathlib import Path

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, REPO)

UID         = 123
INTERVAL    = 300   # 5 minutes
PM2_LOG     = Path.home() / ".pm2/logs/poker44-miner-out.log"
MONITOR_LOG = Path(REPO) / "logs" / "monitor_uid123.log"

MONITOR_LOG.parent.mkdir(exist_ok=True)

def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

def log(msg, also_print=True):
    line = f"[{ts()}] {msg}"
    if also_print:
        print(line, flush=True)
    with open(MONITOR_LOG, "a") as f:
        f.write(line + "\n")

def check_metagraph():
    import bittensor as bt
    mg = bt.metagraph(126, network="finney")
    mg.sync()

    incentive = float(mg.I[UID])
    rank_val  = float(mg.R[UID])
    trust     = float(mg.T[UID])
    emission  = float(mg.E[UID])
    block     = int(mg.block.item())

    # Top-10 ranking
    inc_sorted = sorted([(float(mg.I[i]), i) for i in range(int(mg.n.item()))], reverse=True)
    rank_pos   = next((r+1 for r, (v, u) in enumerate(inc_sorted) if u == UID), 999)

    return {
        "incentive": incentive,
        "rank_val":  rank_val,
        "trust":     trust,
        "emission":  emission,
        "block":     block,
        "rank_pos":  rank_pos,
        "top10":     [(u, v) for v, u in inc_sorted[:10]],
    }

def count_scored_queries(since_lines=200):
    """Count how many 'Scored' lines appear in the last N lines of the PM2 log."""
    try:
        result = subprocess.run(
            ["tail", f"-{since_lines}", str(PM2_LOG)],
            capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.splitlines()
        scored = [l for l in lines if "Scored 100 chunks" in l]
        flagged_vals = []
        for l in scored:
            m = re.search(r"flagged=(\d+)", l)
            if m:
                flagged_vals.append(int(m.group(1)))
        return len(scored), flagged_vals
    except Exception:
        return 0, []

prev_incentive = None
prev_rank_pos  = None
checks = 0

log(f"=== Poker44 monitor started — watching UID {UID} ===")
log(f"    Log file: {MONITOR_LOG}")
log(f"    Checking every {INTERVAL//60} min")

while True:
    try:
        data = check_metagraph()
        n_scored, flagged = count_scored_queries()
        avg_flagged = sum(flagged)/len(flagged) if flagged else 0.0

        changed = (prev_incentive is not None and data["incentive"] != prev_incentive)
        rank_changed = (prev_rank_pos is not None and data["rank_pos"] != prev_rank_pos)

        status = "*** CHANGED ***" if (changed or rank_changed) else ""

        log(
            f"UID {UID} | incentive={data['incentive']:.6f} | "
            f"rank_pos={data['rank_pos']} | trust={data['trust']:.4f} | "
            f"block={data['block']} | "
            f"recent_queries={n_scored} avg_flagged={avg_flagged:.1f} {status}"
        )

        if data["incentive"] > 0 and (prev_incentive == 0 or prev_incentive is None):
            log(">>> FIRST NON-ZERO INCENTIVE — we are being scored! <<<")

        if rank_changed and data["rank_pos"] <= 10:
            log(f">>> ENTERED TOP 10 at rank {data['rank_pos']}! <<<")

        # Print top 10 every hour (every 12 checks)
        if checks % 12 == 0:
            log("  Current top 10:")
            for pos, (u, v) in enumerate(data["top10"], 1):
                marker = " ← US" if u == UID else ""
                log(f"    #{pos:2d}  UID {u:3d}: incentive={v:.6f}{marker}")

        prev_incentive = data["incentive"]
        prev_rank_pos  = data["rank_pos"]
        checks += 1

    except Exception as e:
        log(f"ERROR during check: {e}")

    time.sleep(INTERVAL)
