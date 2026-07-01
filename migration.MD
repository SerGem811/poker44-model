# Poker44 Miner — Project Handoff & Context

> Read this first if you are a new Claude session picking up this project. It explains
> what was built, why, the current state, the gotchas, and what to do next. Operational
> steps are at the very bottom (§10).

---

## 1. What this is

A **bot-detection miner for Bittensor subnet 126 (Poker44)**, running as **UID 162**
(wallet `zero` / hotkey `zero-1`, finney). Validators send batches of poker-hand
"chunks"; the miner returns **one bot-risk score per chunk** (0..1). Validators score
those risk_scores and the backend settles a competition leaderboard + on-chain weights.

Goal: maximize the competition **compositeScore** (→ rank → emission). Target: top 5,
ultimately rank 1.

- Repo (this code) published at `https://github.com/SerGem811/poker44-model`
- GitHub token in `/root/work/Poker44-subnet/.secrets` (gitignored; `source .secrets`)
- Upstream subnet code: `https://github.com/Poker44/Poker44-subnet`
- Live commit currently served: see `POKER44_MODEL_REPO_COMMIT` pm2 env var

---

## 2. Current state (one glance)

- **Model**: `models/poker44_gbdt.joblib` — **5-way ensemble** (LightGBM ×3 + ExtraTrees + RandomForest)
  + BlendedIsotonicCalibrator + monotonic scoring head.
  - Features: **145 scale-invariant behavioral features** (19 per-hand × 7 stats = 133 + 8 chunk signatures + 4 global rates)
  - AP: **0.8164** on benchmark holdout; live score spread **0.355–0.762** (not saturated)
  - The model file is git-ignored (weights stay local; only code is published)
- **Live window scores (epoch day_2026-06-27)**:
  R1=0.287 (old model, broken) → R2=0.448 → R3=0.480 → R4=0.0 (port blocked) → R5=0.429
  Composite ~0.329 → Rank ~111. Leaders at 0.55.
- **New epoch**: starts July 2 00:00 UTC (5-day epoch, 5 rounds of ~24h each).
  First full epoch with new 145-feat ensemble + port confirmed open. Expect 0.47–0.52.
- **Manifest**: `transparent`, `manifestReviewFailed=False` (originality OK).
- **Automation**: cron runs `auto_retrain.sh` nightly (01:00 UTC) and `monitor_round.sh`
  every 2h (logs to `logs/round_metrics.csv`).

---

## 3. The solution — files & what they do

Everything the miner adds lives OUTSIDE the stock subnet code. We never edit
validator/scoring/core — see §8.

### Our custom files (in OUR_FILES in sync_upstream.sh — never overwritten by upstream)

| File | Purpose |
|------|---------|
| `poker44/miner_model/features.py` | 145-feature extractor. Deterministic chunk→vector. Same path for train and live (zero skew). All features are **scale-invariant** (no absolute BB magnitudes). |
| `poker44/miner_model/ensemble.py` | `EnsembleClassifier` + `BlendedIsotonicCalibrator`. Lives in the package so joblib can unpickle from any working directory. |
| `poker44/miner_model/scoring_head.py` | Monotonic score shaper: positions the 0.5 cutoff at `t_star`. Under rank-first scoring this is mostly inert but harmless. |
| `neurons/miner_trained.py` | The miner. Loads model, scores chunks, builds manifest (`POKER44_MODEL_REPO_COMMIT` env, NOT git). Falls back to heuristic if no model. |
| `scripts/miner/train/build_dataset.py` | Pulls labeled chunks from public benchmark API → `.npz` via `extract_chunk_features`. |
| `scripts/miner/train/train_model.py` | Trains 5-way ensemble + BlendedIsotonicCalibrator. `train_ensemble(X_train, y_train [, X_val, y_val])` — val args optional (internal 80/20 split if omitted). |
| `scripts/miner/train/auto_retrain.py` | Nightly: rebuild data, train candidate, promote only if it beats incumbent on UNSEEN recent dates + margin. |
| `scripts/miner/train/auto_retrain.sh` | Cron wrapper (01:00 UTC). Acquires lock, runs auto_retrain.py. |
| `scripts/miner/monitor_round.py/.sh` | Appends rank/score to `logs/round_metrics.csv` every 2h. |
| `scripts/miner/deploy_robust.sh` | Atomic deploy: commit → push to GitHub → update POKER44_MODEL_REPO_COMMIT env → pm2 restart. Use whenever `features.py` OR the model changes in a way requiring a new commit. |
| `scripts/miner/sync_upstream.sh` | Safe upstream merge: saves our files, merges upstream/main, restores our files, pushes to origin. Run `source .secrets && bash scripts/miner/sync_upstream.sh`. |

### Stock upstream files (always take their version during sync)
`poker44/score/scoring.py`, `poker44/validator/forward.py`, `poker44/validator/runtime_provider.py`,
`neurons/validator.py`, `poker44/utils/config.py`, `poker44/__init__.py`, `requirements.txt`.

---

## 4. How validator scoring works (the optimization target)

Reward formula lives in `poker44/score/scoring.py`. **Upstream changed it on 2026-06-26:**

- **OLD** (before commit `24bebfd`): `reward = (0.65·AP + 0.35·recall)·(1-FPR)²`, hard
  **zero if FPR ≥ 0.10**. Threshold/flag-rate mattered; over-flagging humans → reward 0.
- **NEW** (current): `reward = 0.75·AP + 0.25·recall_at_fpr(max_fpr=0.05)`. **No FPR
  cliff. Rank-first.** Both terms depend only on **ranking** (do bots rank above humans?),
  NOT where 0.5 falls. Threshold/`t_star` tuning is now a near-no-op.

Scored over a rolling 20-sample window in `poker44/validator/forward.py`.

**Competition epochs**: 120h (5 days), 5 rounds of ~24h each. `winner_take_all` settlement.
Epoch anchor: June 27 00:00 UTC → next epoch starts July 2 00:00 UTC.

**Implication**: the dominant lever is **ranking quality on LIVE data** (AP on bots vs humans).
Threshold games are spent. Focus on feature quality and model AP.

---

## 5. Key design decisions & rationale

- **Scale-invariant features**: live data is distribution-shifted vs benchmark — absolute
  BB magnitude features saturate the model (~0.86–0.91 for everything). We use fractions,
  CVs, entropies, and bucket-based features ONLY. Score spread improved 0.07–0.73 on live
  data. **Single most impactful change.**
- **145 features** (19 per-hand × 7 stats + 8 chunk signatures + 4 global rates): captures
  per-hand behavioral variation that 26 aggregate features miss.
- **5-way ensemble** (lgbm1, lgbm2, lgbm3, ExtraTrees, RandomForest): diverse models
  reduce variance. AP improved 0.775 → 0.816.
- **BlendedIsotonicCalibrator** (blend=0.5): de-saturates scores while preserving ranking
  monotonicity. Calibrator fitted on val split; model stores it inside EnsembleClassifier.
- **Auto-retrain only promotes on UNSEEN-data improvement + margin**: prevents nightly
  retrain from regressing live model. Worst case = "no change".
- **Weights private, code public**: `.joblib` is git-ignored. Only `features.py`,
  `ensemble.py`, `scoring_head.py`, `miner_trained.py` are published (manifest covers them).

---

## 6. Hard-won lessons / gotchas (don't repeat these)

1. **bittensor version**: repo's unpinned `requirements.txt` pulls bittensor 10.x →
   `config.neuron is None` → miner crash-loops. Pin: `bittensor==9.12.2` +
   `async-substrate-interface==1.6.4`. See `requirements-migrate.txt`.
2. **NEVER change `features.py` without retraining + republishing together.** A 145-feat
   `features.py` with a 26-feat model crashes scoring. Always use `deploy_robust.sh` for
   code changes (atomic: commit+push+update env+restart).
3. **Eligibility**: must enter epoch from ~R1; late entry = `competition_late_entry_ineligible`
   (score 0 that epoch). Keep miner up. Port 8092 must be open in BOTH OS UFW AND cloud
   provider firewall.
4. **Manifest originality**: `POKER44_MODEL_REPO_COMMIT` must point to a commit unique to
   YOUR repo (`https://github.com/SerGem811/poker44-model`). Never copy another miner's
   commit — v2.1 rule penalizes the later duplicate.
5. **Model sync invariant**: `served code == published commit == deployed model features`.
   After any push, verify `Manifest status=transparent missing=[]` in pm2 logs.
6. **Upstream sync**: run `scripts/miner/sync_upstream.sh` to pull upstream changes WITHOUT
   losing custom files. Our files are listed in the `OUR_FILES` array in that script.
7. **Auto-retrain fairness**: if the incumbent was trained on recent dates that overlap with
   the auto_retrain holdout, the comparison is unfair (incumbent inflated score). Manual
   retrain with clean date splits fixes this when needed.
8. **git is NOT needed to run** (manifest reads env var). Only needed for `deploy_robust.sh`
   and `sync_upstream.sh`.

---

## 7. What to do next (open work, highest-leverage first)

1. **Phase 2 — ChunkSetTransformer sequence model** (target rank 1–3, composite ~0.54+):
   2-layer Transformer over action tokens per hand, 1-layer Transformer over hand embeddings,
   d_model=64, n_heads=4. Requires PyTorch. Architecture captures action ORDER, not just
   aggregates. Estimated +0.04–0.08 AP over ensemble baseline.
   - Build: `poker44/miner_model/transformer.py`
   - Train: extend `scripts/miner/train/train_model.py` to optionally use Transformer features
   - Must still be importable from the `poker44` package for joblib to unpickle.

2. **More training data freshness**: `auto_retrain.sh` pulls all dates nightly (DISCOVER_LIMIT=60).
   Weight recent dates higher (closer to live distribution). Currently using 37 dates
   (May 26 – July 1); older dates may have different bot mix than live.

3. **Live score monitoring**: after any model change, check live score SPREAD in pm2 logs:
   `Scored N chunks | flagged=X | score[min/max]=A/B`. Width `B-A` > 0.3 is healthy.
   Under 0.2 suggests saturation; check for scale-sensitive features.

4. **Watch leader repos**: top miners publish their code. Pull and study at their leaderboard
   `repoCommit` to understand feature/architecture advances.
   ```bash
   curl -sS 'https://api.poker44.net/api/v1/competition/leaderboard' \
    | python3 -c "import sys,json;[print(r['rank'],r['uid'],round(r.get('compositeScore') or 0,3),r['repoUrl'],r.get('repoCommit')) for r in sorted(json.load(sys.stdin)['data']['rows'],key=lambda x:-(x.get('compositeScore') or 0))[:8]]"
   ```

---

## 8. What was NOT touched (important)

We never modified validator/scoring/core subnet code. `poker44/score/scoring.py`,
`poker44/validator/*.py`, `neurons/validator.py`, `poker44/utils/config.py`,
`poker44/__init__.py` are stock upstream. If `git status` shows them "M", that's a
git-pointer artifact — content matches upstream; reset with `git checkout HEAD -- <file>`.

---

## 9. Operate it

```bash
# Live status
pm2 logs poker44_miner --lines 40 | grep -E "uid 162|Serving miner axon|Manifest status|Scored"

# Score history
column -s, -t logs/round_metrics.csv | tail -8

# Leaderboard + our rank
curl -sS 'https://api.poker44.net/api/v1/competition/leaderboard' \
 | python3 -c "import sys,json;r=[x for x in json.load(sys.stdin)['data']['rows'] if x['uid']==162][0];print('rank',r['rank'],'composite',r['compositeScore'],[w['compositeScore'] for w in r['windowCompositeScores']])"

# Current epoch timing
curl -sS 'https://api.poker44.net/api/v1/competition/current' | python3 -c "import sys,json,pprint;pprint.pprint(json.load(sys.stdin)['data'])"

# After any model change — verify manifest is transparent
pm2 logs poker44_miner --lines 10 | grep "Manifest status"
```

Key pm2 env (persisted via `pm2 save`):
- `POKER44_MODEL_PATH=models/poker44_gbdt.joblib`
- `POKER44_MODEL_REPO_URL=https://github.com/SerGem811/poker44-model`
- `POKER44_MODEL_REPO_COMMIT=<current HEAD of SerGem811/poker44-model>`

---

## 10. Upstream sync workflow

```bash
source .secrets
bash scripts/miner/sync_upstream.sh
```
This script: fetches upstream, shows diffs in scoring.py/forward.py, merges upstream/main,
RESTORES our custom files (features.py, ensemble.py, miner_trained.py, all scripts/miner/),
pushes merged result to SerGem811/poker44-model, restarts pm2.

After sync: **re-read scoring.py for reward formula changes**. If the formula changed
significantly, retrain the model (scoring head t_star might shift).

---

## 11. Migration steps (new box)

Path must stay `/root/work/Poker44-subnet`. Prereqs: `python3.10+`, `nodejs`+`pm2`,
`build-essential`, `git`. Bundle: `poker44_wallet.tar.gz` + repo + `setup_new_box.sh`.

```bash
chmod +x setup_new_box.sh
AXON_PORT=8092 ./setup_new_box.sh        # restores wallet+repo, pinned venv, starts pm2, cron
pm2 logs poker44_miner --lines 40 | grep -E "uid 162|Serving miner axon|Manifest status"
nc -vz NEW_BOX_IP 8092                   # must succeed; open cloud firewall if not
```

IMPORTANT: open port 8092 in BOTH `ufw` AND the cloud provider firewall (they're separate).
Never run two boxes with the same hotkey simultaneously.

Restore git after migration:
```bash
apt install -y git && cd /root/work/Poker44-subnet
git init -q && git remote add upstream https://github.com/Poker44/Poker44-subnet.git
git remote add origin https://github.com/SerGem811/poker44-model.git
git fetch -q origin main && git reset --mixed FETCH_HEAD
```

Common issues:
- `config.neuron is None` → wrong bittensor version (pin 9.12.2 + asi 1.6.4)
- model/feature mismatch → both must have same feature count (currently 145)
- no `Scored` lines → port 8092 blocked (check both ufw AND cloud provider)
- `Manifest status=missing_files` → miner can't reach GitHub to verify repo
