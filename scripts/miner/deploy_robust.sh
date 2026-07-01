#!/bin/bash
# Atomic robust-feature deploy for the Poker44 miner.
#
# Does, in ONE shot, with rollback on any failure so served code == published code:
#   1. apply robust features.py (drop drift-prone absolute BB magnitudes)
#   2. rebuild dataset + retrain the matching model (conservative t_star)
#   3. commit + push features.py to YOUR github repo
#   4. deploy with the NEW commit hash and restart the miner
#   5. verify features.py == model == pushed commit (all 26 features)
#
# Usage:
#   GITHUB_TOKEN=ghp_xxx ./scripts/miner/deploy_robust.sh
# Optional overrides:
#   GITHUB_USER=SerGem811  GITHUB_REPO=poker44-model  T_STAR=0.91
set -euo pipefail

REPO="/root/work/Poker44-subnet"
cd "$REPO"
: "${GITHUB_TOKEN:?Set GITHUB_TOKEN=your_github_pat (repo scope)}"
GITHUB_USER="${GITHUB_USER:-SerGem811}"
GITHUB_REPO="${GITHUB_REPO:-poker44-model}"
T_STAR="${T_STAR:-0.91}"
PM2_NAME="${PM2_NAME:-poker44_miner}"
FEAT="poker44/miner_model/features.py"
MODEL="models/poker44_gbdt.joblib"

# shellcheck disable=SC1091
source "$REPO/.venv/bin/activate"
export PYTHONPATH="$REPO"

echo "== backing up current state for rollback =="
cp "$FEAT"  /tmp/feat.bak
cp "$MODEL" /tmp/model.bak
git config user.email >/dev/null 2>&1 || git config user.email "miner@poker44.local"
git config user.name  >/dev/null 2>&1 || git config user.name  "$GITHUB_USER"

restore_local() {
  echo "!! ROLLBACK: restoring 28-feature model + features.py and restarting"
  cp /tmp/feat.bak "$FEAT"
  cp /tmp/model.bak "$MODEL"
  pm2 restart "$PM2_NAME" --update-env >/dev/null 2>&1 || true
}

# ---- STAGE 1: code + model (local only) -----------------------------------
echo "== [1/4] applying robust features (drop absolute BB magnitudes) =="
python3 - <<'PY'
p="poker44/miner_model/features.py"
s=open(p).read()
for tok in ('    "size_mean_bb",\n', '    "size_std_bb",\n',
            '        size_mean,\n', '        size_std,\n'):
    if tok not in s:
        raise SystemExit(f"expected token not found, aborting: {tok!r}")
    s=s.replace(tok, "")
open(p,"w").write(s)
PY
if ! python3 -c "from poker44.miner_model.features import FEATURE_NAMES as f; assert len(f)==26, len(f)"; then
  restore_local; echo "features.py did not reduce to 26 cleanly"; exit 1
fi
echo "   features.py -> 26 features (self-check passed)"

echo "== [2/4] rebuild dataset + retrain matching model (t_star=$T_STAR) =="
if ! python3 scripts/miner/train/build_dataset.py --out data/train.npz --discover-limit 60 --limit 100 \
   || ! python3 scripts/miner/train/train_model.py --data data/train.npz --out "$MODEL" --fixed-t-star "$T_STAR"; then
  restore_local; echo "train failed"; exit 1
fi
if ! python3 -c "import joblib; n=joblib.load('$MODEL')['estimator'].n_features_in_; assert n==26, n"; then
  restore_local; echo "model is not 26 features"; exit 1
fi
echo "   model -> 26 features (matches features.py)"

# ---- STAGE 2: commit + push (the part that needs your token) ---------------
echo "== [3/4] commit + push features.py to github.com/$GITHUB_USER/$GITHUB_REPO =="
START_HASH="$(git rev-parse HEAD)"
git add "$FEAT"
git commit -q -m "Robust features: drop drift-prone absolute BB magnitudes for live generalization"
HASH="$(git rev-parse HEAD)"
if ! git push -q "https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${GITHUB_REPO}.git" HEAD:main; then
  echo "!! push failed (bad/expired token?) — undoing commit and rolling back"
  git reset -q --soft "$START_HASH"
  restore_local
  exit 1
fi
# confirm the commit is actually public before we point the manifest at it
if [ "$(git ls-remote "https://github.com/${GITHUB_USER}/${GITHUB_REPO}.git" main | cut -f1)" != "$HASH" ]; then
  echo "!! remote main != pushed hash; rolling back"; git reset -q --soft "$START_HASH"; restore_local; exit 1
fi
echo "   pushed + verified public: $HASH"

# ---- STAGE 3: deploy with the new (matching) commit -----------------------
echo "== [4/4] deploy with new commit + restart =="
export POKER44_MODEL_PATH="$MODEL"
export POKER44_MODEL_REPO_URL="https://github.com/${GITHUB_USER}/${GITHUB_REPO}"
export POKER44_MODEL_REPO_COMMIT="$HASH"
pm2 restart "$PM2_NAME" --update-env >/dev/null 2>&1
pm2 save >/dev/null 2>&1

F=$(python3 -c "from poker44.miner_model.features import FEATURE_NAMES as f; print(len(f))")
M=$(python3 -c "import joblib; print(joblib.load('$MODEL')['estimator'].n_features_in_)")
echo
echo "✅ DONE — served == published"
echo "   features.py=$F  model=$M  pushed_commit=$HASH  t_star=$T_STAR"
echo "   set POKER44_MODEL_REPO_COMMIT=$HASH is now live in pm2 (saved)"
echo "   watch: pm2 logs $PM2_NAME | grep Scored   (expect a conservative flag rate)"
