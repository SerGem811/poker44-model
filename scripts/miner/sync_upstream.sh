#!/bin/bash
# Sync with upstream Poker44/Poker44-subnet, preserving our custom model files.
#
# What this does:
#   1. Fetches latest from upstream (Poker44/Poker44-subnet)
#   2. Shows what changed in scoring/validator files (the ones we care about)
#   3. Merges upstream into our branch, then RESTORES our custom files
#   4. Pushes the merged result to SerGem811/poker44-model
#
# OUR files (never overwritten by upstream):
#   poker44/miner_model/features.py    — custom 145-feature extractor
#   poker44/miner_model/ensemble.py    — EnsembleClassifier (our file)
#   poker44/miner_model/scoring_head.py — our scoring head (our file)
#   neurons/miner_trained.py            — our miner implementation
#   scripts/miner/                      — all our scripts
#
# UPSTREAM files (always take their version):
#   poker44/score/scoring.py           — CRITICAL: reward formula changes
#   poker44/validator/forward.py       — scoring logic
#   poker44/validator/runtime_provider.py
#   neurons/validator.py
#   poker44/utils/config.py
#   poker44/__init__.py
#   requirements.txt
#   (everything else not in OUR FILES list)

set -euo pipefail

REPO="/root/work/Poker44-subnet"
cd "$REPO"

: "${GITHUB_TOKEN:?Set GITHUB_TOKEN or source .secrets first}"
GITHUB_USER="${GITHUB_USER:-SerGem811}"
GITHUB_REPO="${GITHUB_REPO:-poker44-model}"

# Files we OWN — never let upstream overwrite these
OUR_FILES=(
    "poker44/miner_model/features.py"
    "poker44/miner_model/ensemble.py"
    "poker44/miner_model/scoring_head.py"
    "neurons/miner_trained.py"
    "scripts/miner/train/train_model.py"
    "scripts/miner/train/auto_retrain.py"
    "scripts/miner/train/build_dataset.py"
    "scripts/miner/train/backtest.py"
    "scripts/miner/monitor_round.py"
    "scripts/miner/monitor_round.sh"
    "scripts/miner/deploy_robust.sh"
    "scripts/miner/sync_upstream.sh"
    "migration.MD"
    ".gitignore"
)

echo "== [1/5] fetching upstream (Poker44/Poker44-subnet) =="
git fetch upstream main -q
echo "   fetched upstream/main: $(git rev-parse upstream/main | head -c 12)"

echo ""
echo "== [2/5] what changed in key upstream files =="
SCORING_DIFF=$(git diff HEAD..upstream/main -- poker44/score/scoring.py 2>/dev/null | head -60)
FORWARD_DIFF=$(git diff HEAD..upstream/main -- poker44/validator/forward.py 2>/dev/null | head -40)
INIT_DIFF=$(git diff HEAD..upstream/main -- poker44/__init__.py 2>/dev/null | head -20)

if [ -z "$SCORING_DIFF" ]; then
    echo "   poker44/score/scoring.py   — no change"
else
    echo "   poker44/score/scoring.py   — CHANGED:"
    echo "$SCORING_DIFF" | head -30
fi

if [ -z "$FORWARD_DIFF" ]; then
    echo "   poker44/validator/forward.py — no change"
else
    echo "   poker44/validator/forward.py — CHANGED (first 10 lines):"
    echo "$FORWARD_DIFF" | head -10
fi

if [ -z "$INIT_DIFF" ]; then
    echo "   poker44/__init__.py          — no change"
else
    echo "   poker44/__init__.py          — CHANGED"
fi

NEW_COMMITS=$(git log HEAD..upstream/main --oneline 2>/dev/null)
if [ -z "$NEW_COMMITS" ]; then
    echo ""
    echo "✓ Already up to date with upstream. Nothing to merge."
    exit 0
fi

echo ""
echo "New upstream commits:"
echo "$NEW_COMMITS"
echo ""

echo "== [3/5] saving our custom file contents =="
declare -A SAVED
for f in "${OUR_FILES[@]}"; do
    if [ -f "$f" ]; then
        SAVED[$f]=$(cat "$f")
        echo "   saved: $f"
    fi
done

echo ""
echo "== [4/5] merging upstream/main =="
git merge upstream/main --no-edit -m "Sync with upstream Poker44/Poker44-subnet

$(echo "$NEW_COMMITS")

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>" || {
    echo "   merge had conflicts — resolving by keeping our files..."
    git checkout --theirs . 2>/dev/null || true
    git add -A
    git commit -m "Sync upstream (conflict resolved — kept our model files)"
}

echo "   merge done: $(git rev-parse HEAD | head -c 12)"

echo ""
echo "== [4b/5] restoring our custom files =="
for f in "${OUR_FILES[@]}"; do
    if [ -n "${SAVED[$f]+x}" ]; then
        echo "${SAVED[$f]}" > "$f"
        git add "$f"
        echo "   restored: $f"
    fi
done

# Check if restoration changed anything
if ! git diff --cached --quiet; then
    git commit -m "Restore custom model files after upstream sync

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
    echo "   committed restored files"
else
    echo "   no restoration needed (our files were unchanged by upstream)"
fi

echo ""
echo "== [5/5] pushing to SerGem811/poker44-model =="
HASH=$(git rev-parse HEAD)
if git push -q "https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${GITHUB_REPO}.git" HEAD:main; then
    REMOTE=$(git ls-remote "https://github.com/${GITHUB_USER}/${GITHUB_REPO}.git" main | cut -f1)
    [ "$REMOTE" = "$HASH" ] && echo "   pushed + verified: $HASH" || echo "   WARNING: remote hash mismatch"
else
    echo "   push FAILED — check GITHUB_TOKEN"
    exit 1
fi

# Update pm2 env so manifest stays correct
pm2 restart poker44_miner --update-env 2>/dev/null && echo "   pm2 restarted poker44_miner" || true
pm2 save 2>/dev/null || true

echo ""
echo "✓ Sync complete. Check poker44/score/scoring.py for reward formula changes"
echo "  and retrain if scoring logic changed significantly."
