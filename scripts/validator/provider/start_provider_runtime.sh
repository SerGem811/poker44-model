#!/bin/bash

set -euo pipefail

# Zero-touch bootstrap for validator-local provider runtime.
# Intended to let an operator update only poker44-subnet and restart the validator.

RUNTIME_ROOT="${POKER44_PROVIDER_RUNTIME_ROOT:-$(pwd)/.poker44-provider-runtime}"
BACKEND_DIR="${POKER44_PROVIDER_BACKEND_DIR:-$RUNTIME_ROOT/backend}"
FRONTEND_DIR="${POKER44_PROVIDER_FRONTEND_DIR:-$RUNTIME_ROOT/frontend}"
BACKEND_REPO_URL="${POKER44_PROVIDER_BACKEND_REPO_URL:-https://github.com/Poker44/poker44-platform-backend.git}"
FRONTEND_REPO_URL="${POKER44_PROVIDER_FRONTEND_REPO_URL:-https://github.com/Poker44/poker44-platform-frontend.git}"
RUNTIME_BRANCH="${POKER44_PROVIDER_RUNTIME_BRANCH:-dev}"
GIT_PULL="${POKER44_PROVIDER_GIT_PULL:-true}"

BACKEND_PM2_NAME="${POKER44_PROVIDER_BACKEND_PM2_NAME:-p44_provider_backend}"
FRONTEND_PM2_NAME="${POKER44_PROVIDER_FRONTEND_PM2_NAME:-p44_provider_frontend}"
BACKEND_DOCKER_UP="${POKER44_PROVIDER_BACKEND_DOCKER_UP:-false}"
RUN_BACKEND_MIGRATIONS="${POKER44_PROVIDER_RUN_MIGRATIONS:-true}"
SYNC_NEXT_STATIC="${POKER44_PROVIDER_SYNC_NEXT_STATIC:-false}"
NEXT_STATIC_TARGET="${POKER44_PROVIDER_NEXT_STATIC_TARGET:-}"
SKIP_FRONTEND="${POKER44_PROVIDER_SKIP_FRONTEND:-false}"

BACKEND_PORT="${POKER44_PROVIDER_BACKEND_PORT:-4001}"
FRONTEND_PORT="${POKER44_PROVIDER_FRONTEND_PORT:-4000}"
DATABASE_URL="${POKER44_PROVIDER_DATABASE_URL:-postgresql://aceguard:aceguard_local_pwd@localhost:5433/aceguard_poker}"
REDIS_URL="${POKER44_PROVIDER_REDIS_URL:-redis://localhost:6379}"
INTERNAL_EVAL_SECRET="${POKER44_PROVIDER_INTERNAL_SECRET:-force-start-secret}"
EVAL_COORDINATOR_BASE_URL="${POKER44_EVAL_COORDINATOR_BASE_URL:-http://185.196.20.208:4010}"
PROVIDER_PUBLIC_HOST="${POKER44_PROVIDER_PUBLIC_HOST:-}"
PROVIDER_PUBLIC_BASE_URL="${POKER44_PROVIDER_PUBLIC_BASE_URL:-}"
PROVIDER_VALIDATOR_ID="${POKER44_PROVIDER_VALIDATOR_ID:-}"
PROVIDER_FIXED_ROOM_CODE="${POKER44_PROVIDER_FIXED_ROOM_CODE:-}"
PROVIDER_JWT_SECRET="${POKER44_PROVIDER_JWT_SECRET:-}"
MIN_EVAL_HANDS="${POKER44_PROVIDER_MIN_EVAL_HANDS:-70}"
MAX_EVAL_HANDS="${POKER44_PROVIDER_MAX_EVAL_HANDS:-120}"

log() {
  echo "[provider-bootstrap] $*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

is_true() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

ensure_repo() {
  local dir="$1"
  local repo="$2"
  local branch="$3"

  mkdir -p "$(dirname "$dir")"
  if [ ! -d "$dir/.git" ]; then
    if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then
      if is_true "$GIT_PULL"; then
        echo "Repository directory $dir exists but is not a git checkout. Clear it or set POKER44_PROVIDER_GIT_PULL=false." >&2
        exit 1
      fi
      log "Using pre-populated runtime directory without git metadata: $dir"
      return
    fi
    log "Cloning $repo into $dir"
    git clone "$repo" "$dir"
  fi

  if is_true "$GIT_PULL"; then
    log "Updating repo in $dir to branch $branch"
    git -C "$dir" fetch origin
    if git -C "$dir" show-ref --verify --quiet "refs/heads/$branch"; then
      git -C "$dir" checkout "$branch"
    else
      git -C "$dir" checkout -b "$branch" "origin/$branch"
    fi
    git -C "$dir" pull --ff-only origin "$branch"
  fi
}

derive_public_host() {
  if [ -n "$PROVIDER_PUBLIC_HOST" ]; then
    printf '%s' "$PROVIDER_PUBLIC_HOST"
    return
  fi

  local from_route
  from_route="$(ip route get 1.1.1.1 2>/dev/null | awk '/src/ {for (i=1; i<=NF; i++) if ($i == "src") {print $(i+1); exit}}')"
  if [ -n "$from_route" ]; then
    printf '%s' "$from_route"
    return
  fi

  local from_host
  from_host="$(hostname -I 2>/dev/null | awk '{print $1}')"
  if [ -n "$from_host" ]; then
    printf '%s' "$from_host"
    return
  fi

  echo "Could not determine provider public host. Set POKER44_PROVIDER_PUBLIC_HOST." >&2
  exit 1
}

hash_value() {
  local value="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    printf '%s' "$value" | sha256sum | awk '{print $1}'
    return
  fi
  printf '%s' "$value" | shasum -a 256 | awk '{print $1}'
}

derive_fixed_room_code() {
  local seed="$1"
  local digest
  digest="$(hash_value "$seed")"
  printf '%s' "${digest^^}" | tr -dc 'A-Z0-9' | cut -c1-6
}

derive_jwt_secret() {
  local seed="$1"
  local digest
  digest="$(hash_value "$seed")"
  printf 'p44-%s-%s' "$seed" "$digest"
}

upsert_env_line() {
  local file="$1"
  local key="$2"
  local value="$3"
  python3 - "$file" "$key" "$value" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]
line = f"{key}={value}"
if path.exists():
    lines = path.read_text().splitlines()
else:
    lines = []
updated = False
out = []
for existing in lines:
    if existing.startswith(f"{key}="):
        out.append(line)
        updated = True
    else:
        out.append(existing)
if not updated:
    out.append(line)
path.write_text("\n".join(out).rstrip("\n") + "\n")
PY
}

pm2_start_or_restart_backend() {
  local name="$1"
  local cwd="$2"
  log "Ensuring PM2 backend process: $name"
  if pm2 describe "$name" >/dev/null 2>&1; then
    PORT="$BACKEND_PORT" pm2 restart "$name" --update-env
  else
    PORT="$BACKEND_PORT" pm2 start npm --name "$name" --cwd "$cwd" -- start
  fi
}

pm2_start_or_restart_frontend() {
  local name="$1"
  local cwd="$2"
  log "Ensuring PM2 frontend process: $name"
  if pm2 describe "$name" >/dev/null 2>&1; then
    PORT="$FRONTEND_PORT" pm2 restart "$name" --update-env
  else
    PORT="$FRONTEND_PORT" pm2 start npm --name "$name" --cwd "$cwd" -- start -- -p "$FRONTEND_PORT" -H 0.0.0.0
  fi
}

require_cmd git
require_cmd npm
require_cmd pm2
require_cmd python3

mkdir -p "$RUNTIME_ROOT"

PUBLIC_HOST="$(derive_public_host)"
if [ -n "$PROVIDER_PUBLIC_BASE_URL" ]; then
  FRONTEND_PUBLIC_BASE_URL="${PROVIDER_PUBLIC_BASE_URL%/}"
else
  FRONTEND_PUBLIC_BASE_URL="http://$PUBLIC_HOST:$FRONTEND_PORT"
fi
BACKEND_PUBLIC_BASE_URL="http://$PUBLIC_HOST:$BACKEND_PORT"

if [ -z "$PROVIDER_VALIDATOR_ID" ]; then
  PROVIDER_VALIDATOR_ID="$PUBLIC_HOST"
fi
if [ -z "$PROVIDER_FIXED_ROOM_CODE" ]; then
  PROVIDER_FIXED_ROOM_CODE="$(derive_fixed_room_code "$PROVIDER_VALIDATOR_ID")"
fi
if [ -z "$PROVIDER_JWT_SECRET" ]; then
  PROVIDER_JWT_SECRET="$(derive_jwt_secret "$PROVIDER_VALIDATOR_ID")"
fi

ensure_repo "$BACKEND_DIR" "$BACKEND_REPO_URL" "$RUNTIME_BRANCH"
if ! is_true "$SKIP_FRONTEND"; then
  ensure_repo "$FRONTEND_DIR" "$FRONTEND_REPO_URL" "$RUNTIME_BRANCH"
fi

log "Writing provider backend env"
upsert_env_line "$BACKEND_DIR/.env" "NODE_ENV" "development"
upsert_env_line "$BACKEND_DIR/.env" "PORT" "$BACKEND_PORT"
upsert_env_line "$BACKEND_DIR/.env" "DATABASE_URL" "$DATABASE_URL"
upsert_env_line "$BACKEND_DIR/.env" "REDIS_URL" "$REDIS_URL"
upsert_env_line "$BACKEND_DIR/.env" "JWT_SECRET" "$PROVIDER_JWT_SECRET"
upsert_env_line "$BACKEND_DIR/.env" "JWT_EXPIRES_IN" "7d"
upsert_env_line "$BACKEND_DIR/.env" "COOKIE_MAX_AGE" "604800000"
upsert_env_line "$BACKEND_DIR/.env" "CORS_ORIGINS" "$FRONTEND_PUBLIC_BASE_URL,$BACKEND_PUBLIC_BASE_URL,http://localhost:$FRONTEND_PORT,http://127.0.0.1:$FRONTEND_PORT"
upsert_env_line "$BACKEND_DIR/.env" "LOG_TO_FILE" "false"
upsert_env_line "$BACKEND_DIR/.env" "PROVIDER_PLATFORM_URL" "$FRONTEND_PUBLIC_BASE_URL"
upsert_env_line "$BACKEND_DIR/.env" "POKER44_VALIDATOR_ID" "$PROVIDER_VALIDATOR_ID"
upsert_env_line "$BACKEND_DIR/.env" "P2P_FIXED_ROOM_CODE" "$PROVIDER_FIXED_ROOM_CODE"
upsert_env_line "$BACKEND_DIR/.env" "INTERNAL_EVAL_SECRET" "$INTERNAL_EVAL_SECRET"
upsert_env_line "$BACKEND_DIR/.env" "EVAL_COORDINATOR_BASE_URL" "$EVAL_COORDINATOR_BASE_URL"
upsert_env_line "$BACKEND_DIR/.env" "POKER44_PROVIDER_MIN_EVAL_HANDS" "$MIN_EVAL_HANDS"
upsert_env_line "$BACKEND_DIR/.env" "POKER44_PROVIDER_MAX_EVAL_HANDS" "$MAX_EVAL_HANDS"

if ! is_true "$SKIP_FRONTEND"; then
  log "Writing provider frontend env"
  upsert_env_line "$FRONTEND_DIR/.env.local" "NEXT_PUBLIC_API_URL" "$BACKEND_PUBLIC_BASE_URL/api/v1"
  upsert_env_line "$FRONTEND_DIR/.env.local" "NEXT_PUBLIC_WS_URL" "$BACKEND_PUBLIC_BASE_URL"
  upsert_env_line "$FRONTEND_DIR/.env.local" "NEXT_PUBLIC_DIRECTORY_URL" "$BACKEND_PUBLIC_BASE_URL"
fi

log "Bootstrapping provider backend in $BACKEND_DIR"
cd "$BACKEND_DIR"
HUSKY=0 npm install
if is_true "$BACKEND_DOCKER_UP" && [ -f "docker-compose.yml" ]; then
  log "Starting provider backend docker dependencies"
  npm run docker:up
fi
if is_true "$RUN_BACKEND_MIGRATIONS"; then
  log "Running provider backend migrations"
  npm run migration:run:dev
fi
log "Building provider backend"
npm run build
pm2_start_or_restart_backend "$BACKEND_PM2_NAME" "$BACKEND_DIR"

if ! is_true "$SKIP_FRONTEND"; then
  log "Bootstrapping provider frontend in $FRONTEND_DIR"
  cd "$FRONTEND_DIR"
  HUSKY=0 npm install
  log "Building provider frontend"
  npm run build
  pm2_start_or_restart_frontend "$FRONTEND_PM2_NAME" "$FRONTEND_DIR"

  if is_true "$SYNC_NEXT_STATIC"; then
    if [ -z "$NEXT_STATIC_TARGET" ]; then
      echo "POKER44_PROVIDER_NEXT_STATIC_TARGET is required when POKER44_PROVIDER_SYNC_NEXT_STATIC=true" >&2
      exit 1
    fi
    require_cmd rsync
    if [ -d ".next/static" ]; then
      log "Syncing Next static assets to $NEXT_STATIC_TARGET"
      mkdir -p "$NEXT_STATIC_TARGET"
      rsync -a ".next/static/" "$NEXT_STATIC_TARGET/"
    fi
  fi
fi

pm2 save

log "Provider runtime bootstrap complete"
log "Backend dir: $BACKEND_DIR"
log "Frontend dir: ${FRONTEND_DIR:-<skipped>}"
log "Validator/provider id: $PROVIDER_VALIDATOR_ID"
log "Room code: $PROVIDER_FIXED_ROOM_CODE"
log "Backend URL: $BACKEND_PUBLIC_BASE_URL"
log "Frontend URL: $FRONTEND_PUBLIC_BASE_URL"
log "Coordinator URL: $EVAL_COORDINATOR_BASE_URL"
