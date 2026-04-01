# Poker44 Validator Guide

Validator guide for Poker44 subnet `126`.

## Current Model

The validator now has one intended operating model:

- the validator runs in `provider_runtime`
- the validator bootstrap manages a local provider runtime on the same server
- the local provider runtime creates or ensures a public table
- hands are stored in the validator's own PostgreSQL
- the local provider backend publishes chunk candidates to a shared eval coordinator
- all validators consume the same canonical chunk from that coordinator
- validators then evaluate miners, compute rewards, and set weights on-chain

The old `mixed_dataset` mode still exists in code for compatibility, but it is no longer the target operating path.

## Pull + Restart Contract

This is the operational promise of the new validator flow.

When a validator operator does only:

1. `git pull`
2. restart the validator process

the system should do the rest automatically.

Concretely, `pull + restart` means:

- the validator starts in `provider_runtime`
- the bootstrap clones or updates `poker44-platform-backend` and `poker44-platform-frontend` under `.poker44-provider-runtime/`
- it writes the provider `.env` files automatically
- it starts provider dependencies if needed
- it runs backend migrations
- it starts or restarts the provider backend and frontend under PM2
- it derives a stable validator/provider id and fixed room code if they were not provided
- it refuses legacy public setups that expose the provider through raw IP / plain HTTP unless explicitly overridden
- it aligns provider cookies and CORS for `*.poker44.net` hosts automatically
- it opens `80/tcp` and `443/tcp` in `ufw` automatically when available
- it can bootstrap the public edge itself once DNS already points to the validator:
  - `nginx` site
  - `certbot` TLS
  - reload + health path
- it ensures a provider table exists
- that table becomes visible through the public directory
- hands generated on that table are persisted in the validator's own PostgreSQL
- when the provider has enough usable hands, it builds a sanitized chunk candidate
- that candidate is published to the shared eval coordinator
- if a canonical chunk for the current 2h window already exists, the provider uses that one instead
- the validator consumes the coordinator's active chunk
- the validator sends that same chunk to miners
- the validator computes rewards and sets weights

Important:

- every validator keeps its own SQL locally
- every validator/provider backend must point to the same eval coordinator
- all validators must share the same `INTERNAL_EVAL_SECRET` expected by that coordinator

## Requirements

- Linux server
- Python 3.10+
- Node.js and npm
- PM2
- Docker and Docker Compose for bundled provider Postgres/Redis, unless the operator uses their own services
- registered validator hotkey on netuid `126`
- root privileges if you want the bootstrap to automate `ufw`, `nginx`, and `certbot`

## Install

```bash
git clone https://github.com/Poker44/Poker44-subnet
cd Poker44-subnet
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install bittensor-cli
```

Or use:

```bash
./scripts/validator/main/setup.sh
```

## Registration

```bash
btcli subnet register \
  --wallet.name p44_cold \
  --wallet.hotkey p44_validator \
  --netuid 126 \
  --subtensor.network finney

btcli wallet overview --wallet.name p44_cold --subtensor.network finney
```

## Required Environment

Mandatory:

- `POKER44_RUNTIME_MODE=provider_runtime`
- `WALLET_NAME`
- `HOTKEY`

Provider runtime defaults are already wired for the intended flow.

Important defaults:

- `POKER44_PROVIDER_API_BASE_URL=http://127.0.0.1:4001`
- `POKER44_PROVIDER_INTERNAL_SECRET=force-start-secret`
- `POKER44_PROVIDER_MIN_EVAL_HANDS=70`
- `POKER44_PROVIDER_MAX_EVAL_HANDS=120`
- `POKER44_PROVIDER_ATTEMPT_PUBLISH_CURRENT=true`
- `POKER44_EVAL_COORDINATOR_BASE_URL=http://185.196.20.208:4010`
- `POKER44_PROVIDER_RUNTIME_BRANCH=dev`
- `POKER44_PROVIDER_BACKEND_DOCKER_UP=true`
- `POKER44_PROVIDER_RUN_MIGRATIONS=true`
- `POKER44_PROVIDER_CENTRAL_AUTH_ORIGIN=https://dev.poker44.net`
- `POKER44_PROVIDER_UFW_MANAGE=true`
- `POKER44_PROVIDER_PUBLIC_EDGE_ENABLE=true`

The validator itself does not talk directly to the global coordinator.

The flow is:

- validator -> local provider backend
- local provider backend -> shared eval coordinator

## Shared Coordinator Rule

If you want all validators to evaluate miners with the same chunk, all provider backends must use the same:

- `EVAL_COORDINATOR_BASE_URL`
- `INTERNAL_EVAL_SECRET`

Today, the intended coordinator is:

- `http://185.196.20.208:4010`

That is the single source of truth for the active chunk of each 2-hour window.

## What Gets Automated

The validator bootstrap can manage these values automatically if they are not provided:

- `POKER44_PROVIDER_RUNTIME_ROOT`
- `POKER44_PROVIDER_BACKEND_DIR`
- `POKER44_PROVIDER_FRONTEND_DIR`
- `POKER44_PROVIDER_BACKEND_REPO_URL`
- `POKER44_PROVIDER_FRONTEND_REPO_URL`
- `POKER44_PROVIDER_RUNTIME_BRANCH`
- `POKER44_PROVIDER_PUBLIC_HOST`
- `POKER44_PROVIDER_PUBLIC_BASE_URL` for the browser-facing `https` host
- `POKER44_PROVIDER_PUBLIC_API_BASE_URL` when API/WS are exposed on a different public origin
- `POKER44_PROVIDER_BACKEND_PORT`
- `POKER44_PROVIDER_FRONTEND_PORT`
- `POKER44_PROVIDER_DATABASE_URL`
- `POKER44_PROVIDER_REDIS_URL`
- `POKER44_PROVIDER_JWT_SECRET`
- `POKER44_PROVIDER_COOKIE_DOMAIN`
- `POKER44_PROVIDER_PUBLIC_EDGE_EMAIL`
- `POKER44_PROVIDER_FIXED_ROOM_CODE`

Useful overrides:

- `POKER44_PROVIDER_VALIDATOR_ID`
- `POKER44_PROVIDER_PUBLIC_HOST`
- `POKER44_PROVIDER_PUBLIC_BASE_URL`
- `POKER44_PROVIDER_PUBLIC_API_BASE_URL`
- `POKER44_PROVIDER_BACKEND_PORT`
- `POKER44_PROVIDER_FRONTEND_PORT`
- `POKER44_PROVIDER_DATABASE_URL`
- `POKER44_PROVIDER_REDIS_URL`
- `POKER44_PROVIDER_SHARED_JWT_SECRET`
- `POKER44_PROVIDER_COOKIE_DOMAIN`
- `POKER44_PROVIDER_CENTRAL_AUTH_ORIGIN`
- `POKER44_PROVIDER_EXTRA_CORS_ORIGINS`
- `POKER44_PROVIDER_UFW_MANAGE`
- `POKER44_PROVIDER_ALLOW_INSECURE_PUBLIC_BASE_URL`
- `POKER44_PROVIDER_PUBLIC_EDGE_ENABLE`
- `POKER44_PROVIDER_PUBLIC_EDGE_EMAIL`
- `POKER44_PROVIDER_PUBLIC_EDGE_CERTBOT`
- `POKER44_PROVIDER_PUBLIC_EDGE_NGINX_DIR`
- `POKER44_PROVIDER_PUBLIC_EDGE_NGINX_ENABLED_DIR`
- `POKER44_PROVIDER_SKIP_FRONTEND`
- `POKER44_PROVIDER_GIT_PULL`
- `POKER44_PROVIDER_BACKEND_PM2_NAME`
- `POKER44_PROVIDER_FRONTEND_PM2_NAME`
- `POKER44_PROVIDER_BOOTSTRAP_RETRY_SECONDS`
- `POKER44_PROVIDER_BOOTSTRAP_MAX_ATTEMPTS`
- `POKER44_PROVIDER_HEALTH_TIMEOUT_SECONDS`
- `POKER44_PROVIDER_HEALTH_POLL_INTERVAL_SECONDS`
- `POKER44_PROVIDER_ENSURE_ROOM_INTERVAL_SECONDS`
- `POKER44_PROVIDER_REQUEST_TIMEOUT_SECONDS`

Subnet-side evaluation tuning still applies:

- `POKER44_CHUNK_COUNT`
- `POKER44_REWARD_WINDOW`
- `POKER44_POLL_INTERVAL_SECONDS`
- `POKER44_MINERS_PER_CYCLE`
- `POKER44_TARGET_MINER_UIDS`
- `--neuron.timeout`

## Run Validator

Preferred command:

```bash
WALLET_NAME=p44_cold \
HOTKEY=p44_validator \
POKER44_RUNTIME_MODE=provider_runtime \
./scripts/validator/run/run_vali.sh
```

Recommended public-provider env for dev:

```bash
POKER44_PROVIDER_PUBLIC_BASE_URL=https://provider-<validator>.dev.poker44.net
POKER44_PROVIDER_SHARED_JWT_SECRET='<same JWT secret as central dev backend>'
POKER44_PROVIDER_COOKIE_DOMAIN=.poker44.net
POKER44_PROVIDER_CENTRAL_AUTH_ORIGIN=https://dev.poker44.net
POKER44_PROVIDER_PUBLIC_EDGE_EMAIL=ops@poker44.net
```

If the operator does not set a proper `https` hostname, the bootstrap now fails fast instead of silently deploying a provider that will later timeout on `Join`.

## DNS Prerequisite

The bootstrap can automate:

- `ufw`
- `nginx`
- `certbot`
- provider backend/frontend process boot

but it still cannot invent public DNS by itself.

Before restart, the operator must ensure that the chosen public host already resolves to the validator server IP. Once that is true, the bootstrap can finish the rest automatically.

If the validator process does not run with sufficient privileges, set:

```bash
POKER44_PROVIDER_PUBLIC_EDGE_ENABLE=false
```

and manage `ufw` / `nginx` / `certbot` outside the bootstrap.

Script path:

- `scripts/validator/run/run_vali.sh`

The script exports the provider-runtime defaults, bootstraps the local provider runtime if needed, and starts the validator under PM2.

## PM2

```bash
pm2 logs poker44_validator
pm2 restart poker44_validator
pm2 stop poker44_validator
pm2 delete poker44_validator
```

Typical provider PM2 names managed by the bootstrap:

- `p44_provider_backend`
- `p44_provider_frontend`

## Canonical Chunk Behavior

The intended chunk lifecycle is:

- hands are stored locally in the validator's PostgreSQL
- raw hands stay in the provider/backend side
- the validator does not need direct SQL access to raw hands
- when the provider has between `70` and `120` usable hands, it can build a sanitized chunk candidate
- chunks are built from hands in natural order, not reordered
- only one canonical chunk is active per 2-hour window
- all validators consume that same chunk from the shared coordinator
- used chunks are tracked in coordinator-side tables such as `eval_used_chunks`

## Auto-Update

Poker44 supports optional validator auto-update through a separate PM2 watcher.

Files:

- `scripts/validator/update/auto_update_validator.sh`
- `scripts/validator/update/update_validator.sh`
- `scripts/validator/update/update_full.sh`

Start the watcher:

```bash
chmod +x scripts/validator/update/auto_update_validator.sh
pm2 start --name poker44_auto_update \
  --interpreter /bin/bash \
  scripts/validator/update/auto_update_validator.sh
pm2 save
```

## Related Docs

- [VALIDATOR_PROVIDER_SETUP.md](/Users/mac/poker44-launch/documentacion/operaciones/VALIDATOR_PROVIDER_SETUP.md)
- [ENV_MATRIX.md](/Users/mac/poker44-launch/documentacion/operaciones/ENV_MATRIX.md)
- [RUNBOOK.md](/Users/mac/poker44-launch/documentacion/operaciones/RUNBOOK.md)
