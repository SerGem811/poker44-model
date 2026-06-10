# Poker44 Training Benchmark

Public benchmark guide for Poker44 subnet `126`.

## Purpose

Poker44 provides a public training benchmark for miner development. Use it to:

- test your benchmark parser;
- build and validate feature pipelines;
- train and compare detection models;
- run regression tests across model versions;
- calibrate model outputs against labeled chunk data.

The benchmark is a development dataset. Treat live evaluation as a separate
competition surface.

## API Base

```text
https://api.poker44.net/api/v1/benchmark
```

## Endpoints

```text
GET /api/v1/benchmark
GET /api/v1/benchmark/releases
GET /api/v1/benchmark/chunks?sourceDate=YYYY-MM-DD
GET /api/v1/benchmark/chunks/:chunkId
```

## Status

`GET /api/v1/benchmark` returns aggregate availability:

- `releaseVersion`
- `schemaVersion`
- `releaseType`
- `totalChunks`
- `totalHands`
- `latestSourceDate`
- `latestReleasedAt`
- `currentUtcDate`
- `autoRelease`

Example:

```bash
curl -sS https://api.poker44.net/api/v1/benchmark
```

## Releases

`GET /api/v1/benchmark/releases` returns available benchmark dates.

Common query parameters:

- `limit`: number of releases to return.
- `before`: optional `YYYY-MM-DD` cursor for pagination.

Example:

```bash
curl -sS 'https://api.poker44.net/api/v1/benchmark/releases?limit=30'
```

Each release includes:

- `sourceDate`
- `releaseVersion`
- `schemaVersion`
- `chunkCount`
- `handCount`
- `releasedAt`
- `humanExampleCount`
- `syntheticBotExampleCount`
- `audit`
- `metadata`

## Chunks

`GET /api/v1/benchmark/chunks?sourceDate=YYYY-MM-DD` returns chunk payloads for
one release date.

Common query parameters:

- `sourceDate`: required release date in `YYYY-MM-DD` format.
- `limit`: number of chunks to return.
- `cursor`: optional pagination cursor.
- `split`: optional `train` or `validation`.

Example:

```bash
curl -sS 'https://api.poker44.net/api/v1/benchmark/chunks?sourceDate=2026-06-10&limit=24'
```

Each chunk includes:

- `chunkId`
- `chunkHash`
- `sourceDate`
- `releaseVersion`
- `split`
- `handCount`
- `batchCount`
- `chunks`
- `groundTruth`
- `groundTruthLabels`
- `metadata`

## Model Input

The `chunks` field is the miner-visible model input. It is a list of chunk
groups. Each group contains one or more poker hands.

Miners should produce one prediction per chunk group, matching the order of
`chunks`.

The labels are returned separately:

- `groundTruth`: numeric labels, where `1` means bot and `0` means human.
- `groundTruthLabels`: string labels, `bot` or `human`.

Do not read labels from individual hand objects.

## Hand Fields

Hands may include:

- `hand_id`
- `metadata`
- `players`
- `streets`
- `actions`
- `outcome`

Action records may include:

- `action_id`
- `street`
- `actor_seat`
- `action_type`
- `amount`
- `raise_to`
- `call_to`
- `normalized_amount_bb`
- `pot_before`
- `pot_after`

Code should tolerate missing optional fields and empty arrays.

## Recommended Workflow

1. Fetch release dates from `/releases`.
2. Download chunks by `sourceDate`.
3. Split by the returned `split` field when present.
4. Train on `train` chunks.
5. Tune and compare on `validation` chunks.
6. Keep a held-out local set for model regression tests.
7. Track performance by release date and model version.

## Notes

- New releases may be added over time.
- Response fields may expand, so clients should ignore unknown fields.
- The chunk order and label order are significant.
- Avoid tuning a model against a single release only.
- Prefer evaluation across multiple release dates.
