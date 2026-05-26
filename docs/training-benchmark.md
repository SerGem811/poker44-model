# Poker44 Training Benchmark

Public benchmark guide for Poker44 subnet `126`.

## Overview

Poker44 exposes a public training benchmark designed to help miners improve their
models without turning the public surface into a copy of the live competition exam.

Its purpose is to support:

- miner iteration;
- supervised training;
- offline validation;
- regression testing;
- calibration against miner-visible payloads.

The benchmark is served by the central backend API.

## Scope

The benchmark is intended as a public development surface.

Live evaluation and benchmark access remain separate surfaces:

- live evaluation is used for active competition settlement;
- benchmark responses are for offline experimentation and training;
- benchmark releases are recomposed from consumed human material plus benchmark-specific bot profiles;
- benchmark availability, release cadence, and payload details may evolve over time.

## API Base

- `https://api.poker44.net/api/v1/benchmark`

## Available Endpoints

- `GET /api/v1/benchmark`
- `GET /api/v1/benchmark/releases`
- `GET /api/v1/benchmark/chunks?sourceDate=YYYY-MM-DD`
- `GET /api/v1/benchmark/chunks/:chunkId`

## Response Shape

Benchmark responses include:

- release metadata;
- miner-visible chunk payloads;
- label data kept separate from the hand payload itself;
- split metadata (`train` / `validation`) when applicable.

Consumers should rely on the returned response fields rather than assuming the
format will remain fixed beyond the documented API surface.

## Intended Use

Typical uses include:

- supervised training;
- offline evaluation;
- calibration;
- feature engineering;
- repeated comparison across model versions.

## Guidance

Use the benchmark as a development aid, not as a guarantee that future live
evaluation will match any single public release.

In particular:

- avoid overfitting to one release group;
- validate out of sample;
- assume benchmark composition and release policy can change over time;
- optimize for robust generalization, not for any single benchmark snapshot.
