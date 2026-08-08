# Contributing

Eagle Eye X is a small research team working on a UAV tracking system. Keep it
simple, keep it reproducible.

## Branches

- `main` — production tracking pipeline (ASPS mainline). Keep green.
- Feature/experiment work happens on short-lived branches, then a PR into `main`.
- `without-lidar` is a legacy lineage branch; do not build on it.

## Workflow

1. `git checkout -b feat/<short-name>` (or `experiment/<short-name>`).
2. Make changes, run the validation below.
3. Open a PR with a clear description; tag the relevant maintainer.
4. After review, merge.

## Code conventions

- Python 3.10. Keep module-level constants in `configs/*.yaml` — do **not** hardcode
  new tunables in scripts (see `src/ASPS_LIDAR/config.py`).
- Do not delete or modify anything in `archive/` — it is the preserved research
  lineage (see `archive/README.md`).
- Personal machine settings go in `configs/local/` (gitignored), never in `configs/`.
- Model weights, flight logs, and datasets stay out of git (see `.gitignore`).
- New class ids must be mirrored in `model/classes.yaml` and `configs/asps.yaml`.
- Always run scripts **from the repo root** — model/video paths are CWD-relative.

## Onboarding a teammate

```bash
git clone git@github.com:WOAH-KAMRAN/EAGLE_EYE_X.git && cd EAGLE_EYE_X
make setup && make models     # venv + dev weights (best.pt stays private)
# then hand them RUN.md
```

## Validation (before pushing)

```bash
# static check of the production scripts
source VENV/uav/bin/activate
python -m py_compile src/ASPS_LIDAR/ASPS_LIDAR_TR.py src/ASPS_LIDAR/prioritization.py src/ASPS_LIDAR/config.py

# config layering sanity (no camera needed)
python -c "import sys; sys.path.insert(0,'src/ASPS_LIDAR'); import config; print(config.load_flat_config().get('ALPHA'))"

# flight-log tooling on a sample log
python tools/analyze_flight.py flights1/flight_20251016_073717/log_20251016_073717.csv
```

If you change the flight-log schema, update `docs/ARCHITECTURE.md` and
`tools/analyze_flight.py` column aliases.

## Commits

- Concise, imperative subject lines matching the existing log style.
- One logical change per commit.
- Never commit secrets or model weights.
