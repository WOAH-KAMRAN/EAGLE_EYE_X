# Experiments

Research-facing code that is not part of the production flight loop.

```
depth/      Monocular-depth estimation experiments (MiDaS / ZoeDepth) +
            focal-length calibration tools and midas_calibration.json.
ablation/   SITL config-matrix runner for ASPS vs baseline comparisons.
results/    (gitignored) ablation output: per-trial configs, metadata,
            flight logs, and comparison_summary.csv.
```

See `docs/RESEARCH.md` for the protocol, and `docs/ARCHITECTURE.md` for where these
fit in the system.
