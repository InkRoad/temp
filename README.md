# SatEdgeSim Chapter 4 Updated Reproduction Package

Generated: 2026-05-09 15:45:05

This repository contains the latest updated SatEdgeSim source/configuration and the newly rerun Chapter 4 reproduction results.

## Contents

- `source/`: current Java source, `pom.xml`, examples, and project tools.
- `settings/`: current SatEdgeSim settings including Chapter 4 simulation parameters.
- `latest_output/chapter4/2026-05-09_14-16-16/`: final full Chapter 4 device-count experiment, regenerated CSV files, DDLDO logs, and figures.
- `latest_output/chapter4/<speed-run-dir>/`: raw real simulation outputs for speed=30..120, used to build `chapter4_speed_results.csv`.
- `run_info/`: commands and verification notes.
- `checks/`: important file list, result summary, and hardcoded-result scan.

## Latest Main Results

- `latest_output/chapter4/2026-05-09_14-16-16/chapter4_results.csv`
- `latest_output/chapter4/2026-05-09_14-16-16/chapter4_speed_results.csv`
- `latest_output/chapter4/2026-05-09_14-16-16/chapter4_ddldo_decisions.csv`
- `latest_output/chapter4/2026-05-09_14-16-16/chapter4_ddldo_training.csv`
- `latest_output/chapter4/2026-05-09_14-16-16/chapter4_figures/`

## GPU

The full run used CUDA device 0:

```text
NVIDIA GeForce RTX 3050 Ti Laptop GPU
ND4J backend: org.nd4j.linalg.jcublas.JCublasBackend
ND4J executioner: org.nd4j.linalg.jcublas.ops.executioner.CudaExecutioner
DDLDO DNN using GPU: true
```

## Important Caveat

The final full run shows DDLDO better than most baselines, but not better than `TASK_CLF_Q_LEARNING`. This is preserved honestly in the uploaded CSV and should be manually reviewed.
