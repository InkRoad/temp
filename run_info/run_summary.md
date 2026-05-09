# SatEdgeSim Chapter 4 Step 3 Full Run Summary

Generated: 2026-05-09 15:45:05

Purpose: upload the latest fixed SatEdgeSim source, settings, full Chapter 4 simulation outputs, speed experiment raw outputs, DDLDO logs, and regenerated figures to GitHub for verification.

Latest Chapter 4 main output: `latest_output/chapter4/2026-05-09_14-16-16/`

Raw speed run outputs included under `latest_output/chapter4/`:
- speed 30: `2026-05-09_12-47-37`
- speed 40: `2026-05-09_12-58-23`
- speed 50: `2026-05-09_13-07-14`
- speed 60: `2026-05-09_13-32-12`
- speed 70: `2026-05-09_13-38-27`
- speed 80: `2026-05-09_13-44-40`
- speed 90: `2026-05-09_13-50-52`
- speed 100: `2026-05-09_13-57-08`
- speed 110: `2026-05-09_14-03-25`
- speed 120: `2026-05-09_14-09-42`

Commands used locally:

```powershell
$env:CUDA_VISIBLE_DEVICES='0'
mvn.cmd -q exec:java -Dexec.mainClass=edu.weijunyong.satedgesim.MainApplication -Dexec.args=chapter4 -Dvehicle.speed=<30..120> -Dquick.single.device.count=1000
python tools\create_chapter4_speed_results.py
mvn.cmd -q exec:java -Dexec.mainClass=edu.weijunyong.satedgesim.MainApplication -Dexec.args=chapter4 -Dvehicle.speed=60
python tools\extract_chapter4_results.py
python tools\plot_chapter4_results.py
```

GPU proof observed in full run logs:

```text
ND4J backend: org.nd4j.linalg.jcublas.JCublasBackend
ND4J executioner: org.nd4j.linalg.jcublas.ops.executioner.CudaExecutioner
DDLDO DNN using GPU: true
```

Notes:
- DDLDO uses 3 parallel DNNs with 20 -> 64 -> 128 -> 64 -> 1 structure.
- `energyFeedbackType` remains `datacenter_delta`.
- `coverage=-1` remains Not Applicable and does not trigger coverage infeasibility.
- The speed figure is generated from `chapter4_speed_results.csv`, which was aggregated from the raw speed output directories included here.
