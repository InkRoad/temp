# Verification Notes

Generated: 2026-05-09 15:45:05

Important implementation files:
- `source/edu/weijunyong/satedgesim/TasksOrchestration/DefaultEdgeOrchestrator.java`
- `source/edu/weijunyong/satedgesim/ScenarioManager/FilesParser.java`
- `source/edu/weijunyong/satedgesim/LocationManager/DefaultMobilityModel.java`
- `settings/chapter4_simulation_parameters.properties`

Key checks:
- Chapter 4 config now uses `edge_device_counter_time = 10`, giving device counts 100,200,...,1000 with the 1000-node location file.
- `chapter4_results.csv` contains 70 rows = 7 algorithms x 10 device counts.
- `chapter4_speed_results.csv` contains 70 rows = 7 algorithms x 10 speeds.
- Regenerated figures are in `latest_output/chapter4/2026-05-09_14-16-16/chapter4_figures/`.
- DDLDO decisions log rows: 69604.
- Energy feedback type counts: {'datacenter_delta': 69604}.

DDLDO is better than most baseline algorithms in average ETE/success, but it is not better than `TASK_CLF_Q_LEARNING` in the final full run. This should be manually reviewed rather than hidden.
