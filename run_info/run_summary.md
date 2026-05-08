# 运行摘要

## 目的
本 zip 用于验证修复后 SatEdgeSim 是否正确复现论文第三章和第四章算法与图像。

## 选择的结果目录
- 第三章结果目录: output/chapter3/2026-05-07_19-38-31
- 第四章结果目录: output/chapter4/2026-05-07_22-11-09
- 说明: 扫描 output/chapter3 与 output/chapter4 后，上述目录是当前最新结果目录，与用户指定目录一致。

## 第三章结果文件
- chapter3_results.csv
- Sequential_simulation.csv
- Sequential_simulation.txt
- chapter3_figures/，图像数量: 9

## 第四章结果文件
- chapter4_results.csv
- chapter4_speed_results.csv
- chapter4_ddldo_decisions.csv，数据行数: 100445
- chapter4_ddldo_training.csv，数据行数: 100446
- Sequential_simulation.csv
- Sequential_simulation.txt
- chapter4_figures/，图像数量: 9

## 代码修复扫描结果
- issetlink / 星间链路海伦公式: 已在 $simMgr 中扫描到，包含 p * (p - h1) * (p - d) * (p - h2)。
- 第三章 Q-learning reward: 位于 $orch 的 qLearningReward 方法，local 动作使用 CHAPTER3_LOCAL_ETE_WEIGHT / eteDelay + CHAPTER3_LOCAL_EXE_WEIGHT / executionDelay，offload 动作使用 -chapter3Objective(...)。
- 第三章权重: $orch 中 CHAPTER3_WEIGHTS = { 6.0, 6.0, 3.0, 5.0 }。
- 第四章 DDLDO K 值: $orch 中 DDLDO_DNN_COUNT = 3。
- DDLDO 决策日志: 包含 selectedAction / selectedVmId / objective / delay / energy。
- 覆盖时间字段: 决策日志和代码中包含 coverageRemainingTime / estimatedFinishTime / coverageFeasible。
- 车速实验结果: chapter4_speed_results.csv 由 source/tools/create_chapter4_speed_results.py 从真实 chapter4 输出目录聚合；未发现 SERIES = { 等硬编码曲线数组。

## 运行日志
父级 un_logs/ 已复制到 un_info/run_logs/，包含第三章完整运行日志、第四章主实验日志和第四章各车速真实仿真日志。

## 需要人工确认
- 当前未发现明确 .tle 或 Walker 星座参数文件，但 settings/locationflie 下包含 STK 或固定位置 CSV。需要人工确认这些坐标是否由论文表中 Walker 参数生成。
