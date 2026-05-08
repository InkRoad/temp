# SatEdgeSim 第三章第四章复现检查包 v2

## 用途
该 zip 用于让 ChatGPT 验证修复后代码和新仿真结果是否正确复现论文第三章、第四章，包括算法实现、CSV 原始结果、DDLDO 日志和图像。

## 打包时间
2026-05-07 23:22:00 +08:00

## 项目路径
D:\Project\homework\dachuang\SatEdgeSim-master\SatEdgeSim

## 选择的最新结果
第三章:  
output/chapter3/2026-05-07_19-38-31

第四章:  
output/chapter4/2026-05-07_22-11-09

## 包含内容
- source/: 最新 Java 源码、examples、父级 	ools/、父级 pom.xml。
- settings/: 当前仿真配置与 settings/locationflie 位置 CSV。
- latest_output/: 最新第三章结果、最新第四章结果、论文第三章/第四章对照图。
- un_info/: 环境信息、运行摘要、验证备注、运行脚本和 un_logs/。
- checks/: 重要文件清单、结果文件摘要、硬编码结果检查。

## 关键修复点
- 星间链路海伦公式: $simMgr 中 issetlink 使用 p * (p - h1) * (p - d) * (p - h2)，并对 d <= 0、reaSquared <= 0、NaN/Infinity 做保护。
- 第三章 Q-learning reward: $orch 中 qLearningReward 区分 local/offload，offload 使用 6:6:3:5 加权目标，失败任务有惩罚。
- 第四章 DDLDO K=3: $orch 中 DDLDO_DNN_COUNT = 3。
- DDLDO 决策日志: chapter4_ddldo_decisions.csv 包含 selectedAction、selectedVmId、objective、delay、energy 等字段。
- DDLDO 训练日志: chapter4_ddldo_training.csv 包含 memorySize、batchSize、learningRate、loss、avgObjective 等字段。
- 覆盖时间判断: 代码和日志中包含 coverageRemainingTime、estimatedFinishTime、coverageFeasible。
- 车速实验去硬编码: create_chapter4_speed_results.py 从真实 chapter4 输出目录聚合 speed 结果；未发现明显硬编码曲线数组。

## 结果文件说明
- latest_output/chapter3/2026-05-07_19-38-31/chapter3_results.csv: 第三章汇总结果，60 行。
- latest_output/chapter4/2026-05-07_22-11-09/chapter4_results.csv: 第四章汇总结果，105 行。
- latest_output/chapter4/2026-05-07_22-11-09/chapter4_speed_results.csv: 第四章车速实验结果，70 行，speed 包含 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0。
- latest_output/chapter4/2026-05-07_22-11-09/chapter4_ddldo_decisions.csv: DDLDO 逐任务决策日志，100445 行。
- latest_output/chapter4/2026-05-07_22-11-09/chapter4_ddldo_training.csv: DDLDO 训练日志，100446 行。
- latest_output/chapter3/2026-05-07_19-38-31/chapter3_figures/: 第三章复现图像。
- latest_output/chapter4/2026-05-07_22-11-09/chapter4_figures/: 第四章复现图像。

## 需要 ChatGPT 重点检查
1. 第三章 TaskClfQLearning 成功率是否达到论文趋势。
2. 第三章平均端到端延迟、执行延迟、能耗是否与论文趋势一致。
3. 第四章 DDLDO 是否优于或接近优于 TaskClfQLearning 和其他基准。
4. 第四章 DDLDO 成功率是否达到论文描述。
5. 第四章车速实验是否来自真实仿真。
6. DDLDO 决策日志是否能证明算法不是简单贪婪。

## 已知问题
当前未发现明确 .tle 或 Walker 星座参数文件，但 settings/locationflie 下包含 STK 或固定位置 CSV。需要人工确认这些坐标是否由论文表中 Walker 参数生成。

## GitHub upload note
The original settings/locationflie/edge_devices/mist Fixed Position1.csv in the local package was only a missing Git LFS pointer. For this GitHub upload, it is stored as an explanatory placeholder containing the original pointer text, because GitHub rejects unresolved LFS pointers. The large mist Fixed Position.csv file remains tracked through Git LFS.
