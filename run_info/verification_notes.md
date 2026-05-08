# 验证备注

## 1. issetlink / 星间链路可见性判断方法位置
文件: edu/weijunyong/satedgesim/SimulationManager/SimulationManager.java

``java
443:	public static boolean issetlink(DataCenter device1, DataCenter device2) {	//几何可见建立链路
444:		double h1 = getHight(device1), h2 = getHight(device2), d = getdistance(device1, device2);
445:		if(d <= 0) {
446:			return true;
447:		}
448:		double p = (h1 + h2 + d) / 2.0;
449:		double areaSquared = p * (p - h1) * (p - d) * (p - h2);
450:		if (areaSquared <= 0 || Double.isNaN(areaSquared) || Double.isInfinite(areaSquared)) {
451:			return false;
452:		}
453:		double L = 2.0 * Math.sqrt(areaSquared) / d;
454:		return !Double.isNaN(L) && !Double.isInfinite(L)
455:				&& L > simulationParameters.MIN_HEIGHT + simulationParameters.EARTH_RADIUS;
``

确认点: 第 449 行包含 p * (p - h1) * (p - d) * (p - h2)。

## 2. 第三章权重 6, 6, 3, 5
文件: edu/weijunyong/satedgesim/TasksOrchestration/DefaultEdgeOrchestrator.java

``java
35:	private static final double[] CHAPTER3_WEIGHTS = { 6.0, 6.0, 3.0, 5.0 };
36:	private static final double CHAPTER3_LOCAL_ETE_WEIGHT = 0.5;
37:	private static final double CHAPTER3_LOCAL_EXE_WEIGHT = 0.5;
38:	private static final double REWARD_EPSILON = 1.0E-6;
39:	private static final double LATENCY_FAILURE_PENALTY = 1.0;
40:	private static final double MOBILITY_FAILURE_PENALTY = 0.8;
``

## 3. 第三章 Q-learning reward 方法位置
文件: edu/weijunyong/satedgesim/TasksOrchestration/DefaultEdgeOrchestrator.java

``java
523:	private double qLearningReward(Task task, QLearningDecision decision) {
524:		double reward;
525:		if (decision.action == ACTION_LOCAL) {
526:			double eteDelay = Math.max(REWARD_EPSILON, decision.propagationDelay + decision.executionDelay);
527:			double executionDelay = Math.max(REWARD_EPSILON, decision.executionDelay);
528:			reward = CHAPTER3_LOCAL_ETE_WEIGHT / eteDelay + CHAPTER3_LOCAL_EXE_WEIGHT / executionDelay;
529:		} else {
530:			reward = -chapter3Objective(decision.propagationDelayStand, decision.executionDelayStand,
531:					decision.energyStand, decision.parallelTasksStand);
532:		}
533:		if (task.getFailureReason() == Task.Status.FAILED_DUE_TO_LATENCY) {
534:			reward -= LATENCY_FAILURE_PENALTY;
535:		} else if (task.getFailureReason() == Task.Status.FAILED_DUE_TO_DEVICE_MOBILITY) {
536:			reward -= MOBILITY_FAILURE_PENALTY;
537:		} else if (task.getFailureReason() != null && task.getFailureReason() != Task.Status.NULL) {
538:			reward -= 0.5;
539:		}
540:		return reward;
541:	}
``

## 4. 第三章 objective 计算
文件: edu/weijunyong/satedgesim/TasksOrchestration/DefaultEdgeOrchestrator.java

``java
723:	private double chapter3Objective(double propagationDelay, double executionDelay, double energy, double parallelTasks) {
724:		return CHAPTER3_WEIGHTS[0] * propagationDelay + CHAPTER3_WEIGHTS[1] * executionDelay
725:				+ CHAPTER3_WEIGHTS[2] * energy + CHAPTER3_WEIGHTS[3] * parallelTasks;
726:	}
``

## 5. DDLDO_DNN_COUNT 是否为 3 / DDLDO_ACTIONS 位置
文件: edu/weijunyong/satedgesim/TasksOrchestration/DefaultEdgeOrchestrator.java

``java
43:	private static final int ACTION_LOCAL = 0;
44:	private static final int ACTION_OFFLOAD = 1;
45:	private static final int DDLDO_ACTIONS = 4;
46:	private static final int DDLDO_INPUTS = 4;
47:	private static final int DDLDO_DNN_COUNT = 3;
48:	private static final int DDLDO_HIDDEN_1 = 12;
49:	private static final int DDLDO_HIDDEN_2 = 8;
50:	private static final int DDLDO_BATCH_SIZE = 256;
51:	private static final int DDLDO_MEMORY_SIZE = 4096;
52:	private static final int DDLDO_EPOCHS = 1000;
53:	private static final int DDLDO_EPOCHS_PER_STEP = 1;
54:	private static final double DDLDO_LEARNING_RATE = 0.0001;
``

## 6. DDLDO 候选决策生成
文件: edu/weijunyong/satedgesim/TasksOrchestration/DefaultEdgeOrchestrator.java

``java
771:	private List<DdldoDecision> ddldoCandidateDecisions(Task task, double[] input, List<Integer> possibleVms,
772:			MetricSnapshot metrics) {
773:		List<DdldoDecision> candidates = new ArrayList<>();
774:		List<Integer> usedActions = new ArrayList<>();
775:		for (SimpleDnn network : ddldoNetworks) {
776:			double[] scores = adjustedDdldoScores(task, network.predict(input));
777:			DdldoDecision decision = bestDecisionFromScores(task, scores, possibleVms, metrics, usedActions);
778:			if (decision != null) {
779:				candidates.add(decision);
780:				usedActions.add(decision.action);
781:			}
782:		}
783:		if (candidates.isEmpty()) {
784:			DdldoDecision fallback = fallbackDdldoDecision(task, possibleVms, metrics);
785:			if (fallback != null) {
786:				candidates.add(fallback);
787:			}
788:		}
789:		return candidates;
790:	}
``

## 7. DDLDO 最优候选选择
文件: edu/weijunyong/satedgesim/TasksOrchestration/DefaultEdgeOrchestrator.java

``java
851:	private DdldoDecision selectBestDdldoDecision(List<DdldoDecision> candidates) {
852:		DdldoDecision best = null;
853:		for (DdldoDecision decision : candidates) {
854:			if (!decision.feasible) {
855:				continue;
856:			}
857:			if (best == null || decision.objective < best.objective) {
858:				best = decision;
859:			}
860:		}
861:		return best;
862:	}
``

## 8. coverageRemainingTime / estimatedFinishTime / coverageFeasible 覆盖时间字段
文件: edu/weijunyong/satedgesim/TasksOrchestration/DefaultEdgeOrchestrator.java

``java
891:	private DdldoDecision evaluateDdldoDecision(Task task, int vm, MetricSnapshot metrics) {
892:		DataCenter destination = (DataCenter) vmList.get(vm).getHost().getDatacenter();
893:		int action = ddldoActionForVm(task, vm);
894:		double delay = metrics.propagationDelay.get(vm) + metrics.executionDelay.get(vm);
895:		double energy = metrics.energyRaw.get(vm);
896:		double coverageRemainingTime = coverageRemainingTime(task, destination, delay);
897:		double estimatedFinishTime = simulationManager.getSimulation().clock() + delay;
898:		boolean coverageFeasible = coverageRemainingTime < 0 || delay <= coverageRemainingTime;
899:		boolean latencyFeasible = delay <= task.getMaxLatency();
900:		boolean energyFeasible = energyFeasible(destination, energy);
901:		boolean linkFeasible = destination == task.getEdgeDevice() || SimulationManager.issetlink(task.getEdgeDevice(), destination);
902:		boolean resourceFeasible = !destination.isDead() && vmList.get(vm).getMips() > 0;
903:		String failureReason = ddldoFailureReason(coverageFeasible, latencyFeasible, energyFeasible, linkFeasible,
904:				resourceFeasible);
905:		double objective = chapter4Objective(vm, metrics);
906:		if (!latencyFeasible) {
907:			objective += 2.0 + (delay - task.getMaxLatency()) / Math.max(1.0, task.getMaxLatency());
908:		}
909:		if (!coverageFeasible) {
910:			objective += 1.5;
911:		}
912:		if (!energyFeasible) {
913:			objective += 1.0;
914:		}
915:		if (!linkFeasible || !resourceFeasible) {
916:			objective += 3.0;
917:		}
918:		return new DdldoDecision(action, vm, targetId(destination), objective, delay, energy, coverageRemainingTime,
919:				estimatedFinishTime, coverageFeasible, latencyFeasible, energyFeasible, linkFeasible, resourceFeasible,
920:				failureReason);
``

## 9. DDLDO decision log 写入位置
文件: edu/weijunyong/satedgesim/TasksOrchestration/DefaultEdgeOrchestrator.java

``java
1082:	private void appendDdldoDecisionLog(Task task, DdldoDecision decision, boolean success, String failureReason) {
1083:		String header = "time,taskId,deviceId,taskLength,taskFileSize,delaySensitivity,vehicleSpeed,sourceLeoId,"
1084:				+ "candidateCount,candidateActions,candidateVmIds,selectedAction,selectedVmId,objective,delay,energy,"
1085:				+ "coverageRemainingTime,estimatedFinishTime,coverageFeasible,latencyFeasible,energyFeasible,success,failureReason";
1086:		String line = joinCsv(
1087:				csvDouble(simulationManager.getSimulation().clock()),
1088:				Long.toString(task.getId()),
1089:				Integer.toString(targetId(task.getEdgeDevice())),
1090:				Long.toString(task.getLength()),
1091:				Long.toString(task.getFileSize()),
1092:				csvDouble(latencySensitivity(task)),
1093:				csvDouble(simulationParameters.VEHICLE_SPEED),
1094:				Integer.toString(targetId(task.getEdgeDevice())),
1095:				Integer.toString(decision.candidateCount),
1096:				decision.candidateActions,
1097:				decision.candidateVmIds,
1098:				Integer.toString(decision.action),
1099:				Integer.toString(decision.targetId),
1100:				csvDouble(decision.objective),
1101:				csvDouble(decision.delay),
1102:				csvDouble(decision.energy),
1103:				csvDouble(decision.coverageRemainingTime),
1104:				csvDouble(decision.estimatedFinishTime),
1105:				Boolean.toString(decision.coverageFeasible),
1106:				Boolean.toString(decision.latencyFeasible),
1107:				Boolean.toString(decision.energyFeasible),
1108:				Boolean.toString(success),
1109:				failureReason == null ? "" : failureReason);
1110:		appendCsvLine(ddldoLogPath("chapter4_ddldo_decisions.csv"), header, line);
``

## 10. DDLDO training log 写入位置
文件: edu/weijunyong/satedgesim/TasksOrchestration/DefaultEdgeOrchestrator.java

``java
1113:	private void appendDdldoTrainingLog(double loss, DdldoDecision decision) {
1114:		String header = "time,episodeOrStep,memorySize,batchSize,learningRate,loss,avgObjective,selectedAction,selectedVmId";
1115:		String line = joinCsv(
1116:				csvDouble(simulationManager.getSimulation().clock()),
1117:				Integer.toString(ddldoEpochsTrained),
1118:				Integer.toString(ddldoMemory.size()),
1119:				Integer.toString(DDLDO_BATCH_SIZE),
1120:				csvDouble(DDLDO_LEARNING_RATE),
1121:				csvDouble(loss),
1122:				csvDouble(averageDdldoObjective()),
1123:				Integer.toString(decision.action),
1124:				Integer.toString(decision.targetId));
1125:		appendCsvLine(ddldoLogPath("chapter4_ddldo_training.csv"), header, line);
1126:	}
``

## 11. create_chapter4_speed_results.py 是否仍有硬编码 SERIES 数组
脚本已复制到 source/tools/create_chapter4_speed_results.py。检查文件: checks/no_hardcoded_result_check.txt。
当前未发现 SERIES = {、DDLDO": [ 或 TaskClfQLearning": [ 这类硬编码曲线数组；脚本中存在 chapter4_speed_results.csv 输出路径相关命中，属于正常结果文件写入。
