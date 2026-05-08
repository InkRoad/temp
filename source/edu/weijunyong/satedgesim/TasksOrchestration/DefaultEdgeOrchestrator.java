package edu.weijunyong.satedgesim.TasksOrchestration;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.ArrayDeque;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.cloudbus.cloudsim.vms.Vm;

import edu.weijunyong.satedgesim.DataCentersManager.DataCenter;
import edu.weijunyong.satedgesim.ScenarioManager.simulationParameters;
import edu.weijunyong.satedgesim.SimulationManager.SimLog;
import edu.weijunyong.satedgesim.SimulationManager.SimulationManager;
import edu.weijunyong.satedgesim.TasksGenerator.Task;

public class DefaultEdgeOrchestrator extends Orchestrator {
	public DefaultEdgeOrchestrator(SimulationManager simulationManager) {
		super(simulationManager);
	}
	
	public static int FindVmId_TP =0;
	public static int Counttask = 0;
	private static final String TASK_CLF_Q_LEARNING = "TASK_CLF_Q_LEARNING";
	private static final double Q_LEARNING_RATE = 0.35;
	private static final double Q_DISCOUNT_FACTOR = 0.80;
	private static final double Q_EPSILON = 0.10;
	private static final double Q_INITIAL_SPREAD = 1.0E-6;
	private static final double[] CHAPTER3_WEIGHTS = { 6.0, 6.0, 3.0, 5.0 };
	private static final double CHAPTER3_LOCAL_ETE_WEIGHT = 0.5;
	private static final double CHAPTER3_LOCAL_EXE_WEIGHT = 0.5;
	private static final double REWARD_EPSILON = 1.0E-6;
	private static final double LATENCY_FAILURE_PENALTY = 1.0;
	private static final double MOBILITY_FAILURE_PENALTY = 0.8;
	private static final double CHAPTER4_DELAY_WEIGHT = 0.9;
	private static final double CHAPTER4_ENERGY_WEIGHT = 0.1;
	private static final int ACTION_LOCAL = 0;
	private static final int ACTION_OFFLOAD = 1;
	private static final int DDLDO_ACTIONS = 4;
	private static final int DDLDO_INPUTS = 4;
	private static final int DDLDO_DNN_COUNT = 3;
	private static final int DDLDO_HIDDEN_1 = 12;
	private static final int DDLDO_HIDDEN_2 = 8;
	private static final int DDLDO_BATCH_SIZE = 256;
	private static final int DDLDO_MEMORY_SIZE = 4096;
	private static final int DDLDO_EPOCHS = 1000;
	private static final int DDLDO_EPOCHS_PER_STEP = 1;
	private static final double DDLDO_LEARNING_RATE = 0.0001;
	private final Map<String, Map<Integer, Double>> qTable = new HashMap<>();
	private final Map<Long, QLearningDecision> qLearningDecisions = new HashMap<>();
	private final Deque<DdldoSample> ddldoMemory = new ArrayDeque<>();
	private final Map<Long, DdldoDecision> ddldoPendingDecisions = new HashMap<>();
	private final List<SimpleDnn> ddldoNetworks = new ArrayList<>();
	private boolean ddldoInitialized = false;
	private int ddldoEpochsTrained = 0;

	protected int findVM(String[] architecture, Task task) {
		if ("ROUND_ROBIN".equals(algorithm)) {
			return roundRobin(architecture, task);
		} else if ("TRADE_OFF".equals(algorithm)) {
			return tradeOff(architecture, task);
		} else if ("TRADI_POLLING".equals(algorithm)) {
			return TradiPolling(architecture, task);
		} else if ("WEIGHT_GREEDY".equals(algorithm)) {
			return weightGreedy(architecture, task);
		} else if ("DDLDO".equals(algorithm)) {
			return ddldo(architecture, task);
		} else if (TASK_CLF_Q_LEARNING.equals(algorithm) || "TaskClfQLearning".equals(algorithm)) {
			return taskClfQLearning(architecture, task);
		} else if ("RANDOM_VM".equals(algorithm)) {
			return RandomVm(architecture, task);
		} else {
			SimLog.println("");
			SimLog.println("Default Orchestrator- Unknnown orchestration algorithm '" + algorithm
					+ "', please check the simulation parameters file...");
			// Cancel the simulation
			Runtime.getRuntime().exit(0);
		}
		return -1;
	}

	private static class QLearningDecision {
		private final String state;
		private final int action;
		private final int vmIndex;
		private final double propagationDelay;
		private final double executionDelay;
		private final double propagationDelayStand;
		private final double executionDelayStand;
		private final double energyStand;
		private final double parallelTasksStand;

		private QLearningDecision(String state, int action, int vmIndex, double propagationDelay, double executionDelay,
				double propagationDelayStand, double executionDelayStand, double energyStand, double parallelTasksStand) {
			this.state = state;
			this.action = action;
			this.vmIndex = vmIndex;
			this.propagationDelay = propagationDelay;
			this.executionDelay = executionDelay;
			this.propagationDelayStand = propagationDelayStand;
			this.executionDelayStand = executionDelayStand;
			this.energyStand = energyStand;
			this.parallelTasksStand = parallelTasksStand;
		}
	}

	private static class MetricSnapshot {
		private final List<Double> propagationDelay = new ArrayList<>();
		private final List<Double> executionDelay = new ArrayList<>();
		private final List<Double> energy = new ArrayList<>();
		private final List<Double> energyRaw = new ArrayList<>();
		private final List<Double> parallelTasks = new ArrayList<>();
		private final List<Double> propagationDelayStand = new ArrayList<>();
		private final List<Double> executionDelayStand = new ArrayList<>();
		private final List<Double> energyStand = new ArrayList<>();
		private final List<Double> parallelTasksStand = new ArrayList<>();
	}

	private static class DdldoSample {
		private final double[] input;
		private final double[] label;
		private final int action;
		private final int vmIndex;
		private final double objective;
		private final double delay;
		private final double energy;
		private final boolean success;
		private final String failureReason;

		private DdldoSample(double[] input, double[] label, DdldoDecision decision) {
			this.input = input;
			this.label = label;
			this.action = decision.action;
			this.vmIndex = decision.vmIndex;
			this.objective = decision.objective;
			this.delay = decision.delay;
			this.energy = decision.energy;
			this.success = decision.feasible;
			this.failureReason = decision.failureReason;
		}
	}

	private static class DdldoDecision {
		private final int action;
		private final int vmIndex;
		private final int targetId;
		private double objective;
		private final double delay;
		private final double energy;
		private final double coverageRemainingTime;
		private final double estimatedFinishTime;
		private final boolean coverageFeasible;
		private final boolean latencyFeasible;
		private final boolean energyFeasible;
		private final boolean linkFeasible;
		private final boolean resourceFeasible;
		private final boolean feasible;
		private String failureReason;
		private int candidateCount;
		private String candidateActions = "";
		private String candidateVmIds = "";

		private DdldoDecision(int action, int vmIndex, int targetId, double objective, double delay, double energy,
				double coverageRemainingTime, double estimatedFinishTime, boolean coverageFeasible,
				boolean latencyFeasible, boolean energyFeasible, boolean linkFeasible, boolean resourceFeasible,
				String failureReason) {
			this.action = action;
			this.vmIndex = vmIndex;
			this.targetId = targetId;
			this.objective = objective;
			this.delay = delay;
			this.energy = energy;
			this.coverageRemainingTime = coverageRemainingTime;
			this.estimatedFinishTime = estimatedFinishTime;
			this.coverageFeasible = coverageFeasible;
			this.latencyFeasible = latencyFeasible;
			this.energyFeasible = energyFeasible;
			this.linkFeasible = linkFeasible;
			this.resourceFeasible = resourceFeasible;
			this.feasible = coverageFeasible && latencyFeasible && energyFeasible && linkFeasible && resourceFeasible;
			this.failureReason = failureReason;
		}
	}

	private static class SimpleDnn {
		private final double[][] w1;
		private final double[] b1;
		private final double[][] w2;
		private final double[] b2;
		private final double[][] w3;
		private final double[] b3;

		private SimpleDnn() {
			w1 = new double[DDLDO_INPUTS][DDLDO_HIDDEN_1];
			b1 = new double[DDLDO_HIDDEN_1];
			w2 = new double[DDLDO_HIDDEN_1][DDLDO_HIDDEN_2];
			b2 = new double[DDLDO_HIDDEN_2];
			w3 = new double[DDLDO_HIDDEN_2][DDLDO_ACTIONS];
			b3 = new double[DDLDO_ACTIONS];
			init(w1);
			init(w2);
			init(w3);
		}

		private void init(double[][] weights) {
			for (int i = 0; i < weights.length; i++) {
				for (int j = 0; j < weights[i].length; j++) {
					weights[i][j] = (simulationParameters.RANDOM_GENERATOR.nextDouble() - 0.5) * 0.02;
				}
			}
		}

		private double[] predict(double[] input) {
			double[] h1 = sigmoid(add(matmul(input, w1), b1));
			double[] h2 = sigmoid(add(matmul(h1, w2), b2));
			return softmax(add(matmul(h2, w3), b3));
		}

		private double train(List<DdldoSample> samples) {
			double loss = 0.0;
			for (DdldoSample sample : samples) {
				double[] z1 = add(matmul(sample.input, w1), b1);
				double[] h1 = sigmoid(z1);
				double[] z2 = add(matmul(h1, w2), b2);
				double[] h2 = sigmoid(z2);
				double[] output = softmax(add(matmul(h2, w3), b3));
				loss += crossEntropy(output, sample.label);

				double[] delta3 = new double[DDLDO_ACTIONS];
				for (int i = 0; i < DDLDO_ACTIONS; i++) {
					delta3[i] = output[i] - sample.label[i];
				}

				double[] delta2 = new double[DDLDO_HIDDEN_2];
				for (int i = 0; i < DDLDO_HIDDEN_2; i++) {
					double sum = 0.0;
					for (int j = 0; j < DDLDO_ACTIONS; j++) {
						sum += delta3[j] * w3[i][j];
					}
					delta2[i] = sum * h2[i] * (1.0 - h2[i]);
				}

				double[] delta1 = new double[DDLDO_HIDDEN_1];
				for (int i = 0; i < DDLDO_HIDDEN_1; i++) {
					double sum = 0.0;
					for (int j = 0; j < DDLDO_HIDDEN_2; j++) {
						sum += delta2[j] * w2[i][j];
					}
					delta1[i] = sum * h1[i] * (1.0 - h1[i]);
				}

				update(w3, h2, delta3);
				update(b3, delta3);
				update(w2, h1, delta2);
				update(b2, delta2);
				update(w1, sample.input, delta1);
				update(b1, delta1);
			}
			return samples.isEmpty() ? 0.0 : loss / samples.size();
		}

		private static double[] matmul(double[] input, double[][] weights) {
			double[] output = new double[weights[0].length];
			for (int j = 0; j < output.length; j++) {
				double sum = 0.0;
				for (int i = 0; i < input.length; i++) {
					sum += input[i] * weights[i][j];
				}
				output[j] = sum;
			}
			return output;
		}

		private static double[] add(double[] values, double[] bias) {
			double[] output = new double[values.length];
			for (int i = 0; i < values.length; i++) {
				output[i] = values[i] + bias[i];
			}
			return output;
		}

		private static double[] sigmoid(double[] values) {
			double[] output = new double[values.length];
			for (int i = 0; i < values.length; i++) {
				output[i] = 1.0 / (1.0 + Math.exp(-values[i]));
			}
			return output;
		}

		private static double[] softmax(double[] values) {
			double max = -Double.MAX_VALUE;
			for (double value : values) {
				if (value > max) {
					max = value;
				}
			}
			double sum = 0.0;
			double[] output = new double[values.length];
			for (int i = 0; i < values.length; i++) {
				output[i] = Math.exp(values[i] - max);
				sum += output[i];
			}
			if (sum <= 0 || Double.isNaN(sum) || Double.isInfinite(sum)) {
				for (int i = 0; i < output.length; i++) {
					output[i] = 1.0 / output.length;
				}
				return output;
			}
			for (int i = 0; i < output.length; i++) {
				output[i] /= sum;
			}
			return output;
		}

		private static double crossEntropy(double[] output, double[] label) {
			double loss = 0.0;
			for (int i = 0; i < output.length; i++) {
				if (label[i] > 0) {
					loss -= label[i] * Math.log(Math.max(1.0E-12, output[i]));
				}
			}
			return loss;
		}

		private void update(double[][] weights, double[] input, double[] delta) {
			for (int i = 0; i < weights.length; i++) {
				for (int j = 0; j < weights[i].length; j++) {
					weights[i][j] -= DDLDO_LEARNING_RATE * input[i] * delta[j];
				}
			}
		}

		private void update(double[] bias, double[] delta) {
			for (int i = 0; i < bias.length; i++) {
				bias[i] -= DDLDO_LEARNING_RATE * delta[i];
			}
		}
	}
	
	//orchestrationHistory.size() = vmList.size()
	
	
	//Comprehensive weighted greedy algorithm
	//评价指标类型一致化，无量纲化，动态加权，综合评价
	private int weightGreedy(String[] architecture, Task task) {
		//获取样本值
		List<Double> disdelay = new ArrayList<>();	//第一列
		List<Double> exedelay = new ArrayList<>();	//第二列
		List<Double> vmnum = new ArrayList<>();	//第三列
		List<Double> energylim = new ArrayList<>();	//第四列
		for (int i = 0; i < orchestrationHistory.size(); i++) {
			//传播延时
			double disdelay_tem = SimulationManager.getdistance(((DataCenter) vmList.get(i).getHost().getDatacenter())
					, task.getEdgeDevice())/simulationParameters.WAN_PROPAGATION_SPEED;
			disdelay.add(disdelay_tem);
			//处理延时
			double exedelay_tem = task.getLength()/vmList.get(i).getMips();
			exedelay.add(exedelay_tem);
			//VM运行的任务数
			vmnum.add((double)orchestrationHistory.get(i).size());
			//vm的能耗
			double energyuse =10*(Math.log10(((DataCenter) vmList.get(i).getHost().getDatacenter()).getEnergyModel().getTotalEnergyConsumption()));
			energylim.add(energyuse);	
		}
		//标准化（归一化）
		List<Double> disdelay_stand = new ArrayList<>();	//第一列
		List<Double> exedelay_stand = new ArrayList<>();	//第二列
		List<Double> vmnum_stand = new ArrayList<>();	//第三列
		List<Double> energylim_stand = new ArrayList<>();	//第四列
		disdelay_stand = standardization(disdelay);
		exedelay_stand = standardization(exedelay);
		vmnum_stand = standardization(vmnum);
		energylim_stand = standardization(energylim);
		
		//加权综合评定
		int vm = -1;
		double min = -1;
		double min_factor;// vm with minimum assigned tasks;
		double a=0.3, b=0.3, c=0.25, d=0.15;
		// get best vm for this task
		for (int i = 0; i < orchestrationHistory.size(); i++) {
			if (offloadingIsPossible(task, vmList.get(i), architecture)) {
				
				min_factor = a*disdelay_stand.get(i) + b*exedelay_stand.get(i) + c*vmnum_stand.get(i) + d*energylim_stand.get(i);
				if (min == -1) { // if it is the first iteration
					min = min_factor;
					// if this is the first time, set the first vm as the
					vm = i; // best one
				} else if (min > min_factor) { // if this vm has more cpu mips and less waiting tasks
					// idle vm, no tasks are waiting
					min = min_factor;
					vm = i;
				}
			}
		}
		// assign the tasks to the found vm
		return vm;
	}
	
	public List<Double> standardization (List<Double> Pre_standar){	//极值差法标准化
		List<Double> standard = new ArrayList<>();
		double premax = Collections.max(Pre_standar);
		double premin = Collections.min(Pre_standar);
		if (premax == premin) {
			for(int k=0; k<Pre_standar.size(); k++) {
				standard.add(0.0);
			}
			return standard;
		}
		for(int k=0; k<Pre_standar.size(); k++) {
			double temp =(Pre_standar.get(k)-premin)/(premax-premin);
			standard.add(temp);
		}
		return standard;
	}

	private int taskClfQLearning(String[] architecture, Task task) {
		MetricSnapshot metrics = collectMetrics(task);
		List<Integer> possibleVms = new ArrayList<>();
		for (int i = 0; i < orchestrationHistory.size(); i++) {
			if (offloadingIsPossible(task, vmList.get(i), architecture)) {
				possibleVms.add(i);
			}
		}
		if (possibleVms.isEmpty()) {
			return -1;
		}

		String state = chapter3State(task, metrics, possibleVms);
		Map<Integer, Double> stateActions = qTable.computeIfAbsent(state, k -> newChapter3ActionValues());
		int action = chooseChapter3Action(stateActions);
		int vm = action == ACTION_LOCAL ? selectChapter3LocalVm(task, possibleVms, metrics)
				: selectChapter3OffloadVm(task, possibleVms, metrics);
		if (vm == -1) {
			vm = selectMinimumChapter3Objective(possibleVms, metrics);
		}
		qLearningDecisions.put(task.getId(), new QLearningDecision(state, action, vm,
				metrics.propagationDelay.get(vm), metrics.executionDelay.get(vm), metrics.propagationDelayStand.get(vm),
				metrics.executionDelayStand.get(vm), metrics.energyStand.get(vm), metrics.parallelTasksStand.get(vm)));
		return vm;
	}

	private String chapter3State(Task task, MetricSnapshot metrics, List<Integer> possibleVms) {
		int bestVm = selectMinimumChapter3Objective(possibleVms, metrics);
		double f = bestVm == -1 ? 0.0 : vmList.get(bestVm).getMips();
		return computeBucket(f, 3) + "|" + latencyBucket(task) + "|" + dataBucket(task);
	}

	private String latencyBucket(Task task) {
		double maxLatency = Math.max(1.0, task.getMaxLatency());
		if (maxLatency <= 10) {
			return "LATENCY_SENSITIVE";
		}
		if (maxLatency <= 60) {
			return "BALANCED_LATENCY";
		}
		return "LATENCY_TOLERANT";
	}

	private String dataBucket(Task task) {
		double dataSize = task.getFileSize() + task.getOutputSize();
		if (dataSize <= 1024) {
			return "SMALL_DATA";
		}
		if (dataSize <= 10000) {
			return "MEDIUM_DATA";
		}
		return "LARGE_DATA";
	}

	private double latencySensitivity(Task task) {
		double maxLatency = Math.max(1.0, task.getMaxLatency());
		return Math.min(1.0, 10.0 / maxLatency);
	}

	private double dataIntensity(Task task) {
		double dataSize = task.getFileSize() + task.getOutputSize();
		return Math.min(1.0, dataSize / 10000.0);
	}

	private Map<Integer, Double> newChapter3ActionValues() {
		Map<Integer, Double> actions = new HashMap<>();
		actions.put(ACTION_LOCAL, smallInitialQValue());
		actions.put(ACTION_OFFLOAD, smallInitialQValue());
		return actions;
	}

	private double smallInitialQValue() {
		return (simulationParameters.RANDOM_GENERATOR.nextDouble() - 0.5) * Q_INITIAL_SPREAD;
	}

	private int chooseChapter3Action(Map<Integer, Double> stateActions) {
		if (simulationParameters.RANDOM_GENERATOR.nextDouble() < Q_EPSILON) {
			return randomChapter3Action();
		}
		return bestChapter3Action(stateActions);
	}

	private int randomChapter3Action() {
		return simulationParameters.RANDOM_GENERATOR.nextBoolean() ? ACTION_OFFLOAD : ACTION_LOCAL;
	}

	private int bestChapter3Action(Map<Integer, Double> stateActions) {
		double local = stateActions.getOrDefault(ACTION_LOCAL, smallInitialQValue());
		double offload = stateActions.getOrDefault(ACTION_OFFLOAD, smallInitialQValue());
		if (Math.abs(local - offload) <= 1.0E-12) {
			return randomChapter3Action();
		}
		return offload > local ? ACTION_OFFLOAD : ACTION_LOCAL;
	}

	private double maxFutureQ(String state) {
		Map<Integer, Double> actions = qTable.computeIfAbsent(state, k -> newChapter3ActionValues());
		return Collections.max(actions.values());
	}

	private double qLearningReward(Task task, QLearningDecision decision) {
		double reward;
		if (decision.action == ACTION_LOCAL) {
			double eteDelay = Math.max(REWARD_EPSILON, decision.propagationDelay + decision.executionDelay);
			double executionDelay = Math.max(REWARD_EPSILON, decision.executionDelay);
			reward = CHAPTER3_LOCAL_ETE_WEIGHT / eteDelay + CHAPTER3_LOCAL_EXE_WEIGHT / executionDelay;
		} else {
			reward = -chapter3Objective(decision.propagationDelayStand, decision.executionDelayStand,
					decision.energyStand, decision.parallelTasksStand);
		}
		if (task.getFailureReason() == Task.Status.FAILED_DUE_TO_LATENCY) {
			reward -= LATENCY_FAILURE_PENALTY;
		} else if (task.getFailureReason() == Task.Status.FAILED_DUE_TO_DEVICE_MOBILITY) {
			reward -= MOBILITY_FAILURE_PENALTY;
		} else if (task.getFailureReason() != null && task.getFailureReason() != Task.Status.NULL) {
			reward -= 0.5;
		}
		return reward;
	}

	private int ddldo(String[] architecture, Task task) {
		initializeDdldo();
		MetricSnapshot metrics = collectMetrics(task);
		List<Integer> possibleVms = possibleVms(architecture, task);
		if (possibleVms.isEmpty()) {
			return -1;
		}

		double[] input = ddldoInput(task);
		List<DdldoDecision> candidates = ddldoCandidateDecisions(task, input, possibleVms, metrics);
		DdldoDecision selected = selectBestDdldoDecision(candidates);
		if (selected == null) {
			selected = fallbackDdldoDecision(task, possibleVms, metrics);
		}
		if (selected == null || selected.vmIndex == -1) {
			return -1;
		}
		attachCandidateSummary(selected, candidates);
		rememberDdldo(input, selected);
		ddldoPendingDecisions.put(task.getId(), selected);
		double loss = trainDdldo();
		appendDdldoTrainingLog(loss, selected);
		return selected.vmIndex;
	}

	private MetricSnapshot collectMetrics(Task task) {
		MetricSnapshot metrics = new MetricSnapshot();
		for (int i = 0; i < orchestrationHistory.size(); i++) {
			DataCenter destination = (DataCenter) vmList.get(i).getHost().getDatacenter();
			double distance = SimulationManager.getdistance(destination, task.getEdgeDevice());
			double executionDelay = taskExecutionDelay(task, i);
			metrics.propagationDelay.add(taskPropagationDelay(task, destination, distance) + taskTransferDelay(task, destination));
			metrics.executionDelay.add(executionDelay);
			metrics.parallelTasks.add((double) orchestrationHistory.get(i).size());
			double energy = Math.max(1.0, estimatedTaskEnergy(task, i, distance, executionDelay));
			metrics.energyRaw.add(energy);
			metrics.energy.add(10.0 * Math.log10(energy));
		}
		metrics.propagationDelayStand.addAll(standardization(metrics.propagationDelay));
		metrics.executionDelayStand.addAll(standardization(metrics.executionDelay));
		metrics.energyStand.addAll(standardization(metrics.energy));
		metrics.parallelTasksStand.addAll(standardization(metrics.parallelTasks));
		return metrics;
	}

	private List<Integer> possibleVms(String[] architecture, Task task) {
		List<Integer> possible = new ArrayList<>();
		for (int i = 0; i < orchestrationHistory.size(); i++) {
			if (offloadingIsPossible(task, vmList.get(i), architecture)) {
				possible.add(i);
			}
		}
		return possible;
	}

	private double taskExecutionDelay(Task task, int vm) {
		return task.getLength() / Math.max(1.0, vmList.get(vm).getMips());
	}

	private double taskPropagationDelay(Task task, DataCenter destination, double distance) {
		if (destination == task.getEdgeDevice()) {
			return 0.0;
		}
		return distance / simulationParameters.WAN_PROPAGATION_SPEED;
	}

	private double taskTransferDelay(Task task, DataCenter destination) {
		if (destination == task.getEdgeDevice()) {
			return 0.0;
		}
		double dataKbits = Math.max(0.0, task.getFileSize() + task.getOutputSize()) * 8.0;
		return dataKbits / Math.max(1.0, bandwidthFor(destination));
	}

	private double bandwidthFor(DataCenter destination) {
		if (destination.getType() == simulationParameters.TYPES.CLOUD) {
			return Math.min(simulationParameters.WAN_BANDWIDTH, simulationParameters.BANDWIDTH_WLAN);
		}
		return simulationParameters.BANDWIDTH_WLAN;
	}

	private double estimatedTaskEnergy(Task task, int vm, double distance, double executionDelay) {
		DataCenter destination = (DataCenter) vmList.get(vm).getHost().getDatacenter();
		double cpuEnergy = destination.getEnergyModel().getMaxActiveConsumption() * executionDelay;
		if (destination == task.getEdgeDevice()) {
			return Math.max(1.0, task.getLength() * Math.pow(Math.max(1.0, vmList.get(vm).getMips()), 2.0));
		}
		double inputBits = Math.max(1.0, task.getFileSize() * 8000.0);
		double outputBits = Math.max(1.0, task.getOutputSize() * 8000.0);
		return transmissionEnergy(inputBits, distance) + receptionEnergy(inputBits + outputBits) + cpuEnergy;
	}

	private double transmissionEnergy(double bits, double distance) {
		double threshold = Math.sqrt(simulationParameters.AMPLIFIER_DISSIPATION_FREE_SPACE
				/ simulationParameters.AMPLIFIER_DISSIPATION_MULTIPATH);
		double amplifier = distance <= threshold
				? simulationParameters.AMPLIFIER_DISSIPATION_FREE_SPACE * Math.pow(distance, 2.0)
				: simulationParameters.AMPLIFIER_DISSIPATION_MULTIPATH * Math.pow(distance, 4.0);
		return bits * (simulationParameters.CONSUMED_ENERGY_PER_BIT + amplifier);
	}

	private double receptionEnergy(double bits) {
		return bits * simulationParameters.CONSUMED_ENERGY_PER_BIT;
	}

	private String[] currentArchitecture() {
		if ("CLOUD_ONLY".equals(architecture)) {
			return new String[] { "Cloud" };
		}
		if ("MIST_ONLY".equals(architecture)) {
			return new String[] { "Mist" };
		}
		if ("EDGE_AND_CLOUD".equals(architecture)) {
			return new String[] { "Cloud", "Edge" };
		}
		if ("EDGE_ONLY".equals(architecture)) {
			return new String[] { "Edge" };
		}
		if ("MIST_AND_CLOUD".equals(architecture)) {
			return new String[] { "Cloud", "Mist" };
		}
		if ("MIST_AND_EDGE".equals(architecture)) {
			return new String[] { "Edge", "Mist" };
		}
		return new String[] { "Cloud", "Edge", "Mist" };
	}

	private int selectChapter3LocalVm(Task task, List<Integer> possibleVms, MetricSnapshot metrics) {
		for (int vm : possibleVms) {
			if (vmList.get(vm).getHost().getDatacenter() == task.getEdgeDevice()) {
				return vm;
			}
		}
		return selectBestVmByType(possibleVms, metrics, simulationParameters.TYPES.EDGE_DEVICE);
	}

	private int selectChapter3OffloadVm(Task task, List<Integer> possibleVms, MetricSnapshot metrics) {
		if (latencySensitivity(task) >= 0.5) {
			return selectBestVmByType(possibleVms, metrics, simulationParameters.TYPES.EDGE_DEVICE);
		}
		if (dataIntensity(task) >= 0.5) {
			int cloud = selectBestVmByType(possibleVms, metrics, simulationParameters.TYPES.CLOUD);
			if (cloud != -1) {
				return cloud;
			}
		}
		return selectMinimumChapter3Objective(possibleVms, metrics);
	}

	private int selectBestVmByType(List<Integer> possibleVms, MetricSnapshot metrics, simulationParameters.TYPES type) {
		int bestVm = -1;
		double bestValue = Double.MAX_VALUE;
		for (int vm : possibleVms) {
			DataCenter destination = (DataCenter) vmList.get(vm).getHost().getDatacenter();
			if (destination.getType() == type) {
				double value = chapter3Objective(metrics.propagationDelayStand.get(vm), metrics.executionDelayStand.get(vm),
						metrics.energyStand.get(vm), metrics.parallelTasksStand.get(vm));
				if (value < bestValue) {
					bestValue = value;
					bestVm = vm;
				}
			}
		}
		return bestVm;
	}

	private int selectMinimumChapter3Objective(List<Integer> possibleVms, MetricSnapshot metrics) {
		int bestVm = -1;
		double bestValue = Double.MAX_VALUE;
		for (int vm : possibleVms) {
			double value = chapter3Objective(metrics.propagationDelayStand.get(vm), metrics.executionDelayStand.get(vm),
					metrics.energyStand.get(vm), metrics.parallelTasksStand.get(vm));
			if (value < bestValue) {
				bestValue = value;
				bestVm = vm;
			}
		}
		return bestVm;
	}

	private double chapter3Objective(double propagationDelay, double executionDelay, double energy, double parallelTasks) {
		return CHAPTER3_WEIGHTS[0] * propagationDelay + CHAPTER3_WEIGHTS[1] * executionDelay
				+ CHAPTER3_WEIGHTS[2] * energy + CHAPTER3_WEIGHTS[3] * parallelTasks;
	}

	private int computeBucket(double value, int buckets) {
		if (value <= 0) {
			return 0;
		}
		double scaled = Math.log10(value + 1.0);
		return Math.min(buckets - 1, Math.max(0, (int) Math.floor(scaled)));
	}

	private void initializeDdldo() {
		if (ddldoInitialized) {
			return;
		}
		for (int i = 0; i < DDLDO_DNN_COUNT; i++) {
			ddldoNetworks.add(new SimpleDnn());
		}
		ddldoMemory.clear();
		ddldoEpochsTrained = 0;
		ddldoInitialized = true;
	}

	private double[] ddldoInput(Task task) {
		return new double[] {
				normalize(task.getFileSize(), 100000.0),
				normalize(task.getOutputSize(), 100000.0),
				latencySensitivity(task),
				normalize(simulationParameters.VEHICLE_SPEED, 120.0)
		};
	}

	private double normalize(double value, double max) {
		return Math.max(0.0, Math.min(1.0, value / Math.max(1.0, max)));
	}

	private int argMax(double[] values) {
		int best = 0;
		for (int i = 1; i < values.length; i++) {
			if (values[i] > values[best]) {
				best = i;
			}
		}
		return best;
	}

	private List<DdldoDecision> ddldoCandidateDecisions(Task task, double[] input, List<Integer> possibleVms,
			MetricSnapshot metrics) {
		List<DdldoDecision> candidates = new ArrayList<>();
		List<Integer> usedActions = new ArrayList<>();
		for (SimpleDnn network : ddldoNetworks) {
			double[] scores = adjustedDdldoScores(task, network.predict(input));
			DdldoDecision decision = bestDecisionFromScores(task, scores, possibleVms, metrics, usedActions);
			if (decision != null) {
				candidates.add(decision);
				usedActions.add(decision.action);
			}
		}
		if (candidates.isEmpty()) {
			DdldoDecision fallback = fallbackDdldoDecision(task, possibleVms, metrics);
			if (fallback != null) {
				candidates.add(fallback);
			}
		}
		return candidates;
	}

	private double[] adjustedDdldoScores(Task task, double[] scores) {
		double[] adjusted = scores.clone();
		double latency = latencySensitivity(task);
		double data = dataIntensity(task);
		adjusted[0] += 0.20 * latency;
		adjusted[1] += 0.18 * latency;
		adjusted[2] += 0.08 * (1.0 - latency);
		adjusted[3] += 0.12 * data;
		return adjusted;
	}

	private DdldoDecision bestDecisionFromScores(Task task, double[] scores, List<Integer> possibleVms,
			MetricSnapshot metrics, List<Integer> usedActions) {
		boolean[] visited = new boolean[scores.length];
		for (int step = 0; step < scores.length; step++) {
			int action = -1;
			double bestScore = -Double.MAX_VALUE;
			for (int i = 0; i < scores.length; i++) {
				if (!visited[i] && scores[i] > bestScore && (!usedActions.contains(i) || usedActions.size() >= DDLDO_DNN_COUNT)) {
					bestScore = scores[i];
					action = i;
				}
			}
			if (action == -1) {
				return null;
			}
			visited[action] = true;
			DdldoDecision decision = selectDecisionForAction(task, action, possibleVms, metrics, bestScore);
			if (decision != null) {
				return decision;
			}
		}
		return null;
	}

	private DdldoDecision selectDecisionForAction(Task task, int action, List<Integer> possibleVms,
			MetricSnapshot metrics, double actionScore) {
		DdldoDecision bestFeasible = null;
		DdldoDecision bestAny = null;
		double bestFeasibleScore = Double.MAX_VALUE;
		double bestAnyScore = Double.MAX_VALUE;
		for (int vm : possibleVms) {
			if (ddldoActionForVm(task, vm) != action) {
				continue;
			}
			DdldoDecision decision = evaluateDdldoDecision(task, vm, metrics);
			double score = decision.objective - 0.05 * actionScore;
			if (decision.feasible && score < bestFeasibleScore) {
				bestFeasibleScore = score;
				bestFeasible = decision;
			}
			if (score < bestAnyScore) {
				bestAnyScore = score;
				bestAny = decision;
			}
		}
		return bestFeasible != null ? bestFeasible : bestAny;
	}

	private DdldoDecision selectBestDdldoDecision(List<DdldoDecision> candidates) {
		DdldoDecision best = null;
		for (DdldoDecision decision : candidates) {
			if (!decision.feasible) {
				continue;
			}
			if (best == null || decision.objective < best.objective) {
				best = decision;
			}
		}
		return best;
	}

	private DdldoDecision fallbackDdldoDecision(Task task, List<Integer> possibleVms, MetricSnapshot metrics) {
		DdldoDecision bestDeadline = null;
		DdldoDecision bestAny = null;
		for (int vm : possibleVms) {
			DdldoDecision decision = evaluateDdldoDecision(task, vm, metrics);
			if (bestAny == null || decision.objective < bestAny.objective) {
				bestAny = decision;
			}
			if (decision.latencyFeasible && decision.linkFeasible && decision.resourceFeasible
					&& (bestDeadline == null || ddldoFallbackScore(task, decision) < ddldoFallbackScore(task, bestDeadline))) {
				bestDeadline = decision;
			}
		}
		DdldoDecision selected = bestDeadline != null ? bestDeadline : bestAny;
		if (selected != null && !selected.feasible && selected.failureReason.length() == 0) {
			selected.failureReason = "fallback_infeasible";
		}
		return selected;
	}

	private double ddldoFallbackScore(Task task, DdldoDecision decision) {
		DataCenter destination = (DataCenter) vmList.get(decision.vmIndex).getHost().getDatacenter();
		double distance = SimulationManager.getdistance(destination, task.getEdgeDevice());
		double localityBonus = destination == task.getEdgeDevice() ? -1.0 : 0.0;
		return distance / Math.max(1.0, simulationParameters.CLOUD_RANGE) + decision.objective + localityBonus;
	}

	private DdldoDecision evaluateDdldoDecision(Task task, int vm, MetricSnapshot metrics) {
		DataCenter destination = (DataCenter) vmList.get(vm).getHost().getDatacenter();
		int action = ddldoActionForVm(task, vm);
		double delay = metrics.propagationDelay.get(vm) + metrics.executionDelay.get(vm);
		double energy = metrics.energyRaw.get(vm);
		double coverageRemainingTime = coverageRemainingTime(task, destination, delay);
		double estimatedFinishTime = simulationManager.getSimulation().clock() + delay;
		boolean coverageFeasible = coverageRemainingTime < 0 || delay <= coverageRemainingTime;
		boolean latencyFeasible = delay <= task.getMaxLatency();
		boolean energyFeasible = energyFeasible(destination, energy);
		boolean linkFeasible = destination == task.getEdgeDevice() || SimulationManager.issetlink(task.getEdgeDevice(), destination);
		boolean resourceFeasible = !destination.isDead() && vmList.get(vm).getMips() > 0;
		String failureReason = ddldoFailureReason(coverageFeasible, latencyFeasible, energyFeasible, linkFeasible,
				resourceFeasible);
		double objective = chapter4Objective(vm, metrics);
		if (!latencyFeasible) {
			objective += 2.0 + (delay - task.getMaxLatency()) / Math.max(1.0, task.getMaxLatency());
		}
		if (!coverageFeasible) {
			objective += 1.5;
		}
		if (!energyFeasible) {
			objective += 1.0;
		}
		if (!linkFeasible || !resourceFeasible) {
			objective += 3.0;
		}
		return new DdldoDecision(action, vm, targetId(destination), objective, delay, energy, coverageRemainingTime,
				estimatedFinishTime, coverageFeasible, latencyFeasible, energyFeasible, linkFeasible, resourceFeasible,
				failureReason);
	}

	private String ddldoFailureReason(boolean coverageFeasible, boolean latencyFeasible, boolean energyFeasible,
			boolean linkFeasible, boolean resourceFeasible) {
		List<String> reasons = new ArrayList<>();
		if (!coverageFeasible) {
			reasons.add("coverage");
		}
		if (!latencyFeasible) {
			reasons.add("latency");
		}
		if (!energyFeasible) {
			reasons.add("energy");
		}
		if (!linkFeasible) {
			reasons.add("link");
		}
		if (!resourceFeasible) {
			reasons.add("resource");
		}
		return String.join("|", reasons);
	}

	private boolean energyFeasible(DataCenter destination, double energy) {
		if (Double.isNaN(energy) || Double.isInfinite(energy)) {
			return false;
		}
		return energy >= 0.0;
	}

	private double coverageRemainingTime(Task task, DataCenter destination, double estimatedTaskDelay) {
		if (destination == null || destination == task.getEdgeDevice()
				|| destination.getType() == simulationParameters.TYPES.CLOUD) {
			return -1.0;
		}
		double radius = simulationParameters.EARTH_RADIUS;
		double height = Math.max(0.0, SimulationManager.getHight(destination) - radius);
		if (height <= 0 || simulationParameters.SATELLITE_SPEED <= 0) {
			return -1.0;
		}
		double theta = elevationAngle(task.getEdgeDevice(), destination);
		if (theta < 0 || Double.isNaN(theta) || Double.isInfinite(theta)) {
			return 0.0;
		}
		double ratio = radius / (radius + height);
		double acosArg = clamp(ratio * Math.cos(theta), -1.0, 1.0);
		double gamma = Math.acos(acosArg) - theta;
		if (gamma <= 0 || Double.isNaN(gamma) || Double.isInfinite(gamma)) {
			return 0.0;
		}
		double coverage = 2.0 * (radius + height) * gamma / simulationParameters.SATELLITE_SPEED;
		double locationRemaining = Math.max(0.0,
				simulationParameters.LOCATIONTIMENUM - simulationManager.getSimulation().clock());
		if (locationRemaining > 0) {
			coverage = Math.min(coverage, locationRemaining);
		}
		return Math.max(0.0, coverage - Math.max(0.0, estimatedTaskDelay * 0.05));
	}

	private double elevationAngle(DataCenter source, DataCenter satellite) {
		double sx = source.getLocation().getXPos();
		double sy = source.getLocation().getYPos();
		double sz = source.getLocation().getZPos();
		double dx = satellite.getLocation().getXPos() - sx;
		double dy = satellite.getLocation().getYPos() - sy;
		double dz = satellite.getLocation().getZPos() - sz;
		double sourceNorm = Math.sqrt(sx * sx + sy * sy + sz * sz);
		double linkNorm = Math.sqrt(dx * dx + dy * dy + dz * dz);
		if (sourceNorm <= 0 || linkNorm <= 0) {
			return -1.0;
		}
		double sinElevation = clamp((dx * sx + dy * sy + dz * sz) / (sourceNorm * linkNorm), -1.0, 1.0);
		return Math.asin(sinElevation);
	}

	private double clamp(double value, double min, double max) {
		return Math.max(min, Math.min(max, value));
	}

	private int targetId(DataCenter destination) {
		return destination == null ? -1 : destination.getDeviceID();
	}

	private double chapter4Objective(int vm, MetricSnapshot metrics) {
		double delay = metrics.propagationDelayStand.get(vm) + metrics.executionDelayStand.get(vm);
		double energy = metrics.energyStand.get(vm);
		return CHAPTER4_DELAY_WEIGHT * delay + CHAPTER4_ENERGY_WEIGHT * energy;
	}

	private int ddldoActionForVm(Task task, int vm) {
		DataCenter destination = (DataCenter) vmList.get(vm).getHost().getDatacenter();
		if (destination == null) {
			return 0;
		}
		if (destination == task.getEdgeDevice()) {
			return 0;
		}
		if (destination.getType() == simulationParameters.TYPES.EDGE_DEVICE) {
			return 1;
		}
		if (destination.getType() == simulationParameters.TYPES.EDGE_DATACENTER) {
			return 2;
		}
		if (destination.getType() == simulationParameters.TYPES.CLOUD) {
			return 3;
		}
		return 0;
	}

	private double[] oneHot(int action) {
		double[] label = new double[DDLDO_ACTIONS];
		if (action >= 0 && action < label.length) {
			label[action] = 1.0;
		}
		return label;
	}

	private void attachCandidateSummary(DdldoDecision selected, List<DdldoDecision> candidates) {
		List<String> actions = new ArrayList<>();
		List<String> vmIds = new ArrayList<>();
		for (DdldoDecision candidate : candidates) {
			actions.add(Integer.toString(candidate.action));
			vmIds.add(Integer.toString(candidate.targetId));
		}
		selected.candidateCount = candidates.size();
		selected.candidateActions = String.join("|", actions);
		selected.candidateVmIds = String.join("|", vmIds);
	}

	private void rememberDdldo(double[] input, DdldoDecision decision) {
		if (ddldoMemory.size() >= DDLDO_MEMORY_SIZE) {
			ddldoMemory.removeFirst();
		}
		ddldoMemory.addLast(new DdldoSample(input, oneHot(decision.action), decision));
	}

	private double trainDdldo() {
		if (ddldoMemory.isEmpty() || ddldoEpochsTrained >= DDLDO_EPOCHS) {
			return 0.0;
		}
		List<DdldoSample> memory = new ArrayList<>(ddldoMemory);
		double totalLoss = 0.0;
		int trainedNetworks = 0;
		for (int epoch = 0; epoch < DDLDO_EPOCHS_PER_STEP && ddldoEpochsTrained < DDLDO_EPOCHS; epoch++) {
			for (SimpleDnn network : ddldoNetworks) {
				totalLoss += network.train(randomDdldoBatch(memory));
				trainedNetworks++;
			}
			ddldoEpochsTrained++;
		}
		return trainedNetworks == 0 ? 0.0 : totalLoss / trainedNetworks;
	}

	private List<DdldoSample> randomDdldoBatch(List<DdldoSample> memory) {
		List<DdldoSample> batch = new ArrayList<>();
		for (int i = 0; i < Math.min(DDLDO_BATCH_SIZE, memory.size()); i++) {
			batch.add(memory.get(simulationParameters.RANDOM_GENERATOR.nextInt(memory.size())));
		}
		return batch;
	}

	private void appendDdldoDecisionLog(Task task, DdldoDecision decision, boolean success, String failureReason) {
		String header = "time,taskId,deviceId,taskLength,taskFileSize,delaySensitivity,vehicleSpeed,sourceLeoId,"
				+ "candidateCount,candidateActions,candidateVmIds,selectedAction,selectedVmId,objective,delay,energy,"
				+ "coverageRemainingTime,estimatedFinishTime,coverageFeasible,latencyFeasible,energyFeasible,success,failureReason";
		String line = joinCsv(
				csvDouble(simulationManager.getSimulation().clock()),
				Long.toString(task.getId()),
				Integer.toString(targetId(task.getEdgeDevice())),
				Long.toString(task.getLength()),
				Long.toString(task.getFileSize()),
				csvDouble(latencySensitivity(task)),
				csvDouble(simulationParameters.VEHICLE_SPEED),
				Integer.toString(targetId(task.getEdgeDevice())),
				Integer.toString(decision.candidateCount),
				decision.candidateActions,
				decision.candidateVmIds,
				Integer.toString(decision.action),
				Integer.toString(decision.targetId),
				csvDouble(decision.objective),
				csvDouble(decision.delay),
				csvDouble(decision.energy),
				csvDouble(decision.coverageRemainingTime),
				csvDouble(decision.estimatedFinishTime),
				Boolean.toString(decision.coverageFeasible),
				Boolean.toString(decision.latencyFeasible),
				Boolean.toString(decision.energyFeasible),
				Boolean.toString(success),
				failureReason == null ? "" : failureReason);
		appendCsvLine(ddldoLogPath("chapter4_ddldo_decisions.csv"), header, line);
	}

	private void appendDdldoTrainingLog(double loss, DdldoDecision decision) {
		String header = "time,episodeOrStep,memorySize,batchSize,learningRate,loss,avgObjective,selectedAction,selectedVmId";
		String line = joinCsv(
				csvDouble(simulationManager.getSimulation().clock()),
				Integer.toString(ddldoEpochsTrained),
				Integer.toString(ddldoMemory.size()),
				Integer.toString(DDLDO_BATCH_SIZE),
				csvDouble(DDLDO_LEARNING_RATE),
				csvDouble(loss),
				csvDouble(averageDdldoObjective()),
				Integer.toString(decision.action),
				Integer.toString(decision.targetId));
		appendCsvLine(ddldoLogPath("chapter4_ddldo_training.csv"), header, line);
	}

	private double averageDdldoObjective() {
		if (ddldoMemory.isEmpty()) {
			return 0.0;
		}
		double sum = 0.0;
		for (DdldoSample sample : ddldoMemory) {
			sum += sample.objective;
		}
		return sum / ddldoMemory.size();
	}

	private String ddldoLogPath(String fileName) {
		String base = simLog.getFileName(".csv");
		int slash = Math.max(base.lastIndexOf('/'), base.lastIndexOf('\\'));
		if (slash == -1) {
			return fileName;
		}
		return base.substring(0, slash + 1) + fileName;
	}

	private void appendCsvLine(String fileName, String header, String line) {
		try {
			File file = new File(fileName);
			boolean writeHeader = !file.exists() || file.length() == 0;
			try (BufferedWriter writer = new BufferedWriter(new FileWriter(file, true))) {
				if (writeHeader) {
					writer.write(header);
					writer.newLine();
				}
				writer.write(line);
				writer.newLine();
			}
		} catch (IOException e) {
			SimLog.println("DefaultEdgeOrchestrator- Warning: failed to write DDLDO log " + fileName + ": "
					+ e.getMessage());
		}
	}

	private String joinCsv(String... values) {
		return String.join(",", values);
	}

	private String csvDouble(double value) {
		if (Double.isNaN(value) || Double.isInfinite(value)) {
			return "";
		}
		return Double.toString(value);
	}

	
	
	private int tradeOff(String[] architecture, Task task) {
		int vm = -1;
		double min = -1;
		double new_min;// vm with minimum assigned tasks;

		// get best vm for this task
		for (int i = 0; i < orchestrationHistory.size(); i++) {
			if (offloadingIsPossible(task, vmList.get(i), architecture)) {
				double latency = 1;
				double energy = 1;
				if (((DataCenter) vmList.get(i).getHost().getDatacenter())
						.getType() == simulationParameters.TYPES.CLOUD) {
					latency = 1.6;
					energy = 1.1;
				} else if (((DataCenter) vmList.get(i).getHost().getDatacenter())
						.getType() == simulationParameters.TYPES.EDGE_DEVICE) {
					energy = 1.4;
				}
				new_min = (orchestrationHistory.get(i).size() + 1) * latency * energy * task.getLength() / vmList.get(i).getMips();
				if (min == -1) { // if it is the first iteration
					min = new_min;
					// if this is the first time, set the first vm as the
					vm = i; // best one
				} else if (min > new_min) { // if this vm has more cpu mips and less waiting tasks
					// idle vm, no tasks are waiting
					min = new_min;
					vm = i;
				}
			}
		}
		// assign the tasks to the found vm
		return vm;
	}
	
	
	//执行任务最少的vm
	private int roundRobin(String[] architecture, Task task) {
		List<Vm> vmList = simulationManager.getServersManager().getVmList();
		int vm = -1;
		int minTasksCount = -1; // vm with minimum assigned tasks;
		// get best vm for this task
		for (int i = 0; i < orchestrationHistory.size(); i++) {
			if (offloadingIsPossible(task, vmList.get(i), architecture)) {
				if (minTasksCount == -1) {
					minTasksCount = orchestrationHistory.get(i).size();
					// if this is the first time, set the first vm as the best one
					vm = i;
				} else if (minTasksCount > orchestrationHistory.get(i).size()) {
					minTasksCount = orchestrationHistory.get(i).size();
					// new min found, so we choose it as the best VM
					vm = i;
					break;
				}
			}
		}
		// assign the tasks to the found vm
		return vm;
	}
	
	//轮询算法虚拟机编号轮询只考虑能否建链不考虑资源情况
	private int TradiPolling(String[] architecture, Task task) {
		List<Vm> vmList = simulationManager.getServersManager().getVmList();
		List<Task> tasksList = simulationManager.getTasksList();
		List<Integer>  minfindvmid = new ArrayList<>();		//记录可以调度的虚拟机
		boolean flag = false;
		//当轮询变量比虚拟机编号大的时候，取余
		if(FindVmId_TP> vmList.size()-1) {
			FindVmId_TP = (FindVmId_TP+1) % vmList.size();
		}
		// get best vm for this task
		for (int i = 0; i < orchestrationHistory.size(); i++) {
			if (offloadingIsPossible(task, vmList.get(i), architecture)) {
				flag = true;
				minfindvmid.add(i); //把可以进行调度的虚拟机先记下来
				if(FindVmId_TP <=i) {	//遇到与FindVmId_TP相等或者大的Id结束循环
					FindVmId_TP = i;
					break;
				}
			} else{ 
				if((i == vmList.size()-1) && flag) {	//当FindVmId_TP过大且比它大的vm没有符合的
					FindVmId_TP = minfindvmid.get(0);	//FindVmId_TP取第一个满足的
					flag = false;
					
				}
			}
		}
		// assign the tasks to the found vm
		int vm = FindVmId_TP;
		FindVmId_TP++;
		Counttask++;
		//System.out.println("task "+task.getId() + "vm id is: " + vm+". taskcount: "+Counttask);
		if(Counttask == tasksList.size()) {
			FindVmId_TP=0;
			Counttask=0;
		}
		return vm;
	}
	
	
	//随机一个vm
	private int RandomVm(String[] architecture, Task task) {
		List<Vm> vmList = simulationManager.getServersManager().getVmList();
		int vm = -1;
		int RandomCount = 0; // random time;
		// get random vm for this task
		while(RandomCount<orchestrationHistory.size()) {
			int index = simulationParameters.RANDOM_GENERATOR.nextInt(orchestrationHistory.size());
			if (offloadingIsPossible(task, vmList.get(index), architecture)) {
				vm = index;
				break;
			}
			RandomCount++;
		}
		
		//万一没有随机出来
		if(RandomCount>=orchestrationHistory.size()) {
			for (int i = 0; i < orchestrationHistory.size(); i++) {
				if (offloadingIsPossible(task, vmList.get(i), architecture)) {
					vm =i;
					break;
				}
			}
			RandomCount = 0;
		}
		// assign the tasks to the found vm
		return vm;
	}
	

	@Override
	public void resultsReturned(Task task) { 
		if ("DDLDO".equals(algorithm)) {
			DdldoDecision decision = ddldoPendingDecisions.remove(task.getId());
			if (decision != null) {
				boolean success = task.getFailureReason() == null || task.getFailureReason() == Task.Status.NULL;
				String failureReason = success ? "" : task.getFailureReason().name();
				appendDdldoDecisionLog(task, decision, success, failureReason);
			}
			return;
		}
		if (!(TASK_CLF_Q_LEARNING.equals(algorithm) || "TaskClfQLearning".equals(algorithm))) {
			return;
		}
		QLearningDecision decision = qLearningDecisions.remove(task.getId());
		if (decision == null) {
			return;
		}
		Map<Integer, Double> stateActions = qTable.computeIfAbsent(decision.state, k -> newChapter3ActionValues());
		double oldValue = stateActions.getOrDefault(decision.action, 0.0);
		double reward = qLearningReward(task, decision);
		String nextState = decision.state;
		MetricSnapshot metrics = collectMetrics(task);
		List<Integer> possibleVms = possibleVms(currentArchitecture(), task);
		if (!possibleVms.isEmpty()) {
			nextState = chapter3State(task, metrics, possibleVms);
		}
		double newValue = oldValue + Q_LEARNING_RATE
				* (reward + Q_DISCOUNT_FACTOR * maxFutureQ(nextState) - oldValue);
		stateActions.put(decision.action, newValue);
	}

}
