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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

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
	private static final int DDLDO_OUTPUTS = 1;
	private static final int DDLDO_INPUTS = 20;
	private static final int DDLDO_DNN_COUNT = 3;
	private static final int DDLDO_HIDDEN_1 = 64;
	private static final int DDLDO_HIDDEN_2 = 128;
	private static final int DDLDO_HIDDEN_3 = 64;
	private static final int DDLDO_BATCH_SIZE = 256;
	private static final int DDLDO_MEMORY_SIZE = 4096;
	private static final int DDLDO_EPOCHS = 1000;
	private static final int DDLDO_EPOCHS_PER_STEP = 1;
	private static final double DDLDO_LEARNING_RATE = 0.0001;
	private static final double DDLDO_BASE_VEHICLE_SPEED = 60.0;
	private static final String[] DDLDO_CUDA_RUNTIME_DLLS = { "cudart64_110.dll", "cublas64_11.dll",
			"cudnn64_8.dll" };
	private final Map<String, Map<Integer, Double>> qTable = new HashMap<>();
	private final Map<Long, QLearningDecision> qLearningDecisions = new HashMap<>();
	private final Deque<DdldoSample> ddldoMemory = new ArrayDeque<>();
	private final Map<Long, DdldoDecision> ddldoPendingDecisions = new HashMap<>();
	private final List<SimpleDnn> ddldoNetworks = new ArrayList<>();
	private boolean ddldoInitialized = false;
	private boolean ddldoBackendLogged = false;
	private boolean ddldoUsingGpu = false;
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
		private final double actualDelay;
		private final double estimatedEnergy;
		private final double actualEnergy;
		private final boolean actualEnergyAvailable;
		private final String delayFeedbackType;
		private final String energyFeedbackType;
		private final double observedCost;
		private final double finalTrainingCost;
		private final boolean success;
		private final String failureReason;

		private DdldoSample(double[] input, double label, DdldoDecision decision) {
			this.input = input;
			this.label = new double[] { label };
			this.action = decision.action;
			this.vmIndex = decision.vmIndex;
			this.objective = decision.objective;
			this.delay = decision.delay;
			this.energy = decision.energy;
			this.actualDelay = decision.actualDelay;
			this.estimatedEnergy = decision.energy;
			this.actualEnergy = decision.actualEnergy;
			this.actualEnergyAvailable = decision.actualEnergyAvailable;
			this.delayFeedbackType = decision.delayFeedbackType;
			this.energyFeedbackType = decision.energyFeedbackType;
			this.observedCost = decision.observedCost;
			this.finalTrainingCost = decision.finalTrainingCost;
			this.success = decision.actualSuccess;
			this.failureReason = decision.failureReason;
		}
	}

	private static class DdldoDecision {
		private final int action;
		private final int vmIndex;
		private final int targetId;
		private final double[] input;
		private double objective;
		private double dnnScore;
		private double finalCost;
		private final double delay;
		private final double energy;
		private final double currentDistance;
		private final double projectedDistance;
		private final double propagationDelay;
		private final double transmissionDelay;
		private final double executionDelay;
		private final double vmLoad;
		private final double coverageRemainingTime;
		private final double estimatedFinishTime;
		private final boolean mobilityRisk;
		private final boolean delayRisk;
		private final boolean offloadingPossible;
		private final boolean coverageFeasible;
		private final boolean latencyFeasible;
		private final boolean energyFeasible;
		private final boolean linkFeasible;
		private final boolean resourceFeasible;
		private final boolean feasible;
		private final double energyBefore;
		private double actualDelay = -1.0;
		private double actualEnergy = -1.0;
		private boolean actualEnergyAvailable = false;
		private boolean actualSuccess;
		private String delayFeedbackType = "estimated";
		private String energyFeedbackType = "estimated_at_decision";
		private double observedCost;
		private double finalTrainingCost;
		private String failureReason;
		private int candidateCount;
		private String candidateActions = "";
		private String candidateVmIds = "";

		private DdldoDecision(int action, int vmIndex, int targetId, double[] input, double objective, double dnnScore,
				double finalCost, double delay, double energy, double currentDistance, double projectedDistance,
				double propagationDelay, double transmissionDelay, double executionDelay, double vmLoad,
				double coverageRemainingTime, double estimatedFinishTime, boolean mobilityRisk, boolean delayRisk,
				boolean offloadingPossible, boolean coverageFeasible, boolean latencyFeasible, boolean energyFeasible,
				boolean linkFeasible, boolean resourceFeasible, double energyBefore,
				String failureReason) {
			this.action = action;
			this.vmIndex = vmIndex;
			this.targetId = targetId;
			this.input = input;
			this.objective = objective;
			this.dnnScore = dnnScore;
			this.finalCost = finalCost;
			this.delay = delay;
			this.energy = energy;
			this.currentDistance = currentDistance;
			this.projectedDistance = projectedDistance;
			this.propagationDelay = propagationDelay;
			this.transmissionDelay = transmissionDelay;
			this.executionDelay = executionDelay;
			this.vmLoad = vmLoad;
			this.coverageRemainingTime = coverageRemainingTime;
			this.estimatedFinishTime = estimatedFinishTime;
			this.mobilityRisk = mobilityRisk;
			this.delayRisk = delayRisk;
			this.offloadingPossible = offloadingPossible;
			this.coverageFeasible = coverageFeasible;
			this.latencyFeasible = latencyFeasible;
			this.energyFeasible = energyFeasible;
			this.linkFeasible = linkFeasible;
			this.resourceFeasible = resourceFeasible;
			this.feasible = offloadingPossible && coverageFeasible && latencyFeasible && energyFeasible && linkFeasible
					&& resourceFeasible && !mobilityRisk;
			this.energyBefore = energyBefore;
			this.actualSuccess = this.feasible;
			this.observedCost = objective;
			this.finalTrainingCost = finalCost;
			this.failureReason = failureReason;
		}
	}

	private static boolean cudaRuntimeDllsAvailable() {
		if (cudaRuntimeDllsOnPath()) {
			return true;
		}
		return loadCudaRuntimeFromJavacpp();
	}

	private static boolean cudaRuntimeDllsOnPath() {
		for (String dll : DDLDO_CUDA_RUNTIME_DLLS) {
			if (!isDllOnPath(dll)) {
				return false;
			}
		}
		return true;
	}

	private static boolean loadCudaRuntimeFromJavacpp() {
		try {
			Class<?> loader = Class.forName("org.bytedeco.javacpp.Loader");
			String[] cudaClasses = {
					"org.bytedeco.cuda.global.cudart",
					"org.bytedeco.cuda.global.cublas"
			};
			for (String className : cudaClasses) {
				Class<?> cudaClass = Class.forName(className);
				loader.getMethod("load", Class.class).invoke(null, cudaClass);
			}
			return true;
		} catch (Throwable e) {
			return false;
		}
	}

	private static boolean isDllOnPath(String dllName) {
		String cudaPath = System.getenv("CUDA_PATH");
		if (fileExists(cudaPath, "bin", dllName)) {
			return true;
		}
		String path = System.getenv("PATH");
		if (path == null) {
			return false;
		}
		String[] entries = path.split(File.pathSeparator);
		for (String entry : entries) {
			if (fileExists(entry, null, dllName)) {
				return true;
			}
		}
		return false;
	}

	private static boolean fileExists(String base, String child, String fileName) {
		if (base == null || base.trim().isEmpty()) {
			return false;
		}
		File dir = child == null ? new File(base) : new File(base, child);
		return new File(dir, fileName).exists();
	}

	private static class SimpleDnn {
		private boolean nd4jEnabled;
		private INDArray nw1;
		private INDArray nb1;
		private INDArray nw2;
		private INDArray nb2;
		private INDArray nw3;
		private INDArray nb3;
		private INDArray nw4;
		private INDArray nb4;
		private final double[][] w1;
		private final double[] b1;
		private final double[][] w2;
		private final double[] b2;
		private final double[][] w3;
		private final double[] b3;
		private final double[][] w4;
		private final double[] b4;

		private SimpleDnn() {
			w1 = new double[DDLDO_INPUTS][DDLDO_HIDDEN_1];
			b1 = new double[DDLDO_HIDDEN_1];
			w2 = new double[DDLDO_HIDDEN_1][DDLDO_HIDDEN_2];
			b2 = new double[DDLDO_HIDDEN_2];
			w3 = new double[DDLDO_HIDDEN_2][DDLDO_HIDDEN_3];
			b3 = new double[DDLDO_HIDDEN_3];
			w4 = new double[DDLDO_HIDDEN_3][DDLDO_OUTPUTS];
			b4 = new double[DDLDO_OUTPUTS];
			init(w1);
			init(w2);
			init(w3);
			init(w4);
			if (!cudaRuntimeDllsAvailable()) {
				nd4jEnabled = false;
			} else {
				try {
				nw1 = nd4jWeights(DDLDO_INPUTS, DDLDO_HIDDEN_1);
				nb1 = Nd4j.zeros(1, DDLDO_HIDDEN_1);
				nw2 = nd4jWeights(DDLDO_HIDDEN_1, DDLDO_HIDDEN_2);
				nb2 = Nd4j.zeros(1, DDLDO_HIDDEN_2);
				nw3 = nd4jWeights(DDLDO_HIDDEN_2, DDLDO_HIDDEN_3);
				nb3 = Nd4j.zeros(1, DDLDO_HIDDEN_3);
				nw4 = nd4jWeights(DDLDO_HIDDEN_3, DDLDO_OUTPUTS);
				nb4 = Nd4j.zeros(1, DDLDO_OUTPUTS);
				nd4jEnabled = true;
				} catch (Throwable e) {
					nd4jEnabled = false;
				}
			}
		}

		private void init(double[][] weights) {
			for (int i = 0; i < weights.length; i++) {
				for (int j = 0; j < weights[i].length; j++) {
					weights[i][j] = (simulationParameters.RANDOM_GENERATOR.nextDouble() - 0.5) * 0.02;
				}
			}
		}

		private double predict(double[] input) {
			if (nd4jEnabled) {
				try {
					return predictNd4j(input);
				} catch (Throwable e) {
					nd4jEnabled = false;
					SimLog.println("DefaultEdgeOrchestrator- Warning: ND4J DNN prediction failed, falling back to CPU arrays: "
							+ e.getMessage());
				}
			}
			return predictCpu(input);
		}

		private double[] predictBatch(double[][] inputs) {
			if (inputs.length == 0) {
				return new double[0];
			}
			if (nd4jEnabled) {
				try {
					return predictBatchNd4j(inputs);
				} catch (Throwable e) {
					nd4jEnabled = false;
					SimLog.println("DefaultEdgeOrchestrator- Warning: ND4J DNN batch prediction failed, falling back to CPU arrays: "
							+ e.getMessage());
				}
			}
			double[] output = new double[inputs.length];
			for (int i = 0; i < inputs.length; i++) {
				output[i] = predictCpu(inputs[i]);
			}
			return output;
		}

		private double train(List<DdldoSample> samples) {
			if (nd4jEnabled) {
				try {
					return trainNd4j(samples);
				} catch (Throwable e) {
					nd4jEnabled = false;
					SimLog.println("DefaultEdgeOrchestrator- Warning: ND4J DNN training failed, falling back to CPU arrays: "
							+ e.getMessage());
				}
			}
			return trainCpu(samples);
		}

		private INDArray nd4jWeights(int rows, int cols) {
			return Nd4j.rand(new int[] { rows, cols }).subi(0.5).muli(0.02);
		}

		private double predictNd4j(double[] input) {
			INDArray x = Nd4j.create(input).reshape(1, DDLDO_INPUTS).castTo(nw1.dataType());
			return forwardNd4j(x)[3].getDouble(0);
		}

		private double[] predictBatchNd4j(double[][] inputs) {
			INDArray x = Nd4j.create(flatten(inputs)).reshape(inputs.length, DDLDO_INPUTS).castTo(nw1.dataType());
			INDArray output = forwardNd4j(x)[3];
			double[] predictions = new double[inputs.length];
			for (int i = 0; i < predictions.length; i++) {
				predictions[i] = output.getDouble(i, 0);
			}
			Nd4j.getExecutioner().commit();
			return predictions;
		}

		private double trainNd4j(List<DdldoSample> samples) {
			if (samples.isEmpty()) {
				return 0.0;
			}
			INDArray x = Nd4j.create(flattenInputs(samples)).reshape(samples.size(), DDLDO_INPUTS).castTo(nw1.dataType());
			INDArray y = Nd4j.create(flattenLabels(samples)).reshape(samples.size(), DDLDO_OUTPUTS).castTo(nw4.dataType());
			INDArray[] forward = forwardNd4j(x);
			INDArray h1 = forward[0];
			INDArray h2 = forward[1];
			INDArray h3 = forward[2];
			INDArray output = forward[3];
			INDArray error = output.sub(y);
			double loss = error.mul(error).sumNumber().doubleValue();

			INDArray delta4 = error.mul(sigmoidDerivative(output));
			INDArray delta3 = delta4.mmul(nw4.transpose()).mul(sigmoidDerivative(h3));
			INDArray delta2 = delta3.mmul(nw3.transpose()).mul(sigmoidDerivative(h2));
			INDArray delta1 = delta2.mmul(nw2.transpose()).mul(sigmoidDerivative(h1));

			double scale = DDLDO_LEARNING_RATE / Math.max(1, samples.size());
			nw4.subi(h3.transpose().mmul(delta4).muli(scale));
			nb4.subi(delta4.mean(0).muli(DDLDO_LEARNING_RATE));
			nw3.subi(h2.transpose().mmul(delta3).muli(scale));
			nb3.subi(delta3.mean(0).muli(DDLDO_LEARNING_RATE));
			nw2.subi(h1.transpose().mmul(delta2).muli(scale));
			nb2.subi(delta2.mean(0).muli(DDLDO_LEARNING_RATE));
			nw1.subi(x.transpose().mmul(delta1).muli(scale));
			nb1.subi(delta1.mean(0).muli(DDLDO_LEARNING_RATE));
			Nd4j.getExecutioner().commit();
			return loss / samples.size();
		}

		private double[] flatten(double[][] inputs) {
			double[] flat = new double[inputs.length * DDLDO_INPUTS];
			for (int row = 0; row < inputs.length; row++) {
				System.arraycopy(inputs[row], 0, flat, row * DDLDO_INPUTS, DDLDO_INPUTS);
			}
			return flat;
		}

		private double[] flattenInputs(List<DdldoSample> samples) {
			double[] flat = new double[samples.size() * DDLDO_INPUTS];
			for (int row = 0; row < samples.size(); row++) {
				System.arraycopy(samples.get(row).input, 0, flat, row * DDLDO_INPUTS, DDLDO_INPUTS);
			}
			return flat;
		}

		private double[] flattenLabels(List<DdldoSample> samples) {
			double[] flat = new double[samples.size() * DDLDO_OUTPUTS];
			for (int row = 0; row < samples.size(); row++) {
				System.arraycopy(samples.get(row).label, 0, flat, row * DDLDO_OUTPUTS, DDLDO_OUTPUTS);
			}
			return flat;
		}

		private INDArray[] forwardNd4j(INDArray x) {
			INDArray h1 = Transforms.sigmoid(x.mmul(nw1).addRowVector(nb1), false);
			INDArray h2 = Transforms.sigmoid(h1.mmul(nw2).addRowVector(nb2), false);
			INDArray h3 = Transforms.sigmoid(h2.mmul(nw3).addRowVector(nb3), false);
			INDArray output = Transforms.sigmoid(h3.mmul(nw4).addRowVector(nb4), false);
			return new INDArray[] { h1, h2, h3, output };
		}

		private INDArray sigmoidDerivative(INDArray activated) {
			return activated.mul(activated.rsub(1.0));
		}

		private double predictCpu(double[] input) {
			double[] h1 = sigmoid(add(matmul(input, w1), b1));
			double[] h2 = sigmoid(add(matmul(h1, w2), b2));
			double[] h3 = sigmoid(add(matmul(h2, w3), b3));
			return sigmoid(add(matmul(h3, w4), b4))[0];
		}

		private double trainCpu(List<DdldoSample> samples) {
			double loss = 0.0;
			for (DdldoSample sample : samples) {
				double[] z1 = add(matmul(sample.input, w1), b1);
				double[] h1 = sigmoid(z1);
				double[] z2 = add(matmul(h1, w2), b2);
				double[] h2 = sigmoid(z2);
				double[] z3 = add(matmul(h2, w3), b3);
				double[] h3 = sigmoid(z3);
				double[] output = sigmoid(add(matmul(h3, w4), b4));
				double error = output[0] - sample.label[0];
				loss += error * error;

				double[] delta4 = new double[DDLDO_OUTPUTS];
				for (int i = 0; i < DDLDO_OUTPUTS; i++) {
					delta4[i] = (output[i] - sample.label[i]) * output[i] * (1.0 - output[i]);
				}

				double[] delta3 = new double[DDLDO_HIDDEN_3];
				for (int i = 0; i < DDLDO_HIDDEN_3; i++) {
					double sum = 0.0;
					for (int j = 0; j < DDLDO_OUTPUTS; j++) {
						sum += delta4[j] * w4[i][j];
					}
					delta3[i] = sum * h3[i] * (1.0 - h3[i]);
				}

				double[] delta2 = new double[DDLDO_HIDDEN_2];
				for (int i = 0; i < DDLDO_HIDDEN_2; i++) {
					double sum = 0.0;
					for (int j = 0; j < DDLDO_HIDDEN_3; j++) {
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

				update(w4, h3, delta4);
				update(b4, delta4);
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
		List<Integer> possibleVms = possibleVms(architecture, task);
		if (possibleVms.isEmpty()) {
			return -1;
		}

		List<DdldoDecision> candidates = ddldoCandidateDecisions(task, possibleVms);
		List<DdldoDecision> dnnCandidates = ddldoDnnCandidatePool(candidates);
		DdldoDecision selected = selectBestDdldoDecision(dnnCandidates);
		if (selected == null) {
			selected = selectBestDdldoDecision(candidates);
		}
		if (selected == null) {
			selected = fallbackDdldoDecision(candidates);
		}
		if (selected == null || selected.vmIndex == -1) {
			return -1;
		}
		attachCandidateSummary(selected, candidates);
		ddldoPendingDecisions.put(task.getId(), selected);
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
		logDdldoBackend();
		for (int i = 0; i < DDLDO_DNN_COUNT; i++) {
			ddldoNetworks.add(new SimpleDnn());
		}
		ddldoMemory.clear();
		ddldoEpochsTrained = 0;
		ddldoInitialized = true;
	}

	private void logDdldoBackend() {
		if (ddldoBackendLogged) {
			return;
		}
		ddldoBackendLogged = true;
		String backend = "unavailable";
		String executioner = "unavailable";
		if (!cudaRuntimeDllsAvailable()) {
			backend = "unavailable (CUDA 11.6 runtime DLLs not found)";
			executioner = "unavailable";
			ddldoUsingGpu = false;
		} else {
			try {
				Class<?> nd4j = Class.forName("org.nd4j.linalg.factory.Nd4j");
				Object backendObject = nd4j.getMethod("getBackend").invoke(null);
				Object executionerObject = nd4j.getMethod("getExecutioner").invoke(null);
				backend = backendObject == null ? "null" : backendObject.getClass().getName();
				executioner = executionerObject == null ? "null" : executionerObject.getClass().getName();
				String name = (backend + " " + executioner).toLowerCase();
				ddldoUsingGpu = name.contains("cuda") || name.contains("jcublas");
			} catch (Throwable e) {
				backend = "unavailable (" + e.getClass().getSimpleName() + ")";
				executioner = "unavailable";
				ddldoUsingGpu = false;
			}
		}
		SimLog.println("DDLDO DNN structure: input -> 64 -> 128 -> 64 -> output");
		SimLog.println("ND4J backend: " + backend);
		SimLog.println("ND4J executioner: " + executioner);
		SimLog.println("DDLDO DNN using GPU: " + ddldoUsingGpu);
	}

	private double normalize(double value, double max) {
		return Math.max(0.0, Math.min(1.0, value / Math.max(1.0, max)));
	}

	private List<DdldoDecision> ddldoCandidateDecisions(Task task, List<Integer> possibleVms) {
		List<DdldoDecision> candidates = new ArrayList<>();
		for (int vm : possibleVms) {
			DdldoDecision decision = evaluateDdldoDecision(task, vm);
			if (decision != null) {
				candidates.add(decision);
			}
		}
		return candidates;
	}

	private List<DdldoDecision> ddldoDnnCandidatePool(List<DdldoDecision> candidates) {
		List<DdldoDecision> pool = new ArrayList<>();
		double[][] inputs = new double[candidates.size()][];
		for (int i = 0; i < candidates.size(); i++) {
			inputs[i] = candidates.get(i).input;
		}
		for (SimpleDnn network : ddldoNetworks) {
			DdldoDecision best = null;
			double bestScore = Double.MAX_VALUE;
			double[] predictions = network.predictBatch(inputs);
			for (int i = 0; i < candidates.size(); i++) {
				DdldoDecision candidate = candidates.get(i);
				double predicted = predictions[i];
				candidate.dnnScore += predicted / Math.max(1, ddldoNetworks.size());
				double score = predicted + 0.15 * normalizeCost(candidate.objective);
				if (!candidate.feasible) {
					score += 0.5;
				}
				if (score < bestScore) {
					bestScore = score;
					best = candidate;
				}
			}
			if (best != null && !pool.contains(best)) {
				pool.add(best);
			}
		}
		DdldoDecision bestCost = fallbackDdldoDecision(candidates);
		if (bestCost != null && !pool.contains(bestCost)) {
			pool.add(bestCost);
		}
		for (DdldoDecision candidate : candidates) {
			candidate.finalCost = candidate.objective + 0.25 * candidate.dnnScore;
		}
		return pool;
	}

	private DdldoDecision selectBestDdldoDecision(List<DdldoDecision> candidates) {
		DdldoDecision best = null;
		for (DdldoDecision decision : candidates) {
			if (!decision.feasible) {
				continue;
			}
			if (best == null || decision.finalCost < best.finalCost) {
				best = decision;
			}
		}
		return best;
	}

	private DdldoDecision fallbackDdldoDecision(List<DdldoDecision> candidates) {
		DdldoDecision bestDeadline = null;
		DdldoDecision bestAny = null;
		for (DdldoDecision decision : candidates) {
			if (bestAny == null || decision.finalCost < bestAny.finalCost) {
				bestAny = decision;
			}
			if (decision.latencyFeasible && decision.linkFeasible && decision.resourceFeasible && !decision.mobilityRisk
					&& (bestDeadline == null || decision.finalCost < bestDeadline.finalCost)) {
				bestDeadline = decision;
			}
		}
		DdldoDecision selected = bestDeadline != null ? bestDeadline : bestAny;
		if (selected != null && !selected.feasible && selected.failureReason.length() == 0) {
			selected.failureReason = "fallback_infeasible";
		}
		return selected;
	}

	private DdldoDecision evaluateDdldoDecision(Task task, int vm) {
		DataCenter destination = (DataCenter) vmList.get(vm).getHost().getDatacenter();
		int action = ddldoActionForVm(task, vm);
		double currentDistance = SimulationManager.getdistance(destination, task.getEdgeDevice());
		double executionDelay = taskExecutionDelay(task, vm);
		double projectedDistance = speedAwareDistance(task, destination, currentDistance, executionDelay);
		double propagationDelay = taskPropagationDelay(task, destination, projectedDistance);
		double transmissionDelay = taskTransferDelay(task, destination);
		double delay = propagationDelay + transmissionDelay + executionDelay;
		double energy = Math.max(1.0, estimatedTaskEnergy(task, vm, projectedDistance, executionDelay));
		double coverageRemainingTime = coverageRemainingTime(task, destination, delay);
		double estimatedFinishTime = simulationManager.getSimulation().clock() + delay;
		boolean delayRisk = delay > task.getMaxLatency();
		boolean mobilityRisk = mobilityRisk(task, destination, delay, coverageRemainingTime, currentDistance, projectedDistance);
		boolean offloadingPossible = destination == task.getEdgeDevice()
				|| (SimulationManager.issetlink(task.getEdgeDevice(), destination) && !destination.isDead());
		boolean coverageFeasible = coverageRemainingTime < 0 || delay <= coverageRemainingTime;
		boolean latencyFeasible = !delayRisk;
		boolean energyFeasible = energyFeasible(destination, energy);
		boolean linkFeasible = destination == task.getEdgeDevice() || SimulationManager.issetlink(task.getEdgeDevice(), destination);
		boolean resourceFeasible = !destination.isDead() && vmList.get(vm).getMips() > 0;
		double vmLoad = orchestrationHistory.get(vm).size();
		String failureReason = ddldoFailureReason(coverageFeasible, latencyFeasible, energyFeasible, linkFeasible,
				resourceFeasible, mobilityRisk, offloadingPossible);
		double objective = chapter4Objective(delay, energy, currentDistance, projectedDistance, vmLoad);
		if (delayRisk) {
			objective += 2.0 + (delay - task.getMaxLatency()) / Math.max(1.0, task.getMaxLatency());
		}
		if (!coverageFeasible) {
			objective += 2.0 + (delay - Math.max(0.0, coverageRemainingTime)) / Math.max(1.0, task.getMaxLatency());
		}
		if (!energyFeasible) {
			objective += 1.0;
		}
		if (!linkFeasible || !resourceFeasible) {
			objective += 3.0;
		}
		if (mobilityRisk) {
			objective += 1.5;
		}
		if (!offloadingPossible) {
			objective += 4.0;
		}
		double[] input = ddldoInput(task, destination, vm, currentDistance, projectedDistance, propagationDelay,
				transmissionDelay, executionDelay, energy, coverageRemainingTime, vmLoad, mobilityRisk, delayRisk,
				offloadingPossible);
		double finalCost = objective;
		double energyBefore = energySnapshot(task.getEdgeDevice(), destination);
		return new DdldoDecision(action, vm, targetId(destination), input, objective, 0.0, finalCost, delay, energy,
				currentDistance, projectedDistance, propagationDelay, transmissionDelay, executionDelay, vmLoad,
				coverageRemainingTime, estimatedFinishTime, mobilityRisk, delayRisk, offloadingPossible,
				coverageFeasible, latencyFeasible, energyFeasible, linkFeasible, resourceFeasible, energyBefore,
				failureReason);
	}

	private String ddldoFailureReason(boolean coverageFeasible, boolean latencyFeasible, boolean energyFeasible,
			boolean linkFeasible, boolean resourceFeasible, boolean mobilityRisk, boolean offloadingPossible) {
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
		if (mobilityRisk) {
			reasons.add("mobility");
		}
		if (!offloadingPossible) {
			reasons.add("offloading");
		}
		return String.join("|", reasons);
	}

	private boolean energyFeasible(DataCenter destination, double energy) {
		if (Double.isNaN(energy) || Double.isInfinite(energy)) {
			return false;
		}
		return energy >= 0.0;
	}

	private double energySnapshot(DataCenter source, DataCenter destination) {
		double sourceEnergy = dataCenterEnergy(source);
		double destinationEnergy = destination == source ? 0.0 : dataCenterEnergy(destination);
		if (Double.isNaN(sourceEnergy) || Double.isNaN(destinationEnergy)) {
			return Double.NaN;
		}
		return sourceEnergy + destinationEnergy;
	}

	private double dataCenterEnergy(DataCenter dataCenter) {
		if (dataCenter == null || dataCenter.getEnergyModel() == null) {
			return Double.NaN;
		}
		return dataCenter.getEnergyModel().getTotalEnergyConsumption();
	}

	private void attachDdldoFeedback(Task task, DdldoDecision decision, boolean success, String failureReason) {
		DataCenter destination = (DataCenter) vmList.get(decision.vmIndex).getHost().getDatacenter();
		decision.actualSuccess = success;
		decision.delayFeedbackType = "actual";
		decision.actualDelay = Math.max(0.0, simulationManager.getSimulation().clock() - task.getTime());

		double energyAfter = energySnapshot(task.getEdgeDevice(), destination);
		if (!Double.isNaN(decision.energyBefore) && !Double.isNaN(energyAfter) && energyAfter >= decision.energyBefore) {
			decision.actualEnergy = energyAfter - decision.energyBefore;
			decision.actualEnergyAvailable = true;
			decision.energyFeedbackType = "datacenter_delta";
		} else {
			decision.actualEnergy = -1.0;
			decision.actualEnergyAvailable = false;
			decision.energyFeedbackType = "estimated_at_completion";
		}

		double feedbackEnergy = decision.actualEnergyAvailable ? decision.actualEnergy : decision.energy;
		decision.observedCost = chapter4Objective(decision.actualDelay, feedbackEnergy, decision.currentDistance,
				decision.projectedDistance, decision.vmLoad);
		if (!success) {
			decision.observedCost += 2.0;
		}
		if (decision.actualDelay > task.getMaxLatency()) {
			decision.observedCost += 1.0 + (decision.actualDelay - task.getMaxLatency())
					/ Math.max(1.0, task.getMaxLatency());
		}
		if (decision.coverageRemainingTime >= 0 && decision.actualDelay > decision.coverageRemainingTime) {
			decision.observedCost += 2.0 + (decision.actualDelay - decision.coverageRemainingTime)
					/ Math.max(1.0, task.getMaxLatency());
		}
		if (failureReason != null && failureReason.contains("MOBILITY")) {
			decision.observedCost += 1.5;
		}
		decision.finalTrainingCost = decision.observedCost + 0.25 * decision.dnnScore;
		decision.finalCost = decision.finalTrainingCost;
	}

	private double[] ddldoInput(Task task, DataCenter destination, int vm, double currentDistance,
			double projectedDistance, double propagationDelay, double transmissionDelay, double executionDelay,
			double energy, double coverageRemainingTime, double vmLoad, boolean mobilityRisk, boolean delayRisk,
			boolean offloadingPossible) {
		double safeCoverage = coverageRemainingTime < 0 ? simulationParameters.LOCATIONTIMENUM : coverageRemainingTime;
		return new double[] {
				normalize(task.getLength(), 100000000.0),
				normalize(task.getFileSize(), 100000.0),
				normalize(task.getOutputSize(), 100000.0),
				normalize(task.getMaxLatency(), 20.0),
				normalize(simulationParameters.VEHICLE_SPEED, 120.0),
				normalize(simulationParameters.SATELLITE_SPEED, 10000.0),
				destination.getType() == simulationParameters.TYPES.EDGE_DEVICE ? 1.0 : 0.0,
				destination.getType() == simulationParameters.TYPES.EDGE_DATACENTER ? 1.0 : 0.0,
				destination.getType() == simulationParameters.TYPES.CLOUD ? 1.0 : 0.0,
				normalize(vmList.get(vm).getMips(), 100000.0),
				normalize(vmLoad, 100.0),
				normalize(currentDistance, Math.max(1.0, simulationParameters.CLOUD_RANGE)),
				normalize(propagationDelay, Math.max(1.0, task.getMaxLatency())),
				normalize(transmissionDelay, Math.max(1.0, task.getMaxLatency())),
				normalize(executionDelay, Math.max(1.0, task.getMaxLatency())),
				normalize(Math.log10(Math.max(1.0, energy)), 20.0),
				normalize(safeCoverage, Math.max(1.0, simulationParameters.LOCATIONTIMENUM)),
				mobilityRisk ? 1.0 : 0.0,
				delayRisk ? 1.0 : 0.0,
				offloadingPossible ? 1.0 : 0.0
		};
	}

	private double speedAwareDistance(Task task, DataCenter destination, double currentDistance, double executionDelay) {
		if (destination == task.getEdgeDevice()) {
			return 0.0;
		}
		double speed = Math.max(0.0, simulationParameters.VEHICLE_SPEED);
		double estimatedMotionWindow = Math.max(1.0, executionDelay + taskTransferDelay(task, destination));
		double projected = currentDistance + speed * estimatedMotionWindow;
		if (destination.getType() == simulationParameters.TYPES.CLOUD) {
			return currentDistance;
		}
		return Math.max(0.0, projected);
	}

	private boolean mobilityRisk(Task task, DataCenter destination, double delay, double coverageRemainingTime,
			double currentDistance, double projectedDistance) {
		if (destination == task.getEdgeDevice() || destination.getType() == simulationParameters.TYPES.CLOUD) {
			return false;
		}
		int range = destination.getType() == simulationParameters.TYPES.EDGE_DATACENTER
				? simulationParameters.EDGE_DATACENTERS_RANGE : simulationParameters.EDGE_DEVICES_RANGE;
		boolean rangeRisk = projectedDistance > Math.max(1.0, range);
		boolean coverageRisk = coverageRemainingTime >= 0 && delay > coverageRemainingTime;
		return rangeRisk || coverageRisk;
	}

	private double normalizeCost(double cost) {
		if (Double.isNaN(cost) || Double.isInfinite(cost)) {
			return 1.0;
		}
		return 1.0 - Math.exp(-Math.max(0.0, cost));
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
		double relativeSpeed = simulationParameters.SATELLITE_SPEED + Math.max(0.0, simulationParameters.VEHICLE_SPEED);
		double coverage = 2.0 * (radius + height) * gamma / Math.max(1.0, relativeSpeed);
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

	private String ddldoVmTypeLabel(DdldoDecision decision) {
		switch (decision.action) {
		case 0:
			return "local";
		case 1:
			return "mist";
		case 2:
			return "edge";
		case 3:
			return "cloud";
		default:
			return "unknown";
		}
	}

	private double chapter4Objective(double delay, double energy, double currentDistance, double projectedDistance,
			double vmLoad) {
		double delayCost = normalize(delay, 10.0);
		double energyCost = normalize(Math.log10(Math.max(1.0, energy)), 20.0);
		double distanceGrowth = normalize(Math.max(0.0, projectedDistance - currentDistance),
				Math.max(1.0, simulationParameters.EDGE_DATACENTERS_RANGE));
		double loadCost = normalize(vmLoad, 100.0);
		return CHAPTER4_DELAY_WEIGHT * delayCost + CHAPTER4_ENERGY_WEIGHT * energyCost + 0.10 * distanceGrowth
				+ 0.05 * loadCost;
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
		ddldoMemory.addLast(new DdldoSample(input, normalizeCost(decision.finalCost), decision));
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
				+ "candidateCount,candidateActions,candidateVmIds,selectedAction,selectedVmId,selectedVmType,"
				+ "objective,dnnScore,finalCost,estimatedDelay,actualDelay,observedDelay,estimatedEnergy,"
				+ "actualEnergy,actualEnergyAvailable,delayFeedbackType,energyFeedbackType,observedCost,"
				+ "finalTrainingCost,memorySize,currentDistance,projectedDistance,propagationDelay,transmissionDelay,executionDelay,"
				+ "coverageRemainingTime,estimatedFinishTime,coverageFeasible,latencyFeasible,energyFeasible,"
				+ "mobilityRisk,delayRisk,offloadingPossible,success,failureReason";
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
				ddldoVmTypeLabel(decision),
				csvDouble(decision.objective),
				csvDouble(decision.dnnScore),
				csvDouble(decision.finalCost),
				csvDouble(decision.delay),
				csvDouble(decision.actualDelay),
				csvDouble(decision.actualDelay),
				csvDouble(decision.energy),
				csvDouble(decision.actualEnergy),
				Boolean.toString(decision.actualEnergyAvailable),
				decision.delayFeedbackType,
				decision.energyFeedbackType,
				csvDouble(decision.observedCost),
				csvDouble(decision.finalTrainingCost),
				Integer.toString(ddldoMemory.size()),
				csvDouble(decision.currentDistance),
				csvDouble(decision.projectedDistance),
				csvDouble(decision.propagationDelay),
				csvDouble(decision.transmissionDelay),
				csvDouble(decision.executionDelay),
				csvDouble(decision.coverageRemainingTime),
				csvDouble(decision.estimatedFinishTime),
				Boolean.toString(decision.coverageFeasible),
				Boolean.toString(decision.latencyFeasible),
				Boolean.toString(decision.energyFeasible),
				Boolean.toString(decision.mobilityRisk),
				Boolean.toString(decision.delayRisk),
				Boolean.toString(decision.offloadingPossible),
				Boolean.toString(success),
				failureReason == null ? "" : failureReason);
		appendCsvLine(ddldoLogPath("chapter4_ddldo_decisions.csv"), header, line);
	}

	private void appendDdldoTrainingLog(double loss, DdldoDecision decision) {
		String header = "time,episodeOrStep,memorySize,batchSize,learningRate,loss,avgObjective,selectedAction,"
				+ "selectedVmId,selectedVmType,estimatedDelay,actualDelay,estimatedEnergy,actualEnergy,"
				+ "actualEnergyAvailable,delayFeedbackType,energyFeedbackType,observedCost,finalTrainingCost,"
				+ "success,failureReason";
		String line = joinCsv(
				csvDouble(simulationManager.getSimulation().clock()),
				Integer.toString(ddldoEpochsTrained),
				Integer.toString(ddldoMemory.size()),
				Integer.toString(DDLDO_BATCH_SIZE),
				csvDouble(DDLDO_LEARNING_RATE),
				csvDouble(loss),
				csvDouble(averageDdldoObjective()),
				Integer.toString(decision.action),
				Integer.toString(decision.targetId),
				ddldoVmTypeLabel(decision),
				csvDouble(decision.delay),
				csvDouble(decision.actualDelay),
				csvDouble(decision.energy),
				csvDouble(decision.actualEnergy),
				Boolean.toString(decision.actualEnergyAvailable),
				decision.delayFeedbackType,
				decision.energyFeedbackType,
				csvDouble(decision.observedCost),
				csvDouble(decision.finalTrainingCost),
				Boolean.toString(decision.actualSuccess),
				decision.failureReason == null ? "" : decision.failureReason);
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
				attachDdldoFeedback(task, decision, success, failureReason);
				rememberDdldo(decision.input, decision);
				double loss = trainDdldo();
				appendDdldoTrainingLog(loss, decision);
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
