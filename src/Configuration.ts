/** type of the demonstration. Can change GUI and valid options */
type ConfigurationType = "perceptron" | "nn";
import { int, double } from "./main";
import { InputLayerConfig, OutputLayerConfig, LayerConfig } from "./Presets";

export interface TrainingData {
	input: double[];
	output: double[];
}
/** Configuration interface. Default preset must contain values for all these properties! */
export interface Configuration {
	name: string;
	/** inherit all properties from the preset with this name */
	parent: string;
	data: TrainingData[];
	/** is a custom configuration or a preset */
	custom: boolean;
	inputLayer: InputLayerConfig;
	outputLayer: OutputLayerConfig;
	hiddenLayers: LayerConfig[];
	/** learning rate factor (gamma) between 0 and 1*/
	learningRate: number;
	/** show the bias nodes */
	bias: boolean;
	/** automatically restart when all data is correct */
	autoRestart: boolean;
	/** restart after x ms */
	autoRestartTime: int;
	/** number of steps to run per secon when animating */
	stepsPerSecond: int;
	iterationsPerClick: int;
	/** show the class for every pixel in the background as a gradient instead of thresholded */
	showGradient: boolean;
	/** preset start weights (random if not defined) */
	weights: double[] | null;
	/** training method (one of [[Simulation.trainingMethods]]) */
	trainingMethod: string;
	drawCoordinateSystem: boolean;
	/** draw weight arrows (only possible when single perceptron) */
	drawArrows: boolean;
	arrowScale: int;
	/** when animating, step through all data points on their own */
	animationTrainSinglePoints: boolean;
	/** show train single button. only used for neural network demo */
	showTrainSingleButton: boolean;
	type: ConfigurationType;
}

export type PartialConfiguration = Partial<Configuration> & { name: string };
