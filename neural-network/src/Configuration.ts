type ConfigurationType = "perceptron"|"nn";
/** Configuration interface. Default preset must contain values for all these properties! */
interface Configuration {
	[property: string]: any;
	name: string;
	parent?: string; // inherit from
	data?: TrainingData[];
	custom?: boolean;
	inputLayer?: InputLayerConfig;
	outputLayer?: OutputLayerConfig;
	hiddenLayers?: LayerConfig[];
	learningRate?: number;
	bias?: boolean;
	autoRestart?: boolean;
	autoRestartTime?: int;
	stepsPerSecond?: int;
	iterationsPerClick?: int;
	showGradient?: boolean;
	originalBounds?: Util.Bounds;
	weights?: double[];
	trainingMethod?: string;
	saveLastWeights?: boolean;
	drawArrows?: boolean;
	drawCoordinateSystem?: boolean;
    arrowScale?: int;
	showTrainNextButton?: boolean;
	animationTrainSinglePoints?: boolean;
	type?: ConfigurationType;
}
