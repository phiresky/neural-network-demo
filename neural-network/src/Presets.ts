interface InputLayerConfig {
	neuronCount: int;
	names: string[];
}
interface LayerConfig {
	neuronCount: int;
	activation: string;
}
interface OutputLayerConfig extends LayerConfig {
	names: string[];
}
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
	batchTraining?: boolean;
	saveLastWeights?: boolean;
	drawArrows?: boolean;
	drawCoordinateSystem?: boolean;
    arrowScale?: int;
	showTrainNextButton?: boolean;
	animationTrainSinglePoints?: boolean;
	type?: "perceptron"|"nn";
}
module Presets {
	export const presets: Configuration[] = [
		{
			name: "Default",
			stepsPerSecond: 3000,
			learningRate: 0.05,
			showGradient: false,
			batchTraining: false,
			custom: false,
			bias: false,
			autoRestartTime: 5000,
			autoRestart: false,
			iterationsPerClick: 5000,
			data: <TrainingData[]>[
				{ input: [0, 0], output: [0] },
				{ input: [0, 1], output: [1] },
				{ input: [1, 0], output: [1] },
				{ input: [1, 1], output: [0] }
			],
			inputLayer: { neuronCount: 2, names: ["x", "y"] },
			outputLayer: { neuronCount: 1, activation: "sigmoid", names: ["x XOR y"] },
			hiddenLayers: [
				{ neuronCount: 2, activation: "sigmoid" },
			],
			saveLastWeights: false,
			drawArrows: false,
			arrowScale: 0.3,
			originalBounds: null,
			weights: null,
			drawCoordinateSystem: true,
			showTrainNextButton: false,
			animationTrainSinglePoints: false,
			type: "nn"
		},
		{
			name: "Binary Classifier for XOR"
			//defaults only
		},
		{
			name: "Binary Classifier for circular data",
			iterationsPerClick: 1000,
			hiddenLayers: [
				{ "neuronCount": 3, "activation": "sigmoid" },
			],
			inputLayer: { neuronCount: 2, names: ["x", "y"] },
			outputLayer: { neuronCount: 1, "activation": "sigmoid", names: ["Class"] },
			data: [{ input: [1.46, 1.36], output: [0] },
				{ input: [1.14, 1.26], output: [0] },
				{ input: [0.96, 0.97], output: [0] },
				{ input: [1.04, 0.76], output: [0] },
				{ input: [1.43, 0.81], output: [0] },
				{ input: [1.30, 1.05], output: [0] },
				{ input: [1.45, 1.22], output: [0] },
				{ input: [2.04, 1.10], output: [0] },
				{ input: [1.06, 0.28], output: [0] },
				{ input: [0.96, 0.57], output: [0] },
				{ input: [1.28, 0.46], output: [0] },
				{ input: [1.51, 0.33], output: [0] },
				{ input: [1.65, 0.68], output: [0] },
				{ input: [1.67, 1.01], output: [0] },
				{ input: [1.50, 1.83], output: [1] },
				{ input: [0.76, 1.69], output: [1] },
				{ input: [0.40, 0.71], output: [1] },
				{ input: [0.61, 1.18], output: [1] },
				{ input: [0.26, 1.42], output: [1] },
				{ input: [0.28, 1.89], output: [1] },
				{ input: [1.37, 1.89], output: [1] },
				{ input: [1.11, 1.90], output: [1] },
				{ input: [1.05, 2.04], output: [1] },
				{ input: [2.43, 1.42], output: [1] },
				{ input: [2.39, 1.20], output: [1] },
				{ input: [2.10, 1.53], output: [1] },
				{ input: [1.89, 1.72], output: [1] },
				{ input: [2.69, 0.72], output: [1] },
				{ input: [2.96, 0.44], output: [1] },
				{ input: [2.50, 0.79], output: [1] },
				{ input: [2.85, 1.23], output: [1] },
				{ input: [2.82, 1.37], output: [1] },
				{ input: [1.93, 1.90], output: [1] },
				{ input: [2.18, 1.77], output: [1] },
				{ input: [2.29, 0.39], output: [1] },
				{ input: [2.57, 0.22], output: [1] },
				{ input: [2.70, -0.11], output: [1] },
				{ input: [1.96, -0.20], output: [1] },
				{ input: [1.89, -0.10], output: [1] },
				{ input: [1.77, 0.13], output: [1] },
				{ input: [0.73, 0.01], output: [1] },
				{ input: [0.37, 0.31], output: [1] },
				{ input: [0.46, 0.44], output: [1] },
				{ input: [0.48, 0.11], output: [1] },
				{ input: [0.37, -0.10], output: [1] },
				{ input: [1.03, -0.42], output: [1] },
				{ input: [1.35, -0.25], output: [1] },
				{ input: [1.17, 0.01], output: [1] },
				{ input: [0.12, 0.94], output: [1] },
				{ input: [2.05, 0.32], output: [1] },
				{ input: [1.97, 0.55], output: [0] }]
		},
		{
			name: "Three classes test",
			iterationsPerClick: 500,
			stepsPerSecond: 1500,
			hiddenLayers: [
				{ "neuronCount": 4, "activation": "sigmoid" },
			],
			inputLayer: { neuronCount: 2, names: ["x", "y"] },
			outputLayer: { neuronCount: 3, "activation": "sigmoid", names: ["A", "B", "C"] },
			data: [{ "input": [1.40, 1.3], "output": [1, 0, 0] }, { "input": [1.56, 1.36], "output": [1, 0, 0] }, { "input": [1.36, 1.36], "output": [1, 0, 0] }, { "input": [1.46, 1.36], "output": [1, 0, 0] }, { "input": [1.14, 1.26], "output": [1, 0, 0] }, { "input": [0.96, 0.97], "output": [1, 0, 0] }, { "input": [1.04, 0.76], "output": [1, 0, 0] }, { "input": [1.43, 0.81], "output": [1, 0, 0] }, { "input": [1.3, 1.05], "output": [1, 0, 0] }, { "input": [1.45, 1.22], "output": [1, 0, 0] }, { "input": [2.04, 1.1], "output": [1, 0, 0] }, { "input": [1.06, 0.28], "output": [1, 0, 0] }, { "input": [0.96, 0.57], "output": [1, 0, 0] }, { "input": [1.28, 0.46], "output": [1, 0, 0] }, { "input": [1.51, 0.33], "output": [1, 0, 0] }, { "input": [1.65, 0.68], "output": [1, 0, 0] }, { "input": [1.67, 1.01], "output": [1, 0, 0] }, { "input": [1.5, 1.83], "output": [0, 1, 0] }, { "input": [0.76, 1.69], "output": [0, 1, 0] }, { "input": [0.4, 0.71], "output": [0, 1, 0] }, { "input": [0.61, 1.18], "output": [0, 1, 0] }, { "input": [0.26, 1.42], "output": [0, 1, 0] }, { "input": [0.28, 1.89], "output": [0, 1, 0] }, { "input": [1.37, 1.89], "output": [0, 1, 0] }, { "input": [1.11, 1.9], "output": [0, 1, 0] }, { "input": [1.05, 2.04], "output": [0, 1, 0] }, { "input": [2.43, 1.42], "output": [0, 1, 0] }, { "input": [2.39, 1.2], "output": [0, 1, 0] }, { "input": [2.1, 1.53], "output": [0, 1, 0] }, { "input": [1.89, 1.72], "output": [0, 1, 0] }, { "input": [2.69, 0.72], "output": [0, 1, 0] }, { "input": [2.96, 0.44], "output": [0, 1, 0] }, { "input": [2.5, 0.79], "output": [0, 1, 0] }, { "input": [2.85, 1.23], "output": [0, 1, 0] }, { "input": [2.82, 1.37], "output": [0, 1, 0] }, { "input": [1.93, 1.9], "output": [0, 1, 0] }, { "input": [2.18, 1.77], "output": [0, 1, 0] }, { "input": [2.29, 0.39], "output": [0, 1, 0] }, { "input": [2.57, 0.22], "output": [0, 1, 0] }, { "input": [2.7, -0.11], "output": [0, 1, 0] }, { "input": [1.96, -0.2], "output": [0, 1, 0] }, { "input": [1.89, -0.1], "output": [0, 1, 0] }, { "input": [1.77, 0.13], "output": [0, 1, 0] }, { "input": [0.73, 0.01], "output": [0, 1, 0] }, { "input": [0.37, 0.31], "output": [0, 1, 0] }, { "input": [0.46, 0.44], "output": [0, 1, 0] }, { "input": [0.48, 0.11], "output": [0, 1, 0] }, { "input": [0.37, -0.1], "output": [0, 1, 0] }, { "input": [1.03, -0.42], "output": [0, 1, 0] }, { "input": [1.35, -0.25], "output": [0, 1, 0] }, { "input": [1.17, 0.01], "output": [0, 1, 0] }, { "input": [0.12, 0.94], "output": [0, 1, 0] }, { "input": [2.05, 0.32], "output": [0, 1, 0] }, { "input": [1.97, 0.55], "output": [1, 0, 0] },
				{ "input": [0.7860082304526748, 2.5761316872427984], "output": [0, 0, 1] }, { "input": [-0.09053497942386843, 2.3909465020576133], "output": [0, 0, 1] }, { "input": [-0.23868312757201657, 2.0329218106995888], "output": [0, 0, 1] }, { "input": [-0.32510288065843634, 1.748971193415638], "output": [0, 0, 1] }, { "input": [-0.6707818930041154, 1.4526748971193417], "output": [0, 0, 1] }, { "input": [-0.3991769547325104, 1.094650205761317], "output": [0, 0, 1] }, { "input": [-0.2263374485596709, 0.6131687242798356], "output": [0, 0, 1] }, { "input": [-0.2263374485596709, -0.42386831275720144], "output": [0, 0, 1] }, { "input": [-0.13991769547325114, -0.6584362139917693], "output": [0, 0, 1] }, { "input": [1.5390946502057612, -1.0658436213991767], "output": [0, 0, 1] }, { "input": [2.193415637860082, -1.0781893004115224], "output": [0, 0, 1] }, { "input": [2.6502057613168724, -0.9176954732510286], "output": [0, 0, 1] }, { "input": [3.193415637860082, -0.6460905349794236], "output": [0, 0, 1] }, { "input": [3.526748971193415, -0.42386831275720144], "output": [0, 0, 1] }, { "input": [3.4403292181069953, 0.329218106995885], "output": [0, 0, 1] }, { "input": [3.4773662551440325, 1.0452674897119343], "output": [0, 0, 1] }, { "input": [3.6625514403292176, 1.2798353909465023], "output": [0, 0, 1] }, { "input": [2.8847736625514404, 2.946502057613169], "output": [0, 0, 1] }, { "input": [1.4156378600823043, 2.5514403292181074], "output": [0, 0, 1] }, { "input": [1.045267489711934, 2.526748971193416], "output": [0, 0, 1] }, { "input": [2.5144032921810697, 2.1563786008230457], "output": [0, 0, 1] }, { "input": [3.045267489711934, 1.7983539094650207], "output": [0, 0, 1] }, { "input": [2.366255144032922, 2.9341563786008233], "output": [0, 0, 1] }, { "input": [1.5020576131687242, 3.0576131687242802], "output": [0, 0, 1] }, { "input": [0.5390946502057612, 2.711934156378601], "output": [0, 0, 1] }, { "input": [-0.300411522633745, 2.5761316872427984], "output": [0, 0, 1] }, { "input": [-0.7942386831275722, 2.563786008230453], "output": [0, 0, 1] }, { "input": [-1.1646090534979425, 1.181069958847737], "output": [0, 0, 1] }, { "input": [-1.1275720164609055, 0.5637860082304529], "output": [0, 0, 1] }, { "input": [-0.5226337448559671, 0.46502057613168746], "output": [0, 0, 1] }, { "input": [-0.4115226337448561, -0.05349794238683104], "output": [0, 0, 1] }, { "input": [-0.1646090534979425, -0.7325102880658434], "output": [0, 0, 1] }, { "input": [0.4650205761316871, -0.8436213991769544], "output": [0, 0, 1] }, { "input": [0.8106995884773661, -1.164609053497942], "output": [0, 0, 1] }, { "input": [0.32921810699588466, -1.3004115226337447], "output": [0, 0, 1] }, { "input": [1.1687242798353907, -1.127572016460905], "output": [0, 0, 1] }, { "input": [2.1316872427983538, -1.362139917695473], "output": [0, 0, 1] }, { "input": [1.7119341563786008, -0.6954732510288063], "output": [0, 0, 1] }, { "input": [2.5267489711934155, -0.8930041152263373], "output": [0, 0, 1] }, { "input": [2.8971193415637857, -0.8930041152263373], "output": [0, 0, 1] }, { "input": [2.6378600823045266, -0.6460905349794236], "output": [0, 0, 1] }, { "input": [3.2427983539094645, -0.5349794238683125], "output": [0, 0, 1] }, { "input": [3.8477366255144028, 0.02057613168724303], "output": [0, 0, 1] }, { "input": [3.390946502057613, 0.02057613168724303], "output": [0, 0, 1] }, { "input": [3.4403292181069953, 0.3415637860082307], "output": [0, 0, 1] }, { "input": [3.7983539094650203, 0.6502057613168727], "output": [0, 0, 1] }, { "input": [3.526748971193415, 0.983539094650206], "output": [0, 0, 1] }, { "input": [3.452674897119341, 1.4526748971193417], "output": [0, 0, 1] }, { "input": [3.502057613168724, 1.7242798353909468], "output": [0, 0, 1] }, { "input": [3.415637860082304, 2.205761316872428], "output": [0, 0, 1] }, { "input": [2.736625514403292, 2.292181069958848], "output": [0, 0, 1] }, { "input": [1.9465020576131686, 2.403292181069959], "output": [0, 0, 1] }, { "input": [1.8230452674897117, 2.60082304526749], "output": [0, 0, 1] }, { "input": [3.008230452674897, -1.288065843621399], "output": [0, 0, 1] }, { "input": [1.699588477366255, -1.016460905349794], "output": [0, 0, 1] }, { "input": [2.045267489711934, -0.9053497942386829], "output": [0, 0, 1] }, { "input": [1.8724279835390945, -1.2263374485596705], "output": [0, 0, 1] }]
		},
		{ name: "Vowel frequency response (Peterson and Barney)",
			parent: "Three classes test",
			stepsPerSecond: 100,
			iterationsPerClick: 50,
			inputLayer: { neuronCount: 2, names: ["F1","F2"] },
			outputLayer: { neuronCount: 10, "activation": "sigmoid", names:"IY,IH,EH,AE,AH,AA,AO,UH,UW,ER".split(",")}
		},
		{
			name: "Auto-Encoder for linear data",
			stepsPerSecond: 60,
			iterationsPerClick: 10,
			parent: "Auto-Encoder for circular data",
			data: [
				{ input: [2.25, 0.19], output: [2.25, 0.19] },
				{ input: [1.37, 0.93], output: [1.37, 0.93] },
				{ input: [0.62, 1.46], output: [0.62, 1.46] },
				{ input: [-0.23, 2.16], output: [-0.23, 2.16] },
				{ input: [-0.55, 2.44], output: [-0.55, 2.44] },
				{ input: [1.04, 1.05], output: [1.04, 1.05] },
				{ input: [1.70, 0.85], output: [1.70, 0.85] },
				{ input: [2.01, 0.46], output: [2.01, 0.46] },
				{ input: [0.40, 1.73], output: [0.40, 1.73] },
				{ input: [2.73, 0.01], output: [2.73, 0.01] },
				{ input: [2.86, -0.25], output: [2.86, -0.25] },
				{ input: [0.14, 2.07], output: [0.14, 2.07] }],
			hiddenLayers: [
				{ neuronCount: 1, activation: "sigmoid" },
			],

			showGradient: true
		},
		{
			name: "Auto-Encoder for x^2",
			parent: "Auto-Encoder for circular data",
			netLayers: [
				{
					"activation": "sigmoid",
					"neuronCount": 2
				},
				{
					"activation": "linear",
					"neuronCount": 1
				},
				{
					"neuronCount": 2,
					"activation": "sigmoid"
				},
			],
			data: (<number[]>Array.apply(null, Array(17)))
				.map((e, i) => (i - 8) / 8).map(x => ({ input: [x, x * x], output: [x, x * x] }))
		},
		{
			name: "Auto-Encoder for circular data",
			stepsPerSecond: 3000,
			"learningRate": 0.01,
			"iterationsPerClick": 200,
			inputLayer: { neuronCount: 2, names: ["x", "y"] },
			outputLayer: { neuronCount: 2, activation: "linear", names: ["x", "y"] },
			hiddenLayers: [
				{
					"activation": "sigmoid",
					"neuronCount": 3
				},
				{
					"activation": "linear",
					"neuronCount": 1
				},
				{
					"neuronCount": 3,
					"activation": "sigmoid"
				},
			],
			data: [{ input: [-0.83, 0.55], output: [-0.83, 0.55] },
				{ input: [-0.98, 0.21], output: [-0.98, 0.21] },
				{ input: [-0.77, -0.64], output: [-0.77, -0.64] },
				{ input: [0.95, 0.31], output: [0.95, 0.31] },
				{ input: [-0.86, -0.51], output: [-0.86, -0.51] },
				{ input: [0.99, -0.11], output: [0.99, -0.11] },
				{ input: [0.97, 0.24], output: [0.97, 0.24] },
				{ input: [0.85, 0.52], output: [0.85, 0.52] },
				{ input: [-0.99, 0.15], output: [-0.99, 0.15] },
				{ input: [0.62, 0.78], output: [0.62, 0.78] },
				{ input: [0.46, -0.89], output: [0.46, -0.89] },
				{ input: [-0.68, -0.73], output: [-0.68, -0.73] },
				{ input: [0.60, -0.80], output: [0.60, -0.80] },
				{ input: [0.38, 0.92], output: [0.38, 0.92] },
				{ input: [0.76, 0.65], output: [0.76, 0.65] },
				{ input: [0.33, -0.94], output: [0.33, -0.94] },
				{ input: [-0.99, -0.17], output: [-0.99, -0.17] },
				{ input: [-0.99, -0.17], output: [-0.99, -0.17] },
				{ input: [-0.97, -0.26], output: [-0.97, -0.26] },
				{ input: [-0.79, -0.61], output: [-0.79, -0.61] },
				{ input: [-0.03, -1.00], output: [-0.03, -1.00] },
				{ input: [0.58, 0.81], output: [0.58, 0.81] },
				{ input: [-0.67, -0.74], output: [-0.67, -0.74] },
				{ input: [0.14, 0.99], output: [0.14, 0.99] },
				{ input: [0.13, -0.99], output: [0.13, -0.99] },
				{ input: [0.76, 0.65], output: [0.76, 0.65] },
				{ input: [-0.49, 0.87], output: [-0.49, 0.87] },
				{ input: [-0.28, 0.96], output: [-0.28, 0.96] },
				{ input: [0.47, -0.88], output: [0.47, -0.88] },
				{ input: [-0.03, 1.00], output: [-0.03, 1.00] },
				{ input: [-0.70, 0.71], output: [-0.70, 0.71] },
				{ input: [0.38, 0.93], output: [0.38, 0.93] },
				{ input: [0.62, 0.79], output: [0.62, 0.79] },
				{ input: [0.72, -0.69], output: [0.72, -0.69] },
				{ input: [-0.41, -0.91], output: [-0.41, -0.91] },
				{ input: [0.74, -0.67], output: [0.74, -0.67] },
				{ input: [0.44, 0.90], output: [0.44, 0.90] },
				{ input: [-0.99, -0.16], output: [-0.99, -0.16] },
				{ input: [0.62, 0.78], output: [0.62, 0.78] },
				{ input: [0.95, -0.39], output: [0.95, -0.39] },
				{ input: [0.86, -0.53], output: [0.86, -0.53] }]
		},
		{ "name": "Bit Position Auto Encoder", "learningRate": 0.05, "data": [{ "input": [1, 0, 0, 0], "output": [1, 0, 0, 0] }, { "input": [0, 1, 0, 0], "output": [0, 1, 0, 0] }, { "input": [0, 0, 1, 0], "output": [0, 0, 1, 0] }, { "input": [0, 0, 0, 1], "output": [0, 0, 0, 1] }], "inputLayer": { "neuronCount": 4, "names": ["in1", "in2", "in3", "in4"] }, "outputLayer": { "neuronCount": 4, "activation": "sigmoid", "names": ["out1", "out2", "out3", "out4"] }, "hiddenLayers": [{ "neuronCount": 2, "activation": "sigmoid" }], "netLayers": [{ "activation": "sigmoid", "neuronCount": 2 }, { "activation": "linear", "neuronCount": 1 }, { "neuronCount": 2, "activation": "sigmoid" }] },
		{"name": "!listDivider"},
		{
			"name": "Rosenblatt Perzeptron",
			stepsPerSecond: 2,
			"learningRate": 0.5,
			"showGradient": false,
			"bias": false,
			"autoRestartTime": 5000,
			"autoRestart": false,
			batchTraining: true,
			saveLastWeights: true,
			showTrainNextButton: true,
			drawArrows: true,
			drawCoordinateSystem: false,
			animationTrainSinglePoints: true,
			type: "perceptron",
			"iterationsPerClick": 1,
			"data":[{"input":[0.2101231155778894,0.4947319932998326],"output":[0]},{"input":[0.07838107202680059,0.42886097152428815],"output":[0]},{"input":[0.027711055276381822,0.9000921273031828],"output":[0]},{"input":[0.5344112227805695,0.5910050251256282],"output":[0]},{"input":[0.5445452261306533,0.11977386934673367],"output":[0]},{"input":[0.4482721943048576,-0.07783919597989952],"output":[0]},{"input":[0.7725603015075377,-0.305854271356784],"output":[0]},{"input":[0.5445452261306533,-0.3210552763819096],"output":[0]},{"input":[-0.028025963149078823,0.20084589614740372],"output":[1]},{"input":[0.2506591289782244,-0.36159128978224464],"output":[1]},{"input":[-0.22057202680067015,-0.04237018425460638],"output":[1]},{"input":[-0.3573810720268008,0.2819179229480737],"output":[1]},{"input":[-0.5549941373534341,0.2211139028475712],"output":[1]},{"input":[0.05304606365159121,-0.3109212730318259],"output":[1]},{"input":[-0.4485871021775546,-0.31598827470686774],"output":[1]}],
			"inputLayer": {
				"neuronCount": 2,
				"names": [
					"x",
					"y"
				]
			},
			"outputLayer": {
				"neuronCount": 1,
				"activation": "threshold (≥ 0)",
				"names": [
					"class"
				]
			},
			"hiddenLayers": [],
		}
	];
	export function getNames(): string[] {
		return presets.map(p => p.name).filter(c => c !== "Default");
	}
	export function exists(name: string) {
		return presets.filter(p => p.name === name)[0] !== undefined;
	}
	export function get(name: string): Configuration {
		const chain: any[] = [];
		let preset = presets.filter(p => p.name === name)[0];
		chain.unshift(preset);
		while (true) {
			var parentName = preset.parent || "Default";
			preset = presets.filter(p => p.name === parentName)[0];
			chain.unshift(preset);
			if (parentName === "Default") break;
		}
		chain.unshift({});
		console.log("loading preset chain: " + chain.map((c: any) => c.name));
		return JSON.parse(JSON.stringify($.extend.apply($, chain)));
	}
	export function printPreset(sim: Simulation, parentName = "Default") {
		const parent = get(parentName);
		const outconf: any = {};
		for (const prop in sim.state) {
			if (sim.state[prop] !== parent[prop]) outconf[prop] = sim.state[prop];
		}
		/*outconf.data = config.data.map(
			e => '{input:[' + e.input.map(x=> x.toFixed(2))
				+ '], output:[' +
				(config["simType"] == SimulationType.BinaryClassification
					? e.output
					: e.input.map(x=> x.toFixed(2)))
				+ ']},').join("\n");*/
		return outconf;
	}
	
	function parseBarney(data: [double, double, int, int][]) {
		// _cache = LZString.compressToBase64(JSON.stringify(data));
		const relevantData = data
		.filter(row => /* only male participants (better separation) */ row[3] == 1)
		.map(row => (
			{
				input:row.slice(0,2),
				output:Util.arrayWithOneAt(10, row[2] - 1)
			}
		));
		let preset = presets.filter(p => p.name === "Vowel frequency response (Peterson and Barney)")[0];
		preset.data = relevantData;
		Util.normalizeInputs(preset);
		//presets.forEach(preset => preset.data && normalizeInputs(preset.data));
	}
		
	export function loadPetersonBarney() {
		// include peterson_barney_data for faster page load
		const dataStr = "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";
		parseBarney(JSON.parse(LZString.decompressFromBase64(dataStr)));
		return;
	}
	function loadPetersonBarneyAsync() {
		$.get("lib/peterson_barney_data").then((strData:string) => {
			const cols = {
				gender:0, // 1=M,2=F,3=C
				speaker:1,
				phonemeNum:2,
				phonemeAscii:3,
				F0:4,
				F1:5,
				F2:6,
				F3:7
			}
			const isNum = [true,true,true,,true,true,true,true];
			const data = strData.split("\n")
				.filter(row => row.indexOf("#") !== 0)
				.filter(row => row.trim().length > 2 && row.indexOf("*") < 0)
				.map(row => row.split(/\s+/).map((n,i) => i in isNum ? parseFloat(n):n));
			const relevantData:any = data.map(row => 
				[+row[cols.F1], +row[cols.F2], +row[cols.phonemeNum], +row[cols.gender]]);
			parseBarney(relevantData);
		});
	}
}