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
interface Configuration {
	[property: string]: any;
	name: string;
	parent?: string; // inherit from
	data?: TrainingData[];
	inputLayer?: InputLayerConfig;
	outputLayer?: OutputLayerConfig;
	hiddenLayers?: LayerConfig[];
	learningRate?: number;
	bias?: boolean;
	autoRestart?: boolean;
	autoRestartTime?: int;
	stepsPerFrame?: int;
	iterationsPerClick?: int;
	showGradient?: boolean;
}
module Presets {
	let presets: Configuration[] = [
		{
			name: "Default",
			stepsPerFrame: 50,
			learningRate: 0.05,
			showGradient: false,
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
			]
		},
		{
			name: "Binary Classifier for XOR"
			//defaults only
		},
		{
			name: "Binary Classifier for circular data",
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
			hiddenLayers: [
				{ "neuronCount": 4, "activation": "sigmoid" },
			],
			inputLayer: { neuronCount: 2, names: ["x", "y"] },
			outputLayer: { neuronCount: 3, "activation": "sigmoid", names: ["A", "B", "C"] },
			data: [{ "input": [1.46, 1.36], "output": [1, 0, 0] }, { "input": [1.14, 1.26], "output": [1, 0, 0] }, { "input": [0.96, 0.97], "output": [1, 0, 0] }, { "input": [1.04, 0.76], "output": [1, 0, 0] }, { "input": [1.43, 0.81], "output": [1, 0, 0] }, { "input": [1.3, 1.05], "output": [1, 0, 0] }, { "input": [1.45, 1.22], "output": [1, 0, 0] }, { "input": [2.04, 1.1], "output": [1, 0, 0] }, { "input": [1.06, 0.28], "output": [1, 0, 0] }, { "input": [0.96, 0.57], "output": [1, 0, 0] }, { "input": [1.28, 0.46], "output": [1, 0, 0] }, { "input": [1.51, 0.33], "output": [1, 0, 0] }, { "input": [1.65, 0.68], "output": [1, 0, 0] }, { "input": [1.67, 1.01], "output": [1, 0, 0] }, { "input": [1.5, 1.83], "output": [0, 1, 0] }, { "input": [0.76, 1.69], "output": [0, 1, 0] }, { "input": [0.4, 0.71], "output": [0, 1, 0] }, { "input": [0.61, 1.18], "output": [0, 1, 0] }, { "input": [0.26, 1.42], "output": [0, 1, 0] }, { "input": [0.28, 1.89], "output": [0, 1, 0] }, { "input": [1.37, 1.89], "output": [0, 1, 0] }, { "input": [1.11, 1.9], "output": [0, 1, 0] }, { "input": [1.05, 2.04], "output": [0, 1, 0] }, { "input": [2.43, 1.42], "output": [0, 1, 0] }, { "input": [2.39, 1.2], "output": [0, 1, 0] }, { "input": [2.1, 1.53], "output": [0, 1, 0] }, { "input": [1.89, 1.72], "output": [0, 1, 0] }, { "input": [2.69, 0.72], "output": [0, 1, 0] }, { "input": [2.96, 0.44], "output": [0, 1, 0] }, { "input": [2.5, 0.79], "output": [0, 1, 0] }, { "input": [2.85, 1.23], "output": [0, 1, 0] }, { "input": [2.82, 1.37], "output": [0, 1, 0] }, { "input": [1.93, 1.9], "output": [0, 1, 0] }, { "input": [2.18, 1.77], "output": [0, 1, 0] }, { "input": [2.29, 0.39], "output": [0, 1, 0] }, { "input": [2.57, 0.22], "output": [0, 1, 0] }, { "input": [2.7, -0.11], "output": [0, 1, 0] }, { "input": [1.96, -0.2], "output": [0, 1, 0] }, { "input": [1.89, -0.1], "output": [0, 1, 0] }, { "input": [1.77, 0.13], "output": [0, 1, 0] }, { "input": [0.73, 0.01], "output": [0, 1, 0] }, { "input": [0.37, 0.31], "output": [0, 1, 0] }, { "input": [0.46, 0.44], "output": [0, 1, 0] }, { "input": [0.48, 0.11], "output": [0, 1, 0] }, { "input": [0.37, -0.1], "output": [0, 1, 0] }, { "input": [1.03, -0.42], "output": [0, 1, 0] }, { "input": [1.35, -0.25], "output": [0, 1, 0] }, { "input": [1.17, 0.01], "output": [0, 1, 0] }, { "input": [0.12, 0.94], "output": [0, 1, 0] }, { "input": [2.05, 0.32], "output": [0, 1, 0] }, { "input": [1.97, 0.55], "output": [1, 0, 0] },
				{ "input": [0.7860082304526748, 2.5761316872427984], "output": [0, 0, 1] }, { "input": [-0.09053497942386843, 2.3909465020576133], "output": [0, 0, 1] }, { "input": [-0.23868312757201657, 2.0329218106995888], "output": [0, 0, 1] }, { "input": [-0.32510288065843634, 1.748971193415638], "output": [0, 0, 1] }, { "input": [-0.6707818930041154, 1.4526748971193417], "output": [0, 0, 1] }, { "input": [-0.3991769547325104, 1.094650205761317], "output": [0, 0, 1] }, { "input": [-0.2263374485596709, 0.6131687242798356], "output": [0, 0, 1] }, { "input": [-0.2263374485596709, -0.42386831275720144], "output": [0, 0, 1] }, { "input": [-0.13991769547325114, -0.6584362139917693], "output": [0, 0, 1] }, { "input": [1.5390946502057612, -1.0658436213991767], "output": [0, 0, 1] }, { "input": [2.193415637860082, -1.0781893004115224], "output": [0, 0, 1] }, { "input": [2.6502057613168724, -0.9176954732510286], "output": [0, 0, 1] }, { "input": [3.193415637860082, -0.6460905349794236], "output": [0, 0, 1] }, { "input": [3.526748971193415, -0.42386831275720144], "output": [0, 0, 1] }, { "input": [3.4403292181069953, 0.329218106995885], "output": [0, 0, 1] }, { "input": [3.4773662551440325, 1.0452674897119343], "output": [0, 0, 1] }, { "input": [3.6625514403292176, 1.2798353909465023], "output": [0, 0, 1] }, { "input": [2.8847736625514404, 2.946502057613169], "output": [0, 0, 1] }, { "input": [1.4156378600823043, 2.5514403292181074], "output": [0, 0, 1] }, { "input": [1.045267489711934, 2.526748971193416], "output": [0, 0, 1] }, { "input": [2.5144032921810697, 2.1563786008230457], "output": [0, 0, 1] }, { "input": [3.045267489711934, 1.7983539094650207], "output": [0, 0, 1] }, { "input": [2.366255144032922, 2.9341563786008233], "output": [0, 0, 1] }, { "input": [1.5020576131687242, 3.0576131687242802], "output": [0, 0, 1] }, { "input": [0.5390946502057612, 2.711934156378601], "output": [0, 0, 1] }, { "input": [-0.300411522633745, 2.5761316872427984], "output": [0, 0, 1] }, { "input": [-0.7942386831275722, 2.563786008230453], "output": [0, 0, 1] }, { "input": [-1.1646090534979425, 1.181069958847737], "output": [0, 0, 1] }, { "input": [-1.1275720164609055, 0.5637860082304529], "output": [0, 0, 1] }, { "input": [-0.5226337448559671, 0.46502057613168746], "output": [0, 0, 1] }, { "input": [-0.4115226337448561, -0.05349794238683104], "output": [0, 0, 1] }, { "input": [-0.1646090534979425, -0.7325102880658434], "output": [0, 0, 1] }, { "input": [0.4650205761316871, -0.8436213991769544], "output": [0, 0, 1] }, { "input": [0.8106995884773661, -1.164609053497942], "output": [0, 0, 1] }, { "input": [0.32921810699588466, -1.3004115226337447], "output": [0, 0, 1] }, { "input": [1.1687242798353907, -1.127572016460905], "output": [0, 0, 1] }, { "input": [2.1316872427983538, -1.362139917695473], "output": [0, 0, 1] }, { "input": [1.7119341563786008, -0.6954732510288063], "output": [0, 0, 1] }, { "input": [2.5267489711934155, -0.8930041152263373], "output": [0, 0, 1] }, { "input": [2.8971193415637857, -0.8930041152263373], "output": [0, 0, 1] }, { "input": [2.6378600823045266, -0.6460905349794236], "output": [0, 0, 1] }, { "input": [3.2427983539094645, -0.5349794238683125], "output": [0, 0, 1] }, { "input": [3.8477366255144028, 0.02057613168724303], "output": [0, 0, 1] }, { "input": [3.390946502057613, 0.02057613168724303], "output": [0, 0, 1] }, { "input": [3.4403292181069953, 0.3415637860082307], "output": [0, 0, 1] }, { "input": [3.7983539094650203, 0.6502057613168727], "output": [0, 0, 1] }, { "input": [3.526748971193415, 0.983539094650206], "output": [0, 0, 1] }, { "input": [3.452674897119341, 1.4526748971193417], "output": [0, 0, 1] }, { "input": [3.502057613168724, 1.7242798353909468], "output": [0, 0, 1] }, { "input": [3.415637860082304, 2.205761316872428], "output": [0, 0, 1] }, { "input": [2.736625514403292, 2.292181069958848], "output": [0, 0, 1] }, { "input": [1.9465020576131686, 2.403292181069959], "output": [0, 0, 1] }, { "input": [1.8230452674897117, 2.60082304526749], "output": [0, 0, 1] }, { "input": [3.008230452674897, -1.288065843621399], "output": [0, 0, 1] }, { "input": [1.699588477366255, -1.016460905349794], "output": [0, 0, 1] }, { "input": [2.045267489711934, -0.9053497942386829], "output": [0, 0, 1] }, { "input": [1.8724279835390945, -1.2263374485596705], "output": [0, 0, 1] }]
		},
		{ name: "Peterson and Barney",
			parent: "Three classes test",
			stepsPerFrame: 6,
			inputLayer: { neuronCount: 2, names: ["F1","F2"] },
			outputLayer: { neuronCount: 10, "activation": "sigmoid", names:"IY,IH,EH,AE,AH,AA,AO,UH,UW,ER".split(",")}
		},
		{
			name: "Auto-Encoder for linear data",
			stepsPerFrame: 1,
			iterationsPerClick: 1,
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
			"stepsPerFrame": 250,
			"learningRate": 0.01,
			"iterationsPerClick": 10000,
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
		{ "name": "Auto-Encoder 4D", "learningRate": 0.05, "data": [{ "input": [1, 0, 0, 0], "output": [1, 0, 0, 0] }, { "input": [0, 1, 0, 0], "output": [0, 1, 0, 0] }, { "input": [0, 0, 1, 0], "output": [0, 0, 1, 0] }, { "input": [0, 0, 0, 1], "output": [0, 0, 0, 1] }], "inputLayer": { "neuronCount": 4, "names": ["in1", "in2", "in3", "in4"] }, "outputLayer": { "neuronCount": 4, "activation": "sigmoid", "names": ["out1", "out2", "out3", "out4"] }, "hiddenLayers": [{ "neuronCount": 2, "activation": "sigmoid" }], "netLayers": [{ "activation": "sigmoid", "neuronCount": 2 }, { "activation": "linear", "neuronCount": 1 }, { "neuronCount": 2, "activation": "sigmoid" }] }
	];
	export function getNames(): string[] {
		return presets.map(p => p.name).filter(c => c !== "Default");
	}
	export function exists(name: string) {
		return presets.filter(p => p.name === name)[0] !== undefined;
	}
	export function get(name: string): Configuration {
		let chain: any[] = [];
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
		let parent = get(parentName);
		let outconf: any = {};
		for (let prop in sim.config) {
			if (sim.config[prop] !== parent[prop]) outconf[prop] = sim.config[prop];
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
	
	export function loadPetersonBarney() {
		function parseBarney(data: [double, double, int][]) {
			let relevantData = data.map(row => (
				{
					input:row,
					output:Util.arrayWithOneAt(10, row.pop())
				}
			));
			normalizeInputs(relevantData);
			let preset = presets.filter(p => p.name === "Peterson and Barney")[0];
			preset.data = relevantData;
			//presets.forEach(preset => preset.data && normalizeInputs(preset.data));
		}
		// include peterson_barney_data for faster page load
		let dataStr = "NrBMBYBpVAOSCMBdSZ4QAyJcAzATmg12h1wUX3R3EIVgHZJccA2EhBrcHAVjtC9IvNqES4sI1AygIMTVjlhY5rSItThZCMbBor5kPagiIMUfDlBN8JS6lxMEvMXJpqEuDhitiYrqwpQXA8aCnpCUBpXWCEo1F4VBjUWVFY6VgoeaRV0yGzgViE5KA1Cpzl1PnheeAY+NV41es0hfChjYHAKWBJOuHU2qzVWNXtgXnDQcJ8E8M8zKzj6bBNwADooYJV9ShJ4rpV8SL5i2ApUidlz5jYOBiy2XnX4TmehAsz11wRGtlkMPAygxXBhCGVeEJGJAWlcjFBYeAPM4jDRnh5iKiHBYhgkVI5FgkOF5CXgsDBZFZ0OBijQthg4nwnAx4JdIeszoRLgxipl8jgGE5avy0h5QJIBcVKmVlIgdFUEh0hLD2VDCIixmpOkizFhOrh4PhcXhYByjMb+IhwAZTmaEOBvGQguRVmA1DBaZp6e6aBxYNQEq98BdTohBbdUL0w2ICjzEHlPsSJWkpUwpMBkogYAq4fgmCqoXUaE5zFiuoRjmWDZAjTWyH0sOMmohJqTW/aMU7vtBcJ68PsHa7wH4MOSu/tR6Q0ipYCkmYgo5dWAx1h5lyKM6A7axTUxY1uxWmBdosGUaauzPBtVKOuOjLgXnWTKbDQ/DXwMOsk3bZhNP8VcFBMg/EA10XRgV4wiILY2AQTYFygblrm4AUVzOR40lkQRhCeGsgV9GExBVHosFhasSifCYnAkNtWDtAk3BMOD0HIqxP3dBYDlYWQaPTVgtx4oR014FdDXzdx1gqHpiztbN+hsChxlAMY0MonV7RtTQt2KbpSXIC8YB2Z9oGsO00QXPxfXg+hGUDbtOG0iMJlNDxZSXJIPn+dZ5jEdMuEfORnlKWD4NgDYi2nIxmg/XUy0SWL9QfbRjX00ExCbZjxH/NtPywmZhhM0ITDicUh3JUsDgNdZIknA54voJg2SSVSlzXTy0gEsMUJTLNfLYFdXGUnDpGKQEc0FSSzAPHNhVlFVTShOD1QkxT8C/MsBlrJS6LGdKYs8AE+AfQbVMYsBwpMxzkBMN9oE4MzNGyhB8Ecg5ugvZ6YOkK6uA3B543awotxLW7hK1BEaDW0arwIuQ+isdUxi7KELTg8Imm8tsOBpUlswgcIrC2LQhzOAMTT2KcJjapzmwcpyd2jDd2HjeBYyw4LHqzPVoizeGHEbRsCrkRTdntUayA8XSztASIZddIaYDoMg6FrLi1xIT4nGZz4VGzPjdYoPiKlPPgKBBGEaDqM26TMPm8GVPaHDqCxYYY39ic8TsTC3Fi+zx3shw2OIMDES5wCYcVQ4FLdwgZ2Nin+z4XLMUSZpGOjlVDIoLa9R8qDLXT2g2471DWpHve7UYzXGB17Jos7hyy3H9kqa7zqIIy8Azygx356DKdGBdusKDFkxHms/jxdQiNDX6yPVO2QhhW8TGaR23WXtTimCUl1Pr38aOCd4wPJdSyAWkyNksx7Jo9SnEng8UuT4DYOHzpcg2eVmBU/V4wqZ06oE/K5XEGtEg6ZYBXV+I+GUeUcxaHWjoFcdtzwCGTvqKGTca74nXrwEuzhNITFuh2XGdEthyzboIAyEgHrk3oGTXAGwPC2HvuEFkNNXAMBIEue4e47jxivkzT8UxkHDQzJ+bwa0xAQm7uw+aoUNjLS0nKbmWl7KlX1HBaGlFAIfR0NXOk1l94STQQFX80w7qdzxmNSh6BapOkoLZLoThgz01kPxJmxRhSfHmGPDxCB5Tph1ObRECglF4CIkwfoFB1xKQRHYZW4ggLKJ0IdVAz1oDZzbhk6wncGJxkqk4WUdUokz2KrFJSily5dybmdBixjynTEgqvEyBNObPV7vbSgZMGb2j4R1Fs38BkBLHs2Bg4S+T/zIlqLUux/79CBDYJYhE1KvClu7aijpjKYBPj2AIrQiBON4LIfAWA2QnKjsM2Unw6CQKZq8OWfltAxNwv6HOXQOALG1J8qEVIawVgRhUnmnhxYHJBXpck9cyAlUpNIZhMZ+oLkBnGAJIxQx5irKeQWN1hBLISB4FwbZZAiVxlsVuhM7p9nItCN69wggCi0S2Ye/1axnhGNbTQ+YOV4ArKorou0NpCDyEpBQFosKvF/O2AY0tqR+xfNAequxxQMNNGS+laQ/6Ax3HadhnwPAEj8hUDmf4jCkTYAYM1TszCzM5b1KsnVZRKUII6ho4hQWFBUOHXGywlbFUfoCWhJJnp0QjrDIub0yGWJpiA56mC2SYI6QAhcahPiDT8Z84GYiUW/HGslM0bLN6IgMLocycgfZVikUChwao84v0mgdNscFBoSsJgZegtDlK32lTzbYlMqpqspuAIMlygYBV6JNVNAUqFQD4nBJwVChKmxhJav88EeQXg1NXZ1El0orhtREtdv8dG7hhMnGu8qOxmJkq4JEP4rCqXFEVMAEcQ5gUUUQfYZATlfUKBe1gw9B7OH6RMKUi7pBNqmvBMo645D4QSG0OCJACxGHCUvMtj5+hrQsAk4ychsVdCwu6odcovYRLuk+ztEcd5tw9gUxJnAyYuDtG5PgnChCtXjMy1wcdzVZjA8AWcWY4NUyMNy2oCj3nnjaFEtEH1Sz6ihOvEIG6BpqVU/aLZ/Z1rbEDdNGAT7uJhgwoURldMCjic+YbVC9bODjVTPmqyYkzTanSrdRK60XUOF0C7TQrxQIN1NMSB87rKM9isSBTutcggVQlkQdVcJxRnL4CrZFnySBnlGnyrwHnguUQGHGc9AUGnlnEC2hwEHgjNLJD2PstAO4D3pEl+DzHGpsDWlrYz2r3FJSZg+DgQ0/L9YkXZ+y+ihUfkmuuzORIMNwWiq0OTK8uivHlApqtRDqriBLHWugvZEHmMrYIKxO0exODIKp/wfblifp6pwW6erEF3J1ltrNuW5AKAZYglY0HHKMDNIiH4fLG4BJLdWvCOj0qZSbEFK0ZXlvWQfiF5O+NA2n1M7aN+uWlyyAmhrf4CEAHqNU3xL+WYNhWekCtj7lOAoBIc5oOgchuzakre9ya+p0rVOOYgl0Z0H5+c/HQA+kRMw0boixG7XRnSMZKk1qiUbLiymekMgT8LeuyTHr9HQY8oza5mlEpDUEi7XhrDJ3zNZj5NhnO62mKLzER1F4C3JQ53TxTekEa0A8Iv032DRRXfgVhszugiyM+JNeSx11KaRSgHnR80GMFd9V5OQwSrsIuNcejVI9rTd23HCFy2d7YzJHAe14xwHhhcePUD53oMy8ION0y1h0MJ6EOg0VpEUiu5mUybaBQLqCX5VrBPjBJAV6oVobfEtIzLR+Sv73QDeTRvwg43ogVDZGVwGKCj/0S0zE72uLxgxhPNkT4ZETeCy30HzaBzRqT9IQ4HMg9JxDnw4W6cAzuc0ED+xuylGOZQgFa0ih2W5FEV323wfUqCD2bzEWbHwBPzwQvFiEk06gok6EJVUCrDqDxTwARGNFoEQRYTqVei1wcWsCpTiGFEqhF06WbGCG4VYwVTl2UA+i3yUCukE1jCPXtAviNg+niggV1hNiJFNUkx0n3VvScE9xN1gzUgtUoiYxx0IQfiKUIT93JTUXQBJTei2EMzqnLW/ycnDWwm3z8FKm3zmwnzEXgKExml0AiiolClQz1HwPwXGT0mYTUIfAdzyS2HNhoxXC2AJDegfU930OoTnEDB7EiMKAyHASUCDnEB6xlAvTlmgwfEUi3EN0VHWnzhVG8PEAkL80SO1BUBWBHzLjlHXhxl+DSXhziNxhIIo32GQKX020EB9ASBXEiBJTZAfFdyAIEyyMug3GKSgMlCZVeyKx1x4mEMcOdXRHeW5zaGyiLT4zkJ/HfHjw+kEz5xxzhzlgkDyVelOVdDd2CIFHdGrCQgMkOPL1HACkrAKAonAKUA4GOWsJUFbEb2olfhzGzlONhG1RLBP0kJI3Z3Mj+3wNcwMQ6nsjtyeHhPyhMAfVf3OjNDgE7kQK2Dd1Y0fGUh/V+kqycgKWI0uDLR7g3DwwxIZD+i8UjytDHjzCnXBBwFsFknSxijjCBJLmQLIlTALkyyrFJ0MxH1IgtFcBz32iTysH6OgHcMoUEWsE/wI0flVEHT8CJLuEfl0XY3NXxPDjNHJIwGixTTZPQBzQsICgbwmJgIgTTRzFOJB3gQRHfRVBDRhGeDKWWxUWc2BUnBHylHwyHXrRlO2KkOSRqRln2VwIX1hQmDghxPXxNT/wHjiAIL1PSVfXFCr2AAr0Mj+gvVvT6mkDlIITEWpLoHenGzhJLESJVCEILmREHy6HSgBSHzAUoj5F+E03EwTnzwvhljySCCdMtkyWTMhFOw4R7EQgFEiEDwSPI1GNkE9z8n1XiOr11lZJyDMGm03EvCWKwj5RJT11KPBxrgsHbPl13z53CFaIPj8KpXdEd3N2CBOAGUEBHQ8QgFnPSQCSXOeMMGDWXLlBeUjCjxmlIhXWrBCTTyW2rEE0SmPx0WVBFngzDDh3/TDHdVAmCCfRoSCKq3OMzI9RMmYI8BBHgn3CtBnR/nrS9WAQPOgz1BPywsBLrQxG3W2MGngrGBwwIxUQUJXIf31SI3YkyUiyuM6V/xoSOQjkqD6PxPuhItaNiHWm33vz+klJ109SZNMwGGBGGICUqIhFOnkHWiBLKMT0Gmkm2IeVUn1BLHwxJAxSbDXGUMTJbDqOrDgCqx8ruQCO00Hjqg9IgBiKYw9xiJYLsWAyV0YTETCi+23IzG0HbwmCw3+Uk1VRrCyxsD6wh3+3wN5Dh1wpRWhQMimzAnlWUi3GvlbIMh1DqkOwgG7CXEymcEyj4lQMHDXL0WEzt27jPAdO1DFH3UcDtH0WvwfinzvTmEQREivWrVvjRPirgCfWI2Umku7kEDWjJi4Vvl1KMI4mHjhjIo3BehbAg0b1jnlUbwEAKLKCoBs1UwhBPVUEfHdMyoBzMESIqInjUlclI2moslJXlIoz8FaK7B6M6V0RxM6S+CuNZBCjiD9yUC+JLIzDWSPGrwTQIErnuN0v+MWPaDXWjn+rIje3ME2xc11GpuAmppXBwPitOOwVK13l1jqLgAMjRK5siBWsC2gF/3PkfjCPcDCyclnExmmGHlgEqPFCgE+Bh1RTETyGMrNAgWRGGxlHEswOYovFcXkT6BP2ZqB2NkxTMUchHy3EZ2eDQo22xiuklTe3ekjJoQjhgy/XlMiznXFq4gjlRrhXok12JEIAgW3mp3KB+pQNDPEpGpe0cutMsJHzWX4vUlKgbnRhKuyrWrAhaknDF0yW2okndp6SJjl2ZkEAYOkAyvtA2GxuAGZL8zojjwbvyvBWBAMAcP8LBCWINj9KnlSQLicp0VTi2jHPtGRPLHmpKr8FHJMBFz7CbzYOr0/j+l+NgL6HEkinYq9CzCvPUjGnGCwvLLOm7IwPMXQFVr+WsHjK9SO0HSJj2t9zlwKXisV3dEMG3xVhDwzCFFotDytApwzAEBrMOBgUwW+stAbN5gLgeRQSWl3J0SFyz04UjPbF6XtwXxjOwl8tzoVUlzCKwq4iCJiOzggCrobvKkIG3w4jNJGi8qzSFA3IbtcADkMtAveS10qFCS5iHrlC4vJnzhHwsFwQ4Bg0lSSFdqCCf0oWbsySqxxmCHi0NLMOTJuHoDoYE1uQAxcLEOc25U93e0FMQcw3UCZvS34vqmZgbjoGFAbhSChU5kYU7nzJomeP/N+FDrYE1veX8V0m1AsGDNPGND5F7QbmWBwtDkVTB2cddEM0AnhvY3sXAugFOPJM4mbHccaypIWE4H/rzPztlCP1rriikJbIDhgOEZIy7JnDUNZHtDAguAqZimUk6IE1eAEKUH7OsJ4mEz13LOBG3j3OXGrmGfRmPPxDN2q0qdFhoitwYb51eH8XdguHTp5lieyTGggBVz7wgB/pGStDSsM3BRpoCTtsR0QaPu/VJFCdbndndDRLq2sBL2rwiapNqhRG31eF+kb1yDHj708BbsnLVpbOBfivQKDMohYP7NmozDYTUMQkMeS2YG7LYFZBwe8dSbl0Hud0AsoMBgCoYvuN2xb11m9OOcMCBP1X3S4UmM6H8UqCPt1i2MjumE00zApD0m4Wwjbjfo0LzIaf0xyaCB3ljD/oSs9S8er2EpmiNqWMhyWKmFBZ+F4vUGqRdFpeh0mICwBghVSapVDgsTbknOUniyJPincmiKMIjk9wKFVmsELN1b8iFJVBXLtjwUGX70mJrgAksaKTh3QcnoGDgBjP0kajGxdwVTqt/w2sRP9vLou1MhUu4NrAgPqy4I+l0j8hvRqB/iqN8YrG9MNKGabKzFsrAc+WcrojsGqK9M9dvJbCR2YAad5bOSRaiOUhItcWCGAxYSPjXqdba04xVoqBP38KGmgdqOMZKbW2VpH0kTvyHl3iKW8u4VpN5a2F2JafNi4nJDo1LIxJxm5C0JiPDVfUAv9p/t2I9Z+MmKCSrP3RPNAm1CkOctHYUNjlI2I04DqeYDWZiZ5fPlSbacJSHIHgd2fizP9p2c4lAn3DtBJXTDOfjE5L/OxjmI+KMcRAAmVeHbjpIVZp/aErDG20VEOT1cAj9ham7S6IXycRDLiArtwlwbZAjjSfL0PgGGeMPktK6YYYgWJVzfSWoxojKDkHupzGQL5JfitFB1Ipnfo2Zs9uQ6bDEZt15FI2CKb3O1SbaQTIxKfzekHKIf4U/Jpm2uJkuAxRzPOsZTZdGNjn+fzqXlMrpaXSI6bFqsryxiXbqXY1MA3fgnIBjMtGsBA62FOIOCjHIIlvJHifL0FZxncf2EnLtfePyab3XCQ4yT6UdKmBXV12wh5PEHLbPtw50AtDKOvxkfvMxfoEdCQCAA===";
		parseBarney(JSON.parse(LZString.decompressFromBase64(dataStr)));
		return;
		
		$.get("lib/peterson_barney_data").then((strData:string) => {
			let cols = {
				gender:0,
				speaker:1,
				phonemeNum:2,
				phonemeAscii:3,
				F0:4,
				F1:5,
				F2:6,
				F3:7
			}
			let isNum = [true,true,true,,true,true,true,true];
			let data = strData.split("\n")
				.filter(row => row.indexOf("#") !== 0)
				.filter(row => row.trim().length > 2 && row.indexOf("*") < 0)
				.map(row => row.split(/\s+/).map((n,i) => i in isNum ? parseFloat(n):n));
			let relevantData:any = data.map(row => 
				[+row[cols.F1]/10, +row[cols.F2]/10, +row[cols.phonemeNum]]);
			parseBarney(relevantData);
		});
	}
	
	function normalizeInputs(data:TrainingData[]) {
		let i = Util.bounds2dTrainingsInput(data);
		data.forEach(data => data.input = [(data.input[0]-i.minx)/(i.maxx-i.minx),(data.input[1]-i.miny)/(i.maxy-i.miny)]);
	}
}