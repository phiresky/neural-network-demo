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
		return $.get("peterson_barney_data").then((strData:string) => {
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
			let relevantData = data.map(row => (
				{
					input:[+row[cols.F1]/10, +row[cols.F2]/10],
					output:Util.arrayWithOneAt(10, +row[cols.phonemeNum] - 1)
				}
			));
			normalizeInputs(relevantData);
			let preset = presets.filter(p => p.name === "Peterson and Barney")[0];
			preset.data = relevantData;
			//presets.forEach(preset => preset.data && normalizeInputs(preset.data));
		});
	}
	
	function normalizeInputs(data:TrainingData[]) {
		let i = Util.bounds2dTrainingsInput(data);
		data.forEach(data => data.input = [(data.input[0]-i.minx)/(i.maxx-i.minx),(data.input[1]-i.miny)/(i.maxy-i.miny)]);
	}
}