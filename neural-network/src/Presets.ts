///<reference path='Simulation.ts' />
enum SimulationType {
	BinaryClassification,
	AutoEncoder
}
interface Configuration {
	[name: string]: any;
	data?: TrainingData[];
}
module Presets {
	let presets: { [name: string]: Configuration } = {
		"Default": {
			stepsPerFrame: 50,
			learningRate: 0.05,
			showGradient: false,
			bias: true,
			autoRestartTime: 5000,
			autoRestart: true,
			iterationsPerClick: 5000,
			simType: SimulationType.BinaryClassification,
			data: <TrainingData[]>[
				{ input: [0, 0], output: [0] },
				{ input: [0, 1], output: [1] },
				{ input: [1, 0], output: [1] },
				{ input: [1, 1], output: [0] }
			],
			netLayers: <LayerConfig[]>[
				{ neuronCount: 2 },
				{ neuronCount: 2, activation: "sigmoid" },
				{ neuronCount: 1, activation: "sigmoid" }
			]
		},
		"XOR": {
			//defaults only
		},
		"Circular data": {
			"netLayers": [
				{ "neuronCount": 2 },
				{ "neuronCount": 3, "activation": "sigmoid" },
				{ "neuronCount": 1, "activation": "sigmoid" }
			],
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
		"Auto-Encoder": {
			simType: SimulationType.AutoEncoder,
			stepsPerFrame: 1,
			iterationsPerClick: 1,
			data: <TrainingData[]>[
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
			netLayers: <LayerConfig[]>[
				{ neuronCount: 2 },
				{ neuronCount: 1, activation: "sigmoid" },
				{ neuronCount: 2, activation: "linear" }
			],
			showGradient: true
		}
	}
	export function get(name: string) {
		return $.extend(true, {}, presets["Default"], presets[name]);
	}
	export function printPreset(parent = presets["Default"]) {
		let config = <Configuration>(<any>window).simulation.config;
		let outconf: any = {};
		for (let prop in config) {
			if (config[prop] !== parent[prop]) outconf[prop] = config[prop];
		}
		outconf.data = config.data.map(
			e => '{input:[' + e.input.map(x=> x.toFixed(2))
				+ '], output:[' +
				(config["simType"] == SimulationType.BinaryClassification
					? e.output
					: e.input.map(x=> x.toFixed(2)))
				+ ']},').join("\n");
		return outconf;
	}
}