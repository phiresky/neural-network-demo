///<reference path='Simulation.ts' />
enum SimulationType {
	BinaryClassification,
	AutoEncoder
}
module Presets {
	let presets: { [name: string]: {} } = {
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
	export function printDataPoints() {
		return (<any>window).simulation.config.data.map(
			e => '{input:[' + e.input.map(x=> x.toFixed(2)) 
			+ '], output:[' + e.input.map(x=> x.toFixed(2)) + ']},').join("\n")
	}
}