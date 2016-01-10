/**
 * Simple implementation of a neural network (multilayer perceptron)
 * 
 * Uses stochastic gradient descent with squared error as the loss function
 */
module Net {
	/** tangens hyperbolicus polyfill */
	const tanh = function(x: double) {
		if (x === Infinity) {
			return 1;
		} else if (x === -Infinity) {
			return -1;
		} else {
			var y = Math.exp(2 * x);
			return (y - 1) / (y + 1);
		}
	}
	/** an activation function (non-linearity) and its derivative */
	interface ActivationFunction {
		f: (x: double) => double,
		df: (x: double) => double
	}
	/** list of known activation functions */
	export var NonLinearities: { [name: string]: ActivationFunction } = {
		sigmoid: {
			f: x => 1 / (1 + Math.exp(-x)),
			df: x => { x = 1 / (1 + Math.exp(-x)); return x * (1 - x) }
		},
		tanh: {
			f: x => tanh(x),
			df: x => { x = tanh(x); return 1 - x * x }
		},
		linear: {
			f: x => x,
			df: x => 1
		},
		relu: {
			f: x => Math.max(x, 0),
			df: x => x <= 0 ? 0 : 1
		},
		// used for Rosenblatt Perceptron (df is fake and unimportant)
		"threshold (≥ 0)": {
			f: x => (x >= 0) ? 1 : 0,
			df: x => 1
		}
	}
	
		
	/**
	 * A training method for a neural network.
	 * 
	 * optionally returns WeightsSteps for displaying of intermediate data in the NetworkVisualization
	 * 
	 * if trainSingle is null, only batch training is possible
	 */
	export interface TrainingMethod {
		trainAll:(net: NeuralNet, data: TrainingData[]) => WeightsStep[];
		trainSingle:(net: NeuralNet, data: TrainingData) => WeightsStep;
	};
	
	/** list of training methods for each Configuration#type */
	export const trainingMethods: { [type: string]: { [name: string]: TrainingMethod } } = {
		"nn": {
			"Batch Training": {
				trainAll: (net, data) => net.trainAll(data, false, false),
				trainSingle: null
			},
			"Online Training": {
				trainAll: (net, data) => net.trainAll(data, true, false),
				trainSingle: (net, data) => net.train(data, true, false)
			}
		},
		/** A Perceptron is a special type of NeuralNet that has an input layer with 3 neurons (incl. bias), an output layer with 1 neuron, and no hidden layers */
		"perceptron": {
			"Rosenblatt Perceptron": {
				trainAll(net, data) { return data.map(val => this.trainSingle(net, val)) },
				trainSingle: (net, val) => {
					// this algorithm is equivalent to what the neural network would do via net.train(data, true, true)
					// (when the output neuron has this activation function:
					//   {f: x => (x >= 0) ? 1 : 0, df: x => 1})
					net.setInputsAndCalculate(val.input);
					const outp = net.outputs[0];
					outp.targetOutput = val.output[0];
					const err = (outp.targetOutput - outp.output);
					for (const conn of outp.inputs) {
						conn.weight += net.learnRate * err * conn.inp.output;
					}
					var weights = net.connections.map(conn => conn.weight);
					return { weights, dataPoint: val };
				}
			},
			"Batch Perceptron": {
				trainAll: (net, data) => net.trainAll(data, false, true),
				trainSingle: null
			},
			/** averaged perceptron (from http://ciml.info/dl/v0_8/ciml-v0_8-ch03.pdf , p.48) */
			"Averaged Perceptron": {
				trainAll: (net, data) => {
					const storeWeightSteps = true; // false for optimizing (but then can't Configuration#drawArrows)
					if (net.layers.length !== 2 || net.outputs.length !== 1 || net.outputs[0].activation !== "threshold (≥ 0)")
						throw Error("can only be used for single perceptron");
					if (storeWeightSteps) var weights: WeightsStep[] = [];
					let u = net.connections.map(w => 0);
					let c = 1;
					for (const val of data) {
						net.train(val, true, storeWeightSteps);
						if (net.outputs[0].error !== 0) {
							const y = val.output[0] === 1 ? -1 : 1;
							u = net.connections.map((conn, i) => u[i] + y * c * conn.inp.output);
							if (storeWeightSteps) weights.push({ dataPoint: val, weights: net.connections.map((conn, i) => conn.weight - u[i] / c) });
						}
						++c;
					}
					net.connections.forEach((conn, i) => conn.weight -= u[i] / c);
					return weights;
				},
				trainSingle: null
			},
			/** from http://www.jmlr.org/papers/volume3/crammer03a/crammer03a.pdf , page 965 
			 * equivalent to Rosenblatt Perceptron except for how G is defined (see pdf)
			 */
			"Binary MIRA": {
				trainAll(net, data) {
					return data.map(d => this.trainSingle(net, d));
				},
				trainSingle: (net, val) => {
					net.setInputsAndCalculate(val.input);
					const outp = net.outputs[0];
					const x = val.input;
					const y = val.output[0] == 1 ? 1 : -1;
					const yGot = net.outputs[0].output == 1 ? 1 : -1;
					//const err = (outp.targetOutput - outp.output);
					const G = (x:double) => Math.min(Math.max(0, x), 1);
					const err = G(-y * net.outputs[0].weightedInputs / x.reduce((a,b) => a + b*b, 0));
					for (const conn of outp.inputs) {
						conn.weight += net.learnRate * err * y * conn.inp.output;
					}
					var weights = net.connections.map(conn => conn.weight);
					return { weights, dataPoint: val };
				}
			}
		}
	}

	/** 
	 * intermediate result of a single data point training step and the resulting weights vector
	 * 
	 * used to visualize the weights vector in the perceptron demo 
	 */
	export class WeightsStep {
		dataPoint: TrainingData;
		weights: number[];
	}
	
	/**
	 * back propagation code adapted from https://de.wikipedia.org/wiki/Backpropagation
	 */
	export class NeuralNet {
		/** layers including input, hidden, and output */
		layers: Neuron[][] = [];
		/** bias input neurons */
		biases: InputNeuron[] = [];
		/** actual input neurons */
		inputs: InputNeuron[];
		outputs: OutputNeuron[];
		/** a flat list of all the neuron connections */
		connections: NeuronConnection[] = [];
		constructor(input: InputLayerConfig, hidden: LayerConfig[], output: OutputLayerConfig, public learnRate: number,
			startWeight = () => Math.random() - 0.5, public startWeights?: double[]) {
			let nid = 0;
			this.inputs = Util.makeArray(input.neuronCount, i => new InputNeuron(nid++, i, input.names[i]));
			this.layers.push(this.inputs.slice());
			for (var layer of hidden) {
				this.layers.push(Util.makeArray(layer.neuronCount, i => new Neuron(layer.activation, nid++, i)));
			}
			this.outputs = Util.makeArray(output.neuronCount, i => new OutputNeuron(output.activation, nid++, i, output.names[i]));
			this.layers.push(this.outputs);
			for (let i = 0; i < this.layers.length - 1; i++) {
				const inLayer = this.layers[i];
				const outLayer = this.layers[i + 1];
				inLayer.push(new InputNeuron(nid++, -1, "Bias", 1));

				for (const input of inLayer) for (const output of outLayer) {
					var conn = new Net.NeuronConnection(input, output);
					input.outputs.push(conn);
					output.inputs.push(conn);
					this.connections.push(conn);
				}
				this.biases[i] = inLayer.pop() as InputNeuron;
			}
			if (!this.startWeights) {
				this.startWeights = this.connections.map(c => c.weight = startWeight());
			} else this.startWeights.forEach((w, i) => this.connections[i].weight = w);
		}
		setInputsAndCalculate(inputVals: double[]) {
			for (let i = 0; i < this.inputs.length; i++)
				this.inputs[i].output = inputVals[i];
			for (const layer of this.layers.slice(1)) for (const neuron of layer)
				neuron.calculateOutput();
		}
		getOutput(inputVals: double[]) {
			this.setInputsAndCalculate(inputVals);
			return this.outputs.map(output => output.output);
		}
		/** get root-mean-square error */
		getLoss(expectedOutput: double[]) {
			let sum = 0;
			for (let i = 0; i < this.outputs.length; i++) {
				const neuron = this.outputs[i];
				sum += Math.pow(neuron.output - expectedOutput[i], 2);
			}
			return Math.sqrt(sum / this.outputs.length);
		}

		/** if individual is true, train individually, else batch train */
		trainAll(data: TrainingData[], individual: boolean, storeWeightSteps: boolean) {
			if (storeWeightSteps) var weights: WeightsStep[] = [];
			if (!individual) for (const conn of this.connections) conn.zeroDeltaWeight();
			for (const val of data) {
				const step = this.train(val, individual, storeWeightSteps);
				if (storeWeightSteps) weights.push(step);
			}
			if (!individual) for (const conn of this.connections) conn.flushDeltaWeight();
			return weights;
		}

		/** if flush is false, only calculate deltas but don't reset or add them */
		train(val: TrainingData, flush = true, storeWeightSteps: boolean) {
			this.setInputsAndCalculate(val.input);
			for (let i = 0; i < this.outputs.length; i++)
				this.outputs[i].targetOutput = val.output[i];
			for (let i = this.layers.length - 1; i > 0; i--) {
				for (const neuron of this.layers[i]) {
					neuron.calculateError();
					for (const conn of neuron.inputs) {
						if (flush) conn.zeroDeltaWeight();
						conn.addDeltaWeight(this.learnRate);
					}
				}
			}
			if (storeWeightSteps) var weights = this.connections.map(conn => conn.weight + conn.deltaWeight);
			if (flush) for (const conn of this.connections) conn.flushDeltaWeight();
			return { weights, dataPoint: val };
		}
	}

	/** a weighted connection between two neurons */
	export class NeuronConnection {
		public weight = 0;
		/** cached delta weight for training */
		deltaWeight = NaN;
		constructor(public inp: Neuron, public out: Neuron) { }
		zeroDeltaWeight() {
			this.deltaWeight = 0;
		}
		addDeltaWeight(learnRate: double) {
			this.deltaWeight += learnRate * this.out.error * this.inp.output;
		}
		flushDeltaWeight() {
			this.weight += this.deltaWeight;
			this.deltaWeight = NaN; // set to NaN to prevent flushing bugs
		}
	}
	/** a single Neuron / Perceptron */
	export class Neuron {
		public inputs: NeuronConnection[] = [];
		public outputs: NeuronConnection[] = [];
		/** weighted sum of inputs without activation function */
		public weightedInputs = 0;
		public output = 0;
		public error = 0;
		constructor(public activation: string, public id: int, public layerIndex: int) { }
		calculateWeightedInputs() {
			this.weightedInputs = 0;
			for (const conn of this.inputs) {
				this.weightedInputs += conn.inp.output * conn.weight;
			}
		}
		calculateOutput() {
			this.calculateWeightedInputs();
			this.output = NonLinearities[this.activation].f(this.weightedInputs);
		}
		calculateError() {
			var δ = 0;
			for (const output of this.outputs) {
				δ += output.out.error * output.weight;
			}
			this.error = δ * NonLinearities[this.activation].df(this.weightedInputs);
		}
	}
	/** an input neuron (no inputs, fixed output value, can't be trained) */
	export class InputNeuron extends Neuron {
		constant: boolean = false; // value won't change
		constructor(id: int, layerIndex: int, public name: string, constantOutput?: number) {
			super(null, id, layerIndex);
			if (constantOutput !== undefined) {
				this.output = constantOutput;
				this.constant = true;
			}
		}
		calculateOutput() { throw Error("input neuron") }
		calculateWeightedInputs() { throw Error("input neuron") }
		calculateError() { throw Error("input neuron") }
	}
	/** an output neuron (error calculated via target output */
	export class OutputNeuron extends Neuron {
		targetOutput: double;
		constructor(public activation: string, id: int, layerIndex: int, public name: string) {
			super(activation, id, layerIndex);
		}

		calculateError() {
			this.error = NonLinearities[this.activation].df(this.weightedInputs) * (this.targetOutput - this.output);
			// ⇔ for perceptron: (this.targetOutput - this.output)
			// ⇔  1 if (sign(w*x) =  1 and y = -1);
			//   -1 if (sign(w*x) = -1 and y =  1);
			//    0 if (sign(w*x) = y)
			//    where x = input vector, w = weight vector, y = output label (-1 or +1)
		}
	}
}
