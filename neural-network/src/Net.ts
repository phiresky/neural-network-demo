// this neural network uses stochastic gradient descent with the squared error as the loss function
module Net {
	var tanh = function(x: double) {
		if (x === Infinity) {
			return 1;
		} else if (x === -Infinity) {
			return -1;
		} else {
			var y = Math.exp(2 * x);
			return (y - 1) / (y + 1);
		}
	}
	interface ActivationFunction {
		f: (x: double) => double,
		df: (x: double) => double
	}
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
		// used for Rosenblatt Perceptron (fake df)
		"threshold (≥ 0)": {
			f: x => (x>=0)?1:0,
			df: x => 1
		}
	}

	export module Util {
		export function makeArray<T>(len: int, supplier: (i: int) => T): T[] {
			var arr = new Array<T>(len);
			for (let i = 0; i < len; i++) arr[i] = supplier(i);
			return arr;
		}
	}

	// back propagation code adapted from https://de.wikipedia.org/wiki/Backpropagation
	export class NeuralNet {
		layers: Neuron[][] = [];
		inputs: InputNeuron[];
		outputs: OutputNeuron[];
		biases: InputNeuron[] = [];
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
		// get root-mean-square error
		getLoss(expectedOutput: double[]) {
			let sum = 0;
			for (let i = 0; i < this.outputs.length; i++) {
				const neuron = this.outputs[i];
				sum += Math.pow(neuron.output - expectedOutput[i], 2);
			}
			return Math.sqrt(sum / this.outputs.length);
		}
		
		/** if individual is true, train individually, else train as a set */
		trainAll(data: TrainingData[], individual: boolean) {
			if(!individual) for (const conn of this.connections) conn.zeroDeltaWeight();
			for (const val of data) {
				this.train(val.input, val.output, individual);
			}
			if(!individual) for (const conn of this.connections) conn.flushDeltaWeight();
		}

		/** if flush is false, only calculate deltas but don't reset or add them */
		train(inputVals: double[], expectedOutput: double[], flush = true) {
			this.setInputsAndCalculate(inputVals);
			for (var i = 0; i < this.outputs.length; i++)
				this.outputs[i].targetOutput = expectedOutput[i];
			for (let i = this.layers.length - 1; i > 0; i--) {
				for (const neuron of this.layers[i]) {
					neuron.calculateError();
					for (const conn of neuron.inputs) {
						if(flush) conn.zeroDeltaWeight();
						conn.addDeltaWeight(this.learnRate);
					}
				}
			}
			if(flush) for (const conn of this.connections) conn.flushDeltaWeight();
		}
	}

	export class NeuronConnection {
		private deltaWeight = NaN; public weight = 0;
		constructor(public inp: Neuron, public out: Neuron) {

		}
		zeroDeltaWeight() {
			this.deltaWeight = 0;
		}
		addDeltaWeight(learnRate: double) {
			this.deltaWeight += learnRate * this.out.error * this.inp.output;
		}
		flushDeltaWeight() {
			this.weight += this.deltaWeight;
			this.deltaWeight = NaN;
		}
	}
	export class Neuron {
		public inputs: NeuronConnection[] = [];
		public outputs: NeuronConnection[] = [];
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
	export class InputNeuron extends Neuron {
		constant: boolean = false; // value won't change
		constructor(id: int, layerIndex: int, public name: string, constantOutput?: number) {
			super(null, id, layerIndex);
			if (constantOutput !== undefined) {
				this.output = constantOutput;
				this.constant = true;
			}
		}
		calculateOutput() { }
		calculateWeightedInputs() { }
		calculateError() { }
	}
	export class OutputNeuron extends Neuron {
		targetOutput: double;
		constructor(public activation: string, id: int, layerIndex: int, public name: string) {
			super(activation, id, layerIndex);
		}

		calculateError() {
			this.error = NonLinearities[this.activation].df(this.weightedInputs) * (this.targetOutput - this.output);
		}
	}
}