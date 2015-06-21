module Net {
	type int = number;
	type double = number;
	let _nextGaussian: double;
	export function randomGaussian(mean = 0, standardDeviation = 1) {

		if (_nextGaussian !== undefined) {
			var nextGaussian = _nextGaussian;
			_nextGaussian = undefined;
			return (nextGaussian * standardDeviation) + mean;
		} else {
			let v1: double, v2: double, s: double, multiplier: double;
			do {
				v1 = 2 * Math.random() - 1; // between -1 and 1
				v2 = 2 * Math.random() - 1; // between -1 and 1
				s = v1 * v1 + v2 * v2;
			} while (s >= 1 || s == 0);
			multiplier = Math.sqrt(-2 * Math.log(s) / s);
			_nextGaussian = v2 * multiplier;
			return (v1 * multiplier * standardDeviation) + mean;
		}

	};
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
	var NonLinearities: { [name: string]: ActivationFunction } = {
		sigmoid: {
			f: (x: double) => 1 / (1 + Math.exp(-x)),
			df: (x: double) => { x = 1 / (1 + Math.exp(-x)); return x * (1 - x) }
		},
		tanh: {
			f: (x: double) => tanh(x),
			df: (x: double) => { x = tanh(x); return 1 - x * x }
		},
		linear: {
			f: (x: double) => x,
			df: (x: double) => 1
		}
	}

	function makeArray<T>(len: int, supplier: () => T): T[] {
		var arr = new Array<T>(len);
		for (let i = 0; i < len; i++) arr[i] = supplier();
		return arr;
	}

	// back propagation code adapted from https://de.wikipedia.org/wiki/Backpropagation
	export class NeuralNet {
		layers: Neuron[][] = [];
		inputs: InputNeuron[];
		outputs: OutputNeuron[];
		connections: NeuronConnection[] = [];
		learnRate: number = 0.01;
		bias: boolean;
		constructor(layout: LayerConfig[], inputnames: string[], learnRate: number,
			bias = true, startWeight = () => Math.random() - 0.5, weights?: double[]) {
			this.learnRate = learnRate;
			layout = layout.slice();
			if (layout.length < 2) throw "Need at least two layers";
			let nid = 0;
			this.inputs = makeArray(layout.shift().neuronCount, () => new InputNeuron(nid, inputnames[nid++]));
			this.layers.push(this.inputs);
			while (layout.length > 1) {
				var layer = layout.shift();
				this.layers.push(makeArray(layer.neuronCount, () => new Neuron(layer.activation, nid++)));
			}
			let outputLayer = layout.shift();
			this.outputs = makeArray(outputLayer.neuronCount, () => new OutputNeuron(outputLayer.activation, nid++));
			this.layers.push(this.outputs);
			this.bias = bias;
			for (let i = 0; i < this.layers.length - 1; i++) {
				let inLayer = this.layers[i];
				let outLayer = this.layers[i + 1];
				if (bias)
					inLayer.push(new InputNeuron(nid++, "1 (bias)", 1));

				for (let input of inLayer) for (let output of outLayer) {
					var conn = new Net.NeuronConnection(input, output, startWeight());
					input.outputs.push(conn);
					output.inputs.push(conn);
					this.connections.push(conn);
				}
			}
			if (weights) weights.forEach((w, i) => this.connections[i].weight = w);
		}
		setInputsAndCalculate(inputVals: double[]) {
			if (inputVals.length != this.inputs.length - +this.bias) throw "invalid input size";
			for (let i = 0; i < inputVals.length; i++)
				this.inputs[i].output = inputVals[i];
			for (let layer of this.layers.slice(1)) for (let neuron of layer)
				neuron.calculateOutput();
		}
		getOutput(inputVals: double[]) {
			this.setInputsAndCalculate(inputVals);
			return this.outputs.map(output => output.output);
		}

		train(inputVals: double[], expectedOutput: double[]) {
			this.setInputsAndCalculate(inputVals);
			for (var i = 0; i < this.outputs.length; i++)
				this.outputs[i].targetOutput = expectedOutput[i];
			for (let i = this.layers.length - 1; i > 0; i--) {
				for (let neuron of this.layers[i]) {
					neuron.calculateError();
					for (let conn of neuron.inputs)
						conn.calculateDeltaWeight(this.learnRate);
				}
			}
			for (let conn of this.connections) conn.weight += conn.deltaWeight;
		}
	}

	export class NeuronConnection {
		public deltaWeight = 0;
		constructor(public inp: Neuron, public out: Neuron, public weight: double) {

		}
		calculateDeltaWeight(learnRate: double) {
			this.deltaWeight = learnRate * this.out.error * this.inp.output;
		}
	}
	export class Neuron {
		public inputs: NeuronConnection[] = [];
		public outputs: NeuronConnection[] = [];
		public weightedInputs = 0;
		public output = 0;
		public error = 0;
		constructor(public activation: string, public id: int) { }

		calculateWeightedInputs() {
			this.weightedInputs = 0;
			for (let conn of this.inputs) {
				this.weightedInputs += conn.inp.output * conn.weight;
			}
		}
		calculateOutput() {
			this.calculateWeightedInputs();
			this.output = NonLinearities[this.activation].f(this.weightedInputs);
		}

		calculateError() {
			var δ = 0;
			for (let output of this.outputs) {
				δ += output.out.error * output.weight;
			}
			this.error = δ * NonLinearities[this.activation].df(this.weightedInputs);
		}
	}
	export class InputNeuron extends Neuron {
		constructor(id: int, public name: string, output: number = 0) {
			super(null, id);
			this.output = output;
		}
		calculateOutput() { }
		calculateWeightedInputs() { }
		calculateError() { }
	}
	export class OutputNeuron extends Neuron {
		targetOutput: double;

		calculateError() {
			this.error = NonLinearities[this.activation].df(this.weightedInputs) * (this.targetOutput - this.output);
		}
	}
}