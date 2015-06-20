module Net {
	type int = number;
	type double = number;
	let _nextGaussian:double;
	export function randomGaussian(mean = 0, standardDeviation = 1) {

		if (_nextGaussian !== undefined) {
			var nextGaussian = _nextGaussian;
			_nextGaussian = undefined;
			return (nextGaussian * standardDeviation) + mean;
		} else {
			let v1:double, v2:double, s:double, multiplier:double;
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
			df: (x: double) => x * (1 - x)
		},
		tanh: {
			f: (x: double) => tanh(x),
			df: (x: double) => 1 - x * x
		},
	}
	export var activation: ActivationFunction;
	export function setLinearity(name: string) {
		activation = NonLinearities[name];
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
		constructor(counts: int[], inputnames: string[], learnRate: number, 
			bias = true, startWeight = () => Math.random() - 0.5, weights?: double[]) {
			this.learnRate = learnRate;
			counts = counts.slice();
			if (counts.length < 2) throw "Need at least two layers";
			let nid = 0;
			this.inputs = makeArray(counts.shift(), () => new InputNeuron(nid, inputnames[nid++]));
			this.layers.push(this.inputs);
			while (counts.length > 1) {
				this.layers.push(makeArray(counts.shift(), () => new Neuron(nid++)));
			}
			this.outputs = makeArray(counts.shift(), () => new OutputNeuron(nid++));
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
		setInputs(inputVals: double[]) {
			if (inputVals.length != this.inputs.length - +this.bias) throw "invalid input size";
			for (let i = 0; i < inputVals.length; i++)
				this.inputs[i].input = inputVals[i];
		}
		getOutput(inputVals: double[]) {
			this.setInputs(inputVals);
			return this.outputs.map(output => output.getOutput());
		}

		train(inputVals: double[], expectedOutput: double[]) {
			this.setInputs(inputVals);
			for (var i = 0; i < this.outputs.length; i++)
				this.outputs[i].targetOutput = expectedOutput[i];
			for (let conn of this.connections) {
				(<any>conn)._tmpw = conn.getDeltaWeight(this.learnRate);
			}
			for (let conn of this.connections) {
				conn.weight += (<any>conn)._tmpw;
			}
		}
	}

	export class NeuronConnection {
		constructor(public inp: Neuron, public out: Neuron, public weight: double) {

		}
		getDeltaWeight(learnRate: double) {
			return learnRate * this.out.getError() * this.inp.getOutput();
		}
	}
	export class Neuron {
		public inputs: NeuronConnection[] = [];
		public outputs: NeuronConnection[] = [];
		constructor(public id: int) { }

		weightedInputs() {
			var output = 0;
			for (let conn of this.inputs) {
				output += conn.inp.getOutput() * conn.weight;
			}
			return output;
		}
		getOutput() {
			return activation.f(this.weightedInputs());
		}

		getError() {
			var δ = 0;
			for (let output of this.outputs) {
				δ += output.out.getError() * output.weight;
			}
			return δ * activation.df(this.getOutput());
		}
	}
	export class InputNeuron extends Neuron {
		constructor(id: int, public name: string, public input: number = 0) {
			super(id);
		}
		weightedInputs() {
			return this.input;
		}
		getOutput() {
			return this.input;
		}
	}
	export class OutputNeuron extends Neuron {
		targetOutput: double;

		getError() {
			let oup = this.getOutput();
			return activation.df(oup) *
				(this.targetOutput - oup);
		}
	}
}