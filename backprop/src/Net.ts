module Net {
	type int = number;
	type double = number;

	var NonLinearity = {
		sigmoid: (x: double) => 1 / (1 + Math.exp(-x)),
		sigDiff: (x: double) => x * (1 - x)
	}

	function makeArray<T>(len: int, supplier: () => T): T[] {
		var arr = new Array<T>(len);
		for (let i = 0; i < len; i++) arr[i] = supplier();
		return arr;
	}


	export class NeuralNet {
		layers: Neuron[][];
		inputs: InputNeuron[];
		outputs: OutputNeuron[];
		conns: NeuronConnection[] = [];
		learnRate: number = 0.01;
		constructor(counts: int[]) {
			this.inputs = makeArray(counts[0], () => new InputNeuron());
			var hidden = makeArray(counts[1], () => new Neuron());
			this.outputs = makeArray(counts[2], () => new OutputNeuron());
			this.layers = [this.inputs, hidden, this.outputs];
			var onNeuron = new InputNeuron(1);
			this.inputs.push(onNeuron);
			var startWeight = () => Math.random() - 0.5;
			console.log(this.layers);
			for (let i = 0; i < this.layers.length - 1; i++) {
				let inLayer = this.layers[i];
				let outLayer = this.layers[i + 1];

				for (let input of inLayer) for (let output of outLayer) {
					var conn = new Net.NeuronConnection(input, output, startWeight());
					input.outputs.push(conn);
					output.inputs.push(conn);
					this.conns.push(conn);
				}
			}
		}
		setInputs(inputVals: double[]) {
			if (inputVals.length != this.inputs.length - 1) throw "invalid input size";
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
			for (let conn of this.conns) {
				(<any>conn)._tmpw = conn.getDeltaWeight(this.learnRate);
			}
			for (let conn of this.conns) {
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

		weightedInputs() {
			var output = 0;
			for (let conn of this.inputs) {
				output += conn.inp.getOutput() * conn.weight;
			}
			return output;
		}
		getOutput() {
			return NonLinearity.sigmoid(this.weightedInputs());
		}

		getError() {
			var δ = 0;
			for (let output of this.outputs) {
				δ += output.out.getError() * output.weight;
			}
			return δ * NonLinearity.sigDiff(this.getOutput());
		}
	}
	export class InputNeuron extends Neuron {
		constructor(public input: number = 0) {
			super();
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

		getOutput() {
			return Math.max(Math.min(super.weightedInputs(), 1), 0);
		}
		getError() {
			let oup = this.getOutput();
			return NonLinearity.sigDiff(oup) *
				(this.targetOutput - oup);
		}
	}
}