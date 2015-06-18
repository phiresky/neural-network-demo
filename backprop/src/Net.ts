module Net {
	type int = number;
	type double = number;
	
	var NonLinearity  = {
		sigmoid : (x:double) => 1 / (1 + Math.exp(-x)),
		sigDiff : (x:double) => x * (1-x)
	}
	export var learnRate = 0.01;

	export class NeuronConnection {
		constructor(public inp:Neuron, public out:Neuron, public weight:double) {
			
		}
		getDeltaWeight() {
			return learnRate * this.out.getError() * this.inp.getOutput();
		}
	}
	interface Neuron {
		getOutput():double;
		getError():double;
	}
	export class HiddenNeuron implements Neuron {
		public inputs:NeuronConnection[] = [];
		public outputs:NeuronConnection[] = [];
		δ:double;
		
		getOutputRaw() {
			var output = 0;
			for(let conn of this.inputs) {
				output += conn.inp.getOutput() * conn.weight;
			}
			return output;
		}
		getOutput() {
			return NonLinearity.sigmoid(this.getOutputRaw());
		}
		
		getError() {
			var δ = 0;
			for(let output of this.outputs) {
				δ += output.out.getError() * output.weight;
			}
			return δ * NonLinearity.sigDiff(this.getOutput());
		}
	}
	
	export class InputNeuron implements Neuron {
		public outputs:NeuronConnection[] = [];
		constructor(public input:number = 0) {}
		getOutputRaw() {
			return this.input;
		}
		getOutput() {
			return this.input;
		}
		getError() {
			return 0;
		}
	}
	export class OutputNeuron extends HiddenNeuron {
		public inputs:NeuronConnection[] = [];
		targetOutput:double;
		getOutput() {
			return Math.max(Math.min(super.getOutputRaw(),0.999),0.001);
		}
		getError() {
			let oup = Math.abs(this.getOutput());
			return NonLinearity.sigDiff(oup) * 
				(this.targetOutput - oup);
		}
	}
}