import { int, double } from "./main";
import { InputLayerConfig, LayerConfig, OutputLayerConfig } from "./Presets";
import { makeArray } from "./Util";
import { Configuration, TrainingData, TrainingDataEx } from "./Configuration";

/**
 * Simple implementation of a neural network (multilayer perceptron)
 *
 * Uses stochastic gradient descent with squared error as the loss function
 */
export namespace Net {
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
	};
	/** an activation function (non-linearity) and its derivative */
	interface ActivationFunction {
		f: (x: double) => double;
		df: (x: double) => double;
	}
	/** list of known activation functions */
	export var NonLinearities: { [name: string]: ActivationFunction } = {
		sigmoid: {
			f: x => 1 / (1 + Math.exp(-x)),
			df: x => {
				x = 1 / (1 + Math.exp(-x));
				return x * (1 - x);
			}
		},
		tanh: {
			f: x => tanh(x),
			df: x => {
				x = tanh(x);
				return 1 - x * x;
			}
		},
		linear: {
			f: x => x,
			df: x => 1
		},
		relu: {
			f: x => Math.max(x, 0),
			df: x => (x <= 0 ? 0 : 1)
		},
		lrelu: {
			f: x => (x > 0 ? x : x * 0.1),
			df: x => (x <= 0 ? 0 : 1)
		},
		// used for Rosenblatt Perceptron (df is fake and unimportant)
		"threshold (≥ 0)": {
			f: x => (x >= 0 ? 1 : 0),
			df: x => 1
		}
	};

	/**
	 * A training method for a neural network.
	 *
	 * optionally returns WeightsSteps for displaying of intermediate data in the NetworkVisualization
	 *
	 * if trainSingle is null, only batch training is possible
	 */
	export interface TrainingMethod {
		trainAll: (
			net: NeuralNet,
			data: TrainingDataEx[]
		) => WeightsStep[] | null;
		trainSingle:
			| ((net: NeuralNet, data: TrainingData) => WeightsStep | null)
			| null;
	}

	/** list of training methods for each Configuration#type */
	export const trainingMethods: {
		[type: string]: { [name: string]: TrainingMethod };
	} = {
		nn: {
			"Batch Training": {
				trainAll: (net, data) => net.trainAll(data, false, false),
				trainSingle: null
			},
			"Online Training": {
				trainAll: (net, data) => net.trainAll(data, true, false),
				trainSingle: (net, data) => net.train(data, true, false)
			},
			"Time Delayed Training": {
				trainAll: (net, data) => net.trainAll(data, true, false),
				trainSingle: (net, data) => net.train(data, true, false)
			}
		},
		/** A Perceptron is a special type of NeuralNet that has an input layer with 3 neurons (incl. bias), an output layer with 1 neuron, and no hidden layers */
		perceptron: {
			"Rosenblatt Perceptron": {
				trainAll(net, data) {
					return data.map(val =>
						this.trainSingle!(net, val)
					) as WeightsStep[];
				},
				trainSingle: (net, val) => {
					// this algorithm is equivalent to what the neural network would do via net.train(data, true, true)
					// (when the output neuron has this activation function:
					//   {f: x => (x >= 0) ? 1 : 0, df: x => 1})
					net.setInputsAndCalculate(val.input);
					const outp = net.outputs[0];
					outp.targetOutput = val.output[0];
					const err = outp.targetOutput - outp.output;
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
					if (!(net as any).tmpStore)
						(net as any).tmpStore = {
							c: 1,
							w: net.connections.map(c => c.weight),
							u: net.connections.map(c => 0)
						};
					const vars = (net as any).tmpStore;
					const storeWeightSteps = true; // false for optimizing (but then can't use Configuration#drawArrows)
					if (
						net.layers.length !== 2 ||
						net.outputs.length !== 1 ||
						net.outputs[0].activation !== "threshold (≥ 0)"
					)
						throw Error("can only be used for single perceptron");
					if (storeWeightSteps) var weights: WeightsStep[] = [];
					for (const val of data) {
						const y = val.output[0] ? 1 : -1;
						const x = val.input.concat([1]);
						let yReal = 0;
						for (let i = 0; i < x.length; i++)
							yReal += x[i] * vars.w[i];
						if (y * yReal <= 0) {
							for (let i = 0; i < x.length; i++) {
								vars.w[i] += net.learnRate * y * x[i];
								vars.u[i] += net.learnRate * y * vars.c * x[i];
							}
							if (storeWeightSteps)
								weights!.push({
									dataPoint: val,
									weights: x.map(
										(x, i) => vars.w[i] - vars.u[i] / vars.c
									)
								});
						}
						++vars.c;
					}
					net.connections.forEach(
						(conn, i) =>
							(conn.weight = vars.w[i] - vars.u[i] / vars.c)
					);
					return weights!;
				},
				trainSingle: null
			},
			/** from http://www.jmlr.org/papers/volume3/crammer03a/crammer03a.pdf , page 965
			 * equivalent to Rosenblatt Perceptron except for how G is defined (see pdf)
			 */
			"Binary MIRA": {
				trainAll(net, data) {
					return data.map(d => this.trainSingle!(net, d)!);
				},
				trainSingle: (net, val) => {
					net.setInputsAndCalculate(val.input);
					const outp = net.outputs[0];
					const x = val.input;
					const y = val.output[0] == 1 ? 1 : -1;
					const yGot = net.outputs[0].output == 1 ? 1 : -1;
					//const err = (outp.targetOutput - outp.output);
					const G = (x: double) => Math.min(Math.max(0, x), 1);
					const err = G(
						(-y * net.outputs[0].weightedInputs) /
							x.reduce((a, b) => a + b * b, 0)
					);
					for (const conn of outp.inputs) {
						conn.weight +=
							net.learnRate * err * y * conn.inp.output;
					}
					var weights = net.connections.map(conn => conn.weight);
					return { weights, dataPoint: val };
				}
			}
		}
	};

	/**
	 * intermediate result of a single data point training step and the resulting weights vector
	 *
	 * used to visualize the weights vector in the perceptron demo
	 */
	export interface WeightsStep {
		dataPoint: TrainingData | null;
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
		isTDNN: boolean = false;
		/** a flat list of all the neuron connections */
		connections: NeuronConnection[] = [];
		constructor(
			input: InputLayerConfig,
			hidden: LayerConfig[],
			output: OutputLayerConfig,
			public learnRate: number,
			timeDelayed?: number[],
			startWeight = () => Math.random() - 0.5,
			public startWeights?: double[] | null
		) {
			let nid = 0;
			this.inputs = makeArray(
				input.neuronCount,
				i => new InputNeuron(nid++, i, input.names[i])
			);
			// If timeDelayed -> TDNN
			if (timeDelayed) this.isTDNN = true;
			this.layers.push(this.inputs.slice());
			for (var layer of hidden) {
				this.layers.push(
					makeArray(
						layer.neuronCount,
						i =>
							timeDelayed != undefined
								? new TimeDelayedNeuron(
										layer.activation,
										nid++,
										i,
										timeDelayed[hidden.indexOf(layer)]
								  )
								: new Neuron(layer.activation, nid++, i)
					)
				);
			}
			this.outputs = makeArray(
				output.neuronCount,
				i =>
					new OutputNeuron(
						output.activation,
						nid++,
						i,
						output.names[i]
					)
			);
			this.layers.push(this.outputs);
			for (let i = 0; i < this.layers.length - 1; i++) {
				const inLayer = this.layers[i];
				const outLayer = this.layers[i + 1];
				if (timeDelayed == undefined)
					inLayer.push(new InputNeuron(nid++, -1, "Bias", 1));
				// If next layer is Output
				if (
					timeDelayed != undefined &&
					outLayer[0] instanceof OutputNeuron
				) {
					for (let inputI = 0; inputI < inLayer.length; inputI++) {
						const input = inLayer[inputI];
						var conn = new Net.NeuronConnection(
							input,
							outLayer[inputI]
						);
						input.outputs.push(conn);
						outLayer[inputI].inputs.push(conn);
						this.connections.push(conn);
					}
				} else {
					for (const input of inLayer)
						for (const output of outLayer) {
							var conn = new Net.NeuronConnection(input, output);
							input.outputs.push(conn);
							output.inputs.push(conn);
							this.connections.push(conn);
						}
				}
				if (timeDelayed == undefined)
					this.biases[i] = inLayer.pop() as InputNeuron;
			}
			if (!this.startWeights) {
				this.startWeights = this.connections.map(
					c => (c.weight = startWeight())
				);
			} else
				this.startWeights.forEach(
					(w, i) => (this.connections[i].weight = w)
				);
		}
		setInputsAndCalculate(inputVals: double[]) {
			for (let i = 0; i < this.inputs.length; i++)
				this.inputs[i].output = inputVals[i];
			for (const layer of this.layers.slice(1))
				for (const neuron of layer) neuron.calculateOutput();
		}

		initTDNN(inputVals: double[][]) {
			for (let i = 0; i < this.inputs.length; i++)
				this.inputs[i].outputVector = inputVals[i];
			for (const layer of this.layers.slice(1))
				for (const neuron of layer) {
					for (const conn of neuron.inputs) {
						for (
							var i = 0;
							i <=
							conn.inp.outputVector.length - neuron.timeDelayed;
							i++
						) {
							neuron.outputVector[i] = 0;
						}
					}
				}
			// for (let i = 0; i < this.inputs.length; i++)
			// 	this.inputs[i].outputVector = inputVals[i];
		}
		setInputVectorsAndCalculate(inputVals: double[][]) {
			for (let i = 0; i < this.inputs.length; i++) {
				this.inputs[i].outputVector = inputVals[i];
				for (let j = 0; j < inputVals.length; j++) {
					this.inputs[i].outputVector[j] = inputVals[i][j] /= 255;
				}
			}
			for (let layer of this.layers.slice(1, this.layers.length - 1)) {
				// Get output for TDNN Neuron
				for (let neuron of layer) neuron.calculateOutput();
			}
			const lastHiddenLayer = this.layers[this.layers.length - 2];
			const outputLayer = this.outputs;
			for (let neuron of outputLayer) {
				//Get output for Output neuron
				neuron.calculateOutputTDNN(
					lastHiddenLayer[outputLayer.indexOf(neuron)].outputVector
				);
			}
			this.outputs.map(output => output.output);
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
		trainAll(
			data: TrainingDataEx[],
			individual: boolean,
			storeWeightSteps: boolean
		) {
			let weights: WeightsStep[] | null = null;
			if (storeWeightSteps) weights = [];
			if (!individual)
				for (const conn of this.connections) conn.zeroDeltaWeight();
			for (const val of data) {
				const step = this.train(val, individual, storeWeightSteps);
				if (weights) weights.push(step!);
			}
			if (!individual)
				for (const conn of this.connections) conn.flushDeltaWeight();
			return weights;
		}

		/** if flush is false, only calculate deltas but don't reset or add them */
		train(val: TrainingDataEx, flush = true, storeWeightSteps: boolean) {
			if (this.isTDNN) this.setInputVectorsAndCalculate(val.inputVector!);
			else this.setInputsAndCalculate(val.input);
			// console.log("Output length: " + this.outputs.length);
			for (let i = 0; i < this.outputs.length; i++) {
				// console.log(val.output[i]);
				this.outputs[i].targetOutput = val.output[i];
			}

			for (let i = this.layers.length - 1; i > 0; i--) {
				// console.log("Layer " + i);
				for (const neuron of this.layers[i]) {
					neuron.calculateError();
					for (const conn of neuron.inputs) {
						if (flush) conn.zeroDeltaWeight();
						conn.addDeltaWeight(this.learnRate);
					}
				}
			}

			var w = storeWeightSteps
				? {
						weights: this.connections.map(
							conn => conn.weight + conn.deltaWeight
						),
						dataPoint: val
				  }
				: null;
			if (flush)
				for (const conn of this.connections) conn.flushDeltaWeight();
			return w;
		}
	}

	/** a weighted connection between two neurons */
	export class NeuronConnection {
		public weight = 0;
		public weightVector?: number[];
		/** cached delta weight for training */
		deltaWeight = NaN;
		deltaWeightVector: number[] = [];
		constructor(
			/**Input neuron of this connection */ public inp: Neuron,
			/**Output neuron of this connection */ public out: Neuron
		) {}
		zeroDeltaWeight() {
			this.deltaWeight = 0;
			this.deltaWeightVector = [];
		}
		addDeltaWeight(learnRate: double) {
			if (this.out instanceof TimeDelayedNeuron) {
				var tmp = 0;
				for (
					var outputNeuronIndex = 0;
					outputNeuronIndex < this.out.outputVector!.length;
					outputNeuronIndex++
				) {
					for (
						var time = 0;
						time < this.out.timeDelayed;
						time++ // deltaWeight for each time step
					) {
						var tmp1 = 0;
						if (this.deltaWeightVector[time] == null)
							this.deltaWeightVector[time] = 0;
						tmp1 =
							learnRate *
							this.out.errorVector[outputNeuronIndex] *
							this.inp.outputVector![outputNeuronIndex + time];
						this.deltaWeightVector[time] += tmp1;
						//tmp+=learnRate*this.out.errorVector[outputNeuronIndex]*this.inp.outputVector![outputNeuronIndex+time];
					}
				}
				// console.log("Delta weight: ");
				// console.log(this.deltaWeightVector);
				for (
					var time = 0;
					time < this.out.timeDelayed;
					time++ // deltaWeight for each time step
				) {
					this.deltaWeightVector[
						time
					] /= this.out.outputVector!.length; // Average deltaWeight
					// this.deltaWeightVector[time] *=learnRate;
				}
				// console.log("Average Delta weight: ");
				// console.log(this.deltaWeightVector);
			} else if (this.inp instanceof TimeDelayedNeuron) {
				for (let time = 0; time < this.weightVector!.length; time++) {
					var tmp = 0;
					tmp =
						learnRate *
						this.out.error *
						this.inp.outputVector![time];
					this.deltaWeightVector[time] = tmp;
				}
				// console.log(this.deltaWeightVector);
			} else
				this.deltaWeight +=
					learnRate * this.out.error * this.inp.output;
		}
		flushDeltaWeight() {
			if (this.weightVector != undefined) {
				// console.log("Before update weight");
				// console.log(this.weightVector);
				// console.log("---Deltaweight Vector---");
				// console.log(this.deltaWeightVector);
				for (
					var time = 0;
					time < this.out.timeDelayed;
					time++ // deltaWeight for each time step
				) {
					this.weightVector![time] += this.deltaWeightVector[time];
				}
				this.deltaWeightVector = [];
				// console.log("After updated weight");
				// console.log(this.weightVector);
			} else {
				this.weight += this.deltaWeight;
				this.deltaWeight = NaN; // set to NaN to prevent flushing bugs
			}
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
		/** Variable for TDNN */
		public weightedInputsVector: double[] = [];
		public errorVector: number[] = [];
		//public weightVector?: number[];
		public outputVector: double[] = [];
		public timeDelayed: int = 0;
		constructor(
			public activation: string,
			public id: int,
			public layerIndex: int
		) {}
		calculateWeightedInputs() {
			this.weightedInputs = 0;
			for (const conn of this.inputs) {
				this.weightedInputs += conn.inp.output * conn.weight;
			}
		}
		calculateOutput() {
			this.calculateWeightedInputs();
			this.output = NonLinearities[this.activation].f(
				this.weightedInputs
			);
		}
		calculateOutputTDNN(inputVals: number[]) {
			this.timeDelayed = inputVals.length;
			this.weightedInputs = 0;
			for (const conn of this.inputs) {
				if (conn.weightVector == undefined) {
					conn.weightVector = [];
					for (let i = 0; i < inputVals.length; i++)
						conn.weightVector.push(Math.random() - 0.5);
				}
				for (const weight of conn.weightVector!) {
					for (const input of inputVals)
						this.weightedInputs += weight * input;
				}
			}
			this.output = NonLinearities[this.activation].f(
				this.weightedInputs
			);
		}
		calculateError() {
			var δ = 0;
			for (const output of this.outputs) {
				δ += output.out.error * output.weight;
			}
			this.error =
				δ * NonLinearities[this.activation].df(this.weightedInputs);
			// console.log("Error of neuron: %f",[this.error]);
		}
	}
	/** an input neuron (no inputs, fixed output value, can't be trained) */
	export class InputNeuron extends Neuron {
		constant: boolean = false; // value won't change
		constructor(
			id: int,
			layerIndex: int,
			public name: string,
			constantOutput?: number
		) {
			super(null!, id, layerIndex);
			if (constantOutput !== undefined) {
				this.output = constantOutput;
				this.constant = true;
			}
		}
		calculateOutput() {
			throw Error("input neuron");
		}
		calculateWeightedInputs() {
			throw Error("input neuron");
		}
		calculateError() {
			throw Error("input neuron");
		}
	}
	/** an output neuron (error calculated via target output */
	export class OutputNeuron extends Neuron {
		targetOutput: double = 0;
		constructor(
			public activation: string,
			id: int,
			layerIndex: int,
			public name: string
		) {
			super(activation, id, layerIndex);
		}

		calculateError() {
			this.error =
				NonLinearities[this.activation].df(this.weightedInputs) *
				(this.targetOutput - this.output);
			// console.log("Target: " +this.targetOutput);
			// console.log("Output: " + this.output);
			// console.log("Weighted inputs: " + this.weightedInputs);
			// console.log("Derivative of function: " + NonLinearities[this.activation].df(this.weightedInputs))
			// console.log("Error of output neuron: " + this.error);
			// ⇔ for perceptron: (this.targetOutput - this.output)
			// ⇔  1 if (sign(w*x) =  1 and y = -1);
			//   -1 if (sign(w*x) = -1 and y =  1);
			//    0 if (sign(w*x) = y)
			//    where x = input vector, w = weight vector, y = output label (-1 or +1)
		}
	}

	export class TimeDelayedNeuron extends Neuron {
		constructor(
			public activation: string,
			public id: int,
			public layerIndex: int,
			public timeDelayed: int
		) {
			super(activation, id, layerIndex);
		}
		calculateWeightedInputs() {
			this.weightedInputsVector = [];

			for (const conn of this.inputs) {
				if (!conn.weightVector) {
					conn.weightVector = [];
					for (let i = 0; i < this.timeDelayed; i++)
						if (conn.weightVector[i] == null)
							conn.weightVector[i] = Math.random() - 0.5;
				}
				for (
					var i = 0;
					i <= conn.inp.outputVector.length - this.timeDelayed;
					i++
				) {
					if (this.weightedInputsVector[i] == null)
						this.weightedInputsVector[i] = 0;
					for (var j = i; j < i + this.timeDelayed; j++) {
						this.weightedInputsVector[i] +=
							conn.inp.outputVector[j] *
							conn.weightVector![j - i];
					}
				}
			}
		}
		calculateOutput() {
			this.calculateWeightedInputs();
			for (var i = 0; i < this.weightedInputsVector.length; i++) {
				this.outputVector[i] = NonLinearities[this.activation].f(
					this.weightedInputsVector[i]
				);
			}
		}
		calculateError() {
			var δ = 0;
			if (this.outputs[0].out instanceof OutputNeuron) {
				// If next layer is output layer
				for (const output of this.outputs) {
					for (
						var weightIndex = 0;
						weightIndex < output.weightVector!.length;
						weightIndex++
					) {
						var tmpδ = 0;
						tmpδ +=
							output.out.error *
							output.weightVector![weightIndex];
						// console.log("Error "+ output.out.error);
						// console.log("Weight: "+ output.weightVector![weightIndex]);
						// console.log("Tmp Delta "+ tmpδ);
						// console.log("Weighted input vector: " + this.weightedInputsVector[weightIndex]);
						this.errorVector[weightIndex] =
							tmpδ *
							NonLinearities[this.activation].df(
								this.weightedInputsVector[weightIndex]
							);
					}
				}
				// console.log("Error of hidden layer next to output:");
				// console.log(this.errorVector);
			} else {
				// Loop all output of the current neuron
				var tmpδVector: number[] = [];
				for (
					var outputVectorIndex = 0;
					outputVectorIndex < this.outputVector!.length;
					outputVectorIndex++
				) {
					for (const conn of this.outputs) { // Loop all connection to next layer
						var outputNeuron = conn.out;
						for (
							var outputIndexOfNextLayer =
								outputVectorIndex -
								outputNeuron.timeDelayed +
								1;
							outputIndexOfNextLayer <= outputVectorIndex &&
							outputIndexOfNextLayer <
								outputNeuron.outputVector!.length;
							outputIndexOfNextLayer++ // Loop all time value connected to next layer
						) {
							if (outputIndexOfNextLayer >= 0) {
								if (tmpδVector[outputVectorIndex] == undefined)
									tmpδVector[outputVectorIndex] = 0;
								// console.log("outputNeuronError " + outputNeuron.errorVector[outputIndexOfNextLayer]);
								// console.log("+++++++++weightVector index : " + (outputVectorIndex-outputIndexOfNextLayer));
								// console.log("---Weight vector " + conn.weightVector![outputVectorIndex-outputIndexOfNextLayer]);
								tmpδVector[outputVectorIndex] +=
									outputNeuron.errorVector[
										outputIndexOfNextLayer
									] *
									conn.weightVector![
										outputVectorIndex -
											outputIndexOfNextLayer
									];
								// console.log("-------------TmpDelta Vector 2 - " + tmpδVector[outputVectorIndex]);
							}
						}
					}
					this.errorVector[outputVectorIndex] =
						tmpδVector[outputVectorIndex] *
						NonLinearities[this.activation].df(
							this.weightedInputsVector[outputVectorIndex]
						);
				}
				// console.log(this.errorVector);
				// for (const output of this.outputs) {
				// 	δ += output.out.error * output.weight;
				// }
			}
			// this.error =
			// 	δ * NonLinearities[this.activation].df(this.weightedInputs);
		}
	}
}

export default Net;
