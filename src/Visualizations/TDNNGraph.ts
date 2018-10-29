import * as $ from "jquery";
import * as vis from "vis";
import { int, double } from "../main";
import { Visualization } from "./Visualization";
import Net from "../Net";
import Simulation from "../Simulation";
import NetworkVisualization from "./NetworkVisualization";
import { TrainingData, Configuration, TrainingDataEx } from "../Configuration";

interface Point3d {
	x: double;
	y: double;
	z: double;
	style?: double;
}

/** step for [[Simulation#forwardPass]] */
export interface TDNNGraphUpdate {
	nodes: any[];
	layerNumber?: number;
	currentTime?: number;
}
/** show the network as a ordered left-to-right graph using arrow color, width and label to show weights */
export default class TDNNGraph implements Visualization {
	actions = ["TDNN Graph"];
	graph: any; // vis.Network
	nodes: any; // vis.DataSet
	edges: any; // vis.DataSet
	net!: Net.NeuralNet;
	container = document.createElement("div");

	showbias!: boolean;
	currentlyDisplayingForwardPass = false;
	biasBeforeForwardPass = false;

	offsetBetweenLayers = 2;
	xyToConnectionTDNN: { [xcommay: string]: [int, int, double] } = {};
	calculatedNetwork = false;
	currentLayer = 1;
	currentNeuron = 0;
	updates: TDNNGraphUpdate[] = [{ nodes: [] }];
	// canvas=document.createElement("canvas");
	// ctx=this.canvas.getContext("2d");
	constructor(public sim: Simulation) {
		this.instantiateGraph();
		// this.container.appendChild(this.canvas);
		//this.instantiateGraph1();
	}
	instantiateGraph1() {
		// hack to get grayscale colors
		vis.Graph3d.prototype._hsv2rgb = (h: double, s: double, v: double) => {
			h = Math.min(h, 250) | 0;
			return "rgb(" + [h, h, h] + ")";
		};
		// hack to disable axis drawing
		vis.Graph3d.prototype._redrawAxis = function() {};
		this.graph = new vis.Graph3d(this.container, undefined, {
			style: "bar",
			showPerspective: false,
			cameraPosition: { horizontal: -0.001, vertical: Math.PI / 2 },
			width: "100%",
			height: "100%",
			xLabel: "Layer",
			yLabel: "Neuron",
			zLabel: "",
			showGrid: true,
			axisColor: "red",
			xBarWidth: 0.9,
			yBarWidth: 0.9,
			xCenter: "50%",
			legendLabel: "Weight",
			//zMin: 0,
			//zMax: 5,
			tooltip: (point: Point3d) => {
				const [neuron, time, value] = this.xyToConnectionTDNN[
					point.x + "," + point.y
				];
				let inStr: string, outStr: string;
				return (
					"ID : " +
					neuron +
					" - Time Delay: +" +
					time +
					" (" +
					value.toFixed(2) +
					")"
				);
			},
			//xValueLabel: (x: int) => this.xToLayer[x] || "",
			yValueLabel: (y: int) => ((y | 0) == y ? y + 1 : ""),
			zValueLabel: (z: int) => ""
		});
	}
	instantiateGraph() {
		this.nodes = new vis.DataSet([], { queue: true });
		this.edges = new vis.DataSet([], { queue: true });
		const graphData = {
			nodes: this.nodes,
			edges: this.edges
		};
		const options = {
			nodes: { shape: "box", physics: false },
			// edges: {
			// 	smooth: { type: "curvedCW", roundness: 0 },
			// 	font: { align: "top", background: "white" }
			// 	/*scaling: {
			// 		label: {min:1,max:2}
			// 	}*/
			// },
			// layout: { hierarchical: { direction: "DU" } },
			physics: {
				stabilization: true
			},
			interaction: { dragNodes: false, hover: true }
		};
		if (this.graph) this.graph.destroy();
		this.graph = new vis.Network(this.container, graphData, options);
		let thisVar = this;
		// this.graph.on("afterDrawing", function (ctx: any) {
		// 	console.log("after drawing");
		// 	console.log(thisVar);
		// 	ctx.lineWidth="10";
		// 	ctx.strokeStyle="red";
		// 	ctx.rect(thisVar.nodes[0].x,thisVar.nodes[0].y,500,850);
		// 	ctx.stroke();
		// });
	}
	private edgeId(conn: Net.NeuronConnection) {
		return conn.inp.id * this.net.connections.length + conn.out.id;
	}
	onNetworkLoaded1(net: Net.NeuralNet, forceRedraw = false) {
		if (
			!forceRedraw &&
			this.net &&
			this.net.layers.length == net.layers.length &&
			this.net.layers.every(
				(layer, index) => layer.length == net.layers[index].length
			) &&
			this.showbias === this.sim.state.bias
		) {
			// same net layout, only update
			this.net = net;
			this.onFrame();
			return;
		}
		this.showbias = this.sim.state.bias;
		this.net = net;
		this.drawGraph();
	}
	/** initialize graph nodes and edges */
	drawGraph() {
		this.nodes.clear();
		this.edges.clear();
		const net = this.net;
		for (let lid = 0; lid < net.layers.length; lid++) {
			const layer = net.layers[lid];
			let nid = 1;
			let layerWithBias = layer;
			if (this.showbias && net.biases[lid])
				layerWithBias = layer.concat(net.biases[lid]);
			for (const neuron of layerWithBias) {
				let type = "Hidden Neuron " + nid++;
				let color = "#000";
				if (neuron instanceof Net.InputNeuron) {
					type = "Input: " + neuron.name;
					if (neuron.constant) {
						color = NetworkVisualization.colors.autoencoder.bias;
					} else
						color = NetworkVisualization.colors.autoencoder.input;
				}
				if (neuron instanceof Net.OutputNeuron) {
					type = "Output: " + neuron.name;
					color = NetworkVisualization.colors.autoencoder.output;
				}
				if (
					this.sim.state.type == "nn" &&
					this.sim.currentTrainingDataPoint >= 0
				) {
					let v = 1 - Math.min(Math.max(neuron.output, 0), 1);
					v = (v * 250) | 0;
					color = "rgb(" + [v, v, v] + ")";
				}
				this.nodes.add({
					id: neuron.id,
					label: `${type}`,
					level: lid,
					color: color
				});
			}
		}
		for (const conn of net.connections) {
			this.edges.add({
				id: this.edgeId(conn),
				from: conn.inp.id,
				to: conn.out.id,
				arrows: "to",
				label: conn.weight.toFixed(2)
			});
		}
		this.nodes.flush();
		this.edges.flush();
		this.graph.stabilize();
		this.graph.fit();
	}
	/** calculate the visualization of the individual calculation steps for a single forward pass */
	forwardPass(data: TrainingDataEx) {
		if (!this.calculatedNetwork || this.updates == undefined) {
			this.updates = [{ nodes: [] }];
			this.net.setInputVectorsAndCalculate(data.inputVector!);
			this.calculatedNetwork = true;
			this.currentLayer = 1;
			this.currentNeuron = 0;
			this.parseData(this.net);
		}
		// else{

		// 	this.currentNeuron++;
		// 	if (this.currentNeuron>=this.net.layers[this.currentLayer].length)
		// 	{
		// 		this.currentLayer++;
		// 		if (this.currentLayer>=this.net.layers.length){
		// 			this.currentLayer=1;
		// 		}
		// 		this.currentNeuron=0;
		// 	}
		// }
		this.applyUpdate(this.updates.shift()!);
		// this.graph.setData(this.parseData(this.net));
	}
	applyUpdate(update: TDNNGraphUpdate) {
		this.nodes.update(update.nodes);
		this.nodes.flush();
	}
	onView(previouslyHidden: boolean, action: int) {
		// this.graph.redraw();
		this.graph.fit();
		this.graph.stabilize();
	}
	onHide() {}
	/** parse network layout into weights graph ordering */
	maxHeight = 0;
	parseData(net: Net.NeuralNet) {
		console.log(this.nodes);
		this.xyToConnectionTDNN = {};
		const data: Point3d[] = [];
		let maxy = 0;
		let boxSize = 50;
		let lastUpdate = 1;
		if (!this.calculatedNetwork)
			this.updates = [{ nodes: [], currentTime: 0, layerNumber: 0 }];
		this.createForwardPassStep();
		lastUpdate = 1;
		this.maxHeight = Math.max.apply(
			null,
			net.layers.map(layer => layer.length * boxSize)
		);
		if (!this.calculatedNetwork) this.nodes.clear();
		try {
			for (
				let outputLayer = 0;
				outputLayer < net.layers.length;
				outputLayer++
			) {
				// if (this.calculatedNetwork)
				// {
				// 	if (outputLayer!=this.currentLayer)
				// 	{
				// 		if (outputLayer>this.currentLayer)
				// 			break;
				// 		// nodeID+=net.layers[outputLayer].length*net.layers[outputLayer][0].outputVector.length;
				// 		continue;
				// 	}

				// }
				// console.log("layer " + outputLayer);
				const layer = net.layers[outputLayer];
				const layerY = maxy + this.offsetBetweenLayers * boxSize;
				let maxValue = 0;
				let minValue = 9999999999999;
				for (let i = 0; i < layer.length; i++) {
					for (let j = 0; j < layer[i].outputVector.length; j++) {
						if (maxValue < layer[i].outputVector[j])
							maxValue = layer[i].outputVector[j];
						if (minValue > layer[i].outputVector[j])
							minValue = layer[i].outputVector[j];
					}
				}
				// const maxValue=Math.max.apply(
				// 	null,
				// 	layer.map(neuron=> Math.max.apply(null,neuron.outputVector))//.map(val=> val)))
				// )

				// const minValue=Math.min.apply(
				// 	null,
				// 	layer.map(neuron=> Math.min.apply(null,neuron.outputVector))//.map(val=> val)))
				// )

				for (
					let outputNeuron = 0;
					outputNeuron < layer.length;
					outputNeuron++
				) {
					// console.log("Neuron " + outputNeuron);
					const outN = layer[outputNeuron];
					maxy = Math.max(maxy, layerY + layer.length * boxSize);
					// console.log(outN);
					if (outN instanceof Net.OutputNeuron) {
						const p = {
							x: outputNeuron * boxSize,
							y: layerY + boxSize,
							z: outN.output
						};

						if (this.maxHeight != layer.length * boxSize) {
							// console.log(maxHeight);
							// console.log(layer.length);
							p.y += (this.maxHeight - layer.length) / 2;
						}
						let node = {
							id: outN.id * 1000 + outputNeuron,
							size: boxSize / 2,
							shape: "square",
							color: { background: "rgba(0,0,0," + p.z + ")" }, // + ((p.z-minValue)/(maxValue-minValue)).toFixed(1),
							title: "" + p.z.toFixed(2),
							x: p.x,
							y: -p.y
						};
						if (!this.calculatedNetwork) {
							this.nodes.add(node);
						} else {
							this.updates[lastUpdate + outputNeuron].nodes.push(
								node
							);
							this.updates[
								lastUpdate + outputNeuron
							].currentTime = outputNeuron;
							this.updates[
								lastUpdate + outputNeuron
							].layerNumber = outputLayer;
							// if (outputLayer==this.currentLayer && outputNeuron<=this.currentNeuron)
							// 	this.nodes.update([node]);
						}

						data.push(p);
						this.xyToConnectionTDNN[p.x + "," + p.y] = [
							outN.id,
							0,
							outN.output
						];
						// nodeID++;
					} else {
						for (
							let timeStep = 0;
							timeStep < outN.outputVector.length;
							timeStep++
						) {
							// if (this.calculatedNetwork)
							// 	if (timeStep!=this.currentNeuron)
							// 	{
							// 		console.log("timeStep " + timeStep+ " # " +this.currentNeuron+ " currentNeuron")
							// 		// nodeID++;
							// 		if (timeStep>this.currentNeuron)
							// 			break;
							// 		continue;
							// 	}

							const output =
								outN.outputVector[timeStep] == null
									? 0
									: outN.outputVector[timeStep];
							const p = {
								x: timeStep * boxSize,
								y: layerY + outputNeuron * boxSize,
								z: output
							};

							if (this.maxHeight != layer.length) {
								// console.log(maxHeight);
								// console.log(layer.length);
								p.y += (this.maxHeight - layer.length) / 2;
							}
							let node = {
								id: outN.id * 1000 + timeStep,
								size: boxSize / 2,
								shape: "square",
								color: {
									background:
										"rgba(0,0,0," +
										(
											(p.z - minValue) /
											(maxValue - minValue)
										).toFixed(1) +
										")"
								},
								// fixed:{
								// 	x:true,
								// 	y:true
								// },
								title: "" + p.z.toFixed(2),
								x: p.x,
								y: -p.y
							};
							data.push(p);
							if (!this.calculatedNetwork) {
								this.nodes.add(node);
							} else {
								this.updates[lastUpdate + timeStep].nodes.push(
									node
								);
								this.updates[
									lastUpdate + timeStep
								].currentTime = timeStep;
								this.updates[
									lastUpdate + timeStep
								].layerNumber = outputLayer;
							}
							// else
							// {
							// 	if (outputLayer==this.currentLayer && timeStep<=this.currentNeuron)
							// 		this.nodes.update([node]);
							// }
							this.xyToConnectionTDNN[p.x + "," + p.y] = [
								outN.id,
								timeStep,
								output
							];
							// nodeID++;
						}
					}
					// this.graph.on("afterDrawing", function (ctx: any) {
					// 	console.log("haha");
					// 	ctx.lineWidth="6";
					// 	ctx.strokeStyle="red";
					// 	ctx.rect(data[0].x,data[0].y,500,850);
					// 	ctx.stroke();
					// });
				}
				if (layer[0] instanceof Net.OutputNeuron)
					lastUpdate += layer.length;
				else {
					if (layer[0] instanceof Net.TimeDelayedNeuron)
						lastUpdate += layer[0].outputVector.length;
				}
			}
		} catch (e) {
			console.log(e);
		}
		let thisVar = this;
		// this.graph.on("afterDrawing", function (ctx: any) {
		// 				console.log("haha");
		// 				ctx.lineWidth="1";
		// 				ctx.strokeStyle="red";
		// 				ctx.rect(,data[0].y,500,850);
		// 				ctx.stroke();
		// 			});
		if (!this.calculatedNetwork) this.updates[0].nodes.push(this.nodes);
		this.nodes.flush();
		this.graph.stabilize();
		this.graph.fit();
		// this.ctx!.translate(1,1);
		// this.ctx!.scale(10,10);
		// this.ctx!.translate(-1,-1);
		// for(let i=0;i<data.length;i++)
		// {
		// 	this.ctx!.lineWidth=1;
		// 	this.ctx!.strokeStyle="red";
		// 	this.ctx!.rect(data[i].x-1,data[i].y-1,data[i].x+1,data[i].y+1);
		// 	this.ctx!.stroke();
		// }

		return data;
	}
	createForwardPassStep() {
		for (let i = 1; i < this.net.layers.length; i++) {
			const layer = this.net.layers[i];
			if (layer[0] instanceof Net.OutputNeuron) {
				for (let neuron = 0; neuron < layer.length; neuron++)
					this.updates.push({ nodes: [] });
			} else {
				for (
					let time = 0;
					time < layer[0].outputVector.length;
					time++
				) {
					this.updates.push({ nodes: [] });
					// let update={};
					// for (let neuronNumer=0;neuronNumer<layer.length;neuronNumer++)
					// {
					// 	let neuron=layer[neuronNumer];

					// }
				}
			}
		}
	}

	onNetworkLoaded(net: Net.NeuralNet) {
		if (!net.isTDNN) return;
		console.log("On network loaded TDNN");
		this.actions = ["TDNN Graph"];
		this.net = net;
		this.calculatedNetwork = false;
		// let abc=[[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]];
		// this.net.initTDNN(abc);
		this.parseData(net);
		// this.graph.setData(this.parseData(net));
	}
	onFrame() {
		console.log("On frame TDNN");
		if (!this.net.isTDNN) return;
		this.graph.setData(this.parseData(this.sim.net));
	}
}
