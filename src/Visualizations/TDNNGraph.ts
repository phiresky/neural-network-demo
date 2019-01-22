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
interface TDNNUpdateLayer {
	boxSize: number;
	maxX: number;
	numberofNeuron: number;
	timeDelayed: number;
}
export interface TDNNGraphUpdate {
	nodes: any[];
	layerNumber?: number;
	currentTime?: number;
	timeDelayed?: number;
	numberofneuron?: number;
	previousIndex?: number;
	layersUpdate: { [layer_id: number]: TDNNUpdateLayer };
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

	alreadySetNet = false;
	offsetBetweenLayers = 2;
	xyToConnectionTDNN: { [xcommay: string]: [int, int, double] } = {};
	calculatedNetwork = false;
	currentLayer = 1;
	currentNeuron = 0;
	updates: TDNNGraphUpdate[] = [{ nodes: [], layersUpdate: {} }];
	minYofLayer: { [layer_id: number]: number } = {};
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
			layout: {
				improvedLayout: false
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
	next = 0;
	/** calculate the visualization of the individual calculation steps for a single forward pass */
	forwardPass(data: TrainingDataEx) {
		if (
			!this.calculatedNetwork ||
			this.updates == undefined ||
			this.next == this.updates.length - 1
		) {
			this.onFrame();
			console.log("First step of forward pass");
			this.next = 1;
			this.calculatedNetwork = true;
			// this.updates = [{ nodes: [] }];
			// console.log("Forward Pass");
			this.minYofLayer = {};
			this.net.setInputVectorsAndCalculate(data.inputVector!);
			this.currentLayer = 1;
			this.currentNeuron = 0;
			this.parseData(this.net);
			this.currentlyDisplayingForwardPass = true;
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
		// console.log(this.updates, this.next);
		if (this.next >= this.updates.length - 1) {
			// this.onFrame();
			// this.next = 1;
		} else {
			if (this.next == this.updates.length - 2) {
				// console.log("Last step");
				this.alreadySetNet = false;
			}
			this.applyUpdate(this.updates[this.next++]);
		}

		// this.graph.setData(this.parseData(this.net));
	}
	applyUpdate(update: TDNNGraphUpdate) {
		// console.log("ApplyUpdate");
		this.graph.off("afterDrawing");
		this.graph.redraw();
		// console.log(update.nodes);
		// console.log(update.nodes);
		if (this.next != 0) {
			// let x1 = update.nodes[0].x - 30;
			// let y2 = update.nodes[update.nodes.length - 1].y - 30;
			// let h1 = 50 * update.numberofneuron! + 5; //update.nodes[0].y+30;
			// let previousLayer = this.updates[update.previousIndex!];
			// let isOutputLayer =
			// 	this.net.layers[update.layerNumber!][0] instanceof
			// 	Net.OutputNeuron;
			let x1,
				y2,
				h1,
				px1 = 0,
				py2 = 0,
				timeDelayed = 0,
				width = 0,
				height = 0;
			// if (!isOutputLayer) {
			// 	px1 = x1;
			// 	// px1=previousLayer.nodes[0].x-30;
			// 	// py1=previousLayer.nodes[0].y+30;
			// 	py2 =
			// 		previousLayer.nodes[previousLayer.nodes.length - 1].y - 30;
			// 	timeDelayed = update.timeDelayed!;
			// 	width = 50 * timeDelayed + 5;
			// 	height = 50 * previousLayer.numberofneuron! + 5;
			// } else {
			// 	h1 = 55;
			// 	// let minValue = 9999999999999;
			// 	// for (let i = 0; i < previousLayer.nodes.length; i++) {
			// 	// 	if (minValue > previousLayer.nodes[i].x-30)
			// 	// 		minValue = previousLayer.nodes[i].x-30;
			// 	// }
			// 	// px1=minValue;
			// 	let firstPreviousLayer = this.updates[
			// 		update.previousIndex! - previousLayer.currentTime!
			// 	];
			// 	px1 = firstPreviousLayer.nodes[0].x - 30;
			// 	// py1=previousLayer.nodes[0].y+30;
			// 	py2 = previousLayer.nodes[update.currentTime!].y - 30;
			// 	// py2=previousLayer.nodes[previousLayer.nodes.length-1].y-30;
			// 	timeDelayed = this.net.layers[previousLayer.layerNumber!][0]
			// 		.outputVector.length;
			// 	width = 50 * timeDelayed + 5;
			// 	height = 55;
			// }
			// console.log(x1, h1, y2);
			// console.log(px1, py2, width, height);
			var thisVar = this;
			if (update.layerNumber != 0)
				this.graph.on("afterDrawing", function(ctx: any) {
					ctx.lineWidth = "6";
					ctx.strokeStyle = "red";
					for (var layer in update.layersUpdate) {
						x1 = update.layersUpdate[layer].maxX - 30;
						y2 = thisVar.minYofLayer[layer] - 30;
						h1 = 50 * update.layersUpdate[layer].numberofNeuron + 5; //update.nodes[0].y+30;
						if (
							!(
								thisVar.net.layers[layer][0] instanceof
								Net.OutputNeuron
							)
						) {
							px1 = x1;
							py2 = thisVar.minYofLayer[Number(layer) - 1] - 30;
							timeDelayed =
								update.layersUpdate[layer].timeDelayed;
							height =
								50 *
									thisVar.net.layers[Number(layer) - 1]
										.length +
								5;
						} else {
							px1 = -30;
							py2 = thisVar.minYofLayer[Number(layer) - 1] + x1;
							timeDelayed =
								thisVar.net.layers[Number(layer) - 1][0]
									.outputVector.length;
							height = 55;
						}
						width = 50 * timeDelayed + 5;
						ctx.rect(x1, y2, 55, h1);
						ctx.stroke();
						ctx.rect(px1, py2, width!, height!);
						ctx.stroke();
						ctx.beginPath();
						ctx.moveTo(x1, y2 + h1);
						ctx.lineTo(px1, py2);
						ctx.moveTo(x1 + 55, y2 + h1);
						ctx.lineTo(px1 + width!, py2);
						ctx.stroke();
					}
					// ctx.rect(x1, y2, 55, h1);
					// ctx.stroke();
					// ctx.rect(px1, py2, width!, height!);
					// ctx.stroke();
					// ctx.beginPath();
					// ctx.moveTo(x1, y2 + h1);
					// ctx.lineTo(px1, py2);
					// ctx.moveTo(x1 + 55, y2 + h1);
					// ctx.lineTo(px1 + width!, py2);
					// ctx.stroke();
				});
		}
		this.nodes.update(update.nodes);
		this.nodes.flush();
		this.graph.stabilize();
	}
	onView(previouslyHidden: boolean, action: int) {
		// this.graph.redraw();
		this.graph.fit();
		// this.graph.stabilize();
	}
	onHide() {}
	/** parse network layout into weights graph ordering */
	maxHeight = 0;
	parseData(net: Net.NeuralNet) {
		// console.log(this.nodes);
		this.xyToConnectionTDNN = {};
		const data: Point3d[] = [];
		let maxy = 0;
		let boxSize = 50;
		let lastUpdate = 1;
		if (!this.calculatedNetwork && !this.currentlyDisplayingForwardPass) {
			this.updates = [
				{ nodes: [], currentTime: 0, layerNumber: 0, layersUpdate: {} }
			];
			this.createForwardPassStep();
		}

		lastUpdate = 1;
		this.maxHeight = Math.max.apply(
			null,
			net.layers.map(layer => layer.length * boxSize)
		);
		if (!this.calculatedNetwork && !this.currentlyDisplayingForwardPass)
			this.nodes.clear();
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
						let p = {
							x: outputNeuron * boxSize,
							y: layerY + boxSize,
							z: outN.output
						};
						if (
							!this.calculatedNetwork &&
							!this.currentlyDisplayingForwardPass
						)
							p.z = 0;
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
						var updateIndex =
							net.inputs[0].outputVector.length -
							net.layers[1][0].timeDelayed +
							outputNeuron +
							1;
						if (
							!this.calculatedNetwork &&
							!this.currentlyDisplayingForwardPass
						) {
							// console.log("Add node" + node.title);
							this.nodes.add(node);
							this.updates[0].nodes.push(node);
						} else {
							this.updates[updateIndex].nodes.push(node);
							// this.updates[
							// 	lastUpdate + outputNeuron
							// ].currentTime = outputNeuron;
							// this.updates[
							// 	lastUpdate + outputNeuron
							// ].layerNumber = outputLayer;
							// this.updates[
							// 	lastUpdate + outputNeuron
							// ].numberofneuron =
							// 	layer.length;
							// this.updates[
							// 	lastUpdate + outputNeuron
							// ].previousIndex =
							// 	lastUpdate - 1;
							var tmpMinX, tmpMinY;
							if (
								this.updates[updateIndex].layersUpdate[
									outputLayer
								] == undefined
							) {
								tmpMinX = 9999999999;
							} else {
								tmpMinX = this.updates[updateIndex]
									.layersUpdate[outputLayer].maxX;
							}
							if (this.minYofLayer[outputLayer] == undefined)
								tmpMinY = 99999999;
							else tmpMinY = this.minYofLayer[outputLayer];
							if (node.y < tmpMinY)
								this.minYofLayer[outputLayer] = node.y;
							this.updates[updateIndex].layersUpdate[
								outputLayer
							] = {
								boxSize: boxSize,
								maxX: node.x < tmpMinX ? node.x : tmpMinX,
								numberofNeuron: layer.length,
								timeDelayed: outN.timeDelayed
							};
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
							let p = {
								x: timeStep * boxSize,
								y: layerY + outputNeuron * boxSize,
								z: output
							};
							if (
								!this.calculatedNetwork &&
								!this.currentlyDisplayingForwardPass &&
								!(outN instanceof Net.InputNeuron)
							)
								p.z = 0;
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
							if (
								!this.calculatedNetwork &&
								!this.currentlyDisplayingForwardPass
							) {
								// console.log("Add node" + node.title);
								this.nodes.add(node);
								this.updates[0].nodes.push(node);
							} else {
								var tmpMinY;
								if (outN instanceof Net.InputNeuron) {
									this.updates[0].nodes.push(node);
								} else {
									this.updates[
										lastUpdate + timeStep
									].nodes.push(node);
									// this.updates[
									// 	lastUpdate + timeStep
									// ].currentTime = timeStep;
									// this.updates[
									// 	lastUpdate + timeStep
									// ].layerNumber = outputLayer;
									// this.updates[
									// 	lastUpdate + timeStep
									// ].timeDelayed =
									// 	outN.timeDelayed;
									// this.updates[
									// 	lastUpdate + timeStep
									// ].numberofneuron =
									// 	layer.length;
									// this.updates[
									// 	lastUpdate + timeStep
									// ].previousIndex =
									// 	lastUpdate - 1;

									var tmpMinX;
									if (
										this.updates[lastUpdate + timeStep]
											.layersUpdate[outputLayer] ==
										undefined
									) {
										tmpMinX = 9999999999;
									} else {
										tmpMinX = this.updates[
											lastUpdate + timeStep
										].layersUpdate[outputLayer].maxX;
									}

									this.updates[
										lastUpdate + timeStep
									].layersUpdate[outputLayer] = {
										boxSize: boxSize,
										maxX:
											node.x < tmpMinX ? node.x : tmpMinX,
										numberofNeuron: layer.length,
										timeDelayed: outN.timeDelayed
									};
								}
								if (this.minYofLayer[outputLayer] == undefined)
									tmpMinY = 99999999;
								else tmpMinY = this.minYofLayer[outputLayer];
								if (node.y < tmpMinY)
									this.minYofLayer[outputLayer] = node.y;
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
				}
				if (layer[0] instanceof Net.OutputNeuron)
					lastUpdate += layer.length;
				else {
					if (layer[0] instanceof Net.TimeDelayedNeuron)
						lastUpdate += layer[0].timeDelayed + 1; //layer[0].outputVector.length;
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
		// if (!this.calculatedNetwork) this.updates[0].nodes.push(this.nodes);
		this.updates[0].numberofneuron = this.net.layers[0].length;
		this.nodes.flush();
		// this.graph.stabilize();
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
		for (let neuron = 0; neuron < this.net.outputs.length; neuron++) {
			this.updates.push({ nodes: [], layersUpdate: {} });
		}
		for (
			let outputTime = 0;
			outputTime <=
			this.net.inputs[0].outputVector.length -
				this.net.inputs[0].outputs[0].out.timeDelayed;
			outputTime++
		) {
			this.updates.push({ nodes: [], layersUpdate: {} });
		}
		console.log(this.updates);
		// for (let i = 1; i < this.net.layers.length; i++) {
		// 	const layer = this.net.layers[i];
		// 	if (layer[0] instanceof Net.OutputNeuron) {
		// 		for (let neuron = 0; neuron < layer.length; neuron++)
		// 			this.updates.push({ nodes: [] });
		// 	} else {
		// 		for (
		// 			let time = 0;
		// 			time < layer[0].outputVector.length;
		// 			time++
		// 		) {
		// 			this.updates.push({ nodes: [] });
		// 			// let update={};
		// 			// for (let neuronNumer=0;neuronNumer<layer.length;neuronNumer++)
		// 			// {
		// 			// 	let neuron=layer[neuronNumer];

		// 			// }
		// 		}
		// 	}
		// }
	}

	onNetworkLoaded(net: Net.NeuralNet) {
		if (!net.isTDNN) return;
		console.log("On network loaded TDNN");
		this.actions = ["TDNN Graph"];
		// if (!this.alreadySetNet)
		// {
		// 	this.alreadySetNet=true;
		// }
		this.net = net;
		this.parseData(net);
		//this.calculatedNetwork = true;
		this.next = 0;
		this.applyUpdate(this.updates[this.next++]);
		// let abc=[[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]];
		// this.net.initTDNN(abc);
		// this.graph.setData(this.parseData(net));
	}
	onFrame() {
		console.log("On frame TDNN");
		// this.alreadySetNet = false;
		if (!this.net.isTDNN) return;
		// console.log("OnFrame step 2");
		this.calculatedNetwork = false;
		this.graph.off("afterDrawing");
		// this.parseData(this.net);
		this.graph.redraw();
	}
}
