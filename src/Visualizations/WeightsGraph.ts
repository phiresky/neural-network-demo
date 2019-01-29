import { Visualization } from "./Visualization";
import Simulation from "../Simulation";
import * as $ from "jquery";
import { int, double } from "../main";
import Net from "../Net";
import * as vis from "vis";
interface Point3d {
	x: double;
	y: double;
	z: double;
	style?: double;
}
/**
 * Visualization that displays all the weights in the network as a black-white gradient
 *
 * hidden when in perceptron mode because then it's pretty boring
 */
export default class WeightsGraph implements Visualization {
	actions = ["Weights"];
	container = document.createElement("div");
	offsetBetweenLayers = 2;
	graph: any; //vis.Graph3d;
	xyToConnection: { [xcommay: string]: [Net.NeuronConnection, int] } = {};
	constructor(public sim: Simulation) {
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
			zStep: 0.1,
			showGrid: true,
			axisColor: "red",
			xBarWidth: 0.9,
			yBarWidth: 0.9,
			xCenter: "50%",
			legendLabel: "Weight",
			//zMin: 0,
			//zMax: 5,
			tooltip: (point: Point3d) => {
				const [conn, outputLayer] = this.xyToConnection[
					point.x + "," + point.y
				];
				const inLayer = outputLayer - 1;
				let inStr: string, outStr: string;
				const inN = conn.inp,
					outN = conn.out;
				if (inN instanceof Net.InputNeuron) inStr = inN.name;
				else inStr = `Hidden(${inLayer + 1},${inN.layerIndex + 1})`;
				if (outN instanceof Net.OutputNeuron) outStr = outN.name;
				else
					outStr = `Hidden(${outputLayer + 1},${outN.layerIndex +
						1})`;
				return inStr + " to " + outStr + ": " + point.z.toFixed(2); //conn.weight.toFixed(2);
			},
			//xValueLabel: (x: int) => this.xToLayer[x] || "",
			yValueLabel: (y: int) => ((y | 0) == y ? y + 1 : ""),
			zValueLabel: (z: int) => ""
		});
	}
	onView(previouslyHidden: boolean, action: int) {
		this.graph.redraw();
	}
	onHide() {}
	parseDataTDNN(net: Net.NeuralNet) {
		this.xyToConnection = {};
		const data: Point3d[] = [];
		let maxx = 0;
		let maxy = 0;
		let lastUpdate = 1;
		const maxHeight = Math.max.apply(
			null,
			net.layers.map(layer => layer.length)
		);
		for (
			let outputLayer = 1;
			outputLayer < net.layers.length;
			outputLayer++
		) {
			const layer = net.layers[outputLayer];
			const layerY = maxy + this.offsetBetweenLayers;

			for (
				let outputNeuron = 0;
				outputNeuron < layer.length;
				outputNeuron++
			) {
				let layerX = 0;
				const outN = layer[outputNeuron];
				maxy = Math.max(maxy, layerY + layer.length);
				for (
					let inputNeuron = 0;
					inputNeuron < outN.inputs.length;
					inputNeuron++
				) {
					const conn = outN.inputs[inputNeuron];
					const inN = conn.inp;
					layerX = (conn.weightVector!.length + 1) * inputNeuron;
					maxx = Math.max(maxx, layerX + conn.weightVector!.length);
					if (
						!this.sim.state.bias &&
						inN instanceof Net.InputNeuron &&
						inN.constant
					) {
						continue;
					}
					for (
						let timeDelayWeight = 0;
						timeDelayWeight < conn.weightVector!.length;
						timeDelayWeight++
					) {
						const p = {
							x: layerX + timeDelayWeight,
							y: layerY + outputNeuron,
							z: conn.weightVector![timeDelayWeight]
						};
						if (maxHeight != layer.length)
							p.y += (maxHeight - layer.length) / 2;
						data.push(p);
						this.xyToConnection[p.x + "," + p.y] = [
							conn,
							outputLayer
						];
					}
				}
			}
		}
		return data;
	}
	/** parse network layout into weights graph ordering */
	parseData(net: Net.NeuralNet) {
		console.log(this.sim.tdnngraph.alreadySetNet);
		if (net.isTDNN) {
			let data1: Point3d[] = [{ x: 1, y: 1, z: 0 }];
			try {
				data1 = this.parseDataTDNN(net);
				console.log(data1);
				return data1;
			} catch {
				console.log(data1);
				return data1;
			}
		}

		this.xyToConnection = {};
		const data: Point3d[] = [];
		let maxx = 0;
		const maxHeight = Math.max.apply(
			null,
			net.layers.map(layer => layer.length)
		);
		for (
			let outputLayer = 1;
			outputLayer < net.layers.length;
			outputLayer++
		) {
			const layer = net.layers[outputLayer];
			const layerX = maxx + this.offsetBetweenLayers;
			for (
				let outputNeuron = 0;
				outputNeuron < layer.length;
				outputNeuron++
			) {
				const outN = layer[outputNeuron];
				maxx = Math.max(maxx, layerX + outN.inputs.length);
				for (
					let inputNeuron = 0;
					inputNeuron < outN.inputs.length;
					inputNeuron++
				) {
					const conn = outN.inputs[inputNeuron];
					const inN = conn.inp;
					if (
						!this.sim.state.bias &&
						inN instanceof Net.InputNeuron &&
						inN.constant
					) {
						continue;
					}
					const p = {
						x: layerX + inputNeuron,
						y: outputNeuron,
						z: conn.weight
					};
					if (maxHeight != layer.length)
						p.y += (maxHeight - layer.length) / 2;
					data.push(p);
					this.xyToConnection[p.x + "," + p.y] = [conn, outputLayer];
				}
			}
		}
		return data;
	}
	onNetworkLoaded(net: Net.NeuralNet) {
		if (this.sim.state.type == "perceptron") this.actions = [];
		else this.actions = ["Weights"];
		this.xyToConnection = {};
		this.graph.setData(this.parseData(net));
	}
	onFrame() {
		this.xyToConnection = {};
		this.graph.redraw();
		this.graph.setData(this.parseData(this.sim.net));
	}
}
