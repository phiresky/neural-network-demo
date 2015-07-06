interface Point3d {
	x: double; y: double; z: double;
	style?: double;
}
class WeightsGraph implements Visualization {
	actions = ["Weights"];
	container = $("<div>");
	graph: any;//vis.Graph3d;
	xToLayer: number[] = [];
	xToNeuron: number[] = [];
	constructor(public sim: Simulation) {
		this.graph = new vis.Graph3d(this.container[0], undefined, {
			style: 'bar',
			showPerspective: false,
			cameraPosition: { horizontal: -0.001, vertical: 1.57, distance: 1.7 },
			width: "100%",
			height: "100%",
			xLabel: 'Layer',
			yLabel: 'Neuron',
			zLabel: '',
			xBarWidth: 0.9,
			yBarWidth: 0.9,
			legendLabel: "Weight",
			tooltip: (point: Point3d) => {
				let inLayer = this.xToLayer[point.x];
				let outLayer = inLayer + 1;
				let inNeuron = point.y;
				let outNeuron = this.xToNeuron[point.x];
				let inN = this.sim.net.layers[inLayer][inNeuron];
				let outN = this.sim.net.layers[outLayer][outNeuron];
				let inStr: string, outStr: string;
				if (inN instanceof Net.InputNeuron)
					inStr = inN.name;
				else inStr = `Layer ${inLayer + 1} Neuron ${inNeuron + 1}`;
				if (outN instanceof Net.OutputNeuron)
					outStr = outN.name;
				else outStr = `Layer ${outLayer + 1} Neuron ${outNeuron + 1}`;
				return inStr + " to " + outStr;
			},
			xValueLabel: (x: int) => {let l = this.xToLayer[x] + 1; if(!l) return ""; return l+"-"+(l+1);},
			yValueLabel: (y: int) => (y | 0) == y ? y + 1 : "",
			zValueLabel: (z: int) => "",
		});
	}
	onView(previouslyHidden: boolean, action: int) {
		this.graph.redraw();
	}
	onHide() {

	}
	parseData(net: Net.NeuralNet) {
		let data: Point3d[] = [];
		let maxx = 0;
		for (let layerNum = 0; layerNum < net.layers.length; layerNum++) {
			let layer = net.layers[layerNum];
			let layerX = maxx + 1;
			for (let y = 0; y < layer.length; y++) {
				let neuron = layer[y];
				maxx = Math.max(maxx, layerX + neuron.outputs.length);
				for (let output = 0; output < neuron.outputs.length; output++) {
					let conn = neuron.outputs[output];
					data.push({ x: layerX + output, y: y, z: conn.weight });
					this.xToLayer[layerX + output] = layerNum;
					this.xToNeuron[layerX + output] = output;
				}
			}
		}
		return data;
	}
	onNetworkLoaded(net: Net.NeuralNet) {
		this.graph.setData(this.parseData(net));
	}
	onFrame() {
		this.graph.setData(this.parseData(this.sim.net));
	}
}