interface Point3d {
	x: double; y: double; z: double;
	style?: double;
}
class WeightsGraph implements Visualization {
	actions = ["Weights"];
	container = $("<div>");
	offsetBetweenLayers = 2;
	graph: any;//vis.Graph3d;
	xToLayer: number[] = [];
	xToNeuron: number[] = [];
	constructor(public sim: Simulation) {
		// hack to get grayscale colors
		vis.Graph3d.prototype._hsv2rgb = (h:double,s:double,v:double) => {
			h = Math.min(h,250)|0
			return 'rgb('+[h,h,h]+')'
		};
		// hack to disable axis drawing
		vis.Graph3d.prototype._redrawAxis = function(){};
		this.graph = new vis.Graph3d(this.container[0], undefined, {
			style: 'bar',
			showPerspective: false,
			cameraPosition: { horizontal: -0.001, vertical: Math.PI/2, distance: 1.2 },
			width: "100%",
			height: "100%",
			xLabel: 'Layer',
			yLabel: 'Neuron',
			zLabel: '',
			showGrid: true,
			axisColor: 'red',
			xBarWidth: 0.9,
			yBarWidth: 0.9,
			xCenter: "50%",
			legendLabel: "Weight",
			//zMin: 0,
			//zMax: 5,
			tooltip: (point: Point3d) => {
				let outLayer = this.xToLayer[point.x];
				let inLayer = outLayer - 1;
				let inNeuron = this.xToNeuron[point.x];
				let outNeuron = point.y;
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
			xValueLabel: (x: int) => this.xToLayer[x] || "",
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
		for (let layerNum = 1; layerNum < net.layers.length; layerNum++) {
			let layer = net.layers[layerNum];
			let layerX = maxx + this.offsetBetweenLayers;
			for (let y = 0; y < layer.length; y++) {
				let neuron = layer[y];
				maxx = Math.max(maxx, layerX + neuron.inputs.length);
				for (let input = 0; input < neuron.inputs.length; input++) {
					let conn = neuron.inputs[input];
					data.push({ x: layerX + input, y: y, z: conn.weight });
					this.xToLayer[layerX + input] = layerNum;
					this.xToNeuron[layerX + input] = input;
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