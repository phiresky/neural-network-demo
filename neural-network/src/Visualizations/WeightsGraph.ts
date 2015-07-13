interface Point3d {
	x: double; y: double; z: double;
	style?: double;
}
class WeightsGraph implements Visualization {
	actions = ["Weights"];
	container = $("<div>");
	offsetBetweenLayers = 2;
	graph: any;//vis.Graph3d;
	xyToConnection: {[xcommay:string]:[Net.NeuronConnection, int]} = {};
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
			cameraPosition: { horizontal: -0.001, vertical: Math.PI/2 },
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
				let [conn, outputLayer] = this.xyToConnection[point.x+","+point.y];
				let inLayer = outputLayer -1;
				let inStr: string, outStr: string;
				let inN = conn.inp, outN = conn.out;
				if (inN instanceof Net.InputNeuron) inStr = inN.name;
				else inStr = `Hidden(${inLayer + 1},${inN.layerIndex + 1})`;
				if (outN instanceof Net.OutputNeuron) outStr = outN.name;
				else outStr = `Hidden(${outputLayer + 1},${outN.layerIndex + 1})`;
				return inStr + " to " + outStr;
			},
			//xValueLabel: (x: int) => this.xToLayer[x] || "",
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
		this.xyToConnection = {};
		let data: Point3d[] = [];
		let maxx = 0;
		let maxHeight = Math.max.apply(null, net.layers.map(layer => layer.length));
		for (let outputLayer = 1; outputLayer < net.layers.length; outputLayer++) {
			let layer = net.layers[outputLayer];
			let layerX = maxx + this.offsetBetweenLayers;
			for (let outputNeuron = 0; outputNeuron < layer.length; outputNeuron++) {
				let outN = layer[outputNeuron];
				maxx = Math.max(maxx, layerX + outN.inputs.length);
				for (let inputNeuron = 0; inputNeuron < outN.inputs.length; inputNeuron++) {
					let conn = outN.inputs[inputNeuron];
					let inN = conn.inp;
					if(!this.sim.config.bias && inN instanceof Net.InputNeuron && inN.constant) {
						continue;
					}
					let p = { x: layerX + inputNeuron, y: outputNeuron, z: conn.weight };
					if(maxHeight != layer.length) p.y += (maxHeight - layer.length) / 2;
					data.push(p);
					this.xyToConnection[p.x+","+p.y] = [conn, outputLayer];
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