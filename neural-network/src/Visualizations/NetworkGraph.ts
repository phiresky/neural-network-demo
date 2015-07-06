declare let vis:any; // vis.js library
class NetworkGraph implements Visualization {
	actions = ["Network Graph"];
	graph:any; //vis.Network
	nodes = new vis.DataSet();
	edges = new vis.DataSet();
	net:Net.NeuralNet;
	container = $("<div>");
	constructor(public sim: Simulation) {
		this.instantiateGraph();
	}
	instantiateGraph() {
		// need only be run once, but removes bounciness if run every time
		let graphData = {
			nodes: this.nodes,
			edges: this.edges };
		let options = {
			nodes: { shape: 'dot' },
			edges: { 
				smooth: {type: 'curvedCW',roundness:0},
				font:{align:'top', background:'white'},
				/*scaling: {
					label: {min:1,max:2}
				}*/
			},
			layout: { hierarchical: { direction: "LR" } },
			interaction: { dragNodes: false }
		}
		this.graph = new vis.Network(this.container[0], graphData, options);
	}
	onNetworkLoaded(net:Net.NeuralNet) {
		let showbias = this.sim.config.bias;
		if(this.net
			&& this.net.layers.length == net.layers.length 
			&& this.net.layers.every((layer,index) => layer.length == net.layers[index].length)) {
			// same net layout, only update
			this.net = net;
			this.onFrame();
			return;
		}
		this.instantiateGraph();
		this.net = net;
		this.nodes.clear();
		this.edges.clear();
		let nodes: any[] = [], edges: any[] = [];
		for (let lid = 0; lid < net.layers.length; lid++) {
			let layer = net.layers[lid];
			for (let nid = 0; nid < layer.length; nid++) {
				let neuron = layer[nid];
				let type = 'Hidden Neuron '+(nid+1);
				let color = '#000';
				if (neuron instanceof Net.InputNeuron) {
					type = 'Input: '+neuron.name;
					if(neuron.constant) {
						if(!showbias) continue;
						color = NetworkVisualization.colors.autoencoder.bias;
					}
					else color = NetworkVisualization.colors.autoencoder.input;
				} if (neuron instanceof Net.OutputNeuron) {
					type = 'Output: '+neuron.name;
					color = NetworkVisualization.colors.autoencoder.output;
				}
				nodes.push({
					id: neuron.id,
					label: `${type}`,
					level: lid,
					color: color
				});
			}
		}
		for (let conn of net.connections) {
			edges.push({
				id: conn.inp.id * net.connections.length + conn.out.id,
				from: conn.inp.id,
				to: conn.out.id,
				arrows:'to',
				label: conn.weight.toFixed(2),
			})
		}
		this.nodes.add(nodes);
		this.edges.add(edges);
	}
	onFrame() {
		for (let conn of this.net.connections) {
			this.edges.update({
				id: conn.inp.id * this.net.connections.length + conn.out.id,
				label: conn.weight.toFixed(2),
				width: Math.min(6, Math.abs(conn.weight*2)),
				color: conn.weight > 0 ? 'blue':'red'
			})
		}
	}
	onView() {
		this.graph.stabilize();
	}
	onHide() {
		
	}
}