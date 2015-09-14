declare let vis:any; // vis.js library
class NetworkGraph implements Visualization {
	actions = ["Network Graph"];
	graph:any; //vis.Network
	nodes:any; // vis.DataSet
	edges:any;
	net:Net.NeuralNet;
	container = $("<div>");
	showbias: boolean;
	constructor(public sim: Simulation) {
		this.instantiateGraph();
	}
	instantiateGraph() {
		this.nodes = new vis.DataSet([], {queue:true});
		this.edges = new vis.DataSet([], {queue:true});
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
		if(this.graph) this.graph.destroy();
		this.graph = new vis.Network(this.container[0], graphData, options);
	}
	onNetworkLoaded(net:Net.NeuralNet) {
		if(this.net
			&& this.net.layers.length == net.layers.length 
			&& this.net.layers.every((layer,index) => layer.length == net.layers[index].length)
			&& this.showbias === this.sim.config.bias) {
			// same net layout, only update
			this.net = net;
			this.onFrame(0);
			return;
		}
		this.showbias = this.sim.config.bias;
		this.nodes.clear();
		this.edges.clear();
		this.net = net;
		for (let lid = 0; lid < net.layers.length; lid++) {
			let layer = net.layers[lid];
			for (let nid = 0; nid < layer.length; nid++) {
				let neuron = layer[nid];
				let type = 'Hidden Neuron '+(nid+1);
				let color = '#000';
				if (neuron instanceof Net.InputNeuron) {
					type = 'Input: '+neuron.name;
					if(neuron.constant) {
						color = NetworkVisualization.colors.autoencoder.bias;
					}
					else color = NetworkVisualization.colors.autoencoder.input;
				} if (neuron instanceof Net.OutputNeuron) {
					type = 'Output: '+neuron.name;
					color = NetworkVisualization.colors.autoencoder.output;
				}
				this.nodes.add({
					id: neuron.id,
					label: `${type}`,
					level: lid,
					color: color
				});
			}
		}
		for (let conn of net.connections) {
			this.edges.add({
				id: conn.inp.id * net.connections.length + conn.out.id,
				from: conn.inp.id,
				to: conn.out.id,
				arrows:'to',
				label: conn.weight.toFixed(2),
			})
		}
		this.nodes.flush();
		this.edges.flush();
		this.graph.stabilize();
		this.graph.fit();
	}
	onFrame(framenum:int) {
		if(this.net.connections.length > 20 && framenum % 15 !== 0) {
			// skip some frames because slow
			return;
		}
		for (let conn of this.net.connections) {
			this.edges.update({
				id: conn.inp.id * this.net.connections.length + conn.out.id,
				label: conn.weight.toFixed(2),
				width: Math.min(6, Math.abs(conn.weight*2)),
				color: conn.weight > 0 ? 'blue':'red'
			})
		}
		this.edges.flush();
	}
	onView() {
		this.graph.stabilize();
	}
	onHide() {
		
	}
}