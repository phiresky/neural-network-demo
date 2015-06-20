///<reference path='Net.ts' />

declare let vis:any; // vis.js library
class NetworkGraph {
	graph:any;
	nodes = new vis.DataSet();
	edges = new vis.DataSet();
	net:Net.NeuralNet;
	constructor(networkGraphContainer:HTMLElement) {
		let graphData = {
			nodes: this.nodes,
			edges: this.edges };
		let options = {
			nodes: { shape: 'dot' },
			edges: { 
				smooth: {type: 'curvedCW',roundness:0.2},
				font:{align:'top', background:'white'},
				/*scaling: {
					label: {min:1,max:2}
				}*/
			},
			layout: { hierarchical: { direction: "LR" } },
			interaction: { dragNodes: false }
		}
		this.graph = new vis.Network(networkGraphContainer, graphData, options);
	}
	loadNetwork(net:Net.NeuralNet) {
		if(this.net
			&& this.net.layers.length == net.layers.length 
			&& this.net.layers.every((layer,index) => layer.length == net.layers[index].length)) {
			// same net layout, only update
			this.net = net;
			this.update();
			return;
		}
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
					color = '#008';
				} if (neuron instanceof Net.OutputNeuron) {
					type = 'Output Neuron ' + (nid+1);
					color = '#800';
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
	update() {
		for (let conn of this.net.connections) {
			this.edges.update({
				id: conn.inp.id * this.net.connections.length + conn.out.id,
				label: conn.weight.toFixed(2),
				width: Math.min(10, Math.abs(conn.weight*2)),
				color: conn.weight > 0 ? 'blue':'red'
			})
		}
	}
}