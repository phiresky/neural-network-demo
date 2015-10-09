declare const vis:any; // vis.js library
interface NetGraphUpdate {
	nodes: any[];
	edges: any[];
	highlightNodes?: number[];
	highlightEdges?: number[];
}
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
		const graphData = {
			nodes: this.nodes,
			edges: this.edges };
		const options = {
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
	private edgeId(conn:Net.NeuronConnection) {
		return conn.inp.id * this.net.connections.length + conn.out.id
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
			const layer = net.layers[lid];
			for (let nid = 0; nid < layer.length; nid++) {
				const neuron = layer[nid];
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
		for (const conn of net.connections) {
			this.edges.add({
				id: this.edgeId(conn),
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
	forwardPass(data:TrainingData): NetGraphUpdate[] {
		this.net.setInputsAndCalculate(data.input);
		const updates:NetGraphUpdate[] = [{nodes:[], edges:[]}];
		// reset all names
		
		for(const layer of this.net.layers) for(const neuron of layer) {
			updates[0].nodes.push({
				id:neuron.id,
				label: `0`
			});
		}
		for(let i = 0; i < data.input.length; i++) {
			updates[0].nodes.push({
				id:this.net.inputs[i].id,
				label: `${this.net.inputs[i].name} = ${data.input[i]}`
			});
		}
		const allEdgesInvisible = () => this.net.connections.map(conn => ({
			id:this.edgeId(conn),
			color:"rgba(255,255,255,0)",
			label:undefined
		}));
		updates[0].edges = allEdgesInvisible();
		for(const layer of this.net.layers.slice(1)) {
			for(const neuron of layer) {
				if(neuron instanceof Net.InputNeuron) continue; // bias neuron
				updates.push({
					highlightNodes:[neuron.id],
					nodes: [],
					edges:allEdgesInvisible().concat(neuron.inputs.map(i => ({
						id:this.edgeId(i),
						color:"black",
						label: ""
					})))
				});
				let neuronVal = 0;
				for(const input of neuron.inputs) {
					const add = input.inp.output * input.weight;
					neuronVal += add;
					const update:NetGraphUpdate = {
						nodes:[{id:neuron.id, label: `∑ = ${neuronVal.toFixed(2)}`}],
						edges:[{id:this.edgeId(input), label:`+ ${input.inp.output.toFixed(2)} · (${input.weight.toFixed(2)})`}],
						highlightNodes:[],
						highlightEdges:[this.edgeId(input)]
					}
					updates.push(update);
				}
				updates.push({
					nodes:[{id:neuron.id, label: `σ(${neuronVal.toFixed(2)}) = ${neuron.output.toFixed(2)}`}],
					edges:allEdgesInvisible()
				})
			}
		}
		return updates;
	}
	applyUpdate(update: NetGraphUpdate) {
		this.edges.update(update.edges);
		this.nodes.update(update.nodes);
		this.nodes.flush();
		this.edges.flush();
		if(update.highlightNodes) this.graph.selectNodes(update.highlightNodes, false);
		if(update.highlightEdges) this.graph.selectEdges(update.highlightEdges);
	}
	onFrame(framenum:int) {
		if(this.net.connections.length > 20 && framenum % 15 !== 0) {
			// skip some frames because slow
			return;
		}
		for (const conn of this.net.connections) {
			this.edges.update({
				id: this.edgeId(conn),
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