class NeuronGui {
	layerDiv: JQuery = $("#hiddenLayersModify > div").clone();

	removeLayer() {
		$("#hiddenLayersModify > div").eq(0).remove();
	}
	addLayer() {
		$("#hiddenLayersModify > div").eq(0).before(this.layerDiv.clone());
	}
	setActivation(layer: int, activ: string) {

	}
	constructor(public sim: Simulation) {
		$("#hiddenLayersModify").on("click", "button", e => {
			const inc = e.target.textContent == '+';
			const layer = $(e.target.parentNode).index();
			const newval = sim.config.hiddenLayers[layer].neuronCount + (inc ? 1 : -1);
			if (newval < 1) return;
			sim.config.hiddenLayers[layer].neuronCount = newval;
			$("#hiddenLayersModify .neuronCount").eq(layer).text(newval);
			sim.setIsCustom();
			sim.initializeNet();
		});
		$("#inputLayerModify,#outputLayerModify").on("click", "button", e => {
			const isInput = $(e.target).closest("#inputLayerModify").length > 0;
			const name = isInput ? "input" : "output";
			const targetLayer = isInput ? sim.config.inputLayer : sim.config.outputLayer;
			const inc = e.target.textContent == '+';
			const newval = targetLayer.neuronCount + (inc ? 1 : -1);
			if (newval < 1 || newval > 10) return;
			targetLayer.neuronCount = newval;
			$(`#${name}LayerModify .neuronCount`).text(newval);
			sim.config.data = [];
			sim.setIsCustom(true);
			sim.initializeNet();
		});
		$("#layerCountModifier").on("click", "button", e => {
			const inc = e.target.textContent == '+';
			if (!inc) {
				if (sim.config.hiddenLayers.length == 0) return;
				sim.config.hiddenLayers.shift();
				this.removeLayer();
			} else {
				sim.config.hiddenLayers.unshift({ activation: 'sigmoid', neuronCount: 2 });
				this.addLayer();
			}
			$("#layerCount").text(sim.config.hiddenLayers.length + 2);
			sim.setIsCustom();
			sim.initializeNet();
		});
		$("#outputLayerModify").on("change", "select", e=> {
			sim.config.outputLayer.activation = (<HTMLSelectElement>e.target).value;
			sim.setIsCustom();
			sim.initializeNet();
		});
		$("#hiddenLayersModify").on("change", "select", e=> {
			const layer = $(e.target.parentNode).index();
			sim.config.hiddenLayers[layer].activation = (<HTMLSelectElement>e.target).value;
			sim.setIsCustom();
			sim.initializeNet();
		});
	}
	regenerate() {
		const targetCount = this.sim.config.hiddenLayers.length;
		while ($("#hiddenLayersModify > div").length > targetCount)
			this.removeLayer();
		while ($("#hiddenLayersModify > div").length < targetCount)
			this.addLayer();
		this.sim.config.hiddenLayers.forEach((c: LayerConfig, i: int) => {
			$("#hiddenLayersModify .neuronCount").eq(i).text(c.neuronCount);
			$("#hiddenLayersModify > div").eq(i).children("select.activation").val(c.activation);
		});
		$("#outputLayerModify").children("select.activation").val(this.sim.config.outputLayer.activation);
		$("#inputLayerModify .neuronCount").text(this.sim.config.inputLayer.neuronCount);
		$("#outputLayerModify .neuronCount").text(this.sim.config.outputLayer.neuronCount);
		$("#layerCount").text(this.sim.config.hiddenLayers.length + 2);
	}
}