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
			let inc = e.target.textContent == '+';
			let layer = $(e.target.parentNode).index();
			let newval = sim.config.hiddenLayers[layer].neuronCount + (inc ? 1 : -1);
			if (newval < 1) return;
			sim.config.hiddenLayers[layer].neuronCount = newval;
			$("#hiddenLayersModify .neuronCount").eq(layer).text(newval);
			sim.setIsCustom();
			sim.initializeNet();
		});
		$("#inputLayerModify,#outputLayerModify").on("click", "button", e => {
			let isInput = $(e.target).closest("#inputLayerModify").length > 0;
			let name = isInput ? "input" : "output";
			let targetLayer = isInput ? sim.config.inputLayer : sim.config.outputLayer;
			let inc = e.target.textContent == '+';
			let newval = targetLayer.neuronCount + (inc ? 1 : -1);
			if (newval < 1) return;
			targetLayer.neuronCount = newval;
			$(`#${name}LayerModify .neuronCount`).text(newval);
			sim.config.data = [];
			sim.setIsCustom()
			sim.initializeNet();
		});
		$("#layerCountModifier").on("click", "button", e => {
			let inc = e.target.textContent == '+';
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
			sim.config.outputLayer.activation = (<any>e.target).value;
			sim.setIsCustom()
			sim.initializeNet();
		});
		$("#hiddenLayersModify").on("change", "select", e=> {
			let layer = $(e.target.parentNode).index();
			sim.config.hiddenLayers[layer].activation = (<HTMLSelectElement>e.target).value;
			sim.setIsCustom()
			sim.initializeNet();
		});
	}
	regenerate() {
		let targetCount = this.sim.config.hiddenLayers.length;
		while ($("#hiddenLayersModify > div").length > targetCount)
			this.removeLayer();
		while ($("#hiddenLayersModify > div").length < targetCount)
			this.addLayer();
		this.sim.config.hiddenLayers.forEach(
			(c: LayerConfig, i: int) => {
				$("#hiddenLayersModify .neuronCount").eq(i).text(c.neuronCount);
				$("#hiddenLayersModify > div").eq(i).children("select.activation").val(c.activation);
			});
	}
}