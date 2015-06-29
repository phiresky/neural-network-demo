///<reference path='../lib/typings/jquery/jquery.d.ts' />
///<reference path='../lib/typings/jquery-handsontable/jquery-handsontable.d.ts' />
///<reference path='Net.ts' />
///<reference path='NetworkGraph.ts' />
///<reference path='NetworkVisualization.ts' />
///<reference path='Presets.ts' />
interface JQuery { slider: any };
declare var Handsontable: any, LZString: any;
class TableEditor {
	hot: any; // handsontable instance
	headerCount = 2;
	lastUpdate = 0;
	constructor(public container: JQuery, sim: Simulation) {
		let headerRenderer = function firstRowRenderer(instance: any, td: HTMLTableCellElement) {
			Handsontable.renderers.TextRenderer.apply(this, arguments);
			td.style.fontWeight = 'bold';
			td.style.background = '#CCC';
		}
		container.handsontable({
			minSpareRows: 1,
			cells: (row, col, prop) => {
				if (row >= this.headerCount) return { type: 'numeric', format: '0.[000]' };
				else return { renderer: headerRenderer };
			},
			//customBorders: true,
			allowInvalid: false,
			afterChange: this.afterChange.bind(this)
		});
		this.hot = container.handsontable('getInstance');
		$("<div>").addClass("btn btn-default")
			.css({ position: "absolute", right: "2em", bottom: "2em" })
			.text("Remove all")
			.click(e => { sim.config.data = []; this.loadData(sim) })
			.appendTo(container);
		this.loadData(sim);
	}
	afterChange(changes: [number, number, number, number][], reason: string) {
		if (reason === 'loadData') return;
		this.reparseData();
	}
	reparseData() {
		let data: number[][] = this.hot.getData();
		let headers = <string[]><any>data[1];
		let ic = sim.config.inputLayer.neuronCount, oc = sim.config.outputLayer.neuronCount
		sim.config.inputLayer.names = headers.slice(0, ic);
		sim.config.outputLayer.names = headers.slice(ic, ic + oc);
		sim.config.data = data.slice(2).map(row => row.slice(0, ic + oc)).filter(row => row.every(cell => typeof cell === 'number'))
			.map(row => <TrainingData>{ input: row.slice(0, ic), output: row.slice(ic) });
		sim.setIsCustom();
	}
	updateRealOutput() {
		if ((Date.now() - this.lastUpdate) < 500) return;
		this.lastUpdate = Date.now();
		let xOffset = sim.config.inputLayer.neuronCount + sim.config.outputLayer.neuronCount;
		let vals: [number, number, number][] = [];
		for (let y = 0; y < sim.config.data.length; y++) {
			let p = sim.config.data[y];
			let op = sim.net.getOutput(p.input);
			for (let x = 0; x < op.length; x++) {
				vals.push([y + this.headerCount, xOffset + x, op[x]]);
			}
		}
		this.hot.setDataAtCell(vals, "loadData");
	}
	loadData(sim: Simulation) { // needs sim as arg because called from constructor
		let data: (number|string)[][] = [[], sim.config.inputLayer.names.concat(sim.config.outputLayer.names).concat(sim.config.outputLayer.names)];
		let ic = sim.config.inputLayer.neuronCount, oc = sim.config.outputLayer.neuronCount;
		data[0][0] = 'Inputs';
		data[0][ic] = 'Expected Output';
		data[0][ic + oc + oc - 1] = ' ';
		data[0][ic + oc] = 'Actual Output';

		sim.config.data.forEach(t => data.push(t.input.concat(t.output)));
		this.hot.loadData(data);
		/*this.hot.updateSettings({customBorders: [
				{
					range: {
						from: { row: 0, col: ic },
						to: { row: 100, col: ic }
					},
					left: { width: 2, color: 'black' }
				}, {
					range: {
						from: { row: 0, col: ic+oc },
						to: { row: 100, col: ic+oc }
					},
					left: { width: 2, color: 'black' }
				}
			]});
		this.hot.runHooks('afterInit');*/
	}
}
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
interface InputLayerConfig {
	neuronCount: int;
	names: string[];
}
interface LayerConfig {
	neuronCount: int;
	activation: string;
}
interface OutputLayerConfig extends LayerConfig {
	names: string[];
}
class Simulation {
	netviz: NetworkVisualization;
	netgraph: NetworkGraph;
	backgroundResolution = 10;
	stepNum = 0;
	running = false; runningId = -1;
	restartTimeout = -1;
	isCustom = false;

	net: Net.NeuralNet;
	neuronGui: NeuronGui;
	table: TableEditor;
	config: Configuration;

	constructor() {
		let canvas = <HTMLCanvasElement>$("#neuralInputOutput canvas")[0];
		this.netviz = new NetworkVisualization(canvas,
			new CanvasMouseNavigation(canvas, () => this.netviz.inputMode == 3, () => this.draw()),
			this,
			this.backgroundResolution);
		this.netgraph = new NetworkGraph($("#neuralNetworkGraph")[0]);
		
		(<any>$("#learningRate")).slider({
			min: 0.01, max: 1, step: 0.005, scale: "logarithmic", value: 0.05
		}).on('change', (e: any) => $("#learningRateVal").text(e.value.newValue.toFixed(3)));
		this.neuronGui = new NeuronGui(this);
		for (let name of Presets.getNames())
			$("#presetLoader").append($("<li>").append($("<a>").text(name)));
		$("#presetLoader").on("click", "a", e => {
			let name = e.target.textContent;
			$("#presetName").text(`Preset: ${name}`);
			this.config = Presets.get(name);
			this.setConfig();
			this.isCustom = false;
			history.replaceState({}, "", "?" + $.param({ preset: name }));
			this.initializeNet();
		});

		$("#dataInputSwitch").on("click", "a", e => {
			$("#dataInputSwitch li.active").removeClass("active");
			let li = $(e.target).parent();
			li.addClass("active");
			let mode = li.index();
			let modeSwitched = ((this.netviz.inputMode == InputMode.Table) != (mode == InputMode.Table));
			this.netviz.inputMode = mode;
			if (!modeSwitched) return;
			if (mode == InputMode.Table) {
				$("#neuralInputOutput > *").detach(); // keep event handlers
				$("#neuralInputOutput").append(this.table.container);
				this.table.loadData(this);
			} else {
				this.table.reparseData();
				$("#neuralInputOutput > *").detach();
				$("#neuralInputOutput").append(this.netviz.canvas);
				this.draw();
			}
		});
		let doSerialize = () => {
			this.stop();
			console.log("ser");
			$("#urlExport").val(sim.serializeToUrl(+$("#exportWeights").val()));
		};
		$("#exportModal").on("shown.bs.modal", doSerialize);
		$("#exportModal select").on("change", doSerialize);
		this.deserializeFromUrl();
		this.table = new TableEditor($("<div class='fullsize'>"), this);
		this.run();
	}

	initializeNet(weights?: double[]) {
		if (this.net) this.stop();
		this.net = new Net.NeuralNet(this.config.inputLayer, this.config.hiddenLayers, this.config.outputLayer, this.config.learningRate, this.config.bias, undefined, weights);
		let isBinClass = this.config.outputLayer.neuronCount === 1;
		$("#dataInputSwitch > li").eq(1).toggle(isBinClass);
		let firstButton = $("#dataInputSwitch > li > a").eq(0);
		firstButton.text(isBinClass ? "Add Red" : "Add point")
		if (!isBinClass && this.netviz.inputMode == 1) firstButton.click();
		this.stepNum = 0;
		this.netgraph.loadNetwork(this.net);
		if(this.table) this.table.loadData(this);
		this.draw();
		this.updateStatusLine();
	}
	statusIterEle = document.getElementById('statusIteration');
	statusCorrectEle = document.getElementById('statusCorrect');
	step() {
		this.stepNum++;
		for (let val of this.config.data) {
			this.net.train(val.input, val.output);
		}
	}

	draw() {
		if (this.netviz.inputMode === InputMode.Table)
			this.table.updateRealOutput();
		else this.netviz.draw();
		this.netgraph.update();
	}

	run() {
		if (this.running) return;
		$("#runButton").text("Stop").addClass("btn-danger").removeClass("btn-primary");
		this.running = true;
		this.animationStep();
	}

	stop() {
		clearTimeout(this.restartTimeout);
		$("#runButton").text("Run").addClass("btn-primary").removeClass("btn-danger");
		this.restartTimeout = -1;
		this.running = false;
		cancelAnimationFrame(this.runningId);
	}

	reset() {
		this.stop();
		this.loadConfig(true);
		this.initializeNet();
	}

	updateStatusLine() {
		let correct = 0;
		if (this.config.outputLayer.neuronCount === 1) {
			for (var val of this.config.data) {
				let res = this.net.getOutput(val.input);
				if (+(res[0] > 0.5) == val.output[0]) correct++;
			}
			this.statusCorrectEle.innerHTML = `Correct: ${correct}/${this.config.data.length}`;
		} else {
			let sum = 0;
			for (let val of this.config.data) {
				let res = this.net.getOutput(val.input);
				let sum1 = 0;
				for (let i = 0; i < this.net.outputs.length; i++) {
					let dist = res[i] - val.output[i];
					sum1 += dist * dist;
				}
				sum += Math.sqrt(sum1);
			}
			this.statusCorrectEle.innerHTML = `Avg. distance: ${(sum / this.config.data.length).toFixed(2) }`;
		}

		this.statusIterEle.innerHTML = this.stepNum.toString();

		if (correct == this.config.data.length) {
			if (this.config.autoRestart && this.running && this.restartTimeout == -1) {
				this.restartTimeout = setTimeout(() => {
					this.stop();
					this.restartTimeout = -1;
					setTimeout(() => { this.reset(); this.run(); }, 1000);
				}, this.config.autoRestartTime - 1);
			}
		} else {
			if (this.restartTimeout != -1) {
				clearTimeout(this.restartTimeout);
				this.restartTimeout = -1;
			}
		}
	}

	aniFrameCallback = this.animationStep.bind(this);
	animationStep() {
		for (let i = 0; i < this.config.stepsPerFrame; i++) this.step();
		this.draw();
		this.updateStatusLine();
		if (this.running) this.runningId = requestAnimationFrame(this.aniFrameCallback);
	}

	iterations() {
		this.stop();
		for (var i = 0; i < this.config.iterationsPerClick; i++)
			this.step();
		this.draw();
		this.updateStatusLine();
	}

	setIsCustom() {
		if (this.isCustom) return;
		this.isCustom = true;
		$("#presetName").text("Custom Network");
		for (let name of ["input", "output"]) {
			let layer = this.config[`${name}Layer`];
			layer.names = [];
			for (let i = 0; i < layer.neuronCount; i++)
				layer.names.push(`${name} ${i + 1}`);
		}
		this.table.loadData(this);
	}

	loadConfig(nochange = false) { // from gui
		let config = <any>this.config;
		let oldConfig = $.extend({}, config);
		for (let conf in config) {
			let ele = <HTMLInputElement>document.getElementById(conf);
			if (!ele) continue;
			if (ele.type == 'checkbox') config[conf] = ele.checked;
			else if (typeof config[conf] === 'number')
				config[conf] = +ele.value;
			else config[conf] = ele.value;
		}
		if (oldConfig.simType != config.simType) config.data = [];
		if (this.net) this.net.learnRate = this.config.learningRate;
		if (!nochange) this.setIsCustom();
	}
	setConfig() { // in gui
		let config = <any>this.config;
		for (let conf in config) {
			let ele = <HTMLInputElement>document.getElementById(conf);
			if (!ele) continue;
			if (ele.type == 'checkbox') ele.checked = config[conf];
			else ele.value = config[conf];
		}
		$("#learningRate").slider('setValue', this.config.learningRate);
		$("#learningRateVal").text(this.config.learningRate.toFixed(3));
		this.neuronGui.regenerate();
	}

	runtoggle() {
		if (this.running) this.stop();
		else this.run();
	}

	// 0 = no weights, 1 = current weights, 2 = start weights
	serializeToUrl(exportWeights = 0) {
		let url = location.protocol + '//' + location.host + location.pathname + "?";
		let params:any = {};
		if(exportWeights === 1) params.weights = LZString.compressToBase64(JSON.stringify(this.net.connections.map(c => c.weight)));
		if(exportWeights === 2) params.weights = LZString.compressToBase64(JSON.stringify(this.net.startWeights));
		if (this.isCustom) {
			params.config = LZString.compressToBase64(JSON.stringify(this.config));
		} else {
			params.preset = this.config.name;
		}
		
		return url + $.param(params);
	}
	deserializeFromUrl() {
		function getUrlParameter(name: string) {
			let match = RegExp('[?&]' + name + '=([^&]*)').exec(window.location.search);
			return match && decodeURIComponent(match[1].replace(/\+/g, ' '));
		}
		let preset = getUrlParameter("preset"), config = getUrlParameter("config");
		if(preset && Presets.exists(preset))
			this.config = Presets.get(preset);
		else if(config) this.config = JSON.parse(LZString.decompressFromBase64(config));
		else
			this.config = Presets.get("Binary Classifier for XOR");
		let weights = getUrlParameter("weights");
		if(weights) this.initializeNet(JSON.parse(LZString.decompressFromBase64(weights)));
		else this.initializeNet();
	}
}