///<reference path='../lib/typings/jquery/jquery.d.ts' />
///<reference path='../lib/typings/jquery-handsontable/jquery-handsontable.d.ts' />
///<reference path='Net.ts' />
///<reference path='NetworkGraph.ts' />
///<reference path='NetworkVisualization.ts' />
///<reference path='Presets.ts' />
interface JQuery { slider: any };

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
class LearnRateGraph {
	chart: HighstockChartObject;
	constructor(container: JQuery, data: [number, number][]) {
		container.highcharts({
			title: { text: 'Average error' },
			chart: { type: 'line', animation: false },
			plotOptions: { line: { marker: { enabled: false } } },
			legend: { enabled: false },
			yAxis: { min: 0, title: { text: '' }, labels: { format: "{value:%.2f}" } },
			series: [{ name: 'Error', data: data }],
		});
		this.chart = container.highcharts();
	}
	clear() {
		this.chart.series[0].setData([]);
	}
	addPoint(step: int, error: double) {
		this.chart.series[0].addPoint([step, error], true, false);
	}
}
class Simulation {
	netviz: NetworkVisualization;
	netgraph: NetworkGraph;
	backgroundResolution = 10;
	stepNum = 0;
	running = false; runningId = -1;
	restartTimeout = -1;
	isCustom = false;
	averageError = 1;
	errorHistory: [number, number][];

	net: Net.NeuralNet;
	neuronGui: NeuronGui;
	table: TableEditor;
	config: Configuration;
	learnrateGraph: LearnRateGraph;

	constructor(autoRun: boolean) {
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
			this.loadPreset(name);
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
			$("#urlExport").text(sim.serializeToUrl(+$("#exportWeights").val()));
		};
		$("#exportModal").on("shown.bs.modal", doSerialize);
		$("#exportModal select").on("change", doSerialize);
		this.deserializeFromUrl();
		this.table = new TableEditor(this);
		if (autoRun) this.run();
	}

	initializeNet(weights?: double[]) {
		if (this.net) this.stop();
		this.errorHistory = [];
		if (this.learnrateGraph) this.learnrateGraph.clear();
		this.net = new Net.NeuralNet(this.config.inputLayer, this.config.hiddenLayers, this.config.outputLayer, this.config.learningRate, true, undefined, weights);
		let isBinClass = this.config.outputLayer.neuronCount === 1;
		$("#dataInputSwitch > li").eq(1).toggle(isBinClass);
		let firstButton = $("#dataInputSwitch > li > a").eq(0);
		firstButton.text(isBinClass ? "Add Red" : "Add point")
		if (!isBinClass && this.netviz.inputMode == 1) firstButton.click();
		this.stepNum = 0;
		this.netgraph.loadNetwork(this.net, this.config.bias);
		if (this.table) this.table.loadData(this);
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
		this.averageError = 0;
		for (let val of this.config.data) {
			let res = this.net.getOutput(val.input);
			let sum1 = 0;
			for (let i = 0; i < this.net.outputs.length; i++) {
				let dist = res[i] - val.output[i];
				sum1 += dist * dist;
			}
			this.averageError += Math.sqrt(sum1);
		}
		this.averageError /= this.config.data.length;
		if (this.config.outputLayer.neuronCount === 1) {
			for (var val of this.config.data) {
				let res = this.net.getOutput(val.input);
				if (+(res[0] > 0.5) == val.output[0]) correct++;
			}
			this.statusCorrectEle.innerHTML = `Correct: ${correct}/${this.config.data.length}`;
		} else {
			this.statusCorrectEle.innerHTML = `Avg. error: ${(this.averageError).toFixed(2) }`;
		}
		this.errorHistory.push([this.stepNum, this.averageError]);
		if (this.learnrateGraph) this.learnrateGraph.addPoint(this.stepNum, this.averageError);
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

	setIsCustom(neuronCountsChanged: boolean, loadData: boolean = true) {
		if (this.isCustom && !neuronCountsChanged) return;
		this.isCustom = true;
		$("#presetName").text("Custom Network");
		let layer = this.config.inputLayer;
		layer.names = Net.Util.makeArray(layer.neuronCount, i => `in${i + 1}`);
		layer = this.config.outputLayer;
		layer.names = Net.Util.makeArray(layer.neuronCount, i => `out${i + 1}`);
		if (neuronCountsChanged) this.table.createNewTable(this);
		if (loadData) this.table.loadData(this);
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
		if (!nochange) this.setIsCustom(true);
	}

	loadPreset(name: string) {
		this.isCustom = false;
		$("#presetName").text(`Preset: ${name}`);
		this.config = Presets.get(name);
		if (this.table) this.table.createNewTable(this);
		this.setConfig();
		history.replaceState({}, "", "?" + $.param({ preset: name }));
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

	showLearnrateGraph() {
		let container = $("<div>");
		$(this.netgraph.networkGraphContainer).children("*").detach();
		$(this.netgraph.networkGraphContainer).append(container);
		this.learnrateGraph = new LearnRateGraph(container, this.errorHistory)
	}

	// 0 = no weights, 1 = current weights, 2 = start weights
	serializeToUrl(exportWeights = 0) {
		let url = location.protocol + '//' + location.host + location.pathname + "?";
		let params: any = {};
		if (exportWeights === 1) params.weights = LZString.compressToBase64(JSON.stringify(this.net.connections.map(c => c.weight)));
		if (exportWeights === 2) params.weights = LZString.compressToBase64(JSON.stringify(this.net.startWeights));
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
		if (preset && Presets.exists(preset))
			this.loadPreset(preset);
		else if (config) {
			this.config = JSON.parse(LZString.decompressFromBase64(config));
			this.setIsCustom(true);
		} else
			this.loadPreset("Binary Classifier for XOR");
		let weights = getUrlParameter("weights");
		if (weights) this.initializeNet(JSON.parse(LZString.decompressFromBase64(weights)));
		else this.initializeNet();
	}
}