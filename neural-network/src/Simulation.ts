interface JQuery { slider: any };
class Simulation {
	netviz: NetworkVisualization;
	netgraph: NetworkGraph;
	table: TableEditor;
	errorGraph: ErrorGraph;
	weightsGraph: WeightsGraph;
	
	stepNum = 0;
	running = false; runningId = -1;
	restartTimeout = -1;
	isCustom = false;
	averageError = 1;

	net: Net.NeuralNet;
	neuronGui: NeuronGui;
	config: Configuration;
	leftVis:TabSwitchVisualizationContainer;
	rightVis:TabSwitchVisualizationContainer;
	
	constructed = false;
	errorHistory:[number,number][];

	constructor(autoRun: boolean) {
		(<any>$("#learningRate")).slider({
			min: 0.01, max: 1, step: 0.005, scale: "logarithmic", value: 0.05
		}).on('change', (e: any) => $("#learningRateVal").text(e.value.newValue.toFixed(3)));
		for (let name of Presets.getNames())
			$("#presetLoader").append($("<li>").append($("<a>").text(name)));
		$("#presetLoader").on("click", "a", e => {
			let name = e.target.textContent;
			this.loadPreset(name);
		});
		let doSerialize = () => {
			this.stop();
			$("#urlExport").text(this.serializeToUrl(+$("#exportWeights").val()));
		};
		$("#exportModal").on("shown.bs.modal", doSerialize);
		$("#exportModal select").on("change", doSerialize);
		this.neuronGui = new NeuronGui(this);
		
		this.netviz = new NetworkVisualization(this);
		this.netgraph = new NetworkGraph(this);
		this.errorGraph = new ErrorGraph(this);
		this.table = new TableEditor(this);
		this.weightsGraph = new WeightsGraph(this);
		
		this.leftVis = new TabSwitchVisualizationContainer($("#leftVis"), "leftVis", [
			this.netgraph, this.errorGraph, this.weightsGraph]);
		this.rightVis = new TabSwitchVisualizationContainer($("#rightVis"), "rightVis", [
			this.netviz, this.table]);
		this.deserializeFromUrl();
		this.leftVis.setMode(0);
		this.rightVis.setMode(0);
		this.constructed = true;
		this.onFrame();
		if (autoRun) this.run();
	}

	initializeNet(weights?: double[]) {
		console.log(`initializeNet(${weights})`);
		if (this.net) this.stop();
		this.net = new Net.NeuralNet(this.config.inputLayer, this.config.hiddenLayers, this.config.outputLayer, this.config.learningRate, true, undefined, weights);
		this.stepNum = 0;
		this.errorHistory = [];
		this.leftVis.onNetworkLoaded(this.net);
		this.rightVis.onNetworkLoaded(this.net);
		if(this.constructed) this.onFrame();
	}
	statusIterEle = document.getElementById('statusIteration');
	statusCorrectEle = document.getElementById('statusCorrect');
	step() {
		this.stepNum++;
		for (let val of this.config.data) {
			this.net.train(val.input, val.output);
		}
	}

	onFrame() {
		this.calculateAverageError();
		this.rightVis.currentVisualization.onFrame();
		this.leftVis.currentVisualization.onFrame();
		this.updateStatusLine();
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
		this.initializeNet();
		this.onFrame();
	}
	
	calculateAverageError() {
		this.averageError = 0;
		/*for (let val of this.config.data) {
			let res = this.net.getOutput(val.input);
			let sum1 = 0;
			for (let i = 0; i < this.net.outputs.length; i++) {
				let dist = res[i] - val.output[i];
				sum1 += dist * dist;
			}
			this.averageError += Math.sqrt(sum1);
		}
		this.averageError /= this.config.data.length;*/
		for (let val of this.config.data) {
			this.net.setInputsAndCalculate(val.input);
			this.averageError += this.net.getLoss(val.output);
		}
		this.averageError /= this.config.data.length;
		this.errorHistory.push([this.stepNum, this.averageError]);
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
			this.statusCorrectEle.innerHTML = `Avg. error: ${(this.averageError).toFixed(2) }`;
		}
		this.statusIterEle.innerHTML = this.stepNum.toString();

		if (correct == this.config.data.length) {
			if (this.config.autoRestart && this.running && this.restartTimeout == -1) {
				this.restartTimeout = setTimeout(() => {
					this.stop();
					this.restartTimeout = -1;
					setTimeout(() => { this.reset(); this.run(); }, 100);
				}, this.config.autoRestartTime);
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
		this.onFrame();
		if (this.running) this.runningId = requestAnimationFrame(this.aniFrameCallback);
	}

	iterations() {
		this.stop();
		for (var i = 0; i < this.config.iterationsPerClick; i++)
			this.step();
		this.onFrame();
	}

	setIsCustom(forceNeuronRename = false) {
		if (this.isCustom && !forceNeuronRename) return;
		this.isCustom = true;
		$("#presetName").text("Custom Network");
		let layer = this.config.inputLayer;
		layer.names = Net.Util.makeArray(layer.neuronCount, i => `in${i + 1}`);
		layer = this.config.outputLayer;
		layer.names = Net.Util.makeArray(layer.neuronCount, i => `out${i + 1}`);
	}

	loadConfig() { // from gui
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
		if(!this.config.autoRestart) clearTimeout(this.restartTimeout);
	}

	loadPreset(name: string, weights?:double[]) {
		this.isCustom = false;
		$("#presetName").text(`Preset: ${name}`);
		this.config = Presets.get(name);
		this.setConfig();
		history.replaceState({}, "", "?" + $.param({ preset: name }));
		this.initializeNet(weights);
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
		let weightString = getUrlParameter("weights");
		let weights:double[];
		if(weightString) 
			weights = JSON.parse(LZString.decompressFromBase64(weightString));
		if (preset && Presets.exists(preset))
			this.loadPreset(preset, weights);
		else if (config) {
			this.config = JSON.parse(LZString.decompressFromBase64(config));
			this.setIsCustom();
			this.initializeNet();
		} else
			this.loadPreset("Binary Classifier for XOR");
	}
}