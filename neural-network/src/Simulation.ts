///<reference path='../lib/typings/jquery/jquery.d.ts' />
///<reference path='Net.ts' />
///<reference path='NetworkGraph.ts' />
///<reference path='NetworkVisualization.ts' />
///<reference path='Presets.ts' />

class NeuronGui {
	layerDiv: JQuery = $("#neuronCountModifier > div").eq(1).clone();

	removeLayer() {
		$("#neuronCountModifier > div").eq(1).remove();
	}
	addLayer() {
		$("#neuronCountModifier > div").eq(1).before(this.layerDiv.clone());
	}
	setNeuronCount(layer: int, newval: int) {
		$("#neuronCountModifier .neuronCount").eq(layer).text(newval);
	}
	setActivation(layer: int, activ: string) {
		$("#neuronCountModifier > div").eq(layer).children("select.activation").val(activ);
	}
	constructor(public sim: Simulation) {
		$("#neuronCountModifier").on("click", "button", e => {
			let inc = e.target.textContent == '+';
			let layer = $(e.target.parentNode).index();
			let newval = sim.config.netLayers[layer].neuronCount + (inc ? 1 : -1);
			if (newval < 1) return;
			sim.config.netLayers[layer].neuronCount = newval;
			this.setNeuronCount(layer, newval);
			sim.initializeNet();
		});
		$("#layerCountModifier").on("click", "button", e => {
			let inc = e.target.textContent == '+';
			if (!inc) {
				if (sim.config.netLayers.length == 2) return;
				sim.config.netLayers.splice(1, 1);
				this.removeLayer();
			} else {
				this.addLayer();
				sim.config.netLayers.splice(1, 0, { activation: 'sigmoid', neuronCount: 2 });
			}
			$("#layerCount").text(sim.config.netLayers.length);
			sim.initializeNet();
		});
		$("#neuronCountModifier").on("change", "select", e => {
			let layer = $(e.target.parentNode).index();
			sim.config.netLayers[layer].activation = (<HTMLSelectElement>e.target).value;
			sim.initializeNet();
		});
	}
	regenerate() {
		let targetCount = this.sim.config.netLayers.length;
		while ($("#neuronCountModifier > div").length > targetCount)
			this.removeLayer();
		while ($("#neuronCountModifier > div").length < targetCount)
			this.addLayer();
		this.sim.config.netLayers.forEach(
			(c: LayerConfig, i: int) => {
				this.setNeuronCount(i, c.neuronCount);
				this.setActivation(i, c.activation);
			});
	}
}
interface LayerConfig {
	neuronCount: int;
	activation?: string;
}
class Simulation {
	netviz: NetworkVisualization;
	netgraph: NetworkGraph;
	backgroundResolution = 10;
	stepNum = 0;
	running = false; runningId = -1;
	restartTimeout = -1;

	net: Net.NeuralNet;
	neuronGui: NeuronGui;
	config = Presets.get('XOR');

	constructor() {
		let canvas = <HTMLCanvasElement>$("#neuralOutputCanvas")[0];
		this.netviz = new NetworkVisualization(canvas,
			new CanvasMouseNavigation(canvas, () => this.draw()),
			this,
			(x, y) => this.net.getOutput([x, y])[0],
			this.backgroundResolution);
		this.netgraph = new NetworkGraph($("#neuralNetworkGraph")[0]);
		(<any>$("#learningRate")).slider({
			min: 0.01, max: 1, step: 0.005, scale: "logarithmic", value: 0.05
		}).on('slide', (e: any) => $("#learningRateVal").text(e.value.toFixed(2)));
		this.neuronGui = new NeuronGui(this);
		$("#presetLoader").on("click", "a", e => {
			let name = e.target.textContent;
			this.config = Presets.get(name);
			this.setConfig();
			this.initializeNet();
		});
		$("#dataInputSwitch").on("click","a", e => {
			$("#dataInputSwitch li.active").removeClass("active");
			let li = $(e.target).parent();
			li.addClass("active");
			let mode = li.index();
			this.netviz.inputMode = mode;
		});
		this.reset();
		this.run();
	}

	initializeNet(weights?: double[]) {
		if (this.net) this.stop();
		this.net = new Net.NeuralNet(this.config.netLayers, ["x", "y"], this.config.learningRate, this.config.bias, undefined, weights);
		let isBinClass = this.config.simType == SimulationType.BinaryClassification;
		$("#dataInputSwitch > li").eq(1).toggle(isBinClass);
		$("#dataInputSwitch > li > a").eq(0).text(isBinClass?"Add Red":"Add point");
		console.log("net:" + JSON.stringify(this.net.connections.map(c => c.weight)));
		this.stepNum = 0;
		this.netgraph.loadNetwork(this.net);
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
		this.netviz.draw();
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
		this.loadConfig();
		this.initializeNet();
	}

	updateStatusLine() {
		let correct = 0;
		switch (this.config.simType) {
			case SimulationType.BinaryClassification:
				for (var val of this.config.data) {
					let res = this.net.getOutput(val.input);
					if (+(res[0] > 0.5) == val.output[0]) correct++;
				}
				this.statusCorrectEle.innerHTML = `Correct: ${correct}/${this.config.data.length}`;
				break;
			case SimulationType.AutoEncoder:
				let avgDist = (<TrainingData[]>this.config.data)
					.map(point => ({ a: point.output, b: this.net.getOutput(point.input) }))
					.map(x => ({ dx: x.a[0] - x.b[0], dy: x.a[1] - x.b[1] }))
					.reduce((a, b) => a + Math.sqrt(b.dx * b.dx + b.dy * b.dy), 0) / this.config.data.length;
				this.statusCorrectEle.innerHTML = `Avg. distance: ${avgDist.toFixed(2) }`;
				break;
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

	loadConfig() { // from gui
		let config = <any>this.config;
		let oldConfig = $.extend({}, config);
		for (let conf in config) {
			let ele = <HTMLInputElement>document.getElementById(conf);
			if (!ele) continue;
			if (ele.type == 'checkbox') config[conf] = ele.checked;
			else if(typeof config[conf] === 'number')
				config[conf] = +ele.value;
			else config[conf] = ele.value;
		}
		if(oldConfig.simType != config.simType) config.data = [];
		if (this.net) this.net.learnRate = this.config.learningRate;
	}
	setConfig() { // in gui
		let config = <any>this.config;
		for (let conf in config) {
			let ele = <HTMLInputElement>document.getElementById(conf);
			if (!ele) continue;
			if (ele.type == 'checkbox') ele.checked = config[conf];
			else ele.value = config[conf];
		}
		this.neuronGui.regenerate();
	}

	randomizeData() {
		if (this.config.netLayers[0].neuronCount !== 2 || this.config.simType !== SimulationType.BinaryClassification)
			throw "can't create random data for this network";
		let count = Math.random() * 5 + 4;
		this.config.data = [];
		for (let i = 0; i < count; i++) {
			this.config.data[i] = { input: [Math.random() * 2, Math.random() * 2], output: [+(Math.random() > 0.5)] };
		}
		this.draw();
	}
	runtoggle() {
		if (this.running) this.stop();
		else this.run();
	}
}