///<reference path='../lib/typings/jquery/jquery.d.ts' />
///<reference path='Net.ts' />
///<reference path='NetworkGraph.ts' />
///<reference path='NetworkVisualization.ts' />
///<reference path='Presets.ts' />
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
	hiddenLayerDiv: JQuery = $("#neuronCountModifier div").eq(1).clone();

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
		$("#neuronCountModifier").on("click", "button", e => {
			let inc = e.target.textContent == '+';
			let layer = $(e.target.parentNode).index();
			let newval = this.config.netLayers[layer].neuronCount + (inc ? 1 : -1);
			if (newval < 1) return;
			this.config.netLayers[layer].neuronCount = newval;
			$("#neuronCountModifier .neuronCount").eq(layer).text(newval);
			this.initializeNet();
		});
		$("#layerCountModifier").on("click", "button", e => {
			let inc = e.target.textContent == '+';
			if (!inc) {
				if (this.config.netLayers.length == 2) return;
				this.config.netLayers.splice(1, 1);
				$("#neuronCountModifier div").eq(1).remove();
			} else {
				$("#neuronCountModifier div").eq(1).before(this.hiddenLayerDiv.clone());
				this.config.netLayers.splice(1, 0, { activation: 'sigmoid', neuronCount: 2 });
			}
			$("#layerCount").text(this.config.netLayers.length);
			this.initializeNet();
		});
		$("#neuronCountModifier").on("change", "select", e => {
			let layer = $(e.target.parentNode).index();
			this.config.netLayers[layer].activation = (<HTMLSelectElement>e.target).value;
			this.initializeNet();
		});
		$("#presetLoader").on("click", "a", e => {
			let name = e.target.textContent;
			this.config = Presets.get(name);
			this.initializeNet();
		});
		this.reset();
		this.run();
	}

	initializeNet(weights?: double[]) {
		if (this.net) this.stop();
		//let cache = [0.18576880730688572,-0.12869677506387234,0.08548374730162323,-0.19820863520726562,-0.09532690420746803,-0.3415223266929388,-0.309354952769354,-0.157513455953449];
		//let cache = [-0.04884958150796592,-0.3569231238216162,0.11143312812782824,0.43614205135963857,0.3078767384868115,-0.22759653301909566,0.09250503336079419,0.3279339636210352];
		this.net = new Net.NeuralNet(this.config.netLayers, ["x", "y"], this.config.learningRate, this.config.bias, undefined, weights);
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
				this.statusCorrectEle.innerHTML = `Avg. distance: ${avgDist.toFixed(2)}`;
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

	loadConfig() {
		let config = <any>this.config;
		for (let conf in config) {
			let ele = <HTMLInputElement>document.getElementById(conf);
			if (!ele) continue;
			if (ele.type == 'checkbox') config[conf] = ele.checked;
			else config[conf] = ele.value;
		}
		if (this.net) this.net.learnRate = this.config.learningRate;
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