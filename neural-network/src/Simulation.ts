///<reference path='../lib/typings/jquery/jquery.d.ts' />
///<reference path='Net.ts' />
///<reference path='NetworkGraph.ts' />
///<reference path='NetworkVisualization.ts' />

class Simulation {
	netviz: NetworkVisualization;
	netgraph: NetworkGraph;
	backgroundResolution = 10;
	stepNum = 0;
	running = false; runningId = -1;
	restartTimeout = -1;
	data: Data[] = [
		{ x: 0, y: 0, label: 0 },
		{ x: 0, y: 1, label: 1 },
		{ x: 1, y: 0, label: 1 },
		{ x: 1, y: 1, label: 0 }
	];
	net: Net.NeuralNet;
	config = {
		stepsPerFrame: 50,
		learningRate: 0.05,
		activation: "sigmoid",
		showGradient: false,
		bias: true,
		autoRestartTime: 5000
		//lossType: "svm"
	};

	constructor() {
		let canvas = <HTMLCanvasElement>$("#neuralOutputCanvas")[0];
		this.netviz = new NetworkVisualization(canvas,
			new CanvasMouseNavigation(canvas, () => this.draw()));
		this.netgraph = new NetworkGraph($("#neuralNetworkGraph")[0]);
		(<any>$("#learningRate")).slider({
			min: 0.01, max: 1, step: 0.005, scale: "logarithmic", value: 0.05
		}).on('slide', (e: any) => $("#learningRateVal").text(e.value.toFixed(2)));
		this.reset();
		this.run();
	}

	initializeNet() {
		//let cache = [0.18576880730688572,-0.12869677506387234,0.08548374730162323,-0.19820863520726562,-0.09532690420746803,-0.3415223266929388,-0.309354952769354,-0.157513455953449];
		//let cache = [-0.04884958150796592,-0.3569231238216162,0.11143312812782824,0.43614205135963857,0.3078767384868115,-0.22759653301909566,0.09250503336079419,0.3279339636210352];
		this.net = new Net.NeuralNet([2, 2, 1], ["x", "y"], this.config.bias);
		console.log("net:" + JSON.stringify(this.net.connections.map(c => c.weight)));
		this.stepNum = 0;
		this.netgraph.loadNetwork(this.net);
	}

	statusIterEle = document.getElementById('statusIteration');
	statusCorrectEle = document.getElementById('statusCorrect');
	step() {
		this.stepNum++;
		for (let val of this.data) {
			let stats = this.net.train([val.x, val.y], [val.label]);
		}
		let correct = 0;
		for (let val of this.data) {
			let res = this.net.getOutput([val.x, val.y]);
			let label = (res[0] > 0.5) ? 1 : 0;
			if (val.label == label) correct++;
		}
		this.statusIterEle.innerHTML = this.stepNum.toString();
		this.statusCorrectEle.innerHTML = `${correct}/${this.data.length}`;
		if (correct == this.data.length) {
			if (this.running && this.restartTimeout == -1) {
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

	draw() {
		this.netviz.drawBackground(this.backgroundResolution,
			(x, y) => this.net.getOutput([x, y])[0]);
		this.netviz.drawCoordinateSystem();
		this.netviz.drawDataPoints(this.data);
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
		this.draw();
	}

	aniFrameCallback = this.animationStep.bind(this);
	animationStep() {
		for (let i = 0; i < this.config.stepsPerFrame; i++) this.step();
		this.draw();
		if (this.running) this.runningId = requestAnimationFrame(this.aniFrameCallback);
	}

	iterations() {
		this.stop();
		var count = +$("#iterations").val();
		for (var i = 0; i < count; i++)
			this.step();
		this.draw();
	}

	loadConfig() {
		let config = <any>this.config;
		for (let conf in config) {
			let ele = <HTMLInputElement>document.getElementById(conf);
			if (!ele) continue;
			if (ele.type == 'checkbox') config[conf] = ele.checked;
			else config[conf] = ele.value;
		}
		Net.setLinearity(this.config.activation);
		if (this.net) this.net.learnRate = this.config.learningRate;
		this.netviz.showGradient = this.config.showGradient;
	}

	randomizeData() {
		let count = 4;
		for (let i = 0; i < count; i++) {
			this.data[i] = { x: Math.random() * 2, y: Math.random() * 2, label: +(Math.random() > 0.5) };
		}
		this.draw();
	}
	runtoggle() {
		if (this.running) {
			this.stop();
		} else {
			this.run();
		}
	}
}