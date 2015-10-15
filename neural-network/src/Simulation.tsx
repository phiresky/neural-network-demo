class Simulation {
	netviz: NetworkVisualization;
	netgraph: NetworkGraph;
	table: TableEditor;
	errorGraph: ErrorGraph;
	weightsGraph: WeightsGraph;

	stepNum = 0;
	frameNum = 0;
	running = false; runningId = -1;
	restartTimeout = -1;
	isCustom = false;
	averageError = 1;

	net: Net.NeuralNet;
	config: Configuration;
	lrVis: LRVis;

	constructed = false;
	errorHistory: [number, number][];

	constructor(autoRun: boolean) {
		const doSerialize = () => {
			this.stop();
			$("#urlExport").val(this.serializeToUrl(+$("#exportWeights").val()));
		};
		const ioError = (txt: string) => {
			$("#importexporterror")
				.clone()
				.append(txt)
				.appendTo("#exportModal .modal-body")
				.show();
		}
		$("#exportModal").on("shown.bs.modal", doSerialize);
		$("#exportModal .exportWeights").on("change", doSerialize);
		$("#exportModal .exportJSON").click(() => {
			Util.download(JSON.stringify(this.config, null, '\t'), this.config.name + ".json");
		});
		$("#exportModal .exportCSV").click(() => {
			const csv = this.config.inputLayer.names.concat(this.config.outputLayer.names)
				.map(Util.csvSanitize).join(",") + "\n"
				+ this.config.data.map(data => 
					data.input.concat(data.output).join(",")).join("\n");
			Util.download(csv, this.config.name + ".csv");
		});
		$("#exportModal .importJSON").change(e => {
			const ev = e.originalEvent;
			const files = (ev.target as HTMLInputElement).files;
			if (files.length !== 1) ioError("invalid selection");
			const file = files.item(0);
			const r = new FileReader();
			r.onload = t => {
				try {
					const text = r.result;
					const conf = JSON.parse(text);
					this.config = conf;
					$("#exportModal").modal('hide');
					$("#presetName").text(file.name);
					this.renderConfigGui();
					this.initializeNet();
				} catch (e) {
					ioError("Error while reading " + file.name + ": " + e);
				}
			}
			r.readAsText(file);
		});
		$("#exportModal .importCSV").change(e => {
			const ev = e.originalEvent;
			const files = (ev.target as HTMLInputElement).files;
			if (files.length !== 1) ioError("invalid selection");
			const file = files.item(0);
			const r = new FileReader();
			r.onload = t => {
				try {
					const text = r.result as string;
					const data = text.split("\n").map(l => l.split(","));
					const lens = data.map(l => l.length);
					const len = Math.min(...lens);
					if (len !== Math.max(...lens))
						throw `line lengths varying between ${len} and ${Math.max(...lens) }, must be constant`;
					const inps = this.config.inputLayer.neuronCount;
					const oups = this.config.outputLayer.neuronCount;
					if (len !== inps + oups)
						throw `invalid line length, expected (${inps} inputs + ${oups} outputs = ) ${inps+oups} columns, got ${len} columns`;
					if(!data[0][0].match(/^\d+$/)) {
						const headers = data.shift();
						this.config.inputLayer.names = headers.slice(0, inps);
						this.config.outputLayer.names = headers.slice(inps, inps + oups);
					}
					const trainingsData:TrainingData[] = [];
					for(let l = 0; l < data.length; l++) {
						const ele:TrainingData = {input:[], output:[]};
						for(let i = 0; i < len; i++) {
							const v = parseFloat(data[l][i]);
							if(isNaN(v)) throw `can't parse ${data[l][i]} as a number in line ${l+1}`;
							(i < inps ? ele.input:ele.output).push(v);
						}
						trainingsData.push(ele);
					}
					this.config.data = trainingsData;
					this.table.loadData();
					$("#presetName").text(file.name);
					$("#exportModal").modal('hide');
				} catch (e) {
					ioError("Error while reading " + file.name + ": " + e);
				}
			}
			r.readAsText(file);
		});
		this.netviz = new NetworkVisualization(this);
		this.netgraph = new NetworkGraph(this);
		this.errorGraph = new ErrorGraph(this);
		this.table = new TableEditor(this);
		this.weightsGraph = new WeightsGraph(this);
		this.lrVis = React.render(<LRVis sim={this}/>, document.getElementById("lrVisTarget")) as LRVis;
		this.deserializeFromUrl();
		this.renderConfigGui();
		this.constructed = true;
		this.onFrame(true);
		if (autoRun) this.run();
	}

	initializeNet(weights?: double[]) {
		console.log(`initializeNet(${weights})`);
		if (this.net) this.stop();
		this.net = new Net.NeuralNet(this.config.inputLayer, this.config.hiddenLayers, this.config.outputLayer, this.config.learningRate, undefined, weights);
		this.stepNum = 0;
		this.errorHistory = [];
		this.lrVis.leftVis.onNetworkLoaded(this.net);
		this.lrVis.rightVis.onNetworkLoaded(this.net);
		if (this.constructed) this.onFrame(true);
	}
	statusIterEle = document.getElementById('statusIteration');
	statusCorrectEle = document.getElementById('statusCorrect');
	step() {
		this.stepNum++;
		this.net.trainAll(this.config.data);
	}
	
	forwardPassState = -1;
	forwardPassEles:NetGraphUpdate[] = [];
	forwardPassStep() {
		if(!this.netgraph.currentlyDisplayingForwardPass) {
			this.forwardPassEles = [];
			this.forwardPassState = -1;
		}
		this.stop();
		if(this.forwardPassEles.length > 0) {
			this.netgraph.applyUpdate(this.forwardPassEles.shift());
		} else {
			if(this.forwardPassState < this.config.data.length - 1) {
				// start next
				this.lrVis.leftVis.setMode(0);
				this.forwardPassState++;
				this.forwardPassEles = this.netgraph.forwardPass(this.config.data[this.forwardPassState]);
				this.netgraph.applyUpdate(this.forwardPassEles.shift());
			} else {
				// end
				this.forwardPassState = -1;
				this.netgraph.onFrame(0);
			}
		}
	}

	onFrame(forceDraw: boolean) {
		this.frameNum++;
		this.calculateAverageError();
		this.lrVis.state.leftVisBody.onFrame(forceDraw ? 0 : this.frameNum);
		this.lrVis.state.rightVisBody.onFrame(forceDraw ? 0 : this.frameNum);
		this.updateStatusLine();
	}

	run() {
		if (this.running) return;
		this.running = true;
		this.lrVis.setState({running:true});
		this.animationStep();
	}

	stop() {
		clearTimeout(this.restartTimeout);
		this.restartTimeout = -1;
		this.running = false;
		this.lrVis.setState({running:false});
		cancelAnimationFrame(this.runningId);
	}

	reset() {
		this.stop();
		this.initializeNet();
		this.onFrame(true);
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
		for (const val of this.config.data) {
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
				const res = this.net.getOutput(val.input);
				if (+(res[0] > 0.5) == val.output[0]) correct++;
			}
			this.lrVis.setState({correct:`Correct: ${correct}/${this.config.data.length}`});
		} else {
			this.lrVis.setState({correct:`Error: ${(this.averageError).toFixed(2) }`});
		}
		this.lrVis.setState({stepNum: this.stepNum});

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
		this.onFrame(false);
		if (this.running) this.runningId = requestAnimationFrame(this.aniFrameCallback);
	}

	iterations() {
		this.stop();
		for (var i = 0; i < this.config.iterationsPerClick; i++)
			this.step();
		this.onFrame(true);
	}

	setIsCustom(forceNeuronRename = false) {
		if (this.isCustom && !forceNeuronRename) return;
		this.isCustom = true;
		$("#presetName").text("Custom Network");
		this.config.name = "Custom Network";
		let layer = this.config.inputLayer;
		layer.names = Net.Util.makeArray(layer.neuronCount, i => `in${i + 1}`);
		layer = this.config.outputLayer;
		layer.names = Net.Util.makeArray(layer.neuronCount, i => `out${i + 1}`);
	}

	loadConfig() { // from gui
		const config = this.config as any;
		const oldConfig = $.extend({}, config);
		for (const conf in config) {
			const ele = document.getElementById(conf) as HTMLInputElement;
			if (!ele) continue;
			if (ele.type == 'checkbox') config[conf] = ele.checked;
			else if (typeof config[conf] === 'number')
				config[conf] = +ele.value;
			else config[conf] = ele.value;
		}
		this.config.learningRate = Util.expScale(this.config.learningRate);
		if (oldConfig.simType != config.simType) config.data = [];
		if (this.net) {
			if(oldConfig.bias != config.bias) {
				//this.net.
				this.netgraph.onNetworkLoaded(this.net);
			}
			this.net.learnRate = this.config.learningRate;
		}
		if (!this.config.autoRestart) clearTimeout(this.restartTimeout);
		this.renderConfigGui();
	}
	
	renderConfigGui() {
		React.render(<ConfigurationGui {...this.config}/>, document.getElementById("configurationTarget"));
	}

	loadPreset(name: string, weights?: double[]) {
		this.isCustom = false;
		$("#presetName").text(`Preset: ${name}`);
		this.config = Presets.get(name);
		this.renderConfigGui();
		history.replaceState({}, "", "?" + $.param({ preset: name }));
		this.initializeNet(weights);
	}

	runtoggle() {
		if (this.running) this.stop();
		else this.run();
	}

	// 0 = no weights, 1 = current weights, 2 = start weights
	serializeToUrl(exportWeights = 0) {
		const url = location.protocol + '//' + location.host + location.pathname + "?";
		const params: any = {};
		if (exportWeights === 1) params.weights = LZString.compressToEncodedURIComponent(JSON.stringify(this.net.connections.map(c => c.weight)));
		if (exportWeights === 2) params.weights = LZString.compressToEncodedURIComponent(JSON.stringify(this.net.startWeights));
		if (this.isCustom) {
			params.config = LZString.compressToEncodedURIComponent(JSON.stringify(this.config));
		} else {
			params.preset = this.config.name;
		}

		return url + $.param(params);
	}
	deserializeFromUrl() {
		const urlParams = Util.parseUrlParameters();
		const preset = urlParams["preset"], config = urlParams["config"];
		const weightString = urlParams["weights"];
		let weights: double[];
		if (weightString)
			weights = JSON.parse(LZString.decompressFromEncodedURIComponent(weightString));
		if (preset && Presets.exists(preset))
			this.loadPreset(preset, weights);
		else if (config) {
			this.config = JSON.parse(LZString.decompressFromEncodedURIComponent(config));
			this.setIsCustom();
			this.initializeNet();
		} else
			this.loadPreset("Binary Classifier for XOR");
	}
}