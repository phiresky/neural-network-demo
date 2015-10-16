class Simulation extends React.Component<{autoRun: boolean}, Configuration> {
	netviz: NetworkVisualization;
	netgraph: NetworkGraph;
	table: TableEditor;
	errorGraph: ErrorGraph;
	weightsGraph: WeightsGraph;

	stepNum = 0;
	frameNum = 0;
	running = false; runningId = -1;
	restartTimeout = -1;
	averageError = 1;

	net: Net.NeuralNet;
	lrVis: LRVis;

	errorHistory: [number, number][];

	constructor(props:{autoRun: boolean}) {
		super(props);
		const doSerialize = () => {
			this.stop();
			$("#urlExport").val(this.serializeToUrl(+$(".exportWeights").val()));
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
		const config = this.state;
		$("#exportModal .exportJSON").click(() => {
			Util.download(JSON.stringify(this.state, null, '\t'), this.state.name + ".json");
		});
		$("#exportModal .exportCSV").click(() => {
			const csv = this.state.inputLayer.names.concat(this.state.outputLayer.names)
				.map(Util.csvSanitize).join(",") + "\n"
				+ this.state.data.map(data => 
					data.input.concat(data.output).join(",")).join("\n");
			Util.download(csv, this.state.name + ".csv");
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
					this.setState(conf);
					$("#exportModal").modal('hide');
					$("#presetName").text(file.name);
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
					const inps = this.state.inputLayer.neuronCount;
					const oups = this.state.outputLayer.neuronCount;
					if (len !== inps + oups)
						throw `invalid line length, expected (${inps} inputs + ${oups} outputs = ) ${inps+oups} columns, got ${len} columns`;
					const newState:any = {};
					if(!data[0][0].match(/^\d+$/)) {
						const headers = data.shift();
						newState.inputLayer = {names: headers.slice(0, inps), neuronCount: this.state.inputLayer.neuronCount};
						newState.outputLayer = {names: headers.slice(inps, inps + oups), neuronCount: this.state.outputLayer.neuronCount};
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
					newState.data= trainingsData;
					this.setState(newState, () => this.table.loadData());
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
		this.state = this.deserializeFromUrl();
	}

	initializeNet() {
		if (this.net) this.stop();
		console.log("initializeNet()"+this.state.weights);
		this.net = new Net.NeuralNet(this.state.inputLayer, this.state.hiddenLayers, this.state.outputLayer, this.state.learningRate, undefined, this.state.weights);
		this.stepNum = 0;
		this.errorHistory = [];
		this.lrVis.leftVis.onNetworkLoaded(this.net);
		this.lrVis.rightVis.onNetworkLoaded(this.net);
		this.onFrame(true);
	}
	statusIterEle = document.getElementById('statusIteration');
	statusCorrectEle = document.getElementById('statusCorrect');
	step() {
		this.stepNum++;
		this.net.trainAll(this.state.data, !this.state.batchTraining);
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
			if(this.forwardPassState < this.state.data.length - 1) {
				// start next
				this.lrVis.leftVis.setMode(0);
				this.forwardPassState++;
				this.forwardPassEles = this.netgraph.forwardPass(this.state.data[this.forwardPassState]);
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
		for (const val of this.state.data) {
			this.net.setInputsAndCalculate(val.input);
			this.averageError += this.net.getLoss(val.output);
		}
		this.averageError /= this.state.data.length;
		this.errorHistory.push([this.stepNum, this.averageError]);
	}

	updateStatusLine() {
		let correct = 0;
		if (this.state.outputLayer.neuronCount === 1) {
			for (var val of this.state.data) {
				const res = this.net.getOutput(val.input);
				if (+(res[0] > 0.5) == val.output[0]) correct++;
			}
			this.lrVis.setState({correct:`Correct: ${correct}/${this.state.data.length}`});
		} else {
			this.lrVis.setState({correct:`Error: ${(this.averageError).toFixed(2) }`});
		}
		this.lrVis.setState({stepNum: this.stepNum});

		if (correct == this.state.data.length) {
			if (this.state.autoRestart && this.running && this.restartTimeout == -1) {
				this.restartTimeout = setTimeout(() => {
					this.stop();
					this.restartTimeout = -1;
					setTimeout(() => { this.reset(); this.run(); }, 100);
				}, this.state.autoRestartTime);
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
		for (let i = 0; i < this.state.stepsPerFrame; i++) this.step();
		this.onFrame(false);
		if (this.running) this.runningId = requestAnimationFrame(this.aniFrameCallback);
	}

	iterations() {
		this.stop();
		for (var i = 0; i < this.state.iterationsPerClick; i++)
			this.step();
		this.onFrame(true);
	}
	
	componentWillUpdate(nextProps: any, newConfig: Configuration) {
		if(this.state.hiddenLayers.length !== newConfig.hiddenLayers.length && newConfig.custom) {
			if (this.state.custom/* && !forceNeuronRename*/) return;
			$("#presetName").text("Custom Network");
			const inN = newConfig.inputLayer.neuronCount;
			const outN = newConfig.outputLayer.neuronCount;
			newConfig.name = "Custom Network";
			newConfig.inputLayer = {names: Net.Util.makeArray(inN, i => `in${i + 1}`), neuronCount: inN};
			newConfig.outputLayer = {names: Net.Util.makeArray(outN, i => `out${i + 1}`), activation: newConfig.outputLayer.activation, neuronCount: outN};
		}
	}
	
	componentDidUpdate(prevProps: any, oldConfig: Configuration) {
		if (!this.state.autoRestart) clearTimeout(this.restartTimeout);
		const layerDifferent = (l1:any, l2:any) =>
			l1.activation !== l2.activation || l1.neuronCount !== l2.neuronCount || (l1.names&&l1.names.some((name:string, i:number) => l2.names[i] !== name));
		if(this.state.hiddenLayers.length !== oldConfig.hiddenLayers.length
			|| layerDifferent(this.state.inputLayer, oldConfig.inputLayer)
			|| layerDifferent(this.state.outputLayer, oldConfig.outputLayer)
			|| this.state.hiddenLayers.some((layer,i) => layerDifferent(layer, oldConfig.hiddenLayers[i]))
			|| this.state.weights && (!oldConfig.weights || this.state.weights.some((weight, i) => oldConfig.weights[i] !== weight))) {
			this.initializeNet();
		}
		if(!this.state.custom)
			history.replaceState({}, "", "?" + $.param({ preset: this.state.name }));
		if (this.net) {
			if(oldConfig.bias != this.state.bias) {
				this.netgraph.onNetworkLoaded(this.net);
			}
			this.net.learnRate = this.state.learningRate;
		}
	}
	componentDidMount() {
		this.initializeNet();
		this.onFrame(true);
		if (this.props.autoRun) this.run();
	}

	loadConfig() { // from gui
		const config = $.extend(true, {}, this.state);
		for (const conf in config) {
			const ele = document.getElementById(conf) as HTMLInputElement;
			if (!ele) continue;
			if (ele.type == 'checkbox') config[conf] = ele.checked;
			else if (typeof config[conf] === 'number')
				config[conf] = +ele.value;
			else config[conf] = ele.value;
		}
		config.learningRate = Util.expScale(config.learningRate);
		this.setState(config);
	}

	runtoggle() {
		if (this.running) this.stop();
		else this.run();
	}

	// 0 = no weights, 1 = current weights, 2 = start weights
	serializeToUrl(exportWeights = 0) {
		const url = location.protocol + '//' + location.host + location.pathname + "?";
		const params: any = {};
		console.log("cust"+exportWeights);
		if (this.state.custom || exportWeights > 0) {
			params.config = Util.cloneConfig(this.state);
		} else {
			params.preset = this.state.name;
		}
		console.log(exportWeights);
		if (exportWeights === 1) params.config.weights = this.net.connections.map(c => c.weight);
		if (exportWeights === 2) params.config.weights = this.net.startWeights;
		
		if(params.config) params.config = LZString.compressToEncodedURIComponent(JSON.stringify(params.config));
		return url + $.param(params);
	}
	deserializeFromUrl(): Configuration {
		const urlParams = Util.parseUrlParameters();
		const preset = urlParams["preset"], config = urlParams["config"];
		if (preset && Presets.exists(preset))
			return Presets.get(preset);
		else if (config) {
			console.log(JSON.parse(LZString.decompressFromEncodedURIComponent(config)));
			return JSON.parse(LZString.decompressFromEncodedURIComponent(config));
		} else
			return Presets.get("Binary Classifier for XOR");
	}
	shouldComponentUpdate() {
		return true;
	}
	
	render() {
		return (
			<div className="container">
				<div className="page-header">
					<h1>Neural Network demo
						<small>{this.state.custom?" Custom Network":" Preset: "+this.state.name}</small>
					</h1>
				</div>
				<LRVis sim={this} ref={(e:LRVis) => this.lrVis = e} />
				<div className="panel panel-default">
					<div className="panel-heading">
						<h3 className="panel-title">
							<a data-toggle="collapse" data-target=".panel-body">Configuration</a>
						</h3>
					</div>
					<div className="panel-body collapse in">
						<ConfigurationGui {...this.state} />
					</div>
				</div>
				<footer className="small">
					<a href="https://github.com/phiresky/kogsys-demos/">Source on GitHub</a>
				</footer>
			</div>
		);
	}
}