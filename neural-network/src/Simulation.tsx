class Simulation extends React.Component<{autoRun: boolean}, Configuration> {
	netviz: NetworkVisualization;
	netgraph: NetworkGraph;
	table: TableEditor;
	errorGraph: ErrorGraph;
	weightsGraph: WeightsGraph;

	/** training steps that should be done by now (used for animation) */
	stepsWanted = 0;
	stepsCurrent = 0;
	frameNum = 0;
	running = false; runningId = -1;
	restartTimeout = -1;
	lastTimestamp = 0;
	averageError = 1;

	net: Net.NeuralNet;
	lrVis: LRVis;

	errorHistory: [number, number][];
	
	/** data of the last training steps. first entry has .dataPoint set to undefined and contains the previous weights */
	lastWeights: Net.WeightsStep[];
	
	static trainingMethods:{[type:string]: {[name:string]: (net:Net.NeuralNet, data:TrainingData[]) => Net.WeightsStep[]}} = {
		"nn": {
			"Batch Training": (net,data) => net.trainAll(data, false, false),
			"Online Training": (net,data) => net.trainAll(data, true, false)
		},
		"perceptron": {
			"Batch Training": (net,data) => net.trainAll(data, false, true),
			"Online Training": (net,data) => net.trainAll(data, true, true),
			"Averaged Perceptron": (net,data) => net.trainAllAveraged(data, true)
		}
	}
	trainingMethod: (data:TrainingData[]) => void;
	

	constructor(props:{autoRun: boolean}) {
		super(props);
		this.netviz = new NetworkVisualization(this, p => p === this.state.data[this.currentTrainingDataPoint]);
		this.netgraph = new NetworkGraph(this);
		this.errorGraph = new ErrorGraph(this);
		this.table = new TableEditor(this);
		this.weightsGraph = new WeightsGraph(this);
		this.state = this.deserializeFromUrl();
	}

	initializeNet() {
		if (this.net) this.stop();
		console.log("initializeNet()");
		this.net = new Net.NeuralNet(this.state.inputLayer, this.state.hiddenLayers, this.state.outputLayer, this.state.learningRate, undefined, this.state.weights);
		this.stepsWanted = this.stepsCurrent = 0;
		this.errorHistory = [];
		this.lastWeights = [];
		this.lrVis.leftVis.onNetworkLoaded(this.net);
		this.lrVis.rightVis.onNetworkLoaded(this.net);
		this.onFrame(true);
	}
	statusIterEle = document.getElementById('statusIteration');
	statusCorrectEle = document.getElementById('statusCorrect');
	trainAll() {
		this.currentTrainingDataPoint = -1;
		this.stepsCurrent++;
		if(this.state.saveLastWeights)
			this.lastWeights = [{dataPoint: null, weights: this.net.connections.map(c => c.weight)}];
		const steps = Simulation.trainingMethods[this.state.type][this.state.trainingMethod](this.net, this.state.data);
		this.lastWeights = this.lastWeights.concat(steps);
	}
	
	trainAllButton() {
		this.stop();
		for (var i = 0; i < this.state.iterationsPerClick; i++)
			this.trainAll();
		this.stepsWanted = this.stepsCurrent;
		this.onFrame(true);
	}
	
	trainNextButton() {
		this.stop();
		this.trainNext();
		this.stepsWanted = this.stepsCurrent;
		this.onFrame(true);
	}
	
	currentTrainingDataPoint = -1;
	trainNext() {
		this.currentTrainingDataPoint++;
		if(this.state.saveLastWeights)
			this.lastWeights = [{dataPoint: null, weights: this.net.connections.map(c => c.weight)}];
		this.stepsCurrent++;
		if(this.currentTrainingDataPoint >= this.state.data.length) {
			this.currentTrainingDataPoint -= this.state.data.length;
		}
		const newWeights = this.net.train(this.state.data[this.currentTrainingDataPoint], true, this.state.saveLastWeights);
		if(this.state.saveLastWeights) this.lastWeights.push(newWeights);
	}
	

	forwardPassEles:NetGraphUpdate[] = [];
	forwardPassStep() {
		if(!this.netgraph.currentlyDisplayingForwardPass) {
			this.forwardPassEles = [];
			this.currentTrainingDataPoint = -1;
		}
		this.stop();
		if(this.forwardPassEles.length > 0) {
			this.netgraph.applyUpdate(this.forwardPassEles.shift());
		} else {
			if(this.currentTrainingDataPoint < this.state.data.length - 1) {
				// start next
				this.lrVis.leftVis.setMode(0);
				this.currentTrainingDataPoint++;
				this.forwardPassEles = this.netgraph.forwardPass(this.state.data[this.currentTrainingDataPoint]);
				this.netgraph.applyUpdate(this.forwardPassEles.shift());
				this.netviz.onFrame();
			} else {
				// end
				this.currentTrainingDataPoint = -1;
				this.netgraph.onFrame(0);
				this.netviz.onFrame();
			}
		}
	}

	onFrame(forceDraw: boolean) {
		this.frameNum++;
		this.calculateAverageError();
		this.lrVis.onFrame(forceDraw ? 0 : this.frameNum);
		this.updateStatusLine();
	}

	run() {
		if (this.running) return;
		this.running = true;
		this.lrVis.setState({running:true});
		this.lastTimestamp = performance.now(); 
		requestAnimationFrame(this.aniFrameCallback);
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
		this.errorHistory.push([this.stepsCurrent, this.averageError]);
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
		this.lrVis.setState({stepNum: this.stepsCurrent});

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
	animationStep(timestamp: number) {
		let delta = timestamp - this.lastTimestamp;
		this.lastTimestamp = timestamp;
		if(delta > 1000 / 5) {
			console.warn(`only ${(1000/delta).toFixed(1)} fps`);
			delta = 1000 / 5;
		}
		this.stepsWanted += delta / 1000 * this.state.stepsPerSecond;  
		while(this.stepsCurrent < this.stepsWanted) this.state.animationTrainSinglePoints? this.trainNext():this.trainAll();
		this.onFrame(false);
		if (this.running) this.runningId = requestAnimationFrame(this.aniFrameCallback);
	}
	
	componentWillUpdate(nextProps: any, newConfig: Configuration) {
		if(this.state.hiddenLayers.length !== newConfig.hiddenLayers.length && newConfig.custom) {
			if (this.state.custom/* && !forceNeuronRename*/) return;
			const inN = newConfig.inputLayer.neuronCount;
			const outN = newConfig.outputLayer.neuronCount;
			newConfig.name = "Custom Network";
			newConfig.inputLayer = {names: Net.Util.makeArray(inN, i => `in${i + 1}`), neuronCount: inN};
			newConfig.outputLayer = {names: Net.Util.makeArray(outN, i => `out${i + 1}`), activation: newConfig.outputLayer.activation, neuronCount: outN};
		}
	}
	
	componentDidUpdate(prevProps: any, oldConfig: Configuration) {
		const co = oldConfig, cn = this.state;
		if (!cn.autoRestart) clearTimeout(this.restartTimeout);
		const layerDifferent = (l1:any, l2:any) =>
			l1.activation !== l2.activation || l1.neuronCount !== l2.neuronCount || (l1.names&&l1.names.some((name:string, i:number) => l2.names[i] !== name));
		if(cn.hiddenLayers.length !== co.hiddenLayers.length
			|| layerDifferent(cn.inputLayer, co.inputLayer)
			|| layerDifferent(cn.outputLayer, co.outputLayer)
			|| cn.hiddenLayers.some((layer,i) => layerDifferent(layer, co.hiddenLayers[i]))
			|| cn.weights && (!co.weights || cn.weights.some((weight, i) => co.weights[i] !== weight))) {
			this.initializeNet();
		}
		if(!cn.custom)
			history.replaceState({}, "", "?" + $.param({ preset: cn.name }));
		if (this.net) {
			if(co.bias != cn.bias) {
				this.netgraph.onNetworkLoaded(this.net);
			}
			if(co.type !== cn.type) {
				// gui layout may change, trigger resize
				window.dispatchEvent(new Event('resize'));
			}
			this.net.learnRate = cn.learningRate;
			if(cn.showGradient != co.showGradient || cn.drawCoordinateSystem != co.drawCoordinateSystem || cn.drawArrows != co.drawArrows)
				this.onFrame(false);
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
		if (this.state.custom || exportWeights > 0) {
			params.config = Util.cloneConfig(this.state);
		} else {
			params.preset = this.state.name;
		}
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
			<div>
				<div className="container">
					<div className="page-header">
						<div className="btn-toolbar pull-right dropdown" style={{marginTop:"5px"}}>
							<button className="btn btn-info dropdown-toggle" data-toggle="dropdown">{"Load Preset "}
								<span className="caret" />
							</button>
							<ul className="dropdown-menu">
								<li className="dropdown-header">Neural Network</li>
								{Presets.getNames().map(name => {
									const ele = <li key={name}><a onClick={e => sim.setState(Presets.get(name))}>{name}</a></li>;
									if(name === "Rosenblatt Perceptron")
										return [<li className="divider" />, <li className="dropdown-header">Perceptron</li>, ele];
									else return ele;
								})}
							</ul>
						</div>
						<h1>{this.state.type === "perceptron" ?"Perceptron":"Neural Network"} demo
							<small>{this.state.custom?" Custom Network":" Preset: "+this.state.name}</small>
						</h1>
					</div>
					<LRVis sim={this} ref={(e:LRVis) => this.lrVis = e}
						leftVis={[this.netgraph, this.errorGraph, this.weightsGraph]}
						rightVis={[this.netviz, this.table]}
					/>
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
				<ExportModal sim={this}/>
			</div>
		);
	}
}