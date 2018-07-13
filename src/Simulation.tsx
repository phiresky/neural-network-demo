import * as React from "react";
import * as LZString from "lz-string";
import ErrorGraph from "./Visualizations/ErrorGraph";
import ConfigurationGui from "./ConfigurationGui";
import ExportModal from "./ExportModal";
import {
	TableEditor,
	WeightsGraph,
	LRVis,
	NetworkVisualization,
	NetworkGraph,
	NetGraphUpdate
} from "./Visualizations";
import Net from "./Net";
import Presets from "./Presets";
import * as Util from "./Util";
import { TrainingData, Configuration } from "./Configuration";
import * as $ from "jquery";
import "jquery";
import "bootstrap/dist/js/bootstrap";
/**
 * the interface between the GUI and the Simulation / Neural network
 *
 * handles buttons, animation and configuration updates with the help of React
 *
 * the [[#state]] of this object contains the [[Configuration]]
 */
export default class Simulation extends React.Component<
	{ autoRun: boolean },
	Configuration
> {
	static instance: Simulation;

	netviz: NetworkVisualization;
	netgraph: NetworkGraph;
	table: TableEditor;
	errorGraph: ErrorGraph;
	weightsGraph: WeightsGraph;

	/** training steps that should be done by now (used for animation) */
	private stepsWanted = 0;
	/**
	 * training steps that have been done since [[#initializeNet]]
	 * single training steps or batch training steps both count as 1 step
	 */
	stepsCurrent = 0;
	/**
	 * global frame counter
	 * used to limit computationally expensive functions to frameNum modulo n (see NetworkGraph#onFrame)
	 */
	private frameNum = 0;
	/** animation is currently running */
	private running = false;
	/** the current requestAnimationFrame ID */
	private runningId = -1;
	/** the setTimeout ID for restarting the simulation after some time (see Configuration#autoRestartTime) */
	private restartTimeout = -1;
	private lastTimestamp = 0;
	/** current average error (see Net.NeuralNet#getLoss) */
	averageError = 1;

	net: Net.NeuralNet;
	lrVis: LRVis;
	exportModal: ExportModal;

	/** list of [stepNum, averageError] elements */
	errorHistory: [number, number][];

	/** data of the last training steps. first entry has .dataPoint set to undefined and contains the previous weights */
	lastWeights: Net.WeightsStep[];

	/** current training method (one of Net.trainingMethods) */
	get trainingMethod(): Net.TrainingMethod {
		return Net.trainingMethods[this.state.type][this.state.trainingMethod];
	}

	constructor(props: { autoRun: boolean }) {
		super(props);
		if (Simulation.instance) throw Error("Already instantiated");
		else Simulation.instance = this;
		this.netviz = new NetworkVisualization(
			this,
			p => p === this.state.data[this.currentTrainingDataPoint]
		);
		this.netgraph = new NetworkGraph(this);
		this.errorGraph = new ErrorGraph(this);
		this.table = new TableEditor(this);
		this.weightsGraph = new WeightsGraph(this);
		this.state = this.deserializeFromUrl();
	}

	/** initialize a new random network */
	initializeNet() {
		if (this.net) this.stop();
		console.log("initializeNet()");
		this.net = new Net.NeuralNet(
			this.state.inputLayer,
			this.state.hiddenLayers,
			this.state.outputLayer,
			this.state.learningRate,
			undefined,
			this.state.weights
		);
		this.stepsWanted = this.stepsCurrent = 0;
		this.errorHistory = [];
		this.lastWeights = [];
		this.lrVis.leftVis.onNetworkLoaded(this.net);
		this.lrVis.rightVis.onNetworkLoaded(this.net);
		this.currentTrainingDataPoint = -1;
		this.onFrame(true);
	}

	/** train all data points according to [[#trainingMethod]] */
	trainAll() {
		this.currentTrainingDataPoint = -1;
		this.stepsCurrent++;
		if (this.state.drawArrows)
			this.lastWeights = [
				{
					dataPoint: null,
					weights: this.net.connections.map(c => c.weight)
				}
			];
		const steps = this.trainingMethod.trainAll(this.net, this.state.data);
		if (this.state.drawArrows) this.lastWeights.push(...steps);
	}

	/** handle Train All button press */
	trainAllButton() {
		this.stop();
		for (var i = 0; i < this.state.iterationsPerClick; i++) this.trainAll();
		this.stepsWanted = this.stepsCurrent;
		this.onFrame(true);
	}

	/** handle Train Single button press */
	trainNextButton() {
		this.stop();
		this.trainNext();
		this.stepsWanted = this.stepsCurrent;
		this.onFrame(true);
	}

	/** -1 when not training single data points, otherwise index into [[Configuration#data]] */
	private _currentTrainingDataPoint = -1;
	get currentTrainingDataPoint() {
		return this._currentTrainingDataPoint;
	}
	set currentTrainingDataPoint(val) {
		if (val != this._currentTrainingDataPoint) {
			if (val >= this.state.data.length) {
				val -= this.state.data.length;
			}
			this._currentTrainingDataPoint = val;
			this.table.hot.render();
			if (this._currentTrainingDataPoint >= 0)
				this.net.setInputsAndCalculate(
					this.state.data[this._currentTrainingDataPoint].input
				);
			this.netgraph.drawGraph();
		}
	}
	/** train the next single data point */
	trainNext() {
		this.currentTrainingDataPoint++;
		if (this.state.drawArrows)
			this.lastWeights = [
				{
					dataPoint: null,
					weights: this.net.connections.map(c => c.weight)
				}
			];
		this.stepsCurrent++;

		const newWeights = this.trainingMethod.trainSingle(
			this.net,
			this.state.data[this.currentTrainingDataPoint]
		);
		if (this.state.drawArrows) this.lastWeights.push(newWeights);
	}

	/** cache for all the steps that the [[NetGraph]] will go through for a single forward pass step */
	forwardPassEles: NetGraphUpdate[] = [];
	/** do a single forward pass step, start the stepthrough if not already running */
	forwardPassStep() {
		if (!this.netgraph.currentlyDisplayingForwardPass) {
			this.forwardPassEles = [];
			this.currentTrainingDataPoint = -1;
		}
		this.stop();
		if (this.forwardPassEles.length > 0) {
			this.netgraph.applyUpdate(this.forwardPassEles.shift());
		} else {
			if (this.currentTrainingDataPoint < this.state.data.length - 1) {
				// start next
				this.lrVis.leftVis.setMode(0);
				this.currentTrainingDataPoint++;
				this.forwardPassEles = this.netgraph.forwardPass(
					this.state.data[this.currentTrainingDataPoint]
				);
				this.netgraph.applyUpdate(this.forwardPassEles.shift());
				// redraw for highlighted data point
				this.netviz.onFrame();
			} else {
				// end
				this.currentTrainingDataPoint = -1;
				// this clears the forward pass step
				this.netgraph.onFrame(0);
				// clear highlighted data point
				this.netviz.onFrame();
			}
		}
	}

	/** handle animation frame */
	onFrame(forceDraw: boolean) {
		this.frameNum++;
		this.calculateAverageError();
		this.lrVis.onFrame(forceDraw ? 0 : this.frameNum);
		this.updateStatusLine();
	}

	/** begin animation */
	run() {
		if (this.running) return;
		this.running = true;
		this.lrVis.setState({ running: true });
		this.lastTimestamp = performance.now();
		requestAnimationFrame(this.aniFrameCallback);
	}

	/** stop animation */
	stop() {
		clearTimeout(this.restartTimeout);
		this.restartTimeout = -1;
		this.running = false;
		this.lrVis.setState({ running: false });
		cancelAnimationFrame(this.runningId);
	}

	/** stop animation and reset the network */
	reset() {
		this.stop();
		this.initializeNet();
		this.onFrame(true);
	}

	/** calculate the average error ([[#averageError]]) */
	calculateAverageError() {
		this.averageError = 0;
		for (const val of this.state.data) {
			this.net.setInputsAndCalculate(val.input);
			this.averageError += this.net.getLoss(val.output);
		}
		this.averageError /= this.state.data.length;
		this.errorHistory.push([this.stepsCurrent, this.averageError]);
	}

	/** update the information in the status line ([[#averageError]] / correct count and [[#stepNum]]) */
	updateStatusLine() {
		let correct = 0;
		if (this.state.outputLayer.neuronCount === 1) {
			for (var val of this.state.data) {
				const res = this.net.getOutput(val.input);
				if (+(res[0] > 0.5) == val.output[0]) correct++;
			}
			this.lrVis.setState({
				correct: `Correct: ${correct}/${this.state.data.length}`
			});
		} else {
			this.lrVis.setState({
				correct: `Error: ${this.averageError.toFixed(2)}`
			});
		}
		this.lrVis.setState({ stepNum: this.stepsCurrent });

		if (correct == this.state.data.length) {
			if (
				this.state.autoRestart &&
				this.running &&
				this.restartTimeout == -1
			) {
				this.restartTimeout = setTimeout(() => {
					this.stop();
					this.restartTimeout = -1;
					setTimeout(() => {
						this.reset();
						this.run();
					}, 100);
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
	/** do a single step in the animation, called by [[window.requestAnimationFrame]] */
	animationStep(timestamp: number) {
		let delta = timestamp - this.lastTimestamp;
		this.lastTimestamp = timestamp;
		if (delta > 1000 / 5) {
			console.warn(`only ${(1000 / delta).toFixed(1)} fps`);
			delta = 1000 / 5;
		}
		this.stepsWanted += (delta / 1000) * this.state.stepsPerSecond;
		while (this.stepsCurrent < this.stepsWanted) {
			if (
				this.state.animationTrainSinglePoints &&
				this.trainingMethod.trainSingle
			)
				this.trainNext();
			else this.trainAll();
		}
		this.onFrame(false);
		if (this.running)
			this.runningId = requestAnimationFrame(this.aniFrameCallback);
	}

	/** called by React when configuration ([[#state]]) is about to change */
	componentWillUpdate(nextProps: any, newConfig: Configuration) {
		if (
			this.state.hiddenLayers.length !== newConfig.hiddenLayers.length &&
			newConfig.custom
		) {
			if (this.state.custom /* && !forceNeuronRename*/) return;
			const inN = newConfig.inputLayer.neuronCount;
			const outN = newConfig.outputLayer.neuronCount;
			newConfig.name = "Custom Network";
			newConfig.inputLayer = {
				names: Util.makeArray(inN, i => `in${i + 1}`),
				neuronCount: inN
			};
			newConfig.outputLayer = {
				names: Util.makeArray(outN, i => `out${i + 1}`),
				activation: newConfig.outputLayer.activation,
				neuronCount: outN
			};
		}
	}

	/** called by React after the configuration changed; updates the gui components not handled by react accordingly */
	componentDidUpdate(prevProps: any, oldConfig: Configuration) {
		const co = oldConfig,
			cn = this.state;
		if (!cn.autoRestart) clearTimeout(this.restartTimeout);
		const layerDifferent = (l1: any, l2: any) =>
			l1.activation !== l2.activation ||
			l1.neuronCount !== l2.neuronCount ||
			(l1.names &&
				l1.names.some(
					(name: string, i: number) => l2.names[i] !== name
				));
		if (
			cn.hiddenLayers.length !== co.hiddenLayers.length ||
			layerDifferent(cn.inputLayer, co.inputLayer) ||
			layerDifferent(cn.outputLayer, co.outputLayer) ||
			cn.hiddenLayers.some((layer, i) =>
				layerDifferent(layer, co.hiddenLayers[i])
			) ||
			(cn.weights &&
				(!co.weights ||
					cn.weights.some((weight, i) => co.weights[i] !== weight)))
		) {
			this.initializeNet();
		}
		if (!cn.custom)
			history.replaceState({}, "", "?" + $.param({ preset: cn.name }));
		if (this.net) {
			if (co.bias != cn.bias) {
				this.netgraph.onNetworkLoaded(this.net);
			}
			if (co.type !== cn.type) {
				// gui layout may change, trigger resize
				window.dispatchEvent(new Event("resize"));
			}
			if (co.trainingMethod != cn.trainingMethod)
				delete (this.net as any).tmpStore;
			this.net.learnRate = cn.learningRate;
			this.onFrame(false);
		}
	}
	/** called once after the GUI has been added to the DOM */
	componentDidMount() {
		this.initializeNet();
		this.onFrame(true);
		if (this.props.autoRun) this.run();
	}

	/** parse the [[ConfigurationGui]] contents into [[#state]] */
	loadConfig() {
		// from gui
		const config = $.extend(true, {}, this.state);
		for (const conf in config) {
			const ele = document.getElementById(conf) as HTMLInputElement;
			if (!ele) continue;
			if (ele.type == "checkbox") config[conf] = ele.checked;
			else if (typeof config[conf] === "number")
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

	/**
	 * serialize the current configuration into the url (query string)
	 * @param exportWeights 0 = no weights, 1 = current weights, 2 = start weights
	 */
	serializeToUrl(exportWeights = 0) {
		const url =
			location.protocol + "//" + location.host + location.pathname + "?";
		console.log("serializing to url");
		const params: any = {};
		if (this.state.custom || exportWeights > 0) {
			params.config = Util.cloneConfig(this.state);
		} else {
			params.preset = this.state.name;
		}
		if (exportWeights === 1)
			params.config.weights = this.net.connections.map(c => c.weight);
		if (exportWeights === 2) params.config.weights = this.net.startWeights;

		if (params.config)
			params.config = LZString.compressToEncodedURIComponent(
				JSON.stringify(params.config)
			);
		return url + $.param(params);
	}
	/**
	 * parse the configuration from the url created by [[#serializeToUrl]]
	 */
	deserializeFromUrl(): Configuration {
		const urlParams = Util.parseUrlParameters();
		const preset = urlParams["preset"],
			config = urlParams["config"];
		if (preset && Presets.exists(preset)) return Presets.get(preset);
		else if (config) {
			console.log(
				JSON.parse(LZString.decompressFromEncodedURIComponent(config))
			);
			return JSON.parse(
				LZString.decompressFromEncodedURIComponent(config)
			);
		} else return Presets.get("Binary Classifier for XOR");
	}
	/**
	 * (this should be default in newer React versions?)
	 */
	shouldComponentUpdate() {
		return true;
	}

	/**
	 * called by React to render the GUI
	 */
	render() {
		const pageTitle =
			this.state.type === "perceptron"
				? "Perceptron demo"
				: "Neural Network demo";
		const presetName = this.state.custom
			? " Custom Network"
			: " Preset: " + this.state.name;
		document.title = `${pageTitle} — ${presetName}`;
		return (
			<div>
				<div className="container">
					<div className="page-header">
						<div
							className="btn-toolbar pull-right dropdown"
							style={{ marginTop: "5px" }}
						>
							<button
								className="btn btn-info dropdown-toggle"
								data-toggle="dropdown"
							>
								{"Load Preset "}
								<span className="caret" />
							</button>
							<ul className="dropdown-menu">
								<li className="dropdown-header">
									Neural Network
								</li>
								{Presets.getNames().map(name => {
									const ele = (
										<li key={name}>
											<a
												onClick={e =>
													this.setState(
														Presets.get(name)
													)
												}
											>
												{name}
											</a>
										</li>
									);
									if (name === "Rosenblatt Perceptron")
										return [
											<li className="divider" />,
											<li className="dropdown-header">
												Perceptron
											</li>,
											ele
										];
									else return ele;
								})}
							</ul>
						</div>
						<h1>
							{pageTitle}
							<small>{presetName}</small>
						</h1>
					</div>
					<LRVis
						sim={this}
						ref={(e: LRVis) => (this.lrVis = e)}
						leftVis={[
							this.netgraph,
							this.errorGraph,
							this.weightsGraph
						]}
						rightVis={[this.netviz, this.table]}
					/>
					<div className="panel panel-default">
						<div className="panel-heading">
							<h3 className="panel-title">
								<a
									data-toggle="collapse"
									data-target=".panel-body"
								>
									Configuration
								</a>
							</h3>
						</div>
						<div className="panel-body collapse in">
							<ConfigurationGui {...this.state} />
						</div>
					</div>
					<footer className="small">
						<a href="https://github.com/phiresky/neural-network-demo/">
							Source on GitHub
						</a>
					</footer>
				</div>
				<ExportModal
					sim={this}
					ref={(e: ExportModal) => (this.exportModal = e)}
				/>
			</div>
		);
	}
}

(window as any).Simulation = Simulation;
