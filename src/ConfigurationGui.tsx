import * as React from 'react';
import Simulation from "./Simulation";
import * as Util from "./Util";
import { LayerConfig, InputLayerConfig } from "./Presets";
import Net from "./Net";
import { Configuration, TrainingData } from "./Configuration";

/** small wrapper for bootstrap form groups */
class BSFormGroup extends React.Component<{
	label: string, children?: any, id: string, isStatic?: boolean
}, {}> {
	render() {
		return <div className="form-group">
			<label htmlFor={this.props.id} className="col-sm-6 control-label">{this.props.label}</label>
			<div className={"col-sm-6 " + (this.props.isStatic ? "form-control-static" : "")}>{this.props.children}</div>
		</div>;
	}
}
/** small wrapper for bootstrap form checkboxes */
class BSCheckbox extends React.Component<{ conf: Configuration, label: string, id: string, onChange: () => void }, {}> {
	render() {
		return (
			<BSFormGroup label={this.props.label} id={this.props.id} isStatic>
				<input type="checkbox" checked={this.props.conf[this.props.id]} id={this.props.id} onChange={this.props.onChange} />
			</BSFormGroup>
		);
	}
}

/** GUI for displaying and modifying [[Configuration]] */
export default class ConfigurationGui extends React.Component<Configuration, {}> {
	render() {
		const conf = this.props;
		const loadConfig = () => Simulation.instance.loadConfig();
		const trainingMethods = Net.trainingMethods[conf.type];
		return <div className="form-horizontal">
			<div className="col-sm-6">
				<h4>Display</h4>
				<BSFormGroup label="Iterations per click on 'Train'" id="iterationsPerClick">
					<input className="form-control" type="number" min={0} max={10000} id="iterationsPerClick" value={"" + conf.iterationsPerClick} onChange={loadConfig} />
				</BSFormGroup>
				<BSFormGroup label="Steps per Second" id="stepsPerSecond">
					<input className="form-control" type="number" min={0.1} max={1000} id="stepsPerSecond" value={"" + conf.stepsPerSecond} onChange={loadConfig} />
				</BSFormGroup>
				<BSCheckbox label="When correct, restart after 5 seconds" id="autoRestart" onChange={loadConfig} conf={conf} />
				{conf.type !== "perceptron" ?
					<BSCheckbox label="Show class propabilities as gradient" id="showGradient" onChange={loadConfig} conf={conf} />
					: ""}
				<BSCheckbox label="Show bias input" id="bias" onChange={loadConfig} conf={conf} />
				<BSCheckbox label="Show Train Single button" id="showTrainSingleButton" onChange={loadConfig} conf={conf} />
				<button className="btn btn-default" data-toggle="modal" data-target="#exportModal">Import / Export</button>
			</div>
			<div className="col-sm-6">
				<h4>{conf.type === "perceptron" ? "Perceptron" : "Net"}</h4>
				<BSFormGroup id="learningRate" label="Learning Rate" isStatic>
					<span id="learningRateVal" style={{ marginRight: '1em' }}>{conf.learningRate.toFixed(3)}</span>
					<input type="range" min={0.005} max={1} step={0.005} id="learningRate" value={Util.logScale(conf.learningRate) + ""} onChange={loadConfig} />
				</BSFormGroup>
				<BSFormGroup id="trainingMethod" label="Training Method">
					<select id="trainingMethod" className="btn btn-default"
						onChange={loadConfig}
						value={conf.trainingMethod}>
						{Object.keys(trainingMethods).map(name => <option key={name} value={name}>{name}</option>)}
					</select>
				</BSFormGroup>
				{conf.type === "perceptron" ?
					<div>
						{trainingMethods[conf.trainingMethod].trainSingle ?
							<BSCheckbox label="Animate single data points" id="animationTrainSinglePoints" onChange={loadConfig} conf={conf} />
							: ""
						}
						<BSCheckbox label="Draw Arrows" id="drawArrows" onChange={loadConfig} conf={conf} />
						<BSCheckbox label="Draw coordinate system" id="drawCoordinateSystem" onChange={loadConfig} conf={conf} />
					</div>
					:
					<div>

						<NeuronGui {...this.props} />
					</div>
				}
			</div>
		</div>;
	}
}

/** GUI for displaying the configuration of a single neuron layer */
class NeuronLayer extends React.Component<{
	layer: { neuronCount: number, activation?: string },
	name: string,
	activationChanged: (activation: string) => void,
	countChanged: (delta: number) => void,
	key?: number
}, {}> {
	render() {
		const p = this.props;
		return <div>{p.name} layer: {p.layer.neuronCount} neurons&nbsp;
			<button className="btn btn-xs btn-default" onClick={() => p.countChanged(1)}>+</button>
			<button className="btn btn-xs btn-default" onClick={() => p.countChanged(-1)}>-</button>
			{p.layer.activation ?
				<select className="btn btn-xs btn-default activation"
					onChange={(e) => p.activationChanged(e.currentTarget.value)}
					value={p.layer.activation}>
					{Object.keys(Net.NonLinearities).map(name => <option key={name} value={name}>{name}</option>)}
				</select>
				: ""}
		</div>
	}
}

/** GUI for configuring the neuron layers */
class NeuronGui extends React.Component<Configuration, {}> {
	comicShown = false;
	addLayer() {
		const hiddenLayers = this.props.hiddenLayers.slice();
		hiddenLayers.unshift({ activation: 'sigmoid', neuronCount: 2 });
		if (hiddenLayers.length === 4 && !this.comicShown) {
			window.open("https://i.imgur.com/GOTeNFr.png");
			this.comicShown = true;
		}
		Simulation.instance.setState({ hiddenLayers, custom: true });
	}
	removeLayer() {
		if (this.props.hiddenLayers.length == 0) return;
		const hiddenLayers = this.props.hiddenLayers.slice();
		hiddenLayers.shift();
		Simulation.instance.setState({ hiddenLayers, custom: true });
	}
	activationChanged(i: number, a: string) {
		const newConf = Util.cloneConfig(this.props);
		if (i == this.props.hiddenLayers.length)
			newConf.outputLayer.activation = a;
		else newConf.hiddenLayers[i].activation = a;
		newConf.custom = true;
		Simulation.instance.setState(newConf);
	}
	countChanged(i: number, inc: number) {
		const newState = Util.cloneConfig(this.props);
		let targetLayer: (LayerConfig | InputLayerConfig);
		let ioDimensionChanged = true;
		if (i === this.props.hiddenLayers.length) {
			// is output layer
			targetLayer = newState.outputLayer;
			if (targetLayer.neuronCount >= 10) return;
		} else if (i >= 0) {
			// is hidden layer
			targetLayer = newState.hiddenLayers[i];
			ioDimensionChanged = false;
		} else {
			// < 0: is input layer
			targetLayer = newState.inputLayer;
		}
		const newval = targetLayer.neuronCount + inc;
		if (newval < 1) return;
		targetLayer.neuronCount = newval;
		if (ioDimensionChanged) newState.data = [];
		newState.custom = true;
		Simulation.instance.setState(newState);
	}
	render() {
		const conf = this.props;
		const neuronListeners = (i: number) => ({
			activationChanged: (a: string) => this.activationChanged(i, a),
			countChanged: (c: number) => this.countChanged(i, c)
		})
		return <div>
			{(conf.hiddenLayers.length + 2) + " layers "}
			<button className="btn btn-xs btn-default" onClick={() => this.addLayer()}>+</button>
			<button className="btn btn-xs btn-default" onClick={() => this.removeLayer()}>-</button>
			<NeuronLayer key={-1} layer={conf.inputLayer} name="Input" {...neuronListeners(-1) } />
			{conf.hiddenLayers.map((layer, i) =>
				<NeuronLayer key={i} layer={layer} name="Hidden" {...neuronListeners(i) } />
			)}
			<NeuronLayer key={-2} layer={conf.outputLayer} name="Output" {...neuronListeners(conf.hiddenLayers.length) } />
		</div>;
	}
}