//class NumberInput extends React.Component<{name:string, min:number, max:number, , {}
class NeuronLayer extends React.Component<{
		layer: LayerConfig,
		name: string, 
		activationChanged: (activation: string) => void,
		countChanged: (delta: number) => void
	}, {}> {
	render() {
		return <div>{this.props.name} layer:
			<span className="neuronCount">{this.props.layer.neuronCount}</span>&nbsp;neurons
			<button className="btn btn-xs btn-default" onClick={() => this.props.countChanged(1)}>+</button>
			<button className="btn btn-xs btn-default" onClick={() => this.props.countChanged(-1)}>-</button>
			<select className="btn btn-xs btn-default activation"
					onChange={(e) => this.props.activationChanged((e.target as HTMLInputElement).value)} 
					value={this.props.layer.activation}>
				<option>sigmoid</option>
				<option>tanh</option>
				<option>linear</option>
				<option>relu</option>
			</select>
		</div>
	}
}

class ConfigurationGui extends React.Component<Configuration, {}> {
	constructor(props:any) {
		super(props);
	}
	loadConfig() {
		sim.loadConfig();
	}
	addLayer() {
		sim.config.hiddenLayers.unshift({ activation: 'sigmoid', neuronCount: 2 });
		sim.setIsCustom();
		sim.initializeNet();
		sim.renderConfigGui();
	}
	removeLayer() {
		if (sim.config.hiddenLayers.length == 0) return;
		sim.config.hiddenLayers.shift();
		sim.setIsCustom();
		sim.initializeNet();
		sim.renderConfigGui();
	}
	activationChanged(i:number, a:string) {
		if(i == this.props.hiddenLayers.length)
			sim.config.outputLayer.activation = a;
		else sim.config.hiddenLayers[i].activation = a;
		sim.setIsCustom();
		sim.initializeNet();
		sim.renderConfigGui();
	}
	countChanged(i:number, inc:number) {
		let targetLayer:(LayerConfig|InputLayerConfig) = sim.config.inputLayer;
		if(i === this.props.hiddenLayers.length) {
			targetLayer = sim.config.outputLayer;
			if(targetLayer.neuronCount >= 10) return;
		} else if (i >= 0) {
			targetLayer = sim.config.hiddenLayers[i];
		}
		const newval = targetLayer.neuronCount + inc;
		if (newval < 1) return;
		targetLayer.neuronCount = newval;
		sim.config.data = [];
		sim.setIsCustom(true);
		sim.initializeNet();
		sim.renderConfigGui();
	}
	render() {
		const conf = this.props;
		return <div id="configuration" className="form-horizontal">
				<div className="col-sm-6">
					<h4>Display</h4>
					<div className="form-group">
						<label htmlFor="iterationsPerClick" className="col-sm-6 control-label">Iterations per click on 'Step'</label>
						<div className="col-sm-6">
							<input className="form-control" type="number" min={0} max={10000} id="iterationsPerClick" value={""+conf.iterationsPerClick} onChange={this.loadConfig.bind(this)} />
						</div>
					</div>
					<div className="form-group">
						<label htmlFor="stepsPerFrame" className="col-sm-6 control-label">Steps per Frame</label>
						<div className="col-sm-6">
							<input className="form-control" type="number" min={1} max={1000} id="stepsPerFrame" value={""+conf.stepsPerFrame} onChange={this.loadConfig.bind(this)} />
						</div>
					</div>
					<div className="form-group">
						<label htmlFor="autoRestart" className="col-sm-6 control-label">When correct, restart after 5 seconds</label>
						<div className="col-sm-6 form-control-static">
							<input type="checkbox" id="autoRestart" checked={conf.autoRestart} onChange={this.loadConfig.bind(this)} />
						</div>
					</div>
					<div className="form-group">
						<label htmlFor="showGradient" className="col-sm-6 control-label">Show class propabilities as gradient</label>
						<div className="col-sm-6 form-control-static">
							<input type="checkbox" checked={conf.showGradient} id="showGradient" onChange={() => {this.loadConfig();sim.onFrame(false);}} />
						</div>
					</div>
					<button className="btn btn-default" data-toggle="modal" data-target="#exportModal">Import / Export</button>
				</div>
				<div className="col-sm-6">
					<h4>Net</h4>
					<div className="form-group">
						<label htmlFor="learningRate" className="col-sm-6 control-label">Learning Rate</label>
						<div className="col-sm-6 form-control-static">
							<span id="learningRateVal" style={{marginRight: '1em'}}>{conf.learningRate}</span>
							<input type="range" min={0.01} max={1} step={0.01} id="learningRate" value={""+conf.learningRate} onChange={this.loadConfig.bind(this)} />
						</div>
					</div>
					<div className="form-group">
						<label htmlFor="bias" className="col-sm-6 control-label">Show bias input</label>
						<div className="col-sm-6 form-control-static">
							<input type="checkbox" checked={conf.bias} id="bias" onChange={() => {this.loadConfig(); sim.initializeNet()}} />
						</div>
					</div>
					<div id="layerCountModifier">
						<span id="layerCount">{conf.hiddenLayers.length + 2}</span>&nbsp;layers
						<button className="btn btn-xs btn-default" onClick={()=>this.addLayer()}>+</button>
						<button className="btn btn-xs btn-default" onClick={()=>this.removeLayer()}>-</button>
					</div>
					<div id="layersModify">
						<div id="inputLayerModify">Input layer:
							<span className="neuronCount">{conf.inputLayer.neuronCount}</span>&nbsp;neurons
							<button className="btn btn-xs btn-default">+</button>
							<button className="btn btn-xs btn-default">-</button>
						</div>
						<div id="hiddenLayersModify">
						{conf.hiddenLayers.map((layer,i) => 
							<NeuronLayer layer={layer} name="Hidden"
								activationChanged={a => this.activationChanged(i, a)}
								countChanged={c => this.countChanged(i, c)}
							/>
						)}
						</div>
						<div id="outputLayerModify">
							<NeuronLayer layer={conf.outputLayer} name="Output"
								activationChanged={a => this.activationChanged(conf.hiddenLayers.length, a)}
								countChanged={c => this.countChanged(conf.hiddenLayers.length, c)}
							/>
						</div>
					</div>
				</div>
			</div>;
	}
}