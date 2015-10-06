//class NumberInput extends React.Component<{name:string, min:number, max:number, , {}
class BSFormGroup extends React.Component<{
		label: string, children?: any, id:string, isStatic?:boolean
	}, {}> {
	render() {
		return <div className="form-group">
			<label htmlFor={this.props.id} className="col-sm-6 control-label">{this.props.label}</label>
			<div className={"col-sm-6 "+(this.props.isStatic?"form-control-static":"")}>{this.props.children}</div>
		</div>;
	}
}

class ConfigurationGui extends React.Component<Configuration, {}> {
	render() {
		const conf = this.props;
		const loadConfig = () => sim.loadConfig();
		return <div id="configuration" className="form-horizontal">
				<div className="col-sm-6">
					<h4>Display</h4>
					<BSFormGroup label="Iterations per click on 'Step'" id="iterationsPerClick">
						<input className="form-control" type="number" min={0} max={10000} id="iterationsPerClick" value={""+conf.iterationsPerClick} onChange={loadConfig} />
					</BSFormGroup>
					<BSFormGroup label="Steps per Frame" id="stepsPerFrame">
						<input className="form-control" type="number" min={1} max={1000} id="stepsPerFrame" value={""+conf.stepsPerFrame} onChange={loadConfig} />
					</BSFormGroup>
					<BSFormGroup label="When correct, restart after 5 seconds" id="autoRestart" isStatic>
						<input type="checkbox" id="autoRestart" checked={conf.autoRestart} onChange={loadConfig} />
					</BSFormGroup>
					<BSFormGroup label="Show class propabilities as gradient" id="showGradient" isStatic>
						<input type="checkbox" checked={conf.showGradient} id="showGradient" onChange={() => {loadConfig();sim.onFrame(false);}} />
					</BSFormGroup>
					<button className="btn btn-default" data-toggle="modal" data-target="#exportModal">Import / Export</button>
				</div>
				<div className="col-sm-6">
					<h4>Net</h4>
					<BSFormGroup id="learningRate" label="Learning Rate" isStatic>
						<span id="learningRateVal" style={{marginRight: '1em'}}>{conf.learningRate}</span>
						<input type="range" min={0.01} max={1} step={0.01} id="learningRate" value={""+conf.learningRate} onChange={loadConfig} />
					</BSFormGroup>
					<BSFormGroup label="Show bias input" id="bias" isStatic>
						<input type="checkbox" checked={conf.bias} id="bias" onChange={() => {loadConfig(); sim.initializeNet()}} />
					</BSFormGroup>
					<NeuronGui {...this.props} />
				</div>
			</div>;
	}
}

class NeuronLayer extends React.Component<{
		layer: {neuronCount: number, activation?: string},
		name: string,
		activationChanged: (activation: string) => void,
		countChanged: (delta: number) => void
	}, {}> {
	render() {
		const p = this.props;
		return <div>{p.name} layer: {p.layer.neuronCount} neurons&nbsp;
			<button className="btn btn-xs btn-default" onClick={() => p.countChanged(1)}>+</button>
			<button className="btn btn-xs btn-default" onClick={() => p.countChanged(-1)}>-</button>
			{p.layer.activation?
				<select className="btn btn-xs btn-default activation"
						onChange={(e) => p.activationChanged((e.target as HTMLInputElement).value)} 
						value={p.layer.activation}>
					<option>sigmoid</option>
					<option>tanh</option>
					<option>linear</option>
					<option>relu</option>
				</select>
			:""}
		</div>
	}
}

class NeuronGui extends React.Component<Configuration, {}> {
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
		const neuronListeners = (i:number) => ({
			activationChanged: (a:string) => this.activationChanged(i, a),
			countChanged: (c:number) => this.countChanged(i, c)
		})
		return <div>
			{(conf.hiddenLayers.length + 2) + " layers "}
			<button className="btn btn-xs btn-default" onClick={()=>this.addLayer()}>+</button>
			<button className="btn btn-xs btn-default" onClick={()=>this.removeLayer()}>-</button>
			<NeuronLayer layer={conf.inputLayer} name="Input" {...neuronListeners(-1)} />
			{conf.hiddenLayers.map((layer,i) => 
				<NeuronLayer layer={layer} name="Hidden" {...neuronListeners(i)} />
			)}
			<NeuronLayer layer={conf.outputLayer} name="Output" {...neuronListeners(conf.hiddenLayers.length)} />
		</div>;
	}
}