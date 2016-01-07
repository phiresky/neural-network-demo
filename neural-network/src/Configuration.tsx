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

class BSCheckbox extends React.Component<{conf: Configuration, label: string, id:string}, {}> {
	render() {
		return (
			<BSFormGroup label={this.props.label} id={this.props.id} isStatic>
				<input type="checkbox" checked={this.props.conf[this.props.id]} id={this.props.id} onChange={() => sim.loadConfig()} />
			</BSFormGroup>
		);
	}
}

class ConfigurationGui extends React.Component<Configuration, {}> {
	render() {
		const conf = this.props;
		const loadConfig = () => sim.loadConfig();
		return <div className="form-horizontal">
				<div className="col-sm-6">
					<h4>Display</h4>
					<BSFormGroup label="Iterations per click on 'Train'" id="iterationsPerClick">
						<input className="form-control" type="number" min={0} max={10000} id="iterationsPerClick" value={""+conf.iterationsPerClick} onChange={loadConfig} />
					</BSFormGroup>
					<BSFormGroup label="Steps per Second" id="stepsPerSecond">
						<input className="form-control" type="number" min={0.1} max={1000} id="stepsPerSecond" value={""+conf.stepsPerSecond} onChange={loadConfig} />
					</BSFormGroup>
					<BSCheckbox label="When correct, restart after 5 seconds" id="autoRestart" conf={conf} />
					{conf.type !== "perceptron"?
						<BSCheckbox label="Show class propabilities as gradient" id="showGradient" conf={conf}/>
					:""}
					<BSCheckbox label="Show bias input" id="bias" conf={conf} />
					<button className="btn btn-default" data-toggle="modal" data-target="#exportModal">Import / Export</button>
				</div>
				<div className="col-sm-6">
					<h4>{conf.type==="perceptron"?"Perceptron":"Net"}</h4>
					<BSFormGroup id="learningRate" label="Learning Rate" isStatic>
						<span id="learningRateVal" style={{marginRight: '1em'}}>{conf.learningRate.toFixed(3)}</span>
						<input type="range" min={0.005} max={1} step={0.005} id="learningRate" value={Util.logScale(conf.learningRate)+""} onChange={loadConfig} />
					</BSFormGroup>
					{conf.type === "perceptron"?
						<div>
							<BSCheckbox label="Animate online training" id="animationTrainSinglePoints" conf={conf} />
							<BSCheckbox label="Draw Arrows" id="drawArrows" conf={conf} />
							<BSCheckbox label="Draw coordinate system" id="drawCoordinateSystem" conf={conf} />
						</div>
					:
						<div>
							<BSCheckbox id="batchTraining" label="Batch training" conf={conf} />
							<NeuronGui {...this.props} />
						</div>
					}
				</div>
			</div>;
	}
}

class NeuronLayer extends React.Component<{
		layer: {neuronCount: number, activation?: string},
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
			{p.layer.activation?
				<select className="btn btn-xs btn-default activation"
						onChange={(e) => p.activationChanged((e.target as HTMLInputElement).value)} 
						value={p.layer.activation}>
					{Object.keys(Net.NonLinearities).map(name => <option key={name} value={name}>{name}</option>)}
				</select>
			:""}
		</div>
	}
}

class NeuronGui extends React.Component<Configuration, {}> {
	addLayer() {
		const hiddenLayers = this.props.hiddenLayers.slice();
		hiddenLayers.unshift({ activation: 'sigmoid', neuronCount: 2 });
		sim.setState({hiddenLayers, custom:true});
	}
	removeLayer() {
		if (this.props.hiddenLayers.length == 0) return;
		const hiddenLayers = this.props.hiddenLayers.slice();
		hiddenLayers.shift();
		sim.setState({hiddenLayers, custom:true});
	}
	activationChanged(i:number, a:string) {
		const newConf = Util.cloneConfig(this.props);
		if(i == this.props.hiddenLayers.length)
			newConf.outputLayer.activation = a;
		else newConf.hiddenLayers[i].activation = a;
		newConf.custom = true;
		sim.setState(newConf);
	}
	countChanged(i:number, inc:number) {
		const newState = Util.cloneConfig(this.props);
		let targetLayer:(LayerConfig|InputLayerConfig);
		let ioDimensionChanged = true;
		if(i === this.props.hiddenLayers.length) {
			// is output layer
			targetLayer = newState.outputLayer;
			if(targetLayer.neuronCount >= 10) return;
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
		if(ioDimensionChanged) newState.data = [];
		newState.custom = true;
		sim.setState(newState);
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
			<NeuronLayer key={-1} layer={conf.inputLayer} name="Input" {...neuronListeners(-1)} />
			{conf.hiddenLayers.map((layer,i) => 
				<NeuronLayer key={i} layer={layer} name="Hidden" {...neuronListeners(i)} />
			)}
			<NeuronLayer key={-2} layer={conf.outputLayer} name="Output" {...neuronListeners(conf.hiddenLayers.length)} />
		</div>;
	}
}