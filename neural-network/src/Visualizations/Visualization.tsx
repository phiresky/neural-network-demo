interface Visualization {
	container: JQuery;
	actions: (string|{name:string, color:string})[];
	onView: (previouslyHidden: boolean, mode: int) => void;
	onNetworkLoaded: (net:Net.NeuralNet) => void;
	onHide: () => void;
	onFrame: (framenum:int) => void;
}

interface VisualizationConstructor {
	new (sim:Simulation): Visualization;
}
interface LRVisState {
	running: boolean, leftVisBody: Visualization, rightVisBody: Visualization,
	correct: string,
	stepNum: number
}
class LRVis extends React.Component<{sim:Simulation, ref:any}, LRVisState> {
	leftVis: TabSwitcher;
	rightVis: TabSwitcher;
	constructor(props:{sim:Simulation, ref:any}) {
		super(props);
		this.state = {
			running: false,
			leftVisBody: props.sim.netgraph,
			rightVisBody: props.sim.netviz,
			correct: "",
			stepNum: 0
		};
	}
	render() {
		const sim = this.props.sim;
		return <div>
				<div className="row">
					<div className="col-sm-6">
						<TabSwitcher ref={(c:TabSwitcher) => this.leftVis = c}
							things={[sim.netgraph, sim.errorGraph, sim.weightsGraph]}
							onChangeVisualization={(vis,aft) => this.setState({leftVisBody: vis}, aft)}
						/>
					</div>
					<div className="col-sm-6">
						<TabSwitcher ref={(c:TabSwitcher) => this.rightVis = c}
							things={[sim.netviz, sim.table]}
							onChangeVisualization={(vis,aft) => this.setState({rightVisBody: vis}, aft)}
						/>
					</div>
				</div>
				<div className="row">
					<div className="col-sm-6">
						<div id="leftVisBody" className="visbody" />
						<div className="h3">
							<button className={this.state.running?"btn btn-danger":"btn btn-primary"} onClick={sim.runtoggle.bind(sim)}>{this.state.running?"Stop":"Animate"}</button>&nbsp;
							<button className="btn btn-warning" onClick={sim.reset.bind(sim)}>Reset</button>&nbsp;
							<button className="btn btn-default" onClick={sim.iterations.bind(sim)}>Train</button>&nbsp;
							<button className="btn btn-default" onClick={sim.forwardPassStep.bind(sim)}>Forward Pass Step</button>
							<div className="btn-group pull-right">
								<button className="btn btn-default dropdown-toggle" data-toggle="dropdown">{"Load "}
									<span className="caret" /></button>
								<ul className="dropdown-menu">
									{Presets.getNames().map(name =>
										<li key={name}><a onClick={e => sim.setState(Presets.get((e.target as Element).textContent))}>{name}</a></li>)
									}
								</ul>
							</div>
						</div>
						<hr />
					</div>
					<div className="col-sm-6">
						<div id="rightVisBody" className="visbody" />
						<div id="status">
							<h2>
								{this.state.correct} — Iteration:&nbsp;{this.state.stepNum}
							</h2>
						</div>
						<hr />
					</div>
				</div>
			</div>
	}
	componentDidMount() {
		$("#leftVisBody").append(this.props.sim.netgraph.container);
		$("#rightVisBody").append(this.props.sim.netviz.container);
	}
	componentDidUpdate(prevProps:any, prevState:LRVisState) {
		if(prevState.leftVisBody !== this.state.leftVisBody) {
			$("#leftVisBody").children().detach();
			$("#leftVisBody").append(this.state.leftVisBody.container);
		}
		if(prevState.rightVisBody !== this.state.rightVisBody) {
			$("#rightVisBody").children().detach();
			$("#rightVisBody").append(this.state.rightVisBody.container);
		}
	}
}
interface _Mode { thing: int, action: int, text: string, color: string }
interface TSProps {things: Visualization[], onChangeVisualization: (v:Visualization, aft:()=>void) => void, ref?:any}
class TabSwitcher extends React.Component<TSProps, {modes: _Mode[], currentMode: number}> {
	constructor(props: TSProps) {
		super(props);
		this.state = {
			modes:this.createButtonsAndActions(),
			currentMode: 0
		};
	}
	render() {
		const isDark = (color: string) => Util.parseColor(color).reduce((a,b)=>a+b)/3 < 127;
		return <div><ul className="nav nav-pills">
			{this.state.modes.map((mode, i) =>
				<li key={i} className={this.state.currentMode === i?"custom-active":""}>
					<a style={mode.color?{backgroundColor: mode.color, color: isDark(mode.color)?"white":"black"}:{}}
							onClick={e => this.setMode(i)}>
						{mode.text}
					</a>
				</li>
			)}
		</ul></div>;
	}
	createButtonsAndActions() {
		const modes:_Mode[] = [];
		this.props.things.forEach((thing, thingid) =>
			thing.actions.forEach((button, buttonid) => {
				let text = "", color = "";
				if(typeof button === 'string') {
					text = button;
				} else {
					text = button.name;
					color = button.color;
				}
				modes.push({ thing: thingid, action: buttonid, text: text, color: color});
			})
		);
		return modes;
	}
	setMode(mode:int, force = false) {
		if (!force && mode == this.state.currentMode) return;
		const action = this.state.modes[mode];
		const lastAction = this.state.modes[this.state.currentMode];
		this.setState({currentMode: mode});
		const currentVisualization = this.props.things[action.thing];
		if (force || !lastAction || action.thing != lastAction.thing) {
			if(lastAction) this.props.things[lastAction.thing].onHide();
			this.props.onChangeVisualization(currentVisualization, () => 
				currentVisualization.onView(true, action.action)
			);
		} else if (action.action !== lastAction.action) {
			currentVisualization.onView(false, action.action);
		}
	}
	onNetworkLoaded(net: Net.NeuralNet) {
		//todo: ugly hack
		const beforeActions = JSON.stringify(this.props.things.map(t => t.actions));
		this.props.things.forEach(thing => thing.onNetworkLoaded(net));
		const afterActions = JSON.stringify(this.props.things.map(t => t.actions));
		if(beforeActions !== afterActions)
			this.setState({
				modes:this.createButtonsAndActions(),
				currentMode:0
			}, () => this.setMode(0, true));
	}
}