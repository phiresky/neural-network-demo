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
	running: boolean, bodies: Visualization[],
	correct: string,
	stepNum: number
}
abstract class MultiVisDisplayer<T> extends React.Component<{sim:Simulation}&T, LRVisState> {
	bodyDivs: HTMLDivElement[] = [];
	
	constructor(props:{sim:Simulation}&T) {
		super(props);
		this.state = {
			running: false,
			bodies: [null, null],
			correct: "",
			stepNum: 0
		};
	}
	onFrame(framenum:int) {
		for(const body of this.state.bodies) if(body) body.onFrame(framenum);
	}
	/// hack to prevent overwriting from react
	bodiesTmp: Visualization[];
	changeBody(i: int, vis: Visualization, aft: () => void) {
		this.bodiesTmp = this.bodiesTmp || this.state.bodies.slice();
		this.bodiesTmp[i] = vis;
		this.setState({bodies: this.bodiesTmp}, () => {this.bodiesTmp = undefined; aft()});
	}
	componentDidUpdate(prevProps:any, prevState:LRVisState) {
		for(let i = 0; i < prevState.bodies.length; i++) {
			if(prevState.bodies[i] !== this.state.bodies[i]) {
				$(this.bodyDivs[i]).children().detach();
				$(this.bodyDivs[i]).append(this.state.bodies[i].container);
			}
		}
	}
}
class ControlButtonBar extends React.Component<{running: boolean, sim:Simulation}, {}>{
	render() {
		const sim = this.props.sim;
		return <div className="h3">
			<button className={this.props.running?"btn btn-danger":"btn btn-primary"} onClick={sim.runtoggle.bind(sim)}>{this.props.running?"Stop":"Animate"}</button>&nbsp;
			<button className="btn btn-warning" onClick={sim.reset.bind(sim)}>Reset</button>&nbsp;
			<button className="btn btn-default" onClick={sim.trainAllButton.bind(sim)}>{sim.state.showTrainNextButton?"Batch Train":"Train"}</button>&nbsp;
			{sim.state.showTrainNextButton?
				<button className="btn btn-default" onClick={sim.trainNextButton.bind(sim)}>Train Next</button>
			:
				<button className="btn btn-default" onClick={sim.forwardPassStep.bind(sim)}>Forward Pass Step</button>
			}
			<div className="btn-group pull-right">
				<button className="btn btn-default dropdown-toggle" data-toggle="dropdown">{"Load "}
					<span className="caret" /></button>
				<ul className="dropdown-menu">
					{Presets.getNames().map(name =>
						<li key={name}><a onClick={e => sim.setState(Presets.get((e.target as Element).textContent))}>{name}</a></li>)
					}
				</ul>
			</div>
		</div>;
	}
}
class StatusBar extends React.Component<{correct: string, iteration: int}, {}> {
	render() {
		return <h2>
			{this.props.correct} — Iteration:&nbsp;{this.props.iteration}
		</h2>
	}
}
class LRVis extends MultiVisDisplayer<{leftVis: Visualization[], rightVis: Visualization[]}> {
	leftVis: TabSwitcher;
	rightVis: TabSwitcher;
	constructor(props:{sim:Simulation, leftVis: Visualization[], rightVis: Visualization[]}) {
		super(props);
	}
	render() {
		const sim = this.props.sim;
		return <div>
				<div className="row">
					<div className="col-sm-6">
						<TabSwitcher ref={(c:TabSwitcher) => this.leftVis = c}
							things={this.props.leftVis}
							onChangeVisualization={(vis,aft) => this.changeBody(0, vis, aft)}
						/>
					</div>
					<div className="col-sm-6">
						<TabSwitcher ref={(c:TabSwitcher) => this.rightVis = c}
							things={this.props.rightVis}
							onChangeVisualization={(vis,aft) => this.changeBody(1, vis, aft)}
						/>
					</div>
				</div>
				<div className="row">
					<div className="col-sm-6">
						<div className="visbody" ref={b => this.bodyDivs[0] = b } />
						<ControlButtonBar running={this.state.running} sim={sim}/>
						<hr />
					</div>
					<div className="col-sm-6">
						<div className="visbody" ref={b => this.bodyDivs[1] = b } />
						<div>
							<StatusBar correct={this.state.correct} iteration={this.state.stepNum} />
						</div>
						<hr />
					</div>
				</div>
			</div>
	}
}

interface _Mode { thing: int, action: int, text: string, color: string }
interface TSProps {things: Visualization[], onChangeVisualization: (v:Visualization, aft:()=>void) => void, ref?:any}
class TabSwitcher extends React.Component<TSProps, {modes: _Mode[], currentMode: number}> {
	constructor(props: TSProps) {
		super(props);
		this.state = {
			modes:this.createButtonsAndActions(),
			currentMode: -1
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
	componentDidMount() {
		this.setMode(0, true);
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