/** some GUI component that displays some visualization of information about the neural network */
interface Visualization {
	/** DOM element that contains all the stuff */
	container: JQuery;
	/**
	 * List of possible modes of input for this visualization.
	 * Switching between these is handled externally, see [[TabSwitcher]].
	 */
	actions: (string | { name: string, color: string })[];
	/** called when the action is changed or the visualization becomes visible */
	onView: (previouslyHidden: boolean, mode: int) => void;
	onNetworkLoaded: (net: Net.NeuralNet) => void;
	onHide: () => void;
	onFrame: (framenum: int) => void;
}
interface MultiVisState {
	running: boolean, bodies: Visualization[],
	correct: string,
	stepNum: number
}
/** display multiple Visualizations */
abstract class MultiVisDisplayer<T> extends React.Component<{ sim: Simulation } & T, MultiVisState> {
	bodyDivs: HTMLDivElement[] = [];

	constructor(props: { sim: Simulation } & T) {
		super(props);
		this.state = {
			running: false,
			bodies: [null, null],
			correct: "",
			stepNum: 0
		};
	}
	onFrame(framenum: int) {
		for (const body of this.state.bodies) if (body) body.onFrame(framenum);
	}
	/// hack to prevent overwriting from react because setState is called multiple times before it is done
	bodiesTmp: Visualization[];
	changeBody(i: int, vis: Visualization, aft: () => void) {
		this.bodiesTmp = this.bodiesTmp || this.state.bodies.slice();
		this.bodiesTmp[i] = vis;
		this.setState({ bodies: this.bodiesTmp }, () => { this.bodiesTmp = undefined; aft() });
	}
	componentDidUpdate(prevProps: any, prevState: MultiVisState) {
		for (let i = 0; i < prevState.bodies.length; i++) {
			if (prevState.bodies[i] !== this.state.bodies[i]) {
				$(this.bodyDivs[i]).children().detach();
				$(this.bodyDivs[i]).append(this.state.bodies[i].container);
			}
		}
	}
}
class ControlButtonBar extends React.Component<{ running: boolean, sim: Simulation }, {}>{
	render() {
		const sim = this.props.sim;
		return <div className="h3">
			<button className={this.props.running ? "btn btn-danger" : "btn btn-primary"} onClick={sim.runtoggle.bind(sim) }>{this.props.running ? "Stop" : "Animate"}</button>&nbsp;
			<button className="btn btn-warning" onClick={sim.reset.bind(sim) }>Reset</button>&nbsp;
			<button className="btn btn-default" onClick={sim.trainAllButton.bind(sim) }>{sim.state.type === "perceptron" ? "Train All" : "Train"}</button>&nbsp;
			{(() => {
				if(sim.state.type === "perceptron" && sim.trainingMethod.trainSingle)
					return <button className="btn btn-default" onClick={sim.trainNextButton.bind(sim) }>Train Single</button>;
				if(sim.state.type === "nn")
					return <button className="btn btn-default" onClick={sim.forwardPassStep.bind(sim) }>Forward Pass Step</button>;
				return null;
			})()}
		</div>;
	}
}
class StatusBar extends React.Component<{ correct: string, iteration: int }, {}> {
	render() {
		return <h2>
			{this.props.correct} —&nbsp; Iteration: &nbsp; {this.props.iteration}
		</h2>
	}
}
/** Display two visualizations next to each other with tabbed navigation */
class LRVis extends MultiVisDisplayer<{ leftVis: Visualization[], rightVis: Visualization[] }> {
	leftVis: TabSwitcher;
	rightVis: TabSwitcher;
	constructor(props: { sim: Simulation, leftVis: Visualization[], rightVis: Visualization[] }) {
		super(props);
	}
	render() {
		const sim = this.props.sim;
		const isPerceptron = this.props.sim.state.type === "perceptron";
		const leftSize = isPerceptron ? 4 : 6;
		const rightSize = 12 - leftSize;
		return <div>
			<div className="row">
				<div className={`col-sm-${leftSize}`}>
					<TabSwitcher ref={(c: TabSwitcher) => this.leftVis = c}
						things={this.props.leftVis}
						onChangeVisualization={(vis, aft) => this.changeBody(0, vis, aft) }
						/>
				</div>
				<div className={`col-sm-${rightSize}`}>
					<TabSwitcher ref={(c: TabSwitcher) => this.rightVis = c}
						things={this.props.rightVis}
						onChangeVisualization={(vis, aft) => this.changeBody(1, vis, aft) }
						/>
				</div>
			</div>
			<div className="row">
				<div className={`col-sm-${leftSize}`}>
					<div className="visbody" ref={b => this.bodyDivs[0] = b } />
					<ControlButtonBar running={this.state.running} sim={sim}/>
					<hr />
				</div>
				<div className={`col-sm-${rightSize}`}>
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
/** Mode for TabSwitcher */
interface _Mode { thing: int, action: int, text: string, color: string }
interface TSProps { things: Visualization[], onChangeVisualization: (v: Visualization, aft: () => void) => void, ref?: any }
/** switch between multiple visualizations using a tabbed interface */
class TabSwitcher extends React.Component<TSProps, { modes: _Mode[], currentMode: number }> {
	constructor(props: TSProps) {
		super(props);
		this.state = {
			modes: this.createButtonsAndActions(),
			currentMode: -1
		};
	}
	render() {
		const isDark = (color: string) => Util.parseColor(color).reduce((a, b) => a + b) / 3 < 127;
		return <div><ul className="nav nav-pills">
			{this.state.modes.map((mode, i) =>
				<li key={i} className={this.state.currentMode === i ? "custom-active" : ""}>
					<a style={mode.color ? { backgroundColor: mode.color, color: isDark(mode.color) ? "white" : "black" } : {}}
						onClick={e => this.setMode(i) }>
						{mode.text}
					</a>
				</li>
			) }
		</ul></div>;
	}
	componentDidMount() {
		this.setMode(0, true);
	}
	createButtonsAndActions() {
		const modes: _Mode[] = [];
		this.props.things.forEach((thing, thingid) =>
			thing.actions.forEach((button, buttonid) => {
				let text = "", color = "";
				if (typeof button === 'string') {
					text = button;
				} else {
					text = button.name;
					color = button.color;
				}
				modes.push({ thing: thingid, action: buttonid, text: text, color: color });
			})
		);
		return modes;
	}
	setMode(mode: int, force = false) {
		if (!force && mode == this.state.currentMode) return;
		const action = this.state.modes[mode];
		const lastAction = this.state.modes[this.state.currentMode];
		this.setState({ currentMode: mode });
		const currentVisualization = this.props.things[action.thing];
		if (force || !lastAction || action.thing != lastAction.thing) {
			if (lastAction) this.props.things[lastAction.thing].onHide();
			this.props.onChangeVisualization(currentVisualization, () =>
				currentVisualization.onView(true, action.action)
			);
		} else if (action.action !== lastAction.action) {
			currentVisualization.onView(false, action.action);
		}
	}
	/** network was loaded, check if the children actions have changed and update the tab bar accordingly */
	onNetworkLoaded(net: Net.NeuralNet) {
		//todo: ugly hack
		const beforeActions = JSON.stringify(this.props.things.map(t => t.actions));
		this.props.things.forEach(thing => thing.onNetworkLoaded(net));
		const afterActions = JSON.stringify(this.props.things.map(t => t.actions));
		if (beforeActions !== afterActions)
			this.setState({
				modes: this.createButtonsAndActions(),
				currentMode: 0
			}, () => this.setMode(0, true));
	}
}