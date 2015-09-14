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

class TabSwitchVisualizationContainer {
	modes: { thing: int, action: int }[] = [];
	currentVisualization: Visualization;
	ul = $("<ul class='nav nav-pills'>");
	body = $("<div class='visbody'>");
	currentMode = -1;
	constructor(public headContainer: JQuery, public bodyContainer: JQuery, 
		public name: string, public things: Visualization[]) {
		this.createButtonsAndActions();
		this.ul.on("click", "a", e => this.setMode($(e.target).parent().index()));
		headContainer.append(this.ul);
		bodyContainer.append(this.body);
	}
	createButtonsAndActions() {
		this.ul.empty();
		this.modes = [];
		this.things.forEach((thing, thingid) =>
			thing.actions.forEach((button, buttonid) => {
				this.modes.push({ thing: thingid, action: buttonid });
				const a = $("<a>")
				if(typeof button === 'string') {
					a.text(button);
				} else {
					a.text(button.name).css("background-color", button.color);
					const dark = Util.parseColor(button.color).reduce((a,b)=>a+b)/3 < 127;
					a.css("color", dark?'white':'black');
				}
				
				const li = $("<li>").append(a);
				if(!button) li.hide();
				this.ul.append(li);
			})
		);
	}
	setMode(mode:int) {
		this.ul.children("li.custom-active").removeClass("custom-active");
		this.ul.children().eq(mode).addClass("custom-active");
		if (mode == this.currentMode) return;
		const action = this.modes[mode];
		const lastAction = this.modes[this.currentMode];
		this.currentMode = mode;
		this.currentVisualization = this.things[action.thing];
		if (!lastAction || action.thing != lastAction.thing) {
			if(lastAction) this.things[lastAction.thing].onHide();
			this.body.children().detach(); // keep event handlers
			this.body.append(this.currentVisualization.container);
			this.currentVisualization.onView(true, action.action);
		} else if (action.action !== lastAction.action) {
			this.currentVisualization.onView(false, action.action);
		}
	}
	onNetworkLoaded(net: Net.NeuralNet) {
		//todo: ugly hack
		const beforeActions = JSON.stringify(this.things.map(t => t.actions));
		this.things.forEach(thing => thing.onNetworkLoaded(net));
		const afterActions = JSON.stringify(this.things.map(t => t.actions));
		if(beforeActions !== afterActions) {
			this.createButtonsAndActions();
			this.currentMode = -1;
			this.setMode(0);
		}
	}
}