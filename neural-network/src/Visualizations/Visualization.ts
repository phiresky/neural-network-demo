interface Visualization {
	container: JQuery;
	actions: string[];
	onView: (previouslyHidden: boolean, mode: int) => void;
	onNetworkLoaded: (net:Net.NeuralNet) => void;
	onHide: () => void;
	onFrame: () => void;
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
	constructor(public container: JQuery, public name: string, public things: Visualization[]) {
		this.createButtonsAndActions();
		this.ul.on("click", "a", e => this.setMode($(e.target).parent().index()));
		container.append(this.ul);
		container.append(this.body);
	}
	createButtonsAndActions() {
		this.ul.empty();
		this.modes = [];
		this.things.forEach((thing, thingid) =>
			thing.actions.forEach((button, buttonid) => {
				this.modes.push({ thing: thingid, action: buttonid });
				let li = $("<li>").append($("<a>").text(button));
				if(!button) li.hide();
				this.ul.append(li);
			})
		);
	}
	setMode(mode:int) {
		console.log("setting mode to "+mode);
		this.ul.children("li.active").removeClass("active");
		this.ul.children().eq(mode).addClass("active");
		if (mode == this.currentMode) return;
		let action = this.modes[mode];
		let lastAction = this.modes[this.currentMode];
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
		let beforeActions = JSON.stringify(this.things.map(t => t.actions));
		this.things.forEach(thing => thing.onNetworkLoaded(net));
		let afterActions = JSON.stringify(this.things.map(t => t.actions));
		if(beforeActions !== afterActions) {
			this.createButtonsAndActions();
			this.setMode(0);
		}
	}
}