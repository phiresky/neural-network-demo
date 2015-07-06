interface Visualization {
	container: JQuery;
	onView: (previouslyHidden: boolean, mode: int) => void;
	onNetworkLoaded: (net:Net.NeuralNet) => void;
	onHide: () => void;
	onFrame: () => void;
}

interface VisualizationConstructor {
	new (sim:Simulation): Visualization;
}

interface TabSwitchEle {
	visualization: Visualization;
	buttons: string[];
}

class TabSwitchVis {
	modes: { thing: int, action: int }[] = [];
	currentVisualization: Visualization;
	ul = $("<ul class='nav nav-pills'>");
	body = $("<div class='visbody'>");
	currentMode = -1;
	constructor(public container: JQuery, public name: string, public things: TabSwitchEle[]) {
		this.createButtonsAndActions(things);
		this.ul.on("click", "a", e => this.setMode($(e.target).parent().index()));
		container.append(this.ul);
		container.append(this.body);
	}
	createButtonsAndActions(things: TabSwitchEle[]) {
		this.ul.empty();
		things.forEach((thing, thingid) =>
			thing.buttons.forEach((button, buttonid) => {
				this.modes.push({ thing: thingid, action: buttonid });
				this.ul.append($("<li>").append($("<a>").text(button)));
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
		this.currentVisualization = this.things[action.thing].visualization;
		if (!lastAction || action.thing != lastAction.thing) {
			if(lastAction) this.things[lastAction.thing].visualization.onHide();
			this.body.children().detach(); // keep event handlers
			this.body.append(this.currentVisualization.container);
			this.currentVisualization.onView(true, action.action);
		} else if (action.action !== lastAction.action) {
			this.currentVisualization.onView(false, action.action);
		}
	}
	onNetworkLoaded(net: Net.NeuralNet) {
		this.things.forEach(thing => thing.visualization.onNetworkLoaded(net));
	}
}