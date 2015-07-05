interface Visualization {
	container: JQuery;
	onView: (previouslyHidden:boolean, mode: int) => void;
	onHide: () => void;
}

interface TabSwitchEle {
	visualization: Visualization;
	buttons: string[];
}

class TabSwitchVis {
	actions:{thing:int, action:int}[] = [];
	constructor(public container: JQuery, public currentAction: int, public name: string, public things: TabSwitchEle[]) {
		let ul = $("<ul class='nav nav-pills'>");
		let body = $("<div class='visbody'>");
		things.forEach((thing, thingid) =>
			thing.buttons.forEach((button,buttonid) => {
				this.actions.push({thing:thingid, action:buttonid});
				ul.append($("<li>").append($("<a>").text(button)));
			})
		);
		ul.on("click", "a", e => {
			ul.children("li.active").removeClass("active");
			let li = $(e.target).parent();
			li.addClass("active");
			let inx = li.index();
			if(inx == this.currentAction) return;
			let action = this.actions[inx];
			let lastAction = this.actions[this.currentAction];
			let vis = this.things[action.thing].visualization;
			if(action.thing != lastAction.thing) {
				this.currentAction = inx;
				console.log(`switching thing: ${lastAction.thing}->${action.thing}`)
				this.things[lastAction.thing].visualization.onHide();
				body.children().detach(); // keep event handlers
				body.append(vis.container);
				vis.onView(true, action.action);
			} else if(action.action != lastAction.action) {
				vis.onView(false, action.action);
			}
		});
		ul.children().first().addClass("active");
		container.append(ul);
		container.append(body);
		let first = this.actions[currentAction], vis = this.things[first.thing].visualization;
		body.append(vis.container);
		vis.onView(true, first.action);
	}
}