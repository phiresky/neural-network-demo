import * as $ from "jquery";
import {Visualization} from "./Visualization";
import Simulation from "../Simulation";
import * as highstock from "highstock-release";
/** display the error history from [[Simulation#errorHistory]] as a line graph */
export default class ErrorGraph implements Visualization {
	chart: any;
	actions = ["Error History"];
	container = document.createElement("div");
	constructor(public sim: Simulation) {
		this.chart = new highstock.Chart({
			title: { text: 'Average RMSE' },
			chart: { type: 'line', renderTo: this.container, animation: false },
			plotOptions: { line: { marker: { enabled: false } } },
			legend: { enabled: false },
			yAxis: { min: 0, title: { text: '' }, labels: { format: "{value:%.2f}" } },
			series: [{ name: 'Error', data: [] }],
			colors: ["black"],
			credits: { enabled: false }
		});
	}
	onFrame() {
		const data:[number,number] = [this.sim.stepsCurrent, this.sim.averageError];
		this.chart.series[0].addPoint(data, true, false);
	}
	onView() {
		this.chart.series[0].setData(this.sim.errorHistory.map(x => x.slice()));
		this.chart.reflow();
	}
	onNetworkLoaded() {
		this.chart.series[0].setData([]);
	}
	onHide() { }
}