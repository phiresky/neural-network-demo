/** display the error history from [[Simulation#errorHistory]] as a line graph */
class ErrorGraph implements Visualization {
	chart: HighstockChartObject;
	actions = ["Error History"];
	container = $("<div>");
	constructor(public sim: Simulation) {
		this.container.highcharts(<any>{
			title: { text: 'Average RMSE' },
			chart: { type: 'line', animation: false },
			plotOptions: { line: { marker: { enabled: false } } },
			legend: { enabled: false },
			yAxis: { min: 0, title: { text: '' }, labels: { format: "{value:%.2f}" } },
			series: [{ name: 'Error', data: [] }],
			colors: ["black"],
			credits: { enabled: false }
		});
		this.chart = this.container.highcharts();
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