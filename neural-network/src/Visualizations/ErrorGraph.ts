class ErrorGraph implements Visualization {
	chart: HighstockChartObject;
	actions = ["Error History"];
	container = $("<div>");
	constructor(public sim: Simulation) {
		this.container.highcharts({
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
		let data:[number,number] = [this.sim.stepNum, this.sim.averageError];
		this.chart.series[0].addPoint(data, true, false);
	}
	onView() {
		this.chart.series[0].setData(this.sim.errorHistory);
		this.chart.reflow();
	}
	onNetworkLoaded() {
		this.chart.series[0].setData([]);
	}
	onHide() {
		
	}
}