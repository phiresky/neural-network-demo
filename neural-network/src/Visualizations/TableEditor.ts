declare var Handsontable: any, LZString: any;
/**
 * Edit training data and display network output using a table interface
 */
class TableEditor implements Visualization {
	/** handsontable instance */
	hot: any;
	actions = ["Table input"];
	headerCount = 2;
	lastUpdate = 0;
	container: JQuery = $("<div>");
	constructor(public sim: Simulation) { }
	/** called by Handsontable after some data was changed in the table */
	afterChange(changes: [number, number, number, number][], reason: string) {
		if (reason === 'loadData') return;
		this.reparseData();
	}
	onNetworkLoaded(net:Net.NeuralNet) {
		if (this.hot) this.hot.destroy();
		const oldContainer = this.container;
		this.container = $("<div class='fullsize' style='overflow:hidden'>");
		if (oldContainer) oldContainer.replaceWith(this.container);
		$("<div>").addClass("btn btn-default")
			.css({ position: "absolute", right: "2em", bottom: "2em" })
			.text("Remove all")
			.click(e => this.sim.setState({data: []}, () => this.loadData()))
			.appendTo(this.container);
		const headerRenderer = function firstRowRenderer(instance: any, td: HTMLTableCellElement) {
			Handsontable.renderers.TextRenderer.apply(this, arguments);
			td.style.fontWeight = 'bold';
			td.style.background = '#CCC';
		}
		const mergeCells: {}[] = [];
		const ic = net.inputs.length, oc = net.outputs.length;
		//console.log(`creating new table (${ic}, ${oc})`);
		if (ic > 1) mergeCells.push({ row: 0, col: 0, rowspan: 1, colspan: ic });
		if (oc > 1) {
			mergeCells.push({ row: 0, col: ic, rowspan: 1, colspan: oc });
			mergeCells.push({ row: 0, col: ic + oc, rowspan: 1, colspan: oc });
		}
		const _conf = <Handsontable.Options>{
			minSpareRows: 1,
			colWidths: ic + oc + oc <= 6 ? 80 : 45,
			cells: (row, col, prop) => {
				if (row >= this.headerCount) return { type: 'numeric', format: '0.[000]' };
				else {
					const conf: any = { renderer: headerRenderer };
					if (row == 0) conf.readOnly = true;
					return conf;
				}
			},
			/*customBorders: false[{ // bug when larger than ~4
				range: {
					from: { row: 0, col: ic },
					to: { row: 100, col: ic }
				},
				left: { width: 2, color: 'black' }
			}, {
					range: {
						from: { row: 0, col: ic + oc },
						to: { row: 100, col: ic + oc }
					},
					left: { width: 2, color: 'black' }
				}],*/
			allowInvalid: false,
			mergeCells: mergeCells,
			afterChange: this.afterChange.bind(this)
		};
		this.container.handsontable(_conf);
		this.hot = this.container.handsontable('getInstance');
		this.loadData();
	}
	/** read data from table into the [[Configuration]] */
	reparseData() {
		const sim = this.sim;
		const data: number[][] = this.hot.getData();
		const headers = <string[]><any>data[1];
		const newConfig = Util.cloneConfig(sim.state);
		const ic = newConfig.inputLayer.neuronCount, oc = newConfig.outputLayer.neuronCount
		newConfig.inputLayer.names = headers.slice(0, ic);
		newConfig.outputLayer.names = headers.slice(ic, ic + oc);
		newConfig.data = data.slice(2).map(row => row.slice(0, ic + oc))
			.filter(row => row.every(cell => typeof cell === 'number'))
			.map(row => <TrainingData>{ input: row.slice(0, ic), output: row.slice(ic) });
		newConfig.custom = true;
		sim.setState(newConfig);
	}
	onFrame() {
		const sim = this.sim;
		if ((Date.now() - this.lastUpdate) < 500) return;
		this.lastUpdate = Date.now();
		const xOffset = sim.state.inputLayer.neuronCount + sim.state.outputLayer.neuronCount;
		const vals: [number, number, number][] = [];
		for (let y = 0; y < sim.state.data.length; y++) {
			const p = sim.state.data[y];
			const op = sim.net.getOutput(p.input);
			for (let x = 0; x < op.length; x++) {
				vals.push([y + this.headerCount, xOffset + x, op[x]]);
			}
		}
		this.hot.setDataAtCell(vals, "loadData");
	}
	/** load data from [[Configuration]] into the table */
	loadData() {
		const sim = this.sim;
		const data: (number|string)[][] = [[], sim.state.inputLayer.names.concat(sim.state.outputLayer.names).concat(sim.state.outputLayer.names)];
		const ic = sim.state.inputLayer.neuronCount, oc = sim.state.outputLayer.neuronCount;
		data[0][0] = 'Inputs';
		data[0][ic] = 'Expected Output';
		data[0][ic + oc + oc - 1] = ' ';
		data[0][ic + oc] = 'Actual Output';
		const mergeCells: {}[] = [];
		if (ic > 1) mergeCells.push({ row: 0, col: 0, rowspan: 1, colspan: ic });
		if (oc > 1) {
			mergeCells.push({ row: 0, col: ic + oc, rowspan: 1, colspan: oc });
			mergeCells.push({ row: 0, col: ic + oc * 2, rowspan: 1, colspan: oc });
		}
		if (mergeCells.length > 0) this.hot.updateSettings({ mergeCells: mergeCells });

		sim.state.data.forEach(t => data.push(t.input.concat(t.output)));
		this.hot.loadData(data);
		/*this.hot.updateSettings({customBorders: [
				
			]});
		this.hot.runHooks('afterInit');*/
	}
	onView() {
		this.onNetworkLoaded(this.sim.net);
		this.onFrame();
	}
	onHide() {
		//this.reparseData();
	}
}