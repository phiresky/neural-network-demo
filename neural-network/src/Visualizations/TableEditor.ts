declare var Handsontable: any, LZString: any;
class TableEditor implements Visualization {
	hot: any; // handsontable instance
	actions = ["Table input"];
	headerCount = 2;
	lastUpdate = 0;
	container: JQuery = $("<div>");
	constructor(public sim: Simulation) {
		this.sim = sim;
	}
	afterChange(changes: [number, number, number, number][], reason: string) {
		if (reason === 'loadData') return;
		this.reparseData();
	}
	onNetworkLoaded(net:Net.NeuralNet) {
		if (this.hot) this.hot.destroy();
		let oldContainer = this.container;
		this.container = $("<div class='fullsize' style='overflow:hidden'>");
		console.log("new cont");
		if (oldContainer) oldContainer.replaceWith(this.container);
		$("<div>").addClass("btn btn-default")
			.css({ position: "absolute", right: "2em", bottom: "2em" })
			.text("Remove all")
			.click(e => { sim.config.data = []; this.loadData() })
			.appendTo(this.container);
		let headerRenderer = function firstRowRenderer(instance: any, td: HTMLTableCellElement) {
			Handsontable.renderers.TextRenderer.apply(this, arguments);
			td.style.fontWeight = 'bold';
			td.style.background = '#CCC';
		}
		let mergeCells: {}[] = [];
		let ic = net.inputs.length, oc = net.outputs.length;
		//console.log(`creating new table (${ic}, ${oc})`);
		if (ic > 1) mergeCells.push({ row: 0, col: 0, rowspan: 1, colspan: ic });
		if (oc > 1) {
			mergeCells.push({ row: 0, col: ic, rowspan: 1, colspan: oc });
			mergeCells.push({ row: 0, col: ic + oc, rowspan: 1, colspan: oc });
		}
		let _conf: Handsontable.Options = {
			minSpareRows: 1,
			colWidths: ic + oc + oc <= 6 ? 80 : 45,
			cells: (row, col, prop) => {
				if (row >= this.headerCount) return { type: 'numeric', format: '0.[000]' };
				else {
					let conf: any = { renderer: headerRenderer };
					if (row == 0) conf.readOnly = true;
					return conf;
				}
			},
			customBorders: false/*[{ // bug when larger than ~4
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
				}]*/,
			allowInvalid: false,
			mergeCells: mergeCells,
			afterChange: this.afterChange.bind(this)
		};
		this.container.handsontable(_conf);
		this.hot = this.container.handsontable('getInstance');
		this.loadData();
	}
	reparseData() {
		let sim = this.sim;
		let data: number[][] = this.hot.getData();
		let headers = <string[]><any>data[1];
		let ic = sim.config.inputLayer.neuronCount, oc = sim.config.outputLayer.neuronCount
		sim.config.inputLayer.names = headers.slice(0, ic);
		sim.config.outputLayer.names = headers.slice(ic, ic + oc);
		sim.config.data = data.slice(2).map(row => row.slice(0, ic + oc))
			.filter(row => row.every(cell => typeof cell === 'number'))
			.map(row => <TrainingData>{ input: row.slice(0, ic), output: row.slice(ic) });
		sim.setIsCustom();
	}
	onFrame() {
		let sim = this.sim;
		if ((Date.now() - this.lastUpdate) < 500) return;
		this.lastUpdate = Date.now();
		let xOffset = sim.config.inputLayer.neuronCount + sim.config.outputLayer.neuronCount;
		let vals: [number, number, number][] = [];
		for (let y = 0; y < sim.config.data.length; y++) {
			let p = sim.config.data[y];
			let op = sim.net.getOutput(p.input);
			for (let x = 0; x < op.length; x++) {
				vals.push([y + this.headerCount, xOffset + x, op[x]]);
			}
		}
		this.hot.setDataAtCell(vals, "loadData");
	}
	loadData() {
		let sim = this.sim;
		let data: (number|string)[][] = [[], sim.config.inputLayer.names.concat(sim.config.outputLayer.names).concat(sim.config.outputLayer.names)];
		let ic = sim.config.inputLayer.neuronCount, oc = sim.config.outputLayer.neuronCount;
		data[0][0] = 'Inputs';
		data[0][ic] = 'Expected Output';
		data[0][ic + oc + oc - 1] = ' ';
		data[0][ic + oc] = 'Actual Output';
		let mergeCells: {}[] = [];
		if (ic > 1) mergeCells.push({ row: 0, col: 0, rowspan: 1, colspan: ic });
		if (oc > 1) {
			mergeCells.push({ row: 0, col: ic + oc, rowspan: 1, colspan: oc });
			mergeCells.push({ row: 0, col: ic + oc * 2, rowspan: 1, colspan: oc });
		}
		if (mergeCells.length > 0) this.hot.updateSettings({ mergeCells: mergeCells });

		sim.config.data.forEach(t => data.push(t.input.concat(t.output)));
		this.hot.loadData(data);
		/*this.hot.updateSettings({customBorders: [
				
			]});
		this.hot.runHooks('afterInit');*/
	}
	onView() {
		this.onNetworkLoaded(this.sim.net);
	}
	onHide() {
		//this.reparseData();
	}
}