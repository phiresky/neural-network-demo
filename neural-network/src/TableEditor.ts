declare var Handsontable: any, LZString: any;
class TableEditor {
	hot: any; // handsontable instance
	headerCount = 2;
	lastUpdate = 0;
	constructor(public container: JQuery, sim: Simulation) {
		let headerRenderer = function firstRowRenderer(instance: any, td: HTMLTableCellElement) {
			Handsontable.renderers.TextRenderer.apply(this, arguments);
			td.style.fontWeight = 'bold';
			td.style.background = '#CCC';
		}
		container.handsontable({
			minSpareRows: 1,
			cells: (row, col, prop) => {
				if (row >= this.headerCount) return { type: 'numeric', format: '0.[000]' };
				else return { renderer: headerRenderer };
			},
			//customBorders: true,
			allowInvalid: false,
			afterChange: this.afterChange.bind(this)
		});
		this.hot = container.handsontable('getInstance');
		$("<div>").addClass("btn btn-default")
			.css({ position: "absolute", right: "2em", bottom: "2em" })
			.text("Remove all")
			.click(e => { sim.config.data = []; this.loadData(sim) })
			.appendTo(container);
		this.loadData(sim);
	}
	afterChange(changes: [number, number, number, number][], reason: string) {
		if (reason === 'loadData') return;
		this.reparseData();
	}
	reparseData() {
		let data: number[][] = this.hot.getData();
		let headers = <string[]><any>data[1];
		let ic = sim.config.inputLayer.neuronCount, oc = sim.config.outputLayer.neuronCount
		sim.config.inputLayer.names = headers.slice(0, ic);
		sim.config.outputLayer.names = headers.slice(ic, ic + oc);
		sim.config.data = data.slice(2).map(row => row.slice(0, ic + oc)).filter(row => row.every(cell => typeof cell === 'number'))
			.map(row => <TrainingData>{ input: row.slice(0, ic), output: row.slice(ic) });
		sim.setIsCustom();
	}
	updateRealOutput() {
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
	loadData(sim: Simulation) { // needs sim as arg because called from constructor
		let data: (number|string)[][] = [[], sim.config.inputLayer.names.concat(sim.config.outputLayer.names).concat(sim.config.outputLayer.names)];
		let ic = sim.config.inputLayer.neuronCount, oc = sim.config.outputLayer.neuronCount;
		data[0][0] = 'Inputs';
		data[0][ic] = 'Expected Output';
		data[0][ic + oc + oc - 1] = ' ';
		data[0][ic + oc] = 'Actual Output';

		sim.config.data.forEach(t => data.push(t.input.concat(t.output)));
		this.hot.loadData(data);
		/*this.hot.updateSettings({customBorders: [
				{
					range: {
						from: { row: 0, col: ic },
						to: { row: 100, col: ic }
					},
					left: { width: 2, color: 'black' }
				}, {
					range: {
						from: { row: 0, col: ic+oc },
						to: { row: 100, col: ic+oc }
					},
					left: { width: 2, color: 'black' }
				}
			]});
		this.hot.runHooks('afterInit');*/
	}
}