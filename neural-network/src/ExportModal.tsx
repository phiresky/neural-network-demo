class ExportModal extends React.Component<{sim:Simulation},{exportWeights:string, errors:string[]}> {
	constructor(props:{sim:Simulation}) {
		super(props);
		this.state = {
			exportWeights: "0",
			errors: []
		}
	}
	render() {
		return (
			<div className="modal fade" id="exportModal">
				<div className="modal-dialog">
					<div className="modal-content">
						<div className="modal-header">
							<button type="button" className="close" data-dismiss="modal">×</button>
							<h3 className="modal-title">Import / Export</h3>
						</div>
						<div className="modal-body">
							<h4 className="modal-title">Export to URL</h4>
							<select className="exportWeights"
									onChange={t => this.setState({exportWeights: (t.target as HTMLSelectElement).value})} value={this.state.exportWeights}>
								<option value="0">Don't include weights</option>
								<option value="1">Include current weights</option>
								<option value="2">Include start weights</option>
							</select>
							<p>Copy this URL:
								<input className="url-export" onClick={e => (e.target as HTMLInputElement).select()} readOnly
									value={this.props.sim.serializeToUrl(+this.state.exportWeights)}
								/>
							</p>
							<hr />
							<h4 className="modal-title">Export to file</h4>
							<button className="btn btn-default" onClick={() => this.exportJSON(this.props.sim.state)}>
								Export configuration as json
							</button>
							<button className="btn btn-default" onClick={() => this.exportCSV(this.props.sim.state)}>
								Export training data as CSV
							</button>
							<hr />
							<h4 className="modal-title">Import</h4>
							<span className="btn btn-default btn-file">
								Import JSON file <input type="file" className="importJSON" onChange={this.importJSON.bind(this)} />
							</span>
							<span className="btn btn-default btn-file">
								Import CSV file <input type="file" className="importCSV" onChange={this.importCSV.bind(this)} />
							</span>
							{this.state.errors.map((error,i) => 
								<div key={i} className="alert alert-danger">{error}
									<button type="button" className="close" data-dismiss="alert">×</button>
								</div>
							)}
						</div>
					</div>
				</div>
			</div>
		);
	}
	exportJSON(conf:Configuration) {
		Util.download(JSON.stringify(conf, null, '\t'), conf.name + ".json");
	}
	exportCSV(conf:Configuration) {
		const csv = conf.inputLayer.names.concat(conf.outputLayer.names)
			.map(Util.csvSanitize).join(",") + "\n"
			+ conf.data.map(data => data.input.concat(data.output).join(",")).join("\n");
		Util.download(csv, conf.name + ".csv");
	}
	importJSON(ev: Event) {
		const files = (ev.target as HTMLInputElement).files;
		if (files.length !== 1) this.addIOError("invalid selection");
		const file = files.item(0);
		const r = new FileReader();
		r.onload = t => {
			try {
				const text = r.result;
				this.props.sim.setState(JSON.parse(text));
				$("#exportModal").modal('hide');
				$("#presetName").text(file.name);
			} catch (e) {
				this.addIOError("Error while reading " + file.name + ": " + e);
			}
		}
		r.readAsText(file);
	}
	
	importCSV(ev: Event) {
		console.log("imo");
		const files = (ev.target as HTMLInputElement).files;
		if (files.length !== 1) this.addIOError("invalid selection");
		const file = files.item(0);
		const r = new FileReader();
		const sim = this.props.sim;
		r.onload = t => {
			try {
				const text = r.result as string;
				const data = text.split("\n").map(l => l.split(","));
				const lens = data.map(l => l.length);
				const len = Math.min(...lens);
				if (len !== Math.max(...lens))
					throw `line lengths varying between ${len} and ${Math.max(...lens) }, must be constant`;
				const inps = sim.state.inputLayer.neuronCount;
				const oups = sim.state.outputLayer.neuronCount;
				if (len !== inps + oups)
					throw `invalid line length, expected (${inps} inputs + ${oups} outputs = ) ${inps+oups} columns, got ${len} columns`;
				const newState = Util.cloneConfig(sim.state);
				if(!data[0][0].match(/^\d+$/)) {
					const headers = data.shift();
					newState.inputLayer.names = headers.slice(0, inps);
					newState.outputLayer.names = headers.slice(inps, inps + oups);
				}
				newState.data = [];
				for(let l = 0; l < data.length; l++) {
					const ele:TrainingData = {input:[], output:[]};
					for(let i = 0; i < len; i++) {
						const v = parseFloat(data[l][i]);
						if(isNaN(v)) throw `can't parse ${data[l][i]} as a number in line ${l+1}`;
						(i < inps ? ele.input:ele.output).push(v);
					}
					newState.data.push(ele);
				}
				sim.setState(newState, () => sim.table.loadData());
				$("#presetName").text(file.name);
				$("#exportModal").modal('hide');
			} catch (e) {
				this.addIOError("Error while reading " + file.name + ": " + e);
				console.error(e);
			}
		}
		r.readAsText(file);
	}
	addIOError(err: string) {
		const errors = this.state.errors.slice();
		errors.push(err);
		this.setState({errors});
	}
}