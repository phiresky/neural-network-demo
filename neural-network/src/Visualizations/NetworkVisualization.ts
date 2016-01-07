interface TrainingData {
	input: double[]; output: double[];
}
enum InputMode {
	InputPrimary, InputSecondary, Remove, Move, Table
}
enum NetType {
	BinaryClassify, AutoEncode, MultiClass, CantDraw
}
class NetworkVisualization implements Visualization {
	actions: (string|{name:string, color:string})[] = [];
	canvas: HTMLCanvasElement;
	ctx: CanvasRenderingContext2D;
	inputMode: InputMode = 0;
	trafo: TransformNavigation;
	backgroundResolution = 15;
	container = $("<div>");
	netType: NetType = NetType.BinaryClassify;
	static colors = {
		binaryClassify: {
			bg: ["#f88", "#8f8"],
			fg: ["#f00", "#0f0"],
			gradient: (val: number) => "rgb(" +
				[(((1 - val) * (256 - 60)) | 0) + 60, ((val * (256 - 60)) | 0) + 60, 60] + ")"
		},
		autoencoder: {
			input: '#2188e0',
			output: '#ff931f',
			bias: '#008'
		},
		multiClass: {
			fg: ['#7cb5ec', '#434348', '#90ed7d', '#f7a35c', '#8085e9', '#f15c80', '#e4d354', '#2b908f', '#f45b5b', '#91e8e1'],
			bg: ['']
		}
	}

	constructor(public sim: Simulation) {
		const tmp = NetworkVisualization.colors.multiClass;
		tmp.bg = tmp.fg.map(c => Util.printColor(<any>Util.parseColor(c).map(x => (x*1.3)|0))); 
		this.canvas = <HTMLCanvasElement>$("<canvas class=fullsize>")[0];
		this.canvas.width = 550;
		this.canvas.height = 400;
		this.trafo = new TransformNavigation(this.canvas, () => this.inputMode == 0 /* move view mode*/,
			() => this.onFrame());
		this.ctx = <CanvasRenderingContext2D>this.canvas.getContext('2d');
		window.addEventListener('resize', this.canvasResized.bind(this));
		this.canvas.addEventListener("click", this.canvasClicked.bind(this));
		this.canvas.addEventListener("contextmenu", this.canvasClicked.bind(this));
		this.canvas.addEventListener("mousedown", Util.stopEvent); // prevent select text
		$(this.canvas).appendTo(this.container);
	}

	onNetworkLoaded(net: Net.NeuralNet) {
		if(net.inputs.length != 2) this.netType = NetType.CantDraw;
		else {
			if(net.outputs.length == 1) this.netType = NetType.BinaryClassify;
			else if(net.outputs.length == 2) this.netType = NetType.AutoEncode;
			else this.netType = NetType.MultiClass;
		}
		switch(this.netType) {
			case NetType.BinaryClassify:
				this.actions = ["Move View", "Add Red", "Add Green", "Remove"];
				break;
			case NetType.AutoEncode:
				this.actions = ["Move View", "Add Data point", "", "Remove"];
				break;
			case NetType.MultiClass:
				this.actions = ["Move View"];
				let i = 0;
				for(const name of this.sim.state.outputLayer.names)
					this.actions.push({name:name, color:NetworkVisualization.colors.multiClass.bg[i++]});
				this.actions.push("Remove");
				break;
			case NetType.CantDraw:
				this.actions = [];
				break;
		}
		this.refitData();
	}
	onFrame() {
		if (this.netType === NetType.CantDraw) {
			this.clear('white');
			this.ctx.fillStyle = 'black';
			this.ctx.textBaseline = "middle";
			this.ctx.textAlign = "center";
			this.ctx.font = "20px monospace";
			this.ctx.fillText("Cannot draw this data", this.canvas.width / 2, this.canvas.height / 2);
			return;
		}
		const isSinglePerceptron = this.sim.net.layers.length === 2 && this.netType === NetType.BinaryClassify;
		const separator:Util.Bounds = isSinglePerceptron && this.getSeparator(Util.toLinearFunction(this.sim.net.connections.map(i => i.weight) as any));
		if(isSinglePerceptron)
			this.drawPolyBackground(separator);
		else this.drawBackground();
		if(this.sim.state.drawCoordinateSystem) this.drawCoordinateSystem();
		if(this.sim.state.drawArrows) this.drawArrows();
		this.drawDataPoints();
		if(isSinglePerceptron) {
			if(this.sim.state.drawArrows && this.sim.lastWeights !== undefined) {
				const separator = this.getSeparator(Util.toLinearFunction(this.sim.lastWeights as any));
				this.drawLine(separator.minx, separator.miny, separator.maxx, separator.maxy, "gray");
			}
			this.drawLine(separator.minx, separator.miny, separator.maxx, separator.maxy, "black");
		}
	}

	drawDataPoints() {
		this.ctx.strokeStyle = "#000";
		if (this.netType === NetType.BinaryClassify) {
			for (const val of this.sim.state.data) {
				this.drawPoint(val.input[0], val.input[1], NetworkVisualization.colors.binaryClassify.fg[val.output[0] | 0]);
			}
		} else if (this.netType === NetType.AutoEncode) {
			for (const val of this.sim.state.data) {
				const ix = val.input[0], iy = val.input[1];
				const out = this.sim.net.getOutput(val.input);
				const ox = out[0], oy = out[1];
				this.drawLine(ix, iy, ox, oy, "black");
				this.drawPoint(ix, iy, NetworkVisualization.colors.autoencoder.input);
				this.drawPoint(ox, oy, NetworkVisualization.colors.autoencoder.output);
			}
		} else if(this.netType === NetType.MultiClass) {
			for (const val of this.sim.state.data) {
				this.drawPoint(val.input[0], val.input[1], NetworkVisualization.colors.multiClass.fg[Util.getMaxIndex(val.output)]);
			}
		} else {
			throw "can't draw this"
		}
	}
	
	drawArrows() {
		this.ctx.lineWidth = 2;
		const ww = this.sim.net.connections.map(c => c.weight);
		const oldww = this.sim.lastWeights;
		if (oldww === undefined) return;
		const scale = {
			x:(x:number) => this.trafo.toCanvas.x(x*this.sim.state.arrowScale),
			y:(y:number) => this.trafo.toCanvas.y(y*this.sim.state.arrowScale)
		}
		if(ww.length !== 3) throw Error("arrows only work with 2d data");
		if(this.sim.state.inputLayer.neuronCount !== 2
			|| this.sim.state.outputLayer.neuronCount !== 1
			|| this.sim.state.hiddenLayers.length !== 0)
			throw Error("conf not valid for arrows");
		if(ww.length !== oldww.length) throw Error("size changed");
		const wasPointWrong = (p:TrainingData) => +(oldww[0] * p.input[0] + oldww[1] * p.input[1] + oldww[2] >= 0) !== p.output[0];
		const wasVectorWrong = (dp:TrainingData[]) => dp.some(p => wasPointWrong(p));
		if (ww.some((x, i) => x !== oldww[i])) {
			let oldX = 0, oldY = 0, newX = 0, newY = 0;
			if(wasVectorWrong(this.sim.state.data)) {
	
				newX = oldww[0];
				newY = oldww[1];
	
				this.ctx.strokeStyle = "#808080";
				Util.drawArrow(this.ctx, {x:scale.x(oldX), y:scale.y(oldY)},
						{x:scale.x(newX), y:scale.y(newY)}, 5, 5);
	
				for (const p of this.sim.state.data) {
					if (wasPointWrong(p)) {
						oldX = newX;
						oldY = newY;
	
						if (p.output[0] == 1) {
							newX += p.input[0] * this.sim.net.learnRate;
							newY += p.input[1] * this.sim.net.learnRate;
							this.ctx.strokeStyle = "#008800";
						} else {
							newX -= p.input[0] * this.sim.net.learnRate;
							newY -= p.input[1] * this.sim.net.learnRate;
							this.ctx.strokeStyle = "#880000";
	
						}
						Util.drawArrow(this.ctx, {x:scale.x(oldX), y:scale.y(oldY)}, {x:scale.x(newX), y:scale.y(newY)},
								5, 5);
	
						this.ctx.strokeStyle = "#808080";
						this.ctx.arc(scale.x(p.input[0]), scale.y(p.input[1]), 8, 0, 2*Math.PI);
	
					}
				}
			}
			oldX = 0;
			oldY = 0;
			newX = ww[0];
			newY = ww[1];

			this.ctx.strokeStyle = "#000000";
			Util.drawArrow(this.ctx, {x:scale.x(oldX), y:scale.y(oldY)},
					{x:scale.x(newX), y:scale.y(newY)}, 5, 5);
		}
	}
	
	getSeparator(lineFunction:(x:number) => number):Util.Bounds {
		const minx = this.trafo.toReal.x(0);
		const maxx = this.trafo.toReal.x(this.canvas.width);
		const miny = lineFunction(minx);
		const maxy = lineFunction(maxx);
		return {minx, miny, maxx, maxy};
	}

	drawLine(x: double, y: double, x2: double, y2: double, color: string) {
		x = this.trafo.toCanvas.x(x); x2 = this.trafo.toCanvas.x(x2);
		y = this.trafo.toCanvas.y(y); y2 = this.trafo.toCanvas.y(y2);
		this.ctx.strokeStyle = color;
		this.ctx.beginPath();
		this.ctx.lineWidth = 2;
		this.ctx.moveTo(x, y);
		this.ctx.lineTo(x2, y2);
		this.ctx.stroke();
	}

	drawPoint(x: double, y: double, color: string) {
		x = this.trafo.toCanvas.x(x); y = this.trafo.toCanvas.y(y);
		this.ctx.fillStyle = color;
		this.ctx.beginPath();
		this.ctx.lineWidth = 1;
		this.ctx.arc(x, y, 5, 0, 2 * Math.PI);
		this.ctx.fill();
		this.ctx.arc(x, y, 5, 0, 2 * Math.PI);
		this.ctx.stroke();
	}
	clear(color: string) {
		this.ctx.fillStyle = "white";
		this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
		return;
	}
	drawPolyBackground(sep: Util.Bounds) {
		const colors = NetworkVisualization.colors.binaryClassify.bg;
		const ctx = this.ctx;
		const c = this.trafo.toCanvas;
		const tmp = (y:number) => {
			ctx.beginPath();
			ctx.moveTo(c.x(sep.minx), c.y(sep.miny));
			ctx.lineTo(c.x(sep.minx), y);
			ctx.lineTo(c.x(sep.maxx), y);
			ctx.lineTo(c.x(sep.maxx), c.y(sep.maxy));
			ctx.fill();
		}
		const upperIsClass1 = +(this.sim.net.getOutput([sep.minx, sep.miny - 1])[0] > 0.5);
		ctx.fillStyle = colors[1 - upperIsClass1];
		tmp(0);
		ctx.fillStyle = colors[upperIsClass1];
		tmp(this.canvas.height);
	}
	drawBackground() {
		if (this.sim.state.outputLayer.neuronCount === 2) {
			this.clear('white');
			return;
		}
		for (let x = 0; x < this.canvas.width; x += this.backgroundResolution) {
			for (let y = 0; y < this.canvas.height; y += this.backgroundResolution) {
					const vals = this.sim.net.getOutput([this.trafo.toReal.x(x + this.backgroundResolution / 2), this.trafo.toReal.y(y + this.backgroundResolution / 2)]);
				if(this.sim.state.outputLayer.neuronCount > 2) {
					this.ctx.fillStyle = NetworkVisualization.colors.multiClass.bg[Util.getMaxIndex(vals)];
				} else {	
					if (this.sim.state.showGradient) {
						this.ctx.fillStyle = NetworkVisualization.colors.binaryClassify.gradient(vals[0]);
					} else this.ctx.fillStyle = NetworkVisualization.colors.binaryClassify.bg[+(vals[0] > 0.5)];
				}
				this.ctx.fillRect(x, y, this.backgroundResolution, this.backgroundResolution);
			}
		}
	}
	drawCoordinateSystem() {
		const marklen = 0.1;
		const ctx = this.ctx, toc = this.trafo.toCanvas;
		ctx.strokeStyle = "#000";
		ctx.fillStyle = "#000";
		ctx.textBaseline = "middle";
		ctx.textAlign = "center";
		ctx.font = "20px monospace";
		ctx.beginPath();
		this.ctx.lineWidth = 2;

		ctx.moveTo(toc.x(0), 0);
		ctx.lineTo(toc.x(0), this.canvas.height);

		ctx.moveTo(toc.x(-marklen / 2), toc.y(1));
		ctx.lineTo(toc.x(marklen / 2), toc.y(1));
		ctx.fillText("1", toc.x(-marklen), toc.y(1));

		ctx.moveTo(0, toc.y(0));
		ctx.lineTo(this.canvas.width, toc.y(0));

		ctx.moveTo(toc.x(1), toc.y(-marklen / 2));
		ctx.lineTo(toc.x(1), toc.y(marklen / 2));
		ctx.fillText("1", toc.x(1), toc.y(-marklen));
		ctx.stroke();
	}
	canvasResized() {
		this.canvas.width = $(this.canvas).width();
		this.canvas.height = $(this.canvas).height();
		this.refitData();
		this.onFrame();
	}
	refitData() {
		if(this.sim.state.data.length < 3) return;
		// update transform
		if(this.sim.state.inputLayer.neuronCount == 2) {
			const fillamount = 0.6;
			const bounds = Util.bounds2dTrainingsInput(this.sim.state.data);
			const w = bounds.maxx - bounds.minx, h = bounds.maxy - bounds.miny;
			const scale = Math.min(this.canvas.width / w, this.canvas.height / h) * fillamount;
			this.trafo.scalex = scale;
			this.trafo.scaley = -scale;
			this.trafo.offsetx = -(bounds.maxx + bounds.minx) / 2 * scale + this.canvas.width / 2;
			this.trafo.offsety = (bounds.maxy + bounds.miny) / 2 * scale + this.canvas.height / 2;
		}
	}
	canvasClicked(evt: MouseEvent) {
		Util.stopEvent(evt);
		const data = this.sim.state.data.slice();
		const rect = this.canvas.getBoundingClientRect();
		const x = this.trafo.toReal.x(evt.clientX - rect.left);
		const y = this.trafo.toReal.y(evt.clientY - rect.top);
		const removeMode = this.actions.length - 1;
		if (this.inputMode === removeMode || evt.button == 2 || evt.shiftKey) {
			//remove nearest
			let nearestDist = Infinity, nearest = -1;
			for (let i = 0; i < data.length; i++) {
				const p = data[i];
				const dx = p.input[0] - x, dy = p.input[1] - y, dist = dx * dx + dy * dy;
				if (dist < nearestDist) nearest = i, nearestDist = dist;
			}
			if (nearest >= 0) data.splice(nearest, 1);
		} else if (this.inputMode < removeMode && this.inputMode > 0 /* move mode */) {
			// add data point
			if (this.netType === NetType.AutoEncode) {
				data.push({ input: [x, y], output: [x, y] });
			} else {
				const inv = (x: int) => x == 0 ? 1 : 0;
				let label = this.inputMode;
				if (evt.button != 0) label = inv(label);
				if (evt.ctrlKey || evt.metaKey || evt.altKey) label = inv(label);
				let output = [label];
				if(this.netType === NetType.MultiClass) {
					output = Util.arrayWithOneAt(this.sim.state.outputLayer.neuronCount, label);
				}
				data.push({ input: [x, y], output: output });
			}
		} else return;
		this.sim.setState({data, custom: true});
		this.sim.lastWeights = undefined;
		this.onFrame();
	}
	onView(previouslyHidden: boolean, mode: int) {
		if (previouslyHidden) this.canvasResized();
		this.inputMode = mode;
		this.onFrame();
	}
	onHide() {

	}
}
