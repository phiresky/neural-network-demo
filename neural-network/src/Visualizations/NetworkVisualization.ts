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
		let tmp = NetworkVisualization.colors.multiClass;
		tmp.bg = tmp.fg.map(c => Util.printColor(<any>Util.parseColor(c).map(x => (x*1.3)|0))); 
		this.canvas = <HTMLCanvasElement>$("<canvas class=fullsize>")[0];
		this.canvas.width = 550;
		this.canvas.height = 400;
		this.trafo = new TransformNavigation(this.canvas, () => this.inputMode == this.actions.length - 1, () => this.onFrame());
		this.ctx = <CanvasRenderingContext2D>this.canvas.getContext('2d');
		window.addEventListener('resize', this.canvasResized.bind(this));
		this.canvas.addEventListener("click", this.canvasClicked.bind(this));
		this.canvas.addEventListener("contextmenu", this.canvasClicked.bind(this));
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
				this.actions = ["Add Red", "Add Green", "Remove", "Move View"];
				break;
			case NetType.AutoEncode:
				this.actions = ["Add Data point", "", "Remove", "Move View"];
				break;
			case NetType.MultiClass:
				this.actions = [];
				let i = 0;
				for(let name of this.sim.config.outputLayer.names)
					this.actions.push({name:name, color:NetworkVisualization.colors.multiClass.bg[i++]});
				this.actions.push("Remove");
				this.actions.push("Move View");
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

		this.drawBackground();
		this.drawCoordinateSystem();
		this.drawDataPoints();
	}

	drawDataPoints() {
		this.ctx.strokeStyle = "#000";
		if (this.netType === NetType.BinaryClassify) {
			for (let val of this.sim.config.data) {
				this.drawPoint(val.input[0], val.input[1], NetworkVisualization.colors.binaryClassify.fg[val.output[0] | 0]);
			}
		} else if (this.netType === NetType.AutoEncode) {
			for (let val of this.sim.config.data) {
				let ix = val.input[0], iy = val.input[1];
				let out = this.sim.net.getOutput(val.input);
				let ox = out[0], oy = out[1];
				this.drawLine(ix, iy, ox, oy, "black");
				this.drawPoint(ix, iy, NetworkVisualization.colors.autoencoder.input);
				this.drawPoint(ox, oy, NetworkVisualization.colors.autoencoder.output);
			}
		} else if(this.netType === NetType.MultiClass) {
			for (let val of this.sim.config.data) {
				this.drawPoint(val.input[0], val.input[1], NetworkVisualization.colors.multiClass.fg[Util.getMaxIndex(val.output)]);
			}
		} else {
			throw "can't draw this"
		}
	}

	drawLine(x: double, y: double, x2: double, y2: double, color: string) {
		x = this.trafo.toCanvas.x(x); x2 = this.trafo.toCanvas.x(x2);
		y = this.trafo.toCanvas.y(y); y2 = this.trafo.toCanvas.y(y2);
		this.ctx.strokeStyle = color;
		this.ctx.beginPath();
		this.ctx.moveTo(x, y);
		this.ctx.lineTo(x2, y2);
		this.ctx.stroke();
	}

	drawPoint(x: double, y: double, color: string) {
		x = this.trafo.toCanvas.x(x); y = this.trafo.toCanvas.y(y);
		this.ctx.fillStyle = color;
		this.ctx.beginPath();
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
	drawBackground() {
		if (this.sim.config.outputLayer.neuronCount === 2) {
			this.clear('white');
			return;
		}
		if(this.sim.config.outputLayer.neuronCount > 2) {
		for (let x = 0; x < this.canvas.width; x += this.backgroundResolution) {
			for (let y = 0; y < this.canvas.height; y += this.backgroundResolution) {
				let vals = this.sim.net.getOutput([this.trafo.toReal.x(x + this.backgroundResolution / 2), this.trafo.toReal.y(y + this.backgroundResolution / 2)]);
				let maxi = Util.getMaxIndex(vals);
				this.ctx.fillStyle = NetworkVisualization.colors.multiClass.bg[maxi];
				this.ctx.fillRect(x, y, this.backgroundResolution, this.backgroundResolution);
			}
		}
		} else {
		for (let x = 0; x < this.canvas.width; x += this.backgroundResolution) {
			for (let y = 0; y < this.canvas.height; y += this.backgroundResolution) {
				let val = this.sim.net.getOutput([this.trafo.toReal.x(x + this.backgroundResolution / 2), this.trafo.toReal.y(y + this.backgroundResolution / 2)])[0];

				if (this.sim.config.showGradient) {
					this.ctx.fillStyle = NetworkVisualization.colors.binaryClassify.gradient(val);
				} else this.ctx.fillStyle = NetworkVisualization.colors.binaryClassify.bg[+(val > 0.5)];
				this.ctx.fillRect(x, y, this.backgroundResolution, this.backgroundResolution);
			}
		}}
	}
	drawCoordinateSystem() {
		let marklen = 0.2;
		let ctx = this.ctx, toc = this.trafo.toCanvas;
		ctx.strokeStyle = "#000";
		ctx.fillStyle = "#000";
		ctx.textBaseline = "middle";
		ctx.textAlign = "center";
		ctx.font = "20px monospace";
		ctx.beginPath();

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
		if(this.sim.config.data.length < 3) return;
		// update transform
		if(this.sim.config.inputLayer.neuronCount == 2) {
			let fillamount = 0.6;
			let bounds = Util.bounds2dTrainingsInput(this.sim.config.data);
			let w = bounds.maxx - bounds.minx, h = bounds.maxy - bounds.miny;
			this.trafo.scalex = this.canvas.width / w * fillamount;
			this.trafo.scaley = - this.canvas.height / h * fillamount;
			this.trafo.offsetx -= this.trafo.toCanvas.x(bounds.minx - w*(1-fillamount)/1.5);// / bounds.minx;
			this.trafo.offsety -= this.trafo.toCanvas.y(bounds.maxy + h*(1-fillamount)/1.5);// / bounds.minx;
		}
	}
	canvasClicked(evt: MouseEvent) {
		evt.preventDefault();
		let data = this.sim.config.data;
		let rect = this.canvas.getBoundingClientRect();
		let x = this.trafo.toReal.x(evt.clientX - rect.left);
		let y = this.trafo.toReal.y(evt.clientY - rect.top);
		let removeMode = this.actions.length - 2;
		if (this.inputMode === removeMode || evt.button == 2 || evt.shiftKey) {
			//remove nearest
			let nearestDist = Infinity, nearest = -1;
			for (let i = 0; i < data.length; i++) {
				let p = data[i];
				let dx = p.input[0] - x, dy = p.input[1] - y, dist = dx * dx + dy * dy;
				if (dist < nearestDist) nearest = i, nearestDist = dist;
			}
			if (nearest >= 0) data.splice(nearest, 1);
		} else if (this.inputMode < removeMode) {
			// add data point
			if (this.netType === NetType.AutoEncode) {
				data.push({ input: [x, y], output: [x, y] });
			} else {
				let inv = (x: int) => x == 0 ? 1 : 0;
				let label = this.inputMode;
				if (evt.button != 0) label = inv(label);
				if (evt.ctrlKey || evt.metaKey || evt.altKey) label = inv(label);
				let output = [label];
				if(this.netType === NetType.MultiClass) {
					output = Util.arrayWithOneAt(this.sim.config.outputLayer.neuronCount, label);
				}
				data.push({ input: [x, y], output: output });
			}
		} else return;
		this.sim.setIsCustom();
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