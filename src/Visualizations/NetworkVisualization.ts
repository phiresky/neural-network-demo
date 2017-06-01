import * as $ from "jquery";
import { Visualization } from ".";
import { int, double } from "../main";
import Simulation from "../Simulation";
import Net from "../Net";
import TransformNavigation from "../TransformNavigation";
import * as Util from "../Util";
import { TrainingData } from "../Configuration";

enum NetType {
	/** 2D-Input, Single Output (≤0.5 is class 0, otherwise class 1) */
	BinaryClassify,
	/** 2D Input, 2D output, input values = output values*/
	AutoEncode,
	/** 2D Input, ≥ 3 outputs (argmax(outputs) is the resulting class) */
	MultiClass,
	/** any other configuration */
	CantDraw
}
/**
 * Visualize the training data and network output in a 2d canvas
 * 
 * Only works for problems with 2-dimensional input
 * 
 * For classification problems, draw every data point with the input as the position
 * and it's label/class as the color. Also fill the background with the current network output for that position.
 * 
 * For 2D auto encoding draw every target output point connected to the network output.
 */
export default class NetworkVisualization implements Visualization {
	actions: (string | { name: string, color: string })[] = [];
	canvas: HTMLCanvasElement;
	ctx: CanvasRenderingContext2D;
	/** [0] = Drag View; [length - 1] = Remove on click; [n] = Add data point of class (n-1) */
	inputMode: int = 0;
	trafo: TransformNavigation;
	backgroundResolution = 5;
	container = document.createElement("div");
	netType: NetType = NetType.BinaryClassify;
	static colors = {
		binaryClassify: {
			/** background color */
			bg: ["#f88", "#8f8"],
			/** data point color */
			fg: ["#f00", "#0f0"],
			/** color of weight arrows */
			weightVector: ["#800", "#080"],
			/** used when displaying the class in the background as a gradient*/
			gradient: (val: number) => "rgb(" +
				[(((1 - val) * (256 - 60)) | 0) + 60, ((val * (256 - 60)) | 0) + 60, 60] + ")"
		},
		autoencoder: {
			input: '#2188e0',
			output: '#ff931f',
			bias: '#aaa'
		},
		multiClass: {
			fg: ['#7cb5ec', '#434348', '#90ed7d', '#f7a35c', '#8085e9', '#f15c80', '#e4d354', '#2b908f', '#f45b5b', '#91e8e1'],
			/** filled in constructor by darkening fg colors */
			bg: ['']
		}
	}

	constructor(public sim: Simulation, public dataPointIsHighlighted: (p: TrainingData) => boolean) {
		const tmp = NetworkVisualization.colors.multiClass;
		tmp.bg = tmp.fg.map(c => Util.printColor(<any>Util.parseColor(c).map(x => (x * 1.3) | 0)));
		this.canvas = <HTMLCanvasElement>$("<canvas class=fullsize>")[0];
		this.canvas.width = 550;
		this.canvas.height = 400;
		this.trafo = new TransformNavigation(this.canvas, () => this.inputMode === 0 /* move view mode */,
			() => this.onFrame());
		this.ctx = <CanvasRenderingContext2D>this.canvas.getContext('2d');
		window.addEventListener('resize', this.canvasResized.bind(this));
		this.canvas.addEventListener("click", this.canvasClicked.bind(this));
		this.canvas.addEventListener("contextmenu", this.canvasClicked.bind(this));
		this.canvas.addEventListener("mousedown", Util.stopEvent); // prevent select text
		$(this.canvas).appendTo(this.container);
	}

	onNetworkLoaded(net: Net.NeuralNet) {
		if (net.inputs.length != 2) this.netType = NetType.CantDraw;
		else {
			if (net.outputs.length == 1) this.netType = NetType.BinaryClassify;
			else if (net.outputs.length == 2) this.netType = NetType.AutoEncode;
			else this.netType = NetType.MultiClass;
		}
		switch (this.netType) {
			case NetType.BinaryClassify:
				this.actions = ["Move View", "Add Red", "Add Green", "Remove"];
				break;
			case NetType.AutoEncode:
				this.actions = ["Move View", "Add Data point", "", "Remove"];
				break;
			case NetType.MultiClass:
				this.actions = ["Move View"];
				let i = 0;
				for (const name of this.sim.state.outputLayer.names)
					this.actions.push({ name: name, color: NetworkVisualization.colors.multiClass.bg[i++] });
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
		const isSinglePerceptron = this.sim.state.type === "perceptron";
		const separator = isSinglePerceptron && this.getSeparator(Util.toLinearFunction(this.sim.net.connections.map(i => i.weight) as any));
		if (isSinglePerceptron)
			this.drawPolyBackground(separator);
		else this.drawBackground();
		if (this.sim.state.drawCoordinateSystem) this.drawCoordinateSystem();
		if (this.sim.state.drawArrows) this.drawArrows();
		this.drawDataPoints();
		if (isSinglePerceptron) {
			const tor = this.trafo.toReal;
			if (this.sim.state.drawArrows && this.sim.lastWeights !== undefined && this.sim.lastWeights.length > 0) {
				const separator = this.getSeparator(Util.toLinearFunction(this.sim.lastWeights[0].weights as any));
				this.drawLine(tor.x(0), separator.min, tor.x(this.canvas.width), separator.max, "gray");
			}
			this.drawLine(tor.x(0), separator.min, tor.x(this.canvas.width), separator.max, "black");
		}
	}

	drawDataPoints() {
		this.ctx.strokeStyle = "#000";
		if (this.netType === NetType.BinaryClassify || this.netType === NetType.MultiClass) {
			for (const val of this.sim.state.data) {
				this.drawDataPoint(val);
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
		} else {
			throw "can't draw this"
		}
	}

	drawDataPoint(p: TrainingData) {
		const color =
			this.netType === NetType.BinaryClassify ?
				NetworkVisualization.colors.binaryClassify.fg[p.output[0] | 0]
				: this.netType === NetType.MultiClass ?
					NetworkVisualization.colors.multiClass.fg[Util.getMaxIndex(p.output)]
					: null;
		this.drawPoint(p.input[0], p.input[1], color, this.dataPointIsHighlighted(p));
	}

	drawPoint(x: number, y: number, color: string, highlight = false) {
		x = this.trafo.toCanvas.x(x), y = this.trafo.toCanvas.y(y);
		this.ctx.fillStyle = color;
		this.ctx.beginPath();
		this.ctx.lineWidth = highlight ? 5 : 1;
		this.ctx.strokeStyle = highlight ? "#000000" : "#000000";
		this.ctx.arc(x, y, 5, 0, 2 * Math.PI);
		this.ctx.fill();
		this.ctx.beginPath();
		this.ctx.arc(x, y, highlight ? 7 : 5, 0, 2 * Math.PI);
		this.ctx.stroke();
	}

	/** draw the weight vector arrows according to [[Simulation#lastWeights]] */
	drawArrows() {
		this.ctx.lineWidth = 2;
		const al = 8;
		const aw = 4;
		const steps = this.sim.lastWeights;
		if (steps === undefined || steps.length < 2) return;
		const scale = {
			x: (x: number) => this.trafo.toCanvas.x(x * this.sim.state.arrowScale),
			y: (y: number) => this.trafo.toCanvas.y(y * this.sim.state.arrowScale)
		}
		if (this.sim.state.inputLayer.neuronCount !== 2
			|| this.sim.state.outputLayer.neuronCount !== 1
			|| this.sim.state.hiddenLayers.length !== 0)
			throw Error("conf not valid for arrows");

		const weightVector = (weights: number[]) => ({ x: scale.x(weights[0]), y: scale.y(weights[1]) });
		if (steps[steps.length - 1].weights.some((x, i) => x !== steps[0].weights[i])) {
			let oldWeights = steps[0].weights.map(x => 0);

			for (const { weights, dataPoint } of steps) {
				this.ctx.strokeStyle = this.ctx.fillStyle = dataPoint ?
					NetworkVisualization.colors.binaryClassify.weightVector[dataPoint.output[0]]
					: "#888";
				Util.drawArrow(this.ctx, weightVector(oldWeights), weightVector(weights), al, aw);

				this.ctx.strokeStyle = this.ctx.fillStyle = "#808080";
				// this.ctx.arc(scale.x(p.input[0]), scale.y(p.input[1]), 8, 0, 2*Math.PI);
				oldWeights = weights;
			}
			this.ctx.strokeStyle = this.ctx.fillStyle = "#000000";
			Util.drawArrow(this.ctx, { x: scale.x(0), y: scale.y(0) },
				weightVector(steps[steps.length - 1].weights), al, aw);
		}
	}

	/** 
	 * calculate the y position of the given function to the left and right of the canvas in actual/real coordinates
	 */
	getSeparator(lineFunction: (x: number) => number) {
		return {
			min: lineFunction(this.trafo.toReal.x(0)),
			max: lineFunction(this.trafo.toReal.x(this.canvas.width))
		};
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
	clear(color: string) {
		this.ctx.fillStyle = "white";
		this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
		return;
	}
	/** divide the canvas into two regions using the linear function run through [[getSeparator]] as the divider and color the regions */
	drawPolyBackground({ min, max }: { min: number, max: number }) {
		const colors = NetworkVisualization.colors.binaryClassify.bg;
		const ctx = this.ctx;
		const c = this.trafo.toCanvas;
		const tmp = (y: number) => {
			ctx.beginPath();
			ctx.moveTo(0, c.y(min));
			ctx.lineTo(0, y);
			ctx.lineTo(this.canvas.width, y);
			ctx.lineTo(this.canvas.width, c.y(max));
			ctx.fill();
		}
		const upperIsClass1 = +(this.sim.net.getOutput([this.trafo.toReal.x(0), min - 1])[0] > 0.5);
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
				if (this.sim.state.outputLayer.neuronCount > 2) {
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
		if (this.sim.state.data.length < 3) return;
		// update transform
		if (this.sim.state.inputLayer.neuronCount == 2) {
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
				let label = this.inputMode - 1;
				if (evt.button != 0) label = inv(label);
				if (evt.ctrlKey || evt.metaKey || evt.altKey) label = inv(label);
				let output = [label];
				if (this.netType === NetType.MultiClass) {
					output = Util.arrayWithOneAt(this.sim.state.outputLayer.neuronCount, label);
				}
				data.push({ input: [x, y], output: output });
			}
		} else return;
		this.sim.setState({ data, custom: true });
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
