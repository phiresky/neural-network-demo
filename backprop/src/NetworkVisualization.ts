///<reference path='Transform.ts' />
interface Data {
	x: double; y: double; label: int;
}

class NetworkVisualization {
	canvas: HTMLCanvasElement;
	ctx: CanvasRenderingContext2D;
	trafo: Transform;
	showGradient: boolean;
	colors = {
		bg: ["#f88", "#8f8"],
		fg: ["#f00", "#0f0"],
		gradient: (val: number) => "rgb(" + [(val * 256) | 0, ((1 - val) * 256) | 0, 0] + ")"
	}

	constructor(outputCanvas: HTMLCanvasElement, trafo: Transform) {
		this.canvas = outputCanvas;
		this.ctx = <CanvasRenderingContext2D>this.canvas.getContext('2d');
		this.trafo = trafo;
		this.canvasResized();
		window.addEventListener('resize', this.canvasResized.bind(this));
	}

	drawDataPoints(data: Data[]) {
		this.ctx.strokeStyle = "#000";
		for (let val of data) {
			this.ctx.fillStyle = this.colors.fg[val.label | 0];
			this.ctx.beginPath();
			this.ctx.arc(this.trafo.xtoc(val.x), this.trafo.ytoc(val.y), 5, 0, 2 * Math.PI);
			this.ctx.fill();
			this.ctx.arc(this.trafo.xtoc(val.x), this.trafo.ytoc(val.y), 5, 0, 2 * Math.PI);
			this.ctx.stroke();
		}
	}
	drawBackground(resolution: int, classify: (x: double, y: double) => int) {
		for (let x = 0; x < this.canvas.width; x += resolution) {
			for (let y = 0; y < this.canvas.height; y += resolution) {
				let val = classify(this.trafo.ctox(x), this.trafo.ctoy(y));

				if (this.showGradient) {
					this.ctx.fillStyle = this.colors.gradient(val);
				} else this.ctx.fillStyle = this.colors.bg[(val + 0.5) | 0];
				this.ctx.fillRect(x, y, resolution, resolution);
			}
		}
	}
	drawCoordinateSystem() {
		let marklen = 0.2;
		let ctx = this.ctx, xtoc = this.trafo.xtoc, ytoc = this.trafo.ytoc;
		ctx.strokeStyle = "#000";
		ctx.fillStyle = "#000";
		ctx.textBaseline = "middle";
		ctx.textAlign = "center";
		ctx.font = "20px monospace";
		ctx.beginPath();

		ctx.moveTo(xtoc(0), 0);
		ctx.lineTo(xtoc(0), this.canvas.height);

		ctx.moveTo(xtoc(-marklen / 2), ytoc(1));
		ctx.lineTo(xtoc(marklen / 2), ytoc(1));
		ctx.fillText("1", xtoc(-marklen), ytoc(1));

		ctx.moveTo(0, ytoc(0));
		ctx.lineTo(this.canvas.width, ytoc(0));

		ctx.moveTo(xtoc(1), ytoc(-marklen / 2));
		ctx.lineTo(xtoc(1), ytoc(marklen / 2));
		ctx.fillText("1", xtoc(1), ytoc(-marklen));
		ctx.stroke();
	}
	canvasResized() {
		this.canvas.width = $(this.canvas).width();
		this.canvas.height = $(this.canvas).height();
	}
}