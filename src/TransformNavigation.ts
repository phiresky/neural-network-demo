import { int, double } from "./main";
import { stopEvent } from "./Util";

/**
 * this class handles the linear transformations used to offset and scale the drawing in a 2d canvas
 * @see [[#toReal]] and [[#toCanvas]]
 */
export default class TransformNavigation {
	public scalex = 200;
	public scaley = -200;
	offsetx = 0;
	offsety = 0;

	/** converts screen space coordinates to real coordinates */
	public toReal = {
		x: (x: double) => (x - this.offsetx) / this.scalex,
		y: (y: double) => (y - this.offsety) / this.scaley
	};
	/** converts real coordinates to screen space coordinates */
	public toCanvas = {
		x: (c: double) => c * this.scalex + this.offsetx,
		y: (c: double) => c * this.scaley + this.offsety
	};

	/** position where the mouse press started (for dragging) */
	private mousestart: { x: number; y: number } = null;
	/**
	 * @param transformActive function that returns if the mouse transform should act on mouse dragging / scrolling events currently
	 * @param transformChanged callback when the transform has changed (e.g. to redraw)
	 */
	constructor(
		canvas: HTMLCanvasElement,
		transformActive: () => boolean,
		transformChanged: () => void
	) {
		this.offsetx = canvas.width / 4;
		this.offsety = (3 * canvas.height) / 4;
		canvas.addEventListener("wheel", e => {
			if (e.deltaY === 0) return;
			if (!transformActive()) return;
			var delta = e.deltaY / Math.abs(e.deltaY);
			const beforeTransform = {
				x: this.toReal.x(e.offsetX),
				y: this.toReal.y(e.offsetY)
			};
			this.scalex *= 1 - delta / 10;
			this.scaley *= 1 - delta / 10;
			const afterTransform = {
				x: this.toReal.x(e.offsetX),
				y: this.toReal.y(e.offsetY)
			};
			this.offsetx +=
				(afterTransform.x - beforeTransform.x) * this.scalex;
			this.offsety +=
				(afterTransform.y - beforeTransform.y) * this.scaley;
			transformChanged();
			stopEvent(e);
		});
		canvas.addEventListener("mousedown", e => {
			if (!transformActive()) return;
			this.mousestart = { x: e.pageX, y: e.pageY };
			stopEvent(e);
		});
		window.addEventListener("mousemove", e => {
			if (!transformActive()) return;
			if (!this.mousestart) return;
			this.offsetx += e.pageX - this.mousestart.x;
			this.offsety += e.pageY - this.mousestart.y;
			this.mousestart.x = e.pageX;
			this.mousestart.y = e.pageY;
			transformChanged();
			stopEvent(e);
		});
		window.addEventListener("mouseup", e => {
			if (this.mousestart) {
				this.mousestart = null;
				stopEvent(e);
			}
		});
	}
}
