type double = number; type int = number;
interface Transform {
	toReal: { x: (x: double) => double; y: (y: double) => double };
	toCanvas: { x: (x: double) => double; y: (y: double) => double };
}
class CanvasMouseNavigation implements Transform {
	scalex = 100;
	scaley = -100;
	offsetx = 0;
	offsety = 0;
	mousedown: boolean = false;
	mousestart = { x: 0, y: 0 };
	toReal = {
		x: (x: double) => (x - this.offsetx) / this.scalex,
		y: (y: double) => (y - this.offsety) / this.scaley
	}
	toCanvas = {
		x: (c: double) => c * this.scalex + this.offsetx,
		y: (c: double) => c * this.scaley + this.offsety
	}
	constructor(canvas: HTMLCanvasElement, transformActive: () => boolean, transformChanged: () => void) {
		this.offsetx = canvas.width / 3;
		this.offsety = 2 * canvas.height / 3;
		canvas.addEventListener('wheel', e => {
			if (e.deltaY === 0) return;
			var delta = e.deltaY / Math.abs(e.deltaY);
			this.scalex *= 1 - delta / 10;
			this.scaley *= 1 - delta / 10;
			transformChanged();
			e.preventDefault();
		});
		canvas.addEventListener('mousedown', e => {
			if (!transformActive()) return;
			this.mousedown = true;
			this.mousestart.x = e.pageX;
			this.mousestart.y = e.pageY;
		});
		canvas.addEventListener('mousemove', e => {
			if (!transformActive()) return;
			if (!this.mousedown) return;
			this.offsetx += e.pageX - this.mousestart.x;
			this.offsety += e.pageY - this.mousestart.y;
			this.mousestart.x = e.pageX;
			this.mousestart.y = e.pageY;
			transformChanged();
		});
		document.addEventListener('mouseup', e => this.mousedown = false);
	}

}