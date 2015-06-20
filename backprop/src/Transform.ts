type double = number; type int = number;
interface Transform {
	ctox(x: double): double;
	ctoy(x: double): double;
	xtoc(c: double): double;
	ytoc(c: double): double;
}
class CanvasMouseNavigation implements Transform {
	scalex = 100;
	scaley = -100;
	offsetx = 0;
	offsety = 0;
	mousedown: boolean = false;
	mousestart = { x: 0, y: 0 };
	constructor(canvas: HTMLCanvasElement, transformChanged: () => void) {
		this.offsetx = canvas.width / 3;
		this.offsety = 2 * canvas.height / 3;
		canvas.addEventListener('wheel', e => {
			var delta = e.deltaY / Math.abs(e.deltaY);
			this.scalex *= 1 - delta / 10;
			this.scaley *= 1 - delta / 10;
			transformChanged();
			e.preventDefault();
		});
		canvas.addEventListener('mousedown', e => {
			this.mousedown = true;
			this.mousestart.x = e.pageX;
			this.mousestart.y = e.pageY;
		});
		canvas.addEventListener('mousemove', e => {
			if (!this.mousedown) return;
			this.offsetx += e.pageX - this.mousestart.x;
			this.offsety += e.pageY - this.mousestart.y;
			this.mousestart.x = e.pageX;
			this.mousestart.y = e.pageY;
			transformChanged();
		});
		document.addEventListener('mouseup', e => this.mousedown = false);
	}
	ctox(x: double) {
		return (x - this.offsetx) / this.scalex;
	}
	ctoy(x: double) {
		return (x - this.offsety) / this.scaley;
	}
	xtoc(c: double) {
		return c * this.scalex + this.offsetx;
	}
	ytoc(c: double) {
		return c * this.scaley + this.offsety;
	}
}