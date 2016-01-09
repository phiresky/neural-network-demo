module Util {
	/**
	 * @param len array length
	 * @param supplier map from index to array element
	 */
	export function makeArray<T>(len: int, supplier: (i: int) => T): T[] {
		var arr = new Array<T>(len);
		for (let i = 0; i < len; i++) arr[i] = supplier(i);
		return arr;
	}
	/**
	 * return array index that has maximum value
	 * ≈ argmax function
	 */
	export function getMaxIndex(vals: double[]) {
		let max = vals[0], maxi = 0;
		for (let i = 1; i < vals.length; i++) {
			if (vals[i] > max) {
				max = vals[i];
				maxi = i;
			}
		}
		return maxi;
	}
	/**
	 * get an array that has a one at the specified position and zero everywhere else
	 */
	export function arrayWithOneAt(length: int, onePosition: int) {
		const output = new Array<int>(length);
		for (let i = 0; i < length; i++) {
			output[i] = i === onePosition ? 1 : 0;
		}
		return output;
	}
	/** AABB */
	export interface Bounds {
		minx: double, maxx: double, miny: double, maxy: double
	}
	/**
	 * get AABB of a set of points in the form of 2d-TrainingData 
	 */
	export function bounds2dTrainingsInput(data: TrainingData[]): Bounds {
		return {
			minx: Math.min(...data.map(d => d.input[0])),
			miny: Math.min(...data.map(d => d.input[1])),
			maxx: Math.max(...data.map(d => d.input[0])),
			maxy: Math.max(...data.map(d => d.input[1]))
		}
	}
	let _nextGaussian: double;
	/** port of https://docs.oracle.com/javase/7/docs/api/java/util/Random.html#nextGaussian() */
	export function randomGaussian(mean = 0, standardDeviation = 1) {
		if (_nextGaussian !== undefined) {
			var nextGaussian = _nextGaussian;
			_nextGaussian = undefined;
			return (nextGaussian * standardDeviation) + mean;
		} else {
			let v1: double, v2: double, s: double, multiplier: double;
			do {
				v1 = 2 * Math.random() - 1; // between -1 and 1
				v2 = 2 * Math.random() - 1; // between -1 and 1
				s = v1 * v1 + v2 * v2;
			} while (s >= 1 || s == 0);
			multiplier = Math.sqrt(-2 * Math.log(s) / s);
			_nextGaussian = v2 * multiplier;
			return (v1 * multiplier * standardDeviation) + mean;
		}
	};
	/**
	 * deep clone a configuration object
	 */
	export function cloneConfig(config: Configuration): Configuration {
		return $.extend(true, {}, config);
	}
	/**
	 * parse hex color (#ff0000) to number array
	 */
	export function parseColor(input: string): [number, number, number] {
		const m = input.match(/^#([0-9a-f]{6})$/i)[1];
		if (m) {
			return [
				parseInt(m.substr(0, 2), 16),
				parseInt(m.substr(2, 2), 16),
				parseInt(m.substr(4, 2), 16)
			];
		}
	}
	/**
	 * convert three numbers to hex color string
	 */
	export function printColor(c: [int, int, int]) {
		c = <any>c.map(x => x < 0 ? 0 : x > 255 ? 255 : x);
		return '#' + ("000000" + (c[0] << 16 | c[1] << 8 | c[2]).toString(16)).slice(-6);
	}
	/**
	 * parse url query parameters (...?a=b&x=y) to js object ({a:"b",x:"y"})
	 */
	export function parseUrlParameters(): { [name: string]: string } {
		if (!location.search) return {};
		var query: { [name: string]: string } = {};
		for (const p of location.search.slice(1).split('&')) {
			var b = p.split('=').map(c => c.replace(/\+/g, ' '));
			query[decodeURIComponent(b[0])] = decodeURIComponent(b[1]);
		}

		return query;
	}
	/** normalize 2d point in the given bounds to [0,1]×[0,1] */
	export function normalize(i: Bounds, x: double, y: double) {
		return [(x - i.minx) / (i.maxx - i.minx), (y - i.miny) / (i.maxy - i.miny)];
	}
	/** normalize x and y training data of given configuration into [0,1] */
	export function normalizeInputs(conf: Configuration) {
		if (conf.inputLayer.neuronCount !== 2) throw Error("can only normalize 2d data");
		const data = conf.data;
		const i = Util.bounds2dTrainingsInput(data);
		data.forEach(data => data.input = normalize(i, data.input[0], data.input[1]));
	}
	/** 
	 * download some text
	 * @param text file content
	 * @param name file name
	 * @param type mime type
	 */
	export function download(text: string, name: string, type: string = 'text/plain') {
		var a = document.createElement("a");
		var file = new Blob([text], { type: type });
		a.href = URL.createObjectURL(file);
		(<any>a).download = name;
		a.click();
	}
	/** sanitize string for output to CSV */
	export function csvSanitize(s: string) {
		s = s.replace(/"/g, '""');
		if (s.search(/("|,|\n)/g) >= 0)
			return `"${s}"`;
		else return s;
	}
	/** interpret n as a logarithmic scale */
	export function logScale(n: number) {
		return Math.log(n * 9 + 1) / Math.LN10;
	}
	/** interpret n as a exponential scale */
	export function expScale(n: number) {
		return (Math.pow(10, n) - 1) / 9;
	}

	export function stopEvent(e: Event) {
		e.preventDefault();
		e.stopPropagation();
	}

	interface Point { x: double, y: double }
	/** 
	 * Draws a line with a filled arrow head at its end.
	 *
	 * FOUND ON USENET  
	 * @param al Arrowhead length
	 * @param aw Arrowhead width
	 *
	 */
	export function drawArrow(g: CanvasRenderingContext2D, start: Point, end: Point,
		al: double, aw: double) {
		// Compute length of line
		const length = Math.sqrt((end.x - start.x) ** 2 + (end.y - start.y) ** 2);
		// Compute normalized line vector
		const x = (end.x - start.x) / length;
		const y = (end.y - start.y) / length;
		// Compute points for arrow head
		const base = { x: end.x - x * al, y: end.y - y * al };
		const back_top = { x: base.x - aw * y, y: base.y + aw * x };
		const back_bottom = { x: base.x + aw * y, y: base.y - aw * x };
		// Draw lines
		g.beginPath();
		g.moveTo(start.x, start.y);
		g.lineTo(end.x, end.y);
		g.stroke();
		g.moveTo(back_bottom.x, back_bottom.y);
		g.lineTo(end.x, end.y);
		g.lineTo(back_top.x, back_top.y);
		g.fill();
	}
	/** convert perceptron weights to a linear function */
	export function toLinearFunction([wx, wy, wbias]: [number, number, number], threshold = 0): (x: number) => number {
		//   w1*x + w2*y + w3 = thres
		// ⇒ w2*y = thres - w3 - w1*x
		// ⇒ y = (thres - w3 - w1*x) / w2
		if (wy === 0) wy = 0.00001;
		return x => (threshold - wbias - wx * x) / wy;
	}
}