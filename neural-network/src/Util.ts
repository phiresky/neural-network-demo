module Util {
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
	export function arrayWithOneAt(length: int, onePosition: int) {
		const output = new Array<int>(length);
		for (let i = 0; i < length; i++) {
			output[i] = i === onePosition ? 1 : 0;
		}
		return output;
	}
	export interface Bounds {
		minx:double, maxx:double, miny:double, maxy:double
	}
	export function bounds2dTrainingsInput(data: TrainingData[]):Bounds {
		return {
			minx: Math.min(...data.map(d => d.input[0])),
			miny: Math.min(...data.map(d => d.input[1])),
			maxx: Math.max(...data.map(d => d.input[0])),
			maxy: Math.max(...data.map(d => d.input[1]))
		}
	}
	let _nextGaussian: double;
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
	export function benchmark(fun: () => void) {
		const bef = Date.now();
		const r = fun();
		return Date.now() - bef;
	}
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
	export function printColor(c: [int, int, int]) {
		c = <any>c.map(x => x < 0 ? 0 : x > 255 ? 255 : x);
		return '#' + ("000000" + (c[0] << 16 | c[1] << 8 | c[2]).toString(16)).slice(-6);
	}
	export function parseUrlParameters():{[name:string]:string} {
		if(!location.search) return {};
		var query:{[name:string]:string} = {};
		for (const p of location.search.slice(1).split('&')) {
			var b = p.split('=').map(c => c.replace(/\+/g, ' '));
			query[decodeURIComponent(b[0])] = decodeURIComponent(b[1]);
		}

		return query;
	}
	export function normalize(i:Bounds, x:double, y:double) {
		return [(x-i.minx)/(i.maxx-i.minx),(y-i.miny)/(i.maxy-i.miny)];
	}
	export function normalizeInputs(conf:Configuration) {
		const data = conf.data;
		const i = Util.bounds2dTrainingsInput(data);
		data.forEach(data => data.input = normalize(i, data.input[0], data.input[1]));
		conf.originalBounds = i;
	}
	
	export function download(text:string, name:string, type:string = 'text/plain') {
		var a = document.createElement("a");
		var file = new Blob([text], { type: type });
		a.href = URL.createObjectURL(file);
		(<any>a).download = name;
		a.click();
	}
	export function csvSanitize(s:string) {
		s = s.replace(/"/g, '""');
		if(s.search(/("|,|\n)/g) >= 0)
			return `"${s}"`;
		else return s;
	}
	
	export function logScale(n:number) {
		return Math.log(n*9 + 1)/Math.LN10;
	}
	export function expScale(n:number) {
		return (Math.pow(10, n) - 1) / 9;
	}
	
	export function binarySearch(boolFn:(n:number) => boolean, min: number, max: number, epsilon = 0): number {
		var mid = ((max + min) / 2)|0;
		if(Math.abs(mid - min) < epsilon) return mid;
		if(boolFn(mid)) return binarySearch(boolFn, mid, max);
		else return binarySearch(boolFn, min, mid);
	}
	
	export function stopEvent(e:Event) {
		e.preventDefault();
		e.stopPropagation();
	}
}