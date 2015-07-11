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
		let output = new Array<int>(length);
		for (let i = 0; i < length; i++) {
			output[i] = i === onePosition ? 1 : 0;
		}
		return output;
	}
	export function min(input:double[]) {
		return input.reduce((a,b) => Math.min(a,b), Infinity);
	}
	export function max(input:double[]) {
		return input.reduce((a,b) => Math.max(a,b), -Infinity);
	}
	export function bounds2dTrainingsInput(data:TrainingData[]) {
		return {
			minx : Util.min(data.map(d => d.input[0])),
			miny : Util.min(data.map(d => d.input[1])),
			maxx : Util.max(data.map(d => d.input[0])),
			maxy : Util.max(data.map(d => d.input[1]))
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
	export function benchmark(fun:()=>void) {
		let bef = Date.now();
		let r = fun();
		return Date.now() - bef;
	}
}