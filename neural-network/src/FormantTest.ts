declare var sim:Simulation;
var context = new AudioContext();
let analyser = context.createAnalyser();
let nav = <any>navigator;
let getUserMedia = nav.getUserMedia = nav.getUserMedia ||
	nav.webkitGetUserMedia ||
	nav.mozGetUserMedia;
getUserMedia = getUserMedia.bind(navigator);
$(() => {
	getUserMedia({ audio: true }, (stream: any) => {
		//convert audio stream to mediaStreamSource (node)
		console.log(analyser.fftSize);
		let microphone = (<any>context).createMediaStreamSource(stream);
		//connect microphone to analyser
		microphone.connect(analyser);
		requestAnimationFrame(testing);
	}, function(e: any) { throw e });

	function indexToFrequency(index: int) {
		return context.sampleRate / analyser.fftSize * index;
	}
	function testing() {
		var array = new Uint8Array(analyser.frequencyBinCount);
		analyser.getByteFrequencyData(array);
		let form = getFormants(array);
		let trafo = sim.config.originalBounds;
		let vals = Util.normalize(trafo, form[0].freq, form[1].freq);
		sim.config.data[0].input = vals;
		let tostr = (f:any) => `(${f.freq}Hz (${f.db.toFixed(2)}))`;
		$("h1").text(`${sim.config.outputLayer.names[Util.getMaxIndex(sim.net.getOutput(vals))]} F1: ${tostr(form[0])}, F2: ${tostr(form[1])}`);
		requestAnimationFrame(testing);
	}
	function getFormants(frequenciesBuffer: Uint8Array) {
		var size = frequenciesBuffer.length;
		let f1: int, f2: int;
		var maxf1 = -Infinity, maxf2 = -Infinity;
		var period = 16;
		var sma = simple_moving_averager(period);

		for(let i = 0; i < size; i++) {
			let value = frequenciesBuffer[i];
			var freq = indexToFrequency(i);
			var db = sma(value) / 256;
			if (freq < 1500) {
				if (db > maxf1) {
					maxf1 = db;
					f1 = freq;
				}
			}
			else if (freq < 5000) {
				if (db > maxf2) {
					maxf2 = db;
					f2 = freq;
				}
			}
		}
		return [{db: maxf1,freq: f1}, {db: maxf2, freq: f2}];
	}
	function simple_moving_averager(period: number) {
		let nums: number[] = [];
		return function(num: number) {
			nums.push(num);
			if (nums.length > period) nums.shift();
			var sum = 0;
			for (let i of nums) sum += i;
			return sum / nums.length;
		}
	}
});