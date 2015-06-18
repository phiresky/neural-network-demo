///<reference path='../lib/typings/react/react-global.d.ts' />
///<reference path='../lib/typings/jquery/jquery.d.ts' />
///<reference path='convnetjs.d.ts' />
///<reference path='Net.ts' />
type int = number;
type double = number;


let netx = new convnetjs.Vol(1, 1, 2, 0.0);
let stepNum = 0;
let running = false, runningId = -1;
let restartTimeout = -1;
let net:Net.NeuralNet;

function loadTrainer() {
	net.learnRate = config.learningRate;
}
function initializeNet() {
	net = new Net.NeuralNet([2,2,1]);
	stepNum = 0;
}
interface Data {
	x: double; y: double; label: int;
}
let data: Data[] = [
	{ x: 0, y: 0, label: 0 },
	{ x: 0, y: 1, label: 1 },
	{ x: 1, y: 0, label: 1 },
	{ x: 1, y: 1, label: 0 }
];


function step() {
	stepNum++;
	for (let val of data) {
		let stats = net.train([val.x,val.y], [val.label]);
	}
	let correct = 0;
	for (let val of data) {
		let res = net.getOutput([val.x,val.y]);
		let label = (res[0] > 0.5) ? 1 : 0;
		if (val.label == label) correct++;
	}
	document.getElementById('statusIteration').textContent = stepNum.toString();
	document.getElementById('statusCorrect').textContent = `${correct}/${data.length}`;
	if (correct == data.length) {
		$("#status>h3").show();
		if (running && restartTimeout == -1) {
			restartTimeout = setTimeout(() => {
				stop();
				$("#status>h3").hide();
				console.log('hidden');
				restartTimeout = -1;
				setTimeout(() => { reset(); run(); }, 1000);
			}, 3000);
		}
	}
}

let canvas = <HTMLCanvasElement>document.querySelector("canvas");
let ctx = <CanvasRenderingContext2D>canvas.getContext('2d');


let w = 400, h = 400, blocks = 10,
	scalex = 100, scaley = -100, offsetx = w / 3, offsety = 2 * h / 3;
var config = {
	stepsPerFrame: 50,
	learningRate: 0.01,
	activation: "sigmoid",
	showGradient: false,
	lossType: "svm"
};

let color = {
	bg: ["#f88","#8f8"],
	fg:["#f00","#0f0"]
}

function animationStep() {
	for (let i = 0; i < config.stepsPerFrame; i++) step();
	draw();
	if (running) runningId = requestAnimationFrame(animationStep);
}
function ctox(x: double) {
	return (x - offsetx) / scalex;
}
function ctoy(x: double) {
	return (x - offsety) / scaley;
}
function xtoc(c: double) {
	return c * scalex + offsetx;
}
function ytoc(c: double) {
	return c * scaley + offsety;
}

function drawBackground() {

	for (let x = 0; x < w; x += blocks)
		for (let y = 0; y < h; y += blocks) {
			let res = net.getOutput([ctox(x),ctoy(y)]);

			if (config.showGradient) {
				let red = (res[0] * 256) | 0;
				let gre = ((1-res[0]) * 256) | 0;
				ctx.fillStyle = "rgb(" + [red, gre, 0] + ")";
			}
			else ctx.fillStyle = color.bg[(res[0] + 0.5)|0];
			ctx.fillRect(x, y, w, h);
		}
}
function drawData() {
	ctx.strokeStyle = "#000";
	for (let val of data) {
		ctx.fillStyle = color.fg[val.label|0];
		ctx.beginPath();
		ctx.arc(xtoc(val.x), ytoc(val.y), 5, 0, 2 * Math.PI);
		ctx.fill();
		ctx.arc(xtoc(val.x), ytoc(val.y), 5, 0, 2 * Math.PI);
		ctx.stroke();
	}
}
function drawCoordinateSystem() {
	let marklen = 0.2;
	ctx.strokeStyle = "#000";
	ctx.fillStyle = "#000";
	ctx.textBaseline = "middle";
	ctx.textAlign = "center";
	ctx.font = "20px monospace";
	ctx.beginPath();

	ctx.moveTo(xtoc(0), 0);
	ctx.lineTo(xtoc(0), h);

	ctx.moveTo(xtoc(-marklen / 2), ytoc(1));
	ctx.lineTo(xtoc(marklen / 2), ytoc(1));
	ctx.fillText("1", xtoc(-marklen), ytoc(1));

	ctx.moveTo(0, ytoc(0));
	ctx.lineTo(w, ytoc(0));

	ctx.moveTo(xtoc(1), ytoc(-marklen / 2));
	ctx.lineTo(xtoc(1), ytoc(marklen / 2));
	ctx.fillText("1", xtoc(1), ytoc(-marklen));
	ctx.stroke();
}
function draw() {
	ctx.fillStyle = "#ff0000";
	drawBackground();
	drawCoordinateSystem();
	drawData();
}

function run() {
	if(running) return;
	running = true;
	animationStep();
}

function stop() {
	clearTimeout(restartTimeout);
	restartTimeout = -1;
	running = false;
	cancelAnimationFrame(runningId);
}

function reset() {
	stop();
	loadConfig();
	initializeNet();
	draw();
}

function iterations() {
	var count = +$("#iterations").val();
	for (var i = 0; i < count; i++)
		step();
	draw();
}

function loadConfig() {
	let config = (<any>window).config;
	for (let conf in config) {
		let ele = <HTMLInputElement>document.getElementById(conf);
		if (ele.type == 'checkbox') config[conf] = ele.checked;
		else config[conf] = ele.value;
	}
}
let mousedown = false, mousestart = { x: 0, y: 0 };
$(document).ready(function() {
	(<any>$("#learningRate")).slider({
		tooltip: 'always', min: 0.01, max: 1, step: 0.005, scale: "logarithmic", value: 0.01
	}).on('slide', (e: any) => $("#learningRateVal").text(e.value.toFixed(2)));
	canvas.addEventListener('wheel', e=> {
		var delta = e.deltaY / Math.abs(e.deltaY);
		scalex *= 1 - delta / 10;
		scaley *= 1 - delta / 10;
		if (!running) draw();
		return false;
	});
	canvas.addEventListener('mousedown', e=> {
		mousedown = true;
		mousestart.x = e.pageX;
		mousestart.y = e.pageY;
	});
	document.addEventListener('mouseup', e=> mousedown = false);
	canvas.addEventListener('mousemove', e=> {
		if (!mousedown) return;
		offsetx += e.pageX - mousestart.x;
		offsety += e.pageY - mousestart.y;
		mousestart.x = e.pageX;
		mousestart.y = e.pageY;
		if (!running) draw();
	})
	reset();
	run();
})
