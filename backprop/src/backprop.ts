///<reference path='../lib/typings/react/react-global.d.ts' />
///<reference path='../lib/typings/jquery/jquery.d.ts' />
type int = number;
type double = number;

declare module convnetjs {
	interface Layer {
		type:string,
		num_neurons?:int,
		activation?: string
		out_sx?:int,
		out_sy?:int,
		out_depth?:int,
		num_classes?:int
	}
	class Net {
		makeLayers(layers: Layer[]);
		forward(inp: Vol): Vol;
	}
	interface Options {
		learning_rate: double,
		momentum: double,
		batch_size: int,
		l1_decay?: double,
		l2_decay: double
	}
	class Trainer {
		constructor(net: Net, options: Options);
		train(vol: Vol, label: int);
	}
	class Vol {
		constructor(sx: int, sy: int, depth: int);
		sx: int; sy: int; depth: int;
		w: Float64Array;
		dw: Float64Array;
	}
}

let layers = [
	{ type: 'input', out_sx: 1, out_sy: 1, out_depth: 2 },
	{ type: 'fc', num_neurons: 2, activation: 'tanh' },
	{ type: 'softmax', num_classes: 2 }
];
let net:convnetjs.Net, trainer:convnetjs.Trainer;
function initializeNet() {
	net = new convnetjs.Net();
	net.makeLayers(layers);

	trainer = new convnetjs.Trainer(net, {
		learning_rate: 0.01, momentum: 0.1, batch_size: 10, l2_decay: 0.001
	});
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

function update() {
	
	for (let val of data) {
		let x = new convnetjs.Vol(1, 1, 2);
		x.w[0] = val.x;
		x.w[1] = val.y;
		let stats = trainer.train(x, val.label);
	}
}

let canvas = <HTMLCanvasElement>document.querySelector("canvas");
let ctx = <CanvasRenderingContext2D>canvas.getContext('2d');

let w = 400, h = 400, blocks = 5, scalex = 50, scaley = -50;
let config = { steps: 100 };
let running = false;
let color = {
	redbg: "#f88",
	greenbg: "#8f8",
	red: "#f00",
	green: "#0f0"
}

function iteration() {
	for (let i = 0; i < 20; i++) update();
	draw();
	if (running) requestAnimationFrame(iteration);
}
function ctox(x) {
	return (x - w / 2) / scalex;
}
function ctoy(x) {
	return (x - h / 2) / scaley;
}
function xtoc(c) {
	return c * scalex + w / 2;
}
function ytoc(c) {
	return c * scaley + w / 2;
}
function draw() {
	ctx.fillStyle = "#ff0000";
	let netx = new convnetjs.Vol(1, 1, 2);
	for (let x = 0; x < w; x += blocks)
		for (let y = 0; y < h; y += blocks) {
			netx.w[0] = ctox(x);
			netx.w[1] = ctoy(y);
			let res = net.forward(netx);
			ctx.fillStyle = (res.w[0] > res.w[1]) ? color.redbg : color.greenbg;
			ctx.fillRect(x, y, w, h);
		}
	for (let val of data) {
		ctx.fillStyle = val.label ? color.green : color.red;
		ctx.beginPath();
		ctx.arc(xtoc(val.x), ytoc(val.y), 5, 0, 2 * Math.PI);
		ctx.fill();
	}
}

function run() {
	running = true;
	iteration();
}

function stop() {
	running = false;
}

function reset() {
	stop();
	initializeNet();
	iteration();
}
reset();