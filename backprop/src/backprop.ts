///<reference path='../lib/typings/react/react-global.d.ts' />
///<reference path='../lib/typings/jquery/jquery.d.ts' />
///<reference path='convnetjs.d.ts' />
type int = number;
type double = number;

let net: convnetjs.Net, trainer: convnetjs.Trainer;
function loadTrainer() {
	trainer = new convnetjs.Trainer(net, {
		learning_rate: config.learningRate, momentum: 0.9, batch_size: 1, l2_decay: 0.001
	});
}
function initializeNet() {
	net = new convnetjs.Net();
	let layers = [
		{ type: 'input', out_sx: 1, out_sy: 1, out_depth: 2 },
		{ type: 'fc', num_neurons: 2, activation: config.activation },
		{ type: config.lossType, num_classes: 2 }
	];
	net.makeLayers(layers);
	//net.fromJSON({"layers":[{"out_depth":2,"out_sx":1,"out_sy":1,"layer_type":"input"},{"out_depth":2,"out_sx":1,"out_sy":1,"layer_type":"fc","num_inputs":2,"l1_decay_mul":0,"l2_decay_mul":1,"filters":[{"sx":1,"sy":1,"depth":2,"w":{"0":2.0155538859555944,"1":-1.1242570376625403}},{"sx":1,"sy":1,"depth":2,"w":{"0":-1.997246234051715,"1":-1.2874173363849695}}],"biases":{"sx":1,"sy":1,"depth":2,"w":{"0":0.9125948164444501,"1":0.15075782384585915}}},{"out_depth":2,"out_sx":1,"out_sy":1,"layer_type":"tanh"},{"out_depth":2,"out_sx":1,"out_sy":1,"layer_type":"fc","num_inputs":2,"l1_decay_mul":0,"l2_decay_mul":1,"filters":[{"sx":1,"sy":1,"depth":2,"w":{"0":1.407302175440426,"1":1.3603279523598015}},{"sx":1,"sy":1,"depth":2,"w":{"0":-1.0146332894945678,"1":-1.2478560570752113}}],"biases":{"sx":1,"sy":1,"depth":2,"w":{"0":0.10260196288430118,"1":-0.10260196288430096}}},{"out_depth":2,"out_sx":1,"out_sy":1,"layer_type":"softmax","num_inputs":2}]});
	
	loadTrainer();
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
for(let p of data) {p.x+=Math.random()*0.01; p.y+=Math.random()*0.01;}


let netx = new convnetjs.Vol(1, 1, 2, 0.0);
let stepNum = 0;

function step() {
	stepNum++;
	for (let val of data) {
		netx.w[0] = val.x;
		netx.w[1] = val.y;
		let stats = trainer.train(netx, val.label);
	}
	let correct = 0;
	for (let val of data) {
		netx.w[0] = val.x;
		netx.w[1] = val.y;
		let res = net.forward(netx);
		let label = (res.w[0] > res.w[1]) ? 0 : 1;
		if(val.label == label) correct++;
	}
	document.getElementById('statusIteration').textContent = stepNum;
	document.getElementById('statusCorrect').textContent = `${correct}/${data.length}`;
	if(correct == data.length) {
		//statusPre.textContent += "\nAll correct. Restarting in 3s";
		if(running && restartTimeout == -1) 
			restartTimeout = setTimeout(()=>{
				restartTimeout = -1;
				stop();
				setTimeout(()=>{reset();run();}, 1000);
			}, 3000);
	}
}

let canvas = <HTMLCanvasElement>document.querySelector("canvas");
let ctx = <CanvasRenderingContext2D>canvas.getContext('2d');
let restartTimeout = -1;

let w = 400, h = 400, blocks = 5, scalex = 100, scaley = -100, offsetx = w/2, offsety = h/2;
var config = {
	stepsPerFrame: 50,
	learningRate: 0.01,
	activation: "sigmoid",
	showGradient: false,
	lossType: "svm"
};
let running = false;
let color = {
	redbg: "#f88",
	greenbg: "#8f8",
	red: "#f00",
	green: "#0f0"
}

function animationStep() {
	for (let i = 0; i < config.stepsPerFrame; i++) step();
	draw();
	if (running) requestAnimationFrame(animationStep);
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
			netx.w[0] = ctox(x);
			netx.w[1] = ctoy(y);
			let res = net.forward(netx);
			let red = (res.w[0]*256)|0;
			let gre = (res.w[1]*256)|0;
			
			if(config.showGradient) ctx.fillStyle = "rgb("+[red,gre,0]+")";
			else ctx.fillStyle = (res.w[0] > res.w[1]) ? color.redbg : color.greenbg;
			ctx.fillRect(x, y, w, h);
		}
}
function drawData() {
	ctx.strokeStyle = "#000";
	for (let val of data) {
		ctx.fillStyle = val.label ? color.green : color.red;
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
	
	ctx.moveTo(xtoc(-marklen/2), ytoc(1));
	ctx.lineTo(xtoc(marklen/2), ytoc(1));
	ctx.fillText("1", xtoc(-marklen), ytoc(1));
	
	ctx.moveTo(0, ytoc(0));
	ctx.lineTo(w, ytoc(0));
	
	ctx.moveTo(xtoc(1), ytoc(-marklen/2));
	ctx.lineTo(xtoc(1), ytoc(marklen/2));
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
	running = true;
	animationStep();
}

function stop() {
	clearTimeout(restartTimeout);
	restartTimeout = -1;
	running = false;
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
	let config = <any>window.config;
	for (let conf in config) {
		let ele = <HTMLInputElement>document.getElementById(conf);
		if(ele.type == 'checkbox') config[conf] = ele.checked;
		else config[conf] = ele.value;
	}
}
let mousedown = false, mousestart = {x:0,y:0};
$(document).ready(function() {
	$("#learningRate").slider({
		tooltip:'always', min:0.01,max:1,step:0.005,scale:"logarithmic",value:0.01
	}).on('slide',(e:any) => $("#learningRateVal").text(e.value.toFixed(2)));
	canvas.addEventListener('mousewheel', e=>{
		var delta = e.wheelDelta/Math.abs(e.wheelDelta);
		scalex *= 1+delta/10;
		scaley *=1+delta/10;
		if(!running) draw();
		return false;
	});
	canvas.addEventListener('mousedown',e=>{
		mousedown = true;
		mousestart.x = e.pageX;
		mousestart.y = e.pageY;
	});
	document.addEventListener('mouseup',e=>mousedown = false);
	canvas.addEventListener('mousemove', e=>{
		if(!mousedown) return;
		offsetx += e.pageX - mousestart.x;
		offsety += e.pageY - mousestart.y;
		mousestart.x = e.pageX;
		mousestart.y = e.pageY;
		if(!running) draw();
	})
	reset();
})
