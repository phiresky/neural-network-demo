///<reference path='../lib/typings/react/react-global.d.ts' />
///<reference path='../lib/typings/jquery/jquery.d.ts' />
///<reference path='convnetjs.d.ts' />
///<reference path='Net.ts' />
type int = number;
type double = number;

let stepNum = 0;
let running = false, runningId = -1;
let restartTimeout = -1;
let net: Net.NeuralNet;
let graph:any, graphData:any;
declare var vis: any;
interface Data {
	x: double; y: double; label: int;
}
let data: Data[] = [
	{ x: 0, y: 0, label: 0 },
	{ x: 0, y: 1, label: 1 },
	{ x: 1, y: 0, label: 1 },
	{ x: 1, y: 1, label: 0 }
];

function loadTrainer() {
	net.learnRate = config.learningRate;
}
function initializeNet() {
	//let cache = [0.18576880730688572,-0.12869677506387234,0.08548374730162323,-0.19820863520726562,-0.09532690420746803,-0.3415223266929388,-0.309354952769354,-0.157513455953449];
	//let cache = [-0.04884958150796592,-0.3569231238216162,0.11143312812782824,0.43614205135963857,0.3078767384868115,-0.22759653301909566,0.09250503336079419,0.3279339636210352];
	net = new Net.NeuralNet([2, 2, 1],["x","y"]);
	console.log("net:"+JSON.stringify(net.connections.map(c => c.weight)));
	stepNum = 0;
	createGraph(net);
}
function randomizeData() {
	let count = 4;
	for(let i = 0; i < count; i++) {
		data[i] = {x:Math.random()*2,y:Math.random()*2,label:+(Math.random()>0.5)};
	}
	draw();
}
function createGraph(net: Net.NeuralNet) {
	let nodes: any[] = [], edges: any[] = [];
	let id = 0;
	for (let lid = 0; lid < net.layers.length; lid++) {
		let layer = net.layers[lid];
		for (let nid = 0; nid < layer.length; nid++) {
			let neuron = layer[nid];
			let type = 'Hidden Neuron '+(nid+1);
			let color = '#000';
			if (neuron instanceof Net.InputNeuron) {
				type = 'Input: '+neuron.name;
				color = '#008';
			} if (neuron instanceof Net.OutputNeuron) {
				type = 'Output Neuron ' + (nid+1);
				color = '#800';
			}
			nodes.push({
				id: neuron.id,
				label: `${type}`,
				level: lid,
				color: color
				/*x:lid/10,
				y:nid/10,
				size:1,
				color:"#000"*/
			});
		}
	}
	for (let conn of net.connections) {
		edges.push({
			id: conn.inp.id * net.connections.length + conn.out.id,
			from: conn.inp.id,
			to: conn.out.id,
			arrows:'to',
			label: conn.weight.toFixed(2),
			//size:0.01,
			//type:'arrow'
		})
	}
	graphData = { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) };
	let options = {
		nodes: { shape: 'dot' },
		edges: { smooth: {type: 'curvedCW',roundness:0.25}},
		layout: { hierarchical: { direction: "LR" } }
	}
	graph = new vis.Network($('#graph')[0], graphData, options);
}
function drawGraph() {
	for (let conn of net.connections) {
		graphData.edges.update({
			id: conn.inp.id * net.connections.length + conn.out.id,
			label: conn.weight.toFixed(2)
		})
	}
}

function step() {
	stepNum++;
	for (let val of data) {
		let stats = net.train([val.x, val.y], [val.label]);
	}
	let correct = 0;
	for (let val of data) {
		let res = net.getOutput([val.x, val.y]);
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
	} else {
		if(restartTimeout != -1) {
			clearTimeout(restartTimeout);
			restartTimeout = -1;
			$("#status>h3").hide();
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
	//lossType: "svm"
};

let color = {
	bg: ["#f88", "#8f8"],
	fg: ["#f00", "#0f0"]
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
			let res = net.getOutput([ctox(x), ctoy(y)]);

			if (config.showGradient) {
				let gre = (res[0] * 256) | 0;
				let red = ((1 - res[0]) * 256) | 0;
				ctx.fillStyle = "rgb(" + [red, gre, 0] + ")";
			}
			else ctx.fillStyle = color.bg[(res[0] + 0.5) | 0];
			ctx.fillRect(x, y, w, h);
		}
}
function drawData() {
	ctx.strokeStyle = "#000";
	for (let val of data) {
		ctx.fillStyle = color.fg[val.label | 0];
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
	drawGraph();
}

function run() {
	if (running) return;
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
	Net.setLinearity(config.activation);
}
let mousedown = false, mousestart = { x: 0, y: 0 };
function resizeCanvas() {
	w = $("#neuralOutputCanvas").width();
	h = $("#neuralOutputCanvas").height();
	canvas.width = w; canvas.height = h;
}
$(document).ready(function() {
	resizeCanvas();
	(<any>$("#learningRate")).slider({
		min: 0.01, max: 1, step: 0.005, scale: "logarithmic", value: 0.05
	}).on('slide', (e: any) => $("#learningRateVal").text(e.value.toFixed(2)));
	canvas.addEventListener('wheel', e => {
		var delta = e.deltaY / Math.abs(e.deltaY);
		scalex *= 1 - delta / 10;
		scaley *= 1 - delta / 10;
		if (!running) draw();
		e.preventDefault();
	});
	canvas.addEventListener('mousedown', e=> {
		mousedown = true;
		mousestart.x = e.pageX;
		mousestart.y = e.pageY;
	});
	window.addEventListener('resize', resizeCanvas);
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
