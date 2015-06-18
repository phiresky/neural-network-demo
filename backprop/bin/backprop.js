var stepNum = 0;
var running = false, runningId = -1;
var restartTimeout = -1;
var net;
var graph, graphData;
var data = [
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
    net = new Net.NeuralNet([2, 2, 1], ["x", "y"]);
    console.log("net:" + JSON.stringify(net.connections.map(function (c) { return c.weight; })));
    stepNum = 0;
    createGraph(net);
}
function randomizeData() {
    var count = 4;
    for (var i = 0; i < count; i++) {
        data[i] = { x: Math.random() * 2, y: Math.random() * 2, label: +(Math.random() > 0.5) };
    }
    draw();
}
function createGraph(net) {
    var nodes = [], edges = [];
    var id = 0;
    for (var lid = 0; lid < net.layers.length; lid++) {
        var layer = net.layers[lid];
        for (var nid = 0; nid < layer.length; nid++) {
            var neuron = layer[nid];
            var type = 'Hidden Neuron ' + (nid + 1);
            var color_1 = '#000';
            if (neuron instanceof Net.InputNeuron) {
                type = 'Input: ' + neuron.name;
                color_1 = '#008';
            }
            if (neuron instanceof Net.OutputNeuron) {
                type = 'Output Neuron ' + (nid + 1);
                color_1 = '#800';
            }
            nodes.push({
                id: neuron.id,
                label: "" + type,
                level: lid,
                color: color_1
            });
        }
    }
    for (var _i = 0, _a = net.connections; _i < _a.length; _i++) {
        var conn = _a[_i];
        edges.push({
            id: conn.inp.id * net.connections.length + conn.out.id,
            from: conn.inp.id,
            to: conn.out.id,
            arrows: 'to',
            label: conn.weight.toFixed(2),
        });
    }
    graphData = { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) };
    var options = {
        nodes: { shape: 'dot' },
        edges: { smooth: { type: 'curvedCW', roundness: 0.25 } },
        layout: { hierarchical: { direction: "LR" } }
    };
    graph = new vis.Network($('#graph')[0], graphData, options);
}
function drawGraph() {
    for (var _i = 0, _a = net.connections; _i < _a.length; _i++) {
        var conn = _a[_i];
        graphData.edges.update({
            id: conn.inp.id * net.connections.length + conn.out.id,
            label: conn.weight.toFixed(2)
        });
    }
}
function step() {
    stepNum++;
    for (var _i = 0; _i < data.length; _i++) {
        var val = data[_i];
        var stats = net.train([val.x, val.y], [val.label]);
    }
    var correct = 0;
    for (var _a = 0; _a < data.length; _a++) {
        var val = data[_a];
        var res = net.getOutput([val.x, val.y]);
        var label = (res[0] > 0.5) ? 1 : 0;
        if (val.label == label)
            correct++;
    }
    document.getElementById('statusIteration').textContent = stepNum.toString();
    document.getElementById('statusCorrect').textContent = correct + "/" + data.length;
    if (correct == data.length) {
        $("#status>h3").show();
        if (running && restartTimeout == -1) {
            restartTimeout = setTimeout(function () {
                stop();
                $("#status>h3").hide();
                console.log('hidden');
                restartTimeout = -1;
                setTimeout(function () { reset(); run(); }, 1000);
            }, 3000);
        }
    }
    else {
        if (restartTimeout != -1) {
            clearTimeout(restartTimeout);
            restartTimeout = -1;
            $("#status>h3").hide();
        }
    }
}
var canvas = document.querySelector("canvas");
var ctx = canvas.getContext('2d');
var w = 400, h = 400, blocks = 10, scalex = 100, scaley = -100, offsetx = w / 3, offsety = 2 * h / 3;
var config = {
    stepsPerFrame: 50,
    learningRate: 0.01,
    activation: "sigmoid",
    showGradient: false,
};
var color = {
    bg: ["#f88", "#8f8"],
    fg: ["#f00", "#0f0"]
};
function animationStep() {
    for (var i = 0; i < config.stepsPerFrame; i++)
        step();
    draw();
    if (running)
        runningId = requestAnimationFrame(animationStep);
}
function ctox(x) {
    return (x - offsetx) / scalex;
}
function ctoy(x) {
    return (x - offsety) / scaley;
}
function xtoc(c) {
    return c * scalex + offsetx;
}
function ytoc(c) {
    return c * scaley + offsety;
}
function drawBackground() {
    for (var x = 0; x < w; x += blocks)
        for (var y = 0; y < h; y += blocks) {
            var res = net.getOutput([ctox(x), ctoy(y)]);
            if (config.showGradient) {
                var gre = (res[0] * 256) | 0;
                var red = ((1 - res[0]) * 256) | 0;
                ctx.fillStyle = "rgb(" + [red, gre, 0] + ")";
            }
            else
                ctx.fillStyle = color.bg[(res[0] + 0.5) | 0];
            ctx.fillRect(x, y, w, h);
        }
}
function drawData() {
    ctx.strokeStyle = "#000";
    for (var _i = 0; _i < data.length; _i++) {
        var val = data[_i];
        ctx.fillStyle = color.fg[val.label | 0];
        ctx.beginPath();
        ctx.arc(xtoc(val.x), ytoc(val.y), 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.arc(xtoc(val.x), ytoc(val.y), 5, 0, 2 * Math.PI);
        ctx.stroke();
    }
}
function drawCoordinateSystem() {
    var marklen = 0.2;
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
    if (running)
        return;
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
    stop();
    var count = +$("#iterations").val();
    for (var i = 0; i < count; i++)
        step();
    draw();
}
function loadConfig() {
    var config = window.config;
    for (var conf in config) {
        var ele = document.getElementById(conf);
        if (ele.type == 'checkbox')
            config[conf] = ele.checked;
        else
            config[conf] = ele.value;
    }
    Net.setLinearity(config.activation);
}
var mousedown = false, mousestart = { x: 0, y: 0 };
function resizeCanvas() {
    w = $("#neuralOutputCanvas").width();
    h = $("#neuralOutputCanvas").height();
    canvas.width = w;
    canvas.height = h;
}
$(document).ready(function () {
    resizeCanvas();
    $("#learningRate").slider({
        min: 0.01, max: 1, step: 0.005, scale: "logarithmic", value: 0.05
    }).on('slide', function (e) { return $("#learningRateVal").text(e.value.toFixed(2)); });
    canvas.addEventListener('wheel', function (e) {
        var delta = e.deltaY / Math.abs(e.deltaY);
        scalex *= 1 - delta / 10;
        scaley *= 1 - delta / 10;
        if (!running)
            draw();
        e.preventDefault();
    });
    canvas.addEventListener('mousedown', function (e) {
        mousedown = true;
        mousestart.x = e.pageX;
        mousestart.y = e.pageY;
    });
    window.addEventListener('resize', resizeCanvas);
    document.addEventListener('mouseup', function (e) { return mousedown = false; });
    canvas.addEventListener('mousemove', function (e) {
        if (!mousedown)
            return;
        offsetx += e.pageX - mousestart.x;
        offsety += e.pageY - mousestart.y;
        mousestart.x = e.pageX;
        mousestart.y = e.pageY;
        if (!running)
            draw();
    });
    reset();
    run();
});
//# sourceMappingURL=backprop.js.map