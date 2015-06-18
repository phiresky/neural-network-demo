var netx = new convnetjs.Vol(1, 1, 2, 0.0);
var stepNum = 0;
var running = false, runningId = -1;
var restartTimeout = -1;
var net;
var graph;
function loadTrainer() {
    net.learnRate = config.learningRate;
}
function initializeNet() {
    net = new Net.NeuralNet([2, 2, 1]);
    stepNum = 0;
    //drawGraph(net);
}
function drawGraph(net) {
    var id = 0;
    for (var lid = 0; lid < net.layers.length; lid++) {
        var layer = net.layers[lid];
        for (var nid = 0; nid < layer.length; nid++) {
            var neuron = layer[nid];
            graph.graph.addNode({
                id: "n" + neuron.id,
                label: "Neuron " + (nid + 1) + " in Layer " + (lid + 1),
                x: lid,
                y: nid,
                size: 1,
                color: "#000"
            });
        }
    }
    for (var _i = 0, _a = net.connections; _i < _a.length; _i++) {
        var conn = _a[_i];
        graph.graph.addEdge({
            id: 'e' + conn.inp.id + '-' + conn.out.id,
            source: "n" + conn.inp.id,
            target: "n" + conn.out.id,
            type: 'arrow'
        });
    }
    graph.refresh();
}
var data = [
    { x: 0, y: 0, label: 0 },
    { x: 0, y: 1, label: 1 },
    { x: 1, y: 0, label: 1 },
    { x: 1, y: 1, label: 0 }
];
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
}
var canvas = document.querySelector("canvas");
var ctx = canvas.getContext('2d');
var w = 400, h = 400, blocks = 10, scalex = 100, scaley = -100, offsetx = w / 3, offsety = 2 * h / 3;
var config = {
    stepsPerFrame: 50,
    learningRate: 0.01,
    activation: "sigmoid",
    showGradient: false,
    lossType: "svm"
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
}
var mousedown = false, mousestart = { x: 0, y: 0 };
$(document).ready(function () {
    //graph = new sigma("graph");
    $("#learningRate").slider({
        tooltip: 'always', min: 0.01, max: 1, step: 0.005, scale: "logarithmic", value: 0.01
    }).on('slide', function (e) { return $("#learningRateVal").text(e.value.toFixed(2)); });
    canvas.addEventListener('wheel', function (e) {
        var delta = e.deltaY / Math.abs(e.deltaY);
        scalex *= 1 - delta / 10;
        scaley *= 1 - delta / 10;
        if (!running)
            draw();
        return false;
    });
    canvas.addEventListener('mousedown', function (e) {
        mousedown = true;
        mousestart.x = e.pageX;
        mousestart.y = e.pageY;
    });
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