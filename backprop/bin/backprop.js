// binds a js var to a html value
var Binding = (function () {
    function Binding(bound) {
        this.bound = bound;
    }
    Object.defineProperty(Binding.prototype, "value", {
        get: function () {
            return bound.value;
        },
        enumerable: true,
        configurable: true
    });
    return Binding;
})();
var net, trainer;
function loadTrainer() {
    trainer = new convnetjs.Trainer(net, {
        learning_rate: config.learningRate, momentum: 0.9, batch_size: 1, l2_decay: 0.001
    });
}
function initializeNet() {
    net = new convnetjs.Net();
    var layers = [
        { type: 'input', out_sx: 1, out_sy: 1, out_depth: 2 },
        { type: 'fc', num_neurons: 2, activation: config.activation },
        { type: config.lossType, num_classes: 2 }
    ];
    net.makeLayers(layers);
    //net.fromJSON({"layers":[{"out_depth":2,"out_sx":1,"out_sy":1,"layer_type":"input"},{"out_depth":2,"out_sx":1,"out_sy":1,"layer_type":"fc","num_inputs":2,"l1_decay_mul":0,"l2_decay_mul":1,"filters":[{"sx":1,"sy":1,"depth":2,"w":{"0":2.0155538859555944,"1":-1.1242570376625403}},{"sx":1,"sy":1,"depth":2,"w":{"0":-1.997246234051715,"1":-1.2874173363849695}}],"biases":{"sx":1,"sy":1,"depth":2,"w":{"0":0.9125948164444501,"1":0.15075782384585915}}},{"out_depth":2,"out_sx":1,"out_sy":1,"layer_type":"tanh"},{"out_depth":2,"out_sx":1,"out_sy":1,"layer_type":"fc","num_inputs":2,"l1_decay_mul":0,"l2_decay_mul":1,"filters":[{"sx":1,"sy":1,"depth":2,"w":{"0":1.407302175440426,"1":1.3603279523598015}},{"sx":1,"sy":1,"depth":2,"w":{"0":-1.0146332894945678,"1":-1.2478560570752113}}],"biases":{"sx":1,"sy":1,"depth":2,"w":{"0":0.10260196288430118,"1":-0.10260196288430096}}},{"out_depth":2,"out_sx":1,"out_sy":1,"layer_type":"softmax","num_inputs":2}]});
    loadTrainer();
    stepNum = 0;
}
var data = [
    { x: 0, y: 0, label: 0 },
    { x: 0, y: 1, label: 1 },
    { x: 1, y: 0, label: 1 },
    { x: 1, y: 1, label: 0 }
];
for (var _i = 0; _i < data.length; _i++) {
    var p = data[_i];
    p.x += Math.random() * 0.01;
    p.y += Math.random() * 0.01;
}
var netx = new convnetjs.Vol(1, 1, 2, 0.0);
var stepNum = 0;
function step() {
    stepNum++;
    for (var _i = 0; _i < data.length; _i++) {
        var val = data[_i];
        netx.w[0] = val.x;
        netx.w[1] = val.y;
        var stats = trainer.train(netx, val.label);
    }
    var correct = 0;
    for (var _a = 0; _a < data.length; _a++) {
        var val = data[_a];
        netx.w[0] = val.x;
        netx.w[1] = val.y;
        var res = net.forward(netx);
        var label = (res.w[0] > res.w[1]) ? 0 : 1;
        if (val.label == label)
            correct++;
    }
    document.getElementById('statusIteration').textContent = stepNum;
    document.getElementById('statusCorrect').textContent = correct + "/" + data.length;
    if (correct == data.length) {
        //statusPre.textContent += "\nAll correct. Restarting in 3s";
        if (running && restartTimeout == -1)
            restartTimeout = setTimeout(function () {
                restartTimeout = -1;
                stop();
                setTimeout(function () { reset(); run(); }, 1000);
            }, 3000);
    }
}
var canvas = document.querySelector("canvas");
var ctx = canvas.getContext('2d');
var restartTimeout = -1;
var w = 400, h = 400, blocks = 5, scalex = 100, scaley = -100, offsetx = w / 2, offsety = h / 2;
var config = {
    stepsPerFrame: 50,
    learningRate: 0.01,
    activation: "sigmoid",
    showGradient: false,
    lossType: "svm"
};
var running = false;
var color = {
    redbg: "#f88",
    greenbg: "#8f8",
    red: "#f00",
    green: "#0f0"
};
function animationStep() {
    for (var i = 0; i < config.stepsPerFrame; i++)
        step();
    draw();
    if (running)
        requestAnimationFrame(animationStep);
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
            netx.w[0] = ctox(x);
            netx.w[1] = ctoy(y);
            var res = net.forward(netx);
            var red = (res.w[0] * 256) | 0;
            var gre = (res.w[1] * 256) | 0;
            if (config.showGradient)
                ctx.fillStyle = "rgb(" + [red, gre, 0] + ")";
            else
                ctx.fillStyle = (res.w[0] > res.w[1]) ? color.redbg : color.greenbg;
            ctx.fillRect(x, y, w, h);
        }
}
function drawData() {
    ctx.strokeStyle = "#000";
    for (var _i = 0; _i < data.length; _i++) {
        var val = data[_i];
        ctx.fillStyle = val.label ? color.green : color.red;
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
    $("#learningRate").slider({
        tooltip: 'always', min: 0.01, max: 1, step: 0.005, scale: "logarithmic", value: 0.01
    }).on('slide', function (e) { return $("#learningRateVal").text(e.value.toFixed(2)); });
    canvas.addEventListener('mousewheel', function (e) {
        var delta = e.wheelDelta / Math.abs(e.wheelDelta);
        scalex *= 1 + delta / 10;
        scaley *= 1 + delta / 10;
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
});
//# sourceMappingURL=backprop.js.map