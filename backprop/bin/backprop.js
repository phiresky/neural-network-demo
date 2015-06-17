var layers = [
    { type: 'input', out_sx: 1, out_sy: 1, out_depth: 2 },
    { type: 'fc', num_neurons: 2, activation: 'tanh' },
    { type: 'softmax', num_classes: 2 }
];
var net, trainer;
function initializeNet() {
    net = new convnetjs.Net();
    net.makeLayers(layers);
    trainer = new convnetjs.Trainer(net, {
        learning_rate: 0.01, momentum: 0.1, batch_size: 10, l2_decay: 0.001
    });
}
var data = [
    { x: 0, y: 0, label: 0 },
    { x: 0, y: 1, label: 1 },
    { x: 1, y: 0, label: 1 },
    { x: 1, y: 1, label: 0 }
];
function update() {
    for (var _i = 0; _i < data.length; _i++) {
        var val = data[_i];
        var x = new convnetjs.Vol(1, 1, 2);
        x.w[0] = val.x;
        x.w[1] = val.y;
        var stats = trainer.train(x, val.label);
    }
}
var canvas = document.querySelector("canvas");
var ctx = canvas.getContext('2d');
var w = 400, h = 400, blocks = 5, scalex = 50, scaley = -50;
var config = { steps: 100 };
var running = false;
var color = {
    redbg: "#f88",
    greenbg: "#8f8",
    red: "#f00",
    green: "#0f0"
};
function iteration() {
    for (var i = 0; i < 20; i++)
        update();
    draw();
    if (running)
        requestAnimationFrame(iteration);
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
    var netx = new convnetjs.Vol(1, 1, 2);
    for (var x = 0; x < w; x += blocks)
        for (var y = 0; y < h; y += blocks) {
            netx.w[0] = ctox(x);
            netx.w[1] = ctoy(y);
            var res = net.forward(netx);
            ctx.fillStyle = (res.w[0] > res.w[1]) ? color.redbg : color.greenbg;
            ctx.fillRect(x, y, w, h);
        }
    for (var _i = 0; _i < data.length; _i++) {
        var val = data[_i];
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
//# sourceMappingURL=backprop.js.map