var __extends = this.__extends || function (d, b) {
    for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p];
    function __() { this.constructor = d; }
    __.prototype = b.prototype;
    d.prototype = new __();
};
var Net;
(function (Net) {
    var _nextGaussian;
    function randomGaussian(mean, standardDeviation) {
        if (mean === void 0) { mean = 0; }
        if (standardDeviation === void 0) { standardDeviation = 1; }
        if (_nextGaussian !== undefined) {
            var nextGaussian = _nextGaussian;
            _nextGaussian = undefined;
            return (nextGaussian * standardDeviation) + mean;
        }
        else {
            var v1, v2, s, multiplier;
            do {
                v1 = 2 * Math.random() - 1; // between -1 and 1
                v2 = 2 * Math.random() - 1; // between -1 and 1
                s = v1 * v1 + v2 * v2;
            } while (s >= 1 || s == 0);
            multiplier = Math.sqrt(-2 * Math.log(s) / s);
            _nextGaussian = v2 * multiplier;
            return (v1 * multiplier * standardDeviation) + mean;
        }
    }
    Net.randomGaussian = randomGaussian;
    ;
    var tanh = function (x) {
        if (x === Infinity) {
            return 1;
        }
        else if (x === -Infinity) {
            return -1;
        }
        else {
            var y = Math.exp(2 * x);
            return (y - 1) / (y + 1);
        }
    };
    var NonLinearities = {
        sigmoid: {
            f: function (x) { return 1 / (1 + Math.exp(-x)); },
            df: function (x) { x = 1 / (1 + Math.exp(-x)); return x * (1 - x); }
        },
        tanh: {
            f: function (x) { return tanh(x); },
            df: function (x) { x = tanh(x); return 1 - x * x; }
        },
        linear: {
            f: function (x) { return x; },
            df: function (x) { return 1; }
        }
    };
    function makeArray(len, supplier) {
        var arr = new Array(len);
        for (var i = 0; i < len; i++)
            arr[i] = supplier();
        return arr;
    }
    // back propagation code adapted from https://de.wikipedia.org/wiki/Backpropagation
    var NeuralNet = (function () {
        function NeuralNet(layout, inputnames, learnRate, bias, startWeight, weights) {
            var _this = this;
            if (bias === void 0) { bias = true; }
            if (startWeight === void 0) { startWeight = function () { return Math.random() - 0.5; }; }
            this.layers = [];
            this.connections = [];
            this.learnRate = 0.01;
            this.learnRate = learnRate;
            layout = layout.slice();
            if (layout.length < 2)
                throw "Need at least two layers";
            var nid = 0;
            this.inputs = makeArray(layout.shift().neuronCount, function () { return new InputNeuron(nid, inputnames[nid++]); });
            this.layers.push(this.inputs);
            while (layout.length > 1) {
                var layer = layout.shift();
                this.layers.push(makeArray(layer.neuronCount, function () { return new Neuron(layer.activation, nid++); }));
            }
            var outputLayer = layout.shift();
            this.outputs = makeArray(outputLayer.neuronCount, function () { return new OutputNeuron(outputLayer.activation, nid++); });
            this.layers.push(this.outputs);
            this.bias = bias;
            for (var i = 0; i < this.layers.length - 1; i++) {
                var inLayer = this.layers[i];
                var outLayer = this.layers[i + 1];
                if (bias)
                    inLayer.push(new InputNeuron(nid++, "1 (bias)", 1));
                for (var _i = 0; _i < inLayer.length; _i++) {
                    var input = inLayer[_i];
                    for (var _a = 0; _a < outLayer.length; _a++) {
                        var output = outLayer[_a];
                        var conn = new Net.NeuronConnection(input, output, startWeight());
                        input.outputs.push(conn);
                        output.inputs.push(conn);
                        this.connections.push(conn);
                    }
                }
            }
            if (weights)
                weights.forEach(function (w, i) { return _this.connections[i].weight = w; });
        }
        NeuralNet.prototype.setInputsAndCalculate = function (inputVals) {
            if (inputVals.length != this.inputs.length - +this.bias)
                throw "invalid input size";
            for (var i = 0; i < inputVals.length; i++)
                this.inputs[i].output = inputVals[i];
            for (var _i = 0, _a = this.layers.slice(1); _i < _a.length; _i++) {
                var layer = _a[_i];
                for (var _b = 0; _b < layer.length; _b++) {
                    var neuron = layer[_b];
                    neuron.calculateOutput();
                }
            }
        };
        NeuralNet.prototype.getOutput = function (inputVals) {
            this.setInputsAndCalculate(inputVals);
            return this.outputs.map(function (output) { return output.output; });
        };
        NeuralNet.prototype.train = function (inputVals, expectedOutput) {
            this.setInputsAndCalculate(inputVals);
            for (var i = 0; i < this.outputs.length; i++)
                this.outputs[i].targetOutput = expectedOutput[i];
            for (var i_1 = this.layers.length - 1; i_1 > 0; i_1--) {
                for (var _i = 0, _a = this.layers[i_1]; _i < _a.length; _i++) {
                    var neuron = _a[_i];
                    neuron.calculateError();
                    for (var _b = 0, _c = neuron.inputs; _b < _c.length; _b++) {
                        var conn = _c[_b];
                        conn.calculateDeltaWeight(this.learnRate);
                    }
                }
            }
            for (var _d = 0, _e = this.connections; _d < _e.length; _d++) {
                var conn = _e[_d];
                conn.weight += conn.deltaWeight;
            }
        };
        return NeuralNet;
    })();
    Net.NeuralNet = NeuralNet;
    var NeuronConnection = (function () {
        function NeuronConnection(inp, out, weight) {
            this.inp = inp;
            this.out = out;
            this.weight = weight;
            this.deltaWeight = 0;
        }
        NeuronConnection.prototype.calculateDeltaWeight = function (learnRate) {
            this.deltaWeight = learnRate * this.out.error * this.inp.output;
        };
        return NeuronConnection;
    })();
    Net.NeuronConnection = NeuronConnection;
    var Neuron = (function () {
        function Neuron(activation, id) {
            this.activation = activation;
            this.id = id;
            this.inputs = [];
            this.outputs = [];
            this.weightedInputs = 0;
            this.output = 0;
            this.error = 0;
        }
        Neuron.prototype.calculateWeightedInputs = function () {
            this.weightedInputs = 0;
            for (var _i = 0, _a = this.inputs; _i < _a.length; _i++) {
                var conn = _a[_i];
                this.weightedInputs += conn.inp.output * conn.weight;
            }
        };
        Neuron.prototype.calculateOutput = function () {
            this.calculateWeightedInputs();
            this.output = NonLinearities[this.activation].f(this.weightedInputs);
        };
        Neuron.prototype.calculateError = function () {
            var δ = 0;
            for (var _i = 0, _a = this.outputs; _i < _a.length; _i++) {
                var output = _a[_i];
                δ += output.out.error * output.weight;
            }
            this.error = δ * NonLinearities[this.activation].df(this.weightedInputs);
        };
        return Neuron;
    })();
    Net.Neuron = Neuron;
    var InputNeuron = (function (_super) {
        __extends(InputNeuron, _super);
        function InputNeuron(id, name, output) {
            if (output === void 0) { output = 0; }
            _super.call(this, null, id);
            this.name = name;
            this.output = output;
        }
        InputNeuron.prototype.calculateOutput = function () { };
        InputNeuron.prototype.calculateWeightedInputs = function () { };
        InputNeuron.prototype.calculateError = function () { };
        return InputNeuron;
    })(Neuron);
    Net.InputNeuron = InputNeuron;
    var OutputNeuron = (function (_super) {
        __extends(OutputNeuron, _super);
        function OutputNeuron() {
            _super.apply(this, arguments);
        }
        OutputNeuron.prototype.calculateError = function () {
            this.error = NonLinearities[this.activation].df(this.weightedInputs) * (this.targetOutput - this.output);
        };
        return OutputNeuron;
    })(Neuron);
    Net.OutputNeuron = OutputNeuron;
})(Net || (Net = {}));
///<reference path='Net.ts' />
var NetworkGraph = (function () {
    function NetworkGraph(networkGraphContainer) {
        this.networkGraphContainer = networkGraphContainer;
        this.nodes = new vis.DataSet();
        this.edges = new vis.DataSet();
        this.instantiateGraph();
    }
    NetworkGraph.prototype.instantiateGraph = function () {
        // need only be run once, but removes bounciness if run every time
        var graphData = {
            nodes: this.nodes,
            edges: this.edges };
        var options = {
            nodes: { shape: 'dot' },
            edges: {
                smooth: { type: 'curvedCW', roundness: 0.2 },
                font: { align: 'top', background: 'white' },
            },
            layout: { hierarchical: { direction: "LR" } },
            interaction: { dragNodes: false }
        };
        this.graph = new vis.Network(this.networkGraphContainer, graphData, options);
    };
    NetworkGraph.prototype.loadNetwork = function (net) {
        if (this.net
            && this.net.layers.length == net.layers.length
            && this.net.layers.every(function (layer, index) { return layer.length == net.layers[index].length; })) {
            // same net layout, only update
            this.net = net;
            this.update();
            return;
        }
        this.instantiateGraph();
        this.net = net;
        this.nodes.clear();
        this.edges.clear();
        var nodes = [], edges = [];
        for (var lid = 0; lid < net.layers.length; lid++) {
            var layer = net.layers[lid];
            for (var nid = 0; nid < layer.length; nid++) {
                var neuron = layer[nid];
                var type = 'Hidden Neuron ' + (nid + 1);
                var color = '#000';
                if (neuron instanceof Net.InputNeuron) {
                    type = 'Input: ' + neuron.name;
                    color = '#008';
                }
                if (neuron instanceof Net.OutputNeuron) {
                    type = 'Output Neuron ' + (nid + 1);
                    color = '#800';
                }
                nodes.push({
                    id: neuron.id,
                    label: "" + type,
                    level: lid,
                    color: color
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
        this.nodes.add(nodes);
        this.edges.add(edges);
    };
    NetworkGraph.prototype.update = function () {
        for (var _i = 0, _a = this.net.connections; _i < _a.length; _i++) {
            var conn = _a[_i];
            this.edges.update({
                id: conn.inp.id * this.net.connections.length + conn.out.id,
                label: conn.weight.toFixed(2),
                width: Math.min(6, Math.abs(conn.weight * 2)),
                color: conn.weight > 0 ? 'blue' : 'red'
            });
        }
    };
    return NetworkGraph;
})();
var CanvasMouseNavigation = (function () {
    function CanvasMouseNavigation(canvas, transformChanged) {
        var _this = this;
        this.scalex = 100;
        this.scaley = -100;
        this.offsetx = 0;
        this.offsety = 0;
        this.mousedown = false;
        this.mousestart = { x: 0, y: 0 };
        this.toReal = {
            x: function (x) { return (x - _this.offsetx) / _this.scalex; },
            y: function (y) { return (y - _this.offsety) / _this.scaley; }
        };
        this.toCanvas = {
            x: function (c) { return c * _this.scalex + _this.offsetx; },
            y: function (c) { return c * _this.scaley + _this.offsety; }
        };
        this.offsetx = canvas.width / 3;
        this.offsety = 2 * canvas.height / 3;
        canvas.addEventListener('wheel', function (e) {
            var delta = e.deltaY / Math.abs(e.deltaY);
            _this.scalex *= 1 - delta / 10;
            _this.scaley *= 1 - delta / 10;
            transformChanged();
            e.preventDefault();
        });
        canvas.addEventListener('mousedown', function (e) {
            _this.mousedown = true;
            _this.mousestart.x = e.pageX;
            _this.mousestart.y = e.pageY;
        });
        canvas.addEventListener('mousemove', function (e) {
            if (!_this.mousedown)
                return;
            _this.offsetx += e.pageX - _this.mousestart.x;
            _this.offsety += e.pageY - _this.mousestart.y;
            _this.mousestart.x = e.pageX;
            _this.mousestart.y = e.pageY;
            transformChanged();
        });
        document.addEventListener('mouseup', function (e) { return _this.mousedown = false; });
    }
    return CanvasMouseNavigation;
})();
///<reference path='Transform.ts' />
var NetworkVisualization = (function () {
    function NetworkVisualization(canvas, trafo, sim, netOutput, backgroundResolution) {
        var _this = this;
        this.canvas = canvas;
        this.trafo = trafo;
        this.sim = sim;
        this.netOutput = netOutput;
        this.backgroundResolution = backgroundResolution;
        this.mouseDownTime = 0; // ignore clicks if dragged
        this.colors = {
            bg: ["#f88", "#8f8"],
            fg: ["#f00", "#0f0"],
            gradient: function (val) { return "rgb(" + [((1 - val) * 256) | 0, (val * 256) | 0, 100] + ")"; }
        };
        this.ctx = this.canvas.getContext('2d');
        this.canvasResized();
        window.addEventListener('resize', this.canvasResized.bind(this));
        canvas.addEventListener("click", this.canvasClicked.bind(this));
        canvas.addEventListener("mousedown", function () { return _this.mouseDownTime = Date.now(); });
        canvas.addEventListener("contextmenu", this.canvasClicked.bind(this));
    }
    NetworkVisualization.prototype.draw = function () {
        this.drawBackground();
        this.drawCoordinateSystem();
        this.drawDataPoints();
    };
    NetworkVisualization.prototype.drawDataPoints = function () {
        this.ctx.strokeStyle = "#000";
        if (this.sim.config.simType === SimulationType.BinaryClassification) {
            for (var _i = 0, _a = this.sim.config.data; _i < _a.length; _i++) {
                var val = _a[_i];
                this.drawDataPoint(val.input[0], val.input[1], val.output[0]);
            }
        }
        else if (this.sim.config.simType === SimulationType.AutoEncoder) {
            for (var _b = 0, _c = this.sim.config.data; _b < _c.length; _b++) {
                var val = _c[_b];
                var ix = val.input[0], iy = val.input[1];
                var out = this.sim.net.getOutput(val.input);
                var ox = out[0], oy = out[1];
                this.drawLine(ix, iy, ox, oy, "black");
                this.drawDataPoint(ix, iy, 1);
                this.drawDataPoint(ox, oy, 0);
            }
        }
        else {
            throw "can't draw this";
        }
    };
    NetworkVisualization.prototype.drawLine = function (x, y, x2, y2, color) {
        x = this.trafo.toCanvas.x(x);
        x2 = this.trafo.toCanvas.x(x2);
        y = this.trafo.toCanvas.y(y);
        y2 = this.trafo.toCanvas.y(y2);
        this.ctx.strokeStyle = color;
        this.ctx.beginPath();
        this.ctx.moveTo(x, y);
        this.ctx.lineTo(x2, y2);
        this.ctx.stroke();
    };
    NetworkVisualization.prototype.drawDataPoint = function (x, y, label) {
        x = this.trafo.toCanvas.x(x);
        y = this.trafo.toCanvas.y(y);
        this.ctx.fillStyle = this.colors.fg[label | 0];
        this.ctx.beginPath();
        this.ctx.arc(x, y, 5, 0, 2 * Math.PI);
        this.ctx.fill();
        this.ctx.arc(x, y, 5, 0, 2 * Math.PI);
        this.ctx.stroke();
    };
    NetworkVisualization.prototype.drawBackground = function () {
        if (this.sim.config.simType == SimulationType.AutoEncoder) {
            this.ctx.fillStyle = "white";
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            return;
        }
        for (var x = 0; x < this.canvas.width; x += this.backgroundResolution) {
            for (var y = 0; y < this.canvas.height; y += this.backgroundResolution) {
                var val = this.netOutput(this.trafo.toReal.x(x), this.trafo.toReal.y(y));
                if (this.sim.config.showGradient) {
                    this.ctx.fillStyle = this.colors.gradient(val);
                }
                else
                    this.ctx.fillStyle = this.colors.bg[+(val > 0.5)];
                this.ctx.fillRect(x, y, this.backgroundResolution, this.backgroundResolution);
            }
        }
    };
    NetworkVisualization.prototype.drawCoordinateSystem = function () {
        var marklen = 0.2;
        var ctx = this.ctx, toc = this.trafo.toCanvas;
        ctx.strokeStyle = "#000";
        ctx.fillStyle = "#000";
        ctx.textBaseline = "middle";
        ctx.textAlign = "center";
        ctx.font = "20px monospace";
        ctx.beginPath();
        ctx.moveTo(toc.x(0), 0);
        ctx.lineTo(toc.x(0), this.canvas.height);
        ctx.moveTo(toc.x(-marklen / 2), toc.y(1));
        ctx.lineTo(toc.x(marklen / 2), toc.y(1));
        ctx.fillText("1", toc.x(-marklen), toc.y(1));
        ctx.moveTo(0, toc.y(0));
        ctx.lineTo(this.canvas.width, toc.y(0));
        ctx.moveTo(toc.x(1), toc.y(-marklen / 2));
        ctx.lineTo(toc.x(1), toc.y(marklen / 2));
        ctx.fillText("1", toc.x(1), toc.y(-marklen));
        ctx.stroke();
    };
    NetworkVisualization.prototype.canvasResized = function () {
        this.canvas.width = $(this.canvas).width();
        this.canvas.height = $(this.canvas).height();
    };
    NetworkVisualization.prototype.canvasClicked = function (evt) {
        if ((Date.now() - this.mouseDownTime) > 200)
            return;
        if (this.sim.config.netLayers[0].neuronCount !== 2) {
            throw "data modification not supported for !=2 inputs";
        }
        var data = this.sim.config.data;
        var rect = this.canvas.getBoundingClientRect();
        var x = this.trafo.toReal.x(evt.clientX - rect.left);
        var y = this.trafo.toReal.y(evt.clientY - rect.top);
        if (evt.button == 2 || evt.shiftKey) {
            //remove nearest
            var nearestDist = Infinity, nearest = -1;
            for (var i = 0; i < data.length; i++) {
                var p = data[i];
                var dx = p.input[0] - x, dy = p.input[1] - y, dist = dx * dx + dy * dy;
                if (dist < nearestDist)
                    nearest = i, nearestDist = dist;
            }
            if (nearest >= 0)
                data.splice(nearest, 1);
        }
        else {
            if (this.sim.config.simType == SimulationType.AutoEncoder) {
                data.push({ input: [x, y], output: [x, y] });
            }
            else if (this.sim.config.simType == SimulationType.BinaryClassification) {
                var label = evt.button == 0 ? 0 : 1;
                if (evt.ctrlKey)
                    label = label == 0 ? 1 : 0;
                data.push({ input: [x, y], output: [label] });
            }
        }
        this.draw();
        evt.preventDefault();
    };
    return NetworkVisualization;
})();
///<reference path='Simulation.ts' />
var SimulationType;
(function (SimulationType) {
    SimulationType[SimulationType["BinaryClassification"] = 0] = "BinaryClassification";
    SimulationType[SimulationType["AutoEncoder"] = 1] = "AutoEncoder";
})(SimulationType || (SimulationType = {}));
var Presets;
(function (Presets) {
    var presets = {
        "Default": {
            stepsPerFrame: 50,
            learningRate: 0.05,
            showGradient: false,
            bias: true,
            autoRestartTime: 5000,
            autoRestart: true,
            iterationsPerClick: 5000,
            simType: SimulationType.BinaryClassification,
            data: [
                { input: [0, 0], output: [0] },
                { input: [0, 1], output: [1] },
                { input: [1, 0], output: [1] },
                { input: [1, 1], output: [0] }
            ],
            netLayers: [
                { neuronCount: 2 },
                { neuronCount: 2, activation: "sigmoid" },
                { neuronCount: 1, activation: "sigmoid" }
            ]
        },
        "XOR": {},
        "Auto-Encoder": {
            simType: SimulationType.AutoEncoder,
            stepsPerFrame: 1,
            iterationsPerClick: 1,
            data: [
                { input: [2.25, 0.19], output: [2.25, 0.19] },
                { input: [1.37, 0.93], output: [1.37, 0.93] },
                { input: [0.62, 1.46], output: [0.62, 1.46] },
                { input: [-0.23, 2.16], output: [-0.23, 2.16] },
                { input: [-0.55, 2.44], output: [-0.55, 2.44] },
                { input: [1.04, 1.05], output: [1.04, 1.05] },
                { input: [1.70, 0.85], output: [1.70, 0.85] },
                { input: [2.01, 0.46], output: [2.01, 0.46] },
                { input: [0.40, 1.73], output: [0.40, 1.73] },
                { input: [2.73, 0.01], output: [2.73, 0.01] },
                { input: [2.86, -0.25], output: [2.86, -0.25] },
                { input: [0.14, 2.07], output: [0.14, 2.07] }],
            netLayers: [
                { neuronCount: 2 },
                { neuronCount: 1, activation: "sigmoid" },
                { neuronCount: 2, activation: "linear" }
            ],
            showGradient: true
        }
    };
    function get(name) {
        return $.extend({}, presets["Default"], presets[name]);
    }
    Presets.get = get;
    function printDataPoints() {
        return window.simulation.config.data.map(function (e) { return '{input:[' + e.input.map(function (x) { return x.toFixed(2); })
            + '], output:[' + e.input.map(function (x) { return x.toFixed(2); }) + ']},'; }).join("\n");
    }
    Presets.printDataPoints = printDataPoints;
})(Presets || (Presets = {}));
///<reference path='../lib/typings/jquery/jquery.d.ts' />
///<reference path='Net.ts' />
///<reference path='NetworkGraph.ts' />
///<reference path='NetworkVisualization.ts' />
///<reference path='Presets.ts' />
var Simulation = (function () {
    function Simulation() {
        var _this = this;
        this.backgroundResolution = 10;
        this.stepNum = 0;
        this.running = false;
        this.runningId = -1;
        this.restartTimeout = -1;
        this.hiddenLayerDiv = $("#neuronCountModifier div").eq(1).clone();
        this.config = Presets.get('XOR');
        this.statusIterEle = document.getElementById('statusIteration');
        this.statusCorrectEle = document.getElementById('statusCorrect');
        this.aniFrameCallback = this.animationStep.bind(this);
        var canvas = $("#neuralOutputCanvas")[0];
        this.netviz = new NetworkVisualization(canvas, new CanvasMouseNavigation(canvas, function () { return _this.draw(); }), this, function (x, y) { return _this.net.getOutput([x, y])[0]; }, this.backgroundResolution);
        this.netgraph = new NetworkGraph($("#neuralNetworkGraph")[0]);
        $("#learningRate").slider({
            min: 0.01, max: 1, step: 0.005, scale: "logarithmic", value: 0.05
        }).on('slide', function (e) { return $("#learningRateVal").text(e.value.toFixed(2)); });
        $("#neuronCountModifier").on("click", "button", function (e) {
            var inc = e.target.textContent == '+';
            var layer = $(e.target.parentNode).index();
            var newval = _this.config.netLayers[layer].neuronCount + (inc ? 1 : -1);
            if (newval < 1)
                return;
            _this.config.netLayers[layer].neuronCount = newval;
            $("#neuronCountModifier .neuronCount").eq(layer).text(newval);
            _this.initializeNet();
        });
        $("#layerCountModifier").on("click", "button", function (e) {
            var inc = e.target.textContent == '+';
            if (!inc) {
                if (_this.config.netLayers.length == 2)
                    return;
                _this.config.netLayers.splice(1, 1);
                $("#neuronCountModifier div").eq(1).remove();
            }
            else {
                $("#neuronCountModifier div").eq(1).before(_this.hiddenLayerDiv.clone());
                _this.config.netLayers.splice(1, 0, { activation: 'sigmoid', neuronCount: 2 });
            }
            $("#layerCount").text(_this.config.netLayers.length);
            _this.initializeNet();
        });
        $("#neuronCountModifier").on("change", "select", function (e) {
            var layer = $(e.target.parentNode).index();
            _this.config.netLayers[layer].activation = e.target.value;
            _this.initializeNet();
        });
        $("#presetLoader").on("click", "a", function (e) {
            var name = e.target.textContent;
            _this.config = Presets.get(name);
            _this.initializeNet();
        });
        this.reset();
        this.run();
    }
    Simulation.prototype.initializeNet = function (weights) {
        if (this.net)
            this.stop();
        //let cache = [0.18576880730688572,-0.12869677506387234,0.08548374730162323,-0.19820863520726562,-0.09532690420746803,-0.3415223266929388,-0.309354952769354,-0.157513455953449];
        //let cache = [-0.04884958150796592,-0.3569231238216162,0.11143312812782824,0.43614205135963857,0.3078767384868115,-0.22759653301909566,0.09250503336079419,0.3279339636210352];
        this.net = new Net.NeuralNet(this.config.netLayers, ["x", "y"], this.config.learningRate, this.config.bias, undefined, weights);
        console.log("net:" + JSON.stringify(this.net.connections.map(function (c) { return c.weight; })));
        this.stepNum = 0;
        this.netgraph.loadNetwork(this.net);
        this.draw();
        this.updateStatusLine();
    };
    Simulation.prototype.step = function () {
        this.stepNum++;
        for (var _i = 0, _a = this.config.data; _i < _a.length; _i++) {
            var val = _a[_i];
            this.net.train(val.input, val.output);
        }
    };
    Simulation.prototype.draw = function () {
        this.netviz.draw();
        this.netgraph.update();
    };
    Simulation.prototype.run = function () {
        if (this.running)
            return;
        $("#runButton").text("Stop").addClass("btn-danger").removeClass("btn-primary");
        this.running = true;
        this.animationStep();
    };
    Simulation.prototype.stop = function () {
        clearTimeout(this.restartTimeout);
        $("#runButton").text("Run").addClass("btn-primary").removeClass("btn-danger");
        this.restartTimeout = -1;
        this.running = false;
        cancelAnimationFrame(this.runningId);
    };
    Simulation.prototype.reset = function () {
        this.stop();
        this.loadConfig();
        this.initializeNet();
    };
    Simulation.prototype.updateStatusLine = function () {
        var _this = this;
        var correct = 0;
        switch (this.config.simType) {
            case SimulationType.BinaryClassification:
                for (var _i = 0, _a = this.config.data; _i < _a.length; _i++) {
                    var val = _a[_i];
                    var res = this.net.getOutput(val.input);
                    if (+(res[0] > 0.5) == val.output[0])
                        correct++;
                }
                this.statusCorrectEle.innerHTML = correct + "/" + this.config.data.length;
                break;
            case SimulationType.AutoEncoder:
                this.statusCorrectEle.innerHTML = "?";
                break;
        }
        this.statusIterEle.innerHTML = this.stepNum.toString();
        if (correct == this.config.data.length) {
            if (this.config.autoRestart && this.running && this.restartTimeout == -1) {
                this.restartTimeout = setTimeout(function () {
                    _this.stop();
                    _this.restartTimeout = -1;
                    setTimeout(function () { _this.reset(); _this.run(); }, 1000);
                }, this.config.autoRestartTime - 1);
            }
        }
        else {
            if (this.restartTimeout != -1) {
                clearTimeout(this.restartTimeout);
                this.restartTimeout = -1;
            }
        }
    };
    Simulation.prototype.animationStep = function () {
        for (var i = 0; i < this.config.stepsPerFrame; i++)
            this.step();
        this.draw();
        this.updateStatusLine();
        if (this.running)
            this.runningId = requestAnimationFrame(this.aniFrameCallback);
    };
    Simulation.prototype.iterations = function () {
        this.stop();
        for (var i = 0; i < this.config.iterationsPerClick; i++)
            this.step();
        this.draw();
        this.updateStatusLine();
    };
    Simulation.prototype.loadConfig = function () {
        var config = this.config;
        for (var conf in config) {
            var ele = document.getElementById(conf);
            if (!ele)
                continue;
            if (ele.type == 'checkbox')
                config[conf] = ele.checked;
            else
                config[conf] = ele.value;
        }
        if (this.net)
            this.net.learnRate = this.config.learningRate;
    };
    Simulation.prototype.randomizeData = function () {
        if (this.config.netLayers[0].neuronCount !== 2 || this.config.simType !== SimulationType.BinaryClassification)
            throw "can't create random data for this network";
        var count = Math.random() * 5 + 4;
        this.config.data = [];
        for (var i = 0; i < count; i++) {
            this.config.data[i] = { input: [Math.random() * 2, Math.random() * 2], output: [+(Math.random() > 0.5)] };
        }
        this.draw();
    };
    Simulation.prototype.runtoggle = function () {
        if (this.running)
            this.stop();
        else
            this.run();
    };
    return Simulation;
})();
///<reference path='../lib/typings/react/react-global.d.ts' />
///<reference path='../lib/typings/jquery/jquery.d.ts' />
///<reference path='Net.ts' />
///<reference path='Simulation.ts' />
///<reference path='Transform.ts' />
///<reference path='NetworkVisualization.ts' />
var simulation;
$(document).ready(function () { return simulation = new Simulation(); });
function checkSanity() {
    var out = [-0.3180095069079748, -0.2749093166215802, -0.038532753589859546, 0.09576201205465842, -0.3460678329225116,
        0.23218797637289554, -0.33191669283980774, 0.5140297481331861, -0.1518989898989732];
    var inp = [-0.3094657452311367, -0.2758470894768834, 0.005968799814581871, 0.13201188389211893, -0.33257930004037917,
        0.24626848078332841, -0.35734778200276196, 0.489376779878512, -0.2165879353415221];
    simulation.stop();
    simulation.config.netLayers = [
        { neuronCount: 2 },
        { neuronCount: 2, activation: "sigmoid" },
        { neuronCount: 1, activation: "sigmoid" }
    ];
    simulation.net.connections.forEach(function (e, i) { return e.weight = inp[i]; });
    for (var i = 0; i < 1000; i++)
        simulation.step();
    var realout = simulation.net.connections.map(function (e) { return e.weight; });
    if (realout.every(function (e, i) { return e !== out[i]; }))
        throw "insanity!";
    return "ok";
}
//# sourceMappingURL=program.js.map