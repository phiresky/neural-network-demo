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
            arr[i] = supplier(i);
        return arr;
    }
    // back propagation code adapted from https://de.wikipedia.org/wiki/Backpropagation
    var NeuralNet = (function () {
        function NeuralNet(input, hidden, output, learnRate, bias, startWeight, startWeights) {
            var _this = this;
            if (bias === void 0) { bias = true; }
            if (startWeight === void 0) { startWeight = function () { return Math.random() - 0.5; }; }
            this.learnRate = learnRate;
            this.bias = bias;
            this.startWeights = startWeights;
            this.layers = [];
            this.connections = [];
            var nid = 0;
            this.inputs = makeArray(input.neuronCount, function (i) { return new InputNeuron(nid++, input.names[i]); });
            this.layers.push(this.inputs.slice());
            for (var _i = 0; _i < hidden.length; _i++) {
                var layer = hidden[_i];
                this.layers.push(makeArray(layer.neuronCount, function (i) { return new Neuron(layer.activation, nid++); }));
            }
            this.outputs = makeArray(output.neuronCount, function (i) { return new OutputNeuron(output.activation, nid++, output.names[i]); });
            this.layers.push(this.outputs);
            this.bias = bias;
            for (var i = 0; i < this.layers.length - 1; i++) {
                var inLayer = this.layers[i];
                var outLayer = this.layers[i + 1];
                if (bias)
                    inLayer.push(new InputNeuron(nid++, "1 (bias)", 1));
                for (var _a = 0; _a < inLayer.length; _a++) {
                    var input_1 = inLayer[_a];
                    for (var _b = 0; _b < outLayer.length; _b++) {
                        var output_1 = outLayer[_b];
                        var conn = new Net.NeuronConnection(input_1, output_1);
                        input_1.outputs.push(conn);
                        output_1.inputs.push(conn);
                        this.connections.push(conn);
                    }
                }
            }
            if (!this.startWeights) {
                this.startWeights = this.connections.map(function (c) { return c.weight = startWeight(); });
            }
            else
                this.startWeights.forEach(function (w, i) { return _this.connections[i].weight = w; });
        }
        NeuralNet.prototype.setInputsAndCalculate = function (inputVals) {
            for (var i = 0; i < this.inputs.length; i++)
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
        function NeuronConnection(inp, out) {
            this.inp = inp;
            this.out = out;
            this.deltaWeight = 0;
            this.weight = 0;
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
        function InputNeuron(id, name, constantOutput) {
            _super.call(this, null, id);
            this.name = name;
            this.constant = false; // value won't change
            if (constantOutput !== undefined) {
                this.output = constantOutput;
                this.constant = true;
            }
        }
        InputNeuron.prototype.calculateOutput = function () { };
        InputNeuron.prototype.calculateWeightedInputs = function () { };
        InputNeuron.prototype.calculateError = function () { };
        return InputNeuron;
    })(Neuron);
    Net.InputNeuron = InputNeuron;
    var OutputNeuron = (function (_super) {
        __extends(OutputNeuron, _super);
        function OutputNeuron(activation, id, name) {
            _super.call(this, activation, id);
            this.activation = activation;
            this.name = name;
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
                    if (neuron.constant)
                        color = NetworkVisualization.colors.autoencoder.bias;
                    else
                        color = NetworkVisualization.colors.autoencoder.input;
                }
                if (neuron instanceof Net.OutputNeuron) {
                    type = 'Output: ' + neuron.name;
                    color = NetworkVisualization.colors.autoencoder.output;
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
    function CanvasMouseNavigation(canvas, transformActive, transformChanged) {
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
            if (e.deltaY === 0)
                return;
            var delta = e.deltaY / Math.abs(e.deltaY);
            _this.scalex *= 1 - delta / 10;
            _this.scaley *= 1 - delta / 10;
            transformChanged();
            e.preventDefault();
        });
        canvas.addEventListener('mousedown', function (e) {
            if (!transformActive())
                return;
            _this.mousedown = true;
            _this.mousestart.x = e.pageX;
            _this.mousestart.y = e.pageY;
        });
        canvas.addEventListener('mousemove', function (e) {
            if (!transformActive())
                return;
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
var InputMode;
(function (InputMode) {
    InputMode[InputMode["InputPrimary"] = 0] = "InputPrimary";
    InputMode[InputMode["InputSecondary"] = 1] = "InputSecondary";
    InputMode[InputMode["Remove"] = 2] = "Remove";
    InputMode[InputMode["Move"] = 3] = "Move";
    InputMode[InputMode["Table"] = 4] = "Table";
})(InputMode || (InputMode = {}));
var NetworkVisualization = (function () {
    function NetworkVisualization(canvas, trafo, sim, backgroundResolution) {
        this.canvas = canvas;
        this.trafo = trafo;
        this.sim = sim;
        this.backgroundResolution = backgroundResolution;
        this.inputMode = 0;
        this.ctx = this.canvas.getContext('2d');
        this.canvasResized();
        window.addEventListener('resize', this.canvasResized.bind(this));
        canvas.addEventListener("click", this.canvasClicked.bind(this));
        canvas.addEventListener("contextmenu", this.canvasClicked.bind(this));
    }
    NetworkVisualization.prototype.draw = function () {
        if (this.sim.config.inputLayer.neuronCount != 2 || this.sim.config.outputLayer.neuronCount > 2) {
            this.clear('white');
            this.ctx.fillStyle = 'black';
            this.ctx.fillText("Cannot draw this data", this.canvas.width / 2, this.canvas.height / 2);
            return;
        }
        this.drawBackground();
        this.drawCoordinateSystem();
        this.drawDataPoints();
    };
    NetworkVisualization.prototype.drawDataPoints = function () {
        this.ctx.strokeStyle = "#000";
        if (this.sim.config.outputLayer.neuronCount === 1) {
            for (var _i = 0, _a = this.sim.config.data; _i < _a.length; _i++) {
                var val = _a[_i];
                this.drawPoint(val.input[0], val.input[1], NetworkVisualization.colors.binaryClassify.fg[val.output[0] | 0]);
            }
        }
        else if (this.sim.config.outputLayer.neuronCount === 2) {
            for (var _b = 0, _c = this.sim.config.data; _b < _c.length; _b++) {
                var val = _c[_b];
                var ix = val.input[0], iy = val.input[1];
                var out = this.sim.net.getOutput(val.input);
                var ox = out[0], oy = out[1];
                this.drawLine(ix, iy, ox, oy, "black");
                this.drawPoint(ix, iy, NetworkVisualization.colors.autoencoder.input);
                this.drawPoint(ox, oy, NetworkVisualization.colors.autoencoder.output);
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
    NetworkVisualization.prototype.drawPoint = function (x, y, color) {
        x = this.trafo.toCanvas.x(x);
        y = this.trafo.toCanvas.y(y);
        this.ctx.fillStyle = color;
        this.ctx.beginPath();
        this.ctx.arc(x, y, 5, 0, 2 * Math.PI);
        this.ctx.fill();
        this.ctx.arc(x, y, 5, 0, 2 * Math.PI);
        this.ctx.stroke();
    };
    NetworkVisualization.prototype.clear = function (color) {
        this.ctx.fillStyle = "white";
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        return;
    };
    NetworkVisualization.prototype.drawBackground = function () {
        if (this.sim.config.outputLayer.neuronCount === 2) {
            this.clear('white');
            return;
        }
        for (var x = 0; x < this.canvas.width; x += this.backgroundResolution) {
            for (var y = 0; y < this.canvas.height; y += this.backgroundResolution) {
                var val = this.sim.net.getOutput([this.trafo.toReal.x(x + this.backgroundResolution / 2), this.trafo.toReal.y(y + this.backgroundResolution / 2)])[0];
                if (this.sim.config.showGradient) {
                    this.ctx.fillStyle = NetworkVisualization.colors.binaryClassify.gradient(val);
                }
                else
                    this.ctx.fillStyle = NetworkVisualization.colors.binaryClassify.bg[+(val > 0.5)];
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
        var data = this.sim.config.data;
        var rect = this.canvas.getBoundingClientRect();
        var x = this.trafo.toReal.x(evt.clientX - rect.left);
        var y = this.trafo.toReal.y(evt.clientY - rect.top);
        if (this.inputMode == 2 || evt.button == 2 || evt.shiftKey) {
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
        else if (this.inputMode < 2) {
            // add data point
            if (this.sim.config.outputLayer.neuronCount === 2) {
                data.push({ input: [x, y], output: [x, y] });
            }
            else if (this.sim.config.outputLayer.neuronCount === 1) {
                var inv = function (x) { return x == 0 ? 1 : 0; };
                var label = this.inputMode;
                if (evt.button != 0)
                    label = inv(label);
                if (evt.ctrlKey || evt.metaKey || evt.altKey)
                    label = inv(label);
                data.push({ input: [x, y], output: [label] });
            }
        }
        this.sim.setIsCustom();
        evt.preventDefault();
        this.draw();
    };
    NetworkVisualization.colors = {
        binaryClassify: {
            bg: ["#f88", "#8f8"],
            fg: ["#f00", "#0f0"],
            gradient: function (val) { return "rgb(" +
                [(((1 - val) * (256 - 60)) | 0) + 60, ((val * (256 - 60)) | 0) + 60, 60] + ")"; }
        },
        autoencoder: {
            input: '#2188e0',
            output: '#ff931f',
            bias: '#008'
        }
    };
    return NetworkVisualization;
})();
///<reference path='Simulation.ts' />
var Presets;
(function (Presets) {
    var presets = [
        {
            name: "Default",
            stepsPerFrame: 50,
            learningRate: 0.05,
            showGradient: false,
            bias: true,
            autoRestartTime: 5000,
            autoRestart: true,
            iterationsPerClick: 5000,
            data: [
                { input: [0, 0], output: [0] },
                { input: [0, 1], output: [1] },
                { input: [1, 0], output: [1] },
                { input: [1, 1], output: [0] }
            ],
            inputLayer: { neuronCount: 2, names: ["x", "y"] },
            outputLayer: { neuronCount: 1, activation: "sigmoid", names: ["x XOR y"] },
            hiddenLayers: [
                { neuronCount: 2, activation: "sigmoid" },
            ]
        },
        {
            name: "Binary Classifier for XOR"
        },
        {
            name: "Binary Classifier for circular data",
            hiddenLayers: [
                { "neuronCount": 3, "activation": "sigmoid" },
            ],
            inputLayer: { neuronCount: 2, names: ["x", "y"] },
            outputLayer: { neuronCount: 1, "activation": "sigmoid", names: ["Class"] },
            data: [{ input: [1.46, 1.36], output: [0] },
                { input: [1.14, 1.26], output: [0] },
                { input: [0.96, 0.97], output: [0] },
                { input: [1.04, 0.76], output: [0] },
                { input: [1.43, 0.81], output: [0] },
                { input: [1.30, 1.05], output: [0] },
                { input: [1.45, 1.22], output: [0] },
                { input: [2.04, 1.10], output: [0] },
                { input: [1.06, 0.28], output: [0] },
                { input: [0.96, 0.57], output: [0] },
                { input: [1.28, 0.46], output: [0] },
                { input: [1.51, 0.33], output: [0] },
                { input: [1.65, 0.68], output: [0] },
                { input: [1.67, 1.01], output: [0] },
                { input: [1.50, 1.83], output: [1] },
                { input: [0.76, 1.69], output: [1] },
                { input: [0.40, 0.71], output: [1] },
                { input: [0.61, 1.18], output: [1] },
                { input: [0.26, 1.42], output: [1] },
                { input: [0.28, 1.89], output: [1] },
                { input: [1.37, 1.89], output: [1] },
                { input: [1.11, 1.90], output: [1] },
                { input: [1.05, 2.04], output: [1] },
                { input: [2.43, 1.42], output: [1] },
                { input: [2.39, 1.20], output: [1] },
                { input: [2.10, 1.53], output: [1] },
                { input: [1.89, 1.72], output: [1] },
                { input: [2.69, 0.72], output: [1] },
                { input: [2.96, 0.44], output: [1] },
                { input: [2.50, 0.79], output: [1] },
                { input: [2.85, 1.23], output: [1] },
                { input: [2.82, 1.37], output: [1] },
                { input: [1.93, 1.90], output: [1] },
                { input: [2.18, 1.77], output: [1] },
                { input: [2.29, 0.39], output: [1] },
                { input: [2.57, 0.22], output: [1] },
                { input: [2.70, -0.11], output: [1] },
                { input: [1.96, -0.20], output: [1] },
                { input: [1.89, -0.10], output: [1] },
                { input: [1.77, 0.13], output: [1] },
                { input: [0.73, 0.01], output: [1] },
                { input: [0.37, 0.31], output: [1] },
                { input: [0.46, 0.44], output: [1] },
                { input: [0.48, 0.11], output: [1] },
                { input: [0.37, -0.10], output: [1] },
                { input: [1.03, -0.42], output: [1] },
                { input: [1.35, -0.25], output: [1] },
                { input: [1.17, 0.01], output: [1] },
                { input: [0.12, 0.94], output: [1] },
                { input: [2.05, 0.32], output: [1] },
                { input: [1.97, 0.55], output: [0] }]
        },
        {
            name: "Auto-Encoder for linear data",
            stepsPerFrame: 1,
            iterationsPerClick: 1,
            parent: "Auto-Encoder for circular data",
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
            hiddenLayers: [
                { neuronCount: 1, activation: "sigmoid" },
            ],
            showGradient: true
        },
        {
            name: "Auto-Encoder for x^2",
            parent: "Auto-Encoder for circular data",
            netLayers: [
                {
                    "activation": "sigmoid",
                    "neuronCount": 2
                },
                {
                    "activation": "linear",
                    "neuronCount": 1
                },
                {
                    "neuronCount": 2,
                    "activation": "sigmoid"
                },
            ],
            data: Array.apply(null, Array(17))
                .map(function (e, i) { return (i - 8) / 8; }).map(function (x) { return ({ input: [x, x * x], output: [x, x * x] }); })
        },
        {
            name: "Auto-Encoder for circular data",
            "stepsPerFrame": 500,
            "learningRate": 0.01,
            "iterationsPerClick": 10000,
            inputLayer: { neuronCount: 2, names: ["x", "y"] },
            outputLayer: { neuronCount: 2, activation: "linear", names: ["x", "y"] },
            hiddenLayers: [
                {
                    "activation": "sigmoid",
                    "neuronCount": 3
                },
                {
                    "activation": "linear",
                    "neuronCount": 1
                },
                {
                    "neuronCount": 3,
                    "activation": "sigmoid"
                },
            ],
            data: [{ input: [-0.83, 0.55], output: [-0.83, 0.55] },
                { input: [-0.98, 0.21], output: [-0.98, 0.21] },
                { input: [-0.77, -0.64], output: [-0.77, -0.64] },
                { input: [0.95, 0.31], output: [0.95, 0.31] },
                { input: [-0.86, -0.51], output: [-0.86, -0.51] },
                { input: [0.99, -0.11], output: [0.99, -0.11] },
                { input: [0.97, 0.24], output: [0.97, 0.24] },
                { input: [0.85, 0.52], output: [0.85, 0.52] },
                { input: [-0.99, 0.15], output: [-0.99, 0.15] },
                { input: [0.62, 0.78], output: [0.62, 0.78] },
                { input: [0.46, -0.89], output: [0.46, -0.89] },
                { input: [-0.68, -0.73], output: [-0.68, -0.73] },
                { input: [0.60, -0.80], output: [0.60, -0.80] },
                { input: [0.38, 0.92], output: [0.38, 0.92] },
                { input: [0.76, 0.65], output: [0.76, 0.65] },
                { input: [0.33, -0.94], output: [0.33, -0.94] },
                { input: [-0.99, -0.17], output: [-0.99, -0.17] },
                { input: [-0.99, -0.17], output: [-0.99, -0.17] },
                { input: [-0.97, -0.26], output: [-0.97, -0.26] },
                { input: [-0.79, -0.61], output: [-0.79, -0.61] },
                { input: [-0.03, -1.00], output: [-0.03, -1.00] },
                { input: [0.58, 0.81], output: [0.58, 0.81] },
                { input: [-0.67, -0.74], output: [-0.67, -0.74] },
                { input: [0.14, 0.99], output: [0.14, 0.99] },
                { input: [0.13, -0.99], output: [0.13, -0.99] },
                { input: [0.76, 0.65], output: [0.76, 0.65] },
                { input: [-0.49, 0.87], output: [-0.49, 0.87] },
                { input: [-0.28, 0.96], output: [-0.28, 0.96] },
                { input: [0.47, -0.88], output: [0.47, -0.88] },
                { input: [-0.03, 1.00], output: [-0.03, 1.00] },
                { input: [-0.70, 0.71], output: [-0.70, 0.71] },
                { input: [0.38, 0.93], output: [0.38, 0.93] },
                { input: [0.62, 0.79], output: [0.62, 0.79] },
                { input: [0.72, -0.69], output: [0.72, -0.69] },
                { input: [-0.41, -0.91], output: [-0.41, -0.91] },
                { input: [0.74, -0.67], output: [0.74, -0.67] },
                { input: [0.44, 0.90], output: [0.44, 0.90] },
                { input: [-0.99, -0.16], output: [-0.99, -0.16] },
                { input: [0.62, 0.78], output: [0.62, 0.78] },
                { input: [0.95, -0.39], output: [0.95, -0.39] },
                { input: [0.86, -0.53], output: [0.86, -0.53] }]
        }
    ];
    function getNames() {
        return presets.map(function (p) { return p.name; }).filter(function (c) { return c !== "Default"; });
    }
    Presets.getNames = getNames;
    function exists(name) {
        return presets.filter(function (p) { return p.name === name; })[0] !== undefined;
    }
    Presets.exists = exists;
    function get(name) {
        var chain = [];
        var preset = presets.filter(function (p) { return p.name === name; })[0];
        chain.unshift(preset);
        while (true) {
            var parentName = preset.parent || "Default";
            preset = presets.filter(function (p) { return p.name === parentName; })[0];
            chain.unshift(preset);
            if (parentName === "Default")
                break;
        }
        chain.unshift({});
        console.log("loading chain=" + chain.map(function (c) { return c.name; }));
        return JSON.parse(JSON.stringify($.extend.apply($, chain)));
    }
    Presets.get = get;
    function printPreset(parentName) {
        if (parentName === void 0) { parentName = "Default"; }
        var parent = presets.filter(function (p) { return p.name === parentName; })[0];
        var config = window.simulation.config;
        var outconf = {};
        for (var prop in config) {
            if (config[prop] !== parent[prop])
                outconf[prop] = config[prop];
        }
        /*outconf.data = config.data.map(
            e => '{input:[' + e.input.map(x=> x.toFixed(2))
                + '], output:[' +
                (config["simType"] == SimulationType.BinaryClassification
                    ? e.output
                    : e.input.map(x=> x.toFixed(2)))
                + ']},').join("\n");*/
        return outconf;
    }
    Presets.printPreset = printPreset;
})(Presets || (Presets = {}));
///<reference path='../lib/typings/jquery/jquery.d.ts' />
///<reference path='../lib/typings/jquery-handsontable/jquery-handsontable.d.ts' />
///<reference path='Net.ts' />
///<reference path='NetworkGraph.ts' />
///<reference path='NetworkVisualization.ts' />
///<reference path='Presets.ts' />
;
var TableEditor = (function () {
    function TableEditor(container, sim) {
        var _this = this;
        this.container = container;
        this.headerCount = 2;
        this.lastUpdate = 0;
        var headerRenderer = function firstRowRenderer(instance, td) {
            Handsontable.renderers.TextRenderer.apply(this, arguments);
            td.style.fontWeight = 'bold';
            td.style.background = '#CCC';
        };
        container.handsontable({
            minSpareRows: 1,
            cells: function (row, col, prop) {
                if (row >= _this.headerCount)
                    return { type: 'numeric', format: '0.[000]' };
                else
                    return { renderer: headerRenderer };
            },
            //customBorders: true,
            allowInvalid: false,
            afterChange: this.afterChange.bind(this)
        });
        this.hot = container.handsontable('getInstance');
        $("<div>").addClass("btn btn-default")
            .css({ position: "absolute", right: "2em", bottom: "2em" })
            .text("Remove all")
            .click(function (e) { sim.config.data = []; _this.loadData(sim); })
            .appendTo(container);
        this.loadData(sim);
    }
    TableEditor.prototype.afterChange = function (changes, reason) {
        if (reason === 'loadData')
            return;
        this.reparseData();
    };
    TableEditor.prototype.reparseData = function () {
        var data = this.hot.getData();
        var headers = data[1];
        var ic = sim.config.inputLayer.neuronCount, oc = sim.config.outputLayer.neuronCount;
        sim.config.inputLayer.names = headers.slice(0, ic);
        sim.config.outputLayer.names = headers.slice(ic, ic + oc);
        sim.config.data = data.slice(2).map(function (row) { return row.slice(0, ic + oc); }).filter(function (row) { return row.every(function (cell) { return typeof cell === 'number'; }); })
            .map(function (row) { return { input: row.slice(0, ic), output: row.slice(ic) }; });
        sim.setIsCustom();
    };
    TableEditor.prototype.updateRealOutput = function () {
        if ((Date.now() - this.lastUpdate) < 500)
            return;
        this.lastUpdate = Date.now();
        var xOffset = sim.config.inputLayer.neuronCount + sim.config.outputLayer.neuronCount;
        var vals = [];
        for (var y = 0; y < sim.config.data.length; y++) {
            var p = sim.config.data[y];
            var op = sim.net.getOutput(p.input);
            for (var x = 0; x < op.length; x++) {
                vals.push([y + this.headerCount, xOffset + x, op[x]]);
            }
        }
        this.hot.setDataAtCell(vals, "loadData");
    };
    TableEditor.prototype.loadData = function (sim) {
        var data = [[], sim.config.inputLayer.names.concat(sim.config.outputLayer.names).concat(sim.config.outputLayer.names)];
        var ic = sim.config.inputLayer.neuronCount, oc = sim.config.outputLayer.neuronCount;
        data[0][0] = 'Inputs';
        data[0][ic] = 'Expected Output';
        data[0][ic + oc + oc - 1] = ' ';
        data[0][ic + oc] = 'Actual Output';
        sim.config.data.forEach(function (t) { return data.push(t.input.concat(t.output)); });
        this.hot.loadData(data);
        /*this.hot.updateSettings({customBorders: [
                {
                    range: {
                        from: { row: 0, col: ic },
                        to: { row: 100, col: ic }
                    },
                    left: { width: 2, color: 'black' }
                }, {
                    range: {
                        from: { row: 0, col: ic+oc },
                        to: { row: 100, col: ic+oc }
                    },
                    left: { width: 2, color: 'black' }
                }
            ]});
        this.hot.runHooks('afterInit');*/
    };
    return TableEditor;
})();
var NeuronGui = (function () {
    function NeuronGui(sim) {
        var _this = this;
        this.sim = sim;
        this.layerDiv = $("#hiddenLayersModify > div").clone();
        $("#hiddenLayersModify").on("click", "button", function (e) {
            var inc = e.target.textContent == '+';
            var layer = $(e.target.parentNode).index();
            var newval = sim.config.hiddenLayers[layer].neuronCount + (inc ? 1 : -1);
            if (newval < 1)
                return;
            sim.config.hiddenLayers[layer].neuronCount = newval;
            $("#hiddenLayersModify .neuronCount").eq(layer).text(newval);
            sim.setIsCustom();
            sim.initializeNet();
        });
        $("#inputLayerModify,#outputLayerModify").on("click", "button", function (e) {
            var isInput = $(e.target).closest("#inputLayerModify").length > 0;
            var name = isInput ? "input" : "output";
            var targetLayer = isInput ? sim.config.inputLayer : sim.config.outputLayer;
            var inc = e.target.textContent == '+';
            var newval = targetLayer.neuronCount + (inc ? 1 : -1);
            if (newval < 1)
                return;
            targetLayer.neuronCount = newval;
            $("#" + name + "LayerModify .neuronCount").text(newval);
            sim.config.data = [];
            sim.setIsCustom();
            sim.initializeNet();
        });
        $("#layerCountModifier").on("click", "button", function (e) {
            var inc = e.target.textContent == '+';
            if (!inc) {
                if (sim.config.hiddenLayers.length == 0)
                    return;
                sim.config.hiddenLayers.shift();
                _this.removeLayer();
            }
            else {
                sim.config.hiddenLayers.unshift({ activation: 'sigmoid', neuronCount: 2 });
                _this.addLayer();
            }
            $("#layerCount").text(sim.config.hiddenLayers.length + 2);
            sim.setIsCustom();
            sim.initializeNet();
        });
        $("#outputLayerModify").on("change", "select", function (e) {
            sim.config.outputLayer.activation = e.target.value;
            sim.setIsCustom();
            sim.initializeNet();
        });
        $("#hiddenLayersModify").on("change", "select", function (e) {
            var layer = $(e.target.parentNode).index();
            sim.config.hiddenLayers[layer].activation = e.target.value;
            sim.setIsCustom();
            sim.initializeNet();
        });
    }
    NeuronGui.prototype.removeLayer = function () {
        $("#hiddenLayersModify > div").eq(0).remove();
    };
    NeuronGui.prototype.addLayer = function () {
        $("#hiddenLayersModify > div").eq(0).before(this.layerDiv.clone());
    };
    NeuronGui.prototype.setActivation = function (layer, activ) {
    };
    NeuronGui.prototype.regenerate = function () {
        var targetCount = this.sim.config.hiddenLayers.length;
        while ($("#hiddenLayersModify > div").length > targetCount)
            this.removeLayer();
        while ($("#hiddenLayersModify > div").length < targetCount)
            this.addLayer();
        this.sim.config.hiddenLayers.forEach(function (c, i) {
            $("#hiddenLayersModify .neuronCount").eq(i).text(c.neuronCount);
            $("#hiddenLayersModify > div").eq(i).children("select.activation").val(c.activation);
        });
    };
    return NeuronGui;
})();
var Simulation = (function () {
    function Simulation() {
        var _this = this;
        this.backgroundResolution = 10;
        this.stepNum = 0;
        this.running = false;
        this.runningId = -1;
        this.restartTimeout = -1;
        this.isCustom = false;
        this.statusIterEle = document.getElementById('statusIteration');
        this.statusCorrectEle = document.getElementById('statusCorrect');
        this.aniFrameCallback = this.animationStep.bind(this);
        var canvas = $("#neuralInputOutput canvas")[0];
        this.netviz = new NetworkVisualization(canvas, new CanvasMouseNavigation(canvas, function () { return _this.netviz.inputMode == 3; }, function () { return _this.draw(); }), this, this.backgroundResolution);
        this.netgraph = new NetworkGraph($("#neuralNetworkGraph")[0]);
        $("#learningRate").slider({
            min: 0.01, max: 1, step: 0.005, scale: "logarithmic", value: 0.05
        }).on('change', function (e) { return $("#learningRateVal").text(e.value.newValue.toFixed(3)); });
        this.neuronGui = new NeuronGui(this);
        for (var _i = 0, _a = Presets.getNames(); _i < _a.length; _i++) {
            var name_1 = _a[_i];
            $("#presetLoader").append($("<li>").append($("<a>").text(name_1)));
        }
        $("#presetLoader").on("click", "a", function (e) {
            var name = e.target.textContent;
            $("#presetName").text("Preset: " + name);
            _this.config = Presets.get(name);
            _this.setConfig();
            _this.isCustom = false;
            history.replaceState({}, "", "?" + $.param({ preset: name }));
            _this.initializeNet();
        });
        $("#dataInputSwitch").on("click", "a", function (e) {
            $("#dataInputSwitch li.active").removeClass("active");
            var li = $(e.target).parent();
            li.addClass("active");
            var mode = li.index();
            var modeSwitched = ((_this.netviz.inputMode == InputMode.Table) != (mode == InputMode.Table));
            _this.netviz.inputMode = mode;
            if (!modeSwitched)
                return;
            if (mode == InputMode.Table) {
                $("#neuralInputOutput > *").detach(); // keep event handlers
                $("#neuralInputOutput").append(_this.table.container);
                _this.table.loadData(_this);
            }
            else {
                _this.table.reparseData();
                $("#neuralInputOutput > *").detach();
                $("#neuralInputOutput").append(_this.netviz.canvas);
                _this.draw();
            }
        });
        var doSerialize = function () {
            _this.stop();
            console.log("ser");
            $("#urlExport").text(sim.serializeToUrl(+$("#exportWeights").val()));
        };
        $("#exportModal").on("shown.bs.modal", doSerialize);
        $("#exportModal select").on("change", doSerialize);
        this.deserializeFromUrl();
        this.table = new TableEditor($("<div class='fullsize'>"), this);
        this.run();
    }
    Simulation.prototype.initializeNet = function (weights) {
        if (this.net)
            this.stop();
        this.net = new Net.NeuralNet(this.config.inputLayer, this.config.hiddenLayers, this.config.outputLayer, this.config.learningRate, this.config.bias, undefined, weights);
        var isBinClass = this.config.outputLayer.neuronCount === 1;
        $("#dataInputSwitch > li").eq(1).toggle(isBinClass);
        var firstButton = $("#dataInputSwitch > li > a").eq(0);
        firstButton.text(isBinClass ? "Add Red" : "Add point");
        if (!isBinClass && this.netviz.inputMode == 1)
            firstButton.click();
        this.stepNum = 0;
        this.netgraph.loadNetwork(this.net);
        if (this.table)
            this.table.loadData(this);
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
        if (this.netviz.inputMode === InputMode.Table)
            this.table.updateRealOutput();
        else
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
        this.loadConfig(true);
        this.initializeNet();
    };
    Simulation.prototype.updateStatusLine = function () {
        var _this = this;
        var correct = 0;
        if (this.config.outputLayer.neuronCount === 1) {
            for (var _i = 0, _a = this.config.data; _i < _a.length; _i++) {
                var val = _a[_i];
                var res = this.net.getOutput(val.input);
                if (+(res[0] > 0.5) == val.output[0])
                    correct++;
            }
            this.statusCorrectEle.innerHTML = "Correct: " + correct + "/" + this.config.data.length;
        }
        else {
            var sum = 0;
            for (var _b = 0, _c = this.config.data; _b < _c.length; _b++) {
                var val_1 = _c[_b];
                var res = this.net.getOutput(val_1.input);
                var sum1 = 0;
                for (var i = 0; i < this.net.outputs.length; i++) {
                    var dist = res[i] - val_1.output[i];
                    sum1 += dist * dist;
                }
                sum += Math.sqrt(sum1);
            }
            this.statusCorrectEle.innerHTML = "Avg. distance: " + (sum / this.config.data.length).toFixed(2);
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
    Simulation.prototype.setIsCustom = function () {
        if (this.isCustom)
            return;
        this.isCustom = true;
        $("#presetName").text("Custom Network");
        for (var _i = 0, _a = ["input", "output"]; _i < _a.length; _i++) {
            var name_2 = _a[_i];
            var layer = this.config[(name_2 + "Layer")];
            layer.names = [];
            for (var i = 0; i < layer.neuronCount; i++)
                layer.names.push(name_2 + " " + (i + 1));
        }
        this.table.loadData(this);
    };
    Simulation.prototype.loadConfig = function (nochange) {
        if (nochange === void 0) { nochange = false; }
        var config = this.config;
        var oldConfig = $.extend({}, config);
        for (var conf in config) {
            var ele = document.getElementById(conf);
            if (!ele)
                continue;
            if (ele.type == 'checkbox')
                config[conf] = ele.checked;
            else if (typeof config[conf] === 'number')
                config[conf] = +ele.value;
            else
                config[conf] = ele.value;
        }
        if (oldConfig.simType != config.simType)
            config.data = [];
        if (this.net)
            this.net.learnRate = this.config.learningRate;
        if (!nochange)
            this.setIsCustom();
    };
    Simulation.prototype.setConfig = function () {
        var config = this.config;
        for (var conf in config) {
            var ele = document.getElementById(conf);
            if (!ele)
                continue;
            if (ele.type == 'checkbox')
                ele.checked = config[conf];
            else
                ele.value = config[conf];
        }
        $("#learningRate").slider('setValue', this.config.learningRate);
        $("#learningRateVal").text(this.config.learningRate.toFixed(3));
        this.neuronGui.regenerate();
    };
    Simulation.prototype.runtoggle = function () {
        if (this.running)
            this.stop();
        else
            this.run();
    };
    // 0 = no weights, 1 = current weights, 2 = start weights
    Simulation.prototype.serializeToUrl = function (exportWeights) {
        if (exportWeights === void 0) { exportWeights = 0; }
        var url = location.protocol + '//' + location.host + location.pathname + "?";
        var params = {};
        if (exportWeights === 1)
            params.weights = LZString.compressToBase64(JSON.stringify(this.net.connections.map(function (c) { return c.weight; })));
        if (exportWeights === 2)
            params.weights = LZString.compressToBase64(JSON.stringify(this.net.startWeights));
        if (this.isCustom) {
            params.config = LZString.compressToBase64(JSON.stringify(this.config));
        }
        else {
            params.preset = this.config.name;
        }
        return url + $.param(params);
    };
    Simulation.prototype.deserializeFromUrl = function () {
        function getUrlParameter(name) {
            var match = RegExp('[?&]' + name + '=([^&]*)').exec(window.location.search);
            return match && decodeURIComponent(match[1].replace(/\+/g, ' '));
        }
        var preset = getUrlParameter("preset"), config = getUrlParameter("config");
        if (preset && Presets.exists(preset))
            this.config = Presets.get(preset);
        else if (config)
            this.config = JSON.parse(LZString.decompressFromBase64(config));
        else
            this.config = Presets.get("Binary Classifier for XOR");
        var weights = getUrlParameter("weights");
        if (weights)
            this.initializeNet(JSON.parse(LZString.decompressFromBase64(weights)));
        else
            this.initializeNet();
    };
    return Simulation;
})();
///<reference path='../lib/typings/react/react-global.d.ts' />
///<reference path='../lib/typings/jquery/jquery.d.ts' />
///<reference path='Net.ts' />
///<reference path='Simulation.ts' />
///<reference path='Transform.ts' />
///<reference path='NetworkVisualization.ts' />
var sim;
$(document).ready(function () { return sim = new Simulation(); });
function checkSanity() {
    var out = [-0.3180095069079748, -0.2749093166215802, -0.038532753589859546, 0.09576201205465842, -0.3460678329225116,
        0.23218797637289554, -0.33191669283980774, 0.5140297481331861, -0.1518989898989732];
    var inp = [-0.3094657452311367, -0.2758470894768834, 0.005968799814581871, 0.13201188389211893, -0.33257930004037917,
        0.24626848078332841, -0.35734778200276196, 0.489376779878512, -0.2165879353415221];
    sim.stop();
    sim.config.inputLayer = { neuronCount: 2, names: ['', ''] };
    sim.config.hiddenLayers = [{ neuronCount: 2, activation: "sigmoid" }];
    sim.config.outputLayer = { neuronCount: 1, activation: "sigmoid", names: [''] };
    sim.net.connections.forEach(function (e, i) { return e.weight = inp[i]; });
    for (var i = 0; i < 1000; i++)
        sim.step();
    var realout = sim.net.connections.map(function (e) { return e.weight; });
    if (realout.every(function (e, i) { return e !== out[i]; }))
        throw "insanity!";
    return "ok";
}
//# sourceMappingURL=program.js.map