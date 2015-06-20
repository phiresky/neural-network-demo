var __extends = this.__extends || function (d, b) {
    for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p];
    function __() { this.constructor = d; }
    __.prototype = b.prototype;
    d.prototype = new __();
};
var Net;
(function (Net) {
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
            df: function (x) { return x * (1 - x); }
        },
        tanh: {
            f: function (x) { return tanh(x); },
            df: function (x) { return 1 - x * x; }
        },
    };
    Net.nonLinearity;
    function setLinearity(name) {
        Net.nonLinearity = NonLinearities[name];
    }
    Net.setLinearity = setLinearity;
    function makeArray(len, supplier) {
        var arr = new Array(len);
        for (var i = 0; i < len; i++)
            arr[i] = supplier();
        return arr;
    }
    // back propagation code adapted from https://de.wikipedia.org/wiki/Backpropagation
    var NeuralNet = (function () {
        function NeuralNet(counts, inputnames, bias, startWeight, weights) {
            var _this = this;
            if (bias === void 0) { bias = true; }
            if (startWeight === void 0) { startWeight = function () { return Math.random(); }; }
            this.connections = [];
            this.learnRate = 0.01;
            var nid = 0;
            this.inputs = makeArray(counts[0], function () { return new InputNeuron(nid, inputnames[nid++]); });
            var hidden = makeArray(counts[1], function () { return new Neuron(nid++); });
            this.outputs = makeArray(counts[2], function () { return new OutputNeuron(nid++); });
            this.layers = [this.inputs, hidden, this.outputs];
            this.bias = bias;
            if (bias) {
                var onNeuron = new InputNeuron(nid++, "1 (bias)", 1);
                this.inputs.push(onNeuron);
            }
            for (var i = 0; i < this.layers.length - 1; i++) {
                var inLayer = this.layers[i];
                var outLayer = this.layers[i + 1];
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
        NeuralNet.prototype.setInputs = function (inputVals) {
            if (inputVals.length != this.inputs.length - +this.bias)
                throw "invalid input size";
            for (var i = 0; i < inputVals.length; i++)
                this.inputs[i].input = inputVals[i];
        };
        NeuralNet.prototype.getOutput = function (inputVals) {
            this.setInputs(inputVals);
            return this.outputs.map(function (output) { return output.getOutput(); });
        };
        NeuralNet.prototype.train = function (inputVals, expectedOutput) {
            this.setInputs(inputVals);
            for (var i = 0; i < this.outputs.length; i++)
                this.outputs[i].targetOutput = expectedOutput[i];
            for (var _i = 0, _a = this.connections; _i < _a.length; _i++) {
                var conn = _a[_i];
                conn._tmpw = conn.getDeltaWeight(this.learnRate);
            }
            for (var _b = 0, _c = this.connections; _b < _c.length; _b++) {
                var conn = _c[_b];
                conn.weight += conn._tmpw;
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
        }
        NeuronConnection.prototype.getDeltaWeight = function (learnRate) {
            return learnRate * this.out.getError() * this.inp.getOutput();
        };
        return NeuronConnection;
    })();
    Net.NeuronConnection = NeuronConnection;
    var Neuron = (function () {
        function Neuron(id) {
            this.id = id;
            this.inputs = [];
            this.outputs = [];
        }
        Neuron.prototype.weightedInputs = function () {
            var output = 0;
            for (var _i = 0, _a = this.inputs; _i < _a.length; _i++) {
                var conn = _a[_i];
                output += conn.inp.getOutput() * conn.weight;
            }
            return output;
        };
        Neuron.prototype.getOutput = function () {
            return Net.nonLinearity.f(this.weightedInputs());
        };
        Neuron.prototype.getError = function () {
            var δ = 0;
            for (var _i = 0, _a = this.outputs; _i < _a.length; _i++) {
                var output = _a[_i];
                δ += output.out.getError() * output.weight;
            }
            return δ * Net.nonLinearity.df(this.getOutput());
        };
        return Neuron;
    })();
    Net.Neuron = Neuron;
    var InputNeuron = (function (_super) {
        __extends(InputNeuron, _super);
        function InputNeuron(id, name, input) {
            if (input === void 0) { input = 0; }
            _super.call(this, id);
            this.name = name;
            this.input = input;
        }
        InputNeuron.prototype.weightedInputs = function () {
            return this.input;
        };
        InputNeuron.prototype.getOutput = function () {
            return this.input;
        };
        return InputNeuron;
    })(Neuron);
    Net.InputNeuron = InputNeuron;
    var OutputNeuron = (function (_super) {
        __extends(OutputNeuron, _super);
        function OutputNeuron() {
            _super.apply(this, arguments);
        }
        OutputNeuron.prototype.getOutput = function () {
            return Math.max(Math.min(_super.prototype.weightedInputs.call(this), 0.999), 0.001);
            //return super.weightedInputs();
        };
        OutputNeuron.prototype.getError = function () {
            //let oup = Math.abs(NonLinearity.sigmoid(this.getOutput()));
            /*return NonLinearity.sigDiff(NonLinearity.sigmoid(oup)) *
                (this.targetOutput - oup);*/
            var oup = this.getOutput();
            return Net.nonLinearity.df(oup) *
                (this.targetOutput - oup);
        };
        return OutputNeuron;
    })(Neuron);
    Net.OutputNeuron = OutputNeuron;
})(Net || (Net = {}));
///<reference path='Net.ts' />
var NetworkGraph = (function () {
    function NetworkGraph(networkGraphContainer) {
        this.nodes = new vis.DataSet();
        this.edges = new vis.DataSet();
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
        this.graph = new vis.Network(networkGraphContainer, graphData, options);
    }
    NetworkGraph.prototype.loadNetwork = function (net) {
        if (this.net
            && this.net.layers.length == net.layers.length
            && this.net.layers.every(function (layer, index) { return layer.length == net.layers[index].length; })) {
            // same net layout, only update
            this.net = net;
            this.update();
            return;
        }
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
                width: Math.min(10, Math.abs(conn.weight * 2)),
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
    function NetworkVisualization(outputCanvas, trafo) {
        this.colors = {
            bg: ["#f88", "#8f8"],
            fg: ["#f00", "#0f0"],
            gradient: function (val) { return "rgb(" + [(val * 256) | 0, ((1 - val) * 256) | 0, 0] + ")"; }
        };
        this.canvas = outputCanvas;
        this.ctx = this.canvas.getContext('2d');
        this.trafo = trafo;
        this.canvasResized();
        window.addEventListener('resize', this.canvasResized.bind(this));
    }
    NetworkVisualization.prototype.drawDataPoints = function (data) {
        this.ctx.strokeStyle = "#000";
        for (var _i = 0; _i < data.length; _i++) {
            var val = data[_i];
            this.ctx.fillStyle = this.colors.fg[val.label | 0];
            this.ctx.beginPath();
            this.ctx.arc(this.trafo.toCanvas.x(val.x), this.trafo.toCanvas.y(val.y), 5, 0, 2 * Math.PI);
            this.ctx.fill();
            this.ctx.arc(this.trafo.toCanvas.x(val.x), this.trafo.toCanvas.y(val.y), 5, 0, 2 * Math.PI);
            this.ctx.stroke();
        }
    };
    NetworkVisualization.prototype.drawBackground = function (resolution, classify) {
        for (var x = 0; x < this.canvas.width; x += resolution) {
            for (var y = 0; y < this.canvas.height; y += resolution) {
                var val = classify(this.trafo.toReal.x(x), this.trafo.toReal.y(y));
                if (this.showGradient) {
                    this.ctx.fillStyle = this.colors.gradient(val);
                }
                else
                    this.ctx.fillStyle = this.colors.bg[(val + 0.5) | 0];
                this.ctx.fillRect(x, y, resolution, resolution);
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
    return NetworkVisualization;
})();
///<reference path='../lib/typings/jquery/jquery.d.ts' />
///<reference path='Net.ts' />
///<reference path='NetworkGraph.ts' />
///<reference path='NetworkVisualization.ts' />
var Simulation = (function () {
    function Simulation() {
        var _this = this;
        this.backgroundResolution = 10;
        this.stepNum = 0;
        this.running = false;
        this.runningId = -1;
        this.restartTimeout = -1;
        this.data = [
            { x: 0, y: 0, label: 0 },
            { x: 0, y: 1, label: 1 },
            { x: 1, y: 0, label: 1 },
            { x: 1, y: 1, label: 0 }
        ];
        this.config = {
            stepsPerFrame: 50,
            learningRate: 0.05,
            activation: "sigmoid",
            showGradient: false,
            bias: true,
            autoRestartTime: 5000
        };
        this.statusIterEle = document.getElementById('statusIteration');
        this.statusCorrectEle = document.getElementById('statusCorrect');
        this.aniFrameCallback = this.animationStep.bind(this);
        var canvas = $("#neuralOutputCanvas")[0];
        this.netviz = new NetworkVisualization(canvas, new CanvasMouseNavigation(canvas, function () { return _this.draw(); }));
        this.netgraph = new NetworkGraph($("#neuralNetworkGraph")[0]);
        $("#learningRate").slider({
            min: 0.01, max: 1, step: 0.005, scale: "logarithmic", value: 0.05
        }).on('slide', function (e) { return $("#learningRateVal").text(e.value.toFixed(2)); });
        this.reset();
        this.run();
    }
    Simulation.prototype.initializeNet = function () {
        //let cache = [0.18576880730688572,-0.12869677506387234,0.08548374730162323,-0.19820863520726562,-0.09532690420746803,-0.3415223266929388,-0.309354952769354,-0.157513455953449];
        //let cache = [-0.04884958150796592,-0.3569231238216162,0.11143312812782824,0.43614205135963857,0.3078767384868115,-0.22759653301909566,0.09250503336079419,0.3279339636210352];
        this.net = new Net.NeuralNet([2, 2, 1], ["x", "y"], this.config.bias);
        console.log("net:" + JSON.stringify(this.net.connections.map(function (c) { return c.weight; })));
        this.stepNum = 0;
        this.netgraph.loadNetwork(this.net);
    };
    Simulation.prototype.step = function () {
        var _this = this;
        this.stepNum++;
        for (var _i = 0, _a = this.data; _i < _a.length; _i++) {
            var val = _a[_i];
            var stats = this.net.train([val.x, val.y], [val.label]);
        }
        var correct = 0;
        for (var _b = 0, _c = this.data; _b < _c.length; _b++) {
            var val = _c[_b];
            var res = this.net.getOutput([val.x, val.y]);
            var label = (res[0] > 0.5) ? 1 : 0;
            if (val.label == label)
                correct++;
        }
        this.statusIterEle.innerHTML = this.stepNum.toString();
        this.statusCorrectEle.innerHTML = correct + "/" + this.data.length;
        if (correct == this.data.length) {
            if (this.running && this.restartTimeout == -1) {
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
    Simulation.prototype.draw = function () {
        var _this = this;
        this.netviz.drawBackground(this.backgroundResolution, function (x, y) { return _this.net.getOutput([x, y])[0]; });
        this.netviz.drawCoordinateSystem();
        this.netviz.drawDataPoints(this.data);
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
        this.draw();
    };
    Simulation.prototype.animationStep = function () {
        for (var i = 0; i < this.config.stepsPerFrame; i++)
            this.step();
        this.draw();
        if (this.running)
            this.runningId = requestAnimationFrame(this.aniFrameCallback);
    };
    Simulation.prototype.iterations = function () {
        this.stop();
        var count = +$("#iterations").val();
        for (var i = 0; i < count; i++)
            this.step();
        this.draw();
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
        Net.setLinearity(this.config.activation);
        if (this.net)
            this.net.learnRate = this.config.learningRate;
        this.netviz.showGradient = this.config.showGradient;
    };
    Simulation.prototype.randomizeData = function () {
        var count = 4;
        for (var i = 0; i < count; i++) {
            this.data[i] = { x: Math.random() * 2, y: Math.random() * 2, label: +(Math.random() > 0.5) };
        }
        this.draw();
    };
    Simulation.prototype.runtoggle = function () {
        if (this.running) {
            this.stop();
        }
        else {
            this.run();
        }
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
//# sourceMappingURL=program.js.map