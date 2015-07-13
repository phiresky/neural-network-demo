var sim;
$(document).ready(function () {
    Presets.loadPetersonBarney();
    sim = new Simulation(false);
});
function checkSanity() {
    // test if network still works like ages ago
    sim.loadPreset("Binary Classifier for XOR");
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
var __extends = this.__extends || function (d, b) {
    for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p];
    function __() { this.constructor = d; }
    __.prototype = b.prototype;
    d.prototype = new __();
};
// this neural network uses stochastic gradient descent with the squared error as the loss function
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
    var Util;
    (function (Util) {
        function makeArray(len, supplier) {
            var arr = new Array(len);
            for (var i = 0; i < len; i++)
                arr[i] = supplier(i);
            return arr;
        }
        Util.makeArray = makeArray;
    })(Util = Net.Util || (Net.Util = {}));
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
            this.inputs = Util.makeArray(input.neuronCount, function (i) { return new InputNeuron(nid++, input.names[i]); });
            this.layers.push(this.inputs.slice());
            for (var _i = 0; _i < hidden.length; _i++) {
                var layer = hidden[_i];
                this.layers.push(Util.makeArray(layer.neuronCount, function (i) { return new Neuron(layer.activation, nid++); }));
            }
            this.outputs = Util.makeArray(output.neuronCount, function (i) { return new OutputNeuron(output.activation, nid++, output.names[i]); });
            this.layers.push(this.outputs);
            this.bias = bias;
            for (var i = 0; i < this.layers.length - 1; i++) {
                var inLayer = this.layers[i];
                var outLayer = this.layers[i + 1];
                if (bias)
                    inLayer.push(new InputNeuron(nid++, "Bias", 1));
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
        // get root-mean-square error
        NeuralNet.prototype.getLoss = function (expectedOutput) {
            var sum = 0;
            for (var i = 0; i < this.outputs.length; i++) {
                var neuron = this.outputs[i];
                sum += Math.pow(neuron.output - expectedOutput[i], 2);
            }
            return Math.sqrt(sum / this.outputs.length);
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
            if (newval < 1 || newval > 10)
                return;
            targetLayer.neuronCount = newval;
            $("#" + name + "LayerModify .neuronCount").text(newval);
            sim.config.data = [];
            sim.setIsCustom(true);
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
        $("#outputLayerModify").children("select.activation").val(this.sim.config.outputLayer.activation);
        $("#inputLayerModify .neuronCount").text(this.sim.config.inputLayer.neuronCount);
        $("#outputLayerModify .neuronCount").text(this.sim.config.outputLayer.neuronCount);
        $("#layerCount").text(this.sim.config.hiddenLayers.length + 2);
    };
    return NeuronGui;
})();
var Presets;
(function (Presets) {
    var presets = [
        {
            name: "Default",
            stepsPerFrame: 50,
            learningRate: 0.05,
            showGradient: false,
            bias: false,
            autoRestartTime: 5000,
            autoRestart: false,
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
            iterationsPerClick: 1000,
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
            name: "Three classes test",
            iterationsPerClick: 500,
            hiddenLayers: [
                { "neuronCount": 4, "activation": "sigmoid" },
            ],
            inputLayer: { neuronCount: 2, names: ["x", "y"] },
            outputLayer: { neuronCount: 3, "activation": "sigmoid", names: ["A", "B", "C"] },
            data: [{ "input": [1.40, 1.3], "output": [1, 0, 0] }, { "input": [1.56, 1.36], "output": [1, 0, 0] }, { "input": [1.36, 1.36], "output": [1, 0, 0] }, { "input": [1.46, 1.36], "output": [1, 0, 0] }, { "input": [1.14, 1.26], "output": [1, 0, 0] }, { "input": [0.96, 0.97], "output": [1, 0, 0] }, { "input": [1.04, 0.76], "output": [1, 0, 0] }, { "input": [1.43, 0.81], "output": [1, 0, 0] }, { "input": [1.3, 1.05], "output": [1, 0, 0] }, { "input": [1.45, 1.22], "output": [1, 0, 0] }, { "input": [2.04, 1.1], "output": [1, 0, 0] }, { "input": [1.06, 0.28], "output": [1, 0, 0] }, { "input": [0.96, 0.57], "output": [1, 0, 0] }, { "input": [1.28, 0.46], "output": [1, 0, 0] }, { "input": [1.51, 0.33], "output": [1, 0, 0] }, { "input": [1.65, 0.68], "output": [1, 0, 0] }, { "input": [1.67, 1.01], "output": [1, 0, 0] }, { "input": [1.5, 1.83], "output": [0, 1, 0] }, { "input": [0.76, 1.69], "output": [0, 1, 0] }, { "input": [0.4, 0.71], "output": [0, 1, 0] }, { "input": [0.61, 1.18], "output": [0, 1, 0] }, { "input": [0.26, 1.42], "output": [0, 1, 0] }, { "input": [0.28, 1.89], "output": [0, 1, 0] }, { "input": [1.37, 1.89], "output": [0, 1, 0] }, { "input": [1.11, 1.9], "output": [0, 1, 0] }, { "input": [1.05, 2.04], "output": [0, 1, 0] }, { "input": [2.43, 1.42], "output": [0, 1, 0] }, { "input": [2.39, 1.2], "output": [0, 1, 0] }, { "input": [2.1, 1.53], "output": [0, 1, 0] }, { "input": [1.89, 1.72], "output": [0, 1, 0] }, { "input": [2.69, 0.72], "output": [0, 1, 0] }, { "input": [2.96, 0.44], "output": [0, 1, 0] }, { "input": [2.5, 0.79], "output": [0, 1, 0] }, { "input": [2.85, 1.23], "output": [0, 1, 0] }, { "input": [2.82, 1.37], "output": [0, 1, 0] }, { "input": [1.93, 1.9], "output": [0, 1, 0] }, { "input": [2.18, 1.77], "output": [0, 1, 0] }, { "input": [2.29, 0.39], "output": [0, 1, 0] }, { "input": [2.57, 0.22], "output": [0, 1, 0] }, { "input": [2.7, -0.11], "output": [0, 1, 0] }, { "input": [1.96, -0.2], "output": [0, 1, 0] }, { "input": [1.89, -0.1], "output": [0, 1, 0] }, { "input": [1.77, 0.13], "output": [0, 1, 0] }, { "input": [0.73, 0.01], "output": [0, 1, 0] }, { "input": [0.37, 0.31], "output": [0, 1, 0] }, { "input": [0.46, 0.44], "output": [0, 1, 0] }, { "input": [0.48, 0.11], "output": [0, 1, 0] }, { "input": [0.37, -0.1], "output": [0, 1, 0] }, { "input": [1.03, -0.42], "output": [0, 1, 0] }, { "input": [1.35, -0.25], "output": [0, 1, 0] }, { "input": [1.17, 0.01], "output": [0, 1, 0] }, { "input": [0.12, 0.94], "output": [0, 1, 0] }, { "input": [2.05, 0.32], "output": [0, 1, 0] }, { "input": [1.97, 0.55], "output": [1, 0, 0] },
                { "input": [0.7860082304526748, 2.5761316872427984], "output": [0, 0, 1] }, { "input": [-0.09053497942386843, 2.3909465020576133], "output": [0, 0, 1] }, { "input": [-0.23868312757201657, 2.0329218106995888], "output": [0, 0, 1] }, { "input": [-0.32510288065843634, 1.748971193415638], "output": [0, 0, 1] }, { "input": [-0.6707818930041154, 1.4526748971193417], "output": [0, 0, 1] }, { "input": [-0.3991769547325104, 1.094650205761317], "output": [0, 0, 1] }, { "input": [-0.2263374485596709, 0.6131687242798356], "output": [0, 0, 1] }, { "input": [-0.2263374485596709, -0.42386831275720144], "output": [0, 0, 1] }, { "input": [-0.13991769547325114, -0.6584362139917693], "output": [0, 0, 1] }, { "input": [1.5390946502057612, -1.0658436213991767], "output": [0, 0, 1] }, { "input": [2.193415637860082, -1.0781893004115224], "output": [0, 0, 1] }, { "input": [2.6502057613168724, -0.9176954732510286], "output": [0, 0, 1] }, { "input": [3.193415637860082, -0.6460905349794236], "output": [0, 0, 1] }, { "input": [3.526748971193415, -0.42386831275720144], "output": [0, 0, 1] }, { "input": [3.4403292181069953, 0.329218106995885], "output": [0, 0, 1] }, { "input": [3.4773662551440325, 1.0452674897119343], "output": [0, 0, 1] }, { "input": [3.6625514403292176, 1.2798353909465023], "output": [0, 0, 1] }, { "input": [2.8847736625514404, 2.946502057613169], "output": [0, 0, 1] }, { "input": [1.4156378600823043, 2.5514403292181074], "output": [0, 0, 1] }, { "input": [1.045267489711934, 2.526748971193416], "output": [0, 0, 1] }, { "input": [2.5144032921810697, 2.1563786008230457], "output": [0, 0, 1] }, { "input": [3.045267489711934, 1.7983539094650207], "output": [0, 0, 1] }, { "input": [2.366255144032922, 2.9341563786008233], "output": [0, 0, 1] }, { "input": [1.5020576131687242, 3.0576131687242802], "output": [0, 0, 1] }, { "input": [0.5390946502057612, 2.711934156378601], "output": [0, 0, 1] }, { "input": [-0.300411522633745, 2.5761316872427984], "output": [0, 0, 1] }, { "input": [-0.7942386831275722, 2.563786008230453], "output": [0, 0, 1] }, { "input": [-1.1646090534979425, 1.181069958847737], "output": [0, 0, 1] }, { "input": [-1.1275720164609055, 0.5637860082304529], "output": [0, 0, 1] }, { "input": [-0.5226337448559671, 0.46502057613168746], "output": [0, 0, 1] }, { "input": [-0.4115226337448561, -0.05349794238683104], "output": [0, 0, 1] }, { "input": [-0.1646090534979425, -0.7325102880658434], "output": [0, 0, 1] }, { "input": [0.4650205761316871, -0.8436213991769544], "output": [0, 0, 1] }, { "input": [0.8106995884773661, -1.164609053497942], "output": [0, 0, 1] }, { "input": [0.32921810699588466, -1.3004115226337447], "output": [0, 0, 1] }, { "input": [1.1687242798353907, -1.127572016460905], "output": [0, 0, 1] }, { "input": [2.1316872427983538, -1.362139917695473], "output": [0, 0, 1] }, { "input": [1.7119341563786008, -0.6954732510288063], "output": [0, 0, 1] }, { "input": [2.5267489711934155, -0.8930041152263373], "output": [0, 0, 1] }, { "input": [2.8971193415637857, -0.8930041152263373], "output": [0, 0, 1] }, { "input": [2.6378600823045266, -0.6460905349794236], "output": [0, 0, 1] }, { "input": [3.2427983539094645, -0.5349794238683125], "output": [0, 0, 1] }, { "input": [3.8477366255144028, 0.02057613168724303], "output": [0, 0, 1] }, { "input": [3.390946502057613, 0.02057613168724303], "output": [0, 0, 1] }, { "input": [3.4403292181069953, 0.3415637860082307], "output": [0, 0, 1] }, { "input": [3.7983539094650203, 0.6502057613168727], "output": [0, 0, 1] }, { "input": [3.526748971193415, 0.983539094650206], "output": [0, 0, 1] }, { "input": [3.452674897119341, 1.4526748971193417], "output": [0, 0, 1] }, { "input": [3.502057613168724, 1.7242798353909468], "output": [0, 0, 1] }, { "input": [3.415637860082304, 2.205761316872428], "output": [0, 0, 1] }, { "input": [2.736625514403292, 2.292181069958848], "output": [0, 0, 1] }, { "input": [1.9465020576131686, 2.403292181069959], "output": [0, 0, 1] }, { "input": [1.8230452674897117, 2.60082304526749], "output": [0, 0, 1] }, { "input": [3.008230452674897, -1.288065843621399], "output": [0, 0, 1] }, { "input": [1.699588477366255, -1.016460905349794], "output": [0, 0, 1] }, { "input": [2.045267489711934, -0.9053497942386829], "output": [0, 0, 1] }, { "input": [1.8724279835390945, -1.2263374485596705], "output": [0, 0, 1] }]
        },
        { name: "Peterson and Barney (male)",
            parent: "Three classes test",
            stepsPerFrame: 6,
            iterationsPerClick: 50,
            inputLayer: { neuronCount: 2, names: ["F1", "F2"] },
            outputLayer: { neuronCount: 10, "activation": "sigmoid", names: "IY,IH,EH,AE,AH,AA,AO,UH,UW,ER".split(",") }
        },
        { name: "Peterson and Barney (all)",
            parent: "Peterson and Barney (male)"
        },
        {
            name: "Auto-Encoder for linear data",
            stepsPerFrame: 1,
            iterationsPerClick: 10,
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
            "stepsPerFrame": 50,
            "learningRate": 0.01,
            "iterationsPerClick": 200,
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
        },
        { "name": "Auto-Encoder 4D", "learningRate": 0.05, "data": [{ "input": [1, 0, 0, 0], "output": [1, 0, 0, 0] }, { "input": [0, 1, 0, 0], "output": [0, 1, 0, 0] }, { "input": [0, 0, 1, 0], "output": [0, 0, 1, 0] }, { "input": [0, 0, 0, 1], "output": [0, 0, 0, 1] }], "inputLayer": { "neuronCount": 4, "names": ["in1", "in2", "in3", "in4"] }, "outputLayer": { "neuronCount": 4, "activation": "sigmoid", "names": ["out1", "out2", "out3", "out4"] }, "hiddenLayers": [{ "neuronCount": 2, "activation": "sigmoid" }], "netLayers": [{ "activation": "sigmoid", "neuronCount": 2 }, { "activation": "linear", "neuronCount": 1 }, { "neuronCount": 2, "activation": "sigmoid" }] }
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
        console.log("loading preset chain: " + chain.map(function (c) { return c.name; }));
        return JSON.parse(JSON.stringify($.extend.apply($, chain)));
    }
    Presets.get = get;
    function printPreset(sim, parentName) {
        if (parentName === void 0) { parentName = "Default"; }
        var parent = get(parentName);
        var outconf = {};
        for (var prop in sim.config) {
            if (sim.config[prop] !== parent[prop])
                outconf[prop] = sim.config[prop];
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
    function loadPetersonBarney() {
        function parseBarney(data) {
            var relevantData = data.filter(function (row) { return row[3] == 1; }).map(function (row) { return ({
                input: row.slice(0, 2),
                output: Util.arrayWithOneAt(10, row[2])
            }); });
            normalizeInputs(relevantData);
            presets.filter(function (p) { return p.name === "Peterson and Barney (male)"; })[0].data = relevantData;
            var relevantData2 = data.map(function (row) { return ({
                input: row.slice(0, 2),
                output: Util.arrayWithOneAt(10, row[2])
            }); });
            normalizeInputs(relevantData2);
            presets.filter(function (p) { return p.name === "Peterson and Barney (all)"; })[0].data = relevantData2;
            //presets.forEach(preset => preset.data && normalizeInputs(preset.data));
        }
        // include peterson_barney_data for faster page load
        var dataStr = "NrBMBYBpVAOSCMiC6kzwgBkStBmATmkz2l2DyQQI3PCIVgHZJSFVgA2Np7KdtAFYGoQZDECuoRHmwSOTfphadysbAkyrVk8PwTT4ujcshGOERJigMLLAqVv4WCQdM10deNhovSY7uSgSKB4OnRUsETuHODusGIxQhpMqmwcnAycSPwKGpmQuWicEtaQOhkuHhVC8ILwLhyCqs2QjWjgYgRQ5h1IsKS96OVdQdqqTsCCVMFW5NOICD5BiYw4khAAdFChGnQaDmTGiATR8xKwSOlC/Jes5NyITDkPgptGTG9iRVwIm+4IFoPJTwGrAJjuTBEMGCMTMNrzHpQdrAcA6VxmOhvHQkTGSPA2UaSQQaPAuXxCNjeOb47AwfhBDCddaxHaYRLzFxMeDXKZfRBRe6ST6IbKFchMFz1cWSTg6UByCUSapqDQGcqIsxiFGwzZwogotGQAiqIZGzTYIZ4eAEIkca16rXG+YMcAmc6OhDgZb4kKUFloUCqGASOhs4N0NiwWjEowEK7nJ4sXkDJ7SH4iwFEH6PJaK2XKljyNCpRAwDXEmwsHVwhp0ck9OhEU54+02u34QbYSatVxUClTKjmgeUf7QPCh/GkCBsOj+TB08ij6cLo4ZDSwNKcgWkXmcJibHT7mUKUCeziwTYsDNn+VFiX6bBgzqHqzwM3Kxu+zaDPDvZ0bJeNp/ja8yYD+MjgaUYGengkJLv4cEBhQIRwMh4AhGUSS/NsApQLyigCnwEoHhcLyyvwojiK8xqgpGbTSDq/TYCi1pWDY255jSTScLB5JBH8GBsVQFjgcGSxrsU/CyNRspntJYjFlMB42tWnhXos/T1p65ZDKA9hIJMQbGqRAGxDobrcR0Z4SBhVkoa+MB7IB0D6Z6WICv4ka4YwHKxmOCAQo6vL1K+jDYHuKTfMCmxUOqSm8P+mhvFAYLZLhsDgP+KKcJaqg6iYlpgVYRVTj5GKTKOkLSD2gmQZ6A4kj5sweBswbGSJgaJAq6F0lhS6XtEq7YSSArJsVgWmXuR7RXJPm8CexQSOWSn7gFxkJaUdHCge5K3hWTTvix8yXnCfwGuphkEBBentmZga8RMNUTQS9mCH+7hudBGxZRgMDuRsIHQIFAOxFBJw2ZJqJ/Do3RQ0FEgLRmVAlItUgaZoQNKa0mUIro11ba2HQ+IMQQGhMS5/HCHaDrFiDNHTrVNGwzJM4G/h2Z1YA7Ho6EXDGbYnNO8wzUKPFpmLxQfOmDxsBeaOEQYqX7GWpWxJ9pO0sa3ZjFYhkq16332hZ/Zk65DC6zAFv4gwtpQ3KoqkDmLiPDmGgrQ87tIKtVRPvMSAQnjsQNAHYZWJr9ras9+INBxuhUvxujSTiQRnkJk5+OOGcdFliSYNIvLgCwCoFxKZ4o/AGaI+RGSXjiKkHUI4yOjWIxBzn/40ET0PGl+FgfeU10UxsZ7SHKjqTN6AUyWzqLuDPA6hMDzkWGyK/4LxEgEIuWsKjsDw6OoaMOxaskZBMQLEk+jGJgtrEGhHG9tH3D0MfdYD5XHsTLT65mQfZMlQj8i5oA80A1HT0jHNhN0GkQxQyamyIgIUspsC7nuOMbxK4SnAkYXGztYLSASuBaS11SBKVgJDBAgJ/xglxgYJAz4UFlgPI/VEbwRB127oQWCOtjhLGjszAKo1Z7vSSsOMYuErbIVEI5WQoN8C4IFvgLKsNhbEioNySWUx3BMF3LLJMx85ZZRlrKcCMwWFnxLOBHw10x4ixbnWfySJNgXV0ACNW1l1oeIcoTSq+0qGQ0nllaS7pdC8Q4UlReIRWqp2IEYRkxAd72ioHbaBLh4xaM4PwTgJiMgSGlDmOK+YMgAkIZ4N+hoVCuPtIxFgekGH2CCMiRwS4GBIVnnEMsShyDUGgKjLmvT9LrwoFyPy9oXBHxGnUm+GxSi8IsIZYext/6zzJCsxeqx4ltVclQA228oYThOEorgRgi7H3cAUh47jLFTHytU4o/RkRLlNKaFWuNbrlEabMt+k9Tmm10C4aks80JYGQoA9pHkFRjJuCcCKmpqClzkkRY+DAKHHyMKAIgCV9AMJotGduqI2ASTNESuECSWxGQYKuSe88jYdAkIc1ZdIZ5Lm6gyYUsNcnFAmbNBQyptCJgIHUpcT55mBjIV88Wrh4KVnponVey9pE7AxCA3Bxd6KBRCBKKm9NiLCkMmIZ82hQ4AraCa+0zZvFGgIIYFYg9376Tbj2SiRhGozFdQkiA2d0D/lEMMyyCpjkOmVVq2UuDeVS09JonMXh7w7UWMrK+ZhjqmJKgS4SWhu5nIMLa/E8lk0OqIEfHsXg6VcA0NmjZwNrYWBSnE+RFBUG8XVfHE4+8wmSI8CFYhJwCZaOEJ6fZzsBSqBzJ9YpxQiVnjjXyxYArhT6H5Ea5+BLLKaFzd/KwacuG2LTZVfUndEQY1eiIv4n0PU/UcowBtQZYFoS5p03YBzLwhqhuAOMiKSlJQGBpMdSUZFQFWn8FwMjFL+zaCmpoPbPivkNJvY0Rb1I1QPK8vNuFEqgRjhpbkr5fliN4he2IB53BogakEUyCpwgbGLvnMFWUhpqPtPwOG2ELxJVysfHQghryJioWB/VAV864TSuSbaTQuh/FIK3QUrEcTbr0tdGwLSXKnyCWWMt7650AP8ICaRzbXKziTk8aFFAGCaJGpQo+IUdFiGmqKPVp5RRYNTQYfjHBNxljE03Mw5ragZSebofkQqs3YnYlwuEAiN6weI+/IuAVvQAL/NOOR3q4KOQ6sCJ4NcHlRq5VMS8RLvYkQxoFRu4JCyOmfEDIVjozQ1SBlaJLBa/FmC/h0Iw7SBzvogksP8Zbb3FxZSPcc/q/yYXbcshUoamhr37bbCN4IiWkGfHMrhTWf00zQiKPDHXQmxDaYR/AZ7xxbPtMy719BEn21myFYB8I9zXRdtlk5PluC4RzE1gwl8FBsAVBBMEMHSmVZgtWfkOoGsw1XaFzQL9URGHVFaVbPY/xtPlTcFxMg3i/0DLuv1enZEuCXDFgIBzViMaWk8IGMaetordjIGdJY+tWBUNqnraw0qQ2YC3OciwrWlPC7Rd+SF4yHuJHWr0h24c+Sav1zhEAdUgLpBhBtupUFM73PwSUEEcy3CgPgz6MXVqYK6XTBK8OWfCiAppYH+NedjjNLuzQu0uE1SWejqk/yoNJTkVS1p0BSwhHMr66k/hYh+loKAcTrk4WR6EMXLt0B3MHCwbH4AHmEXilT295aipU8LQMLn9zv2/ap+42YKTie+i90xKn5kwWMDB2NEFyvUwNzQVLzoTMdI7D+/CKnzF/u9i1+DKNUPVfQVj6kCN+4Wfksx/czplPChlUy1T0fLiiQFAWUL2gBIiw7Fr4xQfv++zi5QbCw3joVKiq1+7BxWv/QKa1/4L2bv1l7NzH79EfPLeQW5hb4+uWJPqfIwE7C3l3OFJngoEOGBqnnbAYHRGviBgKlnoZMdLPq1giM/lYGSrXpCLgYLB5tEPaFJqMKXqcu3jNvTCnF/pIuvr/sXPioHu/jOGQHgeOOqmvu4MFjsO5rNrwcUI9KrK+Jvt5qWGfmIWpLXj4DfiQZgcQYGO+GQbEFGO6OwYFN0qnocnAEPiQb6vCMwaiHSIIPvOwUGBHk0HVBAONFnkrvPiWBYgqMREgRAh4AIeCC6tRB3saPlC3qIqaNqAAfmjDjXk0JCK8loQ0I0lociMoVfj1ocB4FoZDEMjgFoSEJmC3todKJPoQIPmweLKEHoqXv4KNNOEnmFDwS3hQp6B5u4RhogN1rAZUGFKNKIWnu7CXszAWhIWwo0REUHi4JZJfn0ZoKBLfnun4cEfnm/q3hjOvrMTJJQLoZ4hgGXpPrzEGKYWEb6iYTPhwCxlRO4Wik5FAbvpDobF4QcfKIgWEWYHWKXgeD0BdFoZaHEYSgFEwL7lobDGoXocXD/jEf7oELXk8RwQUR0BRgGlMbIluOQeOHCRkFkGQtUbnDIH+MrIfv+tCC3m9oZGeBXqXkwl3L0aIoCgMe1jIGiSMWumsAoaZmFOqPSazFkp/sHBjoCMsP3ikQHr/tOHvoYXEBjqIBGI8cKWXuUZSI5BKdUQSa5KvknoqvUfkjvott7qqXQjJMGE0AeEWqFqSWidvNzhMa5s6Lfg1OMUHofDQVQYFB6l/qwKuIYWlokPsoYaPmspPqWKEDyC3ooOlkgi3haElC2O4SEU4WcWnmwHse0W6QsLGYCkwtqRTqfriZeOSL4cPiVvpjSXCKZMyfVhPLiaPMZmyUtJ8abP3hRvQf3miToWkYIjsKPjCRgKyV6cyr6XnokJppKcAOurCpGcGXnNFF2bqlcbvrZKqTVhijiXAUzgYMtn4SxIEbPvbiuZHIsMKrXqtiMRODQvfmdm0O8W4NQWWa3mOUkf3I5F8Q2YGMYv7oTs/pIrqByGYQtJPm9jsC6fscUDgtAHFrZkGZgJhKOuARgLpscXVJce0ZmGhO0WigXhqHAWqMfjnM/BjqSfpoDlgY3lQuDDSZCDrFocqEReyVKH8R0Ppl6PBD8dADareRQDsHQn4X8E2ZwefuYRCVwN2X2r2f2QqGAanqfKcfUVbl6LxGUnnn+FxqqcGa6DqsmejA2MaefohTSeiAQe/hSlke+KQmaRkCjFyYdFGhRT6q6SsShHRRZZpkGOxUIIkGsr2SeaEPhH6dEGsMccqqBWvvwJZDBV4CichWWLOXkDgThSWJCA8baT1CMWXmpQAd2JaWhdpY8VdpeUIJEJQcov7qGLRfpH3o3qEGcFnqUaXCVf+a5UJVQsDN5RwE7icLVbvjikhe5sqKheeXfDpRUgldXrue+BSY2m0E/huc8PpfZU8PaeuKWelRQIhAVfgD2j6QxR6YBbYdHj+Wnp3vmvUSRoBn6eBEMfeHniYLcb+WYJmVNamUSWFFCKEZRetI2LRYcMyY+GNVML5aZeaIyrMTAsGBxoKcGIAkESNhAn4fHvYVMEDMEFNNURcFjpGamJATtWOfBRWlOQrmhIpUFESkPC1XHg1BYr0f9R4KSZ9FpFmRiqZLueSKRZ2CZG9b2P9bMdMNLpoYLHAPEpEdAAhQAY4Q7JPgzMXEaE5eXP+Yibvn+QqDxmvqSHtT5WziFSWPoCgU0IpghuFVLjYLIc4MaE1vSd+dtgARIPLDNUhPpLlZHNecAjpb6jklAkEY5EaALbjlsGVQZUlCzXUMWT5Alv5YyadeVlYPBopZ0vFcPmWANWSJ6AYLhPSQgtQWRqXpJnKpElkW8IwRZa9HAPNXDn0ouFmYkAOhYYzrAt+b2SaEqUBdOD1O4fRX2GOLGRXP+LGSIH+CznAemZuWOIpaFBmdlH4RMC8erOxEWVoc2MNZRSOmefHYwDRQqvlQxUAQKVkedNAM2ckc+fne7YDZ2e7Q5RDevieTBackdQcX2ksLuopSAZZIpSULhsEhrYHBXb0d4A1PRndXPCVBjruWeEWs7vrVrW9Wuhmj1UrGeXAI5DWRYINNzZnQVv+V5JEZItCeHU+r2ZuHTMEM4e5rjXvJGXfcreOVwAwNQI6PBeiNYnjR0TdSrb+bhqHOQTdBddlfsjSb7B/XIq6oElkb/YsG8PrKXp9kKVlZDZ6MrnPQtbAseM6TsICWHqXVvdytPr2akDwkQ7ojIFirDcwlQ9I2UL0UKeRa+OpfThww3F6JDvraci0uHYhTNZzNKKbS+tzTnVHbZk6QAX9fBhsfppxR+TzBDY8KIMUQoGrV6FlKfWgNOTIBJVQw4D1ocljSdY/ZCEPckGWDqrFe7I9URpMVoQ3HbMyX8mefQD1o44sf4G6b/t/hbVE1vNLQcRgpGZKFSUQ+9GYFIZdegayMFUAziEleWhebMWKNKuA62RbP3hgIrLzWvYo6iDzMXYtso36Q5ZVe5sGKYMcbbAqQ4fTHLYqRY+oyIIaglTQn2gY4Qmk+eQuR/ZpjczSavclG9XkZoBPZ/YFBI4OE5uA4wSCbWtzZzeMmvcLBMUGHM22WEBtajKweXX1NmNUeJI1YHa4Ps7s64IFQcbSlQ4HFQgw6FXhRrdmjFTM1QohlzV3AA8aM9II6KPxKXikJ8/6PpAyP3rE5xYKQ5VqqCVAo6pPncIwEi4KEjbiW8YS/yHcAYyYFuZuk8yMQPsePrctjY6pbmD9QwOU546wCyngGDKwGuqwHVcNDJPhHVdVVmNRDq4IbzgiJa1PoczyD0/skgrq6jM6w8tPl2pusENBLa0hBOHsL67Zvq7a6yXBIuCG+4+G61VZfcLa1QtOK0Ca2gFQvwQa8mxJBoRa4axuIqLa72LzA63Ho0WSoG55s6L64Q264M5AZ6woqwF6DgL61cIymm+eUGBGLa0wW0a2wKQtLZgcdJHRHGz1K4H7La7BYESG/mZOwZWOYW9c2svOzPGhFW8AzrHmxoHsbW/lqKPBLa3ZGG428PQSAG4a0yFgnG+yAg+KHG61EaGkBkApJiPuwCPrBuw1KMCG8xt0iG9Eu6Pu8GPQfu9/rOLa6khiEm32cNBBz2x8LmwcfkPB8m1e/whqHm8qH2vO7CK01hyReWxUYjODNu6NYFP+0xnq6e0IEgiM62/LHBByCG8GPsruHVZYKkZB086INFJ2yRveJewdmhwc+WMmI+8zja+fDE8+z9nO7R9clW/9Su7R/HssOO+1D+3IfR0e9lcsVp32Q2zAKBbe6hGAeO1KHtZ2xWlo2Bx9YJ5SOdeJ0IDVKHHmzMCW3cSm1J0/A7FW/6Bo1W9h6O3MPu0YP0hFIeZiqGL6wXC1K29h0GFquO8YXCiG8yluGB0LcRGB91DxuOyF2BuO6tiJzCl6KTHm9Ge+K26HaNKuwyi0nmxMh6i56KKbLayCjzU2xpJQJQpVyPl5C+/+bQBGwo7Gy0f4F9CNxLSGdx+5jRpl3kGFHZP2xFY0V7Yl/vg58Q1SxtwBepp56iPKFpPu5QwSz57xI4M9Pu28Bco15YfTD69A/WwyL6xFMG4dH0oBWB36DlwccltN2i6jEt5t4CEh1wFUL4eO59ODyq42Jd40TDxuVQhTL6zYvh8TMitu9mnQmF3TbIJF+R9nbp6PoHAXFNebWQOO0zl6hN+CGsWlwO8QDLJe4Noz+s/TCielyjZV66K8vVzILavu0Meu0C4j6j184CCnMF08GR3W4e1cBuZp3L3W5ih23cZilG0o/pIGUzydje6a8lizxFZ6GXoD7i47LZ3pyzGO7aaYEV6MYclh+4BxvO+aNapVykEL8Vy09u3sVCkF4eQr62+N3BPErz3AAx+yYkEE7R4kPiix0W2r9T1xDABe3VYAhBT2/kuZ3wfTKt6x1SFb8h63eb3vnvrb1u50vO/9QW62y869D581f53LPd+NeL37zrRF7p56Rkor7TIkIRCT+1hAhxgP1wAm6XHm947zHH8AMFgJbr1EzqsECZ+5ijCDyAZCw+0W4F1hxoh79ogFIjdj5Dej0f4cnBE90xrhF14T9/irzCgnyP6mGT9P5MqG62x51TxxxJFx/PzP9GVn3U1pZZt02roFnJ91VgbdEaVEW3rmH9CV8qA2HB3rt385qgOI47W4M3y4CHwVOyAZAEAA";
        parseBarney(JSON.parse(LZString.decompressFromBase64(dataStr)));
        return;
        $.get("lib/peterson_barney_data").then(function (strData) {
            var cols = {
                gender: 0,
                speaker: 1,
                phonemeNum: 2,
                phonemeAscii: 3,
                F0: 4,
                F1: 5,
                F2: 6,
                F3: 7
            };
            var isNum = [true, true, true, , true, true, true, true];
            var data = strData.split("\n")
                .filter(function (row) { return row.indexOf("#") !== 0; })
                .filter(function (row) { return row.trim().length > 2 && row.indexOf("*") < 0; })
                .map(function (row) { return row.split(/\s+/).map(function (n, i) { return i in isNum ? parseFloat(n) : n; }); });
            var relevantData = data.map(function (row) {
                return [+row[cols.F1] / 10, +row[cols.F2] / 10, +row[cols.phonemeNum], +row[cols.gender]];
            });
            parseBarney(relevantData);
        });
    }
    Presets.loadPetersonBarney = loadPetersonBarney;
    function normalizeInputs(data) {
        var i = Util.bounds2dTrainingsInput(data);
        data.forEach(function (data) { return data.input = [(data.input[0] - i.minx) / (i.maxx - i.minx), (data.input[1] - i.miny) / (i.maxy - i.miny)]; });
    }
})(Presets || (Presets = {}));
;
var Simulation = (function () {
    function Simulation(autoRun) {
        var _this = this;
        this.stepNum = 0;
        this.frameNum = 0;
        this.running = false;
        this.runningId = -1;
        this.restartTimeout = -1;
        this.isCustom = false;
        this.averageError = 1;
        this.constructed = false;
        this.statusIterEle = document.getElementById('statusIteration');
        this.statusCorrectEle = document.getElementById('statusCorrect');
        this.aniFrameCallback = this.animationStep.bind(this);
        $("#learningRate").slider({
            min: 0.01, max: 1, step: 0.005, scale: "logarithmic", value: 0.05
        }).on('change', function (e) { return $("#learningRateVal").text(e.value.newValue.toFixed(3)); });
        for (var _i = 0, _a = Presets.getNames(); _i < _a.length; _i++) {
            var name_1 = _a[_i];
            $("#presetLoader").append($("<li>").append($("<a>").text(name_1)));
        }
        $("#presetLoader").on("click", "a", function (e) {
            var name = e.target.textContent;
            _this.loadPreset(name);
        });
        var doSerialize = function () {
            _this.stop();
            $("#urlExport").text(_this.serializeToUrl(+$("#exportWeights").val()));
        };
        $("#exportModal").on("shown.bs.modal", doSerialize);
        $("#exportModal select").on("change", doSerialize);
        this.neuronGui = new NeuronGui(this);
        this.netviz = new NetworkVisualization(this);
        this.netgraph = new NetworkGraph(this);
        this.errorGraph = new ErrorGraph(this);
        this.table = new TableEditor(this);
        this.weightsGraph = new WeightsGraph(this);
        this.leftVis = new TabSwitchVisualizationContainer($("#leftVisHeader"), $("#leftVisBody"), "leftVis", [
            this.netgraph, this.errorGraph, this.weightsGraph]);
        this.rightVis = new TabSwitchVisualizationContainer($("#rightVisHeader"), $("#rightVisBody"), "rightVis", [
            this.netviz, this.table]);
        this.deserializeFromUrl();
        this.leftVis.setMode(0);
        this.rightVis.setMode(0);
        this.constructed = true;
        this.onFrame(true);
        if (autoRun)
            this.run();
    }
    Simulation.prototype.initializeNet = function (weights) {
        console.log("initializeNet(" + weights + ")");
        if (this.net)
            this.stop();
        this.net = new Net.NeuralNet(this.config.inputLayer, this.config.hiddenLayers, this.config.outputLayer, this.config.learningRate, true, undefined, weights);
        this.stepNum = 0;
        this.errorHistory = [];
        this.leftVis.onNetworkLoaded(this.net);
        this.rightVis.onNetworkLoaded(this.net);
        if (this.constructed)
            this.onFrame(true);
    };
    Simulation.prototype.step = function () {
        this.stepNum++;
        for (var _i = 0, _a = this.config.data; _i < _a.length; _i++) {
            var val = _a[_i];
            this.net.train(val.input, val.output);
        }
    };
    Simulation.prototype.onFrame = function (forceDraw) {
        this.frameNum++;
        this.calculateAverageError();
        this.rightVis.currentVisualization.onFrame(forceDraw ? 0 : this.frameNum);
        this.leftVis.currentVisualization.onFrame(forceDraw ? 0 : this.frameNum);
        this.updateStatusLine();
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
        this.initializeNet();
        this.onFrame(true);
    };
    Simulation.prototype.calculateAverageError = function () {
        this.averageError = 0;
        /*for (let val of this.config.data) {
            let res = this.net.getOutput(val.input);
            let sum1 = 0;
            for (let i = 0; i < this.net.outputs.length; i++) {
                let dist = res[i] - val.output[i];
                sum1 += dist * dist;
            }
            this.averageError += Math.sqrt(sum1);
        }
        this.averageError /= this.config.data.length;*/
        for (var _i = 0, _a = this.config.data; _i < _a.length; _i++) {
            var val = _a[_i];
            this.net.setInputsAndCalculate(val.input);
            this.averageError += this.net.getLoss(val.output);
        }
        this.averageError /= this.config.data.length;
        this.errorHistory.push([this.stepNum, this.averageError]);
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
            this.statusCorrectEle.innerHTML = "Error: " + (this.averageError).toFixed(2);
        }
        this.statusIterEle.innerHTML = this.stepNum.toString();
        if (correct == this.config.data.length) {
            if (this.config.autoRestart && this.running && this.restartTimeout == -1) {
                this.restartTimeout = setTimeout(function () {
                    _this.stop();
                    _this.restartTimeout = -1;
                    setTimeout(function () { _this.reset(); _this.run(); }, 100);
                }, this.config.autoRestartTime);
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
        this.onFrame(false);
        if (this.running)
            this.runningId = requestAnimationFrame(this.aniFrameCallback);
    };
    Simulation.prototype.iterations = function () {
        this.stop();
        for (var i = 0; i < this.config.iterationsPerClick; i++)
            this.step();
        this.onFrame(true);
    };
    Simulation.prototype.setIsCustom = function (forceNeuronRename) {
        if (forceNeuronRename === void 0) { forceNeuronRename = false; }
        if (this.isCustom && !forceNeuronRename)
            return;
        this.isCustom = true;
        $("#presetName").text("Custom Network");
        var layer = this.config.inputLayer;
        layer.names = Net.Util.makeArray(layer.neuronCount, function (i) { return ("in" + (i + 1)); });
        layer = this.config.outputLayer;
        layer.names = Net.Util.makeArray(layer.neuronCount, function (i) { return ("out" + (i + 1)); });
    };
    Simulation.prototype.loadConfig = function () {
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
        if (!this.config.autoRestart)
            clearTimeout(this.restartTimeout);
    };
    Simulation.prototype.loadPreset = function (name, weights) {
        this.isCustom = false;
        $("#presetName").text("Preset: " + name);
        this.config = Presets.get(name);
        this.setConfig();
        history.replaceState({}, "", "?" + $.param({ preset: name }));
        this.initializeNet(weights);
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
            params.weights = LZString.compressToEncodedURIComponent(JSON.stringify(this.net.connections.map(function (c) { return c.weight; })));
        if (exportWeights === 2)
            params.weights = LZString.compressToEncodedURIComponent(JSON.stringify(this.net.startWeights));
        if (this.isCustom) {
            params.config = LZString.compressToEncodedURIComponent(JSON.stringify(this.config));
        }
        else {
            params.preset = this.config.name;
        }
        return url + $.param(params);
    };
    Simulation.prototype.deserializeFromUrl = function () {
        var urlParams = Util.parseUrlParameters();
        var preset = urlParams["preset"], config = urlParams["config"];
        var weightString = urlParams["weights"];
        var weights;
        if (weightString)
            weights = JSON.parse(LZString.decompressFromEncodedURIComponent(weightString));
        if (preset && Presets.exists(preset))
            this.loadPreset(preset, weights);
        else if (config) {
            this.config = JSON.parse(LZString.decompressFromEncodedURIComponent(config));
            this.setIsCustom();
            this.initializeNet();
        }
        else
            this.loadPreset("Binary Classifier for XOR");
    };
    return Simulation;
})();
var TransformNavigation = (function () {
    function TransformNavigation(canvas, transformActive, transformChanged) {
        var _this = this;
        this.scalex = 200;
        this.scaley = -200;
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
        this.offsetx = canvas.width / 4;
        this.offsety = 3 * canvas.height / 4;
        canvas.addEventListener('wheel', function (e) {
            if (e.deltaY === 0)
                return;
            if (!transformActive())
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
    return TransformNavigation;
})();
var Util;
(function (Util) {
    function getMaxIndex(vals) {
        var max = vals[0], maxi = 0;
        for (var i = 1; i < vals.length; i++) {
            if (vals[i] > max) {
                max = vals[i];
                maxi = i;
            }
        }
        return maxi;
    }
    Util.getMaxIndex = getMaxIndex;
    function arrayWithOneAt(length, onePosition) {
        var output = new Array(length);
        for (var i = 0; i < length; i++) {
            output[i] = i === onePosition ? 1 : 0;
        }
        return output;
    }
    Util.arrayWithOneAt = arrayWithOneAt;
    function min(input) {
        return input.reduce(function (a, b) { return Math.min(a, b); }, Infinity);
    }
    Util.min = min;
    function max(input) {
        return input.reduce(function (a, b) { return Math.max(a, b); }, -Infinity);
    }
    Util.max = max;
    function bounds2dTrainingsInput(data) {
        return {
            minx: Util.min(data.map(function (d) { return d.input[0]; })),
            miny: Util.min(data.map(function (d) { return d.input[1]; })),
            maxx: Util.max(data.map(function (d) { return d.input[0]; })),
            maxy: Util.max(data.map(function (d) { return d.input[1]; }))
        };
    }
    Util.bounds2dTrainingsInput = bounds2dTrainingsInput;
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
    Util.randomGaussian = randomGaussian;
    ;
    function benchmark(fun) {
        var bef = Date.now();
        var r = fun();
        return Date.now() - bef;
    }
    Util.benchmark = benchmark;
    function parseColor(input) {
        var m = input.match(/^#([0-9a-f]{6})$/i)[1];
        if (m) {
            return [
                parseInt(m.substr(0, 2), 16),
                parseInt(m.substr(2, 2), 16),
                parseInt(m.substr(4, 2), 16)
            ];
        }
    }
    Util.parseColor = parseColor;
    function printColor(c) {
        c = c.map(function (x) { return x < 0 ? 0 : x > 255 ? 255 : x; });
        return '#' + ("000000" + (c[0] << 16 | c[1] << 8 | c[2]).toString(16)).slice(-6);
    }
    Util.printColor = printColor;
    function parseUrlParameters() {
        if (!location.search)
            return {};
        var query = {};
        for (var _i = 0, _a = location.search.slice(1).split('&'); _i < _a.length; _i++) {
            var p = _a[_i];
            var b = p.split('=').map(function (c) { return c.replace(/\+/g, ' '); });
            query[decodeURIComponent(b[0])] = decodeURIComponent(b[1]);
        }
        return query;
    }
    Util.parseUrlParameters = parseUrlParameters;
})(Util || (Util = {}));
var ErrorGraph = (function () {
    function ErrorGraph(sim) {
        this.sim = sim;
        this.actions = ["Error History"];
        this.container = $("<div>");
        this.container.highcharts({
            title: { text: 'Average RMSE' },
            chart: { type: 'line', animation: false },
            plotOptions: { line: { marker: { enabled: false } } },
            legend: { enabled: false },
            yAxis: { min: 0, title: { text: '' }, labels: { format: "{value:%.2f}" } },
            series: [{ name: 'Error', data: [] }],
            colors: ["black"],
            credits: { enabled: false }
        });
        this.chart = this.container.highcharts();
    }
    ErrorGraph.prototype.onFrame = function () {
        var data = [this.sim.stepNum, this.sim.averageError];
        this.chart.series[0].addPoint(data, true, false);
    };
    ErrorGraph.prototype.onView = function () {
        this.chart.series[0].setData(this.sim.errorHistory.map(function (x) { return x.slice(); }));
        this.chart.reflow();
    };
    ErrorGraph.prototype.onNetworkLoaded = function () {
        this.chart.series[0].setData([]);
    };
    ErrorGraph.prototype.onHide = function () {
    };
    return ErrorGraph;
})();
var NetworkGraph = (function () {
    function NetworkGraph(sim) {
        this.sim = sim;
        this.actions = ["Network Graph"];
        this.container = $("<div>");
        this.instantiateGraph();
    }
    NetworkGraph.prototype.instantiateGraph = function () {
        // need only be run once, but removes bounciness if run every time
        this.nodes = new vis.DataSet([], { queue: true }); // don't use clear (listener leak)
        this.edges = new vis.DataSet([], { queue: true });
        var graphData = {
            nodes: this.nodes,
            edges: this.edges };
        var options = {
            nodes: { shape: 'dot' },
            edges: {
                smooth: { type: 'curvedCW', roundness: 0 },
                font: { align: 'top', background: 'white' },
            },
            layout: { hierarchical: { direction: "LR" } },
            interaction: { dragNodes: false }
        };
        if (this.graph)
            this.graph.destroy();
        this.graph = new vis.Network(this.container[0], graphData, options);
    };
    NetworkGraph.prototype.onNetworkLoaded = function (net) {
        if (this.net
            && this.net.layers.length == net.layers.length
            && this.net.layers.every(function (layer, index) { return layer.length == net.layers[index].length; })
            && this.showbias === this.sim.config.bias) {
            // same net layout, only update
            this.net = net;
            this.onFrame(0);
            return;
        }
        this.showbias = this.sim.config.bias;
        this.instantiateGraph();
        this.net = net;
        for (var lid = 0; lid < net.layers.length; lid++) {
            var layer = net.layers[lid];
            for (var nid = 0; nid < layer.length; nid++) {
                var neuron = layer[nid];
                var type = 'Hidden Neuron ' + (nid + 1);
                var color = '#000';
                if (neuron instanceof Net.InputNeuron) {
                    type = 'Input: ' + neuron.name;
                    if (neuron.constant) {
                        if (!this.showbias)
                            continue;
                        color = NetworkVisualization.colors.autoencoder.bias;
                    }
                    else
                        color = NetworkVisualization.colors.autoencoder.input;
                }
                if (neuron instanceof Net.OutputNeuron) {
                    type = 'Output: ' + neuron.name;
                    color = NetworkVisualization.colors.autoencoder.output;
                }
                this.nodes.add({
                    id: neuron.id,
                    label: "" + type,
                    level: lid,
                    color: color
                });
            }
        }
        for (var _i = 0, _a = net.connections; _i < _a.length; _i++) {
            var conn = _a[_i];
            this.edges.add({
                id: conn.inp.id * net.connections.length + conn.out.id,
                from: conn.inp.id,
                to: conn.out.id,
                arrows: 'to',
                label: conn.weight.toFixed(2),
            });
        }
        this.nodes.flush();
        this.edges.flush();
    };
    NetworkGraph.prototype.onFrame = function (framenum) {
        if (this.net.connections.length > 20 && framenum % 15 !== 0) {
            // skip some frames because slow
            return;
        }
        for (var _i = 0, _a = this.net.connections; _i < _a.length; _i++) {
            var conn = _a[_i];
            this.edges.update({
                id: conn.inp.id * this.net.connections.length + conn.out.id,
                label: conn.weight.toFixed(2),
                width: Math.min(6, Math.abs(conn.weight * 2)),
                color: conn.weight > 0 ? 'blue' : 'red'
            });
        }
        this.edges.flush();
    };
    NetworkGraph.prototype.onView = function () {
        this.graph.stabilize();
    };
    NetworkGraph.prototype.onHide = function () {
    };
    return NetworkGraph;
})();
var InputMode;
(function (InputMode) {
    InputMode[InputMode["InputPrimary"] = 0] = "InputPrimary";
    InputMode[InputMode["InputSecondary"] = 1] = "InputSecondary";
    InputMode[InputMode["Remove"] = 2] = "Remove";
    InputMode[InputMode["Move"] = 3] = "Move";
    InputMode[InputMode["Table"] = 4] = "Table";
})(InputMode || (InputMode = {}));
var NetType;
(function (NetType) {
    NetType[NetType["BinaryClassify"] = 0] = "BinaryClassify";
    NetType[NetType["AutoEncode"] = 1] = "AutoEncode";
    NetType[NetType["MultiClass"] = 2] = "MultiClass";
    NetType[NetType["CantDraw"] = 3] = "CantDraw";
})(NetType || (NetType = {}));
var NetworkVisualization = (function () {
    function NetworkVisualization(sim) {
        var _this = this;
        this.sim = sim;
        this.actions = [];
        this.inputMode = 0;
        this.backgroundResolution = 15;
        this.container = $("<div>");
        this.netType = NetType.BinaryClassify;
        var tmp = NetworkVisualization.colors.multiClass;
        tmp.bg = tmp.fg.map(function (c) { return Util.printColor(Util.parseColor(c).map(function (x) { return (x * 1.3) | 0; })); });
        this.canvas = $("<canvas class=fullsize>")[0];
        this.canvas.width = 550;
        this.canvas.height = 400;
        this.trafo = new TransformNavigation(this.canvas, function () { return _this.inputMode == _this.actions.length - 1; }, function () { return _this.onFrame(); });
        this.ctx = this.canvas.getContext('2d');
        window.addEventListener('resize', this.canvasResized.bind(this));
        this.canvas.addEventListener("click", this.canvasClicked.bind(this));
        this.canvas.addEventListener("contextmenu", this.canvasClicked.bind(this));
        $(this.canvas).appendTo(this.container);
    }
    NetworkVisualization.prototype.onNetworkLoaded = function (net) {
        if (net.inputs.length != 2)
            this.netType = NetType.CantDraw;
        else {
            if (net.outputs.length == 1)
                this.netType = NetType.BinaryClassify;
            else if (net.outputs.length == 2)
                this.netType = NetType.AutoEncode;
            else
                this.netType = NetType.MultiClass;
        }
        switch (this.netType) {
            case NetType.BinaryClassify:
                this.actions = ["Add Red", "Add Green", "Remove", "Move View"];
                break;
            case NetType.AutoEncode:
                this.actions = ["Add Data point", "", "Remove", "Move View"];
                break;
            case NetType.MultiClass:
                this.actions = [];
                var i = 0;
                for (var _i = 0, _a = this.sim.config.outputLayer.names; _i < _a.length; _i++) {
                    var name_2 = _a[_i];
                    this.actions.push({ name: name_2, color: NetworkVisualization.colors.multiClass.bg[i++] });
                }
                this.actions.push("Remove");
                this.actions.push("Move View");
                break;
            case NetType.CantDraw:
                this.actions = [];
                break;
        }
        this.refitData();
    };
    NetworkVisualization.prototype.onFrame = function () {
        if (this.netType === NetType.CantDraw) {
            this.clear('white');
            this.ctx.fillStyle = 'black';
            this.ctx.textBaseline = "middle";
            this.ctx.textAlign = "center";
            this.ctx.font = "20px monospace";
            this.ctx.fillText("Cannot draw this data", this.canvas.width / 2, this.canvas.height / 2);
            return;
        }
        this.drawBackground();
        this.drawCoordinateSystem();
        this.drawDataPoints();
    };
    NetworkVisualization.prototype.drawDataPoints = function () {
        this.ctx.strokeStyle = "#000";
        if (this.netType === NetType.BinaryClassify) {
            for (var _i = 0, _a = this.sim.config.data; _i < _a.length; _i++) {
                var val = _a[_i];
                this.drawPoint(val.input[0], val.input[1], NetworkVisualization.colors.binaryClassify.fg[val.output[0] | 0]);
            }
        }
        else if (this.netType === NetType.AutoEncode) {
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
        else if (this.netType === NetType.MultiClass) {
            for (var _d = 0, _e = this.sim.config.data; _d < _e.length; _d++) {
                var val = _e[_d];
                this.drawPoint(val.input[0], val.input[1], NetworkVisualization.colors.multiClass.fg[Util.getMaxIndex(val.output)]);
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
        if (this.sim.config.outputLayer.neuronCount > 2) {
            for (var x = 0; x < this.canvas.width; x += this.backgroundResolution) {
                for (var y = 0; y < this.canvas.height; y += this.backgroundResolution) {
                    var vals = this.sim.net.getOutput([this.trafo.toReal.x(x + this.backgroundResolution / 2), this.trafo.toReal.y(y + this.backgroundResolution / 2)]);
                    var maxi = Util.getMaxIndex(vals);
                    this.ctx.fillStyle = NetworkVisualization.colors.multiClass.bg[maxi];
                    this.ctx.fillRect(x, y, this.backgroundResolution, this.backgroundResolution);
                }
            }
        }
        else {
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
        this.refitData();
        this.onFrame();
    };
    NetworkVisualization.prototype.refitData = function () {
        if (this.sim.config.data.length == 0)
            return;
        // update transform
        if (this.sim.config.inputLayer.neuronCount == 2) {
            var fillamount = 0.6;
            var bounds = Util.bounds2dTrainingsInput(this.sim.config.data);
            var w = bounds.maxx - bounds.minx, h = bounds.maxy - bounds.miny;
            this.trafo.scalex = this.canvas.width / w * fillamount;
            this.trafo.scaley = -this.canvas.height / h * fillamount;
            this.trafo.offsetx -= this.trafo.toCanvas.x(bounds.minx - w * (1 - fillamount) / 1.5); // / bounds.minx;
            this.trafo.offsety -= this.trafo.toCanvas.y(bounds.maxy + h * (1 - fillamount) / 1.5); // / bounds.minx;
        }
    };
    NetworkVisualization.prototype.canvasClicked = function (evt) {
        evt.preventDefault();
        var data = this.sim.config.data;
        var rect = this.canvas.getBoundingClientRect();
        var x = this.trafo.toReal.x(evt.clientX - rect.left);
        var y = this.trafo.toReal.y(evt.clientY - rect.top);
        var removeMode = this.actions.length - 2;
        if (this.inputMode === removeMode || evt.button == 2 || evt.shiftKey) {
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
        else if (this.inputMode < removeMode) {
            // add data point
            if (this.netType === NetType.AutoEncode) {
                data.push({ input: [x, y], output: [x, y] });
            }
            else {
                var inv = function (x) { return x == 0 ? 1 : 0; };
                var label = this.inputMode;
                if (evt.button != 0)
                    label = inv(label);
                if (evt.ctrlKey || evt.metaKey || evt.altKey)
                    label = inv(label);
                var output = [label];
                if (this.netType === NetType.MultiClass) {
                    output = Util.arrayWithOneAt(this.sim.config.outputLayer.neuronCount, label);
                }
                data.push({ input: [x, y], output: output });
            }
        }
        else
            return;
        this.sim.setIsCustom();
        this.onFrame();
    };
    NetworkVisualization.prototype.onView = function (previouslyHidden, mode) {
        if (previouslyHidden)
            this.canvasResized();
        this.inputMode = mode;
        this.onFrame();
    };
    NetworkVisualization.prototype.onHide = function () {
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
        },
        multiClass: {
            fg: ['#7cb5ec', '#434348', '#90ed7d', '#f7a35c', '#8085e9', '#f15c80', '#e4d354', '#2b908f', '#f45b5b', '#91e8e1'],
            bg: ['']
        }
    };
    return NetworkVisualization;
})();
var TableEditor = (function () {
    function TableEditor(sim) {
        this.sim = sim;
        this.actions = ["Table input"];
        this.headerCount = 2;
        this.lastUpdate = 0;
        this.container = $("<div>");
        this.sim = sim;
    }
    TableEditor.prototype.afterChange = function (changes, reason) {
        if (reason === 'loadData')
            return;
        this.reparseData();
    };
    TableEditor.prototype.onNetworkLoaded = function (net) {
        var _this = this;
        if (this.hot)
            this.hot.destroy();
        var oldContainer = this.container;
        this.container = $("<div class='fullsize'>");
        $("<div>").addClass("btn btn-default")
            .css({ position: "absolute", right: "2em", bottom: "2em" })
            .text("Remove all")
            .click(function (e) { sim.config.data = []; _this.loadData(); })
            .appendTo(this.container);
        var headerRenderer = function firstRowRenderer(instance, td) {
            Handsontable.renderers.TextRenderer.apply(this, arguments);
            td.style.fontWeight = 'bold';
            td.style.background = '#CCC';
        };
        var mergeCells = [];
        var ic = net.inputs.length, oc = net.outputs.length;
        //console.log(`creating new table (${ic}, ${oc})`);
        if (ic > 1)
            mergeCells.push({ row: 0, col: 0, rowspan: 1, colspan: ic });
        if (oc > 1) {
            mergeCells.push({ row: 0, col: ic, rowspan: 1, colspan: oc });
            mergeCells.push({ row: 0, col: ic + oc, rowspan: 1, colspan: oc });
        }
        var _conf = {
            minSpareRows: 1,
            colWidths: ic + oc + oc <= 6 ? 80 : 45,
            cells: function (row, col, prop) {
                if (row >= _this.headerCount)
                    return { type: 'numeric', format: '0.[000]' };
                else {
                    var conf = { renderer: headerRenderer };
                    if (row == 0)
                        conf.readOnly = true;
                    return conf;
                }
            },
            customBorders: false /*[{ // bug when larger than ~4
                range: {
                    from: { row: 0, col: ic },
                    to: { row: 100, col: ic }
                },
                left: { width: 2, color: 'black' }
            }, {
                    range: {
                        from: { row: 0, col: ic + oc },
                        to: { row: 100, col: ic + oc }
                    },
                    left: { width: 2, color: 'black' }
                }]*/,
            allowInvalid: false,
            mergeCells: mergeCells,
            afterChange: this.afterChange.bind(this)
        };
        this.container.handsontable(_conf);
        this.hot = this.container.handsontable('getInstance');
        if (oldContainer)
            oldContainer.replaceWith(this.container);
        this.loadData();
    };
    TableEditor.prototype.reparseData = function () {
        var sim = this.sim;
        var data = this.hot.getData();
        var headers = data[1];
        var ic = sim.config.inputLayer.neuronCount, oc = sim.config.outputLayer.neuronCount;
        sim.config.inputLayer.names = headers.slice(0, ic);
        sim.config.outputLayer.names = headers.slice(ic, ic + oc);
        sim.config.data = data.slice(2).map(function (row) { return row.slice(0, ic + oc); })
            .filter(function (row) { return row.every(function (cell) { return typeof cell === 'number'; }); })
            .map(function (row) { return { input: row.slice(0, ic), output: row.slice(ic) }; });
        sim.setIsCustom();
    };
    TableEditor.prototype.onFrame = function () {
        var sim = this.sim;
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
    TableEditor.prototype.loadData = function () {
        var sim = this.sim;
        var data = [[], sim.config.inputLayer.names.concat(sim.config.outputLayer.names).concat(sim.config.outputLayer.names)];
        var ic = sim.config.inputLayer.neuronCount, oc = sim.config.outputLayer.neuronCount;
        data[0][0] = 'Inputs';
        data[0][ic] = 'Expected Output';
        data[0][ic + oc + oc - 1] = ' ';
        data[0][ic + oc] = 'Actual Output';
        var mergeCells = [];
        if (ic > 1)
            mergeCells.push({ row: 0, col: 0, rowspan: 1, colspan: ic });
        if (oc > 1) {
            mergeCells.push({ row: 0, col: ic + oc, rowspan: 1, colspan: oc });
            mergeCells.push({ row: 0, col: ic + oc * 2, rowspan: 1, colspan: oc });
        }
        if (mergeCells.length > 0)
            this.hot.updateSettings({ mergeCells: mergeCells });
        sim.config.data.forEach(function (t) { return data.push(t.input.concat(t.output)); });
        this.hot.loadData(data);
        /*this.hot.updateSettings({customBorders: [
                
            ]});
        this.hot.runHooks('afterInit');*/
    };
    TableEditor.prototype.onView = function () {
        this.loadData();
    };
    TableEditor.prototype.onHide = function () {
        //this.reparseData();
    };
    return TableEditor;
})();
var TabSwitchVisualizationContainer = (function () {
    function TabSwitchVisualizationContainer(headContainer, bodyContainer, name, things) {
        var _this = this;
        this.headContainer = headContainer;
        this.bodyContainer = bodyContainer;
        this.name = name;
        this.things = things;
        this.modes = [];
        this.ul = $("<ul class='nav nav-pills'>");
        this.body = $("<div class='visbody'>");
        this.currentMode = -1;
        this.createButtonsAndActions();
        this.ul.on("click", "a", function (e) { return _this.setMode($(e.target).parent().index()); });
        headContainer.append(this.ul);
        bodyContainer.append(this.body);
    }
    TabSwitchVisualizationContainer.prototype.createButtonsAndActions = function () {
        var _this = this;
        this.ul.empty();
        this.modes = [];
        this.things.forEach(function (thing, thingid) {
            return thing.actions.forEach(function (button, buttonid) {
                _this.modes.push({ thing: thingid, action: buttonid });
                var a = $("<a>");
                if (typeof button === 'string') {
                    a.text(button);
                }
                else {
                    a.text(button.name).css("background-color", button.color);
                    var dark = Util.parseColor(button.color).reduce(function (a, b) { return a + b; }) / 3 < 127;
                    a.css("color", dark ? 'white' : 'black');
                }
                var li = $("<li>").append(a);
                if (!button)
                    li.hide();
                _this.ul.append(li);
            });
        });
    };
    TabSwitchVisualizationContainer.prototype.setMode = function (mode) {
        this.ul.children("li.custom-active").removeClass("custom-active");
        this.ul.children().eq(mode).addClass("custom-active");
        if (mode == this.currentMode)
            return;
        var action = this.modes[mode];
        var lastAction = this.modes[this.currentMode];
        this.currentMode = mode;
        this.currentVisualization = this.things[action.thing];
        if (!lastAction || action.thing != lastAction.thing) {
            if (lastAction)
                this.things[lastAction.thing].onHide();
            this.body.children().detach(); // keep event handlers
            this.body.append(this.currentVisualization.container);
            this.currentVisualization.onView(true, action.action);
        }
        else if (action.action !== lastAction.action) {
            this.currentVisualization.onView(false, action.action);
        }
    };
    TabSwitchVisualizationContainer.prototype.onNetworkLoaded = function (net) {
        //todo: ugly hack
        var beforeActions = JSON.stringify(this.things.map(function (t) { return t.actions; }));
        this.things.forEach(function (thing) { return thing.onNetworkLoaded(net); });
        var afterActions = JSON.stringify(this.things.map(function (t) { return t.actions; }));
        if (beforeActions !== afterActions) {
            this.createButtonsAndActions();
            this.currentMode = -1;
            this.setMode(0);
        }
    };
    return TabSwitchVisualizationContainer;
})();
var WeightsGraph = (function () {
    function WeightsGraph(sim) {
        var _this = this;
        this.sim = sim;
        this.actions = ["Weights"];
        this.container = $("<div>");
        this.offsetBetweenLayers = 2;
        this.xToLayer = [];
        this.xToNeuron = [];
        // hack to get grayscale colors
        vis.Graph3d.prototype._hsv2rgb = function (h, s, v) {
            h = Math.min(h, 250) | 0;
            return 'rgb(' + [h, h, h] + ')';
        };
        // hack to disable axis drawing
        vis.Graph3d.prototype._redrawAxis = function () { };
        this.graph = new vis.Graph3d(this.container[0], undefined, {
            style: 'bar',
            showPerspective: false,
            cameraPosition: { horizontal: -0.001, vertical: Math.PI / 2 },
            width: "100%",
            height: "100%",
            xLabel: 'Layer',
            yLabel: 'Neuron',
            zLabel: '',
            showGrid: true,
            axisColor: 'red',
            xBarWidth: 0.9,
            yBarWidth: 0.9,
            xCenter: "50%",
            legendLabel: "Weight",
            //zMin: 0,
            //zMax: 5,
            tooltip: function (point) {
                var outLayer = _this.xToLayer[point.x];
                var inLayer = outLayer - 1;
                var inNeuron = _this.xToNeuron[point.x];
                var outNeuron = point.y;
                var inN = _this.sim.net.layers[inLayer][inNeuron];
                var outN = _this.sim.net.layers[outLayer][outNeuron];
                var inStr, outStr;
                if (inN instanceof Net.InputNeuron)
                    inStr = inN.name;
                else
                    inStr = "Layer " + (inLayer + 1) + " Neuron " + (inNeuron + 1);
                if (outN instanceof Net.OutputNeuron)
                    outStr = outN.name;
                else
                    outStr = "Layer " + (outLayer + 1) + " Neuron " + (outNeuron + 1);
                return inStr + " to " + outStr;
            },
            xValueLabel: function (x) { return _this.xToLayer[x] || ""; },
            yValueLabel: function (y) { return (y | 0) == y ? y + 1 : ""; },
            zValueLabel: function (z) { return ""; },
        });
    }
    WeightsGraph.prototype.onView = function (previouslyHidden, action) {
        this.graph.redraw();
    };
    WeightsGraph.prototype.onHide = function () {
    };
    WeightsGraph.prototype.parseData = function (net) {
        var data = [];
        var maxx = 0;
        for (var layerNum = 1; layerNum < net.layers.length; layerNum++) {
            var layer = net.layers[layerNum];
            var layerX = maxx + this.offsetBetweenLayers;
            for (var y = 0; y < layer.length; y++) {
                var neuron = layer[y];
                maxx = Math.max(maxx, layerX + neuron.inputs.length);
                for (var input = 0; input < neuron.inputs.length; input++) {
                    var conn = neuron.inputs[input];
                    data.push({ x: layerX + input, y: y, z: conn.weight });
                    this.xToLayer[layerX + input] = layerNum;
                    this.xToNeuron[layerX + input] = input;
                }
            }
        }
        return data;
    };
    WeightsGraph.prototype.onNetworkLoaded = function (net) {
        this.graph.setData(this.parseData(net));
    };
    WeightsGraph.prototype.onFrame = function () {
        this.graph.setData(this.parseData(this.sim.net));
    };
    return WeightsGraph;
})();
//# sourceMappingURL=program.js.map