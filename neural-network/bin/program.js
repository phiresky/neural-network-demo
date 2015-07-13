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
        function NeuralNet(input, hidden, output, learnRate, addBiasToLayers, startWeight, startWeights) {
            var _this = this;
            if (startWeight === void 0) { startWeight = function () { return Math.random() - 0.5; }; }
            this.learnRate = learnRate;
            this.addBiasToLayers = addBiasToLayers;
            this.startWeights = startWeights;
            this.layers = [];
            this.connections = [];
            var nid = 0;
            this.inputs = Util.makeArray(input.neuronCount, function (i) { return new InputNeuron(nid++, i, input.names[i]); });
            this.layers.push(this.inputs.slice());
            for (var _i = 0; _i < hidden.length; _i++) {
                var layer = hidden[_i];
                this.layers.push(Util.makeArray(layer.neuronCount, function (i) { return new Neuron(layer.activation, nid++, i); }));
            }
            this.outputs = Util.makeArray(output.neuronCount, function (i) { return new OutputNeuron(output.activation, nid++, i, output.names[i]); });
            this.layers.push(this.outputs);
            for (var i = 0; i < this.layers.length - 1; i++) {
                var inLayer = this.layers[i];
                var outLayer = this.layers[i + 1];
                inLayer.push(new InputNeuron(nid++, -1, "Bias", 1));
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
                if (!addBiasToLayers)
                    inLayer.pop();
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
        function Neuron(activation, id, layerIndex) {
            this.activation = activation;
            this.id = id;
            this.layerIndex = layerIndex;
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
        function InputNeuron(id, layerIndex, name, constantOutput) {
            _super.call(this, null, id, layerIndex);
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
        function OutputNeuron(activation, id, layerIndex, name) {
            _super.call(this, activation, id, layerIndex);
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
    Presets.presets = [
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
        { "name": "Bit Position Auto Encoder", "learningRate": 0.05, "data": [{ "input": [1, 0, 0, 0], "output": [1, 0, 0, 0] }, { "input": [0, 1, 0, 0], "output": [0, 1, 0, 0] }, { "input": [0, 0, 1, 0], "output": [0, 0, 1, 0] }, { "input": [0, 0, 0, 1], "output": [0, 0, 0, 1] }], "inputLayer": { "neuronCount": 4, "names": ["in1", "in2", "in3", "in4"] }, "outputLayer": { "neuronCount": 4, "activation": "sigmoid", "names": ["out1", "out2", "out3", "out4"] }, "hiddenLayers": [{ "neuronCount": 2, "activation": "sigmoid" }], "netLayers": [{ "activation": "sigmoid", "neuronCount": 2 }, { "activation": "linear", "neuronCount": 1 }, { "neuronCount": 2, "activation": "sigmoid" }] }
    ];
    function getNames() {
        return Presets.presets.map(function (p) { return p.name; }).filter(function (c) { return c !== "Default"; });
    }
    Presets.getNames = getNames;
    function exists(name) {
        return Presets.presets.filter(function (p) { return p.name === name; })[0] !== undefined;
    }
    Presets.exists = exists;
    function get(name) {
        var chain = [];
        var preset = Presets.presets.filter(function (p) { return p.name === name; })[0];
        chain.unshift(preset);
        while (true) {
            var parentName = preset.parent || "Default";
            preset = Presets.presets.filter(function (p) { return p.name === parentName; })[0];
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
            // _cache = LZString.compressToBase64(JSON.stringify(data));
            var relevantData = data.filter(function (row) { return row[3] == 1; }).map(function (row) { return ({
                input: row.slice(0, 2),
                output: Util.arrayWithOneAt(10, row[2])
            }); });
            var preset = Presets.presets.filter(function (p) { return p.name === "Peterson and Barney (male)"; })[0];
            preset.data = relevantData;
            Util.normalizeInputs(preset);
            var relevantData2 = data.map(function (row) { return ({
                input: row.slice(0, 2),
                output: Util.arrayWithOneAt(10, row[2])
            }); });
            preset = Presets.presets.filter(function (p) { return p.name === "Peterson and Barney (all)"; })[0];
            preset.data = relevantData2;
            Util.normalizeInputs(preset);
            //presets.forEach(preset => preset.data && normalizeInputs(preset.data));
        }
        // include peterson_barney_data for faster page load
        var dataStr = "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";
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
                return [+row[cols.F1], +row[cols.F2], +row[cols.phonemeNum], +row[cols.gender]];
            });
            parseBarney(relevantData);
        });
    }
    Presets.loadPetersonBarney = loadPetersonBarney;
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
        this.net = new Net.NeuralNet(this.config.inputLayer, this.config.hiddenLayers, this.config.outputLayer, this.config.learningRate, this.config.bias, undefined, weights);
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
    function normalize(i, x, y) {
        return [(x - i.minx) / (i.maxx - i.minx), (y - i.miny) / (i.maxy - i.miny)];
    }
    Util.normalize = normalize;
    function normalizeInputs(conf) {
        var data = conf.data;
        var i = Util.bounds2dTrainingsInput(data);
        data.forEach(function (data) { return data.input = normalize(i, data.input[0], data.input[1]); });
        conf.originalBounds = i;
    }
    Util.normalizeInputs = normalizeInputs;
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
        if (this.sim.config.data.length < 3)
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
        this.xyToConnection = {};
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
                var _a = _this.xyToConnection[point.x + "," + point.y], conn = _a[0], outputLayer = _a[1];
                var inLayer = outputLayer - 1;
                var inStr, outStr;
                var inN = conn.inp, outN = conn.out;
                if (inN instanceof Net.InputNeuron)
                    inStr = inN.name;
                else
                    inStr = "Hidden(" + (inLayer + 1) + "," + (inN.layerIndex + 1) + ")";
                if (outN instanceof Net.OutputNeuron)
                    outStr = outN.name;
                else
                    outStr = "Hidden(" + (outputLayer + 1) + "," + (outN.layerIndex + 1) + ")";
                return inStr + " to " + outStr;
            },
            //xValueLabel: (x: int) => this.xToLayer[x] || "",
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
        this.xyToConnection = {};
        var data = [];
        var maxx = 0;
        var maxHeight = Math.max.apply(null, net.layers.map(function (layer) { return layer.length; }));
        for (var outputLayer = 1; outputLayer < net.layers.length; outputLayer++) {
            var layer = net.layers[outputLayer];
            var layerX = maxx + this.offsetBetweenLayers;
            for (var outputNeuron = 0; outputNeuron < layer.length; outputNeuron++) {
                var outN = layer[outputNeuron];
                maxx = Math.max(maxx, layerX + outN.inputs.length);
                for (var inputNeuron = 0; inputNeuron < outN.inputs.length; inputNeuron++) {
                    var conn = outN.inputs[inputNeuron];
                    var inN = conn.inp;
                    if (!this.sim.config.bias && inN instanceof Net.InputNeuron && inN.constant) {
                        continue;
                    }
                    var p = { x: layerX + inputNeuron, y: outputNeuron, z: conn.weight };
                    if (maxHeight != layer.length)
                        p.y += (maxHeight - layer.length) / 2;
                    data.push(p);
                    this.xyToConnection[p.x + "," + p.y] = [conn, outputLayer];
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