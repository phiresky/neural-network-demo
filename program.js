$(document).ready(function () {
    Presets.loadPetersonBarney();
    ReactDOM.render(React.createElement(Simulation, {autoRun: false}), document.getElementById("mainContainer"));
    window.sim = Simulation.instance;
});
/** check for regressions in net algorithm */
function checkSanity() {
    var sim = Simulation.instance;
    sim.setState(Presets.get("Binary Classifier for XOR"));
    var out = [-0.3180095069079748, -0.2749093166215802, -0.038532753589859546, 0.09576201205465842, -0.3460678329225116,
        0.23218797637289554, -0.33191669283980774, 0.5140297481331861, -0.1518989898989732];
    var inp = [-0.3094657452311367, -0.2758470894768834, 0.005968799814581871, 0.13201188389211893, -0.33257930004037917,
        0.24626848078332841, -0.35734778200276196, 0.489376779878512, -0.2165879353415221];
    sim.stop();
    sim.state.inputLayer = { neuronCount: 2, names: ['', ''] };
    sim.state.hiddenLayers = [{ neuronCount: 2, activation: "sigmoid" }];
    sim.state.outputLayer = { neuronCount: 1, activation: "sigmoid", names: [''] };
    sim.net.connections.forEach(function (e, i) { return e.weight = inp[i]; });
    for (var i = 0; i < 1000; i++)
        sim.trainAll();
    var realout = sim.net.connections.map(function (e) { return e.weight; });
    if (realout.every(function (e, i) { return e !== out[i]; }))
        throw "insanity!";
    return "ok";
}
var __extends = (this && this.__extends) || function (d, b) {
    for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p];
    function __() { this.constructor = d; }
    d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
};
/**
 * Simple implementation of a neural network (multilayer perceptron)
 *
 * Uses stochastic gradient descent with squared error as the loss function
 */
var Net;
(function (Net) {
    /** tangens hyperbolicus polyfill */
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
    /** list of known activation functions */
    Net.NonLinearities = {
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
        },
        relu: {
            f: function (x) { return Math.max(x, 0); },
            df: function (x) { return x <= 0 ? 0 : 1; }
        },
        // used for Rosenblatt Perceptron (df is fake and unimportant)
        "threshold (≥ 0)": {
            f: function (x) { return (x >= 0) ? 1 : 0; },
            df: function (x) { return 1; }
        }
    };
    ;
    /** list of training methods for every Configuration#type */
    Net.trainingMethods = {
        "nn": {
            "Batch Training": {
                trainAll: function (net, data) { return net.trainAll(data, false, false); },
                trainSingle: null
            },
            "Online Training": {
                trainAll: function (net, data) { return net.trainAll(data, true, false); },
                trainSingle: function (net, data) { return net.train(data, true, false); }
            }
        },
        /** Perceptron has an input layer with 3 neurons (incl. bias), an output layer with 1 neuron, and no hidden layers */
        "perceptron": {
            "Rosenblatt Perceptron": {
                trainAll: function (net, data) {
                    var _this = this;
                    return data.map(function (val) { return _this.trainSingle(net, val); });
                },
                trainSingle: function (net, val) {
                    // this algorithm is equivalent to what the neural network would do via net.train(data, true, true)
                    // (when the output neuron has this activation function:
                    //   {f: x => (x >= 0) ? 1 : 0, df: x => 1})
                    net.setInputsAndCalculate(val.input);
                    var outp = net.outputs[0];
                    outp.targetOutput = val.output[0];
                    var err = (outp.targetOutput - outp.output);
                    for (var _i = 0, _a = outp.inputs; _i < _a.length; _i++) {
                        var conn = _a[_i];
                        conn.weight += net.learnRate * err * conn.inp.output;
                    }
                    var weights = net.connections.map(function (conn) { return conn.weight; });
                    return { weights: weights, dataPoint: val };
                }
            },
            "Batch Perceptron": {
                trainAll: function (net, data) { return net.trainAll(data, false, true); },
                trainSingle: null
            },
            /** averaged perceptron (from http://ciml.info/dl/v0_8/ciml-v0_8-ch03.pdf , p.48) */
            "Averaged Perceptron": {
                trainAll: function (net, data) {
                    var storeWeightSteps = true; // false for optimizing (but then can't Configuration#drawArrows)
                    if (net.layers.length !== 2 || net.outputs.length !== 1 || net.outputs[0].activation !== "threshold (≥ 0)")
                        throw Error("can only be used for single perceptron");
                    if (storeWeightSteps)
                        var weights = [];
                    var u = net.connections.map(function (w) { return 0; });
                    var c = 1;
                    var _loop_1 = function(val) {
                        net.train(val, true, storeWeightSteps);
                        if (net.outputs[0].error !== 0) {
                            var y = val.output[0] === 1 ? -1 : 1;
                            u = net.connections.map(function (conn, i) { return u[i] + y * c * conn.inp.output; });
                            if (storeWeightSteps)
                                weights.push({ dataPoint: val, weights: net.connections.map(function (conn, i) { return conn.weight - u[i] / c; }) });
                        }
                        ++c;
                    };
                    for (var _i = 0, data_1 = data; _i < data_1.length; _i++) {
                        var val = data_1[_i];
                        _loop_1(val);
                    }
                    net.connections.forEach(function (conn, i) { return conn.weight -= u[i] / c; });
                    return weights;
                },
                trainSingle: null
            },
            /** from http://www.jmlr.org/papers/volume3/crammer03a/crammer03a.pdf , page 965
             * equivalent to Rosenblatt Perceptron except for how G is defined (see pdf)
             */
            "Binary MIRA": {
                trainAll: function (net, data) {
                    var _this = this;
                    return data.map(function (d) { return _this.trainSingle(net, d); });
                },
                trainSingle: function (net, val) {
                    net.setInputsAndCalculate(val.input);
                    var outp = net.outputs[0];
                    var x = val.input;
                    var y = val.output[0] == 1 ? 1 : -1;
                    var yGot = net.outputs[0].output == 1 ? 1 : -1;
                    //const err = (outp.targetOutput - outp.output);
                    var G = function (x) { return Math.min(Math.max(0, x), 1); };
                    var err = G(-y * net.outputs[0].weightedInputs / x.reduce(function (a, b) { return a + b * b; }, 0));
                    for (var _i = 0, _a = outp.inputs; _i < _a.length; _i++) {
                        var conn = _a[_i];
                        conn.weight += net.learnRate * err * y * conn.inp.output;
                    }
                    var weights = net.connections.map(function (conn) { return conn.weight; });
                    return { weights: weights, dataPoint: val };
                }
            }
        }
    };
    /**
     * intermediate result of a single data point training step and the resulting weights vector
     *
     * used to visualize the weights vector in the perceptron demo
     */
    var WeightsStep = (function () {
        function WeightsStep() {
        }
        return WeightsStep;
    }());
    Net.WeightsStep = WeightsStep;
    /**
     * back propagation code adapted from https://de.wikipedia.org/wiki/Backpropagation
     */
    var NeuralNet = (function () {
        function NeuralNet(input, hidden, output, learnRate, startWeight, startWeights) {
            var _this = this;
            if (startWeight === void 0) { startWeight = function () { return Math.random() - 0.5; }; }
            this.learnRate = learnRate;
            this.startWeights = startWeights;
            /** layers including input, hidden, and output */
            this.layers = [];
            /** bias input neurons */
            this.biases = [];
            /** a flat list of all the neuron connections */
            this.connections = [];
            var nid = 0;
            this.inputs = Util.makeArray(input.neuronCount, function (i) { return new InputNeuron(nid++, i, input.names[i]); });
            this.layers.push(this.inputs.slice());
            for (var _i = 0, hidden_1 = hidden; _i < hidden_1.length; _i++) {
                var layer = hidden_1[_i];
                this.layers.push(Util.makeArray(layer.neuronCount, function (i) { return new Neuron(layer.activation, nid++, i); }));
            }
            this.outputs = Util.makeArray(output.neuronCount, function (i) { return new OutputNeuron(output.activation, nid++, i, output.names[i]); });
            this.layers.push(this.outputs);
            for (var i = 0; i < this.layers.length - 1; i++) {
                var inLayer = this.layers[i];
                var outLayer = this.layers[i + 1];
                inLayer.push(new InputNeuron(nid++, -1, "Bias", 1));
                for (var _a = 0, inLayer_1 = inLayer; _a < inLayer_1.length; _a++) {
                    var input_1 = inLayer_1[_a];
                    for (var _b = 0, outLayer_1 = outLayer; _b < outLayer_1.length; _b++) {
                        var output_1 = outLayer_1[_b];
                        var conn = new Net.NeuronConnection(input_1, output_1);
                        input_1.outputs.push(conn);
                        output_1.inputs.push(conn);
                        this.connections.push(conn);
                    }
                }
                this.biases[i] = inLayer.pop();
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
                for (var _b = 0, layer_1 = layer; _b < layer_1.length; _b++) {
                    var neuron = layer_1[_b];
                    neuron.calculateOutput();
                }
            }
        };
        NeuralNet.prototype.getOutput = function (inputVals) {
            this.setInputsAndCalculate(inputVals);
            return this.outputs.map(function (output) { return output.output; });
        };
        /** get root-mean-square error */
        NeuralNet.prototype.getLoss = function (expectedOutput) {
            var sum = 0;
            for (var i = 0; i < this.outputs.length; i++) {
                var neuron = this.outputs[i];
                sum += Math.pow(neuron.output - expectedOutput[i], 2);
            }
            return Math.sqrt(sum / this.outputs.length);
        };
        /** if individual is true, train individually, else batch train */
        NeuralNet.prototype.trainAll = function (data, individual, storeWeightSteps) {
            if (storeWeightSteps)
                var weights = [];
            if (!individual)
                for (var _i = 0, _a = this.connections; _i < _a.length; _i++) {
                    var conn = _a[_i];
                    conn.zeroDeltaWeight();
                }
            for (var _b = 0, data_2 = data; _b < data_2.length; _b++) {
                var val = data_2[_b];
                var step = this.train(val, individual, storeWeightSteps);
                if (storeWeightSteps)
                    weights.push(step);
            }
            if (!individual)
                for (var _c = 0, _d = this.connections; _c < _d.length; _c++) {
                    var conn = _d[_c];
                    conn.flushDeltaWeight();
                }
            return weights;
        };
        /** if flush is false, only calculate deltas but don't reset or add them */
        NeuralNet.prototype.train = function (val, flush, storeWeightSteps) {
            if (flush === void 0) { flush = true; }
            this.setInputsAndCalculate(val.input);
            for (var i = 0; i < this.outputs.length; i++)
                this.outputs[i].targetOutput = val.output[i];
            for (var i = this.layers.length - 1; i > 0; i--) {
                for (var _i = 0, _a = this.layers[i]; _i < _a.length; _i++) {
                    var neuron = _a[_i];
                    neuron.calculateError();
                    for (var _b = 0, _c = neuron.inputs; _b < _c.length; _b++) {
                        var conn = _c[_b];
                        if (flush)
                            conn.zeroDeltaWeight();
                        conn.addDeltaWeight(this.learnRate);
                    }
                }
            }
            if (storeWeightSteps)
                var weights = this.connections.map(function (conn) { return conn.weight + conn.deltaWeight; });
            if (flush)
                for (var _d = 0, _e = this.connections; _d < _e.length; _d++) {
                    var conn = _e[_d];
                    conn.flushDeltaWeight();
                }
            return { weights: weights, dataPoint: val };
        };
        return NeuralNet;
    }());
    Net.NeuralNet = NeuralNet;
    /** a weighted connection between two neurons */
    var NeuronConnection = (function () {
        function NeuronConnection(inp, out) {
            this.inp = inp;
            this.out = out;
            this.weight = 0;
            /** cached delta weight for training */
            this.deltaWeight = NaN;
        }
        NeuronConnection.prototype.zeroDeltaWeight = function () {
            this.deltaWeight = 0;
        };
        NeuronConnection.prototype.addDeltaWeight = function (learnRate) {
            this.deltaWeight += learnRate * this.out.error * this.inp.output;
        };
        NeuronConnection.prototype.flushDeltaWeight = function () {
            this.weight += this.deltaWeight;
            this.deltaWeight = NaN; // set to NaN to prevent flushing bugs
        };
        return NeuronConnection;
    }());
    Net.NeuronConnection = NeuronConnection;
    /** a single Neuron / Perceptron */
    var Neuron = (function () {
        function Neuron(activation, id, layerIndex) {
            this.activation = activation;
            this.id = id;
            this.layerIndex = layerIndex;
            this.inputs = [];
            this.outputs = [];
            /** weighted sum of inputs without activation function */
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
            this.output = Net.NonLinearities[this.activation].f(this.weightedInputs);
        };
        Neuron.prototype.calculateError = function () {
            var δ = 0;
            for (var _i = 0, _a = this.outputs; _i < _a.length; _i++) {
                var output = _a[_i];
                δ += output.out.error * output.weight;
            }
            this.error = δ * Net.NonLinearities[this.activation].df(this.weightedInputs);
        };
        return Neuron;
    }());
    Net.Neuron = Neuron;
    /** an input neuron (no inputs, fixed output value, can't be trained) */
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
        InputNeuron.prototype.calculateOutput = function () { throw Error("input neuron"); };
        InputNeuron.prototype.calculateWeightedInputs = function () { throw Error("input neuron"); };
        InputNeuron.prototype.calculateError = function () { throw Error("input neuron"); };
        return InputNeuron;
    }(Neuron));
    Net.InputNeuron = InputNeuron;
    /** an output neuron (error calculated via target output */
    var OutputNeuron = (function (_super) {
        __extends(OutputNeuron, _super);
        function OutputNeuron(activation, id, layerIndex, name) {
            _super.call(this, activation, id, layerIndex);
            this.activation = activation;
            this.name = name;
        }
        OutputNeuron.prototype.calculateError = function () {
            this.error = Net.NonLinearities[this.activation].df(this.weightedInputs) * (this.targetOutput - this.output);
            // ⇔ for perceptron: (this.targetOutput - this.output)
            // ⇔  1 if (sign(w*x) =  1 and y = -1);
            //   -1 if (sign(w*x) = -1 and y =  1);
            //    0 if (sign(w*x) = y)
            //    where x = input vector, w = weight vector, y = output label (-1 or +1)
        };
        return OutputNeuron;
    }(Neuron));
    Net.OutputNeuron = OutputNeuron;
})(Net || (Net = {}));
/**
 * A preset is a hardcoded [[Configuration]].
 * It can also have a parent from which it inherits all properties.
 * This module handles loading of presets
 */
var Presets;
(function (Presets) {
    /** get a list of all the preset names */
    function getNames() {
        return Presets.presets.map(function (p) { return p.name; }).filter(function (c) { return c !== "Default"; });
    }
    Presets.getNames = getNames;
    /** check if a preset of the given name exists */
    function exists(name) {
        return Presets.presets.filter(function (p) { return p.name === name; })[0] !== undefined;
    }
    Presets.exists = exists;
    /** get the preset of the given name */
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
        for (var prop in sim.state) {
            if (sim.state[prop] !== parent[prop])
                outconf[prop] = sim.state[prop];
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
    /** all the hardcoded presets */
    Presets.presets = [
        {
            name: "Default",
            stepsPerSecond: 3000,
            learningRate: 0.05,
            showGradient: false,
            trainingMethod: "Online Training",
            custom: false,
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
            ],
            drawArrows: false,
            arrowScale: 0.3,
            weights: null,
            drawCoordinateSystem: true,
            animationTrainSinglePoints: false,
            type: "nn"
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
            stepsPerSecond: 1500,
            hiddenLayers: [
                { "neuronCount": 4, "activation": "sigmoid" },
            ],
            inputLayer: { neuronCount: 2, names: ["x", "y"] },
            outputLayer: { neuronCount: 3, "activation": "sigmoid", names: ["A", "B", "C"] },
            data: [{ "input": [1.40, 1.3], "output": [1, 0, 0] }, { "input": [1.56, 1.36], "output": [1, 0, 0] }, { "input": [1.36, 1.36], "output": [1, 0, 0] }, { "input": [1.46, 1.36], "output": [1, 0, 0] }, { "input": [1.14, 1.26], "output": [1, 0, 0] }, { "input": [0.96, 0.97], "output": [1, 0, 0] }, { "input": [1.04, 0.76], "output": [1, 0, 0] }, { "input": [1.43, 0.81], "output": [1, 0, 0] }, { "input": [1.3, 1.05], "output": [1, 0, 0] }, { "input": [1.45, 1.22], "output": [1, 0, 0] }, { "input": [2.04, 1.1], "output": [1, 0, 0] }, { "input": [1.06, 0.28], "output": [1, 0, 0] }, { "input": [0.96, 0.57], "output": [1, 0, 0] }, { "input": [1.28, 0.46], "output": [1, 0, 0] }, { "input": [1.51, 0.33], "output": [1, 0, 0] }, { "input": [1.65, 0.68], "output": [1, 0, 0] }, { "input": [1.67, 1.01], "output": [1, 0, 0] }, { "input": [1.5, 1.83], "output": [0, 1, 0] }, { "input": [0.76, 1.69], "output": [0, 1, 0] }, { "input": [0.4, 0.71], "output": [0, 1, 0] }, { "input": [0.61, 1.18], "output": [0, 1, 0] }, { "input": [0.26, 1.42], "output": [0, 1, 0] }, { "input": [0.28, 1.89], "output": [0, 1, 0] }, { "input": [1.37, 1.89], "output": [0, 1, 0] }, { "input": [1.11, 1.9], "output": [0, 1, 0] }, { "input": [1.05, 2.04], "output": [0, 1, 0] }, { "input": [2.43, 1.42], "output": [0, 1, 0] }, { "input": [2.39, 1.2], "output": [0, 1, 0] }, { "input": [2.1, 1.53], "output": [0, 1, 0] }, { "input": [1.89, 1.72], "output": [0, 1, 0] }, { "input": [2.69, 0.72], "output": [0, 1, 0] }, { "input": [2.96, 0.44], "output": [0, 1, 0] }, { "input": [2.5, 0.79], "output": [0, 1, 0] }, { "input": [2.85, 1.23], "output": [0, 1, 0] }, { "input": [2.82, 1.37], "output": [0, 1, 0] }, { "input": [1.93, 1.9], "output": [0, 1, 0] }, { "input": [2.18, 1.77], "output": [0, 1, 0] }, { "input": [2.29, 0.39], "output": [0, 1, 0] }, { "input": [2.57, 0.22], "output": [0, 1, 0] }, { "input": [2.7, -0.11], "output": [0, 1, 0] }, { "input": [1.96, -0.2], "output": [0, 1, 0] }, { "input": [1.89, -0.1], "output": [0, 1, 0] }, { "input": [1.77, 0.13], "output": [0, 1, 0] }, { "input": [0.73, 0.01], "output": [0, 1, 0] }, { "input": [0.37, 0.31], "output": [0, 1, 0] }, { "input": [0.46, 0.44], "output": [0, 1, 0] }, { "input": [0.48, 0.11], "output": [0, 1, 0] }, { "input": [0.37, -0.1], "output": [0, 1, 0] }, { "input": [1.03, -0.42], "output": [0, 1, 0] }, { "input": [1.35, -0.25], "output": [0, 1, 0] }, { "input": [1.17, 0.01], "output": [0, 1, 0] }, { "input": [0.12, 0.94], "output": [0, 1, 0] }, { "input": [2.05, 0.32], "output": [0, 1, 0] }, { "input": [1.97, 0.55], "output": [1, 0, 0] },
                { "input": [0.7860082304526748, 2.5761316872427984], "output": [0, 0, 1] }, { "input": [-0.09053497942386843, 2.3909465020576133], "output": [0, 0, 1] }, { "input": [-0.23868312757201657, 2.0329218106995888], "output": [0, 0, 1] }, { "input": [-0.32510288065843634, 1.748971193415638], "output": [0, 0, 1] }, { "input": [-0.6707818930041154, 1.4526748971193417], "output": [0, 0, 1] }, { "input": [-0.3991769547325104, 1.094650205761317], "output": [0, 0, 1] }, { "input": [-0.2263374485596709, 0.6131687242798356], "output": [0, 0, 1] }, { "input": [-0.2263374485596709, -0.42386831275720144], "output": [0, 0, 1] }, { "input": [-0.13991769547325114, -0.6584362139917693], "output": [0, 0, 1] }, { "input": [1.5390946502057612, -1.0658436213991767], "output": [0, 0, 1] }, { "input": [2.193415637860082, -1.0781893004115224], "output": [0, 0, 1] }, { "input": [2.6502057613168724, -0.9176954732510286], "output": [0, 0, 1] }, { "input": [3.193415637860082, -0.6460905349794236], "output": [0, 0, 1] }, { "input": [3.526748971193415, -0.42386831275720144], "output": [0, 0, 1] }, { "input": [3.4403292181069953, 0.329218106995885], "output": [0, 0, 1] }, { "input": [3.4773662551440325, 1.0452674897119343], "output": [0, 0, 1] }, { "input": [3.6625514403292176, 1.2798353909465023], "output": [0, 0, 1] }, { "input": [2.8847736625514404, 2.946502057613169], "output": [0, 0, 1] }, { "input": [1.4156378600823043, 2.5514403292181074], "output": [0, 0, 1] }, { "input": [1.045267489711934, 2.526748971193416], "output": [0, 0, 1] }, { "input": [2.5144032921810697, 2.1563786008230457], "output": [0, 0, 1] }, { "input": [3.045267489711934, 1.7983539094650207], "output": [0, 0, 1] }, { "input": [2.366255144032922, 2.9341563786008233], "output": [0, 0, 1] }, { "input": [1.5020576131687242, 3.0576131687242802], "output": [0, 0, 1] }, { "input": [0.5390946502057612, 2.711934156378601], "output": [0, 0, 1] }, { "input": [-0.300411522633745, 2.5761316872427984], "output": [0, 0, 1] }, { "input": [-0.7942386831275722, 2.563786008230453], "output": [0, 0, 1] }, { "input": [-1.1646090534979425, 1.181069958847737], "output": [0, 0, 1] }, { "input": [-1.1275720164609055, 0.5637860082304529], "output": [0, 0, 1] }, { "input": [-0.5226337448559671, 0.46502057613168746], "output": [0, 0, 1] }, { "input": [-0.4115226337448561, -0.05349794238683104], "output": [0, 0, 1] }, { "input": [-0.1646090534979425, -0.7325102880658434], "output": [0, 0, 1] }, { "input": [0.4650205761316871, -0.8436213991769544], "output": [0, 0, 1] }, { "input": [0.8106995884773661, -1.164609053497942], "output": [0, 0, 1] }, { "input": [0.32921810699588466, -1.3004115226337447], "output": [0, 0, 1] }, { "input": [1.1687242798353907, -1.127572016460905], "output": [0, 0, 1] }, { "input": [2.1316872427983538, -1.362139917695473], "output": [0, 0, 1] }, { "input": [1.7119341563786008, -0.6954732510288063], "output": [0, 0, 1] }, { "input": [2.5267489711934155, -0.8930041152263373], "output": [0, 0, 1] }, { "input": [2.8971193415637857, -0.8930041152263373], "output": [0, 0, 1] }, { "input": [2.6378600823045266, -0.6460905349794236], "output": [0, 0, 1] }, { "input": [3.2427983539094645, -0.5349794238683125], "output": [0, 0, 1] }, { "input": [3.8477366255144028, 0.02057613168724303], "output": [0, 0, 1] }, { "input": [3.390946502057613, 0.02057613168724303], "output": [0, 0, 1] }, { "input": [3.4403292181069953, 0.3415637860082307], "output": [0, 0, 1] }, { "input": [3.7983539094650203, 0.6502057613168727], "output": [0, 0, 1] }, { "input": [3.526748971193415, 0.983539094650206], "output": [0, 0, 1] }, { "input": [3.452674897119341, 1.4526748971193417], "output": [0, 0, 1] }, { "input": [3.502057613168724, 1.7242798353909468], "output": [0, 0, 1] }, { "input": [3.415637860082304, 2.205761316872428], "output": [0, 0, 1] }, { "input": [2.736625514403292, 2.292181069958848], "output": [0, 0, 1] }, { "input": [1.9465020576131686, 2.403292181069959], "output": [0, 0, 1] }, { "input": [1.8230452674897117, 2.60082304526749], "output": [0, 0, 1] }, { "input": [3.008230452674897, -1.288065843621399], "output": [0, 0, 1] }, { "input": [1.699588477366255, -1.016460905349794], "output": [0, 0, 1] }, { "input": [2.045267489711934, -0.9053497942386829], "output": [0, 0, 1] }, { "input": [1.8724279835390945, -1.2263374485596705], "output": [0, 0, 1] }]
        },
        { name: "Vowel frequency response (Peterson and Barney)",
            parent: "Three classes test",
            stepsPerSecond: 100,
            iterationsPerClick: 50,
            inputLayer: { neuronCount: 2, names: ["F1", "F2"] },
            outputLayer: { neuronCount: 10, "activation": "sigmoid", names: "IY,IH,EH,AE,AH,AA,AO,UH,UW,ER".split(",") }
        },
        {
            name: "Auto-Encoder for linear data",
            stepsPerSecond: 60,
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
            hiddenLayers: [
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
            stepsPerSecond: 3000,
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
        { "name": "Bit Position Auto Encoder", "learningRate": 0.05, "data": [{ "input": [1, 0, 0, 0], "output": [1, 0, 0, 0] }, { "input": [0, 1, 0, 0], "output": [0, 1, 0, 0] }, { "input": [0, 0, 1, 0], "output": [0, 0, 1, 0] }, { "input": [0, 0, 0, 1], "output": [0, 0, 0, 1] }], "inputLayer": { "neuronCount": 4, "names": ["in1", "in2", "in3", "in4"] }, "outputLayer": { "neuronCount": 4, "activation": "sigmoid", "names": ["out1", "out2", "out3", "out4"] }, "hiddenLayers": [{ "neuronCount": 2, "activation": "sigmoid" }] },
        {
            "name": "Rosenblatt Perceptron",
            stepsPerSecond: 2,
            "learningRate": 1,
            "showGradient": false,
            bias: true,
            "autoRestartTime": 5000,
            "autoRestart": false,
            trainingMethod: "Rosenblatt Perceptron",
            drawArrows: true,
            drawCoordinateSystem: false,
            animationTrainSinglePoints: false,
            type: "perceptron",
            "iterationsPerClick": 1,
            "data": [{ "input": [0.2101231155778894, 0.4947319932998326], "output": [0] }, { "input": [0.07838107202680059, 0.42886097152428815], "output": [0] }, { "input": [0.027711055276381822, 0.9000921273031828], "output": [0] }, { "input": [0.5344112227805695, 0.5910050251256282], "output": [0] }, { "input": [0.5445452261306533, 0.11977386934673367], "output": [0] }, { "input": [0.4482721943048576, -0.07783919597989952], "output": [0] }, { "input": [0.7725603015075377, -0.305854271356784], "output": [0] }, { "input": [0.5445452261306533, -0.3210552763819096], "output": [0] }, { "input": [-0.028025963149078823, 0.20084589614740372], "output": [1] }, { "input": [0.2506591289782244, -0.36159128978224464], "output": [1] }, { "input": [-0.22057202680067015, -0.04237018425460638], "output": [1] }, { "input": [-0.3573810720268008, 0.2819179229480737], "output": [1] }, { "input": [-0.5549941373534341, 0.2211139028475712], "output": [1] }, { "input": [0.05304606365159121, -0.3109212730318259], "output": [1] }, { "input": [-0.4485871021775546, -0.31598827470686774], "output": [1] }],
            "inputLayer": {
                "neuronCount": 2,
                "names": [
                    "x",
                    "y"
                ]
            },
            "outputLayer": {
                "neuronCount": 1,
                "activation": "threshold (≥ 0)",
                "names": [
                    "class"
                ]
            },
            "hiddenLayers": [],
        },
        {
            "name": "Averaged Perceptron",
            parent: "Rosenblatt Perceptron",
            trainingMethod: "Averaged Perceptron",
            animationTrainSinglePoints: false
        },
        {
            "name": "Binary MIRA",
            parent: "Rosenblatt Perceptron",
            trainingMethod: "Binary MIRA",
            animationTrainSinglePoints: false
        }
    ];
    function parseBarney(data) {
        // _cache = LZString.compressToBase64(JSON.stringify(data));
        var relevantData = data
            .filter(function (row) { return row[3] == 1; })
            .map(function (row) { return ({
            input: row.slice(0, 2),
            output: Util.arrayWithOneAt(10, row[2] - 1)
        }); });
        var preset = Presets.presets.filter(function (p) { return p.name === "Vowel frequency response (Peterson and Barney)"; })[0];
        preset.data = relevantData;
        Util.normalizeInputs(preset);
        //presets.forEach(preset => preset.data && normalizeInputs(preset.data));
    }
    function loadPetersonBarney() {
        // include lib/peterson_barney_data for faster page load
        var dataStr = "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";
        parseBarney(JSON.parse(LZString.decompressFromBase64(dataStr)));
        return;
    }
    Presets.loadPetersonBarney = loadPetersonBarney;
    function loadPetersonBarneyAsync() {
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
})(Presets || (Presets = {}));
var ExportModal = (function (_super) {
    __extends(ExportModal, _super);
    function ExportModal(props) {
        _super.call(this, props);
        this.state = {
            exportWeights: "0",
            errors: []
        };
        $("body").on("shown.bs.modal", "#exportModal", function () { return Simulation.instance.exportModal.forceUpdate(); });
    }
    ExportModal.prototype.render = function () {
        var _this = this;
        var ele = $("#exportModal")[0];
        var visible = true;
        if (ele && getComputedStyle(ele).display == "none")
            visible = false;
        return (React.createElement("div", {className: "modal fade", id: "exportModal"}, React.createElement("div", {className: "modal-dialog"}, React.createElement("div", {className: "modal-content"}, React.createElement("div", {className: "modal-header"}, React.createElement("button", {type: "button", className: "close", "data-dismiss": "modal"}, "×"), React.createElement("h3", {className: "modal-title"}, "Import / Export")), visible ? React.createElement("div", {className: "modal-body"}, React.createElement("h4", {className: "modal-title"}, "Export to URL"), React.createElement("select", {className: "exportWeights", onChange: function (t) { return _this.setState({ exportWeights: t.target.value }); }, value: this.state.exportWeights}, React.createElement("option", {value: "0"}, "Don't include weights"), React.createElement("option", {value: "1"}, "Include current weights"), React.createElement("option", {value: "2"}, "Include start weights")), React.createElement("p", null, "Copy this URL:", React.createElement("input", {className: "url-export", onClick: function (e) { return e.target.select(); }, readOnly: true, value: this.props.sim.serializeToUrl(+this.state.exportWeights)})), React.createElement("hr", null), React.createElement("h4", {className: "modal-title"}, "Export to file"), React.createElement("button", {className: "btn btn-default", onClick: function () { return _this.exportJSON(_this.props.sim.state); }}, "Export configuration and data as json"), React.createElement("button", {className: "btn btn-default", onClick: function () { return _this.exportCSV(_this.props.sim.state); }}, "Export training data as CSV"), React.createElement("hr", null), React.createElement("h4", {className: "modal-title"}, "Import"), React.createElement("span", {className: "btn btn-default btn-file"}, "Import JSON file ", React.createElement("input", {type: "file", className: "importJSON", onChange: this.importJSON.bind(this)})), React.createElement("span", {className: "btn btn-default btn-file"}, "Import CSV file ", React.createElement("input", {type: "file", className: "importCSV", onChange: this.importCSV.bind(this)})), this.state.errors.map(function (error, i) {
            return React.createElement("div", {key: i, className: "alert alert-danger"}, error, React.createElement("button", {type: "button", className: "close", "data-dismiss": "alert"}, "×"));
        })) : "Loading..."))));
    };
    ExportModal.prototype.exportJSON = function (conf) {
        Util.download(JSON.stringify(conf, null, '\t'), conf.name + ".json");
    };
    ExportModal.prototype.exportCSV = function (conf) {
        var csv = conf.inputLayer.names.concat(conf.outputLayer.names)
            .map(Util.csvSanitize).join(",") + "\n"
            + conf.data.map(function (data) { return data.input.concat(data.output).join(","); }).join("\n");
        Util.download(csv, conf.name + ".csv");
    };
    ExportModal.prototype.importJSON = function (ev) {
        var _this = this;
        var files = ev.target.files;
        if (files.length !== 1)
            this.addIOError("invalid selection");
        var file = files.item(0);
        var r = new FileReader();
        r.onload = function (t) {
            try {
                var text = r.result;
                _this.props.sim.setState(JSON.parse(text));
                $("#exportModal").modal('hide');
            }
            catch (e) {
                _this.addIOError("Error while reading " + file.name + ": " + e);
            }
        };
        r.readAsText(file);
    };
    ExportModal.prototype.importCSV = function (ev) {
        var _this = this;
        console.log("imo");
        var files = ev.target.files;
        if (files.length !== 1)
            this.addIOError("invalid selection");
        var file = files.item(0);
        var r = new FileReader();
        var sim = this.props.sim;
        r.onload = function (t) {
            try {
                var text = r.result;
                var data = text.split("\n").map(function (l) { return l.split(","); });
                var lens = data.map(function (l) { return l.length; });
                var len = Math.min.apply(Math, lens);
                if (len !== Math.max.apply(Math, lens))
                    throw "line lengths varying between " + len + " and " + Math.max.apply(Math, lens) + ", must be constant";
                var inps = sim.state.inputLayer.neuronCount;
                var oups = sim.state.outputLayer.neuronCount;
                if (len !== inps + oups)
                    throw "invalid line length, expected (" + inps + " inputs + " + oups + " outputs = ) " + (inps + oups) + " columns, got " + len + " columns";
                var newState = Util.cloneConfig(sim.state);
                if (!data[0][0].match(/^\d+$/)) {
                    var headers = data.shift();
                    newState.inputLayer.names = headers.slice(0, inps);
                    newState.outputLayer.names = headers.slice(inps, inps + oups);
                }
                newState.data = [];
                for (var l = 0; l < data.length; l++) {
                    var ele = { input: [], output: [] };
                    for (var i = 0; i < len; i++) {
                        var v = parseFloat(data[l][i]);
                        if (isNaN(v))
                            throw "can't parse " + data[l][i] + " as a number in line " + (l + 1);
                        (i < inps ? ele.input : ele.output).push(v);
                    }
                    newState.data.push(ele);
                }
                sim.setState(newState, function () { return sim.table.loadData(); });
                $("#exportModal").modal('hide');
            }
            catch (e) {
                _this.addIOError("Error while reading " + file.name + ": " + e);
                console.error(e);
            }
        };
        r.readAsText(file);
    };
    ExportModal.prototype.addIOError = function (err) {
        var errors = this.state.errors.slice();
        errors.push(err);
        this.setState({ errors: errors });
    };
    return ExportModal;
}(React.Component));
/**
 * the interface between the GUI and the Simulation / Neural network
 *
 * handles buttons, animation and configuration updates with the help of React
 *
 * the [[#state]] of this object contains the [[Configuration]]
 */
var Simulation = (function (_super) {
    __extends(Simulation, _super);
    function Simulation(props) {
        var _this = this;
        _super.call(this, props);
        /** training steps that should be done by now (used for animation) */
        this.stepsWanted = 0;
        /**
         * training steps that have been done since [[#initializeNet]]
         * single training steps or batch training steps both count as 1 step
         */
        this.stepsCurrent = 0;
        /**
         * global frame counter
         * used to limit computationally expensive functions to frameNum modulo n (see NetworkGraph#onFrame)
         */
        this.frameNum = 0;
        /** animation is currently running */
        this.running = false;
        /** the current requestAnimationFrame ID */
        this.runningId = -1;
        /** the setTimeout ID for restarting the simulation after some time (see Configuration#autoRestartTime) */
        this.restartTimeout = -1;
        this.lastTimestamp = 0;
        /** current average error (see Net.NeuralNet#getLoss) */
        this.averageError = 1;
        /** -1 when not training single data points, otherwise index into [[Configuration#data]] */
        this.currentTrainingDataPoint = -1;
        /** cache for all the steps that the [[NetGraph]] will go through for a single forward pass step */
        this.forwardPassEles = [];
        this.aniFrameCallback = this.animationStep.bind(this);
        if (Simulation.instance)
            throw Error("Already instantiated");
        else
            Simulation.instance = this;
        this.netviz = new NetworkVisualization(this, function (p) { return p === _this.state.data[_this.currentTrainingDataPoint]; });
        this.netgraph = new NetworkGraph(this);
        this.errorGraph = new ErrorGraph(this);
        this.table = new TableEditor(this);
        this.weightsGraph = new WeightsGraph(this);
        this.state = this.deserializeFromUrl();
    }
    Object.defineProperty(Simulation.prototype, "trainingMethod", {
        /** current training method (one of Net.trainingMethods) */
        get: function () {
            return Net.trainingMethods[this.state.type][this.state.trainingMethod];
        },
        enumerable: true,
        configurable: true
    });
    /** initialize a new random network */
    Simulation.prototype.initializeNet = function () {
        if (this.net)
            this.stop();
        console.log("initializeNet()");
        this.net = new Net.NeuralNet(this.state.inputLayer, this.state.hiddenLayers, this.state.outputLayer, this.state.learningRate, undefined, this.state.weights);
        this.stepsWanted = this.stepsCurrent = 0;
        this.errorHistory = [];
        this.lastWeights = [];
        this.lrVis.leftVis.onNetworkLoaded(this.net);
        this.lrVis.rightVis.onNetworkLoaded(this.net);
        this.onFrame(true);
    };
    /** train all data points according to [[#trainingMethod]] */
    Simulation.prototype.trainAll = function () {
        this.currentTrainingDataPoint = -1;
        this.stepsCurrent++;
        if (this.state.drawArrows)
            this.lastWeights = [{ dataPoint: null, weights: this.net.connections.map(function (c) { return c.weight; }) }];
        var steps = this.trainingMethod.trainAll(this.net, this.state.data);
        if (this.state.drawArrows)
            (_a = this.lastWeights).push.apply(_a, steps);
        var _a;
    };
    /** handle Train All button press */
    Simulation.prototype.trainAllButton = function () {
        this.stop();
        for (var i = 0; i < this.state.iterationsPerClick; i++)
            this.trainAll();
        this.stepsWanted = this.stepsCurrent;
        this.onFrame(true);
    };
    /** handle Train Single button press */
    Simulation.prototype.trainNextButton = function () {
        this.stop();
        this.trainNext();
        this.stepsWanted = this.stepsCurrent;
        this.onFrame(true);
    };
    /** train the next single data point */
    Simulation.prototype.trainNext = function () {
        this.currentTrainingDataPoint++;
        if (this.state.drawArrows)
            this.lastWeights = [{ dataPoint: null, weights: this.net.connections.map(function (c) { return c.weight; }) }];
        this.stepsCurrent++;
        if (this.currentTrainingDataPoint >= this.state.data.length) {
            this.currentTrainingDataPoint -= this.state.data.length;
        }
        var newWeights = this.trainingMethod.trainSingle(this.net, this.state.data[this.currentTrainingDataPoint]);
        if (this.state.drawArrows)
            this.lastWeights.push(newWeights);
    };
    /** do a single forward pass step, start the stepthrough if not already running */
    Simulation.prototype.forwardPassStep = function () {
        if (!this.netgraph.currentlyDisplayingForwardPass) {
            this.forwardPassEles = [];
            this.currentTrainingDataPoint = -1;
        }
        this.stop();
        if (this.forwardPassEles.length > 0) {
            this.netgraph.applyUpdate(this.forwardPassEles.shift());
        }
        else {
            if (this.currentTrainingDataPoint < this.state.data.length - 1) {
                // start next
                this.lrVis.leftVis.setMode(0);
                this.currentTrainingDataPoint++;
                this.forwardPassEles = this.netgraph.forwardPass(this.state.data[this.currentTrainingDataPoint]);
                this.netgraph.applyUpdate(this.forwardPassEles.shift());
                // redraw for highlighted data point
                this.netviz.onFrame();
            }
            else {
                // end
                this.currentTrainingDataPoint = -1;
                // this clears the forward pass step
                this.netgraph.onFrame(0);
                // clear highlighted data point
                this.netviz.onFrame();
            }
        }
    };
    /** handle animation frame */
    Simulation.prototype.onFrame = function (forceDraw) {
        this.frameNum++;
        this.calculateAverageError();
        this.lrVis.onFrame(forceDraw ? 0 : this.frameNum);
        this.updateStatusLine();
    };
    /** begin animation */
    Simulation.prototype.run = function () {
        if (this.running)
            return;
        this.running = true;
        this.lrVis.setState({ running: true });
        this.lastTimestamp = performance.now();
        requestAnimationFrame(this.aniFrameCallback);
    };
    /** stop animation */
    Simulation.prototype.stop = function () {
        clearTimeout(this.restartTimeout);
        this.restartTimeout = -1;
        this.running = false;
        this.lrVis.setState({ running: false });
        cancelAnimationFrame(this.runningId);
    };
    /** stop animation and reset the network */
    Simulation.prototype.reset = function () {
        this.stop();
        this.initializeNet();
        this.onFrame(true);
    };
    /** calculate the average error ([[#averageError]]) */
    Simulation.prototype.calculateAverageError = function () {
        this.averageError = 0;
        for (var _i = 0, _a = this.state.data; _i < _a.length; _i++) {
            var val = _a[_i];
            this.net.setInputsAndCalculate(val.input);
            this.averageError += this.net.getLoss(val.output);
        }
        this.averageError /= this.state.data.length;
        this.errorHistory.push([this.stepsCurrent, this.averageError]);
    };
    /** update the information in the status line ([[#averageError]] / correct count and [[#stepNum]]) */
    Simulation.prototype.updateStatusLine = function () {
        var _this = this;
        var correct = 0;
        if (this.state.outputLayer.neuronCount === 1) {
            for (var _i = 0, _a = this.state.data; _i < _a.length; _i++) {
                var val = _a[_i];
                var res = this.net.getOutput(val.input);
                if (+(res[0] > 0.5) == val.output[0])
                    correct++;
            }
            this.lrVis.setState({ correct: "Correct: " + correct + "/" + this.state.data.length });
        }
        else {
            this.lrVis.setState({ correct: "Error: " + (this.averageError).toFixed(2) });
        }
        this.lrVis.setState({ stepNum: this.stepsCurrent });
        if (correct == this.state.data.length) {
            if (this.state.autoRestart && this.running && this.restartTimeout == -1) {
                this.restartTimeout = setTimeout(function () {
                    _this.stop();
                    _this.restartTimeout = -1;
                    setTimeout(function () { _this.reset(); _this.run(); }, 100);
                }, this.state.autoRestartTime);
            }
        }
        else {
            if (this.restartTimeout != -1) {
                clearTimeout(this.restartTimeout);
                this.restartTimeout = -1;
            }
        }
    };
    /** do a single step in the animation, called by [[window.requestAnimationFrame]] */
    Simulation.prototype.animationStep = function (timestamp) {
        var delta = timestamp - this.lastTimestamp;
        this.lastTimestamp = timestamp;
        if (delta > 1000 / 5) {
            console.warn("only " + (1000 / delta).toFixed(1) + " fps");
            delta = 1000 / 5;
        }
        this.stepsWanted += delta / 1000 * this.state.stepsPerSecond;
        while (this.stepsCurrent < this.stepsWanted) {
            if (this.state.animationTrainSinglePoints && this.trainingMethod.trainSingle)
                this.trainNext();
            else
                this.trainAll();
        }
        this.onFrame(false);
        if (this.running)
            this.runningId = requestAnimationFrame(this.aniFrameCallback);
    };
    /** called by React when configuration ([[#state]]) is about to change */
    Simulation.prototype.componentWillUpdate = function (nextProps, newConfig) {
        if (this.state.hiddenLayers.length !== newConfig.hiddenLayers.length && newConfig.custom) {
            if (this.state.custom /* && !forceNeuronRename*/)
                return;
            var inN = newConfig.inputLayer.neuronCount;
            var outN = newConfig.outputLayer.neuronCount;
            newConfig.name = "Custom Network";
            newConfig.inputLayer = { names: Util.makeArray(inN, function (i) { return ("in" + (i + 1)); }), neuronCount: inN };
            newConfig.outputLayer = { names: Util.makeArray(outN, function (i) { return ("out" + (i + 1)); }), activation: newConfig.outputLayer.activation, neuronCount: outN };
        }
    };
    /** called by React after the configuration changed; updates the gui components not handled by react accordingly */
    Simulation.prototype.componentDidUpdate = function (prevProps, oldConfig) {
        var co = oldConfig, cn = this.state;
        if (!cn.autoRestart)
            clearTimeout(this.restartTimeout);
        var layerDifferent = function (l1, l2) {
            return l1.activation !== l2.activation || l1.neuronCount !== l2.neuronCount || (l1.names && l1.names.some(function (name, i) { return l2.names[i] !== name; }));
        };
        if (cn.hiddenLayers.length !== co.hiddenLayers.length
            || layerDifferent(cn.inputLayer, co.inputLayer)
            || layerDifferent(cn.outputLayer, co.outputLayer)
            || cn.hiddenLayers.some(function (layer, i) { return layerDifferent(layer, co.hiddenLayers[i]); })
            || cn.weights && (!co.weights || cn.weights.some(function (weight, i) { return co.weights[i] !== weight; }))) {
            this.initializeNet();
        }
        if (!cn.custom)
            history.replaceState({}, "", "?" + $.param({ preset: cn.name }));
        if (this.net) {
            if (co.bias != cn.bias) {
                this.netgraph.onNetworkLoaded(this.net);
            }
            if (co.type !== cn.type) {
                // gui layout may change, trigger resize
                window.dispatchEvent(new Event('resize'));
            }
            this.net.learnRate = cn.learningRate;
            if (cn.showGradient != co.showGradient || cn.drawCoordinateSystem != co.drawCoordinateSystem || cn.drawArrows != co.drawArrows)
                this.onFrame(false);
        }
    };
    /** called once after the GUI has been added to the DOM */
    Simulation.prototype.componentDidMount = function () {
        this.initializeNet();
        this.onFrame(true);
        if (this.props.autoRun)
            this.run();
    };
    /** parse the [[ConfigurationGui]] contents into [[#state]] */
    Simulation.prototype.loadConfig = function () {
        var config = $.extend(true, {}, this.state);
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
        config.learningRate = Util.expScale(config.learningRate);
        this.setState(config);
    };
    Simulation.prototype.runtoggle = function () {
        if (this.running)
            this.stop();
        else
            this.run();
    };
    /**
     * serialize the current configuration into the url (query string)
     * @param exportWeights 0 = no weights, 1 = current weights, 2 = start weights
     */
    Simulation.prototype.serializeToUrl = function (exportWeights) {
        if (exportWeights === void 0) { exportWeights = 0; }
        var url = location.protocol + '//' + location.host + location.pathname + "?";
        console.log("serializing to url");
        var params = {};
        if (this.state.custom || exportWeights > 0) {
            params.config = Util.cloneConfig(this.state);
        }
        else {
            params.preset = this.state.name;
        }
        if (exportWeights === 1)
            params.config.weights = this.net.connections.map(function (c) { return c.weight; });
        if (exportWeights === 2)
            params.config.weights = this.net.startWeights;
        if (params.config)
            params.config = LZString.compressToEncodedURIComponent(JSON.stringify(params.config));
        return url + $.param(params);
    };
    /**
     * parse the configuration from the url created by [[#serializeToUrl]]
     */
    Simulation.prototype.deserializeFromUrl = function () {
        var urlParams = Util.parseUrlParameters();
        var preset = urlParams["preset"], config = urlParams["config"];
        if (preset && Presets.exists(preset))
            return Presets.get(preset);
        else if (config) {
            console.log(JSON.parse(LZString.decompressFromEncodedURIComponent(config)));
            return JSON.parse(LZString.decompressFromEncodedURIComponent(config));
        }
        else
            return Presets.get("Binary Classifier for XOR");
    };
    /**
     * (this should be default in newer React versions?)
     */
    Simulation.prototype.shouldComponentUpdate = function () { return true; };
    /**
     * called by React to render the GUI
     */
    Simulation.prototype.render = function () {
        var _this = this;
        var pageTitle = this.state.type === "perceptron" ? "Perceptron demo" : "Neural Network demo";
        var presetName = this.state.custom ? " Custom Network" : " Preset: " + this.state.name;
        document.title = pageTitle + " \u2014 " + presetName;
        return (React.createElement("div", null, React.createElement("div", {className: "container"}, React.createElement("div", {className: "page-header"}, React.createElement("div", {className: "btn-toolbar pull-right dropdown", style: { marginTop: "5px" }}, React.createElement("button", {className: "btn btn-info dropdown-toggle", "data-toggle": "dropdown"}, "Load Preset ", React.createElement("span", {className: "caret"})), React.createElement("ul", {className: "dropdown-menu"}, React.createElement("li", {className: "dropdown-header"}, "Neural Network"), Presets.getNames().map(function (name) {
            var ele = React.createElement("li", {key: name}, React.createElement("a", {onClick: function (e) { return _this.setState(Presets.get(name)); }}, name));
            if (name === "Rosenblatt Perceptron")
                return [React.createElement("li", {className: "divider"}), React.createElement("li", {className: "dropdown-header"}, "Perceptron"), ele];
            else
                return ele;
        }))), React.createElement("h1", null, pageTitle, React.createElement("small", null, presetName))), React.createElement(LRVis, {sim: this, ref: function (e) { return _this.lrVis = e; }, leftVis: [this.netgraph, this.errorGraph, this.weightsGraph], rightVis: [this.netviz, this.table]}), React.createElement("div", {className: "panel panel-default"}, React.createElement("div", {className: "panel-heading"}, React.createElement("h3", {className: "panel-title"}, React.createElement("a", {"data-toggle": "collapse", "data-target": ".panel-body"}, "Configuration"))), React.createElement("div", {className: "panel-body collapse in"}, React.createElement(ConfigurationGui, React.__spread({}, this.state)))), React.createElement("footer", {className: "small"}, React.createElement("a", {href: "https://github.com/phiresky/kogsys-demos/"}, "Source on GitHub"))), React.createElement(ExportModal, {sim: this, ref: function (e) { return _this.exportModal = e; }})));
    };
    return Simulation;
}(React.Component));
/**
 * this class handles the linear transformations used to offset and scale the drawing in a 2d canvas
 * @see [[#toReal]] and [[#toCanvas]]
 */
var TransformNavigation = (function () {
    /**
     * @param transformActive function that returns if the mouse transform should act on mouse dragging / scrolling events currently
     * @param transformChanged callback when the transform has changed (e.g. to redraw)
     */
    function TransformNavigation(canvas, transformActive, transformChanged) {
        var _this = this;
        this.scalex = 200;
        this.scaley = -200;
        this.offsetx = 0;
        this.offsety = 0;
        /** converts screen space coordinates to real coordinates */
        this.toReal = {
            x: function (x) { return (x - _this.offsetx) / _this.scalex; },
            y: function (y) { return (y - _this.offsety) / _this.scaley; }
        };
        /** converts real coordinates to screen space coordinates */
        this.toCanvas = {
            x: function (c) { return c * _this.scalex + _this.offsetx; },
            y: function (c) { return c * _this.scaley + _this.offsety; }
        };
        /** position where the mouse press started (for dragging) */
        this.mousestart = null;
        this.offsetx = canvas.width / 4;
        this.offsety = 3 * canvas.height / 4;
        canvas.addEventListener('wheel', function (e) {
            if (e.deltaY === 0)
                return;
            if (!transformActive())
                return;
            var delta = e.deltaY / Math.abs(e.deltaY);
            var beforeTransform = { x: _this.toReal.x(e.offsetX), y: _this.toReal.y(e.offsetY) };
            _this.scalex *= 1 - delta / 10;
            _this.scaley *= 1 - delta / 10;
            var afterTransform = { x: _this.toReal.x(e.offsetX), y: _this.toReal.y(e.offsetY) };
            _this.offsetx += (afterTransform.x - beforeTransform.x) * _this.scalex;
            _this.offsety += (afterTransform.y - beforeTransform.y) * _this.scaley;
            transformChanged();
            Util.stopEvent(e);
        });
        canvas.addEventListener('mousedown', function (e) {
            if (!transformActive())
                return;
            _this.mousestart = { x: e.pageX, y: e.pageY };
            Util.stopEvent(e);
        });
        window.addEventListener('mousemove', function (e) {
            if (!transformActive())
                return;
            if (!_this.mousestart)
                return;
            _this.offsetx += e.pageX - _this.mousestart.x;
            _this.offsety += e.pageY - _this.mousestart.y;
            _this.mousestart.x = e.pageX;
            _this.mousestart.y = e.pageY;
            transformChanged();
            Util.stopEvent(e);
        });
        window.addEventListener('mouseup', function (e) {
            if (_this.mousestart) {
                _this.mousestart = null;
                Util.stopEvent(e);
            }
        });
    }
    return TransformNavigation;
}());
var Util;
(function (Util) {
    /**
     * @param len array length
     * @param supplier map from index to array element
     */
    function makeArray(len, supplier) {
        var arr = new Array(len);
        for (var i = 0; i < len; i++)
            arr[i] = supplier(i);
        return arr;
    }
    Util.makeArray = makeArray;
    /**
     * return array index that has maximum value
     * ≈ argmax function
     */
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
    /**
     * get an array that has a one at the specified position and zero everywhere else
     */
    function arrayWithOneAt(length, onePosition) {
        var output = new Array(length);
        for (var i = 0; i < length; i++) {
            output[i] = i === onePosition ? 1 : 0;
        }
        return output;
    }
    Util.arrayWithOneAt = arrayWithOneAt;
    /**
     * get AABB of a set of points in the form of 2d-TrainingData
     */
    function bounds2dTrainingsInput(data) {
        return {
            minx: Math.min.apply(Math, data.map(function (d) { return d.input[0]; })),
            miny: Math.min.apply(Math, data.map(function (d) { return d.input[1]; })),
            maxx: Math.max.apply(Math, data.map(function (d) { return d.input[0]; })),
            maxy: Math.max.apply(Math, data.map(function (d) { return d.input[1]; }))
        };
    }
    Util.bounds2dTrainingsInput = bounds2dTrainingsInput;
    var _nextGaussian;
    /** port of https://docs.oracle.com/javase/7/docs/api/java/util/Random.html#nextGaussian() */
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
    /**
     * deep clone a configuration object
     */
    function cloneConfig(config) {
        return $.extend(true, {}, config);
    }
    Util.cloneConfig = cloneConfig;
    /**
     * parse hex color (#ff0000) to number array
     */
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
    /**
     * convert three numbers to hex color string
     */
    function printColor(c) {
        c = c.map(function (x) { return x < 0 ? 0 : x > 255 ? 255 : x; });
        return '#' + ("000000" + (c[0] << 16 | c[1] << 8 | c[2]).toString(16)).slice(-6);
    }
    Util.printColor = printColor;
    /**
     * parse url query parameters (...?a=b&x=y) to js object ({a:"b",x:"y"})
     */
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
    /** normalize 2d point in the given bounds to [0,1]×[0,1] */
    function normalize(i, x, y) {
        return [(x - i.minx) / (i.maxx - i.minx), (y - i.miny) / (i.maxy - i.miny)];
    }
    Util.normalize = normalize;
    /** normalize x and y training data of given configuration into [0,1] */
    function normalizeInputs(conf) {
        if (conf.inputLayer.neuronCount !== 2)
            throw Error("can only normalize 2d data");
        var data = conf.data;
        var i = Util.bounds2dTrainingsInput(data);
        data.forEach(function (data) { return data.input = normalize(i, data.input[0], data.input[1]); });
    }
    Util.normalizeInputs = normalizeInputs;
    /**
     * download some text
     * @param text file content
     * @param name file name
     * @param type mime type
     */
    function download(text, name, type) {
        if (type === void 0) { type = 'text/plain'; }
        var a = document.createElement("a");
        var file = new Blob([text], { type: type });
        a.href = URL.createObjectURL(file);
        a.download = name;
        a.click();
    }
    Util.download = download;
    /** sanitize string for output to CSV */
    function csvSanitize(s) {
        s = s.replace(/"/g, '""');
        if (s.search(/("|,|\n)/g) >= 0)
            return "\"" + s + "\"";
        else
            return s;
    }
    Util.csvSanitize = csvSanitize;
    /** interpret n as a logarithmic scale */
    function logScale(n) {
        return Math.log(n * 9 + 1) / Math.LN10;
    }
    Util.logScale = logScale;
    /** interpret n as a exponential scale */
    function expScale(n) {
        return (Math.pow(10, n) - 1) / 9;
    }
    Util.expScale = expScale;
    function stopEvent(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    Util.stopEvent = stopEvent;
    /**
     * Draws a line with a filled arrow head at its end.
     *
     * FOUND ON USENET
     * @param al Arrowhead length
     * @param aw Arrowhead width
     *
     */
    function drawArrow(g, start, end, al, aw) {
        // Compute length of line
        var length = Math.sqrt(Math.pow((end.x - start.x), 2) + Math.pow((end.y - start.y), 2));
        // Compute normalized line vector
        var x = (end.x - start.x) / length;
        var y = (end.y - start.y) / length;
        // Compute points for arrow head
        var base = { x: end.x - x * al, y: end.y - y * al };
        var back_top = { x: base.x - aw * y, y: base.y + aw * x };
        var back_bottom = { x: base.x + aw * y, y: base.y - aw * x };
        // Draw lines
        g.beginPath();
        g.moveTo(start.x, start.y);
        g.lineTo(end.x, end.y);
        g.stroke();
        g.moveTo(back_bottom.x, back_bottom.y);
        g.lineTo(end.x, end.y);
        g.lineTo(back_top.x, back_top.y);
        g.fill();
    }
    Util.drawArrow = drawArrow;
    /** convert perceptron weights to a linear function */
    function toLinearFunction(_a, threshold) {
        var wx = _a[0], wy = _a[1], wbias = _a[2];
        if (threshold === void 0) { threshold = 0; }
        //   w1*x + w2*y + w3 = thres
        // ⇒ w2*y = thres - w3 - w1*x
        // ⇒ y = (thres - w3 - w1*x) / w2
        if (wy === 0)
            wy = 0.00001;
        return function (x) { return (threshold - wbias - wx * x) / wy; };
    }
    Util.toLinearFunction = toLinearFunction;
})(Util || (Util = {}));
/** small wrapper for bootstrap form groups */
var BSFormGroup = (function (_super) {
    __extends(BSFormGroup, _super);
    function BSFormGroup() {
        _super.apply(this, arguments);
    }
    BSFormGroup.prototype.render = function () {
        return React.createElement("div", {className: "form-group"}, React.createElement("label", {htmlFor: this.props.id, className: "col-sm-6 control-label"}, this.props.label), React.createElement("div", {className: "col-sm-6 " + (this.props.isStatic ? "form-control-static" : "")}, this.props.children));
    };
    return BSFormGroup;
}(React.Component));
/** small wrapper for bootstrap form checkboxes */
var BSCheckbox = (function (_super) {
    __extends(BSCheckbox, _super);
    function BSCheckbox() {
        _super.apply(this, arguments);
    }
    BSCheckbox.prototype.render = function () {
        return (React.createElement(BSFormGroup, {label: this.props.label, id: this.props.id, isStatic: true}, React.createElement("input", {type: "checkbox", checked: this.props.conf[this.props.id], id: this.props.id, onChange: this.props.onChange})));
    };
    return BSCheckbox;
}(React.Component));
/** GUI for displaying and modifying [[Configuration]] */
var ConfigurationGui = (function (_super) {
    __extends(ConfigurationGui, _super);
    function ConfigurationGui() {
        _super.apply(this, arguments);
    }
    ConfigurationGui.prototype.render = function () {
        var conf = this.props;
        var loadConfig = function () { return Simulation.instance.loadConfig(); };
        var trainingMethods = Net.trainingMethods[conf.type];
        return React.createElement("div", {className: "form-horizontal"}, React.createElement("div", {className: "col-sm-6"}, React.createElement("h4", null, "Display"), React.createElement(BSFormGroup, {label: "Iterations per click on 'Train'", id: "iterationsPerClick"}, React.createElement("input", {className: "form-control", type: "number", min: 0, max: 10000, id: "iterationsPerClick", value: "" + conf.iterationsPerClick, onChange: loadConfig})), React.createElement(BSFormGroup, {label: "Steps per Second", id: "stepsPerSecond"}, React.createElement("input", {className: "form-control", type: "number", min: 0.1, max: 1000, id: "stepsPerSecond", value: "" + conf.stepsPerSecond, onChange: loadConfig})), React.createElement(BSCheckbox, {label: "When correct, restart after 5 seconds", id: "autoRestart", onChange: loadConfig, conf: conf}), conf.type !== "perceptron" ?
            React.createElement(BSCheckbox, {label: "Show class propabilities as gradient", id: "showGradient", onChange: loadConfig, conf: conf})
            : "", React.createElement(BSCheckbox, {label: "Show bias input", id: "bias", onChange: loadConfig, conf: conf}), React.createElement("button", {className: "btn btn-default", "data-toggle": "modal", "data-target": "#exportModal"}, "Import / Export")), React.createElement("div", {className: "col-sm-6"}, React.createElement("h4", null, conf.type === "perceptron" ? "Perceptron" : "Net"), React.createElement(BSFormGroup, {id: "learningRate", label: "Learning Rate", isStatic: true}, React.createElement("span", {id: "learningRateVal", style: { marginRight: '1em' }}, conf.learningRate.toFixed(3)), React.createElement("input", {type: "range", min: 0.005, max: 1, step: 0.005, id: "learningRate", value: Util.logScale(conf.learningRate) + "", onChange: loadConfig})), React.createElement(BSFormGroup, {id: "trainingMethod", label: "Training Method"}, React.createElement("select", {id: "trainingMethod", className: "btn btn-default", onChange: loadConfig, value: conf.trainingMethod}, Object.keys(trainingMethods).map(function (name) { return React.createElement("option", {key: name, value: name}, name); }))), conf.type === "perceptron" ?
            React.createElement("div", null, trainingMethods[conf.trainingMethod].trainSingle ?
                React.createElement(BSCheckbox, {label: "Animate single data points", id: "animationTrainSinglePoints", onChange: loadConfig, conf: conf})
                : "", React.createElement(BSCheckbox, {label: "Draw Arrows", id: "drawArrows", onChange: loadConfig, conf: conf}), React.createElement(BSCheckbox, {label: "Draw coordinate system", id: "drawCoordinateSystem", onChange: loadConfig, conf: conf}))
            :
                React.createElement("div", null, React.createElement(NeuronGui, React.__spread({}, this.props)))));
    };
    return ConfigurationGui;
}(React.Component));
/** GUI for displaying the configuration of a single neuron layer */
var NeuronLayer = (function (_super) {
    __extends(NeuronLayer, _super);
    function NeuronLayer() {
        _super.apply(this, arguments);
    }
    NeuronLayer.prototype.render = function () {
        var p = this.props;
        return React.createElement("div", null, p.name, " layer: ", p.layer.neuronCount, " neurons ", React.createElement("button", {className: "btn btn-xs btn-default", onClick: function () { return p.countChanged(1); }}, "+"), React.createElement("button", {className: "btn btn-xs btn-default", onClick: function () { return p.countChanged(-1); }}, "-"), p.layer.activation ?
            React.createElement("select", {className: "btn btn-xs btn-default activation", onChange: function (e) { return p.activationChanged(e.target.value); }, value: p.layer.activation}, Object.keys(Net.NonLinearities).map(function (name) { return React.createElement("option", {key: name, value: name}, name); }))
            : "");
    };
    return NeuronLayer;
}(React.Component));
/** GUI for configuring the neuron layers */
var NeuronGui = (function (_super) {
    __extends(NeuronGui, _super);
    function NeuronGui() {
        _super.apply(this, arguments);
    }
    NeuronGui.prototype.addLayer = function () {
        var hiddenLayers = this.props.hiddenLayers.slice();
        hiddenLayers.unshift({ activation: 'sigmoid', neuronCount: 2 });
        Simulation.instance.setState({ hiddenLayers: hiddenLayers, custom: true });
    };
    NeuronGui.prototype.removeLayer = function () {
        if (this.props.hiddenLayers.length == 0)
            return;
        var hiddenLayers = this.props.hiddenLayers.slice();
        hiddenLayers.shift();
        Simulation.instance.setState({ hiddenLayers: hiddenLayers, custom: true });
    };
    NeuronGui.prototype.activationChanged = function (i, a) {
        var newConf = Util.cloneConfig(this.props);
        if (i == this.props.hiddenLayers.length)
            newConf.outputLayer.activation = a;
        else
            newConf.hiddenLayers[i].activation = a;
        newConf.custom = true;
        Simulation.instance.setState(newConf);
    };
    NeuronGui.prototype.countChanged = function (i, inc) {
        var newState = Util.cloneConfig(this.props);
        var targetLayer;
        var ioDimensionChanged = true;
        if (i === this.props.hiddenLayers.length) {
            // is output layer
            targetLayer = newState.outputLayer;
            if (targetLayer.neuronCount >= 10)
                return;
        }
        else if (i >= 0) {
            // is hidden layer
            targetLayer = newState.hiddenLayers[i];
            ioDimensionChanged = false;
        }
        else {
            // < 0: is input layer
            targetLayer = newState.inputLayer;
        }
        var newval = targetLayer.neuronCount + inc;
        if (newval < 1)
            return;
        targetLayer.neuronCount = newval;
        if (ioDimensionChanged)
            newState.data = [];
        newState.custom = true;
        Simulation.instance.setState(newState);
    };
    NeuronGui.prototype.render = function () {
        var _this = this;
        var conf = this.props;
        var neuronListeners = function (i) { return ({
            activationChanged: function (a) { return _this.activationChanged(i, a); },
            countChanged: function (c) { return _this.countChanged(i, c); }
        }); };
        return React.createElement("div", null, (conf.hiddenLayers.length + 2) + " layers ", React.createElement("button", {className: "btn btn-xs btn-default", onClick: function () { return _this.addLayer(); }}, "+"), React.createElement("button", {className: "btn btn-xs btn-default", onClick: function () { return _this.removeLayer(); }}, "-"), React.createElement(NeuronLayer, React.__spread({key: -1, layer: conf.inputLayer, name: "Input"}, neuronListeners(-1))), conf.hiddenLayers.map(function (layer, i) {
            return React.createElement(NeuronLayer, React.__spread({key: i, layer: layer, name: "Hidden"}, neuronListeners(i)));
        }), React.createElement(NeuronLayer, React.__spread({key: -2, layer: conf.outputLayer, name: "Output"}, neuronListeners(conf.hiddenLayers.length))));
    };
    return NeuronGui;
}(React.Component));
/** display the error history from [[Simulation#errorHistory]] as a line graph */
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
        var data = [this.sim.stepsCurrent, this.sim.averageError];
        this.chart.series[0].addPoint(data, true, false);
    };
    ErrorGraph.prototype.onView = function () {
        this.chart.series[0].setData(this.sim.errorHistory.map(function (x) { return x.slice(); }));
        this.chart.reflow();
    };
    ErrorGraph.prototype.onNetworkLoaded = function () {
        this.chart.series[0].setData([]);
    };
    ErrorGraph.prototype.onHide = function () { };
    return ErrorGraph;
}());
/** show the network as a ordered left-to-right graph using arrow color, width and label to show weights */
var NetworkGraph = (function () {
    function NetworkGraph(sim) {
        this.sim = sim;
        this.actions = ["Network Graph"];
        this.container = $("<div>");
        this.currentlyDisplayingForwardPass = false;
        this.biasBeforeForwardPass = false;
        this.instantiateGraph();
    }
    NetworkGraph.prototype.instantiateGraph = function () {
        this.nodes = new vis.DataSet([], { queue: true });
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
    NetworkGraph.prototype.edgeId = function (conn) {
        return conn.inp.id * this.net.connections.length + conn.out.id;
    };
    NetworkGraph.prototype.onNetworkLoaded = function (net, forceRedraw) {
        if (forceRedraw === void 0) { forceRedraw = false; }
        if (!forceRedraw && this.net
            && this.net.layers.length == net.layers.length
            && this.net.layers.every(function (layer, index) { return layer.length == net.layers[index].length; })
            && this.showbias === this.sim.state.bias) {
            // same net layout, only update
            this.net = net;
            this.onFrame(0);
            return;
        }
        this.showbias = this.sim.state.bias;
        this.net = net;
        this.drawGraph();
    };
    /** initialize graph nodes and edges */
    NetworkGraph.prototype.drawGraph = function () {
        this.nodes.clear();
        this.edges.clear();
        var net = this.net;
        for (var lid = 0; lid < net.layers.length; lid++) {
            var layer = net.layers[lid];
            var nid = 1;
            var layerWithBias = layer;
            if (this.showbias && net.biases[lid])
                layerWithBias = layer.concat(net.biases[lid]);
            for (var _i = 0, layerWithBias_1 = layerWithBias; _i < layerWithBias_1.length; _i++) {
                var neuron = layerWithBias_1[_i];
                var type = 'Hidden Neuron ' + (nid++);
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
        for (var _a = 0, _b = net.connections; _a < _b.length; _a++) {
            var conn = _b[_a];
            this.edges.add({
                id: this.edgeId(conn),
                from: conn.inp.id,
                to: conn.out.id,
                arrows: 'to',
                label: conn.weight.toFixed(2),
            });
        }
        this.nodes.flush();
        this.edges.flush();
        this.graph.stabilize();
        this.graph.fit();
    };
    /** calculate the visualization of the individual calculation steps for a single forward pass */
    NetworkGraph.prototype.forwardPass = function (data) {
        var _this = this;
        if (this.currentlyDisplayingForwardPass)
            this.onFrame(0);
        this.biasBeforeForwardPass = this.showbias;
        this.showbias = true;
        this.currentlyDisplayingForwardPass = true;
        this.drawGraph();
        this.net.setInputsAndCalculate(data.input);
        var updates = [{ nodes: [], edges: [] }];
        // reset all names
        for (var _i = 0, _a = this.net.layers; _i < _a.length; _i++) {
            var layer = _a[_i];
            for (var _b = 0, layer_2 = layer; _b < layer_2.length; _b++) {
                var neuron = layer_2[_b];
                updates[0].nodes.push({
                    id: neuron.id,
                    label: "0"
                });
            }
        }
        for (var _c = 0, _d = this.net.biases; _c < _d.length; _c++) {
            var neuron = _d[_c];
            updates[0].nodes.push({
                id: neuron.id,
                label: "Bias (1)"
            });
        }
        for (var i = 0; i < data.input.length; i++) {
            updates[0].nodes.push({
                id: this.net.inputs[i].id,
                label: this.net.inputs[i].name + " = " + data.input[i].toFixed(2)
            });
        }
        var allEdgesInvisible = function () { return _this.net.connections.map(function (conn) { return ({
            id: _this.edgeId(conn),
            color: "rgba(255,255,255,0)",
            label: undefined
        }); }); };
        updates[0].edges = allEdgesInvisible();
        // passes
        var lastNeuron;
        for (var _e = 0, _f = this.net.layers.slice(1); _e < _f.length; _e++) {
            var layer = _f[_e];
            for (var _g = 0, layer_3 = layer; _g < layer_3.length; _g++) {
                var neuron = layer_3[_g];
                if (neuron instanceof Net.InputNeuron)
                    continue; // bias neuron
                updates.push({
                    highlightNodes: [neuron.id],
                    nodes: lastNeuron ? [{ id: lastNeuron.id, label: lastNeuron.output.toFixed(2) }] : [],
                    edges: allEdgesInvisible().concat(neuron.inputs.map(function (i) { return ({
                        id: _this.edgeId(i),
                        color: "black",
                        label: ""
                    }); }))
                });
                var neuronVal = 0;
                for (var _h = 0, _j = neuron.inputs; _h < _j.length; _h++) {
                    var input = _j[_h];
                    var add = input.inp.output * input.weight;
                    neuronVal += add;
                    var update = {
                        nodes: [{ id: neuron.id, label: "\u2211 = " + neuronVal.toFixed(2) }],
                        edges: [{ id: this.edgeId(input), label: "+ " + input.inp.output.toFixed(2) + " \u00B7 (" + input.weight.toFixed(2) + ")" }],
                        highlightNodes: [],
                        highlightEdges: [this.edgeId(input)]
                    };
                    updates.push(update);
                }
                updates.push({
                    nodes: [{ id: neuron.id, label: "\u03C3(" + neuronVal.toFixed(2) + ") = " + neuron.output.toFixed(2) }],
                    edges: allEdgesInvisible()
                });
                lastNeuron = neuron;
            }
        }
        return updates;
    };
    NetworkGraph.prototype.applyUpdate = function (update) {
        this.edges.update(update.edges);
        this.nodes.update(update.nodes);
        this.nodes.flush();
        this.edges.flush();
        if (update.highlightNodes)
            this.graph.selectNodes(update.highlightNodes, false);
        if (update.highlightEdges)
            this.graph.selectEdges(update.highlightEdges);
    };
    NetworkGraph.prototype.onFrame = function (framenum) {
        if (this.currentlyDisplayingForwardPass) {
            // abort forward pass
            this.showbias = this.biasBeforeForwardPass;
            this.onNetworkLoaded(this.net, true);
            this.currentlyDisplayingForwardPass = false;
        }
        if (this.net.connections.length > 20 && framenum % 15 !== 0) {
            // skip some frames because slow
            return;
        }
        for (var _i = 0, _a = this.net.connections; _i < _a.length; _i++) {
            var conn = _a[_i];
            this.edges.update({
                id: this.edgeId(conn),
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
}());
var NetType;
(function (NetType) {
    /** 2D-Input, Single Output (≤0.5 is class 0, otherwise class 1) */
    NetType[NetType["BinaryClassify"] = 0] = "BinaryClassify";
    /** 2D Input, 2D output, input values = output values*/
    NetType[NetType["AutoEncode"] = 1] = "AutoEncode";
    /** 2D Input, ≥ 3 outputs (argmax(outputs) is the resulting class) */
    NetType[NetType["MultiClass"] = 2] = "MultiClass";
    /** any other configuration */
    NetType[NetType["CantDraw"] = 3] = "CantDraw";
})(NetType || (NetType = {}));
/**
 * Visualize the training data and network output in a 2d canvas
 *
 * Only works for problems with 2-dimensional input
 *
 * For classification problems, draw every data point with the input as the position
 * and it's label/class as the color. Also fill the background with the current network output for that position.
 *
 * For 2D auto encoding draw every target output point connected to the network output.
 */
var NetworkVisualization = (function () {
    function NetworkVisualization(sim, dataPointIsHighlighted) {
        var _this = this;
        this.sim = sim;
        this.dataPointIsHighlighted = dataPointIsHighlighted;
        this.actions = [];
        /** [0] = Drag View; [length - 1] = Remove on click; [n] = Add data point of class (n-1) */
        this.inputMode = 0;
        this.backgroundResolution = 15;
        this.container = $("<div>");
        this.netType = NetType.BinaryClassify;
        var tmp = NetworkVisualization.colors.multiClass;
        tmp.bg = tmp.fg.map(function (c) { return Util.printColor(Util.parseColor(c).map(function (x) { return (x * 1.3) | 0; })); });
        this.canvas = $("<canvas class=fullsize>")[0];
        this.canvas.width = 550;
        this.canvas.height = 400;
        this.trafo = new TransformNavigation(this.canvas, function () { return _this.inputMode === 0; } /* move view mode */ /* move view mode */, function () { return _this.onFrame(); });
        this.ctx = this.canvas.getContext('2d');
        window.addEventListener('resize', this.canvasResized.bind(this));
        this.canvas.addEventListener("click", this.canvasClicked.bind(this));
        this.canvas.addEventListener("contextmenu", this.canvasClicked.bind(this));
        this.canvas.addEventListener("mousedown", Util.stopEvent); // prevent select text
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
                this.actions = ["Move View", "Add Red", "Add Green", "Remove"];
                break;
            case NetType.AutoEncode:
                this.actions = ["Move View", "Add Data point", "", "Remove"];
                break;
            case NetType.MultiClass:
                this.actions = ["Move View"];
                var i = 0;
                for (var _i = 0, _a = this.sim.state.outputLayer.names; _i < _a.length; _i++) {
                    var name_1 = _a[_i];
                    this.actions.push({ name: name_1, color: NetworkVisualization.colors.multiClass.bg[i++] });
                }
                this.actions.push("Remove");
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
        var isSinglePerceptron = this.sim.state.type === "perceptron";
        var separator = isSinglePerceptron && this.getSeparator(Util.toLinearFunction(this.sim.net.connections.map(function (i) { return i.weight; })));
        if (isSinglePerceptron)
            this.drawPolyBackground(separator);
        else
            this.drawBackground();
        if (this.sim.state.drawCoordinateSystem)
            this.drawCoordinateSystem();
        if (this.sim.state.drawArrows)
            this.drawArrows();
        this.drawDataPoints();
        if (isSinglePerceptron) {
            var tor = this.trafo.toReal;
            if (this.sim.state.drawArrows && this.sim.lastWeights !== undefined && this.sim.lastWeights.length > 0) {
                var separator_1 = this.getSeparator(Util.toLinearFunction(this.sim.lastWeights[0].weights));
                this.drawLine(tor.x(0), separator_1.min, tor.x(this.canvas.width), separator_1.max, "gray");
            }
            this.drawLine(tor.x(0), separator.min, tor.x(this.canvas.width), separator.max, "black");
        }
    };
    NetworkVisualization.prototype.drawDataPoints = function () {
        this.ctx.strokeStyle = "#000";
        if (this.netType === NetType.BinaryClassify || this.netType === NetType.MultiClass) {
            for (var _i = 0, _a = this.sim.state.data; _i < _a.length; _i++) {
                var val = _a[_i];
                this.drawDataPoint(val);
            }
        }
        else if (this.netType === NetType.AutoEncode) {
            for (var _b = 0, _c = this.sim.state.data; _b < _c.length; _b++) {
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
    NetworkVisualization.prototype.drawDataPoint = function (p) {
        var color = this.netType === NetType.BinaryClassify ?
            NetworkVisualization.colors.binaryClassify.fg[p.output[0] | 0]
            : this.netType === NetType.MultiClass ?
                NetworkVisualization.colors.multiClass.fg[Util.getMaxIndex(p.output)]
                : null;
        this.drawPoint(p.input[0], p.input[1], color, this.dataPointIsHighlighted(p));
    };
    NetworkVisualization.prototype.drawPoint = function (x, y, color, highlight) {
        if (highlight === void 0) { highlight = false; }
        x = this.trafo.toCanvas.x(x), y = this.trafo.toCanvas.y(y);
        this.ctx.fillStyle = color;
        this.ctx.beginPath();
        this.ctx.lineWidth = highlight ? 5 : 1;
        this.ctx.strokeStyle = highlight ? "#000000" : "#000000";
        this.ctx.arc(x, y, 5, 0, 2 * Math.PI);
        this.ctx.fill();
        this.ctx.beginPath();
        this.ctx.arc(x, y, highlight ? 7 : 5, 0, 2 * Math.PI);
        this.ctx.stroke();
    };
    /** draw the weight vector arrows according to [[Simulation#lastWeights]] */
    NetworkVisualization.prototype.drawArrows = function () {
        var _this = this;
        this.ctx.lineWidth = 2;
        var al = 8;
        var aw = 4;
        var steps = this.sim.lastWeights;
        if (steps === undefined || steps.length < 2)
            return;
        var scale = {
            x: function (x) { return _this.trafo.toCanvas.x(x * _this.sim.state.arrowScale); },
            y: function (y) { return _this.trafo.toCanvas.y(y * _this.sim.state.arrowScale); }
        };
        if (this.sim.state.inputLayer.neuronCount !== 2
            || this.sim.state.outputLayer.neuronCount !== 1
            || this.sim.state.hiddenLayers.length !== 0)
            throw Error("conf not valid for arrows");
        var weightVector = function (weights) { return ({ x: scale.x(weights[0]), y: scale.y(weights[1]) }); };
        if (steps[steps.length - 1].weights.some(function (x, i) { return x !== steps[0].weights[i]; })) {
            var oldWeights = steps[0].weights.map(function (x) { return 0; });
            for (var _i = 0, steps_1 = steps; _i < steps_1.length; _i++) {
                var _a = steps_1[_i], weights = _a.weights, dataPoint = _a.dataPoint;
                this.ctx.strokeStyle = this.ctx.fillStyle = dataPoint ?
                    NetworkVisualization.colors.binaryClassify.weightVector[dataPoint.output[0]]
                    : "#888";
                Util.drawArrow(this.ctx, weightVector(oldWeights), weightVector(weights), al, aw);
                this.ctx.strokeStyle = this.ctx.fillStyle = "#808080";
                // this.ctx.arc(scale.x(p.input[0]), scale.y(p.input[1]), 8, 0, 2*Math.PI);
                oldWeights = weights;
            }
            this.ctx.strokeStyle = this.ctx.fillStyle = "#000000";
            Util.drawArrow(this.ctx, { x: scale.x(0), y: scale.y(0) }, weightVector(steps[steps.length - 1].weights), al, aw);
        }
    };
    /**
     * calculate the y position of the given function to the left and right of the canvas in actual/real coordinates
     */
    NetworkVisualization.prototype.getSeparator = function (lineFunction) {
        return {
            min: lineFunction(this.trafo.toReal.x(0)),
            max: lineFunction(this.trafo.toReal.x(this.canvas.width))
        };
    };
    NetworkVisualization.prototype.drawLine = function (x, y, x2, y2, color) {
        x = this.trafo.toCanvas.x(x);
        x2 = this.trafo.toCanvas.x(x2);
        y = this.trafo.toCanvas.y(y);
        y2 = this.trafo.toCanvas.y(y2);
        this.ctx.strokeStyle = color;
        this.ctx.beginPath();
        this.ctx.lineWidth = 2;
        this.ctx.moveTo(x, y);
        this.ctx.lineTo(x2, y2);
        this.ctx.stroke();
    };
    NetworkVisualization.prototype.clear = function (color) {
        this.ctx.fillStyle = "white";
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        return;
    };
    /** divide the canvas into two regions using the linear function run through [[getSeparator]] as the divider and color the regions */
    NetworkVisualization.prototype.drawPolyBackground = function (_a) {
        var _this = this;
        var min = _a.min, max = _a.max;
        var colors = NetworkVisualization.colors.binaryClassify.bg;
        var ctx = this.ctx;
        var c = this.trafo.toCanvas;
        var tmp = function (y) {
            ctx.beginPath();
            ctx.moveTo(0, c.y(min));
            ctx.lineTo(0, y);
            ctx.lineTo(_this.canvas.width, y);
            ctx.lineTo(_this.canvas.width, c.y(max));
            ctx.fill();
        };
        var upperIsClass1 = +(this.sim.net.getOutput([this.trafo.toReal.x(0), min - 1])[0] > 0.5);
        ctx.fillStyle = colors[1 - upperIsClass1];
        tmp(0);
        ctx.fillStyle = colors[upperIsClass1];
        tmp(this.canvas.height);
    };
    NetworkVisualization.prototype.drawBackground = function () {
        if (this.sim.state.outputLayer.neuronCount === 2) {
            this.clear('white');
            return;
        }
        for (var x = 0; x < this.canvas.width; x += this.backgroundResolution) {
            for (var y = 0; y < this.canvas.height; y += this.backgroundResolution) {
                var vals = this.sim.net.getOutput([this.trafo.toReal.x(x + this.backgroundResolution / 2), this.trafo.toReal.y(y + this.backgroundResolution / 2)]);
                if (this.sim.state.outputLayer.neuronCount > 2) {
                    this.ctx.fillStyle = NetworkVisualization.colors.multiClass.bg[Util.getMaxIndex(vals)];
                }
                else {
                    if (this.sim.state.showGradient) {
                        this.ctx.fillStyle = NetworkVisualization.colors.binaryClassify.gradient(vals[0]);
                    }
                    else
                        this.ctx.fillStyle = NetworkVisualization.colors.binaryClassify.bg[+(vals[0] > 0.5)];
                }
                this.ctx.fillRect(x, y, this.backgroundResolution, this.backgroundResolution);
            }
        }
    };
    NetworkVisualization.prototype.drawCoordinateSystem = function () {
        var marklen = 0.1;
        var ctx = this.ctx, toc = this.trafo.toCanvas;
        ctx.strokeStyle = "#000";
        ctx.fillStyle = "#000";
        ctx.textBaseline = "middle";
        ctx.textAlign = "center";
        ctx.font = "20px monospace";
        ctx.beginPath();
        this.ctx.lineWidth = 2;
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
        if (this.sim.state.data.length < 3)
            return;
        // update transform
        if (this.sim.state.inputLayer.neuronCount == 2) {
            var fillamount = 0.6;
            var bounds = Util.bounds2dTrainingsInput(this.sim.state.data);
            var w = bounds.maxx - bounds.minx, h = bounds.maxy - bounds.miny;
            var scale = Math.min(this.canvas.width / w, this.canvas.height / h) * fillamount;
            this.trafo.scalex = scale;
            this.trafo.scaley = -scale;
            this.trafo.offsetx = -(bounds.maxx + bounds.minx) / 2 * scale + this.canvas.width / 2;
            this.trafo.offsety = (bounds.maxy + bounds.miny) / 2 * scale + this.canvas.height / 2;
        }
    };
    NetworkVisualization.prototype.canvasClicked = function (evt) {
        Util.stopEvent(evt);
        var data = this.sim.state.data.slice();
        var rect = this.canvas.getBoundingClientRect();
        var x = this.trafo.toReal.x(evt.clientX - rect.left);
        var y = this.trafo.toReal.y(evt.clientY - rect.top);
        var removeMode = this.actions.length - 1;
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
        else if (this.inputMode < removeMode && this.inputMode > 0 /* move mode */) {
            // add data point
            if (this.netType === NetType.AutoEncode) {
                data.push({ input: [x, y], output: [x, y] });
            }
            else {
                var inv = function (x) { return x == 0 ? 1 : 0; };
                var label = this.inputMode - 1;
                if (evt.button != 0)
                    label = inv(label);
                if (evt.ctrlKey || evt.metaKey || evt.altKey)
                    label = inv(label);
                var output = [label];
                if (this.netType === NetType.MultiClass) {
                    output = Util.arrayWithOneAt(this.sim.state.outputLayer.neuronCount, label);
                }
                data.push({ input: [x, y], output: output });
            }
        }
        else
            return;
        this.sim.setState({ data: data, custom: true });
        this.sim.lastWeights = undefined;
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
            /** background color */
            bg: ["#f88", "#8f8"],
            /** data point color */
            fg: ["#f00", "#0f0"],
            /** color of weight arrows */
            weightVector: ["#800", "#080"],
            /** used when displaying the class in the background as a gradient*/
            gradient: function (val) { return "rgb(" +
                [(((1 - val) * (256 - 60)) | 0) + 60, ((val * (256 - 60)) | 0) + 60, 60] + ")"; }
        },
        autoencoder: {
            input: '#2188e0',
            output: '#ff931f',
            bias: '#aaa'
        },
        multiClass: {
            fg: ['#7cb5ec', '#434348', '#90ed7d', '#f7a35c', '#8085e9', '#f15c80', '#e4d354', '#2b908f', '#f45b5b', '#91e8e1'],
            /** filled in constructor by darkening fg colors */
            bg: ['']
        }
    };
    return NetworkVisualization;
}());
/**
 * Edit training data and display network output using a table interface
 */
var TableEditor = (function () {
    function TableEditor(sim) {
        this.sim = sim;
        this.actions = ["Table input"];
        this.headerCount = 2;
        this.lastUpdate = 0;
        this.container = $("<div>");
    }
    /** called by Handsontable after some data was changed in the table */
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
        this.container = $("<div class='fullsize' style='overflow:hidden'>");
        if (oldContainer)
            oldContainer.replaceWith(this.container);
        $("<div>").addClass("btn btn-default")
            .css({ position: "absolute", right: "2em", bottom: "2em" })
            .text("Remove all")
            .click(function (e) { return _this.sim.setState({ data: [] }, function () { return _this.loadData(); }); })
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
            /*customBorders: false[{ // bug when larger than ~4
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
                }],*/
            allowInvalid: false,
            mergeCells: mergeCells,
            afterChange: this.afterChange.bind(this)
        };
        this.container.handsontable(_conf);
        this.hot = this.container.handsontable('getInstance');
        this.loadData();
    };
    /** read data from table into the [[Configuration]] */
    TableEditor.prototype.reparseData = function () {
        var sim = this.sim;
        var data = this.hot.getData();
        var headers = data[1];
        var newConfig = Util.cloneConfig(sim.state);
        var ic = newConfig.inputLayer.neuronCount, oc = newConfig.outputLayer.neuronCount;
        newConfig.inputLayer.names = headers.slice(0, ic);
        newConfig.outputLayer.names = headers.slice(ic, ic + oc);
        newConfig.data = data.slice(2).map(function (row) { return row.slice(0, ic + oc); })
            .filter(function (row) { return row.every(function (cell) { return typeof cell === 'number'; }); })
            .map(function (row) { return { input: row.slice(0, ic), output: row.slice(ic) }; });
        newConfig.custom = true;
        sim.setState(newConfig);
    };
    TableEditor.prototype.onFrame = function () {
        var sim = this.sim;
        if ((Date.now() - this.lastUpdate) < 500)
            return;
        this.lastUpdate = Date.now();
        var xOffset = sim.state.inputLayer.neuronCount + sim.state.outputLayer.neuronCount;
        var vals = [];
        for (var y = 0; y < sim.state.data.length; y++) {
            var p = sim.state.data[y];
            var op = sim.net.getOutput(p.input);
            for (var x = 0; x < op.length; x++) {
                vals.push([y + this.headerCount, xOffset + x, op[x]]);
            }
        }
        this.hot.setDataAtCell(vals, "loadData");
    };
    /** load data from [[Configuration]] into the table */
    TableEditor.prototype.loadData = function () {
        var sim = this.sim;
        var data = [[], sim.state.inputLayer.names.concat(sim.state.outputLayer.names).concat(sim.state.outputLayer.names)];
        var ic = sim.state.inputLayer.neuronCount, oc = sim.state.outputLayer.neuronCount;
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
        sim.state.data.forEach(function (t) { return data.push(t.input.concat(t.output)); });
        this.hot.loadData(data);
        /*this.hot.updateSettings({customBorders: [
                
            ]});
        this.hot.runHooks('afterInit');*/
    };
    TableEditor.prototype.onView = function () {
        this.onNetworkLoaded(this.sim.net);
        this.onFrame();
    };
    TableEditor.prototype.onHide = function () {
        //this.reparseData();
    };
    return TableEditor;
}());
/** display multiple Visualizations */
var MultiVisDisplayer = (function (_super) {
    __extends(MultiVisDisplayer, _super);
    function MultiVisDisplayer(props) {
        _super.call(this, props);
        this.bodyDivs = [];
        this.state = {
            running: false,
            bodies: [null, null],
            correct: "",
            stepNum: 0
        };
    }
    MultiVisDisplayer.prototype.onFrame = function (framenum) {
        for (var _i = 0, _a = this.state.bodies; _i < _a.length; _i++) {
            var body = _a[_i];
            if (body)
                body.onFrame(framenum);
        }
    };
    MultiVisDisplayer.prototype.changeBody = function (i, vis, aft) {
        var _this = this;
        this.bodiesTmp = this.bodiesTmp || this.state.bodies.slice();
        this.bodiesTmp[i] = vis;
        this.setState({ bodies: this.bodiesTmp }, function () { _this.bodiesTmp = undefined; aft(); });
    };
    MultiVisDisplayer.prototype.componentDidUpdate = function (prevProps, prevState) {
        for (var i = 0; i < prevState.bodies.length; i++) {
            if (prevState.bodies[i] !== this.state.bodies[i]) {
                $(this.bodyDivs[i]).children().detach();
                $(this.bodyDivs[i]).append(this.state.bodies[i].container);
            }
        }
    };
    return MultiVisDisplayer;
}(React.Component));
var ControlButtonBar = (function (_super) {
    __extends(ControlButtonBar, _super);
    function ControlButtonBar() {
        _super.apply(this, arguments);
    }
    ControlButtonBar.prototype.render = function () {
        var sim = this.props.sim;
        return React.createElement("div", {className: "h3"}, React.createElement("button", {className: this.props.running ? "btn btn-danger" : "btn btn-primary", onClick: sim.runtoggle.bind(sim)}, this.props.running ? "Stop" : "Animate"), " ", React.createElement("button", {className: "btn btn-warning", onClick: sim.reset.bind(sim)}, "Reset"), " ", React.createElement("button", {className: "btn btn-default", onClick: sim.trainAllButton.bind(sim)}, sim.state.type === "perceptron" ? "Train All" : "Train"), " ", (function () {
            if (sim.state.type === "perceptron" && sim.trainingMethod.trainSingle)
                return React.createElement("button", {className: "btn btn-default", onClick: sim.trainNextButton.bind(sim)}, "Train Single");
            if (sim.state.type === "nn")
                return React.createElement("button", {className: "btn btn-default", onClick: sim.forwardPassStep.bind(sim)}, "Forward Pass Step");
            return null;
        })());
    };
    return ControlButtonBar;
}(React.Component));
var StatusBar = (function (_super) {
    __extends(StatusBar, _super);
    function StatusBar() {
        _super.apply(this, arguments);
    }
    StatusBar.prototype.render = function () {
        return React.createElement("h2", null, this.props.correct, " —  Iteration:   ", this.props.iteration);
    };
    return StatusBar;
}(React.Component));
/** Display two visualizations next to each other with tabbed navigation */
var LRVis = (function (_super) {
    __extends(LRVis, _super);
    function LRVis(props) {
        _super.call(this, props);
    }
    LRVis.prototype.render = function () {
        var _this = this;
        var sim = this.props.sim;
        var isPerceptron = this.props.sim.state.type === "perceptron";
        var leftSize = isPerceptron ? 4 : 6;
        var rightSize = 12 - leftSize;
        return React.createElement("div", null, React.createElement("div", {className: "row"}, React.createElement("div", {className: "col-sm-" + leftSize}, React.createElement(TabSwitcher, {ref: function (c) { return _this.leftVis = c; }, things: this.props.leftVis, onChangeVisualization: function (vis, aft) { return _this.changeBody(0, vis, aft); }})), React.createElement("div", {className: "col-sm-" + rightSize}, React.createElement(TabSwitcher, {ref: function (c) { return _this.rightVis = c; }, things: this.props.rightVis, onChangeVisualization: function (vis, aft) { return _this.changeBody(1, vis, aft); }}))), React.createElement("div", {className: "row"}, React.createElement("div", {className: "col-sm-" + leftSize}, React.createElement("div", {className: "visbody", ref: function (b) { return _this.bodyDivs[0] = b; }}), React.createElement(ControlButtonBar, {running: this.state.running, sim: sim}), React.createElement("hr", null)), React.createElement("div", {className: "col-sm-" + rightSize}, React.createElement("div", {className: "visbody", ref: function (b) { return _this.bodyDivs[1] = b; }}), React.createElement("div", null, React.createElement(StatusBar, {correct: this.state.correct, iteration: this.state.stepNum})), React.createElement("hr", null))));
    };
    return LRVis;
}(MultiVisDisplayer));
/** switch between multiple visualizations using a tabbed interface */
var TabSwitcher = (function (_super) {
    __extends(TabSwitcher, _super);
    function TabSwitcher(props) {
        _super.call(this, props);
        this.state = {
            modes: this.createButtonsAndActions(),
            currentMode: -1
        };
    }
    TabSwitcher.prototype.render = function () {
        var _this = this;
        var isDark = function (color) { return Util.parseColor(color).reduce(function (a, b) { return a + b; }) / 3 < 127; };
        return React.createElement("div", null, React.createElement("ul", {className: "nav nav-pills"}, this.state.modes.map(function (mode, i) {
            return React.createElement("li", {key: i, className: _this.state.currentMode === i ? "custom-active" : ""}, React.createElement("a", {style: mode.color ? { backgroundColor: mode.color, color: isDark(mode.color) ? "white" : "black" } : {}, onClick: function (e) { return _this.setMode(i); }}, mode.text));
        })));
    };
    TabSwitcher.prototype.componentDidMount = function () {
        this.setMode(0, true);
    };
    TabSwitcher.prototype.createButtonsAndActions = function () {
        var modes = [];
        this.props.things.forEach(function (thing, thingid) {
            return thing.actions.forEach(function (button, buttonid) {
                var text = "", color = "";
                if (typeof button === 'string') {
                    text = button;
                }
                else {
                    text = button.name;
                    color = button.color;
                }
                modes.push({ thing: thingid, action: buttonid, text: text, color: color });
            });
        });
        return modes;
    };
    TabSwitcher.prototype.setMode = function (mode, force) {
        if (force === void 0) { force = false; }
        if (!force && mode == this.state.currentMode)
            return;
        var action = this.state.modes[mode];
        var lastAction = this.state.modes[this.state.currentMode];
        this.setState({ currentMode: mode });
        var currentVisualization = this.props.things[action.thing];
        if (force || !lastAction || action.thing != lastAction.thing) {
            if (lastAction)
                this.props.things[lastAction.thing].onHide();
            this.props.onChangeVisualization(currentVisualization, function () {
                return currentVisualization.onView(true, action.action);
            });
        }
        else if (action.action !== lastAction.action) {
            currentVisualization.onView(false, action.action);
        }
    };
    /** network was loaded, check if the children actions have changed and update the tab bar accordingly */
    TabSwitcher.prototype.onNetworkLoaded = function (net) {
        var _this = this;
        //todo: ugly hack
        var beforeActions = JSON.stringify(this.props.things.map(function (t) { return t.actions; }));
        this.props.things.forEach(function (thing) { return thing.onNetworkLoaded(net); });
        var afterActions = JSON.stringify(this.props.things.map(function (t) { return t.actions; }));
        if (beforeActions !== afterActions)
            this.setState({
                modes: this.createButtonsAndActions(),
                currentMode: 0
            }, function () { return _this.setMode(0, true); });
    };
    return TabSwitcher;
}(React.Component));
/**
 * Visualization that displays all the weights in the network as a black-white gradient
 *
 * hidden when in perceptron mode because then it's pretty boring
 */
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
                return inStr + " to " + outStr + ": " + conn.weight.toFixed(2);
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
    /** parse network layout into weights graph ordering */
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
                    if (!this.sim.state.bias && inN instanceof Net.InputNeuron && inN.constant) {
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
        if (this.sim.state.type == "perceptron")
            this.actions = [];
        else
            this.actions = ["Weights"];
        this.graph.setData(this.parseData(net));
    };
    WeightsGraph.prototype.onFrame = function () {
        this.graph.setData(this.parseData(this.sim.net));
    };
    return WeightsGraph;
}());
//# sourceMappingURL=program.js.map