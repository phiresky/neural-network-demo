var __extends = this.__extends || function (d, b) {
    for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p];
    function __() { this.constructor = d; }
    __.prototype = b.prototype;
    d.prototype = new __();
};
var Net;
(function (Net) {
    var NonLinearity = {
        sigmoid: function (x) { return 1 / (1 + Math.exp(-x)); },
        sigDiff: function (x) { return x * (1 - x); }
    };
    function makeArray(len, supplier) {
        var arr = new Array(len);
        for (var i = 0; i < len; i++)
            arr[i] = supplier();
        return arr;
    }
    var NeuralNet = (function () {
        function NeuralNet(counts) {
            this.conns = [];
            this.learnRate = 0.01;
            this.inputs = makeArray(counts[0], function () { return new InputNeuron(); });
            var hidden = makeArray(counts[1], function () { return new Neuron(); });
            this.outputs = makeArray(counts[2], function () { return new OutputNeuron(); });
            this.layers = [this.inputs, hidden, this.outputs];
            var onNeuron = new InputNeuron(1);
            this.inputs.push(onNeuron);
            var startWeight = function () { return Math.random() - 0.5; };
            console.log(this.layers);
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
                        this.conns.push(conn);
                    }
                }
            }
        }
        NeuralNet.prototype.setInputs = function (inputVals) {
            if (inputVals.length != this.inputs.length - 1)
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
            for (var _i = 0, _a = this.conns; _i < _a.length; _i++) {
                var conn = _a[_i];
                conn._tmpw = conn.getDeltaWeight(this.learnRate);
            }
            for (var _b = 0, _c = this.conns; _b < _c.length; _b++) {
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
        function Neuron() {
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
            return NonLinearity.sigmoid(this.weightedInputs());
        };
        Neuron.prototype.getError = function () {
            var δ = 0;
            for (var _i = 0, _a = this.outputs; _i < _a.length; _i++) {
                var output = _a[_i];
                δ += output.out.getError() * output.weight;
            }
            return δ * NonLinearity.sigDiff(this.getOutput());
        };
        return Neuron;
    })();
    Net.Neuron = Neuron;
    var InputNeuron = (function (_super) {
        __extends(InputNeuron, _super);
        function InputNeuron(input) {
            if (input === void 0) { input = 0; }
            _super.call(this);
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
            return Math.max(Math.min(_super.prototype.weightedInputs.call(this), 1), 0);
        };
        OutputNeuron.prototype.getError = function () {
            var oup = this.getOutput();
            return NonLinearity.sigDiff(oup) *
                (this.targetOutput - oup);
        };
        return OutputNeuron;
    })(Neuron);
    Net.OutputNeuron = OutputNeuron;
})(Net || (Net = {}));
//# sourceMappingURL=Net.js.map