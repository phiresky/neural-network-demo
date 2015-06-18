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
    Net.learnRate = 0.01;
    var NeuronConnection = (function () {
        function NeuronConnection(inp, out, weight) {
            this.inp = inp;
            this.out = out;
            this.weight = weight;
        }
        NeuronConnection.prototype.getDeltaWeight = function () {
            return Net.learnRate * this.out.getError() * this.inp.getOutput();
        };
        return NeuronConnection;
    })();
    Net.NeuronConnection = NeuronConnection;
    var HiddenNeuron = (function () {
        function HiddenNeuron() {
            this.inputs = [];
            this.outputs = [];
        }
        HiddenNeuron.prototype.getOutputRaw = function () {
            var output = 0;
            for (var _i = 0, _a = this.inputs; _i < _a.length; _i++) {
                var conn = _a[_i];
                output += conn.inp.getOutput() * conn.weight;
            }
            return output;
        };
        HiddenNeuron.prototype.getOutput = function () {
            return NonLinearity.sigmoid(this.getOutputRaw());
        };
        HiddenNeuron.prototype.getError = function () {
            var δ = 0;
            for (var _i = 0, _a = this.outputs; _i < _a.length; _i++) {
                var output = _a[_i];
                δ += output.out.getError() * output.weight;
            }
            return δ * NonLinearity.sigDiff(this.getOutput());
        };
        return HiddenNeuron;
    })();
    Net.HiddenNeuron = HiddenNeuron;
    var InputNeuron = (function () {
        function InputNeuron(input) {
            if (input === void 0) { input = 0; }
            this.input = input;
            this.outputs = [];
        }
        InputNeuron.prototype.getOutputRaw = function () {
            return this.input;
        };
        InputNeuron.prototype.getOutput = function () {
            return this.input;
        };
        InputNeuron.prototype.getError = function () {
            return 0;
        };
        return InputNeuron;
    })();
    Net.InputNeuron = InputNeuron;
    var OutputNeuron = (function (_super) {
        __extends(OutputNeuron, _super);
        function OutputNeuron() {
            _super.apply(this, arguments);
            this.inputs = [];
        }
        OutputNeuron.prototype.getOutput = function () {
            return Math.max(Math.min(_super.prototype.getOutputRaw.call(this), 0.999), 0.001);
        };
        OutputNeuron.prototype.getError = function () {
            var oup = Math.abs(this.getOutput());
            return NonLinearity.sigDiff(oup) *
                (this.targetOutput - oup);
        };
        return OutputNeuron;
    })(HiddenNeuron);
    Net.OutputNeuron = OutputNeuron;
})(Net || (Net = {}));
//# sourceMappingURL=Net.js.map