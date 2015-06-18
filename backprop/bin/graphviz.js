/// <reference path="../lib/typings/jquery/jquery.d.ts" />
var GraphNode = (function () {
    function GraphNode(edges, x, y, vx, vy) {
        if (edges === void 0) { edges = []; }
        if (x === void 0) { x = 0; }
        if (y === void 0) { y = 0; }
        if (vx === void 0) { vx = 0; }
        if (vy === void 0) { vy = 0; }
        this.edges = edges;
        this.x = x;
        this.y = y;
        this.vx = vx;
        this.vy = vy;
        this.ax = 0;
        this.ay = 0;
    }
    return GraphNode;
})();
var Graph = (function () {
    function Graph(nodes) {
        this.nodes = nodes;
        this.targetDist = 100;
        this.friction = 0.9;
        this.gravity = -700;
        this.attraction = 1 / 1000;
        this.leftright = 0.05;
    }
    Graph.prototype.update = function () {
        for (var i = 0; i < this.nodes.length; i++) {
            var n = this.nodes[i];
            for (var e = 0; e < n.edges.length; e++) {
                var m = this.nodes[n.edges[e]];
                var dx = n.x - m.x;
                var dy = n.y - m.y;
                var d = Math.sqrt(dx * dx + dy * dy);
                var f = (d - this.targetDist) * this.attraction;
                n.ax -= f * dx / d;
                n.ay -= f * dy / d;
                m.ax += f * dx / d;
                m.ay += f * dy / d;
                n.ax -= this.leftright;
                m.ax += this.leftright;
            }
            for (var j = i; j < this.nodes.length; j++) {
                var m = this.nodes[j];
                var dx = n.x - m.x;
                var dy = n.y - m.y;
                var d = Math.sqrt(dx * dx + dy * dy);
                if (d === 0)
                    continue;
                var f = this.gravity / d / d; //m1*m2/d^2;
                n.ax -= f * dx / d;
                n.ay -= f * dy / d;
                m.ax += f * dx / d;
                m.ay += f * dy / d;
            }
        }
        for (var i = 0; i < this.nodes.length; i++) {
            var n = this.nodes[i];
            n.vx += n.ax;
            n.vy += n.ay;
            n.vx *= this.friction;
            n.vy *= this.friction;
            n.x += n.vx;
            n.y += n.vy;
            n.ax = 0;
            n.ay = 0;
        }
    };
    return Graph;
})();
var Util = (function () {
    function Util() {
    }
    Util.drawCircle = function (ctx, x, y, radius) {
        ctx.beginPath();
        ctx.arc(x, y, 10, 0, Math.PI * 2);
        ctx.closePath();
        ctx.fill();
    };
    Util.drawLine = function (ctx, x1, y1, x2, y2) {
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        Util.drawArrowHead(ctx, x1, y1, x2, y2);
    };
    Util.drawArrowHead = function (ctx, x2, y2, x1, y1) {
        var radians = Math.atan((y2 - y1) / (x2 - x1));
        radians += ((x2 > x1) ? -90 : 90) * Math.PI / 180;
        ctx.save();
        ctx.beginPath();
        ctx.translate(x1, y1);
        ctx.rotate(radians);
        ctx.translate(0, 9);
        ctx.moveTo(0, 0);
        ctx.lineTo(5, 10);
        ctx.lineTo(-5, 10);
        ctx.closePath();
        ctx.restore();
        ctx.fill();
    };
    return Util;
})();
var GraphViewer = (function () {
    function GraphViewer(containerElement, width, height, data) {
        var _this = this;
        this.mouseInfo = { down: false, downX: 0, downY: 0 };
        this.drawOffset = { x: 0, y: 0, ax: 0, ay: 0, scale: { a: 0, v: 0, p: 0, real: 1 } };
        this.alpha = false;
        var container = $(containerElement);
        this.drawOffset.scale.a = -data.nodes.length / 2000;
        var viz = $("<canvas>");
        if (container[0].__graphViewer) {
            container[0].__graphViewer.destroy();
            container.empty();
        }
        container[0].__graphViewer = this;
        container.append(viz);
        $("<div>")
            .css("position", "absolute")
            .css("top", "0")
            .css("left", "0")
            .css("background-color", "rgba(255,255,255,0.8)")
            .css("margin", "1ex")
            .css("padding", "1ex")
            .css("float", "left")
            .css("border", "1px solid gray")
            .css("font-family", "mono")
            .html(["Graph Visualisation (<a href='https://github.com/phiresky/graphvis'>source</a>)",
            "Commands:",
            "(shift)+a - change attraction strength",
            "(shift)+g - change gravity strength",
            "(shift)+d - change target distance",
            "(shift)+s - change scale",
            "(shift)+r - change left/right offset",
            "ctrl+v    - load a graph file from the clipboard",
            "        t - toggle transparency"].join("<br>"))
            .appendTo(container);
        this.canvas = viz[0];
        this.canvas.width = this.w = width;
        this.canvas.height = this.h = height;
        this.ctx = this.canvas.getContext("2d");
        this.graph = data;
        container.mousedown(function (e) {
            _this.mouseInfo.downX = e.pageX;
            _this.mouseInfo.downY = e.pageY;
            _this.mouseInfo.down = true;
        });
        container.mousemove(function (e) {
            if (_this.mouseInfo.down) {
                _this.drawOffset.x = _this.drawOffset.ax + e.pageX - _this.mouseInfo.downX;
                _this.drawOffset.y = _this.drawOffset.ay + e.pageY - _this.mouseInfo.downY;
            }
        });
        container.mouseup(function (e) {
            _this.mouseInfo.down = false;
            _this.drawOffset.ax = _this.drawOffset.x;
            _this.drawOffset.ay = _this.drawOffset.y;
        });
        var wheelListener = function (e) {
            var delta = e.wheelDelta;
            if (e.detail)
                delta = -40 * e.detail;
            _this.drawOffset.scale.a = delta / 5000;
        };
        window.addEventListener('DOMMouseScroll', wheelListener);
        window.addEventListener('mousewheel', wheelListener);
        $(window).keydown(function (e) {
            if (e.keyCode == 17) {
                console.log("hi");
                window._copyArea = $("<textarea>").appendTo("body").focus();
            }
            var actions = {
                84: function (s) { return _this.alpha = !_this.alpha; },
                71: function (shift) { return _this.graph.gravity *= shift ? 2 : 0.5; },
                68: function (shift) { return _this.graph.targetDist *= shift ? 2 : 0.5; },
                65: function (shift) { return _this.graph.attraction *= shift ? 2 : 0.5; },
                83: function (shift) { return _this.drawOffset.scale.a += shift ? 0.02 : -0.02; },
                82: function (shift) { return _this.graph.leftright -= shift ? 0.1 : -0.1; } // r
            };
            var action = actions[e.keyCode];
            if (action)
                action(e.shiftKey);
            //console.log(e.keyCode);
            //console.log(e);
        });
        $(window).keyup(function (e) {
            if (e.keyCode == 17) {
                var textarea = window._copyArea;
                if (textarea !== undefined) {
                    var val = textarea.val();
                    textarea.remove();
                    if (val.length > 1) {
                        console.log("Loading from clipboard");
                        initGraphViewerFromString(val);
                    }
                }
            }
        });
        window.addEventListener("resize", function (e) {
            _this.canvas.width = _this.w = container.width();
            _this.canvas.height = _this.h = container.height();
        });
    }
    GraphViewer.prototype.beginDrawing = function () {
        this.drawing = true;
        this.draw();
    };
    GraphViewer.prototype.update = function () {
        var s = this.drawOffset.scale;
        s.v += s.a;
        s.v *= 0.9;
        s.p += s.v;
        s.a = 0;
        s.real = Math.exp(s.p);
        this.graph.update();
    };
    GraphViewer.prototype.draw = function () {
        this.update();
        var ctx = this.ctx;
        ctx.fillStyle = "#fff";
        ctx.fillRect(0, 0, this.w, this.h);
        ctx.fillStyle = "#000";
        ctx.strokeStyle = this.alpha ? "rgba(0,0,0,0.5)" : "#000";
        ctx.save();
        ctx.translate(this.drawOffset.x, this.drawOffset.y);
        ctx.translate(this.w / 2, this.h / 2);
        ctx.scale(this.drawOffset.scale.real, this.drawOffset.scale.real);
        ctx.translate(-this.w / 2, -this.h / 2);
        for (var i = 0; i < this.graph.nodes.length; i++) {
            var node = this.graph.nodes[i];
            for (var e = 0; e < node.edges.length; e++) {
                var node2 = this.graph.nodes[node.edges[e]];
                Util.drawLine(this.ctx, node.x, node.y, node2.x, node2.y);
            }
            Util.drawCircle(this.ctx, node.x, node.y, 6);
        }
        ctx.restore();
        if (this.drawing)
            requestAnimationFrame(this.draw.bind(this));
    };
    GraphViewer.prototype.destroy = function () {
        this.drawing = false;
        $(this.canvas).remove();
        $(window).off();
    };
    return GraphViewer;
})();
function graphFromStringInp(str, zeroBased, coordinateGenerator) {
    if (zeroBased === void 0) { zeroBased = false; }
    if (!coordinateGenerator)
        coordinateGenerator = function (n) { return ({ x: window.innerWidth / 2 + Math.random() * 500 - 250, y: window.innerHeight / 2 + Math.random() * 500 - 250 }); };
    var split = str.split("\n");
    var nodeCount = +split[0].split(" ")[0];
    var nodes = new Array(nodeCount);
    var i = 0;
    for (; i < split.length - 1; i++) {
        var lineSplit = [];
        if (split[i + 1].length > 0)
            lineSplit = split[i + 1].split(" ");
        var edges = [];
        lineSplit.forEach(function (s) { return s.length > 0 && edges.push(+s - (zeroBased ? 0 : 1)); });
        var coords = coordinateGenerator(i);
        nodes[i] = new GraphNode(edges, coords.x, coords.y);
    }
    while (i < nodeCount) {
        var coords = coordinateGenerator(i);
        nodes[i++] = new GraphNode([], coords.x, coords.y);
    }
    return new Graph(nodes);
}
var gv;
function initGraphViewerFromString(response) {
    var w = window.innerWidth;
    var h = window.innerHeight;
    var graph = graphFromStringInp(response);
    gv = new GraphViewer($("#viz")[0], w, h, graph);
    gv.beginDrawing();
}
function initGraphViewer(url) {
    $.get(url, initGraphViewerFromString);
}
//# sourceMappingURL=graphviz.js.map