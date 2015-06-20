declare module convnetjs {
	interface Layer {
		type:string,
		num_neurons?:int,
		activation?: string
		out_sx?:int,
		out_sy?:int,
		out_depth?:int,
		num_classes?:int
	}
	class Net {
		makeLayers(layers: Layer[]):void;
		forward(inp: Vol): Vol;
		fromJSON(inp:any):void;
		toJSON():any;
	}
	interface Options {
		learning_rate: double,
		momentum: double,
		batch_size: int,
		l1_decay?: double,
		l2_decay: double
	}
	class Trainer {
		constructor(net: Net, options: Options);
		train(vol: Vol, label: int) : any;
	}
	class Vol {
		constructor(sx: int, sy: int, depth: int, initValue?:double);
		sx: int; sy: int; depth: int;
		w: Float64Array;
		dw: Float64Array;
	}
}
