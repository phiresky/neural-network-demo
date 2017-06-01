var webpack = require("webpack");
var path = require('path');
var CopyWebpackPlugin = require('copy-webpack-plugin');

const production = process.env.NODE_ENV === "production";

module.exports = {
	entry: ['./src/main.tsx'],
	output: {
		filename: 'bin/bundle.js'
	},
	resolve: {
		extensions: ['.webpack.js', '.web.js', '.ts', '.tsx', '.js'],
	},
	plugins: [
		...(production ? [
			new webpack.DefinePlugin({
				'process.env': {
					NODE_ENV: JSON.stringify('production')
				}
			}),
			new webpack.optimize.UglifyJsPlugin(),
		] : []),
		new webpack.ProvidePlugin({
			jQuery: 'jquery',
			$: 'jquery',
			jquery: 'jquery'
		}),
		new CopyWebpackPlugin(
			[
				..."index.html icon.png".split(" ").map(s => ({ from: "src/" + s, to: "bin/" + s })),
				..."screenshot.png screenshot-perceptron.png".split(" ").map(s => ({ from: s, to: "bin/" + s })),
			]
		)
	],
	devtool: 'source-map',
	module: {
		loaders: [
			{ test: /\.tsx?$/, loader: 'ts-loader' },
			{ test: /\.css$/, loader: "style-loader!css-loader?-url" },
		],
		//noParse: [path.join(__dirname, 'node_modules/handsontable/dist/handsontable.full.js')]
	}
}