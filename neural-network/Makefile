all: bin/styles.min.css bin/program.min.js bin/libs.min.js

bin/styles.min.css: lib/bootstrap.min.css lib/bootstrap-slider.min.css lib/vis.min.css lib/handsontable.full.min.css
	cat $^ > bin/styles.min.css

bin/program.js:
	tsc

bin/libs.min.js: lib/jquery.min.js lib/bootstrap.min.js lib/bootstrap-slider.min.js lib/vis.min.js lib/handsontable.full.min.js lib/lz-string.min.js lib/highstock.min.js
	cat $^ > bin/libs.min.js

watch:
	tsc --watch
