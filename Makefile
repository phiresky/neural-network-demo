all: bin bin/index.html bin/icon.png bin/libs.css bin/program.js bin/libs.js

bin/%: src/%
	cp $< $@

bin/program.js:
	tsc

bin/libs.css: lib/bower_components/bootstrap/dist/css/bootstrap.min.css \
		lib/bower_components/vis/dist/vis.min.css \
		lib/bower_components/handsontable/dist/handsontable.full.min.css
	paste -d '\n' -s $^ > bin/libs.css

bin/libs.js: lib/bower_components/jquery/dist/jquery.min.js lib/bower_components/bootstrap/dist/js/bootstrap.min.js \
		lib/bower_components/vis/dist/vis.min.js \
		lib/bower_components/handsontable/dist/handsontable.full.min.js \
		lib/bower_components/lz-string/libs/lz-string.min.js lib/bower_components/highstock/highstock.js \
		lib/bower_components/react/react.min.js \
		lib/bower_components/react/react-dom.min.js
	paste -d '\n' -s $^ > bin/libs.js

bin:
	[ -f bin/.git ] || (echo "bin not setup. see readme" && exit 1)

gh-pages: bin
	cd bin; git add -A; git commit -m'update binaries'; git push

.PHONY: gh-pages

