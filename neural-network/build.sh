#!/bin/bash
touch bin/program.js
tsc&
while inotifywait -e close_write bin/program.js; do
	uglifyjs --source-map bin/program.min.js.map --in-source-map bin/program.js.map --source-map-url program.min.js.map --output bin/program.min.js -m -- bin/program.js
done
