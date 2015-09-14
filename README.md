# Kogsys-Demos

This repository contains various demonstrations for the lecture “Kognitive Systeme” at the [Interactive Systems Lab (ISL)](http://isl.anthropomatik.kit.edu/english/) at the Karlsruher Institute for Technology.

## Building

The compiled files are included in `*/bin/`

```bash
sudo npm -g install typescript bower tsd
(cd lib
	tsd install
	bower install
)
```

Then build via `make` or `make watch`

