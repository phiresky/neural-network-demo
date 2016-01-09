# Kogsys-Demos

This repository contains various demonstrations for the lecture “Kognitive Systeme” at the [Interactive Systems Lab (ISL)](http://isl.anthropomatik.kit.edu/english/) at the Karlsruher Institute for Technology.

## Building

This project is written in [TypeScript](http://www.typescriptlang.org/), a statically typed superset of JavaScript.

It uses [React](https://facebook.github.io/react/) with [JSX](https://facebook.github.io/jsx/) for GUI state handling.

The compiled files are included in `*/bin/`

```bash
sudo npm -g install typescript bower tsd
(cd lib
	tsd install
	bower install
)
```

Then build via `make` or `make watch`

