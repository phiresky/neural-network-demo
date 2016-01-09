# Kogsys-Demos

This repository contains various demonstrations for the lecture “Kognitive Systeme” at the [Interactive Systems Lab (ISL)](http://isl.anthropomatik.kit.edu/english/) at the [Karlsruhe Institute for Technology](https://kit.edu).

Currently there is only the Neural Network demonstration and the Perceptron demonstration, both contained as a single webpage in [neural-network/](neural-network/).

## Development

This project is written in [TypeScript](http://www.typescriptlang.org/), a statically typed superset of JavaScript.

I recommend [Visual Studio Code](https://code.visualstudio.com/) (crossplatform and opensource) for development.
An alternative is [Atom](https://atom.io/) + [atom-typescript](https://atom.io/packages/atom-typescript).


[React](https://facebook.github.io/react/) is used with [JSX](https://facebook.github.io/jsx/) for GUI state handling.

The compiled files are included in the `gh-pages` branch


**Setup**

```bash
git worktree add bin gh-pages
sudo npm -g install typescript bower tsd
(cd lib
	tsd install
	bower install
)
```

Then build via `make`.

Use 

```bash
python3 -m http.server &
tsc --watch &
```

for automatic compiling and deploying to <http://localhost:8000/bin>

Call enableDev() once from the browser console to enable loading of unminified libraries.

Use `Simulation.instance` to access the main object and all it's children.
