# MoSeq2-model 

[![Build Status](https://travis-ci.com/dattalab/moseq2-model.svg?token=gvoikVySDHEmvHT7Dbed&branch=master)](https://travis-ci.com/dattalab/moseq2-model) 

[![codecov](https://codecov.io/gh/dattalab/moseq2_model/branch/master/graph/badge.svg?token=q9xxVhps5o)](https://codecov.io/gh/dattalab/moseq2_model)

Welcome to moseq2, the latest version of a software package for mouse tracking in depth videos first developed by Alex Wiltschko in the Datta Lab at Harvard Medical School.

Latest version is `0.5.0`

## Features
Below are the commands/functionality that moseq2-model currently affords. 
They are accessible via CLI or Jupyter Notebook in [moseq2-app](https://github.com/dattalab/moseq2-app/tree/release).
```bash
Usage: moseq2-model [OPTIONS] COMMAND [ARGS]...

Options:
  --version  Show the version and exit.  [default: False]
  --help     Show this message and exit.  [default: False]

Commands:
  count-frames  Counts number of frames in given h5 file (pca_scores)
  kappa-scan    Batch fit multiple models scanning over different syllable...
  learn-model   Trains ARHMM on PCA Scores with given training parameters
```

### CLI Exclusive Functions
```bash
  count-frames  Counts number of frames in given h5 file (pca_scores)
  kappa-scan    Batch fit multiple models scanning over different syllable...
```

### GUI Functionality Note
The `kappa-scan` functionality is merged into the `learn_model()` function. 
 To use it, simple set `kappa='scan'`, and for additional control, adjust the `scan_scale`, 
 `min_kappa`, `max_kappa`, and `out_script` input parameters. 
 Options can be found in the documentation or the [jupyter notebook]()

Run any command with the `--help` flag to display all available options and their descriptions.

## Documentation

MoSeq2 uses `sphinx` to generate the documentation in HTML and PDF forms. To install `sphinx`, follow the commands below:
```.bash
pip install sphinx==3.0.3 sphinx_click==2.5.0
pip install sphinx-rtd-theme
pip install rst2pdf
``` 

All documentation regarding moseq2-extract can be found in the `Documentation.pdf` file in the root directory,
an HTML ReadTheDocs page can be generated via running the `make html` in the `docs/` directory.

To generate a PDF version of the documentation, simply run `make pdf` in the `docs/` directory.

## Prerequisites

To use this package, you must have already generated a `pca_scores.h5` file and an index file `moseq2-index.yaml` containing all of your
session metadata (specifically data groupings).
 - The index file is generated when aggregating the results in [moseq2-extract](https://github.com/dattalab/moseq2-extract/tree/release) 
 - The pca_scores are generated via [moseq2-pca](https://github.com/dattalab/moseq2-pca/tree/release).

## Contributing

If you would like to contribute, fork the repository and issue a pull request.  
