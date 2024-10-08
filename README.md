# IRAM 30-meter EMIR observations informativity

[![Documentation status](https://readthedocs.org/projects/iram-30m-emir-obs-info/badge/?version=latest)](https://iram-30m-emir-obs-info.readthedocs.io/en/latest/?badge=latest)
![test coverage badge](./coverage.svg)

This package implements tools to quantitatively estimate the usefulness of spectral line observations for estimating physical conditions.
It provides a tool for simply reproducing observations made at IRAM 30-meter millimeter-wave telescope coupled with the EMIR receiver. Other instruments can also be simulated.

Line intensity predictions are made using a neural network emulation of the Meudon PDR code.
This emulator enables a thousand predictions to be made in around 10 ms on a laptop, with an average error of less than 5%.

## Installation

To install `infobs`:

Step 1: Create a virtual environment and activate it:

```shell
python -m venv .venv
source .venv/bin/activate
```

Step 2 : install with pip:

```shell
pip install -e .
```

**Note 1:** to deactivate the virtual env :

```shell
deactivate
```

**Note 2:**To delete the virtual environment:

```shell
rm -r .venv
```

## Get started

To get started, check out the jupyter notebooks presented in the `examples` folder.

## Tests

To test, run:

```shell
python -m pytest && coverage-badge -o coverage.svg -f
```

## Documentation

```bash
cd docs
sphinx-apidoc -o . ../infobs
make html
```

Outputs are in `docs/_build/html`.


## Associated packages

[**A&A paper repository**](https://github.com/einigl/informative-obs-paper): Reproduce the results in Einig et al. (2024)

[**InfoVar**](<https://github.com/einigl/infovar>): Estimating informativity of features.

[**Neural network-based model approximation**](<https://github.com/einigl/ism-model-nn-approximation>): handle the creation and the training of neural networks to approximate interstellar medium numerical models.

## References

[1] Einig, L, Palud, P. & Roueff, A. & Pety, J. & Bron, E. & Le Petit, F. & Gerin, M. & Chanussot, J. & Chainais, P. & Thouvenin, P.-A. & Languignon, D. & Bešlić, I. & Coudé, S. & Mazurek, H. & Orkisz, J. H. & G. Santa-Maria, M. & Ségal, L. & Zakardjian, A. & Bardeau, S. & Demyk, K. & de Souza Magalhães, V. & Javier R. Goicoechea & Gratier, P. & V. Guzmán, V. & Hughes, A. & Levrier, F. & Le Bourlot, J. & Darek C. Lis & Liszt, H. S. & Peretto, N. & Roueff, E & Sievers, A. (2024).
**Quantifying the informativity of emission lines to infer physical conditions in giant molecular clouds. I. Application to model predictions.** *Astronomy & Astrophysics.*
[10.1051/0004-6361/202451588](https://doi.org/10.1051/0004-6361/202451588).

[2] Palud, P. & Einig, L. & Le Petit, F. & Bron, E. & Chainais, P. & Chanussot, J. & Pety, J. & Thouvenin, P.-A. & Languignon, D. & Beslić, I. & G. Santa-Maria, M. & Orkisz, J.H. & Ségal, L. & Zakardjian, A. & Bardeau, S. & Gerin, M. & Goicoechea, J.R. & Gratier, P. & Guzman, V. (2023).
**Neural network-based emulation of interstellar medium models.**
*Astronomy & Astrophysics.*
[10.1051/0004-6361/202347074](https://doi.org/10.1051/0004-6361/202347074).
