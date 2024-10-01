Welcome to Infobs' documentation
================================

The ``infobs`` Python package provides tools to efficiently study the informativity of variables on data of interest.


Context
=======

TODO


Installation
============

*(optional)* Create a virtual environment and activate it:

```shell
python -m venv .venv
source .venv/bin/activate
```

**Note 1:** to deactivate the virtual env :

```shell
deactivate
```

**Note 2:** To delete the virtual environment:

```shell
rm -r .venv
```


From local package
------------------

To get the source code:

```shell
git clone git@github.com:einigl/iram-30m-emir-obs-info.git
```

To install `infobs`:

```shell
pip install -e .
```

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   modules

.. toctree::
   :maxdepth: 2

   gallery-examples


Associated packages
===================

[**A&A paper repository**](https://github.com/einigl/informative-obs-paper): Reproduce the results in Einig et al. (2024)

[**InfoVar**](<https://github.com/einigl/infovar>): Estimating informativity of features.

[**Neural network-based model approximation**](<https://github.com/einigl/ism-model-nn-approximation>): handle the creation and the training of neural networks to approximate interstellar medium numerical models.


References
==========

[1] Einig, L, Palud, P. & Roueff, A. & Pety, J. & Bron, E. & Le Petit, F. & Gerin, M. & Chanussot, J. & Chainais, P. & Thouvenin, P.-A. & Languignon, D. & Bešlić, I. & Coudé, S. & Mazurek, H. & Orkisz, J. H. & G. Santa-Maria, M. & Ségal, L. & Zakardjian, A. & Bardeau, S. & Demyk, K. & de Souza Magalhães, V. & Javier R. Goicoechea & Gratier, P. & V. Guzmán, V. & Hughes, A. & Levrier, F. & Le Bourlot, J. & Darek C. Lis & Liszt, H. S. & Peretto, N. & Roueff, E & Sievers, A. (2024).
**Quantifying the informativity of emission lines to infer physical conditions in giant molecular clouds. I. Application to model predictions.** *Astronomy & Astrophysics.*
[10.1051/0004-6361/202451588](https://doi.org/10.1051/0004-6361/202451588).

[2] Palud, P. & Einig, L. & Le Petit, F. & Bron, E. & Chainais, P. & Chanussot, J. & Pety, J. & Thouvenin, P.-A. & Languignon, D. & Beslić, I. & G. Santa-Maria, M. & Orkisz, J.H. & Ségal, L. & Zakardjian, A. & Bardeau, S. & Gerin, M. & Goicoechea, J.R. & Gratier, P. & Guzman, V. (2023).
**Neural network-based emulation of interstellar medium models.**
*Astronomy & Astrophysics.*
[10.1051/0004-6361/202347074](https://doi.org/10.1051/0004-6361/202347074).
