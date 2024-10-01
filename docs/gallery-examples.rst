Gallery of examples
===================

This gallery contains several application examples for the ``infobs`` package to illustrate diverse features.

**Meudon PDR code fast emulator:**

- ``meudonpdr.ipynb``: illustrates the features of the ``MeudonPDR`` class, which predicts integrated line intensities from cloud visual extinction, UV radiative field intensity, thermal pressure, and observation angle.

**Instruments and IRAM 30m EMIR receiver:**

- ``instruments.ipynb``: illustrates how to use the ``IRAM30mEMIR`` instrument, and how to create your own receivers. Instruments can process Meudon PDR code emulator predictions to mimic real-world observation.
- ``emir-bands.ipynb``: how to easily display available lines for atmospheric bands.

**Simulator of observations:**

- ``simulator.ipynb``: illustrates how to use the ``Simulator`` class, which combines the Meudon PDR code emulator, an instrument, and a prior on the distribution of the astrophysical parameters.
- ``sampling.ipynb``: more detail on how to parametrize astrophysical parameter distributions.

**Tools to compute and display informativity results:**

- ``informativity.ipynb`` : illustrates the features of the ``Infobs`` class, which provides tools to quantify statistical dependence between simulated observations and astrophysical parameters for a given environment, and to easily display the results.

.. toctree::
   :maxdepth: 1
   :caption: Gallery

   meudonpdr
   instruments
   emir-bands
   sampling
   simulator
   informativity
