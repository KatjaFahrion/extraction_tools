# Extraction_tools

A collection of useful functions to extract spectra from MUSE cubes. Mostly used for extracting spectra of star clusters and subtracting the galaxy background. Assumes star clusters to be point-sources

## Prerequisites

Requires python 3.x with astropy, photutils and PyAstronomy


## How to install

* Click on "Clone or download" and chose Download ZIP
* Unpack the ZIP archive somewhere and navigate into the folder in the command line
* Type in the comment line:
```
pip install -e .
```
This should install the package to your python path

## How to use

Import the package and use the functions:

```
import extraction_tools as tools

spectrum, background_spectrum = tools.extract_spectrum(cube, x, y)
```
where cube is the MUSE datacube as a numpy array, x and y are the coordinates of the source. See the jupyter notebook for example on how to use it in a workflow. Unfortunately without the data as MUSE cubes are too big to host here.

## Acknowledgements

This code makes use of Astropy (https://www.astropy.org), a community-developed core Python package and an ecosystem of tools and resources for astronomy, and Photutils (https://photutils.readthedocs.io/en/stable/index.html), an Astropy package for detection and photometry of astronomical sources.

