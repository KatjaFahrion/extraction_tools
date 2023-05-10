# Extraction_tools

A collection of useful functions to extract spectra from MUSE cubes. Mostly used for extracting spectra of star clusters and subtracting the galaxy background. Assumes star clusters to be point-sources

## Prerequisites

Requires python 3.x with astropy and photutils (and numpy)


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
where cube is the MUSE datacube as a numpy array, x and y are the coordinates of the source. See the example jupyter notebook for more details and how to use additional parameters and other functions.