"""
Tools to extract spectra of sources from a MUSE data cube
"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from photutils import DAOStarFinder, CircularAperture, CircularAnnulus
from PyAstronomy import pyasl


def read_MUSE(file):
    """
    Function to read the MUSE cube of the galaxy
    Returns the data and the wavelength array
    """
    hdulist = fits.open(file, menmap=True)
    data = hdulist[1].data
    hdr = hdulist[1].header
    s = np.shape(data)
    wave = hdr['CRVAL3']+(np.arange(s[0]))*hdr['CD3_3']
    return data, wave


def run_dao_starfind(img, fwhm, threshold, sharplo=0.0):
    '''
    Wrapper around the DAOStarFinder routine
    '''
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold, sharplo=sharplo)  # , )
    sources = daofind(img)
    x_dao = sources['xcentroid']
    y_dao = sources['ycentroid']
    return np.round(x_dao, 2), np.round(y_dao, 2)  # can only use the pixel coordinates


def SourceFinder(residual, psf=3, threshold=3, sharplo=0.5, plot=False):
    '''
    Performs the source finding on a single residual
    '''
    x_dao, y_dao = run_dao_starfind(residual, psf, threshold, sharplo=sharplo)

    if plot:
        vmin = -40
        vmax = 40
        fig, ax0 = plt.subplots(1, 1)
        ax0.set_xticks([])
        ax0.set_yticks([])
        s = np.shape(residual)
        xaxis = (np.arange(s[1]))
        yaxis = (np.arange(s[0]))
        ax0.imshow(residual, cmap="Greys", interpolation='none',  extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]], origin='lower',
                   vmin=vmin, vmax=vmax)
        ax0.set_xlim(0, s[1])
        ax0.set_ylim(0, s[0])
        ax0.scatter(x_dao, y_dao, edgecolor='blue', facecolor='none')
        plt.tight_layout()
    coords = []
    for i in range(len(x_dao)):
        coords.append([x_dao[i], y_dao[i]])
    return coords


######### Routines for the spectra extraction ####################################

def aperture_annu_spec(cube, x, y, r1=10, r2=30):
    """
    gets the spectrum in an annulus of inner radius r1 and outer radius r2
    """
    # loop over wavelength
    spec = []
    cut_cube = cube[:, int(y)-r2-1:int(y)+r2+1, int(x)-r2-1:int(x)+r2+1]
    apertures = CircularAnnulus((x, y), r1, r2)
    masks = apertures.to_mask(method='center')
    mask = masks[0]
    masked_img = mask.to_image(np.shape(cube[100, :, :]))
    masked_img_c = masked_img[int(y)-r2-1:int(y)+r2+1, int(x)-r2-1:int(x)+r2+1]
    for i in range(len(cut_cube[:, 0, 0])):
        cut_out_i = cut_cube[i, :, :]*masked_img_c
        cut_out_i[cut_out_i == 0] = np.nan
        spec.append(np.nanmedian(cut_out_i))
    return np.array(spec)


def aperture_spec(cube, x, y, r=10, psf=True, sigma_psf=3.5/2.35):
    """
    gets the spectrum in a circular aperture of radius r
    """
    # loop over wavelength
    cut_cube = cube[:, int(y)-r:int(y)+r+1, int(x)-int(r):int(x)+r+1]
    spec = []
    apertures = CircularAperture((x, y), r)
    masks = apertures.to_mask(method='center')
    mask = masks[0]
    masked_img = mask.to_image(np.shape(cube[100, :, :]))
    masked_img_c = masked_img[int(y)-r:int(y)+r+1, int(x)-int(r):int(x)+r+1]

    if psf:
        # use the psf weighted extraction, but first create the psf weight mask
        psf_mask = gauss_2d(masked_img_c, r, sigma_psf)
    else:
        psf_mask = np.ones_like(masked_img_c)
    for i in range(len(cut_cube[:, 0, 0])):
        cut_out_i = cut_cube[i, :, :]*masked_img_c*psf_mask
        cut_out_i[cut_out_i == 0] = np.nan
        if psf:
            spec.append(np.nansum(cut_out_i))
        else:
            spec.append(np.nanmedian(cut_out_i))
    return np.array(spec)


def gaussian(x, mu, sig):
    '''
    normalized gauss function
    '''
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) * 1/(sig * np.sqrt(2 * np.pi))


def gauss_2d(img, r, sigma):
    """
    For the weighted PSF mask
    """
    gauss_img = np.zeros_like(img)
    xs = np.arange(len(img[0, :]))
    gauss_x = gaussian(xs, r, sigma)
    ys = np.arange(len(img[0, :]))
    gauss_y = gaussian(ys, r, sigma)
    for x in xs:
        for y in ys:
            gauss_img[y, x] = gauss_x[x] * gauss_y[y]
    return gauss_img


def calc_SNR(wave, spec, der=False):
    '''
    Calculate the signal-to-noise ratio of a spectrum
    '''
    if not der:
        mask = (wave > 6020) & (wave < 6500)  # continuum region of the spectrum
        snrEsti = pyasl.estimateSNR(wave[mask], spec[mask], 20, deg=3, controlPlot=False)
        return snrEsti["SNR-Estimate"]
    else:
        return der_snr(spec)


def der_snr(flux):
    """
    REFERENCES  * ST-ECF Newsletter, Issue #42:
                www.spacetelescope.org/about/further_information/newsletters/html/newsletter_42.html
                * Software:
                www.stecf.org/software/ASTROsoft/DER_SNR/
    AUTHOR      Felix Stoehr, ST-ECF
                24.05.2007, fst, initial import
                01.01.2007, fst, added more help text
                28.04.2010, fst, return value is a float now instead of a numpy.float64
    """
    flux = np.array(flux[np.isfinite(flux)])
    # Values that are exactly zero (padded) or NaN are skipped
    flux = flux[(flux != 0.0) & np.isfinite(flux)]
    n = len(flux)
    # For spectra shorter than this, no value can be returned
    if n > 4:
        signal = np.median(flux)
        noise = 0.6052697 * np.median(np.abs(2.0 * flux[2:n-2] - flux[0:n-4] - flux[4:n]))
        return float(signal / noise)
    else:
        return 0.0
