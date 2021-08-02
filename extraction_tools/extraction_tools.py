"""
Author: Katja Fahrion
Tools to extract spectra of sources from a MUSE data cube
"""
# imports
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from photutils import DAOStarFinder, CircularAperture, CircularAnnulus
from PyAstronomy import pyasl
import os
import glob


def read_MUSE(filename):
    """
    Reads a MUSE cube, returns the data and the wavelength array

    Args:
        file (str): filename, including the path

    Returns:
        array: data array of the cube
        array: wavelength array

    """
    hdulist = fits.open(filename, menmap=True)
    data = hdulist[1].data
    hdr = hdulist[1].header
    s = np.shape(data)
    wave = hdr['CRVAL3']+(np.arange(s[0]))*hdr['CD3_3']
    return data, wave


def write_fits(cube, filename, header, direct='./'):
    '''
    Write fits file to direct with filename and header.
    The data will be in extension 0
    '''
    output_filename = direct + filename
    hdu_prime = fits.PrimaryHDU(header=header)
    hdu_prime.data = cube
    hdul = fits.HDUList([hdu_prime])
    if os.path.isfile(output_filename):
        os.remove(output_filename)
    print('Writing {0}'.format(output_filename))
    hdul.writeto(output_filename)


def run_dao_starfind(img, fwhm, threshold, sharplo=0.0, round=2):
    '''
    Wrapper around the DAOStarFinder routine
    '''
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold, sharplo=sharplo)  # , )
    sources = daofind(img)
    x_dao = sources['xcentroid']
    y_dao = sources['ycentroid']
    return np.round(x_dao, round), np.round(y_dao, round)  # can only use the pixel coordinates


def SourceFinder(residual, fwhm=3, threshold=3, sharplo=0.5, plot=False, quiet=True):
    """
    Performs source finding on a residual (or an image)

    Takes a resiudal image and uses photutils DAOStarFinder to look for
    sources

    Args:
        residual (array): image residual where to look for sources
        fwhm (float): FWHM for the souce finding, default is 3 - 5
        threshold (float): Threshold for the source finding, default is 3 - 10
        sharplo (float): lower limit for the sharpness (default is 0.5)
        plot (bool): if true, the residual and the found sources are plotted
        quiet (bool): if true, prints the number of sources found

    Returns:
        array: x_dao (x coordinates of found sources)
        array: y_dao (y coordinates of found sources)

    Raises:
        Exception: description

    """
    psf = fwhm/2.355
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
    if not quiet:
        print('{0} sources found!'.format(len(x_dao)))
    return x_dao, y_dao


def plot_spectrum(wave, spec, lam_range=None, save=False, output_direct='./', filename='spec_plot.png'):
    """
    Function to plot a spectrum

    Args:
        wave (array): wavelength array
        spec (array): spectrum
        lam_range (tuple): wavelength range to show. Default is None (full)
        save (bool): saves the plot if set to true (default is false)
        output_direct (str): path of where to save the plot
        filename (str): filename of the saved file

    """
    fig, ax = plt.subplots(figsize=[10, 3])
    ax.step(wave, spec)
    ax.set_xlabel(r'$\lambda$ [$\rm{\AA}$]')
    ax.set_ylabel('Flux')
    if lam_range is not None:
        ax.set_xlim(lam_range[0], lam_range[1])
    else:
        ax.set_xlim(wave[0], wave[-1])
    if save:
        filename = output_direct + filename
        plt.savefig(filename, dpi=300)


######### Routines for the spectra extraction ####################################
def extract_spectrum(cube, x, y, bg=True, r=10, r1=10, r2=30, psf=True, fwhm=3.5, var=False, eps=None, PA=None):
    """
    The standard routine for extracting a spectrum from a MUSE cube with (or without background spectrum)
    if psf is false, a circular aperture will be used. Otherwise, a Gaussian weighting will be applied
    with fwhm = fwhm.
    """
    spec = aperture_spec(cube, x, y, r=r, psf=psf, sigma_psf=fwhm/2.355, var=var)
    if bg:
        if eps is None:
            spec_bg = aperture_annu_spec(cube, x, y, r1=r1, r2=r2, var=var)
        else:
            spec_bg = aperture_ellip_annu_spec(cube, x, y, r1=r1, r2=r2, PA=PA, eps=eps)
        return spec, spec_bg
    else:
        return spec


def extract_spectrum_no_var(cube, x, y, bg=True, r=10, r1=10, r2=30, psf=True, fwhm=3.5):
    """
    The standard routine for extracting a spectrum from a MUSE cube with (or without background spectrum)
    if psf is false, a circular aperture will be used. Otherwise, a Gaussian weighting will be applied
    with fwhm = fwhm.
    """
    spec = aperture_spec_no_var(cube, x, y, r=r, psf=psf, sigma_psf=fwhm/2.355)
    if bg:
        spec_bg = aperture_annu_spec_no_var(cube, x, y, r1=r1, r2=r2)
        return spec, spec_bg
    else:
        return spec


def square_extraction(data, x1, x2, y1, y2):
    """
    Cut a full MUSE cube down to a little one between (x1, y1) and (x2, y2),
    get a combined spectrum
    """
    cut_cube = data[:, y1:y2, x1:x2]
    s = np.shape(cut_cube)
    flat_cube = np.reshape(cut_cube, [s[0], s[1]*s[2]])
    spec = np.nanmedian(flat_cube, axis=1)
    return spec

# def square_extraction(cube,)


def aperture_ellip_annu_spec(cube, x, y, eps=0, PA=0, r1=10, r2=30, plot=False):
    """
    Annulus aperture spectra extraction

    Extract the spectrum of a source at X, Y from the cube.
    R1 and R2 give the inner and outer radius of the annulus

    Args:
        cube (ndarray): MUSE data cube
        x (int): x-pixel centroid of the source
        y (int): y-pixel centroid of the source
        r1 (int): inner radius of the annulus (default is 10 pix)
        r2 (int): outer radius of the annulus (default is 30 pix)

    Returns:
        array: spectrum of the source
    """
    # loop over wavelength
    spec = []
    cut_cube = cube[:, int(y)-r2-1:int(y)+r2+1, int(x)-r2-1:int(x)+r2+1]
    s = np.shape(cut_cube)
    ellips = dist_ellipse(s[1]/2, s[2]/2, PA=PA, eps=eps, s=(s[1], s[2]))
    mask_img = np.zeros_like(ellips)
    mask_img[(ellips >= r1) & (ellips < r2)] = 1
    #fig, ax = plt.subplots()
    #ax.imshow(cut_cube[10, :, :]*masked_img_c)
    for i in range(len(cut_cube[:, 0, 0])):
        if i == 1000 and plot:
            fig, ax = plt.subplots()
            ax.imshow(cut_cube[i, :, :] * mask_img, origin='lower')
        cut_out_i = cut_cube[i, :, :] * mask_img
        cut_out_i[cut_out_i == 0] = np.nan
        spec.append(np.nanmedian(cut_out_i))
    return np.array(spec)


def aperture_annu_spec(cube, x, y, r1=10, r2=30, var=False):
    """
    gets the spectrum in an annulus of inner radius r1 and outer radius r2
    var = True for extraction from noise cube
    """
    # loop over wavelength
    spec = []

    cut_cube = cube[:, int(np.round(y))-r2-1:int(np.round(y))+r2+1,
                    int(np.round(x))-r2-1:int(np.round(x))+r2+1]
    apertures = CircularAnnulus((x, y), r1, r2)
    masks = apertures.to_mask(method='center')
    #mask = masks[0]
    mask = masks
    masked_img = mask.to_image(np.shape(cube[100, :, :]))
    masked_img_c = masked_img[int(np.round(y))-r2-1:int(np.round(y)) +
                              r2+1, int(np.round(x))-r2-1:int(np.round(x))+r2+1]
    for i in range(len(cut_cube[:, 0, 0])):
        cut_out_i = cut_cube[i, :, :]*masked_img_c
        cut_out_i[cut_out_i == 0] = np.nan
        spec.append(np.nanmedian(cut_out_i))
    n = np.count_nonzero(~np.isnan(cut_out_i))
    if not var:
        return np.array(spec)
    else:
        return np.array(spec)/n


def aperture_spec(cube, x, y, r=10,  var=False, psf=True, sigma_psf=3.5/2.35, model_psf=None):
    """
    Aperture aperture spectra extraction

    Extract the spectrum of a source at X, Y from the cube.
    R gives the radius of the extraction.
    If PSF is true, a Gaussian PSF with sigma_psf is used for
    a weighted extraction

    Args:
        cube (ndarray): MUSE data cube
        x (int): x-pixel centroid of the source
        y (int): y-pixel centroid of the source
        r (int): Radius of the circular aperture
        psf (bool): If true, use psf weighted extraction
        sigma_psf (float): sigma of the used PSF (in pixel)
        model_psf: a model psf cube. Will only work with r=10

    Returns:
        array: spectrum of the source
    """
    # loop over wavelength
    cut_cube = cube[:, int(np.round(y))-r:int(np.round(y))+r+1, int(np.round(x))-int(r):int(np.round(x)) +
                    r+1]  # Cut the cube near the source
    spec = []
    # uses photutils CircularAperture to create a aperture
    apertures = tools.CircularAperture((x, y), r)
    masks = apertures.to_mask(method='center')
    mask = masks
    masked_img = mask.to_image(np.shape(cube[100, :, :]))  # photutils mask
    masked_img_c = masked_img[int(np.round(y))-r:int(np.round(y))+r+1,
                              int(np.round(x))-int(r):int(np.round(x))+r+1]
    if psf:
        # use the psf weighted extraction, but first create the psf weight mask
        psf_mask = tools.gauss_2d(masked_img_c, r, sigma_psf)
    else:
        psf_mask = np.ones_like(masked_img_c)
    for i in range(len(cut_cube[:, 0, 0])):
        if not var:
            if model_psf is not None:
                # read model psf cube for the galaxy
                # check if the size fits
                shape = np.shape(model_psf[0, :, :])
                if shape != np.shape(masked_img_c):
                    print('shape mismatch!!')
                psf_img = model_psf[i, :, :]
            else:
                psf_img = psf_mask
            cut_out_i = cut_cube[i, :, :]*masked_img_c * psf_img

            cut_out_i[cut_out_i == 0] = np.nan
            if i == 1000:
                fig, ax = plt.subplots(ncols=3, figsize=[13, 4])
                ax[0].imshow(cut_cube[i, :, :]*masked_img_c)
                ax[1].imshow(psf_img)
                ax[2].imshow(psf_mask)
            if psf:
                spec.append(np.nansum(cut_out_i))
            else:
                spec.append(np.nanmedian(cut_out_i))
        else:
            if psf:
                # this is because the weights get quard. for variance
                cut_out_i = cut_cube[i, :, :]*masked_img_c*psf_mask**2
                cut_out_i[cut_out_i == 0] = np.nan
                spec.append(np.nansum(cut_out_i))
            else:
                cut_out_i = cut_cube[i, :, :]*masked_img_c
                cut_out_i[cut_out_i == 0] = np.nan
                spec.append(np.nanmedian(cut_out_i))
    n = np.count_nonzero(~np.isnan(cut_out_i))  # number of valid pixels used in the extraction
    if psf:
        n = 1  # psf variance does not need the extra factor n
    if not var:
        return np.array(spec)
    else:
        return np.array(spec)/n


def aperture_annu_spec_no_var(cube, x, y, r1=10, r2=30):
    """
    Annulus aperture spectra extraction

    Extract the spectrum of a source at X, Y from the cube.
    R1 and R2 give the inner and outer radius of the annulus

    Args:
        cube (ndarray): MUSE data cube
        x (int): x-pixel centroid of the source
        y (int): y-pixel centroid of the source
        r1 (int): inner radius of the annulus (default is 10 pix)
        r2 (int): outer radius of the annulus (default is 30 pix)

    Returns:
        array: spectrum of the source
    """
    # loop over wavelength
    spec = []

    cut_cube = cube[:, int(np.round(y))-r2-1:int(np.round(y))+r2+1,
                    int(np.round(x))-r2-1:int(np.round(x))+r2+1]
    apertures = CircularAnnulus((x, y), r1, r2)
    masks = apertures.to_mask(method='center')
    #mask = masks[0]
    mask = masks
    masked_img = mask.to_image(np.shape(cube[100, :, :]))
    masked_img_c = masked_img[int(np.round(y))-r2-1:int(np.round(y)) +
                              r2+1, int(np.round(x))-r2-1:int(np.round(x))+r2+1]
    #fig, ax = plt.subplots()
    #ax.imshow(cut_cube[10, :, :]*masked_img_c)
    for i in range(len(cut_cube[:, 0, 0])):
        cut_out_i = cut_cube[i, :, :]*masked_img_c
        cut_out_i[cut_out_i == 0] = np.nan
        spec.append(np.nanmedian(cut_out_i))
    return np.array(spec)


def dist_circle(xc, yc, s):
    """
    Returns an array in which the value of each element is its distance from
    a specified center. Useful for masking inside a circular aperture.

    The (xc, yc) coordinates are the ones one can read on the figure axes
    e.g. when plotting the result of my find_galaxy() procedure.

    FROM MGEFIT
    """
    x, y = np.ogrid[:s[0], :s[1]] - np.array([yc, xc])  # note yc before xc
    rad = np.sqrt(x**2 + y**2)
    return rad


def aperture_spec_no_var(cube, x, y, r=10, psf=True, sigma_psf=3.5/2.35):
    """
    Aperture aperture spectra extraction

    Extract the spectrum of a source at X, Y from the cube.
    R gives the radius of the extraction.
    If PSF is true, a Gaussian PSF with sigma_psf is used for
    a weighted extraction

    Args:
        cube (ndarray): MUSE data cube
        x (int): x-pixel centroid of the source
        y (int): y-pixel centroid of the source
        r (int): Radius of the circular aperture
        psf (bool): If true, use psf weighted extraction
        sigma_psf (float): sigma of the used PSF (in pixel)

    Returns:
        array: spectrum of the source
    """
    cut_cube = cube[:, int(np.round(y))-r:int(np.round(y))+r+1, int(np.round(x))-int(r):int(np.round(x)) +
                    r+1]  # Cut the cube near the source
    spec = []
    apertures = CircularAperture((x, y), r)  # uses photutils CircularAperture to create a aperture
    masks = apertures.to_mask(method='center')
    mask = masks
    masked_img = mask.to_image(np.shape(cube[100, :, :]))  # photutils mask
    masked_img_c = masked_img[int(np.round(y))-r:int(np.round(y))+r+1,
                              int(np.round(x))-int(r):int(np.round(x))+r+1]  # needed to get the size right
    if psf:
        # use the psf weighted extraction, but first create the psf weight mask
        psf_mask = gauss_2d(masked_img_c, r, sigma_psf)
    else:
        psf_mask = np.ones_like(masked_img_c)
    for i in range(len(cut_cube[:, 0, 0])):
        cut_out_i = cut_cube[i, :, :]*masked_img_c*psf_mask  # apply the mask and the psf weighting
        cut_out_i[cut_out_i == 0] = np.nan
        if psf:
            spec.append(np.nansum(cut_out_i))  # psf is normalized, so no need for median
        else:
            spec.append(np.nanmedian(cut_out_i))  # otherwise, use the median of the used pixels
    return np.array(spec)


def gaussian(x, mu, sig):
    '''
    Normalized Gaussian function. Used to create the 2D gaussian for the PSF extraction
    '''
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) * 1/(sig * np.sqrt(2 * np.pi))


def gauss_2d(img, r, sigma):
    """
    Create 2D gaussian weight map, same size as img
    """
    gauss_img = np.zeros_like(img)
    xs = np.arange(len(img[0, :]))
    gauss_x = gaussian(xs, r, sigma)
    ys = np.arange(len(img[0, :]))
    gauss_y = gaussian(ys, r, sigma)
    for x in xs:
        for y in ys:
            gauss_img[y, x] = gauss_x[x] * gauss_y[y]  # could be optimized, but it is a small array
    return gauss_img

    # ------------------ Signal to noise calculation ---------------------------------


def calc_SNR(wave, spec, der=False, wavelim=[6020, 6500]):
    """
    Calculate the S/N ratio of a spectrum

    Uses a spectrum and the wavelength array to calculate the S/N ratio.
    If der is set to true, it uses the DER_SNR formula.
    If not, pyAstromony is used for the calculation

    Args:
        wave (array): wavelength array of the spectrum
        spec (array): the spectrum
        der (bool): use DER_SNR for SNR calculation

    Returns:
        float: Signal-to-noise ratio of the spectrum
    """
    if not der:  # the default case, uses pyAstronomy
        mask = (wave > wavelim[0]) & (wave < wavelim[1])  # continuum region of the spectrum
        snrEsti = pyasl.estimateSNR(wave[mask], spec[mask], 20, deg=3,
                                    controlPlot=False)  # PyAstronomy routine
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


# Stuff for masking


def dist_ellipse(xc, yc, PA=69, eps=0.28, s=((444, 444))):
    '''
    Creates a new image that contains the semi major axis distance from the specified
    centre at each pixel
    '''
    x, y = np.ogrid[:s[0], :s[1]] - np.array([yc, xc])  # note yc first
    ang = np.radians(PA+90)
    ratio = 1.0 - eps
    xtemp = x*np.cos(ang) + y*np.sin(ang)
    ytemp = -x*np.sin(ang) + y*np.cos(ang)
    rad = np.sqrt(xtemp ** 2 + (ytemp/ratio)**2)
    return rad
