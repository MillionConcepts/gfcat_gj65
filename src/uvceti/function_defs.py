"""
.. module:: function_defs
   :synopsis: Helper functions for the UV Ceti gPhoton notebook tutorials, used
       to re-create the data and figures used in the paper.

.. moduleauthor:: Chase Million, Scott W. Fleming
"""

import os
from astropy.io import fits as pyfits
from astropy import wcs as pywcs
import numpy as np
import pandas as pd
import gPhoton

def listdir_contains(directory, contains_str):
    """This function returns a sorted list of files that match the given
    pattern."""
    listd = os.listdir(directory)
    listd.sort()
    return ['{d}{f}'.format(d=directory, f=fn) for fn in listd if (
        contains_str in fn)]

def write_image(photon_file, band, imsz=[3600, 3600], overwrite=False,
                skypos=(24.76279, -17.94948)):
    """This function writes an image to disk based on the photon event .csv
    file."""
    fitsfilename = photon_file.replace('csv', 'fits')
    if os.path.exists(fitsfilename) and not overwrite:
        print('  Reading image from {f}'.format(f=fitsfilename))
        hdulist = pyfits.open(fitsfilename, memmap=1)
        image = hdulist[0].data
        wcs = pywcs.WCS(hdulist[0].header)
    else:
        print('  Making image {f}'.format(f=fitsfilename))
        image, wcs = make_image(photon_file, band, imsz=imsz, skypos=skypos)
        write_fits(image, wcs, fitsfilename, band, overwrite=overwrite)

def write_fits(image, wcs, fitsfilename, band, overwrite=False):
    """This function creates a FITS file out of a gPhoton image."""
    hdu = pyfits.PrimaryHDU()
    hdu.header['CDELT1'], hdu.header['CDELT2'] = wcs.wcs.cdelt
    hdu.header['CTYPE1'], hdu.header['CTYPE2'] = wcs.wcs.ctype
    hdu.header['CRPIX1'], hdu.header['CRPIX2'] = wcs.wcs.crpix
    hdu.header['CRVAL1'], hdu.header['CRVAL2'] = wcs.wcs.crval
    hdu.header['EQUINOX'], hdu.header['EPOCH'] = 2000., 2000.
    hdu.header['BAND'] = 1 if band == 'NUV' else 2
    hdu.header['VERSION'] = 'v{v}'.format(v=gPhoton.__version__)
    hdu = pyfits.PrimaryHDU(image)
    print('  Writing image to {f}'.format(f=fitsfilename))
    hdu.writeto(fitsfilename, overwrite=overwrite)

def make_image(photon_file, band, events=None, trange=None, imsz=[3600, 3600],
               skypos=(24.76279, -17.94948), pixsz=0.000416666666666667):
    """This function creates an image out of photon events."""
    if events is None:
        events = calibrate_photons(photon_file, band)
    if not trange:
        trange = (events['t'].loc[np.isfinite(events['ra'])].min(),
                  events['t'].loc[np.isfinite(events['ra'])].max())
    tix = np.where((events['t'] >= trange[0]) & (events['t'] < trange[1]))
    wcs = define_wcs(events, imsz=imsz, skypos=skypos, pixsz=pixsz)
    coo = list(zip(events.iloc[tix]['ra'], events.iloc[tix]['dec']))
    foc = wcs.sip_pix2foc(wcs.wcs_world2pix(coo, 1), 1)
    image, _, _ = np.histogram2d(foc[:, 1]-0.5, foc[:, 0]-0.5, bins=imsz,
                                 range=([[0, imsz[0]], [0, imsz[1]]]))
    return image, wcs

def calibrate_photons(photon_file, band, overwrite=False):
    """This function takes the raw photon events and produces calibrated
    photon events."""
    xfilename = '{d}{base}-x.csv'.format(d=photon_file[:-13],
                                         base=photon_file[-13:-4])
    if os.path.exists(xfilename) and not overwrite:
        return pd.read_csv(xfilename, index_col=None)
    data = pd.read_csv(photon_file, names=['t', 'x', 'y', 'xa', 'ya', 'q', 'xi',
                                           'eta', 'ra', 'dec', 'flags'])
    events = pd.DataFrame()
    flat, _ = gPhoton.cal.flat(band)
    col, row = gPhoton.curvetools.xieta2colrow(np.array(data['xi']),
                                               np.array(data['eta']), band)
    # Use only data that is on the detector.
    ix = np.where((col > 0) & (col < 800) & (row > 0) & (row < 800))
    events['t'] = pd.Series(np.array(data.iloc[ix]['t'])/1000.)
    events['ra'] = pd.Series(np.array(data.iloc[ix]['ra']))
    events['dec'] = pd.Series(np.array(data.iloc[ix]['dec']))
    events['flags'] = pd.Series(np.array(data.iloc[ix]['flags']))
    events['col'] = pd.Series(col[ix])
    events['row'] = pd.Series(row[ix])
    flat = flat[np.array(events['col'], dtype='int16'),
                np.array(events['row'], dtype='int16')]
    events['flat'] = pd.Series(flat)
    scale = gPhoton.galextools.compute_flat_scale(
        np.array(data.iloc[ix]['t'])/1000., band)
    events['scale'] = pd.Series(scale)
    response = np.array(events['flat'])*np.array(events['scale'])
    events['response'] = pd.Series(response)

    # Define the hotspot mask
    mask, maskinfo = gPhoton.cal.mask(band)

    events['mask'] = pd.Series(
        (mask[np.array(col[ix], dtype='int64'),
              np.array(row[ix], dtype='int64')] == 0))

    # Add the remaining photon list parameters back in for completeness.
    events['x'] = pd.Series(np.array(data.iloc[ix]['x']))
    events['y'] = pd.Series(np.array(data.iloc[ix]['y']))
    events['xa'] = pd.Series(np.array(data.iloc[ix]['xa']))
    events['ya'] = pd.Series(np.array(data.iloc[ix]['ya']))
    events['q'] = pd.Series(np.array(data.iloc[ix]['q']))
    events['xi'] = pd.Series(np.array(data.iloc[ix]['xi']))
    events['eta'] = pd.Series(np.array(data.iloc[ix]['eta']))

    print('Writing {xf}'.format(xf=xfilename))
    events.to_csv(xfilename, index=None)

    return events

def define_wcs(events, imsz=[3200, 3200], skypos=None,
               pixsz=0.000416666666666667):
    """Generates a WCS for a gPhoton image.
    Note: The default for 'pixsz' is the same resolution as the GALEX intensity
    ("-int") maps."""
    if skypos is None:
        skypos = (events['ra'].min() + (events['ra'].max() -
                                        events['ra'].min())/2,
                  events['dec'].min() + (events['dec'].max()
                                         -events['dec'].min())/2)
    wcs = pywcs.WCS(naxis=2)
    wcs.wcs.cdelt = np.array([-pixsz, pixsz])
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.crpix = [(imsz[1]/2.) + 0.5, (imsz[0]/2.) + 0.5]
    wcs.wcs.crval = skypos
    return wcs
