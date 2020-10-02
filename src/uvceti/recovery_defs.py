import numpy as np
import pandas as pd
from gPhoton.galextools import counts2flux, mag2counts
from function_defs import get_inff, refine_flare_ranges, calculate_flare_energy

def aflare(t, p):
    """
    This is the Analytic Flare Model from the flare-morphology paper.
    Reference Davenport et al. (2014) http://arxiv.org/abs/1411.3723
    Note: this model assumes the flux before the flare is zero centered
    Note: many sub-flares can be modeled by this method by changing the
    number of parameters in "p". As a result, this routine may not work
    for fitting with methods like scipy.optimize.curve_fit, which require
    a fixed number of free parameters. Instead, for fitting a single peak
    use the aflare1 method.
    Parameters
    ----------
    t : 1-d array
        The time array to evaluate the flare over
    p : 1-d array
        p == [tpeak, fwhm (units of time), amplitude (units of flux)] x N
    Returns
    -------
    flare : 1-d array
        The flux of the flare model evaluated at each time
    """
    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]

    # Do not allow inverted flares
    if p[2] < 0:
        p[2] = 0

    Nflare = int(np.floor((len(p)/3.0)))

    flare = np.zeros_like(t)
    # Compute the flare model for each flare
    for i in range(Nflare):
        # The first lambda function goes out to 4th order.
        outm = np.piecewise(t, [(t <= p[0+i*3]) * (t-p[0+i*3])/p[1+i*3] > -1.,
                                (t > p[0+i*3])],
                            [lambda x: (_fr[0] +
                                        _fr[1]*((x-p[0+i*3])/p[1+i*3]) +
                                        _fr[2]*((x-p[0+i*3])/p[1+i*3])**2. +
                                        _fr[3]*((x-p[0+i*3])/p[1+i*3])**3. +
                                        _fr[4]*((x-p[0+i*3])/p[1+i*3])**4.),
                             lambda x: (_fd[0]*
                                        np.exp(((x-p[0+i*3])/p[1+i*3])*_fd[1]) +
                                        _fd[2]*
                                        np.exp(((x-p[0+i*3])/p[1+i*3])*_fd[3]))]
                            ) * p[2+i*3]
        flare = flare + outm
    return flare


def fake_a_flare(band='NUV',
                 quiescent_mag=18,
                 fpeak_mag=17,
                 stepsz=30., # integration depth in seconds
                 trange=[0, 1600], # visit length in seconds
                 tpeak=250, # flare peak time
                 fwidth=60, # flare "fwhm"
                 resolution=0.05, # normal photon time resolution
                 flat_err=0.15 # 15% error in the NUV flat field
                ):

    quiescent_cps = mag2counts(quiescent_mag, band)
    fpeak_cps = mag2counts(fpeak_mag, band)
    t = np.arange(trange[0], trange[1], resolution)
    flare = (aflare(t, [tpeak, fwidth, max(fpeak_cps - quiescent_cps, 0)]) +
             quiescent_cps)

    tbins = np.arange(trange[0], trange[1], stepsz)
    flare_binned = []
    for t0 in tbins:
        ix = np.where((t >= t0) & (t < t0 + stepsz))
        flare_binned += [np.array(flare)[ix].sum()/len(ix[0])]

    flare_binned_counts = np.array(flare_binned) * stepsz
    flare_obs = np.array([np.random.normal(loc=counts,
                                           scale=np.sqrt(counts))/stepsz
                          for counts in flare_binned_counts])
    flare_obs_err = np.array(
        [np.sqrt(counts)/stepsz for counts in flare_binned_counts])

    model_dict = {'t0':t, 't1':t + resolution,
                  'cps':flare, 'cps_err':np.zeros(len(flare)),
                  'flux':counts2flux(flare, band),
                  'flux_err':np.zeros(len(flare)),
                  'flags':np.zeros(len(flare))}

    # Construct a simulated lightcurve dataframe.
    # NOTE: Since we don't need to worry about aperture corrections, we
    # copy the cps and cps_err into those so the paper's pipeline can run on
    # this simulated light curve too.  We assume no missing time coverage in
    # the time bins, since those with bad coverage are avoided in our paper's
    # pipeline anyways, and thus set the 'expt' to be the same as the requested
    # bin size.
    lc_dict = {'t0':tbins, 't1':tbins+stepsz,
               'cps':flare_obs, 'cps_err':flare_obs_err,
               'cps_apcorrected':flare_obs, 'counts':flare_binned_counts,
               'flux':counts2flux(flare_obs, band),
               'flux_err':counts2flux(flare_obs_err, band),
               'expt':np.full(len(tbins), stepsz),
               'flags':np.zeros(len(tbins))}

    # "TRUTH" + simulated lightcurve dataframe
    return pd.DataFrame(model_dict), pd.DataFrame(lc_dict)

def calculate_ideal_flare_energy(model, quiescence, distance, band='NUV',
                                 effective_widths={'NUV':729.94, 'FUV':255.45},
                                 quiet=True):
    """ Because it's an 'ideal flare,' we can make assumptions that improve
    runtime. """
    q = quiescence

    distance_cm = distance * 3.086e+18 # convert from parsecs to cm
    # NOTE: This does not interpolate over small gaps.
    flare_flux = np.array(model['flux'].values) - counts2flux(q, band)
    flare_flux = np.array([0 if f < 0 else f for f in flare_flux])
    tbins = np.array(model['t1']) - np.array(model['t0'])
    integrated_flux = (tbins * flare_flux).sum()
    """
    GALEX effective widths from
    http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?id=GALEX/GALEX.NUV:
    729.94 Angstroms
    http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?id=GALEX/GALEX.FUV:
    255.45 Angstroms
    """
    fluence = integrated_flux * effective_widths[band]
    energy = (4 * np.pi * (distance_cm**2) * fluence)
    return energy

def inject_and_recover(n=1000, omit_incompletes=True,
                       band='NUV', stepsz=30., trange=[0, 1600], resolution=0.05,
                       distance=2.7, # parsecs
                       quiescent_mag=18, # approx. NUV mag
                       mag_range=[13, 18], # approx. GALEX NUV bright limit
                       detection_threshold=5):
    """
    NOTE: deault for distance, quiescent_mag, and mag_range are for UV Ceti.
    'detection_threshold' is specified as a sigma value
    """

    output = pd.DataFrame({'energy_true':[],
                           'energy_measured':[],
                           'energy_measured_err':[],
                           'energy_measured_w_q':[],
                           'energy_measured_w_q_err':[],
                           'q':[], 'q_err':[],
                           'q_true':[]})

    while len(output['energy_true']) < n:
        printed = False
        if not len(output['energy_true']) % 10000 and not printed:
            print('Injecting: {x}% done...'.format(
                x=float(len(output['energy_true']))/float(n)))
            # Turn off counter in cases where the injected flare is rejected
            # for being truncated, so you don't get the same status update
            # printed multiple times.
            printed = True

        fpeak_mag = np.random.uniform(low=mag_range[0], high=mag_range[1])
        # Peaks within the visit
        tpeak = np.random.uniform(low=trange[0], high=trange[1])
        # FWHM in seconds
        fwidth = np.random.uniform(low=1, high=300)

        model, lc = fake_a_flare(
            band=band, quiescent_mag=quiescent_mag, fpeak_mag=fpeak_mag,
            stepsz=stepsz, trange=trange, tpeak=tpeak, fwidth=fwidth,
            resolution=resolution)

        # Calculate the "true" flare energy.
        model_energy = calculate_ideal_flare_energy(
            model, mag2counts(quiescent_mag, band), distance)

        q, q_err = get_inff(lc)

        # 'n_above_sigma' = minimum consecutive points required for detection
        #fr = find_flare_ranges(lc, sigma=detection_threshold,
        #                       n_above_sigma=2)
        fr, quiescence, quiescence_err = refine_flare_ranges(
            lc, sigma=detection_threshold, makeplot=False)

        if not len(fr):
            output = output.append(pd.Series({
                'energy_true':model_energy,
                'energy_measured':0, #energy[0],
                'energy_measured_err':0, #energy[1],
                'energy_measured_w_q':0, #energy_w_q[0],
                'energy_measured_w_q_err':0, #energy_w_q[1],
                'q':q, 'q_err':q_err,
                'q_true':quiescent_mag}),
                                   ignore_index=True)

        for f in fr:
            if omit_incompletes and ((f[0] == 0) or f[-1] == len(lc)-1):
                continue

            energy = calculate_flare_energy(lc, f, distance, band=band)
            energy_w_q = calculate_flare_energy(lc, f, distance, band=band,
                                                quiescence=[mag2counts(
                                                    quiescent_mag, band), 0.0])
            output = output.append(pd.Series({
                'energy_true':model_energy,
                'energy_measured':energy[0],
                'energy_measured_err':energy[1],
                'energy_measured_w_q':energy_w_q[0],
                'energy_measured_w_q_err':energy_w_q[1],
                'q':q, 'q_err':q_err,
                'q_true':quiescent_mag}),
                                   ignore_index=True)
            # Update counter again.
            printed = False
    return output
