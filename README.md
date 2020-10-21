[![DOI](https://zenodo.org/badge/258297519.svg)](https://zenodo.org/badge/latestdoi/258297519)

# gfcat_gj65
Analysis of GJ 65 (aka "UV Ceti" and "BL Ceti") as part of the GALEX Flare Catalog (GFcat) project.  Supported by NASA Grant 80NSSC18K0084.

## Contents
This repository contains Python notebooks and some data files needed to reproduce the data, plots, tables, and some calculations in Fleming et al. 2020 "New Time-Resolved Flares In The GJ 65 System With gPhoton".

### Notebooks
There are eight Python notebooks included in the [src/uvceti/](src/uvceti/) folder, and they are numbered based on the most natural order of running them.  The regular Python files [src/uvceti/function_defs.py](src/uvceti/function_defs.py) and [src/uvceti/recovery_defs.py](src/uvceti/recovery_defs.py) contains the functions used in the notebooks, and are imported at the top of each one as needed.  The other .txt files are filter transmissions from the SVO Filter Service, which are read in by the notebooks (especially the one that estimates bolometric flare flux contributions for different bands).

  - [00-calculate_pbol.ipynb](src/uvceti/00-calculate_pbol.ipynb) = Estimates the bolometric contribution of GALEX and shows that other bolometric contributions for flares in the bandpasses used in the FFD comparison (Kepler, TESS, Evryscope, Johnson U band) are all within an order of magnitude of each other.
  
  - [01-generate_products.ipynb](src/uvceti/01-generate_products.ipynb) = This notebook is used to retrieve the raw photon event files, calibrate the photon events, and create the initial set of light curves.  WARNING: Some of these functions can take a few hours to run, and it generates several hundred GB of data if you are downloading and creating the photon event files.  For those who don't want to do this step, you can skip it and use the 30-second and 5-second light curves provided directly in the repository (see section below).

  - [02-make_visit_thumbnails.ipynb](src/uvceti/02-make_visit_thumbnails.ipynb) = Creates the thumbnail plot of the GALEX images centered on GJ 65, the photometric aperture used, and highlights where the hotspot mask area is relative to the aperture (and why it is OK for us to include them here, since it is not close enough to impact GJ 65 even if the conservative main branch of the gPhoton software would normally not attempt to calibrate photon events so close to the hotstpot area).

  - [03-make_flare_table_and_figs.ipynb](src/uvceti/03-make_flare_table_and_figs.ipynb) = Generates plots of the light curves and the table of flare properties.

  - [04-ffd_analysis.ipynb](src/uvceti/04-ffd_analysis.ipynb) = Calculates a rough estimate for the flare frequemcy rate within the energies of the flares found in our paper, and compares with FFD from other ground- and space-based surveys in the optical.

  - [05-flare8_colordiff_analysis.ipynb](src/uvceti/05-flare8_colordiff_analysis.ipynb) = Demonstrates that the count rate during the large Flare #8 exceeds the local non-linearity threshold, and thus a FUV-NUV ratio analysis of the flare is unfortunately not possible in the absence of a (nonexistent) robust correction for the flux depression caused by the local non-linearity in the GALEX detectors.

  - [06-qpp_analysis.ipynb](src/uvceti/06-qpp_analysis.ipynb) = Analysis of the quasi-periodic pulsation (QPP) during Flare #8, and shows that the strong signal at ~50 seconds in both FUV and NUV bands is not related to the dither pattern, and thus is highly unlikely to be caused by any known gPhoton systematics.  The full analysis of this QPP signal is done with an IDL package by a co-author that is made available online and is linked in the paper itself.

  - [07-injection_recovery.ipynb](src/uvceti/07-injection_recovery.ipynb) = This notebook creates 100,000 simulated flares and adds them to a simulated GJ 65 gPhoton light curve.  It then runs them through our INFF determination and flare detection algorithm.  We then determine the fraction of undetected flares at various energies, and demonstrate that nearly all the simulated flares at the same energies as those found in the paper are detected.

### Light Curve Files
Generating our data products is possible with the software in this repository, but it can take many hours and generates hundreds of GB of data.  Most users will likely want to analyze the light curve files of GJ 65 and the comparison stars themselves, thus we include them in the [src/uvceti/raw_files/](src/uvceti/raw_files/) directory for convenience.  This includes all versions of the 30-second and 5-second light curves of GJ 65 itself, as well as the 5-second light curves of the comparison stars used in the QPP notebook.

### Special gPhoton Software Branch.
[src/uvceti/gPhoton/](src/uvceti/gPhoton/) = Contains a copy of the gPhoton software (a special branch called '1.28.9_nomask') used to generate the products for the paper.  This version reprocesses data near hotspot masks to get the maximum amount of usable GALEX data for the GJ 65 system.  For those who want to re-generate everything from scratch, starting from the raw photon events, you'll need to use this version of gPhoton, and it is generally easiest to run the notebooks from the same directory level as the gPhoton software folder in this repository (or make sure your Python path is pointing to this specific version of gPhoton).
