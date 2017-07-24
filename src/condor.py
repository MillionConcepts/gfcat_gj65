import numpy as np
import pandas as pd
from gPhoton import gFind
from gPhoton import gAperture
import gPhoton.galextools as gt
import gPhoton.dbasetools as dt
from gPhoton.gCalrun import find_random_positions
import os
from astropy import units as u
from astropy.coordinates import SkyCoord

radius = gt.aper2deg(6)
annulus = [gt.aper2deg(6)*4,gt.aper2deg(6)*10]
stepsz=30.

def lc_condor(opath,target,band,gphoton_input,gphoton_output):
    #os.path.isfile(csv)
    with open('{o}/{t}_{b}.condor'.format(o=opath,t=target,b=band),'w') as f:
        print('executable = {p}/{t}_{b}.sh'.format(p=gphoton_input,t=target,b=band),file=f)
        print('output = {p}/{t}_{b}.condor_stdout'.format(p=gphoton_output,t=target,b=band),file=f)
        print('error = {p}/{t}_{b}.condor_stderr'.format(p=gphoton_output,t=target,b=band),file=f)
        print('log = {p}/{t}_{b}.condor_log'.format(p=gphoton_output,t=target,b=band),file=f)
        print('getenv = True',file=f)
        print('notification = Never',file=f)
        print('universe = vanilla',file=f)
        print('queue = 1',file=f)
    return

def lc_py(opath,target,band,skypos,gphoton_input,gphoton_output):
    with open('{o}/{t}_{b}.py'.format(o=opath,t=target,b=band),'w') as f:
        print('from gPhoton.gAperture import gaperture as gAperture',file=f)
        print('def main():',file=f)
        print('    try:',file=f)
        print('        gAperture(band="{b}", skypos=[{ra}, {dec}], stepsz={step}, csvfile="{p}/{t}/{t}_{b}.csv", overwrite=True, radius={rad}, annulus=[{i}, {o}], verbose=3, maxgap=50, photoncsvfile="{p}/{t}/{t}_{b}_photons.csv")'.format(
            p='/'.join(gphoton_output.strip().split('/')[:-2]),
            b=band,ra=skypos[0],dec=skypos[1],step=stepsz,t=target,rad=radius,
            i=annulus[0],o=annulus[1]),file=f)
        print('    except:',file=f)
        print('        with open("{p}/{t}/ERROR","w") as ofile:'.format(
            p='/'.join(gphoton_output.strip().split('/')[:-2]),t=target),file=f)
        print('            ofile.write("ERROR")',file=f)
        print('if __name__ == "__main__":',file=f)
        print('    main()',file=f)
    return

def lc_sh(opath,target,band,gphoton_input,gphoton_output):
    with open('{o}/{t}_{b}.sh'.format(o=opath,t=target,b=band),'w') as f:
        print('#!/bin/tcsh',file=f)
        print('date',file=f)
        print('mkdir -p {p}/{t}/'.format(
            p='/'.join(gphoton_output.strip().split('/')[:-2]),t=target),file=f)
        print('python {t}_{b}.py'.format(t=target,b=band),file=f)
        print('date',file=f)
    return

def make_condor_scripts(opath,target,skypos,band,catalog='MCAT',
                        radius=gt.aper2deg(6),stepsz=10.,
                        annulus=[gt.aper2deg(6)*4,gt.aper2deg(6)*10]):
    gphoton_input='/data2/fleming/GPHOTON_INPUT/{c}/CONDOR/'.format(c=catalog)
    gphoton_output='/data2/fleming/GPHOTON_OUTPUT/LIGHTCURVES/{c}/CONDOR_OUTPUT/'.format(c=catalog)
    #for band in ['FUV','NUV']:
    lc_condor(opath,target,band,gphoton_input,gphoton_output)
    lc_py(opath,target,band,skypos,gphoton_input,gphoton_output)
    lc_sh(opath,target,band,gphoton_input,gphoton_output)
    return

def make_mdwarf_condor_scripts(csvfile='mdwarfs.csv',opath='condorscripts/.'):
    data = pd.read_csv(csvfile)
    for i,_ in enumerate(data.name):
        target = data.ix[i,'name'].replace(
                                ' ','_').replace('(','').replace(')','')
        skypos = [data.ix[i,'RA'],data.ix[i,'dec']]
        catalog = {'Jones & West - DR7':'dr7', 'Jones & West - PMSU':'pmsu',
                   'Lepine & Gaidos':'lepinegaidos',
                   'Miles&Shkolnik2017':'miles2017',
                   'Shkolnik2010':'shkolnik2010',
                   'Shkolnik2014':'shkolnik2014'}[data['catalog'][i]]
        print('{c} {t} {s}'.format(c=catalog,t=target,s=skypos))
        #if data['fexpt'][i]>0:
        make_condor_scripts(opath,target,skypos,band='FUV',catalog=catalog)
        #if data['nexpt'][i]>0:
        make_condor_scripts(opath,target,skypos,band='NUV',catalog=catalog)
    return
