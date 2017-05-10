import numpy as np
import pandas as pd
from gPhoton import gFind
from gPhoton import gAperture
import gPhoton.galextools as gt
import gPhoton.dbasetools as dt
from gPhoton.gCalrun import find_random_positions
from gPhoton.MCUtils import print_inline
import os
from astropy import units as u
from astropy.coordinates import SkyCoord

def catfiles(dirpath='../data/'):
    files = {'GUVV2':'aj274302_mrt2.txt',
            '??':'apj374973t1_ascii.txt',
            'DR7':'apj520168t2_mrt.txt',
            'PMSU':'apj520168t3_mrt.txt',
            'LG1':'aj403664t1_mrt.txt',
            'LG2':'aj403664t2_mrt.txt',
            'GTDS':'apj462590t4_mrt.txt',
            'GUVV':'datafile2.txt'}
    for k in files.keys():
        files[k]='{d}{f}'.format(d=dirpath,f=files[k])
    return files

def read_dr7(data=pd.DataFrame()):
    filename = catfiles()['DR7']
    with open(filename) as f:
        table = f.readlines()[63:]
        for i,line in enumerate(table):
            if line.strip()=='':
                continue
            entry = {
                'catalog':'Jones & West - DR7',
                'catfile':filename,
                'ggoid':int(line[0:19]),
                'RA':float(line[41:52]),
                'dec':float(line[53:64]),
                'nmag':float(line[97:103]),
                'fmag':float(line[110:117]),
                'nexpt':float(line[125:132]),
                'fexpt':float(line[133:140]),
                'spectype':'M',
                'specsubtype':float(line[39:40]),
                'distance':float(line[239:244]),
                }
            data = data.append(pd.Series(entry),ignore_index=True)
    return data

def read_pmsu(data=pd.DataFrame(),verbose=1):
    filename = catfiles()['PMSU']
    with open(filename) as f:
        table = f.readlines()[55:]
        for i,line in enumerate(table):
            if verbose:
                print_inline(i)
            if line.strip()=='':
                continue
            entry = {
                'catalog':'Jones & West - PMSU',
                'catfile':filename,
                'ggoid':int(line[0:19]),
                'RA':float(line[34:45]),
                'dec':float(line[46:57]),
                'nmag':float(line[90:96]),
                'fmag':float(line[103:110]),
                'nexpt':None,
                'fexpt':None,
                'spectype':'M',
                'specsubtype':float(line[30:33]),
                'distance':float(line[214:218]),
                }
            gf = gFind(skypos=[entry['RA'],entry['dec']],quiet=True)
            entry['nexpt']=gf['NUV']['expt']
            entry['fexpt']=gf['FUV']['expt']
            if (entry['fmag']<0) & (entry['fexpt']>0):
                if (gf['FUV']['nearest_source']['distance']<=0.001):
                    entry['fmag']=gf['FUV']['nearest_source']['mag']
            if (entry['nmag']<0) & (entry['nexpt']>0):
                if (gf['NUV']['nearest_source']['distance']<=0.001):
                    entry['nmag']=gf['NUV']['nearest_source']['mag']
            data = data.append(pd.Series(entry),ignore_index=True)
        if verbose:
            print_inline('     ')
    return data

def read_lepinegaidos(data=pd.DataFrame()):
    filename1 = catfiles()['LG1']
    filename2 = catfiles()['LG2']
    with open(filename1) as f1:
        table1 = f1.readlines()[41:]
        with open(filename2) as f2:
            table2 = f2.readlines()[30:]
            for i in range(len(table1)):
                entry={
                'catalog':'Lepine & Gaidos',
                'catfile':filename,
                'name':table1[i][0:16].strip(),
                'RA':float(table1[i][55:65]),
                'dec':float(table1[i][67:77]),
                'spectype':table2[i][109:112].strip()[0].upper(),
                'specsubtype':float(table2[i][109:112].strip()[1:]),
                'distance':1./float(table2[i][94:101])
                }
                try:
                    entry['nmag']=float(table2[i][25:30])
                except ValueError:
                    entry['nmag']=None
                try:
                    entry['fmag']=float(table2[i][31:36])
                except ValueError:
                    entry['nmag']=None

                data = data.append(pd.Series(entry),ignore_index=True)
    return data

def read_guvv(data=pd.DataFrame()):
    return

def read_guvv2(data=pd.DataFrame()):
    return

def read_gtds(data=pd.DataFrame()):
    return
