import numpy as np
import pandas as pd
from gPhoton import gFind
from gPhoton import gAperture
import gPhoton.galextools as gt
import gPhoton.dbasetools as dt
from gPhoton.gCalrun import find_random_positions
from gPhoton.MCUtils import print_inline, angularSeparation
import os
from astropy import units as u
from astropy.coordinates import SkyCoord

def crosscheck_all(data,matchradius=0.005):
    d=0
    for i in np.arange(len(data)):
        ix = np.where(angularSeparation(data['RA'][i],data['dec'][i],
                  np.array(data['RA']),np.array(data['dec']))<matchradius)
        n = ix[0].shape[0]
        if n>1:
            d+=1
            print('DUPLICATE: ',i,ix,data['name'][i],n)
    return d

def crosscheck(entry,data,matchradius=0.005):
    if not len(data):
        return False
    ix = np.where(angularSeparation(entry['RA'],entry['dec'],np.array(data['RA']),np.array(data['dec']))<matchradius)
    return ix[0].shape[0]>0

def catfiles(dirpath='../data/'):
    files = {'GUVV2':'aj274302_mrt2.txt',
            'S1':'apj374973t1_ascii.txt', # Shkolnik, et al. (2010)
            'S2':'aj499985t1_ascii.txt', # Shkolnik, et al. (2014)
            'DR7':'apj520168t2_mrt.txt', # Jones & West (2016)
            'PMSU':'apj520168t3_mrt.txt', # Jones & West (2016)
            'LG1':'aj403664t1_mrt.txt', # Lepine & Gaidos (2011)
            'LG2':'aj403664t2_mrt.txt', # Lepine & Gaidos (2011)
            'GTDS':'apj462590t4_mrt.txt', # Gezari, et al.
            'GUVV':'datafile2.txt',
            'M17':'1705.03583.txt'}
    for k in files.keys():
        files[k]='{d}{f}'.format(d=dirpath,f=files[k])
    return files

def nan2none(entry):
    for k in entry.keys():
        try:
            if np.isnan(entry[k]):
                entry[k]=None
        except TypeError:
            pass
    return entry

def read_dr7(data=pd.DataFrame(),csvfile=None):
    filename = catfiles()['DR7']
    with open(filename) as f:
        table = f.readlines()[63:]
    for i,line in enumerate(table):
        if csvfile:
            print('Reading from {f}'.format(f=csvfile))
            data = pd.read_csv(csvfile,index_col=0)
        try:
            if line[0:19].strip() in data.name.tolist():
                print('Skipping...')
                continue
        except AttributeError:
            pass
        entry = {
            'catalog':'Jones & West - DR7',
            #'citation':'Jones, David O., and Andrew A. West. "A CATALOG OF GALEX ULTRAVIOLET EMISSION FROM SPECTROSCOPICALLY CONFIRMED M DWARFSGALEX is operated for NASA by the California Institute of Technology under NASA contract NAS5-98034." The Astrophysical Journal 817.1 (2016): 1.',
            'catfile':filename,
            'name':line[0:19].strip(),
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
            'line':line, # Just for QA.
            }
        if crosscheck(entry,data):
            print('Skipping duplicate entry: {n}'.format(n=entry['name']))
            continue
        entry=nan2none(entry)
        #print(entry)
        data = data.append(pd.Series(entry),ignore_index=True)
        if csvfile:
            print('Writing to {f}'.format(f=csvfile))
            data.to_csv(csvfile)
    return data

def read_pmsu(data=pd.DataFrame(),verbose=1,csvfile=None):
    filename = catfiles()['PMSU']
    with open(filename) as f:
        table = f.readlines()[55:]
    for i,line in enumerate(table):
        if line.strip()=='':
            continue
        if csvfile:
            print('Reading from {f}'.format(f=csvfile))
            data = pd.read_csv(csvfile,index_col=0)
        try:
            if line[0:19].strip() in data.name.tolist():
                continue
        except AttributeError:
            pass
        entry = {
            'catalog':'Jones & West - PMSU',
            #'citation':'Jones, David O., and Andrew A. West. "A CATALOG OF GALEX ULTRAVIOLET EMISSION FROM SPECTROSCOPICALLY CONFIRMED M DWARFSGALEX is operated for NASA by the California Institute of Technology under NASA contract NAS5-98034." The Astrophysical Journal 817.1 (2016): 1.',
            'catfile':filename,
            'name':line[0:19].strip(),
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
            'line':line, # Just for QA.
            }
        if crosscheck(entry,data):
            print('Skipping duplicate entry: {n}'.format(n=entry['name']))
            continue
        entry=nan2none(entry)
        #print(entry)
        data = data.append(pd.Series(entry),ignore_index=True)
        if csvfile:
            print('Writing to {f}'.format(f=csvfile))
            data.to_csv(csvfile)
    return data

def read_lepinegaidos(data=pd.DataFrame(),csvfile=None):
    filename1 = catfiles()['LG1']
    filename2 = catfiles()['LG2']
    with open(filename1) as f1:
        table1 = f1.readlines()[41:]
    with open(filename2) as f2:
        table2 = f2.readlines()[30:]
    for i in range(len(table1)):
        if csvfile:
            print('Reading from {f}'.format(f=csvfile))
            data = pd.read_csv(csvfile,index_col=0)
        try:
            if table1[i][0:16].strip() in data.name.tolist():
                continue
        except AttributeError:
            pass
        entry={
            'catalog':'Lepine & Gaidos',
            #'citation':'Lépine, Sébastien, and Eric Gaidos. "An all-sky catalog of bright M dwarfs." The Astronomical Journal 142.4 (2011): 138.',
            'catfile':filename1,
            'name':table1[i][0:16].strip(),
            'RA':float(table1[i][55:65]),
            'dec':float(table1[i][67:77]),
            'spectype':table2[i][109:112].strip()[0].upper(),
            'specsubtype':float(table2[i][109:112].strip()[1:]),
            'distance':1./float(table2[i][94:101]),
            'line':'{t1}|||||{t2}'.format(
                            t1=table1[i],t2=table2[i]), # Just for QA.
            }
        try:
            entry['nmag']=float(table2[i][25:30])
        except ValueError:
            entry['nmag']=None
        try:
            entry['fmag']=float(table2[i][31:36])
        except ValueError:
            entry['fmag']=None
        if crosscheck(entry,data):
            print('Skipping duplicate entry: {n}'.format(n=entry['name']))
            continue
        entry=nan2none(entry)
        #print(entry)
        data = data.append(pd.Series(entry),ignore_index=True)
        if csvfile:
            print('Writing to {f}'.format(f=csvfile))
            data.to_csv(csvfile)
    return data

def read_shkolnik2010(data=pd.DataFrame(),csvfile=None):
    filename=catfiles()['S1']
    with open(filename) as f:
        table = f.readlines()[5:35]
    for line in table:
        if csvfile:
            print('Reading from {f}'.format(f=csvfile))
            data = pd.read_csv(csvfile,index_col=0)
        #R.A. & Decl.	l	b	R	K	J - H	H - K	log(F_NUV/F_J)^a	log(F_FUV/F_J)^a	Halpha	Note
        entries = line.split('\t')
        if len(line.split('\t'))==10:
            entries = line.split('\t')+[None]
        try:
            if entries[0] in data.name.tolist():
                continue
        except AttributeError:
            pass
        entry={
            'catalog':'Shkolnik2010',
            #'citation':'Shkolnik, Evgenya L., et al. "Searching for Young M Dwarfs with GALEX." The Astrophysical Journal 727.1 (2010): 6.',
            'catfile':filename,
            'name':entries[0],
            'RA':float(entries[1]),
            'dec':float(entries[2]),
            'spectype':None,
            'specsubtype':None,
            'distance':None,
            'line':line, # Just for QA.
            }
        entry['fmag'],entry['nmag']=None, None
        if crosscheck(entry,data):
            print('Skipping duplicate entry: {n}'.format(n=entry['name']))
            continue
        entry=nan2none(entry)
        #print(entry)
        data = data.append(pd.Series(entry),ignore_index=True)
        if csvfile:
            print('Writing to {f}'.format(f=csvfile))
            data.to_csv(csvfile)
    return data

def read_shkolnik2014(data=pd.DataFrame(),csvfile=None):
    filename=catfiles()['S2']
    with open(filename) as f:
        table = f.readlines()[6:233]
    for line in table:
        if line.strip()=='':
            continue
        if csvfile:
            print('Reading from {f}'.format(f=csvfile))
            data = pd.read_csv(csvfile,index_col=0)
        #Name	R.A._J2000	Decl._J2000	SpT	J_2MASS	Dist.^a	Bin.^b	Bin. Sep.^c	F_FUV	F_NUV	References.^d
        entries = line.split('\t')
        if len(entries)==1:
            #print(line.strip())
            continue
        if len(line.split('\t'))==10:
            entries = line.split('\t')+[None]
        try:
            if entries[0] in data.name.tolist():
                continue
        except AttributeError:
            pass
        entry={
            'catalog':'Shkolnik2014',
            #'citation':'Shkolnik, Evgenya L., and Travis S. Barman. "HAZMAT. I. The evolution of far-UV and near-UV emission from early M stars." The Astronomical Journal 148.4 (2014): 64.',
            'catfile':filename,
            'name':entries[0],
            'RA':float(entries[1]),
            'dec':float(entries[2]),
            'spectype':entries[3][0],
            'specsubtype':np.array(entries[3][1:].split('-{s}'.format(
                                s=entries[3][0])),dtype='float16').mean(),
            'distance':float(entries[5]),
            'line':line, # Just for QA.
        }
        entry['fmag'],entry['nmag']=None, None
        if crosscheck(entry,data):
            print('Skipping duplicate entry: {n}'.format(n=entry['name']))
            continue
        entry=nan2none(entry)
        #print(entry)
        data = data.append(pd.Series(entry),ignore_index=True)
        if csvfile:
            print('Writing to {f}'.format(f=csvfile))
            data.to_csv(csvfile)
    return data

def read_miles2017(data=pd.DataFrame(),csvfile=None):
    filename=catfiles()['M17']
    with open(filename) as f:
        table = f.readlines()[55:426]
    for line in table:
        if csvfile:
            print('Reading from {f}'.format(f=csvfile))
            data = pd.read_csv(csvfile,index_col=0)
        # Name, RA, Dec, SpT, Age, Ref, Dist, ...
        entries = line.split('&')
        try:
            if entries[0].strip() in data.name.tolist():
                print('Skipping {n}...'.format(n=entries[0]))
                continue # Note: This (new) catalog should actually take priority!
        except AttributeError:
            pass
        entry={
            'catalog':'Miles&Shkolnik2017',
            #'citation':'preprint:https://arxiv.org/abs/1705.03583',
            'catfile':filename,
            'name':entries[0].strip(),
            'RA':float(entries[1]),
            'dec':float(entries[2]),
            'spectype':entries[3].strip()[0],
            'distance':float(entries[6]),
            'line':line, # Just for QA.
        }
        if entries[3].strip()[1:]==':':
            entry['specsubtype']=None
        else:
            entry['specsubtype']=float(entries[3].strip()[1:])
        entry['fmag'],entry['nmag']=None, None
        if crosscheck(entry,data):
            print('Skipping duplicate entry: {n}'.format(n=entry['name']))
            continue
        entry=nan2none(entry)
        #print(entry)
        data = data.append(pd.Series(entry),ignore_index=True)
        if csvfile:
            print('Writing to {f}'.format(f=csvfile))
            data.to_csv(csvfile)
    return data

def read_guvv(data=pd.DataFrame()):
    return data

def read_guvv2(data=pd.DataFrame()):
    return data

def read_gtds(data=pd.DataFrame()):
    return data

def add_gfind_data(csvfile='mdwarfs.csv',matchradius=0.005):
    data = pd.read_csv(csvfile,index_col=0)
    for i in np.arange(len(data)):
        print_inline('{i}:{n}   '.format(i=i,n=len(data)))
        if not np.isnan(data.ix[i,'nexpt']):
            continue
        gf = gFind(skypos=[data.ix[i,'RA'],data.ix[i,'dec']],quiet=True)
        data.ix[i,'nexpt']=gf['NUV']['expt']
        data.ix[i,'fexpt']=gf['FUV']['expt']
        if (gf['FUV']['expt']>0):
            if gf['FUV']['nearest_source'] is None:
                data.ix[i,'fmag']=None
            elif (gf['FUV']['nearest_source']['distance']<=matchradius):
                data.ix[i,'fmag']=gf['FUV']['nearest_source']['mag']
            else:
                data.ix[i,'fmag']=None
        if (gf['NUV']['expt']>0):
            if gf['NUV']['nearest_source'] is None:
                data.ix[i,'nmag']=None
            elif (gf['NUV']['nearest_source']['distance']<=crossmatch_dist):
                data.ix[i,'nmag']=gf['NUV']['nearest_source']['mag']
            else:
                data.ix[i,'nmag']=None
        data.to_csv(csvfile)
    return data

def mdw_cat(csvfile='mdwarfs.csv'):
    try:
        data = pd.read_csv(csvfile,index_col=0)
    except FileNotFoundError:
        data = pd.DataFrame()
    #data.to_csv(csvfile)
    data = read_miles2017(data=data)
    data = read_dr7(data=data)
    data = read_pmsu(data=data)
    data = read_shkolnik2014(data=data)
    data = read_lepinegaidos(data=data)
    data = read_shkolnik2010(data=data)
    def find(name, path):
        for root, dirs, files in os.walk(path):
            if name in files:
                return os.path.join(root, name)
        return np.nan
    data['lcfile']=pd.Series()
    for i,_ in enumerate(mdwarfs.name):
        fn=data.ix[i,'name'].replace(' ','_').replace('(','').replace(')','')+'_NUV.csv'
        try:
            data.ix[i,'lcfile']=os.path.abspath(
                find(fn,'../lightcurves/mdwarfs.nope/'))
        except AttributeError:
            continue
    data.to_csv(csvfile)
    if crosscheck_all(data)>0:
        print('DUPLICATE ENTRIES DETECTED!')
    return data

def gaussian(x,mu,sigma,normed=False):
    N = (1/(sigma*np.sqrt(2*np.pi))) if normed else 1.
    return N*np.exp(-(x-mu)**2./(2*sigma**2))

def box_smooth(photons,trange,bw=10,expt_ratio=0.888,verbose=1,ts=[],cps=[]):
    #print('Generating BS curve...')
    ts,cps =[], []
    for t in np.arange(trange[0],trange[1]-bw,.1):
        if verbose:
            print('{p}% completed...'.format(
                p=int(100*(t-trange[0])/(trange[1]-trange[0]))))
        ts+=[t+bw/2.]
        ix = np.where((photons['t']>=t) & (photons['t']<t+bw))
        cps+=[ sum(1./photons['response'][ix])/(expt_ratio*bw)]
    if verbose:
        print('Done.              ')
    return ts,cps

def gauss_smooth(photons,trange,bw=10,sigma=2,expt_ratio=0.888,verbose=1,
                 ts=[],cps=[]):
    ts,cps =[], []
    for t in np.arange(trange[0],trange[1]-bw,1):
        if verbose:
            print('{p}% completed...'.format(
                p=int(100*(t-trange[0])/(trange[1]-trange[0]))))
        ts+=[t+sigma/2.]
        ix = np.where((photons['t']>=trange[0]) & (photons['t']<trange[1]))
        cps+=[ sum(gaussian(photons['t'][ix],t,sigma)/
                photons['response'][ix])/(0.888*sigma*np.sqrt(2*np.pi))]
    if verbose:
        print('Done.              ')
    return ts, cps
