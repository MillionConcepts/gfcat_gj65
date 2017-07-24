from gPhoton.gphoton_utils import read_lc
from gPhoton import gFind
import pandas as pd
import numpy as np
import gPhoton.dbasetools as dt

def compute_mad(lc):
    dev = lc.flux_bgsub-lc.flux_bgsub.median()
    MAD_rel=(np.abs(dev/lc.flux_bgsub.median())).values
    MAD_rel[np.where(dev<0)]=0 # We're only interested in positive deviations.
    return MAD_rel

def get_variable_ts(lc,cutoff=1):
    MAD_rel = compute_mad(lc)
    ts = lc['t_mean'][MAD_rel>cutoff][lc.flags==0].values
    if len(ts)==0:
        ts = lc['t_mean'][MAD_rel>cutoff].values
    return ts

def get_tranges(ts,gf,band='NUV'):
    tranges = []
    for trange in zip(gf[band]['t0'],gf[band]['t1']):
        if any((trange[0]<=ts) & (trange[1]>=ts)):
            tranges+=[trange]
    return tranges

lcpath = '../lightcurves/'

# Lepine & Gaidos
lppath = 'lepinegaidos/'
files = [
'PM_I00370+4224_lc.csv', # (996678658.995,996680100.995), (939199708.995,939201341.995), (780266290.995,780267905.995)
'PM_I00428+3532_lc.csv', # (815314219.995,815315923.995)
'PM_I01433+0419_lc.csv', # (877976004.995,877977696.995)
'PM_I01562+0006_lc.csv', # (975401977.995,975403623.995), (975313263.995,975314926.995)
'PM_I01587+3515_lc.csv', # (1006117722.995,1006119392.995)
#Maybe flaring:
]#'PM_I01369-0647_lc.csv',
#'PM_I01567-0021_lc.csv',
#'PM_I02088+4926_lc.csv',
#'PM_I08011+0315_lc.csv',
#'PM_I14019-0929_lc.csv']

for f in files:
    filename = '{p}{d}{f}'.format(p=lcpath,d=lppath,f=f)
    lc = read_lc(filename)
    name = f[:-7].replace('_',' ')
    entry = data[data.name==f[:-7].replace('_',' ')]
    skypos = [entry.RA.values[0],entry.dec.values[0]]
    print(entry.name,skypos)
    gf = gFind(skypos=skypos,minexp=500)
    for trange in zip(gf['NUV']['t0'],gf['NUV']['t1']):
        ix = np.where((lc['t0']>=trange[0]) & (lc['t1']<=trange[1]))
        plt.figure('{n} {t}'.format(n=f[:-7].replace('_',' '),t=trange))
        plt.errorbar(lc['t_mean'].ix[ix].values-lc['t0'].ix[ix].min(),
            lc['flux'].ix[ix].values,
            yerr=lc['flux_err'].ix[ix].values*3,fmt='kx')

galexEqWth = 795.65
data = pd.read_csv('mdwarfs.csv',index_col=0)
filename = '{p}{d}{f}'.format(p=lcpath,d=lppath,f='PM_I00370+4224_lc.csv')
lc = read_lc(filename)
flarestar = data[data.name=='PM I00370+4224']
distance = flarestar.distance.values[0]
skypos = [flarestar.RA.values[0],flarestar.dec.values[0]]
tranges = [(996678658.995,996680100.995), (939199708.995,939201341.995), (780266290.995,780267905.995)]
trange = tranges[0]
ix = np.where((lc['t0']>=trange[0]) & (lc['t1']<=trange[1]))[0][1:-1]
quiescent_flux = lc['flux_bgsub'].median()
flare_flux = (lc['flux_bgsub'].ix[ix]-quiescent_flux).values
tbins = lc['t_mean'].ix[ix][1:].values-lc['t_mean'].ix[ix][:-1].values
dflux = flare_flux[1:]-flare_flux[:-1]
integrated_flux = (tbins*(flare_flux[:-1]+.5*dflux)).sum()
fluence = integrated_flux*galexEqWth

# CR Draconis
import gPhoton
skypos = [244.27246917, 55.26919386]
gf=gFind(skypos=skypos)
if os.path.exist:
    lc = read_lc('crdra.csv')
    photons = pd.read_csv('crdra_photons.csv')
else:
    nuv=gPhoton.gAperture(band='NUV',skypos=skypos,stepsz=10.,radius=0.006,annulus=[0.007,0.009],verbose=2,csvfile='crdra.csv',photoncsvfile='crdra_photons.csv')
    lc = nuv
    photons = nuv['photons']
for trange in zip(gf['NUV']['t0'],gf['NUV']['t1']):
    print(trange)
    ix = np.where((lc['t0']>=trange[0]) & (lc['t1']<=trange[1]))
    print(len(ix[0]))
    plt.figure('{n} {t}'.format(n=f[:-7].replace('_',' '),t=trange))
    plt.errorbar(lc['t_mean'].ix[ix].values-lc['t0'].ix[ix].min(),
        lc['flux'].ix[ix].values,
        yerr=lc['flux_err'].ix[ix].values*3,fmt='kx')
trange = (799279196.995,799280851.995) # ~1.4e30 ergs
ix = np.where((lc['t0']>=trange[0]) & (lc['t1']<=trange[1]))[0][1:-1][75:]
quiescent_flux = lc['flux_bgsub'].median()
flare_flux = (lc['flux_bgsub'].ix[ix]-quiescent_flux).values
tbins = lc['exptime'].ix[ix].values
integrated_flux = (tbins*flare_flux).sum()
galexEqWth = 795.65
fluence = integrated_flux#*galexEqWth
distance = 6.37192e+19 # cm (20.65 parsecs) -- note: 3.086e+18 cm / parsec
energy =  (4 * np.pi * (distance**2) * fluence)# / galexBPpercentE
print(energy)

def calculate_energy(lc,trange,distance,filtercorr=False,quiescent_flux=None):
    ix = np.where((lc['t0']>=trange[0]) & (lc['t1']<=trange[1]))[0][1:-1]
    quiescent_flux = lc['flux_bgsub'].median()
    flare_flux = (lc['flux_bgsub'].ix[ix]-quiescent_flux).values
    tbins = lc['exptime'].ix[ix].values
    integrated_flux = (tbins*flare_flux).sum()
    galexEqWth = 795.65 if filtercorr else 1.
    fluence = integrated_flux*galexEqWth
    distance *= 3.086e+18 # cm / parsec
    energy =  (4 * np.pi * (distance**2) * fluence)# / galexBPpercentE
    return energy

# Jones & West
jwpath = '../lightcurves/joneswest/'
dr7path = 'DR7/'
filenames = [
'DR7_2431594291587455831_FUV',
'DR7_2431594291587455831_NUV',
'DR7_2492146595950233381_NUV',# (?)
'DR7_2883924579161475595_NUV',
'DR7_2883924579161475595_FUV',
'DR7_3053724358862188680_NUV',
'DR7_3053724358862188680_FUV',
'DR7_3133205855410855589_NUV',
'DR7_3133205855410855589_FUV',
'DR7_3715507213481020668_NUV',
'DR7_3715507213481020668_FUV',]
for filename in filenames:
    band = filename.split('_')[2]
    if band=='FUV':
        continue
    path = '{lcpath}{catpath}{filename}.csv'.format(
                            lcpath=jwpath,catpath=dr7path,filename=filename)
    if not os.path.exists(path):
        print('No file: {p}'.format(p=path))
        continue
    name = filename.split('_')[1]
    print(name)
    star = mdwarfs[mdwarfs.name==name]
    distance = star.distance.values[0]
    skypos = star.RA.values[0],star.dec.values[0]
    data = read_lc(path)
    ts = get_variable_ts(data)
    gf = gFind(skypos=skypos,quiet=True)
    tranges = get_tranges(ts,gf,band=band)
    for trange in tranges:
        energy = calculate_energy(data,trange,distance)
        print('   ',energy)

    #plt.figure()
    #plt.title('MAD: {n}'.format(n=name))
    #plt.plot(MAD_rel,'.')
    #plt.figure()
    #plt.title('flux: {n}'.format(n=name))
    #plt.plot(data['flux_bgsub'],'x')
    #print(os.path.exists(path),path)


pmsupath = 'PMSU/'
filenames = [
'PMSU_2565048614918298564_NUV',# (?)
'PMSU_2908236980274858545_NUV',
'PMSU_2908236980274858545_FUV',
'PMSU_2946623130423012601_NUV',# (big)
'PMSU_3235522009644409893_NUV',# (non-flare / visit-to-visit variable)
'PMSU_4491991121308680773_NUV',# (?, flagged)
'PMSU_6371267618536425133_NUV',# (?)
# G 164-47 [197.3950211, 28.9847562] M4.8 flare star, binary UV Ceti, also in L&G
'PMSU_6374891691541794242_NUV',# (big)
'PMSU_6374891691541794242_FUV',# (big)
'PMSU_6375349012143277863_NUV',# (medium, incomplete)
'PMSU_6375349012143277863_FUV',# (medium, incomplete)
'PMSU_6383441448862222955_NUV',# (multiple)
'PMSU_6400461831951878683_FUV',
'PMSU_6400461831951878683_NUV',]# (double peak)
for filename in filenames:
    band = filename.split('_')[2]
    #if band=='FUV':
    #    continue
    path = '{lcpath}{catpath}{filename}.csv'.format(
                            lcpath=jwpath,catpath=pmsupath,filename=filename)
    if not os.path.exists(path):
        print('No file: {p}'.format(p=path))
        continue
    name = filename.split('_')[1]
    if len(mdwarfs[mdwarfs.name==name])==0:
        print('Skipping duplicate: {n}'.format(n=name))
        continue
    print(band,name)
    star = mdwarfs[mdwarfs.name==name]
    distance = star.distance.values[0]
    skypos = star.RA.values[0],star.dec.values[0]
    data = read_lc(path)
    ts = get_variable_ts(data,cutoff=0.5)
    gf = gFind(skypos=skypos,quiet=True)
    tranges = get_tranges(ts,gf,band=band)
    for trange in tranges:
        energy = calculate_energy(data,trange,distance)
        print('   ',energy)

    plt.figure()
    plt.title('MAD: {n} ({b})'.format(n=name,b=band))
    plt.plot(compute_mad(data),'.')
    plt.figure()
    plt.title('flux: {n} ({b})'.format(n=name,b=band))
    plt.plot(data['flux_bgsub'],'x')
    print(os.path.exists(path),path)

def compute_mad(lc):
    dev = lc.flux_bgsub-lc.flux_bgsub.median()
    MAD_rel=(np.abs(dev)/lc.flux_bgsub.median()).values
    MAD_rel[np.where(dev<0)]=0 # We're only interested in positive deviations.
    return MAD_rel

mad_rel = []
filenames = []
isvar = []
total_exptime=0
cnt=0
for w in os.walk('../lightcurves/mdwarfs.nope/'):
    for f in w[2]:
        if f=='ERROR' or f=='.DS_Store':
            continue
        filename = '{d}/{f}'.format(d=w[0],f=f)
        band = f[-7:-4]
        if band=='FUV':
            continue
        cnt+=1
        name = f[:-8]
        star = []
        for i,n in enumerate(mdwarfs.name):
            if n.replace(' ','_').replace('(','').replace(')','')==name:
                star = mdwarfs.ix[i]
        data = read_lc(filename)
        exptime=data.exptime.sum()
        if exptime<100:
            #print(data.exptime.sum())
            continue
        if not np.isfinite(star.nmag):
            continue
        if star.nmag>20:
            continue
        total_exptime+=exptime
        print(star.nmag,star['name'])#,filename)
        ix = np.where(data['flags']==0)
        if not len(ix[0]):
            continue
        mad_rel+=[max(compute_mad(data)[np.where(data['flags']==0)])]
        filenames+=[filename]
        #checkplot(filename,figout=True,cleanup=True,minbins=3)
plt.plot(mad_rel,'.')

"""
Flares:
WW Ari
Wolf424
UV Cet # would be dumb not to to find some here!
PM I22437+1916 # incomplete
PM I22333-0936 # multiple, small
PM I22278-0113 # multiple
PM I21376+0137 # multiple
PM I21188+0018 # big
PM I16170+5516 # multiple, full range
PM I15474+4507 # starts below bg
PM I15099-0226
PM I14386-0257 # flagged, but pretty fucking obvious, double peak
PM I13260+2735 # ton of visits
PM I13255+2738 # ton of visits
PM I12265+3347 # multiple, small
PM I11476+0015 # huge but flagged, nonlinear?
PM I10552-0335S # multiple
PM I10360+0507 # 1 flare perfectly framed in MIS
PM I10315+5705 # multiple of all sizes
PM I09598+0246 # 1 small one in a ton of visits
PM I09593+4350W # incomplete
PM I09302+2630
PM I09034-0023 # small
PM I08316+1923N # huge and tiny
PM I08306+0421 # multiple, short
PM I08158+4601 # one big one
PM I04556-1917 # short (~1 minute)
PM I03462+1709W # partial tail
PM I01587+3515
PM I01567-0021 # small
PM I01562+0006 # multiple
PM I00428+3532 # missing tail
PM I00370+4224 # multiple partials
NLTT7704 # obvious but barely significant
LHS3776 # 1 flare in a ton of visits, maybe also a micro-flare
LHS513 # 1 obvious one, maybe some micro-flares
GJ3959 # multiple small and some partials
GJ1207 # 1 large, multiple smalls
GJ1167A # 1 very large, flagged (nonlinearity?)
GJ669A # 1 compound
G166-49 # multiple, range of sizes, most partial
G56-11 # obvious but low significance
3133205855410855589
2908236980274858545 # obvious but low significance
2883924579161475595 # obvious but short
2431594291587455831

Possible flares:
PM I23489-0441
PM I23325-1215
PM I23291-0251
PM I22035+0340 # flagged
PM I20532-0221 # barely significant, incomplete
PM I20105+0631 # flagged
PM I16243+1959W #flagged
PM I12444+1532 # noisy and flagged
PM I12214+3038E # barely
PM I08580-0651 # huge jump in 1 bin, flagged
PM I07472+5020 # barely
PM I02167+0112 # small and flagged
PM I01369-0647 # maybe several very small
PM I01036+4051 # noisy with some flags
PM I00413+5550W # jump in center point of AIS visit
PM I00258-0957 # barely
NLTT48651 # large jump in middle of AIS
NLTT48492 # barely
NLTT46734 # maybe micro-activity
NLTT26114 # short and flagged
LHS1363 # noisy, maybe micro-flaring
LHS1051 # barely significant and partial
GJ3942 # maybe some micro-flaring
GJ3729 # maybe some micro-flaring
GJ3572 # maybe some micro-flaring
G41-14 # noisy as hell and lots of flags
6453519865059739235 # maybe one tiny flare (flagged) in a ton of visits
6381963680536332629 # maybe small flares, noisy and flagged
6380415598231295319 # barely significant
6373589777430612946 # barely significant
6372217624502208357 # barely significant
3822608442318654560 # 1 bin, barely significant
2MASS_J10364483+1521394_A_N_NUV # possible partial

Non-flares:
PM I23450+1458
PM I22561-0009
PM I22274-3500 # rises above detection threshhold?
PM I22257+6905
PM I22021+0124 # jumps around a lot at barely significance
PM I21554+2849 # rises above detection threshhold?
PM I20225+0921 # barely significant
PM I19170-5238W # rises above detection, but flagged
PM I17038+2850 # falls below detection
PM I16373-2003
PM I16314+4710 # rises above detection
PM I16159+3852 # drops out
PM I15368+3734W # jumps
PM I15354+6005 # drops off over single AIS
PM I12157+5231 # jumps
PM I12156+5239 # barely
PM I11314+1344 # jumps and then some subvisit variability
PM I09456+0333 # jump and wiggles
PM I08364+6717 # jump and slight downward trend
PM I07319+3613S # downward trend in AIS
PM I07249+3054 # trend
PM I04439+3723W # jump
PM I04238+1455 # jump and wiggle, maybe tail end of a flare?
PM I03094+6732 # jump to barely detectable
PM I02327-3421N # jump and trend in AIS
PM I02274+0310 # jump between AIS
PM I02159-0929 # jump
PM I02101-1548 # sharp trend in AIS
PM I01376+1835 # sharp trend in AIS
PM I01214+3120W # jump at high significance and maybe micro-activity
PM I00298-5441 # sharp trend in AIS
VSV6431 # jump
GJ3966 # noisy but maybe something
GJ493.1 # jump between AIS
6384496925264052860 # huge jump between short (petal?) visits
6379993387913711723 # jump between AIS
6373519479555491781 # jump between AIS
3235522009644409893 # jumps
2485356012139185335 # jumps
03244056-3904227 # jump between AIS
"""

flares_table = [
'WW Ari',
'Wolf424',
'UV Cet', # would be dumb not to to find some here!
'PM I22437+1916', # incomplete
'PM I22333-0936', # multiple, small
'PM I22278-0113', # multiple
'PM I21376+0137', # multiple
'PM I21188+0018', # big
'PM I16170+5516', # multiple, full range
'PM I15474+4507', # starts below bg
'PM I15099-0226',
'PM I14386-0257', # flagged, but pretty fucking obvious, double peak
'PM I13260+2735', # ton of visits
'PM I13255+2738', # ton of visits
'PM I12265+3347', # multiple, small
'PM I11476+0015', # huge but flagged, nonlinear?
'PM I10552-0335S', # multiple
'PM I10360+0507', # 1 flare perfectly framed in MIS
'PM I10315+5705', # multiple of all sizes
'PM I09598+0246', # 1 small one in a ton of visits
'PM I09593+4350W', # incomplete
'PM I09302+2630',
'PM I09034-0023', # small
'PM I08316+1923N', # huge and tiny
'PM I08306+0421', # multiple, short
'PM I08158+4601', # one big one
'PM I04556-1917', # short (~1 minute)
'PM I03462+1709W', # partial tail
'PM I01587+3515',
'PM I01567-0021', # small
'PM I01562+0006', # multiple
'PM I00428+3532', # missing tail
'PM I00370+4224', # multiple partials
'NLTT7704', # obvious but barely significant
'LHS3776', # 1 flare in a ton of visits, maybe also a micro-flare
'LHS513', # 1 obvious one, maybe some micro-flares
'GJ3959', # multiple small and some partials
'GJ1207', # 1 large, multiple smalls
'GJ1167A', # 1 very large, flagged (nonlinearity?)
'GJ669A', # 1 compound
'G166-49', # multiple, range of sizes, most partial
'G56-11', # obvious but low significance
'3133205855410855589',
'2908236980274858545', # obvious but low significance
'2883924579161475595', # obvious but short
'2431594291587455831',
]
mdwarfs = pd.read_csv('mdwarfs.csv',index_col=0)

def checkplot(csvfile,nsigma=3,pprow=5,minbins=5,figout=None,cleanup=False,
              makenuv=False,outdir='../plots/'):
    if not os.path.exists(csvfile):
        print('No data: {f}'.format(f=csvfile))
        return False
    out = read_lc(csvfile)
    tranges = dt.distinct_tranges(np.sort(out['t0']),maxgap=100)
    good_tranges=[]
    for trange in tranges: # Pre-determine the number of plots needed.
        ix = np.where((np.array(out['t0'])>=trange[0]) &
            (np.array(out['t1'])<=trange[1]) & (np.array(out['flags'])==0))
        if len(ix[0])<minbins:
            continue # Not enough data in this visit to do much with right now.
        good_tranges+=[trange]
    if len(good_tranges)==0:
        print('No visits longer than {m} bins.'.format(m=minbins))
        return
    figdims=(int(np.ceil(len(good_tranges)/pprow)),
    pprow if len(good_tranges)>pprow else len(good_tranges))
    plt.figure(figsize=(figdims[1]*3,figdims[0]*3))
    for i,trange in enumerate(good_tranges):
        plt.subplot(figdims[0],figdims[1],i+1)
        if (makenuv & ('NUV' in csvfile) &
                        os.path.exists(csvfile.replace('NUV','FUV'))):
            fuv = read_lc(csvfile.replace('NUV','FUV'))
            if len(ix[0])>0:
                fix = np.where((np.array(fuv['t0'])>=trange[0]) &
                               (np.array(fuv['t1'])<=trange[1]))
                plt.errorbar(fuv.ix[fix].t_mean,fuv.ix[fix].cps_bgsub,
                    yerr=nsigma*fuv.ix[fix].cps_bgsub_err,fmt='bx')
                plt.plot(fuv.ix[fix].ix[fuv.flags>0].t_mean,
                    fuv.ix[fix].ix[fuv.flags>0].cps_bgsub,'ro')
        ix = np.where((np.array(out['t0'])>=trange[0]) &
                      (np.array(out['t1'])<=trange[1]))
        plt.errorbar(out.ix[ix].t_mean,out.ix[ix].cps_bgsub,
                        yerr=nsigma*out.ix[ix].cps_bgsub_err,fmt='kx')
        plt.plot(out.ix[ix].ix[out.flags>0].t_mean,
                 out.ix[ix].ix[out.flags>0].cps_bgsub,'ro')
# Obfuscate the times...
#        plt.errorbar(
#            30*np.arange(len(np.array(out.ix[ix].t_mean))),
#            out.ix[ix].cps_bgsub,
#                        yerr=nsigma*out.ix[ix].cps_bgsub_err,fmt='kx-')
        #plt.plot(np.arange(len(np.array(out.ix[ix].ix[out.flags>0].t_mean))),
        #         out.ix[ix].ix[out.flags>0].cps_bgsub,'ro')
        plt.xlabel('seconds')
        plt.ylabel('counts per second')
        plt.hlines(out.cps_bgsub.median(),trange[0],trange[1])#0,len(np.array(out.ix[ix].t_mean))*30)
        plt.xlim([trange[0],trange[1]])
        #plt.xlim([0,len(np.array(out.ix[ix].t_mean))*30])
        plt.ylim([max(min(out.cps_bgsub-nsigma*out.cps_bgsub_err),-.2e-15),
                  max(out.cps_bgsub+nsigma*out.cps_bgsub_err)])
        #plt.ylim([0,2])
    plt.suptitle(csvfile.split('/')[-1])
    plt.tight_layout()
    if figout:
        plt.savefig('{d}{f}.png'.format(d=outdir,
                                        f=csvfile.split('/')[-1][:-4]))
    if cleanup:
        plt.close('all')
    return

for name in flares_table:
    print(name)
    star = mdwarfs[mdwarfs.name==name]
    catpath = {'Jones & West - DR7':'dr7', 'Jones & West - PMSU':'pmsu',
               'Lepine & Gaidos':'lepinegaidos',
               'Miles&Shkolnik2017':'miles2017',
               'Shkolnik2010':'shkolnik2010',
               'Shkolnik2014':'shkolnik2014'}[star.catalog.values[0]]
    filename=star.name.values[0].replace(
                            ' ','_').replace('(','').replace(')','')
    path = '{lcpath}/{catpath}/{filename}/{filename}_NUV.csv'.format(
        lcpath='../lightcurves/mdwarfs.nope',catpath=catpath,filename=filename)
    if not os.path.exists(path):
        print('No file: {p}'.format(p=path))
        continue
    distance = star.distance.values[0]
    skypos = star.RA.values[0],star.dec.values[0]
    data = read_lc(path)
    mad_rel=compute_mad(data)
    ts = get_variable_ts(data)
    if not ts.any():
        continue
    #gf = gFind(skypos=skypos,quiet=True)
    tranges = dt.distinct_tranges(np.sort(ts),maxgap=100)
    #tranges = get_tranges(ts,gf,band=band)
    checkplot(path,nsigma=1,pprow=5,minbins=5,figout=True,cleanup=True,
                outdir='./')
    for trange in tranges:
        energy = calculate_energy(data,[trange[0],trange[1]],
                                                distance,filtercorr=False)
        if energy==0.0:
            continue
        print('   ',energy)
