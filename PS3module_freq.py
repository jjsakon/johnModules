import pandas as pd
import numpy as np
from cmlreaders import CMLReader, get_data_index
import sys
import os
import matplotlib.pyplot as plt
from copy import copy
import functools
import datetime
import scipy

from ptsa.data.filters import morlet
from ptsa.data.filters import ButterworthFilter
from general import *


def Log(s, logname):
    date = datetime.datetime.now().strftime('%F_%H-%M-%S')
    output = date + ': ' + str(s)
    with open(logname, 'a') as logfile:
        print(output)
        logfile.write(output+'\n')

def LogDFExceptionLine(row, e, logname):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    line_num = exc_tb.tb_lineno
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    
    rd = row._asdict()
    Log('DF Exception: Sub: '+str(rd['subject'])+', Sess: '+str(rd['session'])+\
        ', '+e.__class__.__name__+', '+str(e)+', file: '+fname+', line no: '+str(line_num), logname)
    
def LogDFException(row, e, logname):
    rd = row._asdict()
    Log('DF Exception: Sub: '+str(rd['subject'])+', Exp: '+str(rd['experiment'])+', Sess: '+\
        str(rd['session'])+', '+e.__class__.__name__+', '+str(e), logname)
    
def LogException(e, logname):
    Log(e.__class__.__name__+', '+str(e)+'\n'+
        ''.join(traceback.format_exception(type(e), e, e.__traceback__)), logname)    
    
def get_bp_tal_struct(sub, montage, localization):
    
    from ptsa.data.readers import TalReader    
   
    #Get electrode information -- bipolar
    tal_path = '/protocols/r1/subjects/'+sub+'/localizations/'+str(localization)+'/montages/'+str(montage)+'/neuroradiology/current_processed/pairs.json'
    tal_reader = TalReader(filename=tal_path)
    tal_struct = tal_reader.read()
    monopolar_channels = tal_reader.get_monopolar_channels()
    bipolar_pairs = tal_reader.get_bipolar_pairs()
    
    return tal_struct, bipolar_pairs, monopolar_channels

def get_elec_regions(tal_struct):
    
    regs = []    
    atlas_type = []
    for num,e in enumerate(tal_struct['atlases']):
        try:
            if 'stein' in e.dtype.names:
                if (e['stein']['region'] is not None) and (len(e['stein']['region'])>1) and \
                   (e['stein']['region'] not in 'None') and (e['stein']['region'] not in 'nan'):
                    regs.append(e['stein']['region'].lower())
                    atlas_type.append('stein')
                    continue
                else:
                    pass
            if 'das' in e.dtype.names:
                if (e['das']['region'] is not None) and (len(e['das']['region'])>1) and \
                   (e['das']['region'] not in 'None') and (e['das']['region'] not in 'nan'):
                    regs.append(e['das']['region'].lower())
                    atlas_type.append('das')
                    continue
                else:
                    pass
            if 'ind' in e.dtype.names:
                if (e['ind']['region'] is not None) and (len(e['ind']['region'])>1) and \
                   (e['ind']['region'] not in 'None') and (e['ind']['region'] not in 'nan'):
                    regs.append(e['ind']['region'].lower())
                    atlas_type.append('ind')
                    continue
                else:
                    pass
            if 'dk' in e.dtype.names:
                if (e['dk']['region'] is not None) and (len(e['dk']['region'])>1):
                    regs.append(e['dk']['region'].lower())
                    atlas_type.append('dk')
                    continue
                else:
                    pass                
            if 'wb' in e.dtype.names:
                if (e['wb']['region'] is not None) and (len(e['wb']['region'])>1):
                    regs.append(e['wb']['region'].lower())
                    atlas_type.append('wb')
                    continue
                else:
                    pass
            else:                
                regs.append('')
                atlas_type.append('No atlas')
        except AttributeError:
            regs.append('')
            
    return np.array(regs),np.array(atlas_type)

def findStimbp(evs_on,sub,session,tal_struct,exp):
    stim_pair = [evs_on.iloc[0].stim_params[0]['anode_number'],evs_on.iloc[0].stim_params[0]['cathode_number']]
    # check if anode/cathode change
    stim_electrode_change = 0
    for i in range(evs_on.shape[0]):
        test_pair = [evs_on.iloc[i].stim_params[0]['anode_number'],evs_on.iloc[i].stim_params[0]['cathode_number']]
        if stim_pair != test_pair:
            stim_electrode_change = 1
            print('The anode/cathode pair changed during this session!! Task: '+exp+
                  ', '+sub+', session: '+str(session))
    chs = [list(temp) for temp in tal_struct['channel']]
    stimbp = np.nan
    for idx,ch_pair in enumerate(chs):
        if stim_pair==ch_pair: # if the anode/cathode pair were recorded from (not always the case!)  
            stimbp = idx
    if np.isnan(stimbp): # if anode/cathode pair don't have a site...just grab first from anode       
        stimbp = findAinB([stim_pair[0]],chs)[0]
        print('stimbp set to: '+str(stimbp)+', since anode/cathode pair not in tal_struct for '+sub
              +', '+str(session))
    return stimbp,stim_electrode_change 

# LocationSearch used like 5-10 sites with consistent parameters. Get all the stim sites in a list
def findStimbpLS(evs_on,sub,session,tal_struct):
    stim_list = []
    for i in range(evs_on.shape[0]):
        stim_list.append([evs_on.iloc[i].stim_params[0]['anode_number'],evs_on.iloc[i].stim_params[0]['cathode_number']])
    all_stim,num_of_each = findUniquePairs(stim_list)
    chs = [list(temp) for temp in tal_struct['channel']]
    stimbp = []
    for j,stim_pair in enumerate(all_stim):
        temp_site = np.nan
        for idx,ch_pair in enumerate(chs):
            if sum(stim_pair==np.array(ch_pair))==2: # if the anode/cathode pair were recorded from (not always the case!)
                temp_site = idx
        if np.isnan(temp_site): # if anode/cathode pair don't have a site...just grab first from anode       
            temp_site = findAinB([stim_pair[0]],chs)[0]
            print('stimbp set to: '+str(temp_site)+', since anode/cathode pair not in tal_struct for '+sub
                  +', '+str(session)+', Stim pair: '+str(j))
        stimbp.append(temp_site)    
    return stimbp 

def get_tal_distmat(tal_struct):
        
    #Get distance matrix
    pos = []
    for ts in tal_struct:
        x = ts['atlases']['ind']['x']
        y = ts['atlases']['ind']['y']
        z = ts['atlases']['ind']['z']
        pos.append((x, y, z))
    pos = np.array(pos)
    dist_mat = np.empty((len(pos), len(pos))) # iterate over electrode pairs and build the adjacency matrix
    dist_mat.fill(np.nan)
    for i, e1 in enumerate(pos):
        for j, e2 in enumerate(pos):
            if (i <= j):
                dist_mat[i,j] = np.linalg.norm(e1 - e2, axis=0)
                dist_mat[j,i] = np.linalg.norm(e1 - e2, axis=0)    
    distmat = 1./np.exp(dist_mat/120.)
    
    return distmat  
    
def get_multitaper_power(eegs, time, freqs):
    # note: must be in ptsa format!!
    from ptsa.data.timeseries import TimeSeriesX
    import mne
    
    sr = int(eegs.samplerate) #get samplerate
    
    #Get multitaper power for all channels first
    eegs = eegs[:, :, :].transpose('event', 'channel', 'time')
    
    time = [x/1000 for x in time] # convert to s for MNE
    clips = np.array(eegs[:, :, int(sr*time[0]):int(sr*time[1])])

    #Get powers
    mne_evs = np.empty([clips.shape[0], 3]).astype(int)
    mne_evs[:, 0] = np.arange(clips.shape[0]) # at each timepoint
    mne_evs[:, 1] = clips.shape[2] # 0
    mne_evs[:, 2] = list(np.zeros(clips.shape[0]))
    event_id = dict(resting=0)
    tmin=0.0
    info = mne.create_info([str(i) for i in range(eegs.shape[1])], sr, ch_types='eeg')       

    arr = mne.EpochsArray(np.array(clips), info, mne_evs, tmin, event_id) 

    #Use MNE for multitaper power
    pows, fdone = mne.time_frequency.psd_multitaper(arr, fmin=freqs[0], fmax=freqs[-1], tmin=0.0,
                                                       verbose=False);

    pows = np.mean(np.log10(pows), 2) #will be shaped n_epochs, n_channels
    
    return pows,fdone # powers and freq. done

def get_tfr_multitaper_power(eegs, freqs, n_cycles, TBW, time):
    # input PTSA format
    from ptsa.data.timeseries import TimeSeriesX
    import mne

    time_range = time # how long is eeg?
    arr = ptsa_to_mne(eegs,time_range)
    # this multitaper program allows you to smooth via cycles at each frequency
    pows = mne.time_frequency.tfr_multitaper(arr, freqs=freqs, n_cycles=n_cycles,
                           time_bandwidth=TBW, return_itc=False, average=False)
    pows = np.mean(np.mean(np.log10(pows.data),2),2) # average across freqs and times to get n_epochs X chs
    return pows

def find_sat_events(eegs,acceptable_saturations):
    #Return array of chans x events with 1s where saturation is found   
    
    def zero_runs(a): # this finds strings of zero differences...which is indicative of eeg at ceiling
        a = np.array(a)
        # Create an array that is 1 where a is 0, and pad each end with an extra 0.
        iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        # Runs start and end where absdiff is 1.
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        return ranges
    
    sat_events = np.zeros([eegs.shape[0], eegs.shape[1]])
    
    zero_ct = 0
    for i in range(eegs.shape[0]):
        for j in range(eegs.shape[1]):
            ts = eegs[i, j]
            zr = zero_runs(np.diff(np.array(ts)))
            numzeros = zr[:, 1]-zr[:, 0]
            if np.sum(ts)==0:
                zero_ct+=1
            if (numzeros>acceptable_saturations).any():
                sat_events[i, j] = 1
                continue
    print('Number of electrodes X events with all 0s: '+str(zero_ct))    
    print('Number of saturated electrodes X events: '+str(np.sum(sat_events)))
    return sat_events.astype(bool)

# from mne_pipeline_refactored
def exclude_bad(s, montage, just_bad=None):
    from glob import glob
    try:
#         # shouldn't actually be any different if montage isn't 0... so remove this
#         if montage!=0:  
#             fn = 'home1/john/data/eeg/electrode_categories/electrode_categories_'+s+'_'+str(montage)+'.txt'
#             # copied this over from:
#             #fn = '/scratch/pwanda/electrode_categories/electrode_categories_'+s+'_'+str(montage)+'.txt'
#         else:
        if len(glob('/data/eeg/'+s+'/docs/electrode_categories.txt'))>0:
            fn = '/data/eeg/'+s+'/docs/electrode_categories.txt'
        else:
            print("Didn't find electrode_categories.txt in data/eeg...so find in folder stolen from Paul")
            if len(glob('/home1/john/data/eeg/electrode_categories/electrode_categories_'+s+'.txt'))>0:
                fn = '/home1/john/data/eeg/electrode_categories/electrode_categories_'+s+'.txt'
                #fn = '/scratch/pwanda/electrode_categories/electrode_categories_'+s+'.txt'
            elif len(glob('/home1/john/data/eeg/electrode_categories/'+s+'_electrode_categories.txt'))>0:
                fn = '/home1/john/data/eeg/electrode_categories/'+s+'_electrode_categories.txt'
                #fn = '/scratch/pwanda/electrode_categories/'+s+'_electrode_categories.txt'
            else:
                print("Didn't find any electrode_categories! Even in john/data/eeg/electrode_categories folder")

        with open(fn, 'r') as fh:
            lines = [mystr.replace('\n', '') for mystr in fh.readlines()]
    except:
        print("Didn't load montage file correctly from exclude_bad for subject "+s+', montage = '+str(montage))
        lines = []
        
    if just_bad is True:
        bidx=len(lines)
        try:
            bidx = [s.lower().replace(':', '').strip() for s in lines].index('bad electrodes')
        except:
            try:
                bidx = [s.lower().replace(':', '').strip() for s in lines].index('broken leads')
            except:
                lines = []
        lines = lines[bidx:]
    
    return lines
    
def artifactExclusion(eegs_pre,eegs_post):
    #Return p-values indicating channels with significant artifact: t-test and levene variance test
    
    from scipy.stats import ttest_rel
    from ptsa.data.timeseries import TimeSeriesX
    from scipy.stats import levene
    
    def justfinites(arr):
        return arr[np.isfinite(arr)]

    sr = int(eegs_pre.samplerate)
    
    pvals = []; lev_pvals = [];
    for i in range(eegs_pre.shape[1]): # across channels
        ts_pre = eegs_pre[:, i, :]
        ts_post = eegs_post[:, i, :]
        pre_eeg = np.mean(ts_pre[:, int(sr*0.55):int(sr*0.9)], 1) # this is out of -950:-50 eeg before stim_on
        # was taking -550:-150 before and 150:550 after la Mohan et al. 
        # If you look at code Ethan does -400:-50 and 50:400 tho
        post_eeg = np.mean(ts_post[:, int(sr*0.0):int(sr*0.35)], 1) # this is out of 50:950 after stim_off 
        eeg_t_chan, eeg_p_chan = ttest_rel(post_eeg, pre_eeg, nan_policy='omit')
        pvals.append(eeg_p_chan)
        try:
            lev_t, lev_p = levene(justfinites(post_eeg), justfinites(pre_eeg))
            lev_pvals.append(lev_p)
        except:
            lev_pvals.append(0.0)
        
    return np.array(pvals), np.array(lev_pvals)   

def get_wm_dist(s, tal_struct, stimbp):
    import nibabel
    
    #Get distance to nearest white matter
    coordsR, faces = nibabel.freesurfer.io.read_geometry('/data/eeg/freesurfer/subjects/'+s+'/surf/rh.white')
    coordsL, faces = nibabel.freesurfer.io.read_geometry('/data/eeg/freesurfer/subjects/'+s+'/surf/lh.white')
    coords = np.vstack([coordsR, coordsL])
    
    xyz = [tal_struct[stimbp]['atlases']['ind']['x'], tal_struct[stimbp]['atlases']['ind']['y'], tal_struct[stimbp]['atlases']['ind']['z']]

    coords_diff = np.array([coords[:, 0]-xyz[0], coords[:, 1]-xyz[1], coords[:, 2]-xyz[2]]).T
    coords_dist = np.sqrt(np.sum(coords_diff**2, 1))
    dist_nearest_wm = np.min(coords_dist)
    
    return dist_nearest_wm

def resid_adjmat(distmat, conn):
    from sklearn.linear_model import LinearRegression
    from scipy.special import logit

    finite_idxs = np.where(np.logical_and(np.isfinite(distmat), np.isfinite(conn))) # conn is already logit'd
    X = distmat[finite_idxs]
    y = conn[finite_idxs]

    #Fit the model
    mdl = LinearRegression(fit_intercept=True, normalize=False)
    X = X[:, np.newaxis]
    mdl.fit(X, y);

    #Get residuals
    preds = mdl.predict(X)
    resid_fc = y-preds
    
    resid_conn = np.empty(conn.shape); resid_conn[:] = np.nan
    resid_conn[finite_idxs] = resid_fc

    return resid_conn # return logit transformed coherences for all channels

def resid_fc(distmat, conn, stimbp):
    from sklearn.linear_model import LinearRegression
    from scipy.special import logit
    
    #Residualize functional connectivity with distance
    X = distmat[stimbp] #distances should already be exponential/normalized
    y = conn[stimbp]

    #Fit the model
    mdl = LinearRegression(fit_intercept=True, normalize=False)
    X = X[np.isfinite(y), np.newaxis]
    mdl.fit(X, y[np.isfinite(y)]);

    #Get residuals
    preds = mdl.predict(X)
    preds = np.insert(preds, stimbp, np.nan) # need to put a nan in at stim channel 
    resid_fc = y-preds
    
    return resid_fc #return residuals of fc after regressing out distance

def residTstat(distmat, sess_Ts):
    from sklearn.linear_model import LinearRegression
    from scipy.special import logit
    
    #Residualize functional connectivity with distance
    X = distmat #distances should already be exponential/normalized
    y = sess_Ts

    #Fit the model
    mdl = LinearRegression(fit_intercept=True, normalize=False)
    X = X[np.isfinite(y), np.newaxis]
    y = y[np.isfinite(y)]
    mdl.fit(X, y);

    #Get residuals
    preds = mdl.predict(X)     
    resid_stim = y-preds
    
    return resid_stim #return logit transformed coherences but only for stim channel

def StartFig():
    test = plt.figure();
    plt.rcParams.update({'font.size':14});
    return test;

def PrintTest():
    print('testttt')
    
def SaveFig(basename):
    plt.savefig(basename+'.png')
    plt.savefig(basename+'.pdf')
    print('Saved .png and .pdf')

def SubjectDataFrames(sub_list):
    if isinstance(sub_list, str):
        sub_list = [sub_list]
    
    df = get_data_index('all')
    indices_list = [df['subject']==sub for sub in sub_list]
    indices = functools.reduce(lambda x,y: x|y, indices_list)
    df_matched = df[indices]
    return df_matched

def CMLReadDFRow(row):
    '''for row in df.itertuples():
            reader = CMLReadDFRow(row)
    '''
    rd = row._asdict() # this takes df and takes values from 1 row as a dict
    return CMLReader(rd['subject'], rd['experiment'], rd['session'], \
                     montage=rd['montage'], localization=rd['localization'])
    # dirty secret: Readers needs: eegoffset, experiment, subject, and eegfile...but really should
    # pass in sessions since sampling rate could theoretically change...

def GetElectrodes(sub,start,stop):
    df_sub = SubjectDataFrames(sub)
    reader = CMLReadDFRow(next(df_sub.itertuples()))
    evs = reader.load('events')
    enc_evs = evs[evs.type=='WORD']
    eeg = reader.load_eeg(events=enc_evs, rel_start=start, rel_stop=stop, clean=True)
    return eeg.to_ptsa().channel.values

def get_resting_events(evs,fmin,fmax):
    from copy import copy
    import mne
    #Use the FR countdown period to establish baseline events, spaced every second
    
    # need to load reader in here separately for each task
    exp_list = evs.experiment.unique()
    accum_eeg = None    
    for exp in exp_list:
        exp_evs = evs[evs['experiment']==exp]
        sess_list = exp_evs.session.unique()
        for sess in sess_list: # gotta do this since it's POSSIBLE sampling rate could change
            # note: montage and localization never change within session (I don't think!)            
            reader = CMLReader(exp_evs.subject.iloc[0], exp, sess, 
                               montage=exp_evs.montage.iloc[0], localization=exp_evs.localization.iloc[0])

            #Get samplerate to do eegoffsets
            init_sr = int(reader.load("sources")['sample_rate'])
            
            orig_evs = exp_evs[(exp_evs['type']=='COUNTDOWN_START') & (exp_evs['session']==sess)]
            rest_evs = copy(orig_evs) # store initial one then append next 9, 1 s chunks to it
            for i in range(1, 10): 
                rest_copy = copy(orig_evs)
                rest_copy.eegoffset = rest_copy.eegoffset+init_sr*i
                rest_evs = rest_evs.append(rest_copy, ignore_index=True)    

            #Use MNE to get connectivity for all 10, 1 s resting events during countdowns
            pairs = reader.load('pairs') # voltages across adjacent contacts
            try:  #some bipolar ENS subjects will have nonmatching EEG and pairs information
                eeg = reader.load_eeg(events=rest_evs, rel_start=0, rel_stop=1000, scheme=pairs)  
                eeg = eeg.to_mne()
                print('Loaded eeg for '+str(len(eeg))+' events for Subject '+str(exp_evs.subject.iloc[0])+', Experiment '+exp+', Session '+str(sess))
                if accum_eeg is None:
                    accum_eeg = eeg
                elif accum_eeg is not None:
                    accum_eeg = mne.concatenate_epochs([accum_eeg,eeg])
            except:
                try: 
                    eeg = reader.load_eeg(events=rest_evs, rel_start=0, rel_stop=1000)
                    eeg = eeg.to_mne()
                    if len(pairs)!=eeg.shape[1]:                        
                        raise ValueError('pairs.json and loaded EEG do not match! Probably should not use.')
                except:
                    print(str(exp_evs.subject.iloc[0])+', Experiment '+exp+', Session '+str(sess)+
                                     ": from PS3mod split EEG filenames don't seem to match what are in the events")
                    pass
    
    from mne.connectivity import spectral_connectivity
    method = 'coh'
    mode = 'multitaper'
    #time_bandwidth_product = 4
    cons, freqs, times, n_epochs, n_tapers = spectral_connectivity(
        accum_eeg, method=method, mode=mode, sfreq=init_sr, fmin=fmin, fmax=fmax,
        faverage=True, tmin=0.0, mt_adaptive=False, n_jobs=1, verbose=False) #,
        #mt_bandwidth=time_bandwidth_product)
    
    #Symmetrize and save average network
    mu = np.mean(cons, 2)
    mu_full = np.nansum(np.array([mu, mu.T]), 0)
    mu_full[np.diag_indices_from(mu_full)] = 0.0
    #np.save(self.root+''+self.s+'/'+self.s+'_baseline10trials_network_'+str(self.band)+'.npy', mu_full)
        
    return mu_full

def MakeLocationFilter(scheme, location):
    return [location in s for s in [s if s else '' for s in scheme.iloc()[:]['ind.region']]]

class Settings():
    '''settings = Settings()
      settings.somelist = [1, 2, 3]
      settings.importantstring = 'saveme'
      settings.Save()
      settings = Settings.Load()
    '''
    def Save(self, filename='settings.pkl'):
        with open(filename, 'wb') as fw:
            fw.write(pickle.dumps(self))

    def Load(filename='settings.pkl'):
        return pickle.load(open(filename, 'rb'))

def ClusterRun(function, parameter_list, max_cores=100):
    '''function: The routine run in parallel, which must contain all necessary
       imports internally.
    
       parameter_list: should be an iterable of elements, for which each element
       will be passed as the parameter to function for each parallel execution.
       
       max_cores: Standard Rhino cluster etiquette is to stay within 100 cores
       at a time.  Please ask for permission before using more.
       
       In jupyterlab, the number of engines reported as initially running may
       be smaller than the number actually running.  Check usage from an ssh
       terminal using:  qstat -f | egrep "$USER|node" | less
       
       Undesired running jobs can be killed by reading the JOBID at the left
       of that qstat command, then doing:  qdel JOBID
    '''
    import cluster_helper.cluster
    from pathlib import Path

    num_cores = len(parameter_list)
    num_cores = min(num_cores, max_cores)

    myhomedir = str(Path.home())

    with cluster_helper.cluster.cluster_view(scheduler="sge", queue="RAM.q", \
        num_jobs=num_cores, cores_per_job=1, \
        extra_params={'resources':'pename=python-round-robin'}, \
        profile=myhomedir + '/.ipython/') \
        as view:
        # 'map' applies a function to each value within an interable
        res = view.map(function, parameter_list)
        
    return res

def PowerSpectra(subjects, electrodes, freqs, avg_ref=False, zscore=False, \
        bin_elecs=True, internal_bipolar=False, elec_masks=False, debug=False):
    buf_ms = 1000
    start_time = 0
    end_time = 1600
    morlet_reps = 6
    rec_results = []
    nrec_results = []

    is_string = isinstance(subjects,str)
    if is_string: # if just one string
        subjects = [subjects]
    #import ipdb; ipdb.set_trace()    
    for sub in subjects: # for each subject
        df_sub = SubjectDataFrames(sub) # get their dataframe
        sub_rec_powers = np.zeros(len(freqs))
        sub_nrec_powers = np.zeros(len(freqs))
        first_run = True
        first_channel_flags = None
        num_channels_found = 0
        df_per_sub = 0
        mask_index = -1
        
        for row in df_sub.itertuples(): # for each row (session) in dataframe
            mask_index += 1
            try:
                reader = CMLReadDFRow(row) # read this session
                # This does not work for this data set,
                # so we will get these from load_eeg(...).to_ptsa() later.
                # contacts = reader.load('contacts')
                evs = reader.load('events')
                enc_evs = evs[evs.type=='WORD']               
                if np.sum(enc_evs.recalled == True) == 0:
                    raise IndexError('No recalled events')
                if np.sum(enc_evs.recalled == False) == 0:
                    raise IndexError('No non-recalled events')

                # clean=True for Localized Component Filtering (LCF)
                eeg = reader.load_eeg(events=enc_evs, rel_start= start_time - buf_ms, \
                    rel_stop= end_time + buf_ms, clean=True)
                
                if len(eeg.events) != enc_evs.shape[0]:
                    raise IndexError(str(len(eeg.events)) + ' eeg events for ' + \
                                     str(enc_evs.shape[0]) + ' encoding events')
  
                # added from Assignment2  
                if(evs['eegoffset'].max()<0):
                    print('### Had max eegoffset value of: ' + str(evs['eegoffset'].max()) +  
                        ' in row: ' + str(row))
                    continue   
#                 if eeg.samplerate != 500:
#                     print('### Had to resample values in session: ' + str(sess))
#                     eeg = eeg.resampled(500) # I guess this resamples??                
                if avg_ref == True:
                    # Reference to average
                    avg_ref_data = np.mean(eeg.data, (1))
                    for i in range(eeg.data.shape[1]):
                        eeg.data[:,i,:] = eeg.data[:,i,:] - avg_ref_data                
                if internal_bipolar == True:
                    # Bipolar reference to nearest labeled electrode
                    eeg.data -= np.roll(eeg.data, 1, 1)

                sr = eeg.samplerate
                # print('sampling rate (Hz)',sr)0
            
                eeg_ptsa = eeg.to_ptsa()                 
                
                if elec_masks:
                    if electrodes[mask_index] is None:
                        raise ValueError('No channel mask available for session ' + \
                                         str(mask_index))
                    if isinstance(electrodes[mask_index], np.ndarray):
                        channel_flags = electrodes[mask_index].tolist()
                    else:
                        channel_flags = electrodes[mask_index]
                else:
                    channels = eeg_ptsa.channel.values
                    channel_flags = [c in electrodes for c in channels] # just puts true for all
                    # all you ever need is the mask to get the channels for the eeg_ptsa to eeg_chan 
                
                if np.sum(channel_flags)==0:
                    if elec_masks:
                        raise IndexError('No channels for region index '+str(mask_index))
                    else:
                        raise IndexError('No matching channels found for '+str(electrodes))
                
                eeg_chan = eeg_ptsa[:,channel_flags,:]
                
                freq_range = [58., 62.]
                b_filter = ButterworthFilter(timeseries=eeg_chan, freq_range=freq_range, filt_type='stop', order=4)
                eeg_filtered = b_filter.filter()
            
                wf = morlet.MorletWaveletFilter(timeseries=eeg_filtered, freqs=freqs, \
                    width=morlet_reps, output=['power'], complete=True)
                powers_plusbuf = wf.filter()
                # freqs, events, elecs, and time
                powers = powers_plusbuf[:, :, :, int((buf_ms/1000)*sr):-1*int((buf_ms/1000)*sr)]
                
                # Average over time
                powers = np.mean(powers, (3))
                
                # Across events
                if zscore:
                    powers = scipy.stats.zscore(powers, 1, ddof=1)
                
                if bin_elecs:
                    rec_powers = np.mean(powers[:,enc_evs.recalled == True,:].data, (1,2))
                    nrec_powers = np.mean(powers[:,enc_evs.recalled == False,:].data, (1,2))
                else:
                    if first_run==True:
                        first_run = False
                        first_channel_flags = channel_flags
                        num_channels_found = powers.shape[2]
                        sub_rec_powers = np.zeros((powers.shape[0], powers.shape[2]))
                        sub_nrec_powers = np.zeros((powers.shape[0], powers.shape[2]))
                    else:
                        if np.any(first_channel_flags != channel_flags):
                            raise IndexError('Mismatched electrodes for subject')
                        if num_channels_found != powers.shape[2]:
                            raise IndexError('Inconsistent number of electrodes found')
                    
                    rec_powers = np.mean(powers[:,enc_evs.recalled == True,:].data, (1))
                    nrec_powers = np.mean(powers[:,enc_evs.recalled == False,:].data, (1))
                
                if np.any(np.isnan(rec_powers)) or np.any(np.isnan(nrec_powers)):
                    print('rec_powers', rec_powers)
                    print('nrec_powers', nrec_powers)
                    raise ValueError('nan values in eeg power')
            
                sub_rec_powers += rec_powers
                sub_nrec_powers += nrec_powers
                df_per_sub += 1
            except Exception as e:   
                exc_type, exc_obj, exc_tb = sys.exc_info()
                line_num = exc_tb.tb_lineno
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                LogDFExceptionLine(row, e, line_num, fname)
                if debug:
                    raise

        print('df_per_sub', df_per_sub)
        sub_rec_powers /= df_per_sub
        sub_nrec_powers /= df_per_sub
        
        rec_results.append(sub_rec_powers)
        nrec_results.append(sub_nrec_powers)
        
        print('eeg appended for subject ' + sub)        
        
    return (freqs, rec_results, nrec_results)

class SubjectStats():
    def __init__(self):
        self.sessions = 0
        self.lists = []
        self.recalled = []
        self.intrusions_prior = []
        self.intrusions_extra = []
        self.repeats = []
        self.num_words_presented = []
    
    def Add(self, evs):
        enc_evs = evs[evs.type=='WORD']
        rec_evs = evs[evs.type=='REC_WORD']
        
        # Trigger exceptions before data collection happens
        enc_evs.recalled
        enc_evs.intrusion
        enc_evs.item_name
        if 'trial' in enc_evs.columns:
            enc_evs.trial
        else:
            enc_evs.list

        self.sessions += 1
        if 'trial' in enc_evs.columns:
            self.lists.append(len(enc_evs.trial.unique()))
        else:
            self.lists.append(len(enc_evs.list.unique()))
        self.recalled.append(sum(enc_evs.recalled))
        self.intrusions_prior.append(sum(rec_evs.intrusion > 0))
        self.intrusions_extra.append(sum(rec_evs.intrusion < 0))
        words = rec_evs.item_name
        self.repeats.append(len(words) - len(words.unique()))
        self.num_words_presented.append(len(enc_evs.item_name))
        
    def ListAvg(self, arr):
        return np.sum(arr)/np.sum(self.lists)
    
    def RecallFraction(self):
        return np.sum(self.recalled)/np.sum(self.num_words_presented)
    
def SubjectStatTable(subjects):
    ''' Prepare LaTeX table of subject stats '''   

    table = ''
    try:
        table += '\\begin{tabular}{lrrrrrr}\n'
        table += ' & '.join('\\textbf{{{0}}}'.format(h) for h in [
            'Subject',
            '\\# Sessions',
            '\\# Lists',
            'Avg Recalled',
            'Prior Intrusions',
            'Extra Intrusions',
            'Repeats'
        ])
        table += ' \\\\\n'

        for sub in subjects:
            df_sub = SubjectDataFrames(sub) # at this level 
            stats = SubjectStats()
            for row in df_sub.itertuples():
                reader = CMLReadDFRow(row) # "row" is whole row of dataframe (1 session in this case)
                evs = reader.load('task_events')
                stats.Add(evs)
            table += ' & '.join([sub, str(stats.sessions)] +
                ['{:.2f}'.format(x) for x in [
                    np.mean(stats.lists),
                    stats.ListAvg(stats.recalled),
                    stats.ListAvg(stats.intrusions_prior),
                    stats.ListAvg(stats.intrusions_extra),
                    stats.ListAvg(stats.repeats)
                ]]) + ' \\\\\n'
        
        table += '\\end{tabular}\n'
    except Exception as e:
        print (table)
        raise
    
    return table    

def get_stim_events(evs,sessions): #sub, task, montage=0):
    
    # this doesn't do anything anymore I think. I do through rows of dataframes so not very useful
    # could maybe check to make sure anode and cathode have same numbers
    
#     from ptsa.data.readers import BaseEventReader
#     from ptsa.data.readers import events
#     from ptsa.data.readers import JsonIndexReader
#     reader = JsonIndexReader('/protocols/r1.json')

#     #Get events
#     evfiles = list(reader.aggregate_values('task_events', subject=sub, experiment=task, montage=montage)) #This is supposed to work but often does not
#     if len(evfiles)<1:
#         from glob import glob
#         sessions = [s.split('/')[-1] for s in glob('/protocols/r1/subjects/'+sub+'/experiments/'+task+'/sessions/*')]
#         evfiles = []
#         for sess in sessions:
#             evfiles.append('/protocols/r1/subjects/'+sub+'/experiments/'+task+'/sessions/'+sess+'/behavioral/current_processed/task_events.json')

#     evs_on = np.array([]); 
#     for ef in evfiles:
#         try:
#             base_e_reader = BaseEventReader(filename=ef, eliminate_events_with_no_eeg=True)
#             base_events = base_e_reader.read()
#             if len(evs_on) == 0:
#                 evs_on = base_events[base_events.type=='STIM_ON']
#             else:
#                 evs_on = np.concatenate((evs_on, base_events[base_events.type=='STIM_ON']), axis=0)
#         except:
#             continue
#     evs_on = Events(evs_on);

    evs_on = evs[evs['type']=='STIM_ON']
    
    #Reorganize sessions around stim sites
    #sessions = reader.sessions(subject=sub, experiment=task, montage=montage)
    sess_tags = []
    for i in range(len(evs_on)):
        stimtag = str(evs_on.iloc[i].stim_params[0]['anode_number'])+'-'+str(evs_on.iloc[i].stim_params[0]['cathode_number'])
        sess_tags.append(stimtag)
    sess_tags = np.array(sess_tags)
    if len(np.unique(sess_tags))<=len([sessions]):
        session_array = evs_on['session']
    else:
        session_array = np.empty(len(evs_on))
        for idx, s in enumerate(np.unique(sess_tags)):
            session_array[sess_tags==s] = int(idx)
    session_array = np.unique(session_array)
    
    return evs_on, session_array

def run_stim_regression(row, MTL_labels, test_freq_range, fmin, fmax, fmin_pow, fmax_pow, stim_param):
    
    import pickle
    contact_array = np.array(['dummy']) # array of all locations
    import statsmodels.api as sm 

    # accumlate stats across sessions
    sess_Ts = []; stimbps = []
    Conn_Zs = []; Conn_Ps = []
    regs = []; tal_structs = []
    subjects = []; sessions = []
    good_chans = []
    
    print('start run')
    
    try:
        # get contacts for each session
        reader = CMLReadDFRow(row)
        contacts = reader.load('contacts') # actual electrodes
        contact_array = np.append(contact_array,contacts['avg.region'].unique()) 

        # get bipolar pairs
        sub = row.subject
        mont = int(row.montage)        
        loc = int(row.localization)
        exp = row.experiment
        session = int(row.session)
        evs = reader.load('events')            

        #evs_on, session_array = get_stim_events(evs,session) #sub, exp, montage=mont)  # this seemed like steps before CMLReaders
        evs_on = evs[evs['type']=='STIM_ON'] #Get events, structured around stim electrodes
        evs_off = evs[evs['type']=='STIM_OFF']

        #check to make sure all events have same stimulation sites
        sess_tags = []
        for i in range(len(evs_on)):
            stimtag = str(evs_on.iloc[i].stim_params[0]['anode_number'])+'-'+str(evs_on.iloc[i].stim_params[0]['cathode_number'])
            sess_tags.append(stimtag)
        sess_tags = np.array(sess_tags)
        if len(np.unique(sess_tags))>1:
            raise('There are '+str(len(np.unique(sess_tags)))+' stimulation pairs in session '+str(session)+' from subject '+sub)

        # should really rewrite this to use "pairs" from CMLReaders instead of tal_struct
        tal_struct, bipolar_pairs, mpchans = get_bp_tal_struct(sub, montage=mont, localization=loc)
        elec_regions,_ = get_elec_regions(tal_struct)      
        distmat = get_tal_distmat(tal_struct) 
        # stimbp is the bipolar PAIR from tal_struct that contains the anode and cathode contacts
        stimbp,_ = findStimbp(evs_on,sub,session,tal_struct,exp)

        # check if stim region is in MTL
        if elec_regions[stimbp] in MTL_labels:
            regs.append(elec_regions[stimbp])
        else:
#             Log('Session '+str(session)+' from subject '+sub+' was not stimulated in MTL!!'+'\n',
#                 'run_stim_regression_log.txt')
            regs.append(elec_regions[stimbp])
            #pass

        # get functional connectivity matrix from resting state data (made in getBaseFxlConn)
        if test_freq_range is True:
            conn_file = os.path.join('/home1/john/data/eeg/PS3_fxl_conn/'+sub,
                                            sub+'_'+exp+'_'+str(fmin)+'-'+str(fmax)+'Hz_10s_countdown_network.p')
            if exp == 'PS2' and sub == 'R1108J' and mont == 0: # mont and loc changed and separate FC for both 
                conn_file = os.path.join('/home1/john/data/eeg/PS3_fxl_conn/'+sub,
                                            sub+'_0_3_'+exp+'_'+str(fmin)+'-'+str(fmax)+'Hz_10s_countdown_network.p')
            elif exp == 'PS2' and sub == 'R1108J' and mont == 1: # note only did PS2 so only need to add to this module
                conn_file = os.path.join('/home1/john/data/eeg/PS3_fxl_conn/'+sub,
                                            sub+'_4_9_'+exp+'_'+str(fmin)+'-'+str(fmax)+'Hz_10s_countdown_network.p')
        else:
            conn_file = os.path.join('/home1/john/data/eeg/PS3_fxl_conn/'+sub,
                                            sub+'_'+exp+'_'+str(fmin)+'-'+str(fmax)+'_network.p')
            if exp == 'PS2' and sub == 'R1108J' and mont == 0: # mont and loc changed and separate FC for both 
                conn_file = os.path.join('/home1/john/data/eeg/PS3_fxl_conn/'+sub,
                                            sub+'_0_3_'+exp+'_'+str(fmin)+'-'+str(fmax)+'_network.p')
            elif exp == 'PS2' and sub == 'R1108J' and mont == 1:   
                conn_file = os.path.join('/home1/john/data/eeg/PS3_fxl_conn/'+sub,
                                            sub+'_4_9_'+exp+'_'+str(fmin)+'-'+str(fmax)+'_network.p') 
        with open(conn_file,'rb') as f:          
            conn,num_10s_events = pickle.load(f)
        from scipy.special import logit    
        conn = logit(conn)
                    # #         import h5py
            # #         f = h5py.File('/data10/scratch/jkragel/HCP/adjacency/'+sub+'_adjacency.hdf5', 'r')
            # #         conn = np.array(f['rsfMRI_zCorr']) # some old version

        ## do t-test between pre and post stimulus for each electrode ##

        # eeg start and end times (in ms)
        pre_start = -950
        pre_end = -50
        post_start = 50
        post_end = 950

        # parameters
        notch_filter = True
        internal_bipolar = True
        baseline_correct = True # done in get_stim_eeg in esolo code
        for stim_type in np.arange(2): 
            if stim_type==0: # stim_on
                start = pre_start
                end = pre_end
                stim_evs = evs[evs['type']=='STIM_ON'] # evs = evs.query("(type=='STIM_ON')")
                label = 'pre'
            elif stim_type==1: # stim_off
                start = post_start
                end = post_end
                stim_evs = evs[evs['type']=='STIM_OFF']   
                label = 'post'

            if internal_bipolar == True:
                # Bipolar reference to nearest labeled electrode
                pairs = reader.load('pairs') # voltages across the adjacent contacts
            else: pairs = None            

            # clean=True for Localized Component Filtering (LCF)
            # reader.load_eeg doesn't seem to like it with < 1 s of data or when 0 isn't included
            eeg = reader.load_eeg(events=stim_evs, rel_start=start, rel_stop=end, clean=True, scheme=pairs)

            if len(eeg.events) != stim_evs.shape[0]:
                raise IndexError(str(len(eeg.events)) + ' eeg events for ' + \
                                str(stim_evs.shape[0]) + ' encoding events')   

            eeg = eeg.to_ptsa() # move to ptsa so can correct baseline and apply filters on timeseries

            if baseline_correct == True:
                eeg = eeg.baseline_corrected((start,end))

            sr = eeg.samplerate # per second

            if notch_filter == True:
                from ptsa.data.filters import ButterworthFilter
                eeg = ButterworthFilter(timeseries=eeg, freq_range=[58.,62.], filt_type='stop', order=4).filter()
                eeg = ButterworthFilter(timeseries=eeg, freq_range=[118.,122.], filt_type='stop', order=4).filter()
                eeg = ButterworthFilter(timeseries=eeg, freq_range=[178.,182.], filt_type='stop', order=4).filter()
            else:
                pass

            # Use MNE to get multitaper spectral power
            eeg_length = end-start
            if fmin_pow==fmax_pow: # if single band, use tfr_multitaper
                TBW = 2
                pows = get_tfr_multitaper_power(eeg, np.array([fmin_pow]), np.array([fmin_pow])/2, TBW, time=[0,eeg_length])
            else:
                pows,freqs_done = get_multitaper_power(eeg, time=[0,eeg_length],freqs = np.array([fmin_pow, fmax_pow])) 
            # returns log(powers) evts X channels averaged over time

            # this finds consecutive timepoints with no change in power. 10 or more of these is enough to remove an event
            # 1600 Hz is samplerate for R1034D at least so at least 6 ms here
            # problem might be if there are a bunch of 0s it's going to flag those too. 
            acceptable_saturations = 9 # more than this # of 0s flagged. 
            sat_events = find_sat_events(eeg,acceptable_saturations)
            good_pows = copy(pows); 
            good_pows[sat_events] = np.nan # .T?
            print('Total electrodes X events: '+str(eeg.shape[0]*eeg.shape[1]))

            if stim_type==0: # stim_on
                eeg_pre = copy(eeg)
                pre_pows = copy(good_pows)
            elif stim_type==1:
                eeg_post = copy(eeg)
                post_pows = copy(good_pows)

#                     ## save files ##
#         try:
#             os.mkdir('compiled/PS3_powers')
#         except FileExistsError as e:
#             pass
#         filename = os.path.join('compiled/PS3_powers', sub+'_'+str(session)+'_'+label+'.p')            
#         with open(filename,'wb') as f:
#             pickle.dump([pows,eeg],f)
# #             np.save(filename,[pows,eeg]) # this takes too long          

        # Get bad electrodes (seizure onset/ictal)
        
        badelecs = exclude_bad(sub, mont, just_bad=False)
        bad_filt = np.zeros(len(tal_struct))
        for idx, e in enumerate(tal_struct):
            tagnames = e['tagName'].split('-')
            if tagnames[0] in badelecs or tagnames[1] in badelecs:
                bad_filt[idx] = 1
        
        #Identify channels to exclude due to decay artifact
        pvals, lev_pvals = artifactExclusion(eeg_pre,eeg_post)
        
        # can set a number. Here just grabbing all unique amplitudes:
        desired_amps = {dlist[0]['amplitude'] for dlist in evs_on.stim_params.to_list()} # {750} this was for PS2 to test Ethan Fig. 2 example
        ## FOR PS3 ONLY ##...NOTE YOU HAVE TO CHANGE "PULSE" TO "BURST" in pulse_freq and desired_pulses below
        desired_bursts = {stim_param} #{dlist[0]['burst_freq'] for dlist in evs_on.stim_params.to_list()} # PS3 has 3:8 Hz burst_freq
#         desired_pulses = {stim_param} # {dlist[0]['pulse_freq'] for dlist in evs_on.stim_params.to_list()} [100,200] for PS3
        
        good_pre = np.zeros(evs_on.shape[0]); 
        for i,row in enumerate(evs_on.itertuples()): 
            if row.stim_params[0]['amplitude'] in desired_amps and row.stim_params[0]['burst_freq'] in desired_bursts:
                good_pre[i] = True
        good_post = np.zeros(evs_off.shape[0])        
        for i,row in enumerate(evs_off.itertuples()): 
            if row.stim_params[0]['amplitude'] in desired_amps and row.stim_params[0]['burst_freq'] in desired_bursts:
                good_post[i] = True 
        good_trials = np.logical_and(good_pre,good_post) # there's no reason these should differ, 
                                            #but just in case (since ttest_rel needs equal shape)        
        #T-test post vs. pre powers
        from scipy.stats import ttest_rel
        # mytrials = np.where(good_trials)[0] #mytrials = np.random.choice(mytrials, 50)
        
        # t-test values for all electrodes between pre/post for each trial
        chan_T, p = ttest_rel(post_pows[good_trials==1, :], pre_pows[good_trials==1, :], axis=0, nan_policy='omit') 
        orig_T_stats = copy(chan_T) # for Fig. 2b
        chan_T[pvals<0.01] = np.nan # remove post-stim decay artifacts
        chan_T[lev_pvals<0.01] = np.nan 
        chan_T[bad_filt==1] = np.nan # remove bad electrodes (SOZ/ictal)
        # for Fig. 2b, label channels as 0) good 1) SOZ/ictal 2) artifactual
        electrode_indicator = np.zeros(len(chan_T))
        electrode_indicator[pvals<0.01] = 2 # remove post-stim decay artifacts
        electrode_indicator[lev_pvals<0.01] = 2 
        electrode_indicator[bad_filt==1] = 1 # remove bad electrodes (SOZ/ictal)       
#         if session == 0:
#             print('stopped here to print figures')
#             break 

        #Dont include stim channels (only those with anode)
        stim_channels = tal_struct[stimbp]['channel']
        for idx, i in enumerate(tal_struct['channel']):
            if (stim_channels[0] in i) or (stim_channels[1] in i):
                chan_T[idx] = np.nan

        # remove contacts if no connectivity was computed with stim channel
        chan_T[~np.isfinite(conn[stimbp, :])] = np.nan  

        sess_Ts.append(copy(chan_T))
        stimbps.append(stimbp)
        tal_structs.append(tal_struct)
        subjects.append(sub); sessions.append(session)
        print('For sub '+sub+', session '+str(session)+', kept '+
              str(len(chan_T)-np.sum(np.isnan(chan_T)))+'/'+str(len(chan_T))+' channels (pairs)')

        ### DO REGRESSION ###

        print('Regression for session '+str(session))

        if np.sum(np.abs(chan_T)>10)>0:
            print('Warning! High T-stats detected!')
            #chan_T[np.abs(chan_T)>10] = np.nan

        #Set up feature matrices
        chan_T = np.array(chan_T)
        good_chan = ~np.isnan(chan_T)
        X = np.empty([np.sum(good_chan), 3])
        # this is logit(y) = B0 + B1*conn + B2*dist
        X[:, 2] = distmat[stimbp][good_chan]
        X[:, 1] = conn[stimbp][good_chan]
        X[:, 0] = np.ones(np.sum(good_chan))
        y = chan_T[good_chan]        

        if y.size < 20:
            print('Warning: <20 electrodes left!')

        #Fit the model
        result = sm.OLS(y, X).fit()
        tru_coefs = copy(result.params)

        #Shuffle for null coefficients
        # seems to me like a better null would be to just not use stimbp above and test diff channels
        null_coefs = []
        shuf_idxs = np.arange(X.shape[0])
        for k in range(1000):
            np.random.shuffle(shuf_idxs)
            null_result = sm.OLS(y, X[shuf_idxs, :]).fit()
            null_coefs.append(null_result.params)
        null_coefs = np.array(null_coefs)

        #Z-score and p-value true vs. null coefs
        coef_stats = np.empty([len(tru_coefs), 2])
        for idx, i in enumerate(tru_coefs):
            coef_stats[idx, 0] = (i-np.nanmean(null_coefs[:, idx]))/np.nanstd(null_coefs[:, idx]) #z-score
            coef_stats[idx, 1] = np.sum(null_coefs[:, idx]>i)/float(null_coefs.shape[0]) # empirical p-value

        # Z scores and p-values for connectivity after regressing out effect of distance
        Conn_Zs.append(coef_stats[1, 0]) # this is NMA(theta)!!
        Conn_Ps.append(coef_stats[1, 1])
        good_chans.append(good_chan)    

    except Exception as e:
        LogDFExceptionLine(row, e, 'run_stim_regression_log.txt')

    try:
        os.mkdir(sub) #os.mkdir('compiled/PS3_NMA/'+sub)
    except FileExistsError as e:
        pass

    fn = os.path.join(sub, #'compiled/PS3_NMA/'+sub,
                      sub+'_exp-'+exp+'_session-'+str(session)+'_'+str(fmin)+'-'+str(fmax)+'-fc_'+
                      str(fmin_pow)+'-'+str(fmax_pow)+'-pow_burst-'+str(stim_param)+'.p')   # pow_pulse-         

    with open(fn,'wb') as f:
        pickle.dump({'sess_Ts':sess_Ts, 'stimbps':stimbps, 'Conn_Zs': Conn_Zs, 'Conn_Ps': Conn_Ps, 
                     'sessions':sessions, 'tal_structs': tal_structs, 'subjects':subjects,
                     'good_chans':good_chans, 'regs':regs, 'electrode_indicator':electrode_indicator,
                     'orig_T_stats':orig_T_stats}, f)
    return