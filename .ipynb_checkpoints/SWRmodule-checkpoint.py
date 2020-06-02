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
    rd = row._asdict()
    if type(e) is str: # if it's just a string then this was not an Exception I just wanted to print my own error
        Log('DF Exception: Sub: '+str(rd['subject'])+', Sess: '+str(rd['session'])+\
        ', Manual error, '+e+', file: , line no: XXX', logname)
    else: # if e is an exception then normal print to .txt log
        exc_type, exc_obj, exc_tb = sys.exc_info()
        line_num = exc_tb.tb_lineno
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        Log('DF Exception: Sub: '+str(rd['subject'])+', Sess: '+str(rd['session'])+\
        ', '+e.__class__.__name__+', '+str(e)+', file: '+fname+', line no: '+str(line_num), logname)
    
def LogDFException(row, e, logname):
    rd = row._asdict()
    Log('DF Exception: Sub: '+str(rd['subject'])+', Exp: '+str(rd['experiment'])+', Sess: '+\
        str(rd['session'])+', '+e.__class__.__name__+', '+str(e), logname)
    
def LogException(e, logname):
    Log(e.__class__.__name__+', '+str(e)+'\n'+
        ''.join(traceback.format_exception(type(e), e, e.__traceback__)), logname)  
    
def normFFT(eeg):
    from scipy import fft
    # gets you the frequency spectrum after the fft by removing mirrored signal and taking modulus
    N = len(eeg)
    fft_eeg = 1/N*np.abs(fft(eeg)[:N//2]) # should really normalize by Time/sample rate (e.g. 4 s of eeg/500 hz sampling=8)
    return fft_eeg
    
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

def getStartEndArrays(ripple_array,sr):
    # get separate arrays of SWR starts and SWR ends from the full binarized array
    sr_factor = (1000/sr)
    start_array = np.zeros((ripple_array.shape))
    end_array = np.zeros((ripple_array.shape))
    num_trials = ripple_array.shape[0]
    for trial in range(num_trials):
        ripplelogictrial = ripple_array[trial]
        starts,ends = getLogicalChunks(ripplelogictrial)
        temp_row = np.zeros(len(ripplelogictrial))
        temp_row[starts] = 1
        start_array[trial] = temp_row # time when each SWR starts
        temp_row = np.zeros(len(ripplelogictrial))
        temp_row[ends] = 1
        end_array[trial] = temp_row
    return start_array,end_array

def detectRipplesHamming(eeg_rip,trans_width,sr,iedlogic):
    # detect ripples similar to with Butterworth, but using Norman et al 2019 algo (based on Stark 2014 algo). Description:
#      Then Hilbert, clip extreme to 4 SD, square this clipped, smooth w/ Kaiser FIR low-pass filter with 40 Hz cutoff,
#      mean and SD computed across entire experimental duration to define the threshold for event detection
#      Events from original (squared but unclipped) signal >4 SD above baseline were selected as candidate SWR events. 
#      Duration expanded until ripple power <2 SD. Events <20 ms or >200 ms excluded. Adjacent events <30 ms separation (peak-to-peak) merged.
    from scipy.signal import firwin,filtfilt,kaiserord
    sr_factor = 1000/sr
    ripple_min = 20/sr_factor # convert each to ms
    ripple_max = 250/sr_factor #200/sr_factor
    min_separation = 30/sr_factor # peak to peak
    orig_eeg_rip = copy(eeg_rip)
    clip_SD = 4*np.std(eeg_rip)
    eeg_rip[eeg_rip>clip_SD] = clip_SD # clip at 4SD
    eeg_rip = eeg_rip**2 # square
    
    # FIR lowpass 40 hz filter for Malach dtection algo
    nyquist = sr/2
    ntaps40, beta40 = kaiserord(40, trans_width/nyquist)
    kaiser_40lp_filter = firwin(ntaps40, cutoff=40, window=('kaiser', beta40), scale=False, nyq=nyquist, pass_zero='lowpass')
    
    eeg_rip = filtfilt(kaiser_40lp_filter,1.,eeg_rip)
    mean_detection_thresh = np.mean(eeg_rip)
    std_detection_thresh = np.std(eeg_rip)
    
    # now, find candidate events (>mean+4SD) and expand to >2SD periods around those events
    orig_eeg_rip = orig_eeg_rip**2
    candidate_thresh = mean_detection_thresh+4*std_detection_thresh
    expansion_thresh = mean_detection_thresh+2*std_detection_thresh
    ripplelogic = orig_eeg_rip >= candidate_thresh
    ripplelogic[iedlogic==1] = 0 # remove IEDs detected from Vaz algo...maybe should do this after expansion to 2SD??
    # expand out to 2SD
    num_trials = ripplelogic.shape[0]
    trial_length = ripplelogic.shape[1]
    for trial in range(num_trials):
        ripplelogictrial = ripplelogic[trial]
        starts,ends = getLogicalChunks(ripplelogictrial)
        data_trial = orig_eeg_rip[trial]
        for i,start in enumerate(starts):
            current_time = 0
            while data_trial[start+current_time]>=expansion_thresh:
                if (start+current_time)==-1:
                    break
                else:
                    current_time -=1
            starts[i] = start+current_time+1
        for i,end in enumerate(ends):
            current_time = 0
            while data_trial[end+current_time]>=expansion_thresh:
                if (end+current_time)==trial_length-1:
                    break
                else:
                    current_time +=1
            ends[i] = end+current_time
            
        # remove any duplicates from starts and ends
        starts = np.array(starts); ends = np.array(ends)
        _,start_idxs = np.unique(starts, return_index=True)
        _,end_idxs = np.unique(ends, return_index=True)
        starts = starts[start_idxs & end_idxs]
        ends = ends[start_idxs & end_idxs]

        # remove ripples <min or >max length
        lengths = ends-starts
        ripple_keep = (lengths > ripple_min) & (lengths < ripple_max)
        starts = starts[ripple_keep]; ends = ends[ripple_keep]

        # get peak times of each ripple to combine those < 30 ms separated peak-to-peak
        if len(starts)>1:
            max_idxs = np.zeros(len(starts))
            for ripple in range(len(starts)):
                max_idxs[ripple] = starts[ripple] + np.argmax(data_trial[starts[ripple]:ends[ripple]])                    
            overlappers = np.where(np.diff(max_idxs)<min_separation)

            if len(overlappers[0])>0:
                ct = 0
                for overlap in overlappers:
                    ends = np.delete(ends,overlap-ct)
                    starts = np.delete(starts,overlap+1-ct)
                    ct+=1 # so each time one is removed can still remove the next overlap
                
        # turn starts/ends into a logical array and replace the trial in ripplelogic
        temp_trial = np.zeros(trial_length)
        for i in range(len(starts)):
            temp_trial[starts[i]:ends[i]]=1
        ripplelogic[trial] = temp_trial # place it back in
    return ripplelogic

def detectRipplesButter(eeg_rip,eeg_ied,eeg_mne,sr): #,mstimes):
    ## detect ripples ##
    # input: hilbert amp from 80-120 Hz, hilbert amp from 250-500 Hz, raw eeg. All trials X duration (ms),mstime of each FR event
    # output: ripplelogic and iedlogic, which are trials X duration masks of ripple presence 
    # note: can get all ripple starts/ends using getLogicalChunks custom function
    from scipy import signal,stats

    sr_factor = 1000/sr # have to account for sampling rate since using ms times 
    ripplewidth = 25/sr_factor # ms
    ripthresh = 2 # threshold detection
    ripmaxthresh = 3 # ripple event must meet this maximum
    ied_thresh = 5 # from Staresina, NN 2015 IED rejection
    ripple_separation = 15/sr_factor # from Roux, NN 2017
    artifact_buffer = 100 # per Vaz et al 2019 

    num_trials = eeg_mne.shape[0]
    eeg_rip_z = stats.zscore(eeg_rip) # note that Vaz et al averaged over time bins too, so axis=None instead of 0
    eeg_ied_z = stats.zscore(eeg_ied)
    eeg_diff = np.diff(eeg_mne) # measure eeg gradient and zscore too
    eeg_diff = np.column_stack((eeg_diff,eeg_diff[:,-1]))# make logical arrays same size
    eeg_diff = stats.zscore(eeg_diff)

    # convert to logicals and remove IEDs
    ripplelogic = eeg_rip_z>ripthresh
    broadlogic = eeg_ied_z>ied_thresh 
    difflogic = abs(eeg_diff)>ied_thresh
    iedlogic = broadlogic | difflogic # combine artifactual ripples
    iedlogic = signal.convolve2d(iedlogic,np.ones((1,artifact_buffer)),'same')>0 # expand to +/- 100 ms
    ripplelogic[iedlogic==1] = 0 # remove IEDs

    # loop through trials and remove ripples
    for trial in range(num_trials):
        ripplelogictrial = ripplelogic[trial]        
        if np.sum(ripplelogictrial)==0:
            continue
        hilbamptrial = eeg_rip_z[trial]

        starts,ends = getLogicalChunks(ripplelogictrial) # get chunks of 1s that are putative SWRs
        for ripple in range(len(starts)):
            if ends[ripple]+1-starts[ripple] < ripplewidth or \
            max(abs(hilbamptrial[starts[ripple]:ends[ripple]+1])) < ripmaxthresh:
                ripplelogictrial[starts[ripple]:ends[ripple]+1] = 0
        ripplelogic[trial] = ripplelogictrial # reassign trial with ripples removed

    # join ripples less than 15 ms separated 
    for trial in range(num_trials):
        ripplelogictrial = ripplelogic[trial]
        if np.sum(ripplelogictrial)==0:
            continue
        starts,ends = getLogicalChunks(ripplelogictrial)
        if len(starts)<=1:
            continue
        for ripple in range(len(starts)-1): # loop through ripples before last
            if (starts[ripple+1]-ends[ripple]) < ripple_separation:            
                ripplelogictrial[ends[ripple]+1:starts[ripple+1]] = 1
        ripplelogic[trial] = ripplelogictrial # reassign trial with ripples removed  
        
#     # now that you have ripples, get the mstime for each so can remove those that overlap across recall events later on
#     ## NEVERMIND--only remove RECALLS that are too close together to avoid overlaps
#     ripple_mstimes = np.zeros((ripplelogic.shape))
#     for trial in range(num_trials):
#         ripplelogictrial = ripplelogic[trial]
#         temp_time_array = np.zeros(len(ripplelogictrial))
#         if np.sum(ripplelogictrial)==0:
#             continue
#         starts,_ = getLogicalChunks(ripplelogictrial)    
#         for ripple in range(len(starts)):
#             temp_time_array[starts[ripple]] = int(mstimes[trial]+starts[ripple]*sr_factor)
#         ripple_mstimes[trial] = temp_time_array    
    
    return ripplelogic,iedlogic #,ripple_mstimes

def downsampleBinary(array,factor):
    # input should be trial X time binary matrix
    array_save = np.array([])
    if factor-int(factor)==0: # if an integer
        for t in range(array.shape[0]): #from https://stackoverflow.com/questions/20322079/downsample-a-1d-numpy-array
            array_save = superVstack(array_save,array[t].reshape(-1,int(factor)).mean(axis=1))
    else:
        # when dividing by non-integer, can just use FFT and round to get new sampling
        from scipy.signal import resample
        if array.shape[1]/factor-int(array.shape[1]/factor)!=0:
            print('Did not get whole number array for downsampling')
        new_sampling = int(array.shape[1]/factor)
        for t in range(array.shape[0]):
            array_save = superVstack(array_save,np.round(resample(array[t],new_sampling)))
    return array_save

def ptsa_to_mne(eegs,time_length): # in ms
    # convert ptsa to mne    
    import mne
    
    sr = int(np.round(eegs.samplerate)) #get samplerate...round 1st since get like 499.7 for some reason  
    eegs = eegs[:, :, :].transpose('event', 'channel', 'time') # make sure right order of names
    
    time = [x/1000 for x in time_length] # convert to s for MNE
    clips = np.array(eegs[:, :, int(sr*time[0]):int(sr*time[1])])

    mne_evs = np.empty([clips.shape[0], 3]).astype(int)
    mne_evs[:, 0] = np.arange(clips.shape[0]) # at each timepoint
    mne_evs[:, 1] = clips.shape[2] # 0
    mne_evs[:, 2] = list(np.zeros(clips.shape[0]))
    event_id = dict(resting=0)
    tmin=0.0
    info = mne.create_info([str(i) for i in range(eegs.shape[1])], sr, ch_types='eeg')  
    
    arr = mne.EpochsArray(np.array(clips), info, mne_evs, tmin, event_id)
    return arr

def fastSmooth(a,window_size): # I ended up not using this one. It's what Norman/Malach use (a python
     # implementation of matlab nanfastsmooth, but isn't actually triangular like it says in paper)
    
    # a: NumPy 1-D array containing the data to be smoothed
    # window_size: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    if np.mod(window_size,2)==0:
        print('sliding window must be odd!!')
        print('See https://stackoverflow.com/questions/40443020/matlabs-smooth-implementation-n-point-moving-average-in-numpy-python')
    out0 = np.convolve(a,np.ones(window_size,dtype=int),'valid')/window_size    
    r = np.arange(1,window_size-1,2)
    start = np.cumsum(a[:window_size-1])[::2]/r
    stop = (np.cumsum(a[:-window_size:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

def triangleSmooth(data,smoothing_triangle): # smooth data with triangle filter using padded edges
    
    factor = smoothing_triangle-3 # factor is how many points from middle does triangle go?
    f = np.zeros((1+2*factor))
    for i in range(factor):
        f[i] = i+1
        f[-i-1] = i+1
    f[factor] = factor + 1
    triangle_filter = f / np.sum(f)

    padded = np.pad(data, factor, mode='edge') # pads same value either side
    smoothed_data = np.convolve(padded, triangle_filter, mode='valid')
    return smoothed_data

def fullPSTH(point_array,binsize,smoothing_triangle,sr,start_offset):
    # point_array is binary point time (spikes or SWRs) v. trial
    # binsize in ms, smoothing_triangle is how many points in triangle kernel moving average
    sr_factor = (1000/sr)
    num_trials = point_array.shape[0]
    xtimes = np.where(point_array)[1]*sr_factor # going to do histogram so don't need to know trial #s
    
    nsamples = point_array.shape[1]
    ms_length = nsamples*sr_factor
    last_bin = binsize*np.ceil(ms_length/binsize)

    edges = np.arange(0,last_bin+binsize,binsize)
    bin_centers = edges[0:-1]+binsize/2+start_offset

    count = np.histogram(xtimes,bins=edges);
    norm_count = count/np.array((num_trials*binsize/1000))
    #smoothed = fastSmooth(norm_count[0],5) # use triangular instead, although this gives similar answer
    PSTH = triangleSmooth(norm_count[0],smoothing_triangle)
    return PSTH,bin_centers

def bootPSTH(point_array,binsize,smoothing_triangle,sr,start_offset): # same as above, but single output so can bootstrap
    # point_array is binary point time (spikes or SWRs) v. trial
    # binsize in ms, smoothing_triangle is how many points in triangle kernel moving average
    sr_factor = (1000/sr)
    num_trials = point_array.shape[0]
    xtimes = np.where(point_array)[1]*sr_factor # going to do histogram so don't need to know trial #s
    
    nsamples = point_array.shape[1]
    ms_length = nsamples*sr_factor
    last_bin = binsize*np.ceil(ms_length/binsize)

    edges = np.arange(0,last_bin+binsize,binsize)
    bin_centers = edges[0:-1]+binsize/2+start_offset

    count = np.histogram(xtimes,bins=edges);
    norm_count = count/np.array((num_trials*binsize/1000))
    #smoothed = fastSmooth(norm_count[0],5) # use triangular instead, although this gives similar answer
    PSTH = triangleSmooth(norm_count[0],smoothing_triangle)
    return PSTH

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

def MakeLocationFilter(scheme, location):
    return [location in s for s in [s if s else '' for s in scheme.iloc()[:]['ind.region']]]

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

def run_stim_regression(row, MTL_labels, test_freq_range, fmin, fmax, fmin_pow, fmax_pow):
    
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
        session = row.session
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
                                            sub+'_'+exp+'_'+str(fmin)+'-'+str(fmax)+'Hz_10s_countdown_network'+'.p')    
        else:
            conn_file = os.path.join('/home1/john/data/eeg/PS3_fxl_conn/'+sub,
                                                sub+'_'+exp+'_'+str(fmin)+'-'+str(fmax)+'_network.p')
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
                stim_evs = evs_on # evs = evs.query("(type=='STIM_ON')")
                label = 'pre'
            elif stim_type==1: # stim_off
                start = post_start
                end = post_end
                stim_evs = evs_off   
                label = 'post'   

            if internal_bipolar == True:
                # Bipolar reference to nearest labeled electrode
                pairs = reader.load('pairs') # voltages across the adjacent contacts
            else: pairs = None            
            
            if stim_type == 1 and exp=='PS3': #PS3 evs_off is messed up!! stim_off was just calculated by adding to stim_on but for
                           # the timestamp they used 0.25* instead of 0.25/ in the equation. Stim_duration ok tho. This corrects it:               
                sr = reader.load("sources")['sample_rate']
                durations = []
                for i in range(stim_evs.shape[0]):
                    durations.append(stim_evs.iloc[i].stim_params[0]['stim_duration']) # get stim durations
                stim_evs['mstime'] = evs_on['mstime'].add(durations).values # replace stim_off times with stim_on+duration
                eeg_durations = np.round(np.array(durations)*(sr/1000))
                stim_evs['eegoffset'] = evs_on['eegoffset'].add(eeg_durations).values.astype('int64') # replace eegoffsets
                
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
        desired_amps = {dlist[0]['amplitude'] for dlist in evs_on.stim_params.to_list()} # {750}
        desired_pulses = {dlist[0]['pulse_freq'] for dlist in evs_on.stim_params.to_list()} # {10} 
        
        good_pre = np.zeros(evs_on.shape[0]); 
        for i,row in enumerate(evs_on.itertuples()): 
            if row.stim_params[0]['amplitude'] in desired_amps and row.stim_params[0]['pulse_freq'] in desired_pulses:
                good_pre[i] = True
        good_post = np.zeros(evs_off.shape[0])        
        for i,row in enumerate(evs_off.itertuples()): 
            if row.stim_params[0]['amplitude'] in desired_amps and row.stim_params[0]['pulse_freq'] in desired_pulses:
                good_post[i] = True 
        good_trials = np.logical_and(good_pre,good_post) # there's no reason these should differ, 
                                            #but just in case (since ttest_rel needs equal shape)        
        #T-test post vs. pre powers
        from scipy.stats import ttest_rel
        # mytrials = np.where(good_trials)[0] #mytrials = np.random.choice(mytrials, 50)
        
        # t-test values for all electrodes between pre/post for each trial
        alpha_value = 0.01
        chan_T, p = ttest_rel(post_pows[good_trials==1, :], pre_pows[good_trials==1, :], axis=0, nan_policy='omit') 
        orig_T_stats = copy(chan_T) # for Fig. 2b
        chan_T[pvals<alpha_value] = np.nan # remove post-stim decay artifacts
        chan_T[lev_pvals<alpha_value] = np.nan 
        chan_T[bad_filt==1] = np.nan # remove bad electrodes (SOZ/ictal)
        # for Fig. 2b, label channels as 0) good 1) SOZ/ictal 2) artifactual
        electrode_indicator = np.zeros(len(chan_T))        
        electrode_indicator[bad_filt==1] = 1 # remove bad electrodes (SOZ/ictal)   
        electrode_indicator[pvals<alpha_value] = 2 # remove post-stim decay artifacts
        electrode_indicator[lev_pvals<alpha_value] = 2 
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
                      str(fmin_pow)+'-'+str(fmax_pow)+'-pow.p') #_pulse-10.p')            

    with open(fn,'wb') as f:
        pickle.dump({'sess_Ts':sess_Ts, 'stimbps':stimbps, 'Conn_Zs': Conn_Zs, 'Conn_Ps': Conn_Ps, 
                     'sessions':sessions, 'tal_structs': tal_structs, 'subjects':subjects,
                     'good_chans':good_chans, 'regs':regs, 'electrode_indicator':electrode_indicator,
                     'orig_T_stats':orig_T_stats}, f)
    return