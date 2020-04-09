import numpy as np
import pandas as pd
import functools
import datetime
import scipy
from cmlreaders import CMLReader, get_data_index
from ptsa.data.filters import morlet
from ptsa.data.filters import ButterworthFilter
import matplotlib.pyplot as plt
import sys
import os
# %matplotlib inline


def Log(s):
    date = datetime.datetime.now().strftime('%F_%H-%M-%S')
    output = date + ': ' + str(s)
    with open('analysis_log.txt', 'a') as logfile:
        print(output)
        logfile.write(output+'\n')

def LogDFException(row, e, line_num, fname):
    rd = row._asdict()
    Log('DF Exception: Sub: '+str(rd['subject'])+', Sess: '+str(rd['session'])+\
        ', '+e.__class__.__name__+', '+str(e)+', file: '+fname+', line no: '+str(line_num))
    
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
                     rd['montage'], rd['localization'])

def GetElectrodes(sub,start,stop):
    df_sub = SubjectDataFrames(sub)
    reader = CMLReadDFRow(next(df_sub.itertuples()))
    evs = reader.load('events')
    enc_evs = evs[evs.type=='WORD']
    eeg = reader.load_eeg(events=enc_evs, rel_start=start, rel_stop=stop, clean=True)
    return eeg.to_ptsa().channel.values

def MakeLocationFilter(scheme, location):
    return [location in s for s in [s if s else '' for s in scheme.iloc()[:]['ind.region']]]

def ClusterRun(function, parameter_list):
    '''function: The routine run in parallel, which must contain all necessary
       imports internally.
    
       parameter_list: should be an iterable of elements, for which each element
       will be passed as the parameter to function for each parallel execution.
       
       In jupyterlab, the number of engines reported as initially running may
       be smaller than the number actually running.  Check usage from an ssh
       terminal using:  qstat -f | egrep "$USER|node" | less
       
       Undesired running jobs can be killed by reading the JOBID at the left
       of that qstat command, then doing:  qdel JOBID
    '''
    import cluster_helper.cluster
    from pathlib import Path

    is_string = isinstance(parameter_list,str) 
    if is_string:
        num_cores = 1 # have to do this or [0] gives letters of string for 1st subject
        parameter_list = [parameter_list]
    else:
        num_cores = len(parameter_list)

    # Standard Rhino cluster etiquette is to stay within 100 cores at a time.
    num_cores = min(num_cores, 100)

    myhomedir = str(Path.home())

    with cluster_helper.cluster.cluster_view(scheduler="sge", queue="RAM.q", \
        num_jobs=1, cores_per_job=num_cores, \
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
    import ipdb; ipdb.set_trace()    
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
                                #import ipdb; ipdb.set_trace() 
                df_per_sub += 1
            except Exception as e:   
                exc_type, exc_obj, exc_tb = sys.exc_info()
                line_num = exc_tb.tb_lineno
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                LogDFException(row, e, line_num, fname)
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