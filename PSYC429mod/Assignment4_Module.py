import numpy as np
import pandas as pd
import pylab as plt
import functools
import datetime
import scipy
import statsmodels.stats.multitest
import traceback
from cmlreaders import CMLReader, get_data_index
from ptsa.data.filters import morlet
from ptsa.data.filters import ButterworthFilter


def Log(s, suffix=''):
    date = datetime.datetime.now().strftime('%F_%H-%M-%S')
    is_string = isinstance(s,str) 
    if is_string:
        output = date + ': ' + s
    else:
        output = date + ': ' + str(s)    

    filename = 'analysis_log'
    suffix = str(suffix)
    if suffix != '':
        filename = filename + '_' + suffix
    filename = filename + '.txt'

    with open(filename, 'a') as logfile:
        print(output)
        logfile.write(output+'\n')


def LogException(e, suffix=''):
    Log(e.__class__.__name__+', '+str(e)+'\n'+
        ''.join(traceback.format_exception(type(e), e, e.__traceback__)),
        suffix = suffix)
    

def LogDFException(row, e, line_num, fname): #suffix=''):
    rd = row._asdict()
#     Log('Sub: '+str(rd['subject'])+', Exp: '+str(rd['experiment'])+', Sess: '+\
#         str(rd['session'])+', '+e.__class__.__name__+', '+str(e)+'\n'+
#         ''.join(traceback.format_exception(type(e), e, e.__traceback__)),
#         suffix = suffix)
    Log('DF Exception: Sub: '+str(rd['subject'])+', Sess: '+str(rd['session'])+\
        ', '+e.__class__.__name__+', '+str(e)+', file: '+fname+', line no: '+str(line_num))
    
def StartFig():
    plt.figure()
    plt.rcParams.update({'font.size': 12})


def SaveFig(basename):
    plt.savefig(basename+'.png')
    plt.savefig(basename+'.pdf')


def ExpTypes(search_str=''):
    df = get_data_index('all')
    exp_types = set(df['experiment'])
    exp_list = sorted(exp for exp in exp_types if search_str in exp)
    return exp_list


def DataFramesFor(exp_list):
    if isinstance(exp_list, str):
        exp_list = [exp_list]
    
    df = get_data_index('all')
    indices_list = [df['experiment']==exp for exp in exp_list]
    indices = functools.reduce(lambda x,y: x|y, indices_list)
    df_matched = df[indices]
    return df_matched


def SubjectDataFrames(sub_list):
    if isinstance(sub_list, str):
        sub_list = [sub_list]
    
    df = get_data_index('all')
    indices_list = [df['subject']==sub for sub in sub_list]
    indices = functools.reduce(lambda x,y: x|y, indices_list)
    df_matched = df[indices]
    return df_matched


def GetElectrodes(sub):
    df_sub = SubjectDataFrames(sub)
    reader = CMLReadDFRow(next(df_sub.itertuples()))
    # For scalp data, this is currently only accesible via ptsa.
    # So this is the most general method for all data.
    evs = reader.load('events')
    enc_evs = evs[evs.type=='WORD']
    eeg = reader.load_eeg(events=evs, rel_start=0, rel_stop=500, clean=True)
    return eeg.to_ptsa().channel.values


def CMLReadDFRow(row):
    '''for row in df.itertuples():
            reader = CMLReadDFRow(row)
    '''
    rd = row._asdict()
    return CMLReader(rd['subject'], rd['experiment'], rd['session'], \
                     rd['montage'], rd['localization'])


def ClusterRun(function, parameter_list, max_jobs=100):
    '''function: The routine run in parallel, which must contain all necessary
       imports internally.
    
       parameter_list: should be an iterable of elements, for which each
       element will be passed as the parameter to function for each parallel
       execution.
       
       max_jobs: Standard Rhino cluster etiquette is to stay within 100 jobs
       running at a time.  Please ask for permission before using more.
       
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
        num_jobs = 1 # have to do this or [0] gives letters of string for 1st subject
        parameter_list = [parameter_list]
    else:
        num_jobs = len(parameter_list)
    
    num_jobs = len(parameter_list)
    num_jobs = min(num_jobs, max_jobs)

    myhomedir = str(Path.home())

    with cluster_helper.cluster.cluster_view(scheduler="sge", queue="RAM.q", \
        num_jobs=num_jobs, cores_per_job=1, \
        extra_params={'resources':'pename=python-round-robin'}, \
        profile=myhomedir + '/.ipython/') \
        as view:
        # 'map' applies a function to each value within an interable
        res = view.map(function, parameter_list)
        
    return res


class SpectralAnalysis():
  def __init__(self, freqs, subs=None, dfs=None, electrodes=None, \
      elec_masks=None, morlet_reps=6, buf_ms=1000, bin_Hz=None, \
      time_range=(0,1600), split_recall=True, debug=False):

    self.debug = debug
    self.freqs = freqs
    self.bin_Hz = bin_Hz
    self.split_recall = split_recall

    self.subjects = None
    if dfs is not None:
      self.by_session = True
      self.df_all = dfs
      self.subjects = self.df_all.subject.unique()
    else:
      self.by_session = False

    if subs is not None:
      if self.by_session:
        raise ValueError('Set subs or dfs, but not both')
      self.subjects = subs

    if self.subjects is None:
      raise ValueError('Set either subs or dfs')


    if electrodes is None and elec_masks is None:
      self.use_all_elecs = True
    else:
      self.use_all_elecs = False

    if electrodes is not None:
      self.electrodes = electrodes
      self.use_elec_masks = False
    else:
      self.use_elec_masks = True

    if elec_masks is not None:
      if not self.use_elec_masks:
        raise ValueError('Set electrodes or elec_masks, but not both')
      self.elec_masks = elec_masks


    self.SetTimeRange(time_range[0], time_range[1])
    self.buf_ms = buf_ms
    self.morlet_reps = morlet_reps
    
    self.avg_ref = False
    self.zscore = False
    self.internal_bipolar = False


  def SetTimeRange(self, left_ms, right_ms):
    self.left_ms = left_ms
    self.right_ms = right_ms
    if self.bin_Hz is not None:
      self.time_elements = int(((right_ms - left_ms)/1000.0)*self.bin_Hz + 0.5)


  def LoadEEG(self, row, mask_index=None):
    reader = CMLReadDFRow(row)
    # This does not work for this data set,
    # so we will get these from load_eeg(...).to_ptsa() later.
    #contacts = reader.load('contacts')
    evs = reader.load('events')
    self.enc_evs = evs[evs.type=='WORD']

    # Use a pairs scheme if it exists.
    self.pairs = None
    # Disabled for now due to montage errors
    try:
      self.pairs = reader.load('pairs')
    except:
      pass
    
    if np.sum(self.enc_evs.recalled == True) == 0:
        raise IndexError('No recalled events')
    if np.sum(self.enc_evs.recalled == False) == 0:
        raise IndexError('No non-recalled events')

    if self.pairs is None:
      # clean=True for Localized Component Filtering (LCF)
      eeg = reader.load_eeg(events=self.enc_evs, \
        rel_start=self.left_ms - self.buf_ms, \
        rel_stop=self.right_ms + self.buf_ms, clean=True)
    else:
      # clean=True for Localized Component Filtering (LCF)
      eeg = reader.load_eeg(events=self.enc_evs, \
        rel_start=self.left_ms - self.buf_ms, \
        rel_stop=self.right_ms + self.buf_ms, clean=True)
      if self.pairs.shape[0] != eeg.data.shape[1]:
        eeg = reader.load_eeg(events=self.enc_evs, scheme=self.pairs, \
          rel_start=self.left_ms - self.buf_ms, \
          rel_stop=self.right_ms + self.buf_ms, clean=True)
    
    if len(eeg.events) != self.enc_evs.shape[0]:
        raise IndexError(str(len(eeg.events)) + \
            ' eeg events for ' + str(self.enc_evs.shape[0]) + \
            ' encoding events')

    if self.avg_ref == True:
        # Reference to average
        avg_ref_data = np.mean(eeg.data, (1))
        for i in range(eeg.data.shape[1]):
            eeg.data[:,i,:] = eeg.data[:,i,:] - avg_ref_data
    
    if self.internal_bipolar == True:
        # Bipolar reference to nearest labeled electrode
        eeg.data -= np.roll(eeg.data, 1, 1)

    self.sr = eeg.samplerate

    eeg_ptsa = eeg.to_ptsa()

    if self.use_all_elecs:
      self.eeg_ptsa = eeg_ptsa
      self.channel_flags = [True]*len(eeg_ptsa.channel.values)
    else:
      if self.use_elec_masks:
          if self.elec_masks[mask_index] is None:
              raise ValueError( \
                  'No channel mask available for session ' + \
                  str(mask_index))
          if isinstance(self.elec_masks[mask_index], np.ndarray):
              channel_flags = self.elec_masks[mask_index].tolist()
          else:
              channel_flags = self.elec_masks[mask_index]
      else:
          channels = eeg_ptsa.channel.values
          channel_flags = [c in self.electrodes for c in channels]
      
      if np.sum(channel_flags)==0:
          if self.use_elec_masks:
              raise IndexError('No channels for region index ' + \
                  str(mask_index))
          else:
              raise IndexError('No matching channels found for ' + \
                  str(self.electrodes))
      
      self.eeg_ptsa = eeg_ptsa[:,channel_flags,:]
      self.channel_flags = channel_flags

    self.channels = eeg_ptsa.channel.values[self.channel_flags]


  def FilterLineNoise(self):
    freq_range = [58., 62.]
    b_filter = ButterworthFilter(timeseries=self.eeg_ptsa, \
      freq_range=freq_range, filt_type='stop', order=4)
    self.eeg_ptsa = b_filter.filter()
      

  def MorletPower(self):
    wf = morlet.MorletWaveletFilter(timeseries=self.eeg_ptsa, \
      freqs=self.freqs, width=self.morlet_reps, output=['power'], \
      complete=True)
    
    powers_plusbuf = wf.filter()
    # freqs, events, elecs, and time
    start = int((self.buf_ms/1000.0)*self.sr)
    endp1 = -1*int((self.buf_ms/1000.0)*self.sr) + 1
    if endp1 >= 0:
      self.powers = powers_plusbuf[:, :, :, start:]
    else:
      self.powers = powers_plusbuf[:, :, :, start:endp1]
    
    if np.any(np.isnan(self.powers)):
      raise ValueError('nan values in Morlet Wavelet power')


  def MorletComplex(self):
    wf = morlet.MorletWaveletFilter(timeseries=self.eeg_ptsa, \
      freqs=self.freqs, width=self.morlet_reps, output=['complex'], \
      complete=True)
    phasors_plusbuf = wf.filter()
    # freqs, events, elecs, and time
    start = int((self.buf_ms/1000.0)*self.sr)
    endp1 = -1*int((self.buf_ms/1000.0)*self.sr) + 1
    if endp1 >= 0:
      self.phasors = phasors_plusbuf[:, :, :, start:]
    else:
      self.phasors = phasors_plusbuf[:, :, :, start:endp1]

    if np.any(np.isnan(self.phasors)):
      raise ValueError('nan values in Morlet Wavelet complex values')


  def ResamplePowers(self):
    if self.bin_Hz == None:
      return

    # Bin down to bin_Hz as a sampling rate
    binby = self.sr/self.bin_Hz
    new_time_elements = int(self.powers.shape[3] / binby + 0.5)
    if new_time_elements < self.time_elements-1 or \
        new_time_elements > self.time_elements+1:
      raise ValueError('Got '+str(new_time_elements)+' binned '+\
          'elements but expecting '+str(self.time_elements))
    self.powers = \
        scipy.signal.resample(self.powers, self.time_elements, axis=3)


  def ResamplePhasors(self):
    if self.bin_Hz == None:
      return

    # Bin down to bin_Hz as a sampling rate
    binby = self.sr/self.bin_Hz
    new_time_elements = int(self.phasors.shape[3] / binby + 0.5)
    if new_time_elements < self.time_elements-1 or \
        new_time_elements > self.time_elements+1:
      raise ValueError('Got '+str(new_time_elements)+' binned '+\
          'elements but expecting '+str(self.time_elements))
    self.phasors = \
        scipy.signal.resample(self.phasors, self.time_elements, axis=3)



  def NormalizePhasors(self):
    self.phasors = self.phasors / np.absolute(self.phasors)

  
  def PowerSpectra(self, avg_ref=False, zscore=False, bin_elecs=True, \
        internal_bipolar=False):

    if self.split_recall:
      rec_results = []
      nrec_results = []
    else:
      results = []

    for sub in self.subjects:
      if self.by_session:
        df_sub = self.df_all[self.df_all.subject==sub]
      else:
        df_sub = SubjectDataFrames(sub)

      # Store count and first and second harmonic sums, then divide.
      # bin_elecs == False:  rec, freqs
      # bin_elecs == True:  rec, freqs, elecs  (Set later)
      if self.split_recall:
        sub_data_shape = (2, len(self.freqs))
      else:
        sub_data_shape = (1, len(self.freqs))
      sub_res_data = np.zeros(sub_data_shape)

      df_per_sub = 0

      for mask_index, row in enumerate(df_sub.itertuples()):
        try:
          self.LoadEEG(row, mask_index)

          if avg_ref == True:
            # Reference to average
            avg_ref_data = np.mean(self.eeg_ptsa.data, (1))
            for i in range(self.eeg_ptsa.data.shape[1]):
              self.eeg_ptsa.data[:,i,:] = \
                self.eeg_ptsa.data[:,i,:] - avg_ref_data
          
          if internal_bipolar == True:
            # Bipolar reference to nearest labeled electrode
            self.eeg_ptsa.data -= np.roll(self.eeg_ptsa.data, 1, 1)

          self.FilterLineNoise()
          self.MorletPower()
          self.ResamplePowers()

          # Average over time
          self.powers = np.mean(self.powers, (3))
          
          # Across events
          if zscore:
            self.powers = scipy.stats.zscore(self.powers, 1, ddof=1)
                  
          if bin_elecs:
            if self.split_recall:
              rec_powers = np.mean(self.powers[:, \
                  self.enc_evs.recalled == True, :].data, (1,2))
              nrec_powers = np.mean(self.powers[:, \
                  self.enc_evs.recalled == False, :].data, (1,2))
            else:
              res_powers = np.mean(self.powers[:, :, :].data, (1,2))
          else:
            if df_per_sub == 0:
              first_channel_flags = self.channel_flags
              num_channels_found = self.powers.shape[2]
              sub_res_data = np.zeros((sub_data_shape[0], \
                sub_data_shape[1], self.powers.shape[2]))
            else:
              if np.any(first_channel_flags != self.channel_flags):
                raise IndexError( \
                  'Mismatched electrodes for subject')
              if num_channels_found != self.powers.shape[2]:
                raise IndexError( \
                  'Inconsistent number of electrodes found')
            
            if self.split_recall:
              rec_powers = np.mean(self.powers[:, \
                self.enc_evs.recalled == True, :].data, (1))
              nrec_powers = np.mean(self.powers[:, \
                self.enc_evs.recalled == False, :].data, (1))
          
            else:
              res_powers = np.mean(self.powers[:, :, :].data, (1))
          
          if np.any(np.isnan(rec_powers)) or \
              np.any(np.isnan(nrec_powers)):
            raise ValueError('nan values in eeg power')
      
          if self.split_recall:
            sub_res_data[0] += rec_powers
            sub_res_data[1] += nrec_powers
          else:
            sub_res_data[0] += res_powers

          df_per_sub += 1

        except Exception as e:
          exc_type, exc_obj, exc_tb = sys.exc_info()
          line_num = exc_tb.tb_lineno
          fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
          LogDFException(row, e, line_num, fname)
          if self.debug:
            raise

      sub_res_data /= df_per_sub

      if self.split_recall:
        rec_results.append(sub_res_data[0])
        nrec_results.append(sub_res_data[1])
      else:
        results.append(sub_res_data[0])

    if self.split_recall:
      return (rec_results, nrec_results)
    else:
      return results


  def PowerEventsByFreqsChans(self, avg_ref=False, internal_bipolar=False):

    results = []
    recalls = []
    
    for sub in self.subjects:
      if self.by_session:
        df_sub = self.df_all[self.df_all.subject==sub]
      else:
        df_sub = SubjectDataFrames(sub)

      # events, (freqs*chans)
      sub_results = []
      # events
      sub_recalls = []

      df_per_sub = 0
      print('got to 505')
      for mask_index, row in enumerate(df_sub.itertuples()):
        try:
          self.LoadEEG(row, mask_index)

          if avg_ref == True:
            # Reference to average
            avg_ref_data = np.mean(self.eeg_ptsa.data, (1))
            for i in range(self.eeg_ptsa.data.shape[1]):
              self.eeg_ptsa.data[:,i,:] = \
                self.eeg_ptsa.data[:,i,:] - avg_ref_data
          
          if internal_bipolar == True:
            # Bipolar reference to nearest labeled electrode
            self.eeg_ptsa.data -= np.roll(self.eeg_ptsa.data, 1, 1)
          
          self.FilterLineNoise()
          self.MorletPower()
          self.ResamplePowers()

          # Average over time
          self.powers = np.mean(self.powers, (3))
          if df_per_sub == 0:
            first_channels = self.channels
          else:
            if np.any(self.channels != first_channels):
              raise IndexError( \
                'Mismatched electrodes for subject')

          # events, freqs, elecs
          swapped_axes = np.swapaxes(self.powers.data, 0, 1)
          feature_shape = (swapped_axes.shape[0], \
            swapped_axes.shape[1]*swapped_axes.shape[2])
          sub_results.append(swapped_axes.reshape(feature_shape))
          swapped_axes = None
          sub_recalls.append(self.enc_evs.recalled)

          df_per_sub += 1

        except Exception as e:
          exc_type, exc_obj, exc_tb = sys.exc_info()
          line_num = exc_tb.tb_lineno
          fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
          LogDFException(row, e, line_num, fname)
          if self.debug:
            raise

      sub_results = np.array(sub_results)
      # Merge sessions:
      #sub_results = sub_results.reshape((sub_results.shape[0] * \
      #    sub_results.shape[1], sub_results.shape[2]))
      results.append(sub_results)
      recalls.append(np.array(sub_recalls))

    ret_results = np.array(results)
    ret_recalls = np.array(recalls)
    # ret_results: sub, sessions, events, freqs*elecs
    # ret_recalls: sub, sessions, events
    return (ret_results, ret_recalls)

