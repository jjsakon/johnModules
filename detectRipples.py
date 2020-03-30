import numpy as np

def getLogicalChunks(array):
    # find start and end indices for chunks of values >0 in an array
    foundstart = False
    foundend = False
    startindex = 0
    endindex = 0
    starts = []
    ends = []
    for i in range(0, len(array)):
        if array[i] != 0:
            if not foundstart:
                foundstart = True
                startindex = i
        else:
            if foundstart:
                foundend = True
                endindex = i - 1

        if foundend:
            #print(startindex, endindex)
            starts.append(startindex)
            ends.append(endindex)
            foundstart = False
            foundend = False
            startindex = 0
            endindex = 0

    if foundstart:
        ends.append(len(array)-1)
    return starts,ends

def detectRipples(eeg_rip,eeg_ied,eeg_mne,sr):
    ## detect ripples ##
    # input: hilbert amp from 80-120 Hz, hilbert amp from 250-500 Hz, raw eeg. All are trials X duration (ms)
    # output: ripplelogic and iedlogic, which are trials X duration masks of ripple presence 
    # note: can get all ripple starts/ends using getLogicalChunks custom function
    from scipy import signal,stats

    sr_factor = 1000/sr # have to account for sampling rate since using ms times 
    ripplewidth = 25/sr_factor # ms
    ripthresh = 2 # threshold detection
    ripmaxthresh = 3 # ripple event must meet this maximum
    ied_thresh = 5 # from Staresina, NN 2015 IED rejection
    ripple_separation = 15/sr_factor # from Roux, NN 2017
    artifact_buffer = int(200/sr_factor)

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
    iedlogic = signal.convolve2d(iedlogic,np.ones((1,artifact_buffer)),'same')>0 # expand to +/- buffer in ms
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
    return ripplelogic,iedlogic