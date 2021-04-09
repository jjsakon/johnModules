import pandas as pd
import numpy as np
from cmlreaders import CMLReader
import sys
import os
import matplotlib.pyplot as plt
from copy import copy
import functools
import datetime
import scipy

# functions for closedLoop tasks like TICL

def findStimbpTICL(evs_on,sub,session,tal_struct,exp):
    stim_pair = [evs_on.iloc[0].stim_params['anode_number'],evs_on.iloc[0].stim_params['cathode_number']]
    # check if anode/cathode change
    stim_electrode_change = 0
    for i in range(evs_on.shape[0]):
        test_pair = [evs_on.iloc[i].stim_params['anode_number'],evs_on.iloc[i].stim_params['cathode_number']]
        if stim_pair != test_pair:
            stim_electrode_change = 1
            stim_pair = copy(test_pair)
            print('The anode/cathode pair changed during this session!! Task: '+exp+
                  ', '+sub+', session: '+str(session)+'; now channel: '+str(test_pair))
    chs = [list(temp) for temp in tal_struct['channel']]
    stimbp = np.nan
    for idx,ch_pair in enumerate(chs):
        if stim_pair==ch_pair: # if the anode/cathode pair were recorded from (not always the case!)  
            stimbp = idx
    if np.isnan(stimbp): # if anode/cathode pair don't have a site...just grab first from anode       
        stimbp = findAinBlists([stim_pair[0]],chs)[0]
        print('stimbp set to: '+str(stimbp)+', since anode/cathode pair not in tal_struct for '+sub
              +', '+str(session))
    return stimbp,stim_electrode_change 