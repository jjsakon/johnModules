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

def findStimbpcatFR6(evs_on,sub,session,tal_struct,exp):
    stim_pair = [evs_on.iloc[0].stim_params[0]['anode_number'],evs_on.iloc[0].stim_params['cathode_number']]
    # check if anode/cathode change
    stim_electrode_change = 0
    for i in range(evs_on.shape[0]):
        test_pair = [evs_on.iloc[i].stim_params[0]['anode_number'],evs_on.iloc[i].stim_params[0]['cathode_number']]
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

def makePairwiseComparisonPlotCL(comp_data,comp_names,col_names,figsize=(7,4)):
    # make a pairwise comparison errorbar plot with swarm and FDR significance overlaid
    # comp_data: list of vectors of pairwise comparison data
    # comp_names: list of labels for each pairwise comparison
    # col_names: list of 2 names: 1st is what is in data, 2nd is what the grouping of the labels 
    
    import pandas as pd
    from scipy.stats import ttest_1samp
    from statsmodels.stats.multitest import fdrcorrection
    import matplotlib.pyplot as plt
    import seaborn as sb

    # make dataframe
    comp_df = pd.DataFrame(columns=col_names)
    for i in range(len(comp_data)):
        # remove NaNs
        comp_data[i] = np.array(comp_data[i])[~np.isnan(comp_data[i])]
        
        temp = pd.DataFrame(columns=col_names)
        temp['pairwise_data'] = comp_data[i]
        temp['grouping'] = np.tile(comp_names[i],len(comp_data[i]))
        comp_df = comp_df.append(temp,ignore_index=False, sort=True)

    figSub,axSub = plt.subplots(1,1, figsize=figsize)
    axSub.bar( range(len(comp_names)), [np.mean(i) for i in comp_data], 
              yerr = [2*np.std(i)/np.sqrt(len(i)) for i in comp_data],
              color = (0.5,0.5,0.5), error_kw={'elinewidth':55, 'ecolor':(0.7,0.7,0.7)} )
    sb.swarmplot(x='grouping', y='pairwise_data', data=comp_df, ax=axSub, color=(0.8,0,0.8), alpha=0.3)
    axSub.plot([axSub.get_xlim()[0],axSub.get_xlim()[1]],[0,0],linewidth=2,linestyle='--',color=(0,0,0),label='_nolegend_')
    for i in range(len(comp_names)):
        plt.text(i-0.1,0.9,'N='+str(len(comp_data[i])))
#     # put *s for FDR-corrected significance
#     p_values = []
#     for i in range(len(comp_data)):
#         p_values.append(ttest_1samp(comp_data[i],0)[1])
#     sig_after_correction = fdrcorrection(p_values)[0]
#     for i in range(len(sig_after_correction)):
#         if sig_after_correction[i]==True:
#             plt.text(i-0.07,4.575,'*',size=20)
#     print('FDR-corrected p-values for each:')
#     fdr_pvalues = fdrcorrection(p_values)[1]

    # axSub.set(xticks=[],xticklabels=comp_names)
    axSub.set_ylim(0,1)
    plt.xlabel(col_names[0])
    plt.ylabel(col_names[1])
    figSub.tight_layout()
    
#     print(fdr_pvalues)
#     return fdr_pvalues