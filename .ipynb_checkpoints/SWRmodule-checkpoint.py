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

# all the unique sub names for FR1 task in df
total_sub_names = ['R1001P', 'R1002P', 'R1003P', 'R1006P', 'R1010J', 'R1020J',
       'R1022J', 'R1027J', 'R1032D', 'R1033D', 'R1034D', 'R1035M',
       'R1044J', 'R1045E', 'R1048E', 'R1049J', 'R1052E', 'R1054J',
       'R1056M', 'R1059J', 'R1061T', 'R1063C', 'R1065J', 'R1066P',
       'R1067P', 'R1068J', 'R1077T', 'R1080E', 'R1083J', 'R1089P',
       'R1092J', 'R1094T', 'R1096E', 'R1101T', 'R1102P', 'R1104D',
       'R1105E', 'R1108J', 'R1112M', 'R1113T', 'R1115T', 'R1120E',
       'R1122E', 'R1123C', 'R1125T', 'R1128E', 'R1131M', 'R1134T',
       'R1136N', 'R1137E', 'R1138T', 'R1147P', 'R1150J', 'R1151E',
       'R1154D', 'R1158T', 'R1159P', 'R1161E', 'R1162N', 'R1163T',
       'R1167M', 'R1168T', 'R1171M', 'R1172E', 'R1174T', 'R1176M',
       'R1191J', 'R1195E', 'R1200T', 'R1203T', 'R1204T', 'R1212P',
       'R1215M', 'R1217T', 'R1221P', 'R1229M', 'R1230J', 'R1236J',
       'R1241J', 'R1243T', 'R1260D', 'R1268T', 'R1275D', 'R1281E',
       'R1283T', 'R1288P', 'R1292E', 'R1293P', 'R1297T', 'R1298E',
       'R1299T', 'R1306E', 'R1308T', 'R1310J', 'R1311T', 'R1313J',
       'R1315T', 'R1316T', 'R1320D', 'R1323T', 'R1325C', 'R1328E',
       'R1330D', 'R1332M', 'R1334T', 'R1336T', 'R1338T', 'R1339D',
       'R1341T', 'R1342M', 'R1346T', 'R1349T', 'R1350D', 'R1374T',
       'R1397D']

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

def getSWRpathInfo(remove_soz_ictal,recall_type_switch,selected_period):
    # get strings for path name for save and loading cluster data
    if remove_soz_ictal == False:
        soz_label = 'soz_in'
    else:
        soz_label = 'soz_out'
    recall_selection_name = ''
    if recall_type_switch == 1:
        recall_selection_name = 'FIRSTOFCOMPOUND'
    elif recall_type_switch == 2:
        recall_selection_name = 'RECALLTWO'
    elif recall_type_switch == 3:
        recall_selection_name = 'SOLONOCOMPOUND'  
    if selected_period == 'surrounding_recall':
        if recall_type_switch == 0:
            subfolder = 'IRIonly' # for all recall trials as usual
        else:
            subfolder = recall_selection_name
    elif selected_period == 'whole_retrieval':
        subfolder = 'WHOLE_RETRIEVAL'
    elif selected_period == 'encoding':
        subfolder = 'ENCODING' 
    
    return soz_label,recall_selection_name,subfolder

def selectRecallType(recall_type_switch,evs_free_recall,IRI):
    # input recall type (assigned in SWRanalysis) and output the selected idxs and their associated string name

    if recall_type_switch == 0:
        # remove events with Inter-Recall Intervals too small. IRI = psth_start since that's what will show in PSTH
        selected_recalls_idxs = np.append(True,np.diff(evs_free_recall.mstime)>IRI)
        recall_selection_name = ''
    elif recall_type_switch == 1:
        # subset of recalls < IRI as above, but ONLY keeping those first ones where a second one happens within IRI
        # in other words, removing isolated recalls that don't lead to a subseqent recall 
        
        keep_recall_with_another_recall_within = 2000 # trying 2000 ms but could make thish smaller like 1000 ms
        
        # False at end since another recall never happens
        selected_recalls_idxs = np.append(True,np.diff(evs_free_recall.mstime)>IRI) & \
                np.append(np.diff(evs_free_recall.mstime)<=keep_recall_with_another_recall_within,False) 
        recall_selection_name = 'FIRSTOFCOMPOUND'
    elif recall_type_switch == 2:
        # get ONLY second recalls within 2 s of the first recall (these are removed in selection_type=0)
        selected_recalls_idxs = getSecondRecalls(evs_free_recall,IRI) 
        recall_selection_name = 'RECALLTWO'
    elif recall_type_switch == 3:
        # subset of recalls < IRI
        
        remove_recalls_with_another_recall_within = 2000
        
        # True at end since another recall never happens
        selected_recalls_idxs = np.append(True,np.diff(evs_free_recall.mstime)>IRI) & \
                np.append(np.diff(evs_free_recall.mstime)>remove_recalls_with_another_recall_within,True) 
        recall_selection_name = 'SOLONOCOMPOUND'
    
    return recall_selection_name,selected_recalls_idxs

def removeRepeatedRecalls(evs_free_recall,word_evs):
    # use recall df and list word df to identify repeated recalls and remove them from recall df
    
    good_free_recalls = np.ones(len(evs_free_recall))    
    list_nums = evs_free_recall.list.unique()    
    for ln in list_nums:
        evs_idxs_for_list_recalls = np.where(evs_free_recall.list==ln)[0] # idxs in evs df so can set repeats to 0
        list_recalls = evs_free_recall[evs_free_recall.list==ln] # recalls just for this list
        list_words = word_evs[word_evs.list==ln] # words just for this list
        recalls_serial_pos = [int(list_words[list_words.item_name==w].serialpos) for w in list_recalls.item_name] 
        _,repeats_to_remove = remove_recall_repeats(recalls_serial_pos) # get idxs for this list of which recalls were removed
        if len(repeats_to_remove)>0:
            evs_idxs_for_list_recalls = evs_idxs_for_list_recalls[repeats_to_remove] # grab right indxs for the whole session index
            good_free_recalls[evs_idxs_for_list_recalls] = 0 # remove from session index
    evs_free_recall = evs_free_recall[good_free_recalls==1]
    
    return evs_free_recall

def remove_recall_repeats(serialpositions):
    #Takes array of numbers (serial positions) and removes any repeated ones
    items_to_keep = np.ones(len(serialpositions)).astype(bool)
    items_seen = []
    idx_removed = []
    for idx in range(len(serialpositions)):
        if serialpositions[idx] in items_seen:
            items_to_keep[idx] = False
            idx_removed.append(idx)
        items_seen.append(serialpositions[idx])

    final_vec = np.array(serialpositions)[items_to_keep]
    return final_vec, idx_removed

def getOutputPositions(evs,evs_free_recall):       
    # let's get the recall output positions (after selecting which recalls...since can always use df to get original recall list)
    lists_visited = evs_free_recall.list.unique()
    orig_evs_free_recall = evs[(evs.type=='REC_WORD') & (evs.recalled==True)] # get original 
    session_corrected_list_ops = []
    for list_num in lists_visited:
        original_list_recalls = orig_evs_free_recall[orig_evs_free_recall.list==list_num] # grab trials from this list from original free recalls
        original_op_order = np.arange(len(original_list_recalls))
        corrected_list_recalls = evs_free_recall[evs_free_recall.list==list_num]    
        # I think the easiest way to do this is search for the mstimes in the original list to get index, then grab order
        temp_idxs = [findAinB([pos_mstime],original_list_recalls.mstime)[0] for pos_mstime in corrected_list_recalls.mstime]
        session_corrected_list_ops.extend(original_op_order[temp_idxs])
    return session_corrected_list_ops

def get_recall_clustering(recall_cluster_values, recall_serial_pos):
    from scipy.spatial.distance import euclidean
    from scipy.stats import percentileofscore
    import itertools
    #Get temporal/semantic clustering scores. 

    #recall_cluster_values: array of semantic/temporal values
    #recall_serial_pos: array of indices for true recall sequence (indexing depends on when called), e.g. [1, 12, 3, 5, 9, 6]

    recall_cluster_values = copy(np.array(recall_cluster_values).astype(float))
    all_pcts = []
    all_possible_trans = list(itertools.combinations(range(len(recall_cluster_values)), 2))

    for ridx in np.arange(len(recall_serial_pos)-1):  #Loops through each recall event, except last one
        possible_trans = [comb 
                          for comb in all_possible_trans 
                          if (recall_serial_pos[ridx] in comb)
                         ]
        dists = []
        for c in possible_trans: # all possible trans within list...do it this way since can avoid the true one with the except
            try:
                dists.append(euclidean(recall_cluster_values[c[0]], recall_cluster_values[c[1]]))
            except:
                #If we did this transition, then it's a NaN, so append a NaN
                dists.append(np.nan)
        dists = np.array(dists)
        dists = dists[np.isfinite(dists)]
        true_trans = euclidean(recall_cluster_values[recall_serial_pos[ridx]], recall_cluster_values[recall_serial_pos[ridx+1]])
        pctrank = 1.-percentileofscore(dists, true_trans, kind='strict')/100. # use strict so lag of 1 yields 1.00 percentile
        all_pcts.append(pctrank) # percentile rank within each list

        recall_cluster_values[recall_serial_pos[ridx]] = np.nan # used serialpos get a NaN so won't be included in %ile

    return all_pcts

def getSecondRecalls(evs_free_recall,IRI):
    # instead of removing recalls with <IRI, get ONLY the second recalls have been been removed
    # note that all recalls within IRI of the second recalls are then remove to make it "only"
    mstime_diffs = np.diff(evs_free_recall.mstime)
    second_recalls = np.append(False,mstime_diffs<=IRI) # first one can never be second recall so add a False
    mstime_diffs = np.append(0,mstime_diffs) # add a first trial just to make this align with diffs
    adjusted_second_recalls = copy(second_recalls)

    i=-1
    while i < len(second_recalls)-1:
        i+=1
        second_recall = second_recalls[i]
        if second_recall == True and i < (len(second_recalls)-1): # -1 since adding 1 below
            # now that have a second recall, make sure ones after it aren't within 2000 ms of it
            last_time_diff = 0
            while (last_time_diff+mstime_diffs[i+1])<=IRI:
                adjusted_second_recalls[i+1] = False
                last_time_diff = last_time_diff + mstime_diffs[i+1]
                i+=1
                if i >= (len(second_recalls)-1): # same idea as above. already checked last row so -1!
                    break                
    return adjusted_second_recalls
    
def get_bp_tal_struct(sub, montage, localization):
    
    from ptsa.data.readers import TalReader    
   
    #Get electrode information -- bipolar
    tal_path = '/protocols/r1/subjects/'+sub+'/localizations/'+str(localization)+'/montages/'+str(montage)+'/neuroradiology/current_processed/pairs.json'
    tal_reader = TalReader(filename=tal_path)
    tal_struct = tal_reader.read()
    monopolar_channels = tal_reader.get_monopolar_channels()
    bipolar_pairs = tal_reader.get_bipolar_pairs()
    
    return tal_struct, bipolar_pairs, monopolar_channels

# def get_elec_regions(tal_struct): # old version from Ethan
    
#     regs = []    
#     atlas_type = []
#     for num,e in enumerate(tal_struct['atlases']):
#         try:
#             if 'stein' in e.dtype.names:
#                 if (e['stein']['region'] is not None) and (len(e['stein']['region'])>1) and \
#                    (e['stein']['region'] not in 'None') and (e['stein']['region'] not in 'nan'):
#                     regs.append(e['stein']['region'].lower())
#                     atlas_type.append('stein')
#                     continue
#                 else:
#                     pass
#             if 'das' in e.dtype.names:
#                 if (e['das']['region'] is not None) and (len(e['das']['region'])>1) and \
#                    (e['das']['region'] not in 'None') and (e['das']['region'] not in 'nan'):
#                     regs.append(e['das']['region'].lower())
#                     atlas_type.append('das')
#                     continue
#                 else:
#                     pass
#             if 'wb' in e.dtype.names:
#                 if (e['wb']['region'] is not None) and (len(e['wb']['region'])>1):
#                     regs.append(e['wb']['region'].lower())
#                     atlas_type.append('wb')
#                     continue
#                 else:
#                     pass
#             if 'dk' in e.dtype.names:
#                 if (e['dk']['region'] is not None) and (len(e['dk']['region'])>1):
#                     regs.append(e['dk']['region'].lower())
#                     atlas_type.append('dk')
#                     continue
#                 else:
#                     pass 
#             if 'ind' in e.dtype.names:
#                 if (e['ind']['region'] is not None) and (len(e['ind']['region'])>1) and \
#                    (e['ind']['region'] not in 'None') and (e['ind']['region'] not in 'nan'):
#                     regs.append(e['ind']['region'].lower())
#                     atlas_type.append('ind')
#                     continue
#                 else:
#                     pass               
#             else:                
#                 regs.append('')
#                 atlas_type.append('No atlas')
#         except AttributeError:
#             regs.append('')
            
#     return np.array(regs),np.array(atlas_type)

def get_elec_regions(tal_struct,localizations,pairs): 
    # new version after consulting with Paul 2020-08-13
    # suggested order to use regions is: stein->das->MTL->wb->mni
    
    # unlike previous version this inputs tal_struct (pairs.json as a recArray) and localizations.json like this:
    # pairs = reader.load('pairs')
    # localizations = reader.load('localization')
    
    regs = []    
    atlas_type = []
    pair_number = []
    has_stein_das = 0
    # if localization.json exists get the names from each atlas
    if len(localizations) > 1: 
        # pairs that were recorded and possible pairs from the localization are typically not the same.
        # so need to translate the localization region names to the pairs...which I think is easiest to just do here
        loc_MTL_names = np.array(localizations['atlases.mtl']['pairs']) # MTL field from pairs in localization.json
        loc_dk_names = np.array(localizations['atlases.dk']['pairs']) #dk from same
        loc_wb_names = np.array(localizations['atlases.whole_brain']['pairs']) #wb from same 
        # get an index for every pair in pairs
        loc_to_pairs_idxs = getLocalizationToPairsTranslation(pairs,localizations)
        import ipdb; ipdb.set_trace()

    else:
        loc_MTL_names = 'nan'; loc_dk_names = 'nan'; loc_wb_names = 'nan'

    for pair_ct in range(len(tal_struct)):
        try:
            pair_number.append(pair_ct) # just to keep track of what pair this was in subject
            pair_atlases = tal_struct[pair_ct].atlases
            
            if 'stein' in pair_atlases.dtype.names:
                if (pair_atlases['stein']['region'] is not None) and (len(pair_atlases['stein']['region'])>1) and \
                   (pair_atlases['stein']['region'] not in 'None') and (pair_atlases['stein']['region'] != 'nan'):
                    regs.append(pair_atlases['stein']['region'].lower())
                    atlas_type.append('stein')
                    has_stein_das = 1 # temporary thing just to see where stein/das stopped annotating
                    continue
                else:
                    pass
            if 'das' in pair_atlases.dtype.names:
                if (pair_atlases['das']['region'] is not None) and (len(pair_atlases['das']['region'])>1) and \
                   (pair_atlases['das']['region'] not in 'None') and (pair_atlases['das']['region'] != 'nan'):
                    regs.append(pair_atlases['das']['region'].lower())
                    atlas_type.append('das')
                    has_stein_das = 1
                    continue
                else:
                    pass
            if len(localizations) > 1:             # 'MTL' from localization.json
                if loc_MTL_names[pair_ct] != '':
                    if str(loc_MTL_names[pair_ct]) != 'nan': # looking for "MTL" field in localizations.json
                        regs.append(loc_MTL_names[pair_ct].lower())
                        atlas_type.append('MTL_localization')
                        continue
                    else:
                        pass
            if len(localizations) > 1:             # 'whole_brain' from localization.json
                if loc_wb_names[pair_ct] != '' and loc_wb_names[pair_ct] != ' ':
                    if str(loc_wb_names[pair_ct]) != 'nan': # looking for "MTL" field in localizations.json
                        regs.append(loc_wb_names[pair_ct].lower())
                        atlas_type.append('wb_localization')
                        continue
                    else:
                        pass
            if 'wb' in pair_atlases.dtype.names:
                if (pair_atlases['wb']['region'] is not None) and (len(pair_atlases['wb']['region'])>1) and \
                   (pair_atlases['wb']['region'] not in 'None') and (pair_atlases['wb']['region'] != 'nan'):
                    regs.append(pair_atlases['wb']['region'].lower())
                    atlas_type.append('wb')
                    continue
                else:
                    pass
            if len(localizations) > 1:             # 'dk' from localization.json
                if loc_dk_names[pair_ct] != '':
                    if str(loc_dk_names[pair_ct]) != 'nan': # looking for "dk" field in localizations.json
                        regs.append(loc_dk_names[pair_ct].lower())
                        atlas_type.append('dk_localization')
                        continue
                    else:
                        pass
#             if pair_ct == 29:
#                 import ipdb; ipdb.set_trace()
            if 'dk' in pair_atlases.dtype.names:
                if (pair_atlases['dk']['region'] is not None) and (len(pair_atlases['dk']['region'])>1) and \
                   (pair_atlases['dk']['region'] not in 'None') and (pair_atlases['dk']['region'] != 'nan'):
                    regs.append(pair_atlases['dk']['region'].lower())
                    atlas_type.append('dk')
                    continue
                else:
                    pass
            if 'ind.corrected' in pair_atlases.dtype.names:
                if (pair_atlases['ind.corrected']['region'] is not None) and (len(pair_atlases['ind.corrected']['region'])>1) and \
                   (pair_atlases['ind.corrected']['region'] not in 'None') and (pair_atlases['ind.corrected']['region'] not in 'nan'):
                    regs.append(pair_atlases['ind.corrected']['region'].lower())
                    atlas_type.append('ind.corrected')
                    continue
                else:
                    pass  
            if 'ind' in pair_atlases.dtype.names:
                if (pair_atlases['ind']['region'] is not None) and (len(pair_atlases['ind']['region'])>1) and \
                   (pair_atlases['ind']['region'] not in 'None') and (pair_atlases['ind']['region'] != 'nan'):
                    regs.append(pair_atlases['ind']['region'].lower())
                    atlas_type.append('ind')
                    continue
                else:
                    pass               
            else:                
                regs.append('No atlas')
                atlas_type.append('No atlas')
        except AttributeError:
            regs.append('')
            
    return np.array(regs),np.array(atlas_type),np.array(pair_number),has_stein_das

def getLocalizationToPairsTranslation(pairs,localizations):
    # localizations is all the possible contacts and bipolar pairs locations
    # pairs is the actual bipolar pairs recorded (plugged in to a certain montage of the localization)
    # this goes through all the recorded pairs and finds the index of them in the localization pairs
    loc_pairs = localizations.type.pairs
    loc_pair_labels = np.array(loc_pairs.index)

    loc_to_pairs_idxs = []
    for pair in pairs.label:
        split_pair = pair.split('-') # split into list of 2
        idx = np.where([split_pair==list(lp) for lp in loc_pair_labels])[0]
        if loc_pair_labels[idx].size==0: # if didn't find it, check for reverse
            split_pair.reverse()
            temp_mask = [split_pair==list(lp) for lp in loc_pair_labels]
            idx = np.where(temp_mask)[0]
            if len(idx) == 0:
                idx = ' '
        loc_to_pairs_idxs.extend(idx)

    return loc_to_pairs_idxs

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

def getBadChannels(tal_struct,elecs_cat,remove_soz_ictal):
    # get the bad channels and soz/ictal/lesion channels from electrode_categories.txt files
    bad_bp_mask = np.zeros(len(tal_struct))
    if elecs_cat != []:
        if remove_soz_ictal == True:
            bad_elecs = elecs_cat['bad_channel'] + elecs_cat['soz'] + elecs_cat['interictal']
        else:
            bad_elecs = elecs_cat['bad_channel']
        for idx,tal_row in enumerate(tal_struct):
            elec_labels = tal_row['tagName'].split('-')
            # if there are dashes in the monopolar elec names, need to fix that
            if (len(elec_labels) > 2) & (len(elec_labels) % 2 == 0): # apparently only one of these so don't need an else
                n2 = int(len(elec_labels)/2)
                elec_labels = ['-'.join(elec_labels[0:n2]), '-'.join(elec_labels[n2:])]
            if elec_labels[0] in bad_elecs or elec_labels[1] in bad_elecs:
                bad_bp_mask[idx] = 1
    return bad_bp_mask

def getStartEndArrays(ripple_array):
    # get separate arrays of SWR starts and SWR ends from the full binarized array
    start_array = np.zeros((ripple_array.shape),dtype='uint8')
    end_array = np.zeros((ripple_array.shape),dtype='uint8')
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
    from scipy.signal import firwin,filtfilt,kaiserord,convolve2d
    
    artifact_buffer = 100 # ms around IED events to remove SWRs
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
    
    # now, find candidate events (>mean+4SD) 
    orig_eeg_rip = orig_eeg_rip**2
    candidate_thresh = mean_detection_thresh+4*std_detection_thresh
    expansion_thresh = mean_detection_thresh+2*std_detection_thresh
    ripplelogic = orig_eeg_rip >= candidate_thresh
    # remove IEDs detected from Norman 25-60 algo...maybe should do this after expansion to 2SD??
    iedlogic = convolve2d(iedlogic,np.ones((1,artifact_buffer)),'same')>0 # expand to +/- 50 ms from each ied point
    ripplelogic[iedlogic==1] = 0 
    
    # expand out to 2SD around surviving events
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
            print('Did not get whole number array for downsampling...rounding to nearest 100')
        new_sampling = int( round((array.shape[1]/factor)/100) )*100
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
    
    # problem with this smoothing is when there's a point on the edge it gives too much weight to 
    # first 2 points (3rd is okay). E.g. for a 1 on the edge of all 0s it gives 0.66, 0.33, 0.11
    # while for a 1 in the 2nd position of all 0s it gives 0.22, 0.33, 0.22, 0.11 (as you'd want)
    # so make sure you don't use data in first 2 or last 2 positions since that 0.66/0.33 is overweighted
    
    factor = smoothing_triangle-3 # factor is how many points from middle does triangle go?
    # this all just gets the triangle for given smoothing_triangle length
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
    if smoothing_triangle==1:
        PSTH = norm_count[0]
    else:
        PSTH = triangleSmooth(norm_count[0],smoothing_triangle)
    return PSTH,bin_centers

def binBinaryArray(start_array,bin_size,sr_factor):
    # instead of PSTH, get a binned binary array that keeps the trials but bins over time
    bins = np.arange(0,start_array.shape[1],bin_size/sr_factor) #start_array.shape[1]/bin_size*sr_factor
    bin_in_sr = bin_size/sr_factor
    bin_to_hz = 1000/bin_size*bin_in_sr # factor that converts binned matrix to Hz
    
    binned_array = [] # note this will be at instantaeous rate bin_in_sr multiple lower (e.g. 100 ms bin/2 sr = 50x)
    for row in start_array:
        temp_row = []
        for time_bin in bins:
            temp_row.append(bin_to_hz*np.mean(row[int(time_bin):int(time_bin+bin_in_sr)]))
        binned_array = superVstack(binned_array,temp_row)
    return binned_array

def getSubSessPredictors(sub_names,sub_sess_names,trial_nums):
    # get arrays of predictors for each trial so can set up ME model
    
    subject_name_array = []
    session_name_array = []

    trial_ct = 0
    for ct,subject in enumerate(sub_names):    
        trials_this_loop = int(trial_nums[ct])
        trial_ct = trial_ct + trials_this_loop 
        # update each array with subjects, sessions, and other prdictors
        subject_name_array.extend(np.tile(subject,trials_this_loop))
        session_name_array.extend(np.tile(sub_sess_names[ct],trials_this_loop))
        
    return subject_name_array,session_name_array

def getMixedEffectCIs(binned_start_array,subject_name_array,session_name_array):
    # take a binned array of ripples and find the mixed effect confidence intervals
    # note that output is the net ± distance from mean
    import statsmodels.formula.api as smf

    # now, to set up ME regression, append each time_bin to bottom and duplicate
    mean_values = []
    CIs = []
    for time_bin in range(np.shape(binned_start_array)[1]):
        ripple_rates = binned_start_array[:,time_bin]
        CI_df = pd.DataFrame(data={'session':session_name_array,'subject':subject_name_array,'ripple_rates':ripple_rates})
        # now get the CIs JUST for this time bin
        vc = {'session':'0+session'}
        get_bin_CI_model = smf.mixedlm("ripple_rates ~ 1", CI_df, groups="subject", vc_formula=vc)
        bin_model = get_bin_CI_model.fit(reml=False, method='nm')
        mean_values.append(bin_model.params.Intercept)
        CIs = superVstack(CIs,bin_model.conf_int().iloc[0].values)
    # get CI distances at each bin by subtracting from means
    CI_plot = np.array(CIs.T)
    CI_plot[0,:] = mean_values - CI_plot[0,:] # - difference to subtract from PSTH
    CI_plot[1,:] = CI_plot[1,:] - mean_values # + difference to add to PSTH
    
    return CI_plot

def MEstatsAcrossBins(binned_start_array,subject_name_array,session_name_array):
    # returns mixed effect model for the given trial X bins array by comparing bins
    import statsmodels.formula.api as smf
    
    bin_label = []
    session_name = []
    subject_name = []
    ripple_rates = []
    # now, to set up ME pairwise stats, append each time_bin to bottom and duplicate
    for time_bin in range(np.shape(binned_start_array)[1]): 
        session_name.extend(session_name_array)
        subject_name.extend(subject_name_array)
        bin_label.extend(np.tile(str(time_bin),binned_start_array.shape[0]))
        ripple_rates.extend(binned_start_array[:,time_bin])

    bin_df = pd.DataFrame(data={'session':session_name,'subject':subject_name,
                               'bin':bin_label,'ripple_rates':ripple_rates})
    vc = {'session':'0+session'}
    sig_bin_model = smf.mixedlm("ripple_rates ~ bin", bin_df, groups="subject", vc_formula=vc)
    bin_model = sig_bin_model.fit(reml=False, method='nm')
    return bin_model

def MEstatsAcrossCategories(binned_recalled_array,binned_forgot_array,sub_forgot,sess_forgot,sub_recalled,sess_recalled):
    # here looking at only a single bin but now comparing across categories (e.g. remembered v. forgotten)
    import statsmodels.formula.api as smf
    
    category_label = []
    session_name = []
    subject_name = []
    ripple_rates = []
    # now, to set up ME pairwise stats, append each time_bin to bottom and duplicate
    for category in range(2): 
        if category == 0: # remembered then forgot
            binned_start_array = binned_recalled_array
            session_name_array = sess_recalled
            subject_name_array = sub_recalled
        else:
            binned_start_array = binned_forgot_array
            session_name_array = sess_forgot
            subject_name_array = sub_forgot        
        session_name.extend(session_name_array)
        subject_name.extend(subject_name_array)
        category_label.extend(np.tile(str(category),binned_start_array.shape[0]))
        ripple_rates.extend(binned_start_array[:,0]) # even though only a vector gotta call it out of this list so it's hashable 

    bin_df = pd.DataFrame(data={'session':session_name,'subject':subject_name,
                               'category':category_label,'ripple_rates':ripple_rates})
    vc = {'session':'0+session'}
    sig_bin_model = smf.mixedlm("ripple_rates ~ category", bin_df, groups="subject", vc_formula=vc)
    bin_model = sig_bin_model.fit(reml=False, method='nm')
    return bin_model

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
    # can add in 'mem':Num where Num is # of GB to allow for memory into extra_params
    #...Nora said it doesn't work tho and no sign it does
    # can also try increasing cores_per_job to >1, but should also reduce num_jobs to not hog
    # so like 2 and 50 instead of 1 and 100 etc. Went up to 5 for encoding at points
    # ...actually now went up to 10/10 which seems to stop memory errors 2020-08-12
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
