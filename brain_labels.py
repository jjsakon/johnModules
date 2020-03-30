'''
All regions I've been able to find in PS tasks, either in the stim. electrode (see below)
or in the other electrodes (see get_elec_regions). These can be found in a manner like
the "unique electrode region labels" cell in dataQuality.ipynb
Can import like so: >>>from brain_labels import MTL_labels, LTC_labels, PFC_labels, OTHER_labels, ALL_labels
2020-02-14 JS
'''

MTL_stein = ['left ca1','left ca2','left ca3','left dg','left sub','left prc','left ec','left phc',
             'right ca1','right ca2','right ca3','right dg','right sub','right prc','right ec',
             'right phc','left mtl wm','right mtl wm','left amy','right amy','ca1'] # including amygdala in MTL
LTC_stein = ['left middle temporal gyrus','right middle temporal gyrus','right stg']
PFC_stein = ['left caudal middle frontal cortex','left dlpfc','left precentral gyrus','right precentral gyrus',
             'right caudal middle frontal cortex','right dlpfc','right superior frontal gyrus']
cingulate_stein = ['left acg','left mcg','left pcg','right acg','right pcg']
parietal_stein = ['left supramarginal gyrus','right supramarginal gyrus']
other_TL_stein = ['ba36','left fusiform gyrus wm'] # actually from Das. ba36 is part of fusiform
other_stein = ['left precentral gyrus','none','right insula','right precentral gyrus','nan','misc']

# Using Desikan Neuroimage (2016), the ind localizations come from automated segmentation
# I'm also adding in dk and wb to these, since for some reason those are used for some electrode regions
# -dk comes from the same DesikanKilliany(2006) paper
# -wb (whole-brain) appears to come from FreeSurfer labels here: 
# https://www.slicer.org/wiki/Documentation/4.1/SlicerApplication/LookupTables/Freesurfer_labels
MTL_ind = ['parahippocampal','entorhinal','temporalpole',
           'left hippocampus','right hippocampus','left ent entorhinal area','left tmp temporal pole', # whole-brain names
          'right ent entorhinal area','right tmp temporal pole'] # haven't seen but put right in too
LTC_ind = ['bankssts','middletemporal','inferiortemporal','superiortemporal'] # first 4 defined by Ezzyat NatComm 2018...unsure about bankssts tho
PFC_ind = ['caudalmiddlefrontal','frontalpole','lateralorbitofrontal','medialorbitofrontal','parsopercularis',
          'parsorbitalis','parstriangularis','rostralmiddlefrontal','superiorfrontal']
cingulate_ind = ['caudalanteriorcingulate','isthmuscingulate','posteriorcingulate','rostralanteriorcingulate']
parietal_ind = ['inferiorparietal','postcentral','precuneus','superiorparietal','supramarginal']
occipital_ind = ['cuneus','lateraloccipital','lingual','pericalcarine']
other_TL_ind = ['fusiform','transversetemporal'] # temporal lobe but not MTL
other_ind = ['insula','none','precentral','paracentral','right inf lat vent','left inf lat vent', # not sure where to put these
            'left cerebral white matter','right cerebral white matter', # these wb labels can be anywhere in hemisphere so just put in other
             'nan','left lateral ventricle','right lateral ventricle']

MTL_labels = MTL_stein+MTL_ind
LTC_labels = LTC_stein+LTC_ind
PFC_labels = PFC_stein+PFC_ind
OTHER_labels = cingulate_stein+parietal_stein+other_TL_stein+other_stein+ \
                cingulate_ind+occipital_ind+other_TL_ind+other_ind
ALL_labels = MTL_labels+LTC_labels+PFC_labels+OTHER_labels


'''
# This is the original, which only has labels for those places STIMULATED across PS tasks.
# The above has regions for the other (record-only) electrodes as well. I dunno why you'd want to use
# this smaller set below, but keeping it for posterity
# stim location labels
# these are all the regions that were stimulated in PS and locationSearch tasks
MTL_stein = ['left ca1','left ca2','left ca3','left dg','left sub','left prc','left ec','left phc',
             'right ca1','right ca2','right ca3','right dg','right sub','right prc','right ec',
             'right phc','left mtl wm','right mtl wm','left amy','right amy'] # including amygdala in MTL
LTC_stein = ['left middle temporal gyrus','right middle temporal gyrus','right stg']
PFC_stein = ['left caudal middle frontal cortex','left dlpfc','left precentral gyrus','right precentral gyrus',
             'right caudal middle frontal cortex','right dlpfc','right superior frontal gyrus']
cingulate_stein = ['left acg','left mcg','left pcg','right acg','right pcg']
parietal_stein = ['left supramarginal gyrus','right supramarginal gyrus']
other_TL_stein = ['ba36','left fusiform gyrus wm'] # actually from Das. ba36 is part of fusiform
other_stein = ['left precentral gyrus','none','right insula','right precentral gyrus']
# Using Desikan Neuroimage (2016), the ind localizations come from automated segmentation
MTL_ind = ['parahippocampal','entorhinal','temporalpole']
LTC_ind = ['bankssts','middletemporal','inferiortemporal','superiortemporal'] # as defined by Ezzyat NatComm 2018...unsure about bankssts tho
PFC_ind = ['caudalmiddlefrontal','frontalpole','lateralorbitofrontal','medialorbitofrontal','parsopercularis',
          'parsorbitalis','parstriangularis','rostralmiddlefrontal','superiorfrontal']
cingulate_ind = ['caudalanteriorcingulate','isthmuscingulate','posteriorcingulate','rostralanteriorcingulate']
parietal_ind = ['inferiorparietal','postcentral','precuneus','superiorparietal','supramarginal']
occipital_ind = ['cuneus','lateraloccipital','lingual','pericalcarine']
other_TL_ind = ['fusiform','transversetemporal'] # temporal lobe but not MTL
other_ind = ['insula','none','precentral','paracentral'] # not sure where to put these
'''