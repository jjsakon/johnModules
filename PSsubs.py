'''PS Subjects!!'''

import numpy as np

## PS3 ##
# the following subjects are good with montages/localizations being consistent:
PS3_subs = ['R1034D', 'R1050M', 'R1051J', 'R1054J', 'R1056M', 'R1060M', 'R1062J', # through 1056M session 4 # 50
 'R1067P', 'R1069M', 'R1074M', 'R1077T', 'R1081J',
 'R1086M', 'R1112M', 'R1136N', 'R1141T', 'R1142N', 'R1149N', 'R1162N']
PS3_subs.extend(['R1066P','R1185N']) # these subs have FRs to match PS3 despite mont/loc changes
# the FC code below takes care of this now (it makes sure mont/loc from FR match PS3)
PS3_problem_subs = ['R1059J','R1064E','R1073J'] # no FR/PAL periods for last two. 
# R1059J has two FR periods but diff loc/mont. All 3 have YC periods with right mont/loc just no FRs. 8 sessions total
PS3_MTL_mask = [1., 1., 1., 0., 0., 0., 1., 1., 1., 1., 0., 1., 0., 0., 0., 1., 1.,
       1., 0., 0., 0., 0., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 0., 0.,
       0., 1., 0.] # stimulation sites in MTL (found in PS3analysis.ipynb)

## PS2.1 ##
PS21_subs = ['R1154D', 'R1158T', 'R1161E', 'R1163T', 'R1164E',
       'R1166D', 'R1168T', 'R1170J', 'R1173J', 'R1174T',
       'R1184M', 'R1195E', 'R1200T', 'R1201P',
       'R1202M', 'R1203T', 'R1204T', 'R1217T', 'R1222M',
       'R1223E', 'R1124J', 'R1232N', 'R1240T',
       'R1243T', 'R1247P', 'R1251M', 'R1260D', 'R1274T',
       'R1276D', 'R1284N']
# see dataQuality or log but couldn't get pairs.json when trying to load these
# ps21_problem_subs =  ['R1230J','R1216E'] # these have FR/PAL and PS2.1 sessions! But...
# R1230J has events or pairs load problems. R1216E gets split EEG error
# R1183T only is a single PS2.1 session. R1237C is 4 PS2.1 and 7 THs (no countdown period)
# R1155D only has one FR period (and 4 THs) but with diff montage
# new_problem_subs = ['R1196N','R1236J','R1175N'] # check run_stim_regression.log
# not_LTC_subs = ['R1161E','R1164E','R1166D','R1173J','R1247P'] # don't have LTC electrodes--7 sessions
'''Other sessions with issues:
PS21_df = df[(df.subject.isin(PS21_subs)) & (df.experiment==exp)]
PS21_df = PS21_df[((df.subject!='R1195E') | (df.session!=11)) & # see loadErrorTestbed to see why these don't load
                ((df.subject!='R1240T') | (df.session!=0)) &  
                ((df.subject!='R1243T') | (df.session==2)) & # 2 is good, other 4/5 sessions are bad
                ((df.subject!='R1260D') | (~df.session.isin([0,5]))) # 2 more bad ones of a bunch
                ((df.subject!='R1201P') | (df.session!=1)) ] # 1 is bad, other 2 are good
PS21_df'''
PS21_LTC_mask = [1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 0., 1.,
       1., 1., 0., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1.,
       1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1.]


## PS2 ## 
PS2_subs = np.array([u'R1056M', u'R1068J', u'R1077T', u'R1089P', u'R1096E', u'R1101T',
       u'R1104D', u'R1105E', u'R1108J', u'R1113T', u'R1114C', u'R1115T',
       u'R1120E', u'R1122E', u'R1124J', u'R1125T', u'R1131M', u'R1134T',
       u'R1136N', u'R1144E', u'R1153T', u'R1157C', u'R1162N', u'R1176M',
       u'R1161E', u'R1163T']) 
PS2_MTL_mask = [1., 1., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 0., 1., 1., 0., 1., 1., 1., 0., 0., 1.]

'''other issues:
PS2_df = PS2_df[((df.subject!='R1104D') | (df.session!=3)) &
                ((df.subject!='R1108J') | (df.session.isin([0,1,2,3])))] #even though 4:9 have same mont and more sessions, 
                        #0:3 actually load pairs so I did FC for those 4 sessions instead
# u'R1124J' none of sessions loaded pairs (put in Asana)
'''

'''Ethan's subs:
good_subs = np.array([u'R1056M', u'R1068J', u'R1077T', u'R1089P', u'R1096E', u'R1101T',
       u'R1104D', u'R1105E', u'R1108J', u'R1113T', u'R1114C', u'R1115T',
       u'R1120E', u'R1122E', u'R1124J', u'R1125T', u'R1131M', u'R1134T',
       u'R1136N', u'R1144E', u'R1153T', u'R1157C', u'R1162N', u'R1176M',
       u'R1161E', u'R1163T'])
'''
