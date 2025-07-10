import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import csv
import glob
import re
from scipy.optimize import curve_fit
import json
import pandas as pd
from scipy.stats import sem

import PreprocessingFunctions as pf

#plt.close('all')
SaveDir_Anal = r'/results/Analysis/'
SaveDir_Fig = r'/results/Fig_publication/'

if not os.path.exists(SaveDir_Anal):
    os.mkdir(SaveDir_Anal)
if not os.path.exists(SaveDir_Fig):
    os.mkdir(SaveDir_Fig)

folder_path = '/data'

# for visualization
Roi2Vis=[0,1]
AllPlot=0

# params for pre-processing
nFrame2cut = 100  #crop initial n frames
sampling_rate = 20 #individual channel (not total)
kernelSize = 1 #median filter
degree = 4 #polyfit
b_percentile = 0.70 #To calculare F0, median of bottom x%

sampling_rate=20
StimPeriod = 0.5 #sec for visualization
preW=100 #nframes for PSTH
LickWindow=5.0 #sec window length for Consummatory/Omission licks

#%%
AnalDir = r"/data/combined/behavior_734811_2024-09-20_13-36-31"
print("Now processing: " + AnalDir)


file1  = glob.glob(AnalDir + '/fib' + os.sep + "FIP_DataIso_*")[0]
file2 = glob.glob(AnalDir + '/fib' + os.sep + "FIP_DataG_*")[0]
file3 = glob.glob(AnalDir + '/fib' + os.sep + "FIP_DataR_*")[0]
SubjectID = AnalDir.split('/')[3].split('_')[1]

with open(file1) as f:
    reader = csv.reader(f)
    datatemp = np.array([row for row in reader])
    data1 = datatemp[1:,:].astype(np.float32)
    #del datatemp
    
with open(file2) as f:
    reader = csv.reader(f)
    datatemp = np.array([row for row in reader])
    data2 = datatemp[1:,:].astype(np.float32)
    #del datatemp
    
with open(file3) as f:
    reader = csv.reader(f)
    datatemp = np.array([row for row in reader])
    data3 = datatemp[1:,:].astype(np.float32)
    #del datatemp


TS_CS3file = glob.glob(AnalDir + '/behavior' + os.sep + "TS_CS3*")
TS_Lickfile = glob.glob(AnalDir + '/behavior' + os.sep + "TS_Lick*")
TS_Rewardfile = glob.glob(AnalDir + '/behavior' + os.sep + "TS_Reward_*")
TSdict = {}

with open(TS_CS3file[0], newline='', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    datatemp = np.array([row for row in csvreader])
    TSdict['CS3'] = datatemp.astype(np.float32)

with open(TS_Lickfile[0], newline='', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    datatemp = np.array([row for row in csvreader])
    TSdict['Lick'] = datatemp.astype(np.float32)

with open(TS_Rewardfile[0], newline='', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    header = next(csvreader, None)
    datatemp = np.array([row for row in csvreader])
    TSdict['Reward'] = datatemp.astype(np.float32)

# in case acquisition halted accidentally
Length = np.amin([len(data1),len(data2),len(data3)])

data1 = data1[0:Length] #iso       Time*[TS,ROI0,ROI1,ROI2,..]
data2 = data2[0:Length] #signal
data3 = data3[0:Length] #Stim

PMts= data2[:,0] #SignalTS
time_seconds = np.arange(len(data1)) /sampling_rate

#Preprocess
Ctrl_dF_F=np.zeros((data1.shape[0],data1.shape[1]))
G_dF_F=np.zeros((data1.shape[0],data1.shape[1]))
R_dF_F=np.zeros((data1.shape[0],data1.shape[1]))

for ii in range(data2.shape[1]-1):
    Ctrl_dF_F[:,ii] = pf.tc_preprocess(data1[:,ii+1], nFrame2cut, kernelSize, sampling_rate, degree, b_percentile)
    G_dF_F[:,ii] = pf.tc_preprocess(data2[:,ii+1] , nFrame2cut, kernelSize, sampling_rate, degree, b_percentile)
    R_dF_F[:,ii] = pf.tc_preprocess(data3[:,ii+1] , nFrame2cut, kernelSize, sampling_rate, degree, b_percentile)

# Stim Prep
TSFramesdict={}

for k in TSdict:
    Temp = TSdict[k]
    TempFrames = np.empty(len(Temp))
    for ii in range(len(Temp)):
        idx = np.argmin(np.abs(data1[:,0] - Temp[ii,0]))
        TempFrames[ii] = idx
    
    TSFramesdict[k] = TempFrames

RewardFrames = TSFramesdict['Reward']
CS3Frames = TSFramesdict['CS3']
LickFrames = TSFramesdict['Lick']

# Entire Session Plot
##
figT=plt.figure('Summary:' + AnalDir,figsize=(16, 16))
gs = gridspec.GridSpec(12,9)
plt.subplot(gs[0:4, 0:9])

for ii_ROI in range(len(Roi2Vis)):
    plt.plot(time_seconds, Ctrl_dF_F[:,Roi2Vis[ii_ROI]]*100 - ii_ROI*100, 'blue')
    plt.plot(time_seconds, G_dF_F[:,Roi2Vis[ii_ROI]]*100 - ii_ROI*100, 'green')
    plt.plot(time_seconds, R_dF_F[:,Roi2Vis[ii_ROI]]*100 - ii_ROI*100, 'magenta')
    plt.plot(time_seconds, np.zeros(len(time_seconds))-ii_ROI*100,'--k')
    
plt.plot(LickFrames/20, np.ones(len(LickFrames))*100, marker=3, markersize=10, color=[0, 0, 0, 0.5] ,label='Lick')

plt.xlabel('Time (seconds)')
plt.ylabel('dF/F (%)')
plt.title("SubjectID: " + os.path.basename(AnalDir) + "  Date: " + os.path.basename(os.path.dirname(AnalDir)))
plt.xlim([0, time_seconds[-1]])
plt.grid(True)

for ii in range(len(RewardFrames)):
    plt.axvspan(RewardFrames[ii]/20, RewardFrames[ii]/20 + StimPeriod, color = [0, 0, 1, 0.4])

for ii in range(len(CS3Frames)):
    plt.axvspan(CS3Frames[ii]/20, CS3Frames[ii]/20 + 1, color = [1, 0, 1, 0.4]) 

plt.axvspan(RewardFrames[0]/20, RewardFrames[0]/20, color = [0, 0, 1, 0.4],label='Reward')
#plt.axvspan(CS1Frames[0]/20, CS1Frames[0]/20, color = [1, 0, 0, 0.4],label='CS1')
#plt.axvspan(CS2Frames[0]/20, CS2Frames[0]/20, color = [0, 1, 0, 0.4],label='CS2') 
plt.axvspan(CS3Frames[0]/20, CS3Frames[0]/20, color = [1, 0, 1, 0.4],label='CS3')

plt.legend()

#plt.savefig(SaveDir_Fig + SubjectID + "_SessionTrace.pdf")
#plt.savefig(SaveDir_Fig + SubjectID + "_SessionTrace.png")
    
# PSTH functions (for multiple traces)
def PSTHmaker(TC, Stims, preW, postW):
    
    cnt = 0
    
    for ii in range(len(Stims)):
        if Stims[ii] - preW >= 0 and  Stims[ii] + postW < len(TC):
            
            A = int(Stims[ii]-preW) 
            B = int(Stims[ii]+postW)
            
            if cnt == 0:
                PSTHout = TC[A:B,:]
                cnt = 1
            else:
                PSTHout = np.dstack([PSTHout,TC[A:B,:]])
        else:
            if cnt == 0:
                PSTHout = np.zeros(preW+postW)
                cnt = 1
            #else:
                #PSTHout = np.dstack([PSTHout, np.zeros(preW+postW)])
    return PSTHout

#
def PSTHplot(PSTH, MainColor, SubColor, LabelStr):
    plt.plot(np.arange(np.shape(PSTH)[1])/20 - preW/sampling_rate, np.mean(PSTH.T,axis=1),label=LabelStr,color = MainColor)
    #plt.plot(np.arange(np.shape(PSTH)[1])/20 - 5, np.mean(PSTH.T,axis=1) + np.std(PSTH.T,axis=1)/np.sqrt(np.shape(PSTH)[0]),color = SubColor, linestyle = "dotted")
    #plt.plot(np.arange(np.shape(PSTH)[1])/20 - 5, np.mean(PSTH.T,axis=1) - np.std(PSTH.T,axis=1)/np.sqrt(np.shape(PSTH)[0]),color = SubColor, linestyle = "dotted")
    y11 =  np.mean(PSTH.T,axis=1) + np.std(PSTH.T,axis=1)/np.sqrt(np.shape(PSTH)[0])
    y22 =  np.mean(PSTH.T,axis=1) - np.std(PSTH.T,axis=1)/np.sqrt(np.shape(PSTH)[0])
    plt.fill_between(np.arange(np.shape(PSTH)[1])/20 - preW/sampling_rate, y11, y22, facecolor=SubColor, alpha=0.5)


# PSTH baseline subtraction (multi)
#dim0:trial, dim1:time
def PSTH_baseline(PSTH, preW):

    for ii in range(np.shape(PSTH)[2]):
        
        Trace_this = PSTH[:, :, ii]
        Trace_this_base = Trace_this[0:preW,:]
        Trace_this_subtracted = Trace_this - np.mean(Trace_this_base,axis=0)        
        
        if ii == 0:
            PSTHbase = Trace_this_subtracted
        else:
            PSTHbase = np.dstack([PSTHbase,Trace_this_subtracted])
    
    return PSTHbase


# Rew+/Rew-
# Trial csv handing

if bool(glob.glob(AnalDir + os.sep + "TrialN_*")) == True:
    file_TrialMat = glob.glob(AnalDir + os.sep + "TrialN_*")[0]
    df = pd.read_csv(file_TrialMat)

    Mat_CS1=np.where((df['TrialType']<=10) & (df['TrialType']>=1))[0]
    Mat_CS2=np.where((df['TrialType']<=20) & (df['TrialType']>=11))[0]
    Mat_CS3=np.where((df['TrialType']<=30) & (df['TrialType']>=21))[0]
    
    Mat_CS1R=np.where(df['TrialType']==1)[0]
    Mat_CS1UR=np.where((df['TrialType']<=10) & (df['TrialType']>=2))[0]
    Mat_CS2R=np.where((df['TrialType']<=15) & (df['TrialType']>=11))[0]
    Mat_CS2UR=np.where((df['TrialType']<=20) & (df['TrialType']>=16))[0]
    Mat_CS3R=np.where((df['TrialType']<=29) & (df['TrialType']>=21))[0]
    Mat_CS3UR=np.where(df['TrialType']==30)[0]
    
    RewardedCS1ind = np.where(np.isin(Mat_CS1, Mat_CS1R))[0]
    RewardedCS2ind = np.where(np.isin(Mat_CS2, Mat_CS2R))[0]
    RewardedCS3ind = np.where(np.isin(Mat_CS3, Mat_CS3R))[0]
    UnRewardedCS1ind = np.where(np.isin(Mat_CS1, Mat_CS1UR))[0]
    UnRewardedCS2ind = np.where(np.isin(Mat_CS2, Mat_CS2UR))[0]
    UnRewardedCS3ind = np.where(np.isin(Mat_CS3, Mat_CS3UR))[0]

if bool(glob.glob(AnalDir + os.sep + "Trial_Reversal_*")) == True:
    file_Reversal = glob.glob(AnalDir + os.sep + "Trial_Reversal_*")[0]
    df_Reversal = pd.read_csv(file_Reversal)
    TrialSwitched = df_Reversal['TrialSwitched'][0] 
    
    
    Mat_CS1=np.where((df['TrialType']<=10) & (df['TrialType']>=1) & (df['TrialNumber']<=TrialSwitched))[0]
    Mat_CS2=np.where((df['TrialType']<=20) & (df['TrialType']>=11))[0]
    Mat_CS3=np.where((df['TrialType']<=30) & (df['TrialType']>=21)& (df['TrialNumber']<=TrialSwitched))[0]
    
    Mat_CS1R=np.where(df['TrialType']==1 & (df['TrialNumber']<=TrialSwitched))[0]
    Mat_CS1UR=np.where((df['TrialType']<=10) & (df['TrialType']>=2) & (df['TrialNumber']<=TrialSwitched))[0]
    Mat_CS2R=np.where((df['TrialType']<=15) & (df['TrialType']>=11))[0]
    Mat_CS2UR=np.where((df['TrialType']<=20) & (df['TrialType']>=16))[0]
    Mat_CS3R=np.where((df['TrialType']<=29) & (df['TrialType']>=21) & (df['TrialNumber']<=TrialSwitched))[0]
    Mat_CS3UR=np.where(df['TrialType']==30 & (df['TrialNumber']<=TrialSwitched))[0]
    
    Mat_CS1=np.append(Mat_CS1, np.where((df['TrialType']<=30) & (df['TrialType']>=21) & (df['TrialNumber']>TrialSwitched))[0])
    Mat_CS3=np.append(Mat_CS3, np.where((df['TrialType']<=10) & (df['TrialType']>=1) & (df['TrialNumber']>TrialSwitched))[0])   
    
    Mat_CS1R=np.append(Mat_CS1R, np.where((df['TrialType']<=29) & (df['TrialType']>=21) & (df['TrialNumber']>TrialSwitched))[0])
    Mat_CS1UR=np.append(Mat_CS1UR, np.where((df['TrialType']==30) & (df['TrialNumber']>TrialSwitched))[0])
    Mat_CS3R=np.append(Mat_CS3R, np.where((df['TrialType']==1) & (df['TrialNumber']>TrialSwitched))[0])
    Mat_CS3UR=np.append(Mat_CS3UR, np.where((df['TrialType']<=10) & (df['TrialType']>=2) & (df['TrialNumber']>TrialSwitched))[0])
    
    RewardedCS1ind = np.where(np.isin(Mat_CS1, Mat_CS1R))[0]
    RewardedCS2ind = np.where(np.isin(Mat_CS2, Mat_CS2R))[0]
    RewardedCS3ind = np.where(np.isin(Mat_CS3, Mat_CS3R))[0]
    UnRewardedCS1ind = np.where(np.isin(Mat_CS1, Mat_CS1UR))[0]
    UnRewardedCS2ind = np.where(np.isin(Mat_CS2, Mat_CS2UR))[0]
    UnRewardedCS3ind = np.where(np.isin(Mat_CS3, Mat_CS3UR))[0]


# when no csv    
else:
    RewardedCS1ind=[]
    RewardedCS2ind=[]
    RewardedCS3ind=[]
    
    for ii in range(len(RewardFrames)):
#        idx_CS1 = np.argmin(np.abs(CS1Frames[:] - RewardFrames[ii]))
#        idx_CS2 = np.argmin(np.abs(CS2Frames[:] - RewardFrames[ii]))
        idx_CS3 = np.argmin(np.abs(CS3Frames[:] - RewardFrames[ii]))
        
#        if CS1Frames[idx_CS1] - RewardFrames[ii]>0:
#            idx_CS1 = idx_CS1-1
#        if CS2Frames[idx_CS2] - RewardFrames[ii]>0:
#            idx_CS2 = idx_CS2-1        
        if CS3Frames[idx_CS3] - RewardFrames[ii]>0:
            idx_CS3 = idx_CS3-1        
    
#        if CS1Frames[idx_CS1] == np.max([CS1Frames[idx_CS1],CS2Frames[idx_CS2],CS3Frames[idx_CS3]]):
#            RewardedCS1ind = np.append(RewardedCS1ind,idx_CS1)
#        if CS2Frames[idx_CS2] == np.max([CS1Frames[idx_CS1],CS2Frames[idx_CS2],CS3Frames[idx_CS3]]):
#            RewardedCS2ind = np.append(RewardedCS2ind,idx_CS2)
        if CS3Frames[idx_CS3] == np.max([CS3Frames[idx_CS3]]):
            RewardedCS3ind = np.append(RewardedCS3ind,idx_CS3)
    
#    UnRewardedCS1ind = np.setdiff1d(range(len(CS1Frames)),RewardedCS1ind)
#    UnRewardedCS2ind = np.setdiff1d(range(len(CS2Frames)),RewardedCS2ind)
    UnRewardedCS3ind = np.setdiff1d(range(len(CS3Frames)),RewardedCS3ind)

Psth_G_CS3R = PSTHmaker(G_dF_F*100, CS3Frames[RewardedCS3ind.astype(int)], 100, 300)
Psth_R_CS3R = PSTHmaker(R_dF_F*100, CS3Frames[RewardedCS3ind.astype(int)], 100, 300)
Psth_C_CS3R = PSTHmaker(Ctrl_dF_F*100, CS3Frames[RewardedCS3ind.astype(int)], 100, 300)
Psth_G_CS3R_base = PSTH_baseline(Psth_G_CS3R, 100)
Psth_R_CS3R_base = PSTH_baseline(Psth_R_CS3R, 100)
Psth_C_CS3R_base = PSTH_baseline(Psth_C_CS3R, 100)  

Psth_G_CS3UR = PSTHmaker(G_dF_F*100, CS3Frames[UnRewardedCS3ind.astype(int)], 100, 300)
Psth_R_CS3UR = PSTHmaker(R_dF_F*100, CS3Frames[UnRewardedCS3ind.astype(int)], 100, 300)
Psth_C_CS3UR = PSTHmaker(Ctrl_dF_F*100, CS3Frames[UnRewardedCS3ind.astype(int)], 100, 300)
Psth_G_CS3UR_base = PSTH_baseline(Psth_G_CS3UR, 100)
Psth_R_CS3UR_base = PSTH_baseline(Psth_R_CS3UR, 100)
Psth_C_CS3UR_base = PSTH_baseline(Psth_C_CS3UR, 100)

        
##
ymin=np.empty(len(Roi2Vis)+1)
ymax=np.empty(len(Roi2Vis)+1)
for ii in range(len(Roi2Vis)):
    ymax[ii]=np.max([
    np.max(np.mean(Psth_G_CS3R_base[:,Roi2Vis[ii],:],axis=1)),
    np.max(np.mean(Psth_G_CS3UR_base[:,Roi2Vis[ii],:],axis=1))])
    
    ymin[ii]=np.min([
    np.min(np.mean(Psth_G_CS3R_base[:,Roi2Vis[ii],:],axis=1)),
    np.min(np.mean(Psth_G_CS3UR_base[:,Roi2Vis[ii],:],axis=1))])

ymax[ii+1]=np.max([
np.max(np.mean(Psth_R_CS3R_base[:,0,:],axis=1)),
np.max(np.mean(Psth_R_CS3UR_base[:,0,:],axis=1))])

ymin[ii+1]=np.min([
np.min(np.mean(Psth_R_CS3R_base[:,0,:],axis=1)),
np.min(np.mean(Psth_R_CS3UR_base[:,0,:],axis=1))])


figT=plt.figure('Summary:' + AnalDir)

    
for ii in range(len(Roi2Vis)):
    plt.subplot(gs[4 + ii*2:4 + ii*2+2, 0:3])
    PSTHplot(Psth_G_CS3R_base[:,Roi2Vis[ii],:].T, "g", "darkgreen", "R+")
    PSTHplot(Psth_C_CS3R_base[:,Roi2Vis[ii],:].T, "b", "darkblue", "Iso_R+")
    PSTHplot(Psth_G_CS3UR_base[:,Roi2Vis[ii],:].T, "m", "darkmagenta", "R-")
    PSTHplot(Psth_C_CS3UR_base[:,Roi2Vis[ii],:].T, "k", "k", "Iso_R-")    
    plt.ylim([ymin[ii]*1.1, ymax[ii]*1.1])
    plt.xlim([-5,15])
    plt.grid(True)
    plt.title("CS3(90%Rew) all trials, ROI-Green: " + str(ii))
    plt.xlabel('Time - Tone (s)')
    plt.axvspan(0, 1.0, color = [1, 0, 1, 0.4])
    plt.axvspan(2.0, 2.5, color = [0, 0, 1, 0.4])
    

    plt.subplot(gs[4 + ii*2: 4 + ii*2+2, 3:6])
    PSTHplot(Psth_R_CS3R_base[:,Roi2Vis[ii],:].T, "g", "darkgreen", "R+")
    PSTHplot(Psth_C_CS3R_base[:,Roi2Vis[ii],:].T, "b", "darkblue", "Iso_R+")
    PSTHplot(Psth_R_CS3UR_base[:,Roi2Vis[ii],:].T, "m", "darkmagenta", "R-")
    PSTHplot(Psth_C_CS3UR_base[:,Roi2Vis[ii],:].T, "k", "k", "Iso_R-")    
    plt.ylim([ymin[ii]*1.1, ymax[ii]*1.1])
    plt.xlim([-5,15])
    plt.grid(True)
    plt.title("CS3(90%Rew) all trials, ROI-Red: " + str(ii))
    plt.xlabel('Time - Tone (s)')
    plt.ylabel('dF/F%')
    plt.axvspan(0, 1.0, color = [1, 0, 1, 0.4])
    plt.axvspan(2.0, 2.5, color = [0, 0, 1, 0.4])
    
    plt.subplots_adjust(hspace = 0.5, wspace=0.25)
    plt.tight_layout()

print('TotalTrial: ' + str(np.sum([len(CS3Frames)])))
print('CS3Trial: ' + str(len(CS3Frames)))
print('CS3 Rewarded:' + str(len(RewardedCS3ind)) + ' (' + str(np.single(100*len(RewardedCS3ind)/len(CS3Frames))) + '%)') 

# Lick Quant

Lick_CS3=[]
Lick_CS3_post=[]
for ii in range(len(CS3Frames)):
    count1 = len([x for x in LickFrames if CS3Frames[ii] < x < CS3Frames[ii]+2*20])
    count2 = len([x for x in LickFrames if CS3Frames[ii]+2*20 < x < CS3Frames[ii]+7*20])
    Lick_CS3=np.append(Lick_CS3,count1) 
    Lick_CS3_post=np.append(Lick_CS3_post,count2) 

aveLick_CS3=np.mean(Lick_CS3)
semLick_CS3=np.std(Lick_CS3)/np.sqrt(len(Lick_CS3))

aveLick_CS3_UnR=np.mean(Lick_CS3_post[UnRewardedCS3ind.astype(int)])
semLick_CS3_UnR=np.std(Lick_CS3_post[UnRewardedCS3ind.astype(int)])/np.sqrt(len(Lick_CS3_post[UnRewardedCS3ind.astype(int)]))


figT=plt.figure('Summary:' + AnalDir)

plt.subplot(gs[10:12, 6:9])
plt.plot(Lick_CS3,label='Anticipatory')
plt.plot(Lick_CS3_post,label='Consumatory/Omission')
plt.plot(RewardedCS3ind, Lick_CS3_post[RewardedCS3ind.astype(int)], '.',color='blue',markersize=10)
plt.plot(UnRewardedCS3ind, Lick_CS3_post[UnRewardedCS3ind.astype(int)], '.',color='Red',markersize=10,label='UnRewarded')
plt.xlabel('trial#')
plt.title('CS3 AntiLick:' + str(round(aveLick_CS3,2)) + '+-' +str(round(semLick_CS3,2))+ ' Omission:'+ str(round(aveLick_CS3_UnR,2)) + '+-' +str(round(semLick_CS3_UnR,2)))
plt.subplots_adjust(hspace = 0.5, wspace=0.25)
plt.tight_layout()

#
aID=os.path.basename(AnalDir)
aDate=os.path.basename(os.path.dirname(AnalDir))
#plt.savefig(SaveDir_Fig + os.sep + 'Summary_' + aID + '_' + aDate + '.pdf')
#if bool(glob.glob(AnalDir + os.sep + "TrialN_*")) == True:


# dF/F trial-by-trial quant
CSall=np.sort(np.hstack(TSFramesdict['CS3']))

Resp_Cue = np.empty((len(CSall),Ctrl_dF_F.shape[1])) # Cue:CS onset-offset (1s)
Resp_Rew = np.empty((len(CSall),Ctrl_dF_F.shape[1]))  # Reward: Reward onset to +3s
Resp_Tail = np.empty((len(CSall),Ctrl_dF_F.shape[1]))  # tail:next cue - 2sec to next trial 
Resp_base = np.empty((len(CSall),Ctrl_dF_F.shape[1]))  # base:-2s-0ms
Resp_Cue_based = np.empty((len(CSall),Ctrl_dF_F.shape[1]))  # baseline subtracted
Resp_Rew_based = np.empty((len(CSall),Ctrl_dF_F.shape[1]))  # 
Resp_Tail_based = np.empty((len(CSall),Ctrl_dF_F.shape[1]))  # 

TC=G_dF_F*100 # At StimRig, GreenChannel

for ii in range(len(CSall)):
    Resp_Cue[ii,:] = np.mean(TC[int(CSall[ii]+1):int(CSall[ii]+20),:],axis=0)
    Resp_Rew[ii,:] = np.mean(TC[int(CSall[ii]+41):int(CSall[ii]+100),:],axis=0)
    if ii==len(CSall)-1:
        Resp_Tail[ii,:] = np.mean(TC[int(CSall[ii]+200):int(CSall[ii]+240),:],axis=0)
    else:
        Resp_Tail[ii,:] = np.mean(TC[int(CSall[ii+1]-40):int(CSall[ii+1]),:],axis=0)

    Resp_base[ii,:] = np.mean(TC[int(CSall[ii]-40):int(CSall[ii]),:],axis=0)

    #relative to local mean
    Resp_Cue_based[ii,:] = Resp_Cue[ii,:] - Resp_base[ii,:] 
    Resp_Rew_based[ii,:] = Resp_Rew[ii,:] - Resp_base[ii,:] 
    Resp_Tail_based[ii,:] = Resp_Tail[ii,:] - Resp_base[ii,:] 

#
Psth_G_CSall = PSTHmaker(G_dF_F*100, CSall, 100, 300)
Psth_C_CSall = PSTHmaker(Ctrl_dF_F*100, CSall, 100, 300)
Psth_G_CSall_base = PSTH_baseline(Psth_G_CSall, 100)
Psth_C_CSall_base = PSTH_baseline(Psth_C_CSall, 100)

# PreciseTiming RewardResponse

def find_closest_larger_elements(arr1, arr2):
    arr2_sorted = np.sort(arr2)
    result = []

    for element in arr1:
        # Find the index of the smallest element in arr2_sorted that is larger than element
        idx = np.searchsorted(arr2_sorted, element, side='right')
        
        if idx < len(arr2_sorted):
            result.append(arr2_sorted[idx])
        else:
            break
            #result.append(None)  # Or handle the case where no larger element is found

    return np.array(result)


RewardConsumption = find_closest_larger_elements(RewardFrames, LickFrames)

Psth_G_RewardC = PSTHmaker(G_dF_F*100, RewardConsumption, 100, 300)
Psth_C_RewardC = PSTHmaker(Ctrl_dF_F*100, RewardConsumption, 100, 300)
Psth_R_RewardC = PSTHmaker(R_dF_F*100, RewardConsumption, 100, 300)
Psth_G_RewardC_base = PSTH_baseline(Psth_G_RewardC, 100)
Psth_C_RewardC_base = PSTH_baseline(Psth_C_RewardC, 100)
Psth_R_RewardC_base = PSTH_baseline(Psth_R_RewardC, 100)

plt.figure()
plt.subplot(2,3,1)
PSTHplot(Psth_G_RewardC_base[:,0,:].T, "green", "darkgreen",[])
#PSTHplot(Psth_C_RewardC_base[:,0,:].T, "b", "darkblue",[])
plt.axvspan(0, 2.5, color = [0, 0, 1, 0.2])
#plt.axvspan(2.0, 2.5, color = [0, 0, 1, 0.4])
plt.grid(True)
plt.xlim([-5, 15])
plt.title('aligned reward response R, Left')

plt.subplot(2,3,4)
PSTHplot(Psth_G_RewardC_base[:,1,:].T, "green", "darkgreen",[])
#PSTHplot(Psth_C_RewardC_base[:,0,:].T, "b", "darkblue",[])
plt.axvspan(0, 2.5, color = [0, 0, 1, 0.2])
#plt.axvspan(2.0, 2.5, color = [0, 0, 1, 0.4])
plt.grid(True)
plt.xlim([-5, 15])
plt.title('aligned reward response, Right')

plt.subplot(2,3,2)
PSTHplot(Psth_C_RewardC_base[:,0,:].T, "blue", "darkblue",[])
#PSTHplot(Psth_C_RewardC_base[:,0,:].T, "b", "darkblue",[])
plt.axvspan(0, 2.5, color = [0, 0, 1, 0.2])
#plt.axvspan(2.0, 2.5, color = [0, 0, 1, 0.4])
plt.grid(True)
plt.xlim([-5, 15])
plt.title('aligned reward response R, Left')

plt.subplot(2,3,5)
PSTHplot(Psth_C_RewardC_base[:,1,:].T, "blue", "darkblue",[])
#PSTHplot(Psth_C_RewardC_base[:,0,:].T, "b", "darkblue",[])
plt.axvspan(0, 2.5, color = [0, 0, 1, 0.2])
#plt.axvspan(2.0, 2.5, color = [0, 0, 1, 0.4])
plt.grid(True)
plt.xlim([-5, 15])
plt.title('aligned reward response, Right')


plt.subplot(2,3,3)
PSTHplot(Psth_R_RewardC_base[:,0,:].T, "darkred", "magenta",[])
#PSTHplot(Psth_C_RewardC_base[:,0,:].T, "b", "darkblue",[])
plt.axvspan(0, 2.5, color = [0, 0, 1, 0.2])
#plt.axvspan(2.0, 2.5, color = [0, 0, 1, 0.4])
plt.grid(True)
plt.xlim([-5, 15])
plt.title('aligned reward response R, Left')

plt.subplot(2,3,6)
PSTHplot(Psth_R_RewardC_base[:,1,:].T, "darkred", "magenta",[])
#PSTHplot(Psth_C_RewardC_base[:,0,:].T, "b", "darkblue",[])
plt.axvspan(0, 2.5, color = [0, 0, 1, 0.2])
#plt.axvspan(2.0, 2.5, color = [0, 0, 1, 0.4])
plt.grid(True)
plt.xlim([-5, 15])
plt.title('aligned reward response, Right')



#
yrange=[np.mean(Psth_G_RewardC_base[:,0,:],axis=1), np.mean(Psth_G_RewardC_base[:,1,:],axis=1)]
yMax=np.max(yrange)+0.5
yMin=np.min(yrange)-0.5

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
PSTHplot(Psth_G_RewardC_base[:,0,:].T, "goldenrod", "darkgoldenrod",[])
#PSTHplot(Psth_C_RewardC_base[:,0,:].T, "b", "darkblue",[])
plt.axvspan(0, 2.5, color = [0, 0, 1, 0.2])
#plt.axvspan(2.0, 2.5, color = [0, 0, 1, 0.4])
plt.grid(True)
plt.xlim([-5, 15])
plt.ylim([yMin, yMax])
plt.title(SubjectID + '  iGluSnFR3.v857.PDGFR')
plt.xlabel('Time - Reward Consumption (s)' )
plt.ylabel('dF/F (%)')

plt.subplot(1,2,2)
PSTHplot(Psth_G_RewardC_base[:,1,:].T, "blue", "darkblue",[])
#PSTHplot(Psth_C_RewardC_base[:,0,:].T, "b", "darkblue",[])
plt.axvspan(0, 2.5, color = [0, 0, 1, 0.2])
#plt.axvspan(2.0, 2.5, color = [0, 0, 1, 0.4])
plt.grid(True)
plt.xlim([-5, 15])
plt.ylim([yMin, yMax])
plt.title('iGluSnFR4s')
plt.xlabel('Time - Reward Consumption (s)' )
plt.ylabel('dF/F (%)')
plt.tight_layout

plt.savefig(SaveDir_Fig + "Figure5c_iGlu3vs4_PSTHs.pdf")
#plt.savefig(SaveDir_Fig + SubjectID + "_iGlu3vs4.png")
#
#np.save(SaveDir_Anal + SubjectID + "Psth_G_RewardC.npy", Psth_G_RewardC)
#np.save(SaveDir_Anal + SubjectID + "Psth_R_RewardC.npy", Psth_R_RewardC)
#np.save(SaveDir_Anal + SubjectID + "Psth_C_RewardC.npy", Psth_C_RewardC)

#
plt.figure(figsize=(12,4))
plt.plot(time_seconds, G_dF_F[:,0]*100 - 30, 'goldenrod')
plt.plot(time_seconds, G_dF_F[:,1]*100 - 20, 'blue')
plt.plot(LickFrames/20, np.ones(len(LickFrames)), marker=3, markersize=10, color=[0, 0, 0, 0.5] ,label='Lick')

for ii in range(len(RewardFrames)):
    plt.axvspan(RewardFrames[ii]/20, RewardFrames[ii]/20 + StimPeriod, color = [0, 0, 1, 0.4])

plt.xlabel('Time (seconds)')
plt.ylabel('dF/F (%)')
plt.title("SubjectID: " + SubjectID + "  Date: " + os.path.basename(os.path.dirname(AnalDir)))
plt.xlim([0, time_seconds[-1]])
plt.grid(True)
#plt.savefig(SaveDir_Fig + "Figure5b_iGlu3vs4_RawTraces.pdf")

#%%
plt.figure(figsize=(12,4))
plt.plot(time_seconds, G_dF_F[:,0]*100 - 30, 'goldenrod')
plt.plot(time_seconds, G_dF_F[:,1]*100 - 20, 'blue')
plt.plot(LickFrames/20, np.ones(len(LickFrames)), marker=3, markersize=10, color=[0, 0, 0, 0.5] ,label='Lick')

for ii in range(len(RewardConsumption)):
    plt.axvspan(RewardConsumption[ii]/20, RewardConsumption[ii]/20 + 0.2, color = [0, 0, 1, 0.4])

plt.xlabel('Time (seconds)')
plt.ylabel('dF/F (%)')
plt.title("SubjectID: " + SubjectID + "  Date: " + os.path.basename(os.path.dirname(AnalDir)))
plt.xlim([950, 1000])
plt.savefig(SaveDir_Fig + "Figure5b_iGlu3vs4_Traces.pdf")
# %%
