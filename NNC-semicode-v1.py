#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#from ARAS datasets Selected House A
#Anomaly state is checked for resident 1 in House A
#We divided the datasets into two equal parts
#We used all the second 15 days of the datasets in  to check for anomalies
#You can change this division

# DATASETS :
#           Change to CSV format 
#           Add a row (first row) for the sensor name and resident number (In short)
#           Add a column to show the seconds.


#The first 15 days (two weeks) are used for the initial learning of the algorithm.
#from the sixteenth day, NNC begins to be calculated.

# Pay Attention:
#           To use this code, the reading part of the datasets must be changed

###............................................. Main Part ............................................

import random
import statistics
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score

#........................................... Functions ..................................... :

def randomcheck(testday ,resident, classcode):
    subNNC = 0
    y = testday['r'+str(resident)]
    if classcode == 10 or classcode == 20:
        testday.drop(['r1'], axis=1 , inplace = True)
        testday.drop(['r2'], axis=1 , inplace = True)
    else:
        testday.drop(['r'+str(resident)], axis=1 , inplace = True)
    T = 0
    for i in range (3600):
        rt = random.randint(1,86400)
        yt = y[rt-1:rt]
        xt = testday[rt-1:rt]
        yp = globals()["dt"+str(classcode)].predict(xt)
        if int(yt) == yp:
            T = T + 1
    if T < 900:
        subNNC = 20
    if T > 900 and T < 1400:
        subNNC = 14   
    if T > 1400 and T < 1900:
        subNNC = 12
    if T > 1900 and T < 2400:
        subNNC = 5
    if T > 2400 and T < 2900:
        subNNC = 3
    if T > 2900 and T < 3400:
        subNNC = 1
    return subNNC

#.............////////////////////..............

def TimePart(day,resident):
    for g in range(27):
        globals()["p"+str(g+1)] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for n in range(24):
        for t in range(27):
            globals()["h"+str(t+1)] = 0
        for s in range((n)*3600,((n+1)*3600)-1):
            o = day['r'+str(resident)][s]
            globals()["h"+str(o)] = 1 + globals()["h"+str(o)]
            if globals()["h"+str(o)] > 60:
                globals()["p"+str(o)][n] = 1
    out = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13 , p14,
           p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27]
    return out

#.............////////////////////..............

def predNorm (x, classcode):
    
    for b in range(27):
        globals()["p"+str(b+1)] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for n in range(24):
        for b in range(27):
            globals()["y_pred"+str(b+1)] = 0
        for s in range((n)*3600,((n+1)*3600)-1):
            pred = int(globals()["dt"+str(classcode)].predict(x[s:s+1]))
            globals()["y_pred"+str(pred)] = 1 + globals()["y_pred"+str(pred)] 
            if globals()["y_pred"+str(pred)] > 60:
                globals()["p"+str(pred)][n] = 1
    ex = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13 , p14,
          p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27]               
    return ex

def checkpred (tday, resident, classcode):
    subNNC = 0
    real = TimePart(tday,resident)
    if classcode == 10 or classcode == 20:
        tday.drop(['r1'], axis=1 , inplace = True)
        tday.drop(['r2'], axis=1 , inplace = True)
    else:
        tday.drop(['r'+str(resident)], axis=1 , inplace = True)
    x_test = tday
    pred = predNorm(x_test, classcode)
    for d in range(27):
        if real[d] != pred[d]:
            subNNC = 1 + subNNC
    return subNNC

#.............////////////////////..............

def norm (tday,resident):
    subNNC = 0
    for g in range(27):
        globals()["k"+str(g+1)] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

#In this section, the datasets of the days should be entered in order
    for i in range(15):
        d = pd.read_csv('c://DAY_0'+str(i+1)+'.csv' , encoding = 'ansi')
        y = TimePart(d,resident)    
        for f in range(27):
            for p in range(24):
                globals()["k"+str(f+1)][p] = globals()["k"+str(f+1)][p] + y[f][p]

    j = TimePart(tday,resident)
    for f in range(27):
        for k in range(24):
            timechart = globals()["k"+str(f+1)][k]
            tcharttest = j[f][k]
            if timechart < 3 and tcharttest == 1:
                subNNC = subNNC + 1
            if timechart > 12 and tcharttest == 0:
                subNNC = subNNC + 1
    return subNNC

#.............////////////////////..............

def sectable (dy, resident):

    for g in range(27):
        globals()["sec"+str(g+1)] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]                
    for n in range(24):
        for x in range((n)*3600,((n+1)*3600)-1):
            o = dy['r'+str(resident)][x]
            globals()["sec"+str(o)][n] = 1 + globals()["sec"+str(o)][n]
    secout = [sec1,sec2,sec3,sec4,sec5,sec6,sec7,sec8,sec9,sec10,sec11,sec12,sec13,sec14,
              sec15,sec16,sec17,sec18,sec19,sec20,sec21,sec22,sec23,sec24,sec25,sec26,sec27]
    return secout

def MaxMin (resident):
    for g in range(27):
        globals()["ma"+str(g+1)] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        globals()["mi"+str(g+1)] = [4000,4000,4000,4000,4000,4000,4000,4000,4000,4000,4000,4000,
                                    4000,4000,4000,4000,4000,4000,4000,4000,4000,4000,4000,4000]
                    
#In this section, the datasets of the days should be entered in order
    for i in range(15):
        day = pd.read_csv('c://DAY_0'+str(i+1)+'.csv' , encoding = 'ansi')
        ts = sectable(day,resident)
        for w in range(27):
            for q in range(24):
                if ts[w][q] > globals()["ma"+str(w+1)][q]:
                    globals()["ma"+str(w+1)][q] = ts[w][q]
                if ts[w][q] < globals()["mi"+str(w+1)][q]:
                    globals()["mi"+str(w+1)][q] = ts[w][q]   
    mintabl = [mi1,mi2,mi3,mi4,mi5,mi6,mi7,mi8,mi9,mi10,mi11,mi12,mi13,mi14,
                mi15,mi16,mi17,mi18,mi19,mi20,mi21,mi22,mi23,mi24,mi25,mi26,mi27]
    maxtabl = [ma1,ma2,ma3,ma4,ma5,ma6,ma7,ma8,ma9,ma10,ma11,ma12,ma13,ma14,
                ma15,ma16,ma17,ma18,ma19,ma20,ma21,ma22,ma23,ma24,ma25,ma26,ma27] 
    return (mintabl , maxtabl)

def MaxMincheck(testday, resident):
    subNNC = 0
    st = sectable(testday, resident)
    mi , ma = MaxMin(resident)
    for p in range(27):
        for w in range(24):
            if st[p][w] < mi[p][w]:
                subNNC = 4 + subNNC
            if st[p][w] > ma[p][w]:
                subNNC = 2 + subNNC
    return subNNC

#.............////////////////////..............

def TimeJob (daay, resident):
    
    for v in range(27):
        globals()["ac"+str(v+1)] = [] 
    for l in range(27):
        globals()["st"+str(l+1)] = -1 
    t = 0   
    while t < 86400:
        z = t
        o = daay['r'+str(resident)][t]
        globals()["st"+str(o)] = 1 + globals()["st"+str(o)]
        globals()["ac"+str(o)].append([z,1])
        globals()["ac"+str(o)][globals()["st"+str(o)]][1] = 1 + globals()["ac"+str(o)][globals()["st"+str(o)]][1]
        for a in range(z+1,86400):
            if daay['r'+str(resident)][a] == o:
                globals()["ac"+str(o)][globals()["st"+str(o)]][1] = 1 + globals()["ac"+str(o)][globals()["st"+str(o)]][1]
                if a == 86399:
                    t = 86400
            else:
                t = a
                break
        t = t + 1

    acout = [len(ac1),len(ac2),len(ac3),len(ac4),len(ac5),len(ac6),len(ac7),len(ac8),len(ac9),len(ac10),
             len(ac11),len(ac12),len(ac13),len(ac14),len(ac15),len(ac16),len(ac17),len(ac18),len(ac19),
             len(ac20),len(ac21),len(ac22),len(ac23),len(ac24),len(ac25),len(ac26),len(ac27)]
    
    for m in range(27):
        globals()["tts"+str(m+1)] = 0
    
    for g in range(27):
        h = globals()["ac"+str(g+1)]
        for s in range(len(h)):
            globals()["tts"+str(g+1)] = globals()["ac"+str(g+1)][s][1] + globals()["tts"+str(g+1)]
            
    ttsout = [tts1, tts2, tts3, tts4, tts5, tts6, tts7, tts8, tts9, tts10, tts11, tts12, tts13, tts14,
             tts15, tts16, tts17, tts18, tts19, tts20, tts21, tts22, tts23, tts24, tts25, tts26, tts27]
    
    return acout , ttsout
    
def numericalCheck (day, resident):
    subNNC = 0
    for v in range(27):
        globals()["c"+str(v+1)] = 0 
    for f in range(27):
        globals()["s"+str(f+1)] = 0
    for k in range(27):
        globals()["av"+str(k+1)] = 0
    for d in range(27):
        globals()["mc"+str(d+1)] = 0
    for s in range(27):
        globals()["mt"+str(s+1)] = 0
    for s in range(27):
        globals()["mi"+str(s+1)] = 90000
#In this section, the datasets of the days should be entered in order
    for i in range(15):
        daaay = pd.read_csv('c://DAY_0'+str(i+1)+'.csv' , encoding = 'ansi')
        count , total = TimeJob(daaay,resident)
        for j in range(27):
            if count[j] > globals()["mc"+str(j+1)]:
                globals()["mc"+str(j+1)] = count[j]
            if total[j] > globals()["mt"+str(j+1)]:
                globals()["mt"+str(j+1)] = total[j]
            if total[j] > 0:
                if total[j] < globals()["mi"+str(j+1)]:
                    globals()["mi"+str(j+1)] = total[j]
            if count[j] > 0:
                globals()["c"+str(j+1)] = globals()["c"+str(j+1)] +1
                globals()["s"+str(j+1)] = globals()["s"+str(j+1)] +total[j]
    for g in range(27):
        if globals()["c"+str(g+1)] > 0 :
            globals()["av"+str(g+1)] = globals()["s"+str(g+1)]/globals()["c"+str(g+1)]
                   
    cnt , ttl = TimeJob(day,resident)
    for r in range(27):
        if ttl[r] > 0 and ttl[r] < globals()["mi"+str(r+1)]:
            subNNC = subNNC + 1
        if cnt[r] > globals()["mc"+str(r+1)]:
            subNNC = subNNC + 1
        if ttl[r] > globals()["mt"+str(r+1)]:
            subNNC = subNNC + 1
        if ttl[r] > ((globals()["av"+str(r+1)])*1.5):
            subNNC = subNNC + 1
        if ttl[r] < ((globals()["av"+str(r+1)])*0.5):
            subNNC = subNNC + 1
        if (globals()["c"+str(r+1)]) > 12 and cnt[r] < 1:
            subNNC = subNNC + 1
        if (globals()["c"+str(r+1)]) < 3 and cnt[r] > 0:
            subNNC = subNNC + 1
    return subNNC
#.............////////////////////..............

#In this section, the datasets of the days should be entered in order

for i in range(15):
    globals()["df0" + str(i+1)] = pd.read_csv('c://DAY_0'+str(i+1)+'.csv' , encoding = 'ansi')

w0 = df01.append(df02, ignore_index=True)
for j in range(12):
    globals()["w"+str(j+1)] = globals()["w"+str(j)].append(globals()["df0" + str(j+3)] , ignore_index=True)
    if j==11:
        dataR10 = globals()["w"+str(j+1)].append(globals()["df0" + str(j+4)] , ignore_index=True)
        dataR11 = globals()["w"+str(j+1)].append(globals()["df0" + str(j+4)] , ignore_index=True)
        dataR20 = globals()["w"+str(j+1)].append(globals()["df0" + str(j+4)] , ignore_index=True)
        dataR21 = globals()["w"+str(j+1)].append(globals()["df0" + str(j+4)] , ignore_index=True)

print("\nWELCOME\n")

## ------------------------------------------------------ Resident 1 --------------------------------------------------

dt11 = tree.DecisionTreeClassifier(criterion='gini') 
y11  = dataR11.r1
dataR11.drop(['r1'], axis=1 , inplace = True)
x11 = dataR11
dt11.fit(x11,y11)

dt10 = tree.DecisionTreeClassifier(criterion='gini') 
y10  = dataR10.r1
dataR10.drop(['r1'], axis=1 , inplace = True)
dataR10.drop(['r2'], axis=1 , inplace = True)
x10 = dataR10
dt10.fit(x10,y10)

subNNC11 = []
subNNC21 = []
subNNC10 = []
subNNC20 = []
subNNC3 = []
subNNC4 = []
subNNC5 = []

#In this section, the 16th to the 30th days are called for review

for daycounter in range(15):
    
#sub-NNC for Resident 1, Considering the behavior of the resident 2 in the last 15 days:
    dataSetsNNC11 = pd.read_csv('c://DAY_0'+str(daycounter+16)+'.csv' , encoding = 'ansi')
    sbNNC11 = randomcheck(dataSetsNNC11,1,11)
    subNNC11.append(sbNNC11)
    dataSetsNNC21 = pd.read_csv('c://DAY_0'+str(daycounter+16)+'.csv' , encoding = 'ansi')
    sbNNC21 = checkpred(dataSetsNNC21,1,11)
    subNNC21.append(sbNNC21)
    
#sub-NNC for Resident 1, Regardless of resident behavior 2 in the last 15 days:
    dataSetsNNC10 = pd.read_csv('c://DAY_0'+str(daycounter+16)+'.csv' , encoding = 'ansi')
    sbNNC10 = randomcheck(dataSetsNNC10,1,10)
    subNNC10.append(sbNNC10)
    dataSetsNNC20 = pd.read_csv('c://DAY_0'+str(daycounter+16)+'.csv' , encoding = 'ansi')
    sbNNC20 = checkpred(dataSetsNNC20,1,10)
    subNNC20.append(sbNNC20)

    dataSetsNNC3 = pd.read_csv('c://DAY_0'+str(daycounter+16)+'.csv' , encoding = 'ansi')
    sbNNC3 = numericalCheck(dataSetsNNC3,1)
    subNNC3.append(sbNNC3)
    dataSetsNNC4 = pd.read_csv('c://DAY_0'+str(daycounter+16)+'.csv' , encoding = 'ansi')
    sbNNC4 = MaxMincheck(dataSetsNNC4,1)
    subNNC4.append(sbNNC4)
    dataSetsNNC5 = pd.read_csv('c://DAY_0'+str(daycounter+16)+'.csv' , encoding = 'ansi')
    sbNNC5 = norm(dataSetsNNC5,1)
    subNNC5.append(sbNNC5)

#************************************************ FINAL NNC FOR RESIDENT 1 ************************************************                

DailyNNC = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(15):
    DailyNNC1[i] = subNNC11[i]+subNNC21[i]+subNNC10[i]+subNNC20[i]+subNNC3[i]+subNNC4[i]+subNNC5[i]
NNC1 = np.mean(DailyNNC1)
for i in range(15):
    if DailyNNC1[i] > (NNC1*1.3):
            print("on",i+16,"day Abnormal state was diagnosed for the Resident 1")


print(DailyNNC1)
print("Final Average NNC for Resident 1 in the last 15 days : ",NNC1)

#*************************************************************************************************************************


# In[ ]:





# In[ ]:


for daycounter in range(14):
    dataSetsNNC10 = pd.read_csv('c://DAY_0'+str(daycounter+16)+'.csv' , encoding = 'ansi')
    sbNNC10 = randomcheck(dataSetsNNC10,1,10)
    subNNC10.append(sbNNC10)


# In[ ]:




