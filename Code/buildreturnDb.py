# dfsdfsds encoding: utf-8

import pandas as pd
import numpy as np
from datetime import date
import datetime
import time
#from utility import str2date
#from utility import date2str
import pickle


dfClose = pd.read_csv('stoneClosingPx_1228.csv', encoding='utf-8', index_col='Date')
dfClose.index = pd.to_datetime(dfClose.index)
dfOpen = pd.read_csv('stoneOpeningPx_1228.csv', encoding='utf-8', index_col='Date')
dfOpen.index = pd.to_datetime(dfOpen.index)

dfPreClose = pd.read_csv('stonePreClosingPx_1228.csv', encoding='utf-8', index_col='Date')
dfPreClose.index = pd.to_datetime(dfPreClose.index)

dfTrend = pd.DataFrame()
dfTrend[dfOpen.columns]=dfOpen[dfOpen.columns]
dfTrendRelative = pd.DataFrame()
dfTrendRelative[dfOpen.columns]=dfOpen[dfOpen.columns]
numDate = len(dfTrend.index)
for idt in range(numDate):
    print 'processing',idt+1,'of',numDate, '...'
    dt = dfTrend.index[idt]
    for stockID in dfTrend.columns:
        #print 'processing', stockID
        vOpen_1 = 0
        vClose_1=0
        vPreClose_1=0
        #vOpen_2=0 #该值没用上
        vClose_2 = 0
        vPreClose_2=0
        idtN1=-1#后一天日期
        idtN2=-1#后两天日期
        for idtN1Candidate in range(idt+1,min(numDate,idt+3)):#后一天要在两天内
            dt1 = dfOpen.index[idtN1Candidate]
            vOpenCandidate = dfOpen.loc[dt1,stockID]
            if np.isnan(vOpenCandidate) == False:
                vOpen_1 = vOpenCandidate
                vClose_1=dfClose.loc[dt1,stockID]
                vPreClose_1=dfPreClose.loc[dt1,stockID]
                vClose_2 = vOpen_1#提前赋值，确保结果稳定
                idtN1 = idtN1Candidate
                break
        for idtN2Candidate in range(idtN1+1,min(numDate,idtN1+3)):#后两天要在后一天的两天内
            dt2 = dfClose.index[idtN2Candidate]
            vCloseCandidate = dfClose.loc[dt2, stockID]
            if np.isnan(vCloseCandidate) == False:
                vClose_2 = vCloseCandidate
                vPreClose_2=dfPreClose.loc[dt2, stockID]
                idtN2 = idtN2Candidate
                break
        if idtN1 != -1 and idtN2 != -1:
            ratio = vPreClose_1/vPreClose_2*vClose_2/vClose_1 #复权
            vOpen_1=vOpen_1*ratio#复权后的后一天开盘价，和后两天收盘价等在同一起跑线
            dfTrend.loc[dt, stockID] = vClose_2-vOpen_1
            dfTrendRelative.loc[dt, stockID] = (vClose_2 - vOpen_1)/vOpen_1
        else:
            dfTrend.loc[dt, stockID] = np.nan
            dfTrendRelative.loc[dt, stockID] = np.nan
dfTrend.to_csv('stoneReturnPx_1228.csv',encoding='utf-8',index=True)
dfTrendRelative.to_csv('stoneReturnRelativePx_1228.csv',encoding='utf-8',index=True)
quit()