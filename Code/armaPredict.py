# -*-coding:utf-8-*-
import pandas as pd
import numpy as np
from math import sqrt
from numpy.random import normal
from numpy import mean, median
from statsmodels.tsa.arima_model import ARMA
import sys
from dateutil.relativedelta import relativedelta
from copy import deepcopy
import matplotlib.pyplot as plt
import datetime
import time
from statsmodels.tsa.stattools import adfuller
from arima_model import arima_model,testStationarity
import arch



if __name__ == '__main__':
    #dfResults = pd.read_csv('returnResult_1.csv', encoding='utf-8')
    #dfResults = dfResults.sort_values('Date')  # 按照日期进行排序
    #dfResults['Date'] = pd.to_datetime(dfResults['Date'])
    #dfTest = dfResults[dfResults['Date']>=pd.to_datetime('2015-11-22')]
    #numTotal = len(dfTest.values)
    #nCorrect=0
    #nUp=0
    #nDown=0
    #for i in dfTest.index:
    #    vP = dfTest.loc[i,'ReturnPredict']
    #    vR = dfTest.loc[i,'ReturnReal']
    #    if vP > 0 and vR > 0:
    #        nUp+=1
    #    elif vP < 0 and vR < 0:
    #        nDown+=1
    #nCorrect = nUp + nDown
    #print numTotal,nCorrect,nUp,nDown
    #quit()


    dfReturn = pd.read_csv('stoneReturnRelativePx_1228.csv', encoding='utf-8',index_col='Date')
    dfReturn.index = pd.to_datetime(dfReturn.index)

    dfM = pd.read_csv('dFM-big-oldLabel.csv', encoding='utf-8')


    dfTarget = pd.read_csv('18wStockInfo4ARMA.csv', encoding='utf-8', index_col='Date')
    dfTarget.index = pd.to_datetime(dfTarget.index)

    dfResults = pd.DataFrame()
    dfResults['FileName']=dfTarget['FileName'].values
    dfResults['StockID'] = dfTarget['StockID'].values
    dfResults['Date'] = dfTarget.index.values
    colReturnPredict=[]
    #colReturnReal = []
    colReturnstderr=[]
    for i in dfResults.index:
        print 'processing',i+1,'of',len(dfResults.index)
        fn = dfResults.loc[i,'FileName']
        stockID = dfResults.loc[i,'StockID']
        stockID = stockID.replace('sz','')
        stockID = stockID.replace('sh', '')
        dt = pd.to_datetime(dfResults.loc[i,'Date'])
        dt_1=dt + relativedelta(days=-1)
        numKeyFound=0
        dtTmp = dt
        keyPos=dt_1
        idxUpper = 0
        while numKeyFound < 200:
            if idxUpper > 50:#往前找50天都没找到
                keyPos = dt + relativedelta(days=-1)
                break
            idxUpper+=1
            if dtTmp in dfReturn[stockID].index:
                keyPos = dtTmp
                numKeyFound+=1

            dtTmp = dtTmp + relativedelta(days=-1)

        dt_1 = keyPos

        sequence = deepcopy(dfReturn[stockID][dfReturn.index < dt_1])
        sequence=sequence.dropna()
        if len(sequence) > 5:
            sequence1 = deepcopy(sequence.values)
            #print sequence1
            # stati = testStationarity(sequence)
            # if stati['p-value']>0.3:
            #    print 'non-stationarity:',stati['p-value'],stockID,dt
            model = arima_model(sequence1)
            #  这里使用模型参数自动识别
            model.get_proper_model()
            # model.certain_model(1, 1)
            print 'bic:', model.bic, 'p:', model.p, 'q:', model.q
            meanARMA = model.properModel.forecast(steps=2)[0][1]
            errARMA = model.properModel.forecast(steps=2)[1][1]


            #ARMA拟合结果的长度要短于输入
            #ppp = model.properModel.predict()
            #predict_ts = pd.Series(data=ppp, index=sequence.index[-len(ppp):])
            #real_ts = pd.Series(data=sequence1, index=sequence.index)


            #residual = sequence1[1:] - ppp#残差，用Grach模型拟合


            #am = arch.arch_model(residual,vol='GARCH', p=1,q=1)
            #res = am.fit(disp='off')
            #last_obs = len(residual)-1
            #fMean = res.forecast(horizon=2,start=last_obs).mean.iloc[last_obs][1]
            #fV = res.forecast(horizon=2, start=last_obs).variance.iloc[last_obs][1]
            #fRV = res.forecast(horizon=2, start=last_obs).residual_variance.iloc[last_obs][1]
            ##print meanARMA,fMean,fV,fRV
            #rv = normal( size=1000)
            #rvSum = rv * sqrt(fV)
            #offsetGrach = median(rvSum)


            rtn = meanARMA #+ offsetGrach
            err = errARMA

        else:
            rtn = np.random.normal(0, 0.1, 1)[0]
            err = 100

        #print model.properModel.forecast(steps=1)[0]
        #print model.properModel.forecast(steps=1)[1]
        #print model.properModel.forecast(steps=1)[2]
        colReturnPredict.append(rtn)
        colReturnstderr.append(err)
        #after = dfReturn[stockID][dfReturn.index > dt]
        #after = after.dropna()
        #if len(after) > 0:
        #    rtnReal = after[0]
        #else:
        #    rtnReal=np.nan
        #rtnReal=dfM[dfM['FileName'] == fn]['AfterOneDayTrend_OpenClose'].values[0]
        #colReturnReal.append(rtnReal)  #这种方式非常慢！！！
        print 'predict:', rtn, err#, rtnReal

    #quit()
    dfReal = pd.DataFrame()
    dfReal['FileName']=dfM['FileName']
    dfReal['ReturnReal'] = dfM['AfterOneDayTrendUnlimited_OpenClose']
    dfResults = pd.merge(dfResults, dfReal, how='left', on='FileName')
    dfResults['ReturnPredict']=colReturnPredict
    #dfResults['ReturnReal'] = colReturnReal
    dfResults['ReturnErr'] = colReturnstderr
    dfResults.to_csv('18wreturnResult_Relative_1.csv',encoding='utf-8',index=False)

    dfResults = dfResults.sort_values('Date')  # 按照日期进行排序
    #dfResults['Date'] = pd.to_datetime(dfResults['Date'])
    dfTest = dfResults[dfResults['Date']>=pd.to_datetime('2016-03-23')]
    numTotal = len(dfTest.values)
    nCorrect=0
    nUp=0
    nDown=0
    for i in dfTest.index:
        vP = dfTest.loc[i,'ReturnPredict']
        vR = dfTest.loc[i,'ReturnReal']
        if vP > 0 and vR > 0:
            nUp+=1
        elif vP < 0 and vR < 0:
            nDown+=1
    nCorrect = nUp + nDown
    print 'numTotal,nCorrect,nUp,nDown'
    print numTotal,nCorrect,nUp,nDown
    print 'Relative'

    quit()





        #for j in i.index:
            #if i.loc[j,'FileName'] in setFileName:
                #i.loc[j, 'Label'] = 1 - i.loc[j,'Label']
    #idxTotal = pd.date_range(start=dfStock.index[0], end=dfStock.index[-1])#直接在整个数据上进行填充不妥，应该用股票各自的时间窗
    #dfStock = dfStock.reindex(idxTotal)
    #dfStock.fillna(method='bfill',inplace=True)#直接在整个数据上进行填充不妥，应该用股票各自的时间窗


    stock0 = dfStock['600100']
    #plt.figure(facecolor='white')
    #stock0.plot(color='blue', label='Predict')
    #plt.show()




    # 数据预处理
    stock1 = stock0.diff(1)  # 一阶差分
    stock1.dropna(inplace=True)


    # 模型拟合

    model = arima_model(stock1.values)
    #  这里使用模型参数自动识别
    #model.get_proper_model()
    model.certain_model(1,1)
    print 'bic:', model.bic, 'p:', model.p, 'q:', model.q
    print model.properModel.forecast(steps=1)[0]
    print model.properModel.forecast(steps=1)[1]
    print model.properModel.forecast(steps=1)[2]
    #print model.forecast_next_day_value(type='month')
    #quit()

    # 预测结果还原
    predict_ts = model.properModel.predict()
    numTotal = len(predict_ts)
    nCorrect=0
    for i in range(numTotal):
        vReal = stock1[i]
        vPredict = predict_ts[i]
        if vReal > 0 and  vPredict > 0:
            nCorrect+=1
        elif vReal < 0 and vPredict < 0:
            nCorrect += 1
        elif vReal == 0 and vPredict == 0:
            nCorrect += 1
        #elif vReal == 0:
        #    numTotal-=1
    print nCorrect,numTotal
    quit()

    #diff_shift_ts = ts_diff_1.shift(1)
    #diff_recover_1 = predict_ts.add(diff_shift_ts)
    #rol_shift_ts = rol_mean.shift(1)
    #diff_recover = diff_recover_1.add(rol_shift_ts)
    #rol_sum = ts_log.rolling(window=11).sum()
    #rol_recover = diff_recover*12 - rol_sum.shift(1)
    #log_recover = np.exp(rol_recover)
    #log_recover.dropna(inplace=True)
    diff_shift_ts = stock0.shift(1)
    diff_recover_1 = predict_ts.add(diff_shift_ts)
    diff_recover_1.dropna(inplace=True)



    # 预测结果作图
    fut1Year = [pd.tslib.Timestamp(month_step(df.index[-1], w)) for w in range(1, 13)]
    fut1YearData = model.properModel.forecast(steps=12)[0]
    newF = pd.Series(data=fut1YearData, index=fut1Year)


    stock1 = stock1[diff_recover_1.index]
    stock0 = stock0[diff_recover_1.index]#去除dropNAN的数据范围
    plt.figure(facecolor='white')
    diff_recover_1.plot(color='blue', label='Predict')
    stock0.plot(color='red', label='Original')
    #newF.plot(color='black', label='future')
    plt.legend(loc='best')
    plt.title('RMSE: %.4f'% np.sqrt(sum((diff_recover_1-stock0)**2)/ts.size))
    plt.show()