# -*-coding:utf-8-*-
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARMA
import sys
from dateutil.relativedelta import relativedelta
from copy import deepcopy
import matplotlib.pyplot as plt
import datetime
import time
from statsmodels.tsa.stattools import adfuller

def testStationarity(ts):
    dftest = adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput


class arima_model:

    def __init__(self, ts, maxLag=9):
        self.data_ts = ts
        self.resid_ts = None
        self.predict_ts = None
        self.maxLag = maxLag
        self.p = maxLag
        self.q = maxLag
        self.properModel = None
        self.bic = sys.maxint

    # 计算最优ARIMA模型，将相关结果赋给相应属性
    def get_proper_model(self):
        self._proper_model()
        self.predict_ts = deepcopy(self.properModel.predict())
        self.resid_ts = deepcopy(self.properModel.resid)

    # 对于给定范围内的p,q计算拟合得最好的arima模型，这里是对差分好的数据进行拟合，故差分恒为0
    def _proper_model(self):
        for p in np.arange(self.maxLag):
            for q in np.arange(self.maxLag):
                # print p,q,self.bic
                if p==0 and q==0:#zc
                    p=1
                    q=1
                model = ARMA(self.data_ts, order=(p, q))
                try:
                    results_ARMA = model.fit(disp=-1, method='css')
                except:
                    continue
                bic = results_ARMA.bic
                # print 'bic:',bic,'self.bic:',self.bic
                if bic < self.bic:
                    self.p = p
                    self.q = q
                    self.properModel = results_ARMA
                    self.bic = bic
                    self.resid_ts = deepcopy(self.properModel.resid)
                    self.predict_ts = self.properModel.predict()

    # 参数确定模型
    def certain_model(self, p, q):
            model = ARMA(self.data_ts, order=(p, q))
            try:
                self.properModel = model.fit( disp=-1, method='css')
                self.p = p
                self.q = q
                self.bic = self.properModel.bic
                self.predict_ts = self.properModel.predict()
                self.resid_ts = deepcopy(self.properModel.resid)
            except:
                print 'You can not fit the model with this parameter p,q, ' \
                      'please use the get_proper_model method to get the best model'

    # 预测第二日的值
    def forecast_next_day_value(self, type='day'):
        # 我修改了statsmodels包中arima_model的源代码，添加了constant属性，需要先运行forecast方法，为constant赋值
        self.properModel.forecast()
        if self.data_ts.index[-1] != self.resid_ts.index[-1]:
            raise ValueError('''The index is different in data_ts and resid_ts, please add new data to data_ts.
            If you just want to forecast the next day data without add the real next day data to data_ts,
            please run the predict method which arima_model included itself''')
        if not self.properModel:
            raise ValueError('The arima model have not computed, please run the proper_model method before')
        para = self.properModel.params

        # print self.properModel.params
        if self.p == 0:   # It will get all the value series with setting self.data_ts[-self.p:] when p is zero
            ma_value = self.resid_ts[-self.q:]
            values = ma_value.reindex(index=ma_value.index[::-1])
        elif self.q == 0:
            ar_value = self.data_ts[-self.p:]
            values = ar_value.reindex(index=ar_value.index[::-1])
        else:
            ar_value = self.data_ts[-self.p:]
            ar_value = ar_value.reindex(index=ar_value.index[::-1])
            ma_value = self.resid_ts[-self.q:]
            ma_value = ma_value.reindex(index=ma_value.index[::-1])
            values = ar_value.append(ma_value)

        predict_value = np.dot(para[1:], values) + self.properModel.constant[0]
        self._add_new_data(self.predict_ts, predict_value, type)
        return predict_value

    # 动态添加数据函数，针对索引是月份和日分别进行处理
    def _add_new_data(self, ts, dat, type='day'):
        if type == 'day':
            new_index = ts.index[-1] + relativedelta(days=1)
        elif type == 'month':
            new_index = ts.index[-1] + relativedelta(months=1)
        ts[new_index] = dat

    def add_today_data(self, dat, type='day'):
        self._add_new_data(self.data_ts, dat, type)
        if self.data_ts.index[-1] != self.predict_ts.index[-1]:
            raise ValueError('You must use the forecast_next_day_value method forecast the value of today before')
        self._add_new_data(self.resid_ts, self.data_ts[-1] - self.predict_ts[-1], type)

def month_step(curDt,step):
    year=curDt.year
    month=curDt.month
    newYear = year
    newMonth = month + step
    while newMonth > 12:
        newMonth -= 12
        newYear += 1
    while newMonth < 1:
        newMonth += 12
        newYear -= 1
    res=datetime.datetime(newYear,newMonth,1)
    return res


if __name__ == '__main__':
    dfStock = pd.read_csv('stonePreClosingPx_1205.csv', encoding='utf-8', index_col='Date')
    dfStock.index = pd.to_datetime(dfStock.index)
    idxTotal = pd.date_range(start=dfStock.index[0], end=dfStock.index[-1])#直接在整个数据上进行填充不妥，应该用股票各自的时间窗
    dfStock = dfStock.reindex(idxTotal)
    dfStock.fillna(method='bfill',inplace=True)#直接在整个数据上进行填充不妥，应该用股票各自的时间窗


    stock0 = dfStock['600005']
    plt.figure(facecolor='white')

    stock1 = stock0.diff(1)#一阶差分
    stock1.dropna(inplace=True)
    stock1=stock1[stock1.index<=pd.to_datetime('2015-04-27')]
    print testStationarity(stock1)
    stock1.plot(color='blue', label='Predict')
    plt.show()




    df = pd.read_csv('AirPassengers.csv', encoding='utf-8', index_col='date')

    df.index = pd.to_datetime(df.index)
    ts = df['x']



    # 数据预处理
    ts_log = np.log(ts)
    rol_mean = ts_log.rolling(window=12).mean()
    rol_mean.dropna(inplace=True)
    ts_diff_1 = rol_mean.diff(1)
    ts_diff_1.dropna(inplace=True)
    ts_diff_2 = ts_diff_1.diff(1)
    ts_diff_2.dropna(inplace=True)


    # 模型拟合
    model = arima_model(stock1)
    #  这里使用模型参数自动识别
    #model.get_proper_model()
    model.certain_model(1,1)
    print type(model.properModel)
    quit()
    print 'bic:', model.bic, 'p:', model.p, 'q:', model.q
    print model.properModel.forecast(steps=1)[0]
    print model.properModel.forecast(steps=1)[1]
    print model.properModel.forecast(steps=1)[2]
    print model.forecast_next_day_value(type='month')

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
    #quit()

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