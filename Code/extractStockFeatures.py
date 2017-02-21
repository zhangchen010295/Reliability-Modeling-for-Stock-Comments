# dfsdfsds encoding: utf-8
# 提取股票的特征，包括股票热度、当天或历史的情感趋势、预测准确性

import pandas
import numpy as np
import datetime
from datetime import date
from utility import str2date

def calcOneHot(dicD,k1,k2,lPre=[],bOnehot=True):
    num1=0
    num2=0
    if lPre == []:
        num1 = dicD[k1]
        num2 = dicD[k2]
    else:
        for oP in lPre:
            if oP in dicD:
                num1 += dicD[oP][k1]
                num2 += dicD[oP][k2]
    if bOnehot==False:  #只输出数量
        return num1,num2
    else:
        if num1 == 0 and num2 == 0:
            return 0, 0  # 过去n天都没人发表态度
        elif num1 > num2:
            return 1, 0
        else:
            return 0, 1

def add8Sta2Dics(dicD):
    if 'NegWrong' not in dicD:
        dicD['NegWrong'] = 0
    if 'NegRight' not in dicD:
        dicD['NegRight'] = 0
    if 'PosWrong' not in dicD:
        dicD['PosWrong'] = 0
    if 'PosRight' not in dicD:
        dicD['PosRight'] = 0
    if 'Neg' not in dicD:
        dicD['Neg'] = 0
    if 'Pos' not in dicD:
        dicD['Pos'] = 0
    if 'Wrong' not in dicD:
        dicD['Wrong'] = 0
    if 'Right' not in dicD:
        dicD['Right'] = 0
    if 'pUlist' not in dicD:
        dicD['pUlist'] = set()  # 保存股评员list
    return dicD

def buildStockPopSentiReliFeatures(data):
    data = data.sort_values(['StockID', 'Date'])  # 按照股票ID和日期进行排序

    num_total_comments = len(data.values)
    print num_total_comments, u'条股评'

    # 1. 计算每只股票的历史股评数(可以得到历史上有多少人咨询该股票，历史上咨询该股票的比例)
    stocksHistogram = data['StockID'].value_counts()
    listStockID = list(stocksHistogram._index)
    listStockCount = list(stocksHistogram.get_values())
    numStock = len(listStockID)
    print numStock, u'支股票'
    dicStockCount = dict()  # 保存所有股票在历史上的股评数目
    for i in range(numStock):
        dicStockCount[listStockID[i]] = listStockCount[i]

    # 2. 计算每一天的所有股评数
    datesCommentsHistogram = data['Date'].value_counts()
    listDatesCommentsID = list(datesCommentsHistogram._index)
    listDatesCommentsCount = list(datesCommentsHistogram.get_values())
    numDatesComments = len(listDatesCommentsID)
    print numDatesComments, u'天'
    dicDatesCommentsCount = dict()  # 保存每一天的所有股评数目
    for i in range(numDatesComments):
        dicDatesCommentsCount[str2date(listDatesCommentsID[i])] = listDatesCommentsCount[i]

    # 3. 计算每只股票在每一天的股评数
    dicStockPerDayCount = dict()  # 保存所有股票在每一天的股评数目
    '''
    对每一只股票而言，元素按照日期进行排序(因为dict无法排序)
    用于求取当日的前一周的股评情况
    '''
    dicStockPerDayCountSorted = dict()
    for stock in listStockID:
        dicStockPerDayCount[stock] = dict()  # 保存当前股票stock在每一天的股评数目
        dataS = data[data['StockID'] == stock]
        dateHistogram = dataS['Date'].value_counts()
        listDate = list(dateHistogram._index)
        listDateCount = list(dateHistogram.get_values())
        numDate = len(listDate)
        for i in range(numDate):
            # d=pandas.to_datetime(listDate[i])
            d = str2date(listDate[i])

            dicStockPerDayCount[stock][d] = listDateCount[i]
        dicStockPerDayCountSorted[stock] = sorted(dicStockPerDayCount[stock].iteritems(), key=lambda d: d[0])

    # 4. 计算每只股票在每一天的情感趋势one-hot以及靠谱程度
    sentiFrame = pandas.crosstab(index=[data['StockID'], data['Date'], data['PersonURL']],
                                 columns=[data['Label'], data['Reliability']], margins=False)
    # sentiFrame.to_csv('baimei.csv')


    dicSentiment = dict()  # 按股票，日期，人作为key，value为四个值，跌不靠谱，跌靠谱，涨不靠谱，涨靠谱
    for i in sentiFrame.iterrows():
        sID = i[0][0]  # stock ID
        d = str2date(i[0][1])  # date
        pU = i[0][2]  # user URL
        if sID not in dicSentiment:
            dicSentiment[sID] = dict()
        if d not in dicSentiment[sID]:
            dicSentiment[sID][d] = dict()
            dicSentiment[sID][d] = add8Sta2Dics(dicSentiment[sID][d])

        # 注意，如果dicSentiment已经包含这一个人了，那么跳过去或者叠加，因为默认同一个人在同一天针对同一只股票只能有一个态度
        if pU not in dicSentiment[sID][d]['pUlist']:
            dicSentiment[sID][d]['pUlist'].add(pU)
            sData = i[1]  # values包含4个元素，分别是看跌不靠谱，看跌靠谱，看涨不靠谱，看涨靠谱
            # print pU,sID,d
            # print sData.values
            # quit()
            nW = sData.values[0]
            nR = sData.values[1]
            pW = sData.values[2]
            pR = sData.values[3]
            dicSentiment[sID][d]['NegWrong'] += nW
            dicSentiment[sID][d]['NegRight'] += nR
            dicSentiment[sID][d]['PosWrong'] += pW
            dicSentiment[sID][d]['PosRight'] += pR
            dicSentiment[sID][d]['Neg'] += nW + nR
            dicSentiment[sID][d]['Pos'] += pW + pR
            dicSentiment[sID][d]['Wrong'] += nW + pW
            dicSentiment[sID][d]['Right'] += nR + pR

    '''
    开始提取特征
    '''
    # A. 提取股票的历史流行度
    columnHisStockPopCount = []  # 历史上有多少人咨询该股票
    columnHisStockPopRatio = []  # 历史上咨询该股票的比例
    sAry = data['StockID'].values
    for s in sAry:
        n = dicStockCount[s]
        columnHisStockPopCount.append(n)
        columnHisStockPopRatio.append(float(n) / num_total_comments)

    # B. 提取股票的近期流行度（例如最近一周有多少人咨询该股票）
    dAry = data['Date'].values
    column7DaysStockPopCount = []  # 最近7天有多少人咨询该股票
    column7DaysStockPopRatio = []  # 最近7天咨询该股票的比例
    columnCurrentDayStockPopCount = []  # 当天有多少人咨询该股票
    columnCurrentDayStockPopRatio = []  # 当天咨询该股票的比例
    assert (len(sAry) == len(dAry))
    for i in range(len(sAry)):
        stockID = sAry[i]  # 股票ID
        d = str2date(dAry[i])  # 当前日期

        numStockCurrentDay = dicStockPerDayCount[stockID][d]  # 当天咨询该股票的人数
        numCurrentDay = dicDatesCommentsCount[d]  # 当天咨询所有股票的人数
        assert (numCurrentDay > 0)
        columnCurrentDayStockPopCount.append(numStockCurrentDay)
        columnCurrentDayStockPopRatio.append(float(numStockCurrentDay) / numCurrentDay)

        numStock7Days = numStockCurrentDay  # 最近7天把当天也考虑进去了
        num7Days = numCurrentDay
        # 前6个自然日的日期
        dPre6 = [d + datetime.timedelta(days=w) for w in range(-6, 0)]
        for dP in dPre6:
            if dP in dicStockPerDayCount[stockID]:
                numStock7Days = numStock7Days + dicStockPerDayCount[stockID][dP]

            if dP in dicDatesCommentsCount:
                num7Days = num7Days + dicDatesCommentsCount[dP]
        assert (num7Days > 0)
        column7DaysStockPopCount.append(numStock7Days)
        column7DaysStockPopRatio.append(float(numStock7Days) / num7Days)

    # C. 提取当天对该股票看涨、看跌的趋势 one-hot（01代表看涨的多，10代表看跌的多）
    # D. 最近7天（不包括当天）对该股票看涨、看跌、靠谱、不靠谱的趋势
    clmCurDayNeg = []  # 当天看跌的占优
    clmCurDayPos = []  # 当天看涨的占优
    clmCurDayNeo = []  # 将来考虑中性，即涨跌各半，目前只考虑两种情况
    clm7DaysNeg = []  # 最近7天看跌的占优
    clm7DaysPos = []  # 最近7天看涨的占优
    clm7DaysWrong = []  # 最近7天预测错误的占优
    clm7DaysRight = []  # 最近7天预测正确的占优
    clm7DaysNegWrn = []  # 最近7天的状态分布（4个状态，用num数目代表）
    clm7DaysNegRht = []
    clm7DaysPosWrn = []
    clm7DaysPosRht = []

    clm7DaysWrongNum = []  # 最近7天预测错误的人数
    clm7DaysRightNum = []  # 最近7天预测正确的人数
    clm7DaysRightRatio = []  # 预测正确率（实验证明该特征没什么作用，和上面两个重复了）
    clmCurDayNegNum = []  # 当天看跌的人数
    clmCurDayPosNum = []  # 当天看涨的人数

    clm7DaysNegNum = []  # 最近7天看跌的人数
    clm7DaysPosNum = []  # 最近7天看涨的人数

    for i in range(len(sAry)):
        stockID = sAry[i]  # 股票ID
        d = str2date(dAry[i])  # 当前日期
        hCurNeg, hCurPos = calcOneHot(dicSentiment[stockID][d], 'Neg', 'Pos')
        clmCurDayNeg.append(hCurNeg)
        clmCurDayPos.append(hCurPos)
        numCurNeg, numCurPos = calcOneHot(dicSentiment[stockID][d], 'Neg', 'Pos', bOnehot=False)
        clmCurDayNegNum.append(numCurNeg)
        clmCurDayPosNum.append(numCurPos)

        # 前7个自然日的日期，最近7天没有考虑当天，因为当天的靠谱程度是未知量
        dPre7 = [d + datetime.timedelta(days=w) for w in range(-7, 0)]
        n7DaysNegRht, n7DaysNegWrn, n7DaysPosRht, n7DaysPosWrn = 0, 0, 0, 0
        for dP in dPre7:
            if dP in dicSentiment[stockID]:
                n7DaysNegRht += dicSentiment[stockID][dP]['NegRight']
                n7DaysNegWrn += dicSentiment[stockID][dP]['NegWrong']
                n7DaysPosRht += dicSentiment[stockID][dP]['PosRight']
                n7DaysPosWrn += dicSentiment[stockID][dP]['PosWrong']
        n7DaysSum = n7DaysNegRht + n7DaysNegWrn + n7DaysPosRht + n7DaysPosWrn
        if n7DaysSum == 0:
            clm7DaysNegWrn.append(0)  # 过去7天没有人评价该股票
            clm7DaysNegRht.append(0)
            clm7DaysPosWrn.append(0)
            clm7DaysPosRht.append(0)
        else:
            # clm7DaysNegWrn.append(float(n7DaysNegWrn)/n7DaysSum)
            # clm7DaysNegRht.append(float(n7DaysNegRht)/n7DaysSum)
            # clm7DaysPosWrn.append(float(n7DaysPosWrn)/n7DaysSum)
            # clm7DaysPosRht.append(float(n7DaysPosRht)/n7DaysSum)
            clm7DaysNegWrn.append(n7DaysNegWrn)
            clm7DaysNegRht.append(n7DaysNegRht)
            clm7DaysPosWrn.append(n7DaysPosWrn)
            clm7DaysPosRht.append(n7DaysPosRht)

        h7DaysNeg, h7DaysPos = calcOneHot(dicSentiment[stockID], 'Neg', 'Pos', dPre7)
        clm7DaysNeg.append(h7DaysNeg)
        clm7DaysPos.append(h7DaysPos)
        num7DaysNeg, num7DaysPos = calcOneHot(dicSentiment[stockID], 'Neg', 'Pos', dPre7, bOnehot=False)
        clm7DaysNegNum.append(num7DaysNeg)
        clm7DaysPosNum.append(num7DaysPos)

        h7DaysRight, h7DaysWrong = calcOneHot(dicSentiment[stockID], 'Right', 'Wrong', dPre7)
        clm7DaysRight.append(h7DaysRight)
        clm7DaysWrong.append(h7DaysWrong)
        num7DaysRight, num7DaysWrong = calcOneHot(dicSentiment[stockID], 'Right', 'Wrong', dPre7, False)
        clm7DaysRightNum.append(num7DaysRight)
        clm7DaysWrongNum.append(num7DaysWrong)
        if num7DaysRight == 0 and num7DaysWrong == 0:
            clm7DaysRightRatio.append(0)
        else:
            clm7DaysRightRatio.append(float(num7DaysRight) / (num7DaysRight + num7DaysWrong))

    data['HisStockPopNum'] = columnHisStockPopCount
    data['HisStockPopRatio'] = columnHisStockPopRatio
    data['7DaysStockPopNum'] = column7DaysStockPopCount
    data['7DaysStockPopRatio'] = column7DaysStockPopRatio
    data['CurDayNeg'] = clmCurDayNeg
    data['CurDayPos'] = clmCurDayPos
    data['7DaysNeg'] = clm7DaysNeg
    data['7DaysPos'] = clm7DaysPos
    data['7DaysRight'] = clm7DaysRight
    data['7DaysWrong'] = clm7DaysWrong
    data['7DaysRightNum'] = clm7DaysRightNum
    data['7DaysWrongNum'] = clm7DaysWrongNum
    data['7DaysRightRatio'] = clm7DaysRightRatio
    data['CurDayNegNum'] = clmCurDayNegNum
    data['CurDayPosNum'] = clmCurDayPosNum
    data['7DaysNegNum'] = clm7DaysNegNum
    data['7DaysPosNum'] = clm7DaysPosNum

    data['7DaysProbNegWrong'] = clm7DaysNegWrn
    data['7DaysProbNegRight'] = clm7DaysNegRht
    data['7DaysProbPosWrong'] = clm7DaysPosWrn
    data['7DaysProbPosRight'] = clm7DaysPosRht

    data = data.drop('PersonURL', axis=1)
    data = data.drop('StockID', axis=1)
    data = data.drop('Date', axis=1)
    data = data.drop('Label', axis=1)
    data = data.drop('Reliability', axis=1)

    return data

