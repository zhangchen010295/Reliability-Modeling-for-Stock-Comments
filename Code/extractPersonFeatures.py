# dfsdfsds encoding: utf-8
# 提取股评员的特征，包括历史态度、预测准确率，最近7天态度、预测准确率
#还包括其自身特征，公司，地区，简介，好评率，违规率，...

import pandas
import numpy as np
import datetime
from datetime import date
from utility import str2date
import pickle

def calcNum(dicD, k1,k2,k3,k4,k5,lPre=[]):
    num1 = 0
    num2 = 0
    num3 = 0
    num4 = 0
    num5 = 0
    if lPre == []:
        num1 = dicD[k1]
        num2 = dicD[k2]
        num3 = dicD[k3]
        num4 = dicD[k4]
        num5 = dicD[k5]
    else:
        for oP in lPre:
            if oP in dicD:
                num1 += dicD[oP][k1]
                num2 += dicD[oP][k2]
                num3 += dicD[oP][k3]
                num4 += dicD[oP][k4]
                num5 += dicD[oP][k5]
    return num1, num2, num3, num4, num5

def buildPersonFeatures(data):
    # 股票板块信息
    openStockSector = open('StockSector.pkl', 'rb')
    dictStockSector, setSectorName = pickle.load(openStockSector)
    listSectorName = list(setSectorName)
    print listSectorName
    openStockSector.close()
    # 找到训练集和测试集分界面的时间节点
    dateCalc = pandas.DataFrame()
    dateCalc['Date'] = pandas.to_datetime(data['Date'])
    dateCalc = dateCalc.sort_values('Date')
    numData22 = len(dateCalc.values)
    dtBegin = dateCalc.iloc[0]['Date']
    dtEnd = dateCalc.iloc[numData22 - 1]['Date']
    # dtSplit = pandas.to_datetime('2016-03-23') #大数据
    dtSplit = pandas.to_datetime('2015-11-22')  # 小数据
    dtEnd = str2date(str(dtEnd).split(' ')[0])  # 转乘python的date
    dtSplit = str2date(str(dtSplit).split(' ')[0])  # 转乘python的date
    print dtBegin, dtEnd

    # for data
    data = data.sort_values(['PersonURL', 'StockID', 'Date'])  # 按照股票ID和日期进行排序

    num_total_comments = len(data.values)
    print num_total_comments, u'条股评'

    # 1. 计算每个人的历史股评数及每一天的股评数
    personHistogram = data['PersonURL'].value_counts()
    listPersonID = list(personHistogram._index)
    listPersonCount = list(personHistogram.get_values())
    numPerson = len(listPersonID)
    print numPerson, u'个人'
    del personHistogram

    dicPersonCount = dict()  # 保存所有人在历史上的股评数目
    dicPersonDateCount = dict()  # 所有人在每一天的股评数目
    sentiVoc = ['Neg', 'Pos']  # 0是negative，1是positive
    kaopuVoc = ['Wrn', 'Rht']  # 0是wrong，1是right
    # 1a 统计dicPersonCount
    for i in range(numPerson):
        curPID = listPersonID[i]
        nTotal = listPersonCount[i]
        dicPersonCount[curPID] = dict()
        dicPersonCount[curPID]['Total'] = nTotal
        dicPersonCount[curPID]['Pos'] = 0
        dicPersonCount[curPID]['Neg'] = 0
        dicPersonCount[curPID]['Wrn'] = 0
        dicPersonCount[curPID]['Rht'] = 0
        curPData = data[data['PersonURL'] == curPID]
        personSentiHistogram = curPData['Label'].value_counts()
        listSentiID = list(personSentiHistogram._index)
        listSentiCount = list(personSentiHistogram.get_values())
        numSenti = len(listSentiID)
        for j in range(numSenti):
            s = listSentiID[j]
            ns = listSentiCount[j]
            dicPersonCount[curPID][sentiVoc[s]] = ns
        del personSentiHistogram

        personKaopuHistogram = curPData['Reliability'].value_counts()
        listKaopuID = list(personKaopuHistogram._index)
        listKaopuCount = list(personKaopuHistogram.get_values())
        numKaopu = len(listKaopuID)
        for k in range(numKaopu):
            ka = listKaopuID[k]
            nka = listKaopuCount[k]
            dicPersonCount[curPID][kaopuVoc[ka]] = nka
        del personKaopuHistogram

    # 1b 统计dicPersonDateCount,每个人每一天发布股评的情况
    for i in range(numPerson):
        curPID = listPersonID[i]
        dicPersonDateCount[curPID] = dict()
        curPData = data[data['PersonURL'] == curPID]
        pDateHistogram = curPData['Date'].value_counts()
        listDateID = list(pDateHistogram._index)
        listDateCount = list(pDateHistogram.get_values())
        numDate = len(listDateID)
        for j in range(numDate):
            strdtID = listDateID[j]
            dtID = str2date(strdtID)
            dtNum = listDateCount[j]
            dicPersonDateCount[curPID][dtID] = dict()
            dicPersonDateCount[curPID][dtID]['Total'] = dtNum
            dicPersonDateCount[curPID][dtID]['Pos'] = 0
            dicPersonDateCount[curPID][dtID]['Neg'] = 0
            dicPersonDateCount[curPID][dtID]['Wrn'] = 0
            dicPersonDateCount[curPID][dtID]['Rht'] = 0
            curPDateData = curPData[curPData['Date'] == strdtID]  # 注意，这是一定是str，不能是date

            pDateSentiHistogram = curPDateData['Label'].value_counts()
            listDateSentiID = list(pDateSentiHistogram._index)
            listDateSentiCount = list(pDateSentiHistogram.get_values())
            numDateSenti = len(listDateSentiID)
            for k1 in range(numDateSenti):
                sd = listDateSentiID[k1]
                nsd = listDateSentiCount[k1]
                dicPersonDateCount[curPID][dtID][sentiVoc[sd]] = nsd
            del pDateSentiHistogram

            pDateKaopuHistogram = curPDateData['Reliability'].value_counts()
            listDateKaopuID = list(pDateKaopuHistogram._index)
            listDateKaopuCount = list(pDateKaopuHistogram.get_values())
            numDateKaopu = len(listDateKaopuID)
            for k2 in range(numDateKaopu):
                kao = listDateKaopuID[k2]
                nkao = listDateKaopuCount[k2]
                dicPersonDateCount[curPID][dtID][kaopuVoc[kao]] = nkao
            del pDateKaopuHistogram
        del pDateHistogram
    # npI=0
    # for p in dicPersonDateCount:
    #    if len(dicPersonDateCount[p]) >= 4:
    #        npI+=1
    #        lD=[]
    #        for d in dicPersonDateCount[p]:
    #            lD.append(d)
    #        lD.sort()
    #        print p,len(lD),float((lD[-1]-lD[0]).days)/len(lD)
    #        #print('\n')
    # print npI,u'个人在发布了大于4天的股评'
    # quit()


    # 2. 计算每个人在每一只股票上的历史股评数及每一天在每一只股票上的股评数
    stockHistogram = data['StockID'].value_counts()
    listStockID = list(stockHistogram._index)
    listStockCount = list(stockHistogram.get_values())
    numStock = len(listStockID)
    print numStock, u'支股票'
    del stockHistogram

    dicStockPersonCount = dict()  # 保存针对每一只股票，每个人上在历史上的股评数目
    dicStockPersonDateCount = dict()  # 保存针对每一只股票，每个人在每一天的股评数目
    dicSectorPersonCount = dict()  # 保存针对每一个板块，每个人上在历史上的股评数目
    dicSectorPersonDateCount = dict()  # 保存针对每一个板块，每个人在每一天的股评数目

    for iS in range(numStock):
        curStockID = listStockID[iS]
        sectorID = dictStockSector[curStockID[2:]]  # 板块ID
        # 该data里只包含当前股票的数据，所以统计过程和上面一样
        dataStock = data[data['StockID'] == curStockID]
        dicStockPersonCount[curStockID] = dict()
        dicStockPersonDateCount[curStockID] = dict()
        if sectorID not in dicSectorPersonCount:
            dicSectorPersonCount[sectorID] = dict()
        if sectorID not in dicSectorPersonDateCount:
            dicSectorPersonDateCount[sectorID] = dict()

        stockPersonHistogram = dataStock['PersonURL'].value_counts()
        listStockPersonID = list(stockPersonHistogram._index)
        listStockPersonCount = list(stockPersonHistogram.get_values())
        numStockPerson = len(listStockPersonID)
        print u'股票', curStockID, u'上有', numStockPerson, u'个人'
        del stockPersonHistogram

        # 2a 统计dicStockPersonCount
        for i in range(numStockPerson):
            curPID = listStockPersonID[i]
            nTotal = listStockPersonCount[i]
            dicStockPersonCount[curStockID][curPID] = dict()
            dicStockPersonCount[curStockID][curPID]['Total'] = nTotal
            dicStockPersonCount[curStockID][curPID]['Pos'] = 0
            dicStockPersonCount[curStockID][curPID]['Neg'] = 0
            dicStockPersonCount[curStockID][curPID]['Wrn'] = 0
            dicStockPersonCount[curStockID][curPID]['Rht'] = 0
            if curPID not in dicSectorPersonCount[sectorID]:
                dicSectorPersonCount[sectorID][curPID] = dict()
                dicSectorPersonCount[sectorID][curPID]['Total'] = nTotal
                dicSectorPersonCount[sectorID][curPID]['Pos'] = 0
                dicSectorPersonCount[sectorID][curPID]['Neg'] = 0
                dicSectorPersonCount[sectorID][curPID]['Wrn'] = 0
                dicSectorPersonCount[sectorID][curPID]['Rht'] = 0
            else:
                dicSectorPersonCount[sectorID][curPID]['Total'] += nTotal
            curPData = dataStock[dataStock['PersonURL'] == curPID]

            stockPersonSentiHistogram = curPData['Label'].value_counts()
            listStockSentiID = list(stockPersonSentiHistogram._index)
            listStockSentiCount = list(stockPersonSentiHistogram.get_values())
            numStockSenti = len(listStockSentiID)
            for j in range(numStockSenti):
                s = listStockSentiID[j]
                ns = listStockSentiCount[j]
                dicStockPersonCount[curStockID][curPID][sentiVoc[s]] = ns
                dicSectorPersonCount[sectorID][curPID][sentiVoc[s]] += ns
            del stockPersonSentiHistogram

            stockPersonKaopuHistogram = curPData['Reliability'].value_counts()
            listStockKaopuID = list(stockPersonKaopuHistogram._index)
            listStockKaopuCount = list(stockPersonKaopuHistogram.get_values())
            numStockKaopu = len(listStockKaopuID)
            for k in range(numStockKaopu):
                ka = listStockKaopuID[k]
                nka = listStockKaopuCount[k]
                dicStockPersonCount[curStockID][curPID][kaopuVoc[ka]] = nka
                dicSectorPersonCount[sectorID][curPID][kaopuVoc[ka]] += nka
            del stockPersonKaopuHistogram

        # 2b 统计dicStockPersonDateCount
        for i in range(numStockPerson):
            curPID = listStockPersonID[i]
            dicStockPersonDateCount[curStockID][curPID] = dict()
            if curPID not in dicSectorPersonDateCount[sectorID]:
                dicSectorPersonDateCount[sectorID][curPID] = dict()
            curPData = dataStock[dataStock['PersonURL'] == curPID]
            pStockDateHistogram = curPData['Date'].value_counts()
            listStockDateID = list(pStockDateHistogram._index)
            listStockDateCount = list(pStockDateHistogram.get_values())
            numStockDate = len(listStockDateID)
            for j in range(numStockDate):
                strdtID = listStockDateID[j]
                dtID = str2date(strdtID)
                dtNum = listStockDateCount[j]
                dicStockPersonDateCount[curStockID][curPID][dtID] = dict()
                dicStockPersonDateCount[curStockID][curPID][dtID]['Total'] = dtNum
                dicStockPersonDateCount[curStockID][curPID][dtID]['Pos'] = 0
                dicStockPersonDateCount[curStockID][curPID][dtID]['Neg'] = 0
                dicStockPersonDateCount[curStockID][curPID][dtID]['Wrn'] = 0
                dicStockPersonDateCount[curStockID][curPID][dtID]['Rht'] = 0

                if dtID not in dicSectorPersonDateCount[sectorID][curPID]:
                    dicSectorPersonDateCount[sectorID][curPID][dtID] = dict()
                    dicSectorPersonDateCount[sectorID][curPID][dtID]['Total'] = dtNum
                    dicSectorPersonDateCount[sectorID][curPID][dtID]['Pos'] = 0
                    dicSectorPersonDateCount[sectorID][curPID][dtID]['Neg'] = 0
                    dicSectorPersonDateCount[sectorID][curPID][dtID]['Wrn'] = 0
                    dicSectorPersonDateCount[sectorID][curPID][dtID]['Rht'] = 0
                else:
                    dicSectorPersonDateCount[sectorID][curPID][dtID]['Total'] += dtNum

                curPDateData = curPData[curPData['Date'] == strdtID]  # 注意，这是一定是str，不能是date

                pStockDateSentiHistogram = curPDateData['Label'].value_counts()
                listStockDateSentiID = list(pStockDateSentiHistogram._index)
                listStockDateSentiCount = list(pStockDateSentiHistogram.get_values())
                numStockDateSenti = len(listStockDateSentiID)
                for k1 in range(numStockDateSenti):
                    sd = listStockDateSentiID[k1]
                    nsd = listStockDateSentiCount[k1]
                    dicStockPersonDateCount[curStockID][curPID][dtID][sentiVoc[sd]] = nsd
                    dicSectorPersonDateCount[sectorID][curPID][dtID][sentiVoc[sd]] += nsd
                del pStockDateSentiHistogram

                pStockDateKaopuHistogram = curPDateData['Reliability'].value_counts()
                listStockDateKaopuID = list(pStockDateKaopuHistogram._index)
                listStockDateKaopuCount = list(pStockDateKaopuHistogram.get_values())
                numStockDateKaopu = len(listStockDateKaopuID)
                for k2 in range(numStockDateKaopu):
                    kao = listStockDateKaopuID[k2]
                    nkao = listStockDateKaopuCount[k2]
                    dicStockPersonDateCount[curStockID][curPID][dtID][kaopuVoc[kao]] = nkao
                    dicSectorPersonDateCount[sectorID][curPID][dtID][kaopuVoc[kao]] += nkao
                del pStockDateKaopuHistogram
            del pStockDateHistogram

    '''
    开始提取特征
    '''
    # A. 提取所有人的历史股评情况
    columnHisCommentCount = []  # 历史上发布多少股评
    columnHisCommentCountPos = []  # 历史上发布多少Positive股评
    columnHisCommentCountNeg = []  # 历史上发布多少Negative股评
    columnHisCommentCountWrn = []  # 历史上发布多少正确股评
    columnHisCommentCountRht = []  # 历史上发布多少错误股评
    pAry = data['PersonURL'].values
    sAry = data['StockID'].values
    dateAry = data['Date'].values
    assert (len(dateAry) == num_total_comments)
    assert (len(sAry) == num_total_comments)
    assert (len(pAry) == num_total_comments)
    for i in range(num_total_comments):
        pn = pAry[i]
        # sn = sAry[i]
        dt = str2date(dateAry[i])
        # 历史上的数据要减去当天的
        nTotal = dicPersonCount[pn]['Total'] - dicPersonDateCount[pn][dt]['Total']
        nPos = dicPersonCount[pn]['Pos'] - dicPersonDateCount[pn][dt]['Pos']
        nNeg = dicPersonCount[pn]['Neg'] - dicPersonDateCount[pn][dt]['Neg']
        nWrn = dicPersonCount[pn]['Wrn'] - dicPersonDateCount[pn][dt]['Wrn']
        nRht = dicPersonCount[pn]['Rht'] - dicPersonDateCount[pn][dt]['Rht']
        # 2016-12-12还得去掉当天之后的测试集日期的
        dtFuture = dtSplit
        if dtFuture <= dt:
            dtFuture = dt + datetime.timedelta(days=1)
        while dtFuture <= dtEnd:
            if dtFuture in dicPersonDateCount[pn]:
                nTotal -= dicPersonDateCount[pn][dtFuture]['Total']
                nPos -= dicPersonDateCount[pn][dtFuture]['Pos']
                nNeg -= dicPersonDateCount[pn][dtFuture]['Neg']
                nWrn -= dicPersonDateCount[pn][dtFuture]['Wrn']
                nRht -= dicPersonDateCount[pn][dtFuture]['Rht']
            dtFuture = dtFuture + datetime.timedelta(days=1)

        columnHisCommentCount.append(nTotal)
        columnHisCommentCountPos.append(nPos)
        columnHisCommentCountNeg.append(nNeg)
        columnHisCommentCountWrn.append(nWrn)
        columnHisCommentCountRht.append(nRht)

    # B. 提取所有人在每只股票上及每个板块上的历史股评情况
    columnCurStockHisCommentCount = []  # 历史上发布多少股评
    columnCurStockHisCommentCountPos = []  # 历史上发布多少Positive股评
    columnCurStockHisCommentCountNeg = []  # 历史上发布多少Negative股评
    columnCurStockHisCommentCountWrn = []  # 历史上发布多少正确股评
    columnCurStockHisCommentCountRht = []  # 历史上发布多少错误股评

    columnSectorHisCommentCount = []  # 历史上发布多少股评
    columnSectorHisCommentCountPos = []  # 历史上发布多少Positive股评
    columnSectorHisCommentCountNeg = []  # 历史上发布多少Negative股评
    columnSectorHisCommentCountWrn = []  # 历史上发布多少正确股评
    columnSectorHisCommentCountRht = []  # 历史上发布多少错误股评
    columnSectorHisAmountRatio = []  #历史上在该板块发布的股评占所有股评的比例
    columnSectorHisCorrectRatio = [] #历史上在该板块发布的股评正确性
    columnSectorHisWeight = []  # 历史上在该板块发布的权重，为上面两个ratio的乘积


    for i in range(num_total_comments):
        sn = sAry[i]
        sectorID = dictStockSector[sn[2:]]  # 板块ID
        pn = pAry[i]
        dt = str2date(dateAry[i])
        # 历史上的数据要减去当天的
        nTotalStock = dicStockPersonCount[sn][pn]['Total'] - dicStockPersonDateCount[sn][pn][dt]['Total']
        nPosStock = dicStockPersonCount[sn][pn]['Pos'] - dicStockPersonDateCount[sn][pn][dt]['Pos']
        nNegStock = dicStockPersonCount[sn][pn]['Neg'] - dicStockPersonDateCount[sn][pn][dt]['Neg']
        nWrnStock = dicStockPersonCount[sn][pn]['Wrn'] - dicStockPersonDateCount[sn][pn][dt]['Wrn']
        nRhtStock = dicStockPersonCount[sn][pn]['Rht'] - dicStockPersonDateCount[sn][pn][dt]['Rht']

        nTotalSector = dicSectorPersonCount[sectorID][pn]['Total'] - dicSectorPersonDateCount[sectorID][pn][dt]['Total']
        nPosSector = dicSectorPersonCount[sectorID][pn]['Pos'] - dicSectorPersonDateCount[sectorID][pn][dt]['Pos']
        nNegSector = dicSectorPersonCount[sectorID][pn]['Neg'] - dicSectorPersonDateCount[sectorID][pn][dt]['Neg']
        nWrnSector = dicSectorPersonCount[sectorID][pn]['Wrn'] - dicSectorPersonDateCount[sectorID][pn][dt]['Wrn']
        nRhtSector = dicSectorPersonCount[sectorID][pn]['Rht'] - dicSectorPersonDateCount[sectorID][pn][dt]['Rht']

        # 2016-12-12还得去掉当天之后的测试集日期的
        dtFuture = dtSplit
        if dtFuture <= dt:
            dtFuture = dt + datetime.timedelta(days=1)
        while dtFuture <= dtEnd:
            if dtFuture in dicStockPersonDateCount[sn][pn]:
                nTotalStock -= dicStockPersonDateCount[sn][pn][dtFuture]['Total']
                nPosStock -= dicStockPersonDateCount[sn][pn][dtFuture]['Pos']
                nNegStock -= dicStockPersonDateCount[sn][pn][dtFuture]['Neg']
                nWrnStock -= dicStockPersonDateCount[sn][pn][dtFuture]['Wrn']
                nRhtStock -= dicStockPersonDateCount[sn][pn][dtFuture]['Rht']

            if dtFuture in dicSectorPersonDateCount[sectorID][pn]:
                nTotalSector -= dicSectorPersonDateCount[sectorID][pn][dtFuture]['Total']
                nPosSector -= dicSectorPersonDateCount[sectorID][pn][dtFuture]['Pos']
                nNegSector -= dicSectorPersonDateCount[sectorID][pn][dtFuture]['Neg']
                nWrnSector -= dicSectorPersonDateCount[sectorID][pn][dtFuture]['Wrn']
                nRhtSector -= dicSectorPersonDateCount[sectorID][pn][dtFuture]['Rht']
            dtFuture = dtFuture + datetime.timedelta(days=1)

        columnCurStockHisCommentCount.append(nTotalStock)
        columnCurStockHisCommentCountPos.append(nPosStock)
        columnCurStockHisCommentCountNeg.append(nNegStock)
        columnCurStockHisCommentCountWrn.append(nWrnStock)
        columnCurStockHisCommentCountRht.append(nRhtStock)

        columnSectorHisCommentCount.append(nTotalSector)
        columnSectorHisCommentCountPos.append(nPosSector)
        columnSectorHisCommentCountNeg.append(nNegSector)
        columnSectorHisCommentCountWrn.append(nWrnSector)
        columnSectorHisCommentCountRht.append(nRhtSector)
        if columnHisCommentCount[i] > 0:
            amountRatio = float(nTotalSector)/columnHisCommentCount[i]
        else:
            amountRatio = 0.0
        if nRhtSector+nWrnSector > 0:
            correctRatio = float(nRhtSector)/(nRhtSector+nWrnSector)
        else:
            correctRatio = 0.0
        columnSectorHisAmountRatio.append(amountRatio)
        columnSectorHisCorrectRatio.append(correctRatio)
        columnSectorHisWeight.append(amountRatio*correctRatio)

    # C. 提取所有人的7天股评情况
    column7DaysCommentCount = []  # 最近7天发布多少股评
    column7DaysCommentCountPos = []  # 最近7天发布多少Positive股评
    column7DaysCommentCountNeg = []  # 最近7天发布多少Negative股评
    column7DaysCommentCountWrn = []  # 最近7天发布多少正确股评
    column7DaysCommentCountRht = []  # 最近7天发布多少错误股评

    column30DaysCommentCount = []  # 最近30天发布多少股评
    column30DaysCommentCountPos = []  # 最近30天发布多少Positive股评
    column30DaysCommentCountNeg = []  # 最近30天发布多少Negative股评
    column30DaysCommentCountWrn = []  # 最近30天发布多少正确股评
    column30DaysCommentCountRht = []  # 最近30天发布多少错误股评

    column90DaysCommentCount = []  # 最近90天发布多少股评
    column90DaysCommentCountPos = []  # 最近90天发布多少Positive股评
    column90DaysCommentCountNeg = []  # 最近90天发布多少Negative股评
    column90DaysCommentCountWrn = []  # 最近90天发布多少正确股评
    column90DaysCommentCountRht = []  # 最近90天发布多少错误股评

    for i in range(num_total_comments):
        pn = pAry[i]
        dt = str2date(dateAry[i])

        # 前7个自然日的日期，最近7天不考虑当天
        dPre7 = [dt + datetime.timedelta(days=w) for w in range(-7, 0)]
        num7DaysTotal, num7DaysPos, num7DaysNeg, num7DaysWrn, num7DaysRht = \
            calcNum(dicPersonDateCount[pn], 'Total', 'Pos', 'Neg', 'Wrn', 'Rht', dPre7)

        column7DaysCommentCount.append(num7DaysTotal)
        column7DaysCommentCountPos.append(num7DaysPos)
        column7DaysCommentCountNeg.append(num7DaysNeg)
        column7DaysCommentCountWrn.append(num7DaysWrn)
        column7DaysCommentCountRht.append(num7DaysRht)

        # 前30个自然日的日期，最近30天不考虑当天
        dPre30 = [dt + datetime.timedelta(days=w) for w in range(-30, 0)]
        num30DaysTotal, num30DaysPos, num30DaysNeg, num30DaysWrn, num30DaysRht = \
            calcNum(dicPersonDateCount[pn], 'Total', 'Pos', 'Neg', 'Wrn', 'Rht', dPre30)
        #
        column30DaysCommentCount.append(num30DaysTotal)
        column30DaysCommentCountPos.append(num30DaysPos)
        column30DaysCommentCountNeg.append(num30DaysNeg)
        column30DaysCommentCountWrn.append(num30DaysWrn)
        column30DaysCommentCountRht.append(num30DaysRht)

        # 前90个自然日的日期，最近90天不考虑当天
        dPre90 = [dt + datetime.timedelta(days=w) for w in range(-90, 0)]
        num90DaysTotal, num90DaysPos, num90DaysNeg, num90DaysWrn, num90DaysRht = \
            calcNum(dicPersonDateCount[pn], 'Total', 'Pos', 'Neg', 'Wrn', 'Rht', dPre90)
        #
        column90DaysCommentCount.append(num90DaysTotal)
        column90DaysCommentCountPos.append(num90DaysPos)
        column90DaysCommentCountNeg.append(num90DaysNeg)
        column90DaysCommentCountWrn.append(num90DaysWrn)
        column90DaysCommentCountRht.append(num90DaysRht)

    # D. 提取所有人在每只股票上及每个板块上的7天股评情况
    columnCurStock7DaysCommentCount = []  # 最近7天在每只股票上发布多少股评
    columnCurStock7DaysCommentCountPos = []  # 最近7天在每只股票上发布多少Positive股评
    columnCurStock7DaysCommentCountNeg = []  # 最近7天在每只股票上发布多少Negative股评
    columnCurStock7DaysCommentCountWrn = []  # 最近7天在每只股票上发布多少正确股评
    columnCurStock7DaysCommentCountRht = []  # 最近7天在每只股票上发布多少错误股评

    columnCurStock30DaysCommentCount = []  # 最近30天在每只股票上发布多少股评
    columnCurStock30DaysCommentCountPos = []  # 最近30天在每只股票上发布多少Positive股评
    columnCurStock30DaysCommentCountNeg = []  # 最近30天在每只股票上发布多少Negative股评
    columnCurStock30DaysCommentCountWrn = []  # 最近30天在每只股票上发布多少正确股评
    columnCurStock30DaysCommentCountRht = []  # 最近30天在每只股票上发布多少错误股评

    columnCurStock90DaysCommentCount = []  # 最近90天在每只股票上发布多少股评
    columnCurStock90DaysCommentCountPos = []  # 最近90天在每只股票上发布多少Positive股评
    columnCurStock90DaysCommentCountNeg = []  # 最近90天在每只股票上发布多少Negative股评
    columnCurStock90DaysCommentCountWrn = []  # 最近90天在每只股票上发布多少正确股评
    columnCurStock90DaysCommentCountRht = []  # 最近90天在每只股票上发布多少错误股评

    columnSector7DaysCommentCount = []  # 最近7天在每个板块上发布多少股评
    columnSector7DaysCommentCountPos = []  # 最近7天在每个板块上发布多少Positive股评
    columnSector7DaysCommentCountNeg = []  # 最近7天在每个板块上发布多少Negative股评
    columnSector7DaysCommentCountWrn = []  # 最近7天在每个板块上发布多少正确股评
    columnSector7DaysCommentCountRht = []  # 最近7天在每个板块上发布多少错误股评

    columnSector30DaysCommentCount = []  # 最近30天在每个板块上发布多少股评
    columnSector30DaysCommentCountPos = []  # 最近30天在每个板块上发布多少Positive股评
    columnSector30DaysCommentCountNeg = []  # 最近30天在每个板块上发布多少Negative股评
    columnSector30DaysCommentCountWrn = []  # 最近30天在每个板块上发布多少正确股评
    columnSector30DaysCommentCountRht = []  # 最近30天在每个板块上发布多少错误股评

    columnSector90DaysCommentCount = []  # 最近90天在每个板块上发布多少股评
    columnSector90DaysCommentCountPos = []  # 最近90天在每个板块上发布多少Positive股评
    columnSector90DaysCommentCountNeg = []  # 最近90天在每个板块上发布多少Negative股评
    columnSector90DaysCommentCountWrn = []  # 最近90天在每个板块上发布多少正确股评
    columnSector90DaysCommentCountRht = []  # 最近90天在每个板块上发布多少错误股评

    columnSector7DaysAmountRatio = []  # 最近7天在该板块发布的股评占所有股评的比例
    columnSector7DaysCorrectRatio = []  # 最近7天在该板块发布的股评正确性
    columnSector7DaysWeight = []  # 最近7天在该板块发布的权重，为上面两个ratio的乘积

    columnSector30DaysAmountRatio = []  # 最近30天在该板块发布的股评占所有股评的比例
    columnSector30DaysCorrectRatio = []  # 最近30天在该板块发布的股评正确性
    columnSector30DaysWeight = []  # 最近30天在该板块发布的权重，为上面两个ratio的乘积

    columnSector90DaysAmountRatio = []  # 最近90天在该板块发布的股评占所有股评的比例
    columnSector90DaysCorrectRatio = []  # 最近90天在该板块发布的股评正确性
    columnSector90DaysWeight = []  # 最近90天在该板块发布的权重，为上面两个ratio的乘积

    for i in range(num_total_comments):
        sn = sAry[i]
        sectorID = dictStockSector[sn[2:]]  # 板块ID
        pn = pAry[i]
        dt = str2date(dateAry[i])

        # 前7个自然日的日期，最近7天不考虑当天
        dPre7 = [dt + datetime.timedelta(days=w) for w in range(-7, 0)]
        num7DaysTotal, num7DaysPos, num7DaysNeg, num7DaysWrn, num7DaysRht = \
            calcNum(dicStockPersonDateCount[sn][pn], 'Total', 'Pos', 'Neg', 'Wrn', 'Rht', dPre7)

        columnCurStock7DaysCommentCount.append(num7DaysTotal)
        columnCurStock7DaysCommentCountPos.append(num7DaysPos)
        columnCurStock7DaysCommentCountNeg.append(num7DaysNeg)
        columnCurStock7DaysCommentCountWrn.append(num7DaysWrn)
        columnCurStock7DaysCommentCountRht.append(num7DaysRht)

        num7DaysTotalSector, num7DaysPosSector, num7DaysNegSector, num7DaysWrnSector, num7DaysRhtSector = \
            calcNum(dicSectorPersonDateCount[sectorID][pn], 'Total', 'Pos', 'Neg', 'Wrn', 'Rht', dPre7)

        columnSector7DaysCommentCount.append(num7DaysTotalSector)
        columnSector7DaysCommentCountPos.append(num7DaysPosSector)
        columnSector7DaysCommentCountNeg.append(num7DaysNegSector)
        columnSector7DaysCommentCountWrn.append(num7DaysWrnSector)
        columnSector7DaysCommentCountRht.append(num7DaysRhtSector)

        if column7DaysCommentCount[i] > 0:
            amountRatio = float(num7DaysTotalSector) / column7DaysCommentCount[i]
        else:
            amountRatio = 0.0
        if num7DaysRhtSector + num7DaysWrnSector > 0:
            correctRatio = float(num7DaysRhtSector) / (num7DaysRhtSector + num7DaysWrnSector)
        else:
            correctRatio = 0.0
        columnSector7DaysAmountRatio.append(amountRatio)
        columnSector7DaysCorrectRatio.append(correctRatio)
        columnSector7DaysWeight.append(amountRatio * correctRatio)

        # 前30个自然日的日期，最近30天不考虑当天
        dPre30 = [dt + datetime.timedelta(days=w) for w in range(-30, 0)]
        num30DaysTotal, num30DaysPos, num30DaysNeg, num30DaysWrn, num30DaysRht = \
            calcNum(dicStockPersonDateCount[sn][pn], 'Total', 'Pos', 'Neg', 'Wrn', 'Rht', dPre30)
        #
        columnCurStock30DaysCommentCount.append(num30DaysTotal)
        columnCurStock30DaysCommentCountPos.append(num30DaysPos)
        columnCurStock30DaysCommentCountNeg.append(num30DaysNeg)
        columnCurStock30DaysCommentCountWrn.append(num30DaysWrn)
        columnCurStock30DaysCommentCountRht.append(num30DaysRht)

        num30DaysTotalSector, num30DaysPosSector, num30DaysNegSector, num30DaysWrnSector, num30DaysRhtSector = \
            calcNum(dicSectorPersonDateCount[sectorID][pn], 'Total', 'Pos', 'Neg', 'Wrn', 'Rht', dPre7)

        columnSector30DaysCommentCount.append(num30DaysTotalSector)
        columnSector30DaysCommentCountPos.append(num30DaysPosSector)
        columnSector30DaysCommentCountNeg.append(num30DaysNegSector)
        columnSector30DaysCommentCountWrn.append(num30DaysWrnSector)
        columnSector30DaysCommentCountRht.append(num30DaysRhtSector)

        if column30DaysCommentCount[i] > 0:
            amountRatio = float(num30DaysTotalSector) / column30DaysCommentCount[i]
        else:
            amountRatio = 0.0
        if num30DaysRhtSector + num30DaysWrnSector > 0:
            correctRatio = float(num30DaysRhtSector) / (num30DaysRhtSector + num30DaysWrnSector)
        else:
            correctRatio = 0.0
        columnSector30DaysAmountRatio.append(amountRatio)
        columnSector30DaysCorrectRatio.append(correctRatio)
        columnSector30DaysWeight.append(amountRatio * correctRatio)

        # 前90个自然日的日期，最近90天不考虑当天
        dPre90 = [dt + datetime.timedelta(days=w) for w in range(-90, 0)]
        num90DaysTotal, num90DaysPos, num90DaysNeg, num90DaysWrn, num90DaysRht = \
            calcNum(dicStockPersonDateCount[sn][pn], 'Total', 'Pos', 'Neg', 'Wrn', 'Rht', dPre90)
        #
        columnCurStock90DaysCommentCount.append(num90DaysTotal)
        columnCurStock90DaysCommentCountPos.append(num90DaysPos)
        columnCurStock90DaysCommentCountNeg.append(num90DaysNeg)
        columnCurStock90DaysCommentCountWrn.append(num90DaysWrn)
        columnCurStock90DaysCommentCountRht.append(num90DaysRht)

        num90DaysTotalSector, num90DaysPosSector, num90DaysNegSector, num90DaysWrnSector, num90DaysRhtSector = \
            calcNum(dicSectorPersonDateCount[sectorID][pn], 'Total', 'Pos', 'Neg', 'Wrn', 'Rht', dPre7)

        columnSector90DaysCommentCount.append(num90DaysTotalSector)
        columnSector90DaysCommentCountPos.append(num90DaysPosSector)
        columnSector90DaysCommentCountNeg.append(num90DaysNegSector)
        columnSector90DaysCommentCountWrn.append(num90DaysWrnSector)
        columnSector90DaysCommentCountRht.append(num90DaysRhtSector)

        if column90DaysCommentCount[i] > 0:
            amountRatio = float(num90DaysTotalSector) / column90DaysCommentCount[i]
        else:
            amountRatio = 0.0
        if num90DaysRhtSector + num90DaysWrnSector > 0:
            correctRatio = float(num90DaysRhtSector) / (num90DaysRhtSector + num90DaysWrnSector)
        else:
            correctRatio = 0.0
        columnSector90DaysAmountRatio.append(amountRatio)
        columnSector90DaysCorrectRatio.append(correctRatio)
        columnSector90DaysWeight.append(amountRatio * correctRatio)

    '''
    输出特征
    '''
    newData = pandas.DataFrame()
    newData['FileName'] = data['FileName']
    newData['PersonHisCommentCount'] = columnHisCommentCount
    newData['PersonHisCommentCountPos'] = columnHisCommentCountPos
    newData['PersonHisCommentCountNeg'] = columnHisCommentCountNeg
    newData['PersonHisCommentCountWrn'] = columnHisCommentCountWrn
    newData['PersonHisCommentCountRht'] = columnHisCommentCountRht

    newData['PersonCurStockHisCommentCount'] = columnCurStockHisCommentCount
    newData['PersonCurStockHisCommentCountPos'] = columnCurStockHisCommentCountPos
    newData['PersonCurStockHisCommentCountNeg'] = columnCurStockHisCommentCountNeg
    newData['PersonCurStockHisCommentCountWrn'] = columnCurStockHisCommentCountWrn
    newData['PersonCurStockHisCommentCountRht'] = columnCurStockHisCommentCountRht

    newData['PersonSectorHisCommentCount'] = columnSectorHisCommentCount
    newData['PersonSectorHisCommentCountPos'] = columnSectorHisCommentCountPos
    newData['PersonSectorHisCommentCountNeg'] = columnSectorHisCommentCountNeg
    newData['PersonSectorHisCommentCountWrn'] = columnSectorHisCommentCountWrn
    newData['PersonSectorHisCommentCountRht'] = columnSectorHisCommentCountRht
    newData['PersonSectorHisAmountRatio'] = columnSectorHisAmountRatio
    newData['PersonSectorHisCorrectRatio'] = columnSectorHisCorrectRatio
    newData['PersonSectorHisWeight'] = columnSectorHisWeight

    newData['Person7DaysCommentCount'] = column7DaysCommentCount
    newData['Person7DaysCommentCountPos'] = column7DaysCommentCountPos
    newData['Person7DaysCommentCountNeg'] = column7DaysCommentCountNeg
    newData['Person7DaysCommentCountWrn'] = column7DaysCommentCountWrn
    newData['Person7DaysCommentCountRht'] = column7DaysCommentCountRht

    newData['PersonCurStock7DaysCommentCount'] = columnCurStock7DaysCommentCount
    newData['PersonCurStock7DaysCommentCountPos'] = columnCurStock7DaysCommentCountPos
    newData['PersonCurStock7DaysCommentCountNeg'] = columnCurStock7DaysCommentCountNeg
    newData['PersonCurStock7DaysCommentCountWrn'] = columnCurStock7DaysCommentCountWrn
    newData['PersonCurStock7DaysCommentCountRht'] = columnCurStock7DaysCommentCountRht

    newData['Person30DaysCommentCount'] = column30DaysCommentCount
    newData['Person30DaysCommentCountPos'] = column30DaysCommentCountPos
    newData['Person30DaysCommentCountNeg'] = column30DaysCommentCountNeg
    newData['Person30DaysCommentCountWrn'] = column30DaysCommentCountWrn
    newData['Person30DaysCommentCountRht'] = column30DaysCommentCountRht

    newData['PersonCurStock30DaysCommentCount'] = columnCurStock30DaysCommentCount
    newData['PersonCurStock30DaysCommentCountPos'] = columnCurStock30DaysCommentCountPos
    newData['PersonCurStock30DaysCommentCountNeg'] = columnCurStock30DaysCommentCountNeg
    newData['PersonCurStock30DaysCommentCountWrn'] = columnCurStock30DaysCommentCountWrn
    newData['PersonCurStock30DaysCommentCountRht'] = columnCurStock30DaysCommentCountRht

    newData['Person90DaysCommentCount'] = column90DaysCommentCount
    newData['Person90DaysCommentCountPos'] = column90DaysCommentCountPos
    newData['Person90DaysCommentCountNeg'] = column90DaysCommentCountNeg
    newData['Person90DaysCommentCountWrn'] = column90DaysCommentCountWrn
    newData['Person90DaysCommentCountRht'] = column90DaysCommentCountRht

    newData['PersonCurStock90DaysCommentCount'] = columnCurStock90DaysCommentCount
    newData['PersonCurStock90DaysCommentCountPos'] = columnCurStock90DaysCommentCountPos
    newData['PersonCurStock90DaysCommentCountNeg'] = columnCurStock90DaysCommentCountNeg
    newData['PersonCurStock90DaysCommentCountWrn'] = columnCurStock90DaysCommentCountWrn
    newData['PersonCurStock90DaysCommentCountRht'] = columnCurStock90DaysCommentCountRht

    newData['PersonSector7DaysCommentCount'] = columnSector7DaysCommentCount
    newData['PersonSector7DaysCommentCountPos'] = columnSector7DaysCommentCountPos
    newData['PersonSector7DaysCommentCountNeg'] = columnSector7DaysCommentCountNeg
    newData['PersonSector7DaysCommentCountWrn'] = columnSector7DaysCommentCountWrn
    newData['PersonSector7DaysCommentCountRht'] = columnSector7DaysCommentCountRht
    newData['PersonSector7DaysAmountRatio'] = columnSector7DaysAmountRatio
    newData['PersonSector7DaysCorrectRatio'] = columnSector7DaysCorrectRatio
    newData['PersonSector7DaysWeight'] = columnSector7DaysWeight

    newData['PersonSector30DaysCommentCount'] = columnSector30DaysCommentCount
    newData['PersonSector30DaysCommentCountPos'] = columnSector30DaysCommentCountPos
    newData['PersonSector30DaysCommentCountNeg'] = columnSector30DaysCommentCountNeg
    newData['PersonSector30DaysCommentCountWrn'] = columnSector30DaysCommentCountWrn
    newData['PersonSector30DaysCommentCountRht'] = columnSector30DaysCommentCountRht
    newData['PersonSector30DaysAmountRatio'] = columnSector30DaysAmountRatio
    newData['PersonSector30DaysCorrectRatio'] = columnSector30DaysCorrectRatio
    newData['PersonSector30DaysWeight'] = columnSector30DaysWeight

    newData['PersonSector90DaysCommentCount'] = columnSector90DaysCommentCount
    newData['PersonSector90DaysCommentCountPos'] = columnSector90DaysCommentCountPos
    newData['PersonSector90DaysCommentCountNeg'] = columnSector90DaysCommentCountNeg
    newData['PersonSector90DaysCommentCountWrn'] = columnSector90DaysCommentCountWrn
    newData['PersonSector90DaysCommentCountRht'] = columnSector90DaysCommentCountRht
    newData['PersonSector90DaysAmountRatio'] = columnSector90DaysAmountRatio
    newData['PersonSector90DaysCorrectRatio'] = columnSector90DaysCorrectRatio
    newData['PersonSector90DaysWeight'] = columnSector90DaysWeight

    return newData

