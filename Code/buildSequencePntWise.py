# dfsdfsds encoding: utf-8
# 提取序列数据
# 2016-12-24 逐点构建序列，每个点都有机会成为序列最后一个点
# 2016-12-25 对所有特征scale一下

import pandas
import numpy
import datetime
from datetime import date
from utility import str2date,date2str
from sklearn import preprocessing
import pickle
from utility import aryFeaturesScaled

SEED = 1048
numpy.random.seed(SEED)

listPersonSectorFeatures = [

         'PersonSectorHisCommentCount'
        , 'PersonSectorHisCommentCountPos'
        , 'PersonSectorHisCommentCountNeg'
        , 'PersonSectorHisCommentCountWrn'
        , 'PersonSectorHisCommentCountRht'
        , 'PersonSectorHisAmountRatio'
        , 'PersonSectorHisCorrectRatio'
        , 'PersonSectorHisWeight'

        , 'PersonSector7DaysCommentCount'
        , 'PersonSector7DaysCommentCountPos'
        , 'PersonSector7DaysCommentCountNeg'
        , 'PersonSector7DaysCommentCountWrn'
        , 'PersonSector7DaysCommentCountRht'
        , 'PersonSector7DaysAmountRatio'
        , 'PersonSector7DaysCorrectRatio'
        , 'PersonSector7DaysWeight'

        , 'PersonSector30DaysCommentCount'
        , 'PersonSector30DaysCommentCountPos'
        , 'PersonSector30DaysCommentCountNeg'
        , 'PersonSector30DaysCommentCountWrn'
        , 'PersonSector30DaysCommentCountRht'
        , 'PersonSector30DaysAmountRatio'
        , 'PersonSector30DaysCorrectRatio'
        , 'PersonSector30DaysWeight'

        , 'PersonSector90DaysCommentCount'
        , 'PersonSector90DaysCommentCountPos'
        , 'PersonSector90DaysCommentCountNeg'
        , 'PersonSector90DaysCommentCountWrn'
        , 'PersonSector90DaysCommentCountRht'
        , 'PersonSector90DaysAmountRatio'
        , 'PersonSector90DaysCorrectRatio'
        , 'PersonSector90DaysWeight'
    ]
listPersonFeatures = [
        'PersonHisCommentCount'
        , 'PersonHisCommentCountPos'
        , 'PersonHisCommentCountNeg'
        , 'PersonHisCommentCountWrn'
        , 'PersonHisCommentCountRht'
        ,'PersonCurStockHisCommentCount'
        , 'PersonCurStockHisCommentCountPos'
        , 'PersonCurStockHisCommentCountNeg'
        , 'PersonCurStockHisCommentCountWrn'
        , 'PersonCurStockHisCommentCountRht'

        , 'Person7DaysCommentCount'
        , 'Person7DaysCommentCountPos'
        , 'Person7DaysCommentCountNeg'
        , 'Person7DaysCommentCountWrn'
        , 'Person7DaysCommentCountRht'
         ,'PersonCurStock7DaysCommentCount'
       , 'PersonCurStock7DaysCommentCountPos'
        , 'PersonCurStock7DaysCommentCountNeg'
        , 'PersonCurStock7DaysCommentCountWrn'
        , 'PersonCurStock7DaysCommentCountRht'
        , 'Person30DaysCommentCount'
        , 'Person30DaysCommentCountPos'
        , 'Person30DaysCommentCountNeg'
        , 'Person30DaysCommentCountWrn'
        , 'Person30DaysCommentCountRht'
        , 'PersonCurStock30DaysCommentCount'
        , 'PersonCurStock30DaysCommentCountPos'
        , 'PersonCurStock30DaysCommentCountNeg'
        , 'PersonCurStock30DaysCommentCountWrn'
        , 'PersonCurStock30DaysCommentCountRht'
        , 'Person90DaysCommentCount'
        , 'Person90DaysCommentCountPos'
        , 'Person90DaysCommentCountNeg'
        , 'Person90DaysCommentCountWrn'
        , 'Person90DaysCommentCountRht'
        , 'PersonCurStock90DaysCommentCount'
        , 'PersonCurStock90DaysCommentCountPos'
        , 'PersonCurStock90DaysCommentCountNeg'
        , 'PersonCurStock90DaysCommentCountWrn'
        , 'PersonCurStock90DaysCommentCountRht'

        ,'SentiConfidence'  #新特征
        ,'SentiInconsistence'
    ]
listStockTrendFeatures = [
        "StockTrendPre1"
        ,"StockTrendPre2"
        ,"StockTrendPre3"
        ,"StockTrendPre4"
        ,"StockTrendPre5"
        ,"StockTrendPre6"
        ,"StockTrendPre7"
        ,"StockTrendPre8"
        , "StockTrendPre9"
        , "StockTrendPre10"
        , "StockTrendPre11"
        , "StockTrendPre12"
        , "StockTrendPre13"
        , "StockTrendPre14"
        , "StockTrendPre15"
        , "StockTrendPre16"
        , "StockTrendPre17"
        , "StockTrendPre18"
        , "StockTrendPre19"
        , "StockTrendPre20"
        , "StockTrendPre21"
        , "StockTrendPre22"
        , "StockTrendPre23"
        , "StockTrendPre24"
        , "StockTrendPre25"
        #,'ReturnPredict'
        #,'ReturnConfidence'
        #,'TotalTurnover'
        #, 'TotalTurnoverPre1'
        #, 'TotalTurnoverPre2'
        #, 'TotalTurnoverPre2'
        #,'TotalVolumeTraded'
        #, 'TotalVolumeTradedPre1'
        #, 'TotalVolumeTradedPre2'
        #, 'TotalVolumeTradedPre3'
    ]
listStockPopSentiReliFeatures = [
        'HisStockPopNum'
        ,'HisStockPopRatio'
        ,'7DaysStockPopNum'
        ,'7DaysStockPopRatio'
        ,'CurDayNegNum'
        ,'CurDayPosNum'
        ,'7DaysNegNum'
        ,'7DaysPosNum'
        ,'7DaysRight'
        ,'7DaysWrong'
        ,'7DaysRightNum'
        ,'7DaysWrongNum'
        ,'7DaysRightRatio'
        ,'7DaysProbNegWrong'
        ,'7DaysProbNegRight'
        ,'7DaysProbPosWrong'
        ,'7DaysProbPosRight'
    ]


def addLabelError(TestData,ratio):
    if ratio == 0.:
        return TestData
    setFileName=set()
    for i in TestData:
        for j in i.index:
            setFileName.add(i.loc[j,'FileName'])
    aryFN = numpy.array(list(setFileName))
    numpy.random.shuffle(aryFN)
    aryFN = list(aryFN)
    numTotal = len(aryFN)
    aryFn=aryFN[:int(ratio*numTotal)]
    print 'error label'
    for i in aryFn:
        print i
    setFileName=set(aryFN)
    for i in TestData:
        for j in i.index:
            if i.loc[j,'FileName'] in setFileName:
                i.loc[j, 'Label'] = 1 - i.loc[j,'Label']
    return TestData

def outputStastics(fnOutput,dicStastics):
    dfProfit = pandas.read_csv('file4Profit.csv',encoding='utf-8')
    dicProft = dict()
    for i in dfProfit.index:
        fn = dfProfit.loc[i,'FileName']
        vP = dfProfit.loc[i,'Profit']
        dicProft[fn] = vP
    strSenti=['Down','Up']
    strReli = ['Wrong', 'Right']
    with open (fnOutput,'w') as f:
        f.write('PersonURL,StockID,Length,SentiRatio,SentiConfRatio,ReliaRatio,InconsisRatio,TimeRange,Sequence,Chinese\n')
        for p in dicStastics.keys():
            for s in dicStastics[p].keys():
                l = dicStastics[p][s]['Length']
                senra = dicStastics[p][s]['SentiRatio']
                senconfra= dicStastics[p][s]['SentiConfidenceRatio']
                reliaratio = dicStastics[p][s]['ReliabilityRatio']
                inconsratio = dicStastics[p][s]['SentiShiftRatio']
                tmrng = dicStastics[p][s]['TimeRange']
                seq = dicStastics[p][s]['Sequence']
                nd=''
                seqData=[]
                chnData=[]
                for i in seq:
                    dt=date2str(i[0])
                    onesen=i[1]
                    onesenconf=i[2]
                    onerht=i[3]
                    onefn=i[4]
                    assert onefn in dicProft
                    oneProfit = dicProft[onefn]
                    nd='{0};{1};{2};{3};{4};{5}'.format(dt,onesen,onesenconf,onerht,onefn,oneProfit)
                    seqData.append(nd)
                    chnData.append(strSenti[int(onesen)]+strReli[int(onerht)])
                strseqData=','.join(seqData)
                strChnData=','.join(chnData)
                f.write('%s,%s,%d,%.3f,%.3f,%.3f,%.3f,%d,%s\n'%\
                    (p,s,l,senra,senconfra,reliaratio,inconsratio,tmrng,strseqData))

def buildSequenceData(data,Length,fileOutputSequences,fileOutputStastics = ''):
    Error_Ratio = 0.
    #seBefore = data[aryFeaturesScaled].values
    #seAfter = preprocessing.scale(seBefore)
    #data[aryFeaturesScaled] = seAfter

    # dfARMATotal = pandas.read_csv('TotalreturnResult_1.csv', encoding='utf-8')
    # dfARMA = pandas.DataFrame()
    # dfARMA['FileName']=dfARMATotal['FileName']
    # colPredictARMA = []
    # colConfidenceARMA = []
    # for i in dfARMATotal.index:
    #    vPre = dfARMATotal.loc[i,'ReturnPredict']
    #    vErr = dfARMATotal.loc[i,'ReturnErr']
    #    if vPre > 0:
    #        colPredictARMA.append(1)
    #    else:
    #        colPredictARMA.append(0)
    #    colConfidenceARMA.append(1.0/vErr)
    # dfARMA['ReturnPredict'] = colPredictARMA
    # dfARMA['ReturnConfidence'] = colConfidenceARMA

    # 将ARMA结果融合进特征
    # data = pandas.merge(data,dfARMA,how = 'left',on='FileName')



    data = data.sort_values(['PersonURL', 'StockID', 'Date'])  # 按照人ID股票ID和日期进行排序
    listDataFrameSequence = []

    personHistogram = data['PersonURL'].value_counts()
    listPersonID = list(personHistogram._index)
    listPersonCount = list(personHistogram.get_values())
    numPerson = len(listPersonID)
    print numPerson, u'个人'
    del personHistogram

    listFileName = []
    listFeatures = []
    dicStastics = dict()
    numSequence = 0
    # for i in range(1):
    for i in range(numPerson):
        print i + 1, u'个人正在被处理...'
        curPID = listPersonID[i]
        nP = listPersonCount[i]
        if nP > Length:  # 当前股评员发布股评数目少于Length的直接跳过
            dataP = data[data['PersonURL'] == curPID]  # 这里保存着当前股评员的所有数据
            psnStockHistogram = dataP['StockID'].value_counts()
            lPerStockID = list(psnStockHistogram._index)
            lPerStockCount = list(psnStockHistogram.get_values())
            numStock = len(lPerStockID)
            for j in range(numStock):
                curStockID = lPerStockID[j]
                nS = lPerStockCount[j]
                if nS > Length:  # 针对当前股票发布股评数目少于Length的直接跳过
                    # 这里保存着当前股评员在当前股票的所有数据
                    dataPS = dataP[dataP['StockID'] == curStockID]
                    dateHistogram = dataPS['Date'].value_counts()
                    lDateID = list(dateHistogram._index)
                    lDateID.sort()  # 按日期进行排序(危险，字符串排序！！！)
                    # print curPID,curStockID
                    # print lDateID
                    # quit()
                    nD = len(lDateID)
                    if nD > Length:  # 针对当前股票发布股评天数少于Length的直接跳过
                        xDF = pandas.DataFrame(columns=data.columns)  # 保存raw series data
                        listConfidence = []  # 记录当天评论的confidence，有可能有反复态度
                        listSentiShift = []  # 记录该人对该股票的态度shift次数
                        numShift = 0  # 记录当前人在当前股票的shift次数
                        lastSenti = -1
                        numPos = 0.0
                        numNeg = 0.0
                        numRight = 0.0
                        numTotalConfidence = 0.0
                        for k in range(nD):
                            curDate = lDateID[k]
                            dtData = dataPS[dataPS['Date'] == curDate]
                            numText = len(dtData.values)
                            curSenti = -1
                            sentiConfidence = 1.
                            if numText == 1:
                                xDF = xDF.append(dtData.iloc[0], ignore_index=True)
                                curSenti = dtData.iloc[0]['Label']
                                numTotalConfidence += 1
                                numRight += dtData.iloc[0]['Reliability']
                            else:
                                # 当天针对该股票发布了超过一个股评，选择一个最靠谱的
                                pos = 0.
                                numL0 = 0
                                numL1 = 0

                                for idx7 in range(numText):
                                    curLabel = dtData.iloc[idx7]['Label']
                                    if curLabel == 0:
                                        numL0 += 1
                                    else:
                                        numL1 += 1
                                    pos += curLabel
                                pos /= numText
                                if pos > 0.5:
                                    pos = 1
                                    sentiConfidence = float(numL1) / numText
                                else:
                                    pos = 0
                                    sentiConfidence = float(numL0) / numText
                                curSenti = pos
                                numTotalConfidence += sentiConfidence

                                for idx7 in range(numText):
                                    if pos == dtData.iloc[idx7]['Label']:
                                        xDF = xDF.append(dtData.iloc[idx7], ignore_index=True)
                                        numRight += dtData.iloc[idx7]['Reliability']
                                        break
                            if curSenti == 0:
                                numNeg += 1
                            else:
                                numPos += 1
                            listConfidence.append(sentiConfidence)
                            if lastSenti == -1:  # 第一次
                                lastSenti = curSenti
                            elif lastSenti != curSenti:
                                lastSenti = curSenti
                                numShift += 1
                            listSentiShift.append(float(numShift) / (k + 1))
                        # 添加两行特征
                        xDF['SentiConfidence'] = numpy.asarray(listConfidence)
                        xDF['SentiInconsistence'] = numpy.asarray(listSentiShift)
                        listFeatures.append(xDF)
                        nTotalSenquence = len(xDF.values)
                        # 将当前序列输出
                        if curPID not in dicStastics:
                            dicStastics[curPID] = dict()
                        if curStockID not in dicStastics:
                            dicStastics[curPID][curStockID] = dict()
                        dicStastics[curPID][curStockID]['Length'] = nTotalSenquence
                        assert numPos + numNeg == dicStastics[curPID][curStockID]['Length']
                        dicStastics[curPID][curStockID]['SentiRatio'] = numPos / (numPos + numNeg)
                        dicStastics[curPID][curStockID]['SentiConfidenceRatio'] = numTotalConfidence / (numPos + numNeg)
                        dicStastics[curPID][curStockID]['ReliabilityRatio'] = numRight / (numPos + numNeg)
                        dicStastics[curPID][curStockID]['SentiShiftRatio'] = listSentiShift[-1]
                        dicStastics[curPID][curStockID]['Sequence'] = []
                        for i1212 in range(nTotalSenquence):
                            dt = str2date(xDF.iloc[i1212]['Date'])
                            senti = xDF.iloc[i1212]['Label']
                            sentiConfidence = xDF.iloc[i1212]['SentiConfidence']
                            rht = xDF.iloc[i1212]['Reliability']
                            fn21 = xDF.iloc[i1212]['FileName']
                            dicStastics[curPID][curStockID]['Sequence'].append \
                                ((dt, senti, sentiConfidence, rht, fn21))
                        dicStastics[curPID][curStockID]['Sequence'].sort()
                        dicStastics[curPID][curStockID]['TimeRange'] = \
                            (dicStastics[curPID][curStockID]['Sequence'][-1][0] \
                             - dicStastics[curPID][curStockID]['Sequence'][0][0]).days

                        # 用来输出有效数据的文件名
                        for i1212 in range(nTotalSenquence):
                            listFileName.append(xDF.iloc[i1212]['FileName'])
                        # 添加特征：上一次的态度和靠谱
                        colLastSenti = [0.5]
                        colLastReli = [0.5]
                        for i24 in range(1, nTotalSenquence):
                            colLastSenti.append(xDF.iloc[i24 - 1]['Label'])
                            colLastReli.append(xDF.iloc[i24 - 1]['Reliability'])
                        xDF['LastSentiment'] = numpy.asarray(colLastSenti)
                        xDF['LastReliability'] = numpy.asarray(colLastReli)
                        # 从现在开始对xDF进行分组，每组4个元素
                        numSequence += 1  # 统计原始序列数目
                        # 如果是pointwise，则从Length-1开始，每次移动一位
                        for idx32 in range(Length - 1, nTotalSenquence):
                            # for idx32 in range (0,nTotalSenquence-Length,Length-1):
                            oneDF = pandas.DataFrame(columns=data.columns)
                            for idxInner in range(idx32 - (Length - 1), idx32 + 1):
                                # for idxInner in range(idx32,idx32+Length):
                                oneDF = oneDF.append(xDF.iloc[idxInner])
                            # 特征中加上一个时间差
                            numLenDF = len(oneDF.values)
                            listDateDelta = []
                            listDateDelta.append(0)
                            for idx1212 in range(1, numLenDF):
                                d0 = str2date(oneDF.iloc[idx1212 - 1]['Date'])
                                d1 = str2date(oneDF.iloc[idx1212]['Date'])
                                delta = d1 - d0
                                listDateDelta.append(delta.days)
                            oneDF['TimeRange'] = numpy.asarray(listDateDelta)
                            listDataFrameSequence.append(oneDF)
    print numSequence, u'个原始序列'
    outputStastics(fileOutputStastics,dicStastics)
    # quit()

    numxDF = len(listFeatures)
    dfExtractedFeature = pandas.DataFrame()
    dfExtractedFeature['FileName'] = listFeatures[0]['FileName']
    dfExtractedFeature['SentiConfidence'] = listFeatures[0]['SentiConfidence']
    dfExtractedFeature['SentiInconsistence'] = listFeatures[0]['SentiInconsistence']
    for i in range(1, numxDF):
        dfTmp = pandas.DataFrame()
        dfTmp['FileName'] = listFeatures[i]['FileName']
        dfTmp['SentiConfidence'] = listFeatures[i]['SentiConfidence']
        dfTmp['SentiInconsistence'] = listFeatures[i]['SentiInconsistence']
        dfExtractedFeature = dfExtractedFeature.append(dfTmp, ignore_index=True)
    #dfExtractedFeature.to_csv(fileNewFeatures, encoding='utf-8', index=False)
    # quit()


    # 将listDataFrameSequence按照时间先后进行排序（按照最后一个元素的时间！！！）
    lDateSort = []
    for i in range(len(listDataFrameSequence)):
        section = listDataFrameSequence[i]
        assert (section.values.shape[0] == Length)
        # d1 = str2date(section.iloc[0]['Date'])
        d4 = str2date(section.iloc[Length - 1]['Date'])
        # dC = d1 + (d4-d1)/2
        lDateSort.append((d4, i))

    lDateSort.sort()

    listNewDF = []  # 按日期先后顺序排序的DataFrame
    # 不要排序，乱序
    # sidx = numpy.random.permutation(len(lDateSort))  # 随机数下标
    # for i in sidx:
    # listNewDF.append(listDataFrameSequence[i])
    # 要排序！！！！
    for i in lDateSort:
        listNewDF.append(listDataFrameSequence[i[1]])

    numTotalData = len(listNewDF)
    print numTotalData
    numTrain = int(0.9 * numTotalData)
    TrainData = listNewDF[:numTrain]
    TestData = listNewDF[numTrain:]
    TestData = addLabelError(TestData, Error_Ratio)

    X_Train = []
    y_Train = []
    X_Test = []
    y_Test = []
    fileName_Train = []  # 以y的shape来记录每个序列中每个样本点的fileName，用于后续validation
    fileName_Test = []
    for i in TrainData:
        # if i['ReturnPredict'].values[0] != 0.0 and i['ReturnPredict'].values[0] != 1.0:
        #    print i['ReturnPredict'].values[0]
        #    print 'error ReturnPredict',i['FileName']
        # if numpy.isnan(i['ReturnPredict'].values[0]):
        #    print i['ReturnPredict'].values[0]
        #    print 'error ReturnConfidence',i['FileName']
        #X_Train.append(i.drop(
        #    ['PersonURL', 'StockID', 'Date', 'FileName', 'Reliability', 'StockTrendPre1', 'StockTrendPre2',
        #     'StockTrendPre3', 'StockTrendPre4', 'StockTrendPre5', 'StockTrendPre6', 'StockTrendPre7', 'StockTrendPre8',
        #     'StockTrendPre9', 'StockTrendPre10', 'StockTrendPre11', 'StockTrendPre12', 'StockTrendPre13',
        #     'StockTrendPre14', 'StockTrendPre15', 'StockTrendPre16', 'StockTrendPre17', 'StockTrendPre18',
        #     'StockTrendPre19', 'StockTrendPre20', 'StockTrendPre21', 'StockTrendPre22', 'StockTrendPre23',
        #     'StockTrendPre24', 'StockTrendPre25', 'HisStockPopNum', 'HisStockPopRatio', '7DaysStockPopNum',
        #     '7DaysStockPopRatio', 'CurDayNegNum', 'CurDayPosNum', '7DaysNegNum', '7DaysPosNum', '7DaysRight',
        #     '7DaysWrong', '7DaysRightNum', '7DaysWrongNum', '7DaysRightRatio', '7DaysProbNegWrong',
        #     '7DaysProbNegRight', '7DaysProbPosWrong', '7DaysProbPosRight'], axis=1).values)
        iPart = i.drop(['PersonURL','StockID','Date','FileName','Reliability'],axis=1)
        iPart = iPart.drop(listPersonSectorFeatures,axis=1)
        iPart = iPart.drop(listStockTrendFeatures, axis=1)
        iPart = iPart.drop(listStockPopSentiReliFeatures, axis=1)

        X_Train.append(iPart.values)
        y_Train.append(i['Reliability'].values[-1])  # 只记录序列最后一个元素的Reliability
        #fileName_Train.append(i['FileName'].values[-1])
        fileName_Train.extend(list(i['FileName'].values))
    for i in TestData:
        # if i['ReturnPredict'].values[0] != 0.0 and i['ReturnPredict'].values[0] != 1.0:
        #    print i['ReturnPredict'].values[0]
        #    print 'error ReturnPredict',i['FileName']
        # if numpy.isnan(i['ReturnPredict'].values[0]):
        #    print i['ReturnPredict'].values[0]
        #    print 'error ReturnConfidence',i['FileName']
        #X_Test.append(i.drop(
        #    ['PersonURL', 'StockID', 'Date', 'FileName', 'Reliability', 'StockTrendPre1', 'StockTrendPre2',
        #     'StockTrendPre3', 'StockTrendPre4', 'StockTrendPre5', 'StockTrendPre6', 'StockTrendPre7', 'StockTrendPre8',
        #     'StockTrendPre9', 'StockTrendPre10', 'StockTrendPre11', 'StockTrendPre12', 'StockTrendPre13',
        #     'StockTrendPre14', 'StockTrendPre15', 'StockTrendPre16', 'StockTrendPre17', 'StockTrendPre18',
        #     'StockTrendPre19', 'StockTrendPre20', 'StockTrendPre21', 'StockTrendPre22', 'StockTrendPre23',
        #     'StockTrendPre24', 'StockTrendPre25', 'HisStockPopNum', 'HisStockPopRatio', '7DaysStockPopNum',
        #     '7DaysStockPopRatio', 'CurDayNegNum', 'CurDayPosNum', '7DaysNegNum', '7DaysPosNum', '7DaysRight',
        #     '7DaysWrong', '7DaysRightNum', '7DaysWrongNum', '7DaysRightRatio', '7DaysProbNegWrong',
        #     '7DaysProbNegRight', '7DaysProbPosWrong', '7DaysProbPosRight'], axis=1).values)
        iPart = i.drop(['PersonURL', 'StockID', 'Date', 'FileName', 'Reliability'], axis=1)
        iPart = iPart.drop(listPersonSectorFeatures, axis=1)
        iPart = iPart.drop(listStockTrendFeatures, axis=1)
        iPart = iPart.drop(listStockPopSentiReliFeatures, axis=1)

        X_Test.append(iPart.values)
        y_Test.append(i['Reliability'].values[-1])
        fileName_Test.append(i['FileName'].values[-1])
    # print X_Test
    # print y_Test
    X_Train = numpy.asarray(X_Train)
    y_Train = numpy.asarray(y_Train)
    fileName_Train = numpy.asarray(fileName_Train)
    X_Test = numpy.asarray(X_Test)#
    y_Test = numpy.asarray(y_Test)
    fileName_Test = numpy.asarray(fileName_Test)
    print 'X_Train.shape', X_Train.shape
    print 'y_Train.shape', y_Train.shape
    print 'X_Test.shape', X_Test.shape
    print 'y_Test.shape', y_Test.shape

    #y_Train = y_Train.reshape((y_Train.shape[0],y_Train.shape[1],1))
    #y_Test = y_Test.reshape((y_Test.shape[0],y_Test.shape[1],1))
    #fileName_Train = fileName_Train.reshape((fileName_Train.shape[0],fileName_Train.shape[1],1))
    #fileName_Test = fileName_Test.reshape((fileName_Test.shape[0],fileName_Test.shape[1],1))
    print y_Train.shape
    print y_Test.shape

    output = open(fileOutputSequences, 'wb')

    # Pickle dictionary using protocol 0.
    pickle.dump((X_Train, y_Train, X_Test, y_Test, fileName_Train, fileName_Test), output)
    output.close()
    #quit()

    #with open(fileLSTMUseData, 'w') as f:
    #    for i in listFileName:
    #        f.write(('%s\n') % (i))
    return dfExtractedFeature,listFileName
