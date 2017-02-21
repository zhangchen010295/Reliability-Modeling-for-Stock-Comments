# dfsdfsds encoding: utf-8
#2016-12-20特征中添加当天及前三天的销售额和销售量
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas
import pickle
import numpy as np

from utility import getFullFeatures,save_csv,load_csv
#from gensim.models.doc2vec import TaggedLineDocument
import time
from extractStockFeatures import buildStockPopSentiReliFeatures
from extractPersonFeatures import buildPersonFeatures
from buildSequencePntWise import buildSequenceData

SEED = 1428
np.random.seed(SEED)

Length = 4 #序列长度

#小数据
'''
#fnValidPerson = 'sequenceData\\openclose\\lstmUsedDataLen4FileName_OneDayTrend_OpenClose.txt'
#fileIn = "data\\total\\stopword\\comments_filter.csv"
fileIn = 'data\\dFM-small.csv'
fnStockInfo='StockInfo_new.csv'
fnStockFeatures='StockPopSentiReliabilityFeature_OneDayTrend_OpenClose.csv'
fnPersonFeatures='PersonFeature_Total_OneDayTrend_OpenClose_AdjustHisSplit_30_90.csv'
fnNewFeatures='sequenceData\\openclose\\NewFeatures_4_OneDayTrend_OpenClose.csv'
fnFullFeaturesOutput='fullFeature_OneDayTrend_OpenClose_AdjustHisSplit_30_90.csv'
fileOutputSequences = 'StockDataset_4_OneDayTrend_OpenClose_adderror.pkl'
fileOutputStastics = 'stasticsUsesmallDataSequence_24.csv'
'''

#大数据

#fnValidPerson = 'sequenceData\\openclose\\lstmUsedDataLen4FileNameTotal_OneDayTrend_OpenClose.txt'
#fileIn = "data\\total\\stopword\\TotalCommentsAddLabels2.csv"
fileIn = 'data\\dFM-big-oldLabel.csv'
#fnStockInfo='StockInfoTotal_new.csv'
fnStockFeatures='StockPopSentiReliabilityFeature_Total_OneDayTrend_OpenClose.csv'
#fnPersonFeatures='PersonFeature_Total_OneDayTrend_OpenClose.csv'
fnPersonFeatures='PersonFeature_Total_OneDayTrend_OpenClose_AdjustHisSplit_30_90.csv'
fnNewFeatures='sequenceData\\openclose\\NewFeatures_4_Total_OneDayTrend_OpenClose.csv'
fnFullFeaturesOutput='fullFeature_Total_OneDayTrend_OpenClose_AdjustHisSplit_30_90.csv'
fileOutputSequences = 'StockDataset_4_Total_person_noerr_seq.pkl'
fileOutputStastics = 'stasticsUsebigDataSequence_24.csv'



def buildDfFeature(datafme):
    #该csv用来提取股票热度情感靠谱率特征
    dfOutput=pandas.DataFrame()
    dfOutput['PersonURL']=datafme['PersonURL']
    dfOutput['StockID'] = datafme['StockID']
    dfOutput['Date'] = datafme['Date']
    dfOutput['Label'] = datafme['Label']
    dfOutput['FileName'] = datafme['FileName']
    dfOutput['Reliability'] = datafme['Reliability']
    return dfOutput


def buildFeatures():
    #data = pandas.read_csv(fileIn, encoding='utf-8')

    #dfStockInfo = pandas.read_csv(fnStockInfo, encoding='utf-8')  # 这是股价涨跌信息数据
    #dfStockInfo = dfStockInfo.drop('Date', axis=1)

    #dfM = pandas.merge(data, dfStockInfo, how='left', on='FileName')
    dfM = pandas.read_csv(fileIn, encoding='utf-8')

    PersonURL = dfM.pop('PersonURL')
    dfM.insert(0, 'PersonURL', PersonURL)
    StockID = dfM.pop('StockID')
    dfM.insert(1, 'StockID', StockID)
    DateC = dfM.pop('Date')
    dfM.insert(2, 'Date', DateC)
    Lc = dfM.pop('Label')
    dfM.insert(3, 'Label', Lc)

    YComment = dfM["Label"].values  # 情感是积极还是消极
    # YReal = dfM["StockTrendAfter1"].values
    YReal = dfM["AfterOneDayTrendUnlimited_OpenClose"].values
    after1Value = dfM["StockTrendAfter1"].values
    after2Value = dfM["StockTrendAfter2"].values
    after3Value = dfM["StockTrendAfter3"].values
    after4Value = dfM["StockTrendAfter4"].values
    after5Value = dfM["StockTrendAfter5"].values
    after6Value = dfM["StockTrendAfter6"].values
    YReal=[]
    for i in range(len(after1Value)):
        v1=after1Value[i]
        delta = 0.02
        v2 = after2Value[i]
        v3 = after3Value[i]
        v4 = after4Value[i]
        v5 = after5Value[i]
        v6 = after6Value[i]
        bHigh = False
        bLow = False
        if v2-v1 >= delta or v3-v1 >= delta or v4-v1 >= delta\
            or v5-v1 >= delta or v6-v1 >= delta:
            bHigh =True
            YReal.append(1.0)
        else:
            YReal.append(0.0)
        #if bHigh

    YReal = dfM["AfterOneDayTrendUnlimited_OpenClose"].values
    dtAry = dfM['Date'].values
    assert (len(YComment) == len(YReal))

    YKaopu = np.zeros_like(YComment)
    posiUp, posiDown, posiAver, negaUp, negaDown, negaAver=0,0,0,0,0,0

    numUp, numDown, numAver, sumlR, numLA, numEQ, numLE = 0,0,0,0,0,0,0
    pR, pW, nR, nW = 0,0,0,0

    threshold=0.0
    for idx35 in range(len(YComment)):
        lC = YComment[idx35]
        lR = YReal[idx35]
        sumlR += lR
        if lR >= 0.02:
            numUp+=1
        elif lR <= -0.02:
            numDown+=1
        else:
            numAver+=1
        if lR > 0:
           numLA+=1
        elif lR < 0:
            numLE+=1
        else:
            numEQ +=1

        if lR >= 0.02:
            if lC == 1:
                posiUp+=1
            else:
                negaUp+=1
        elif lR <= -0.02:
            if lC == 1:
                posiDown+=1
            else:
                negaDown+=1
        else:
            if lC == 1:
                posiAver+=1
            else:
                negaAver+=1

        if lR > threshold and lC == 1:
            YKaopu[idx35]=1
            pR+=1
        elif lR <= threshold and lC == 0:
            YKaopu[idx35]=1
            nR+=1
        elif lR > threshold and lC == 0:
            nW+=1
        elif lR <= threshold and lC == 1:
            pW+=1
        else:
            print dtAry[idx35],lR,lC


    #print numLA,numLE,numEQ
    #print numUp,numDown,numAver,sumlR
    #print u'看多且大涨', posiUp
    #print u'看多但大跌', posiDown
    #print u'看空但大涨', negaUp
    #print u'看空且大跌', negaDown
    #print u'看多没反应', posiAver
    #print u'看跌没反应', negaAver

    print u'看多靠谱', pR
    print u'看多不靠谱', pW
    print u'看空靠谱', nR
    print u'看空不靠谱', nW
    print pR + pW + nR + nW
    #quit()


    #dfConfidenceAll = pandas.read_csv('data\\total\\stopword\\svm-linear-confidence-All.csv', encoding='utf-8')
    #dfReliability = pandas.DataFrame()
    #dfReliability['FileName'] = dfM['FileName']
    #dfReliability['Reliability'] = dfM['Reliability']
    #dfConfidenceAll = pandas.merge(dfConfidenceAll,dfReliability,how='left',on='FileName')
    #dfConfidenceAll.to_csv('data\\total\\stopword\\svm-linear-confidence-All.csv', encoding='utf-8', index=False)
    #quit()
    dfM['Reliability'] = YKaopu
    dfFeature = buildDfFeature(dfM)
    #dfFeature.to_csv('validation2.csv', encoding='utf-8', index=False)
    #quit()
    dfStockPopSentiReliFeature = buildStockPopSentiReliFeatures(dfFeature)
    dfStockPopSentiReliFeature.to_csv(fnStockFeatures, encoding='utf-8',index=False)

    dfPersonFeature = buildPersonFeatures(dfFeature)
    dfPersonFeature.to_csv(fnPersonFeatures, encoding='utf-8',index=False)

    # 合并股票及股评员特征
    dfM = pandas.merge(dfM, dfStockPopSentiReliFeature, how='left', on='FileName')
    dfM = pandas.merge(dfM, dfPersonFeature, how='left', on='FileName')

    # 输出当前全特征数据，为后续构建sequence data提供input
    dfFullFeatures = getFullFeatures(dfM)
    #dfFullFeatures.to_csv('validation2.csv',encoding='utf-8',index=False)
    #quit()
    dfNewFeature,listValidFile = buildSequenceData(dfFullFeatures,Length,fileOutputSequences,fileOutputStastics)
    #dfNewFeature = outForSequenceData(fnFullFeaturesOutput,dfM)
    #quit()

    # 合并confidence以及inconsistence特征
    #dfNewFeature = pandas.read_csv(fnNewFeatures, encoding='utf-8')
    dfM = pandas.merge(dfM, dfNewFeature, how='left', on='FileName')

    # 过滤掉发少于5条股评的人
    # 或者根据需要对数据进行过滤
    dfM = dfM[dfM['FileName'].isin(listValidFile)]
    print len(dfM.values)
    personHistogram = dfM['PersonURL'].value_counts()
    listPersonID = list(personHistogram._index)
    listPersonCount = list(personHistogram.get_values())
    numPerson = len(listPersonID)
    print u'序列化后剩下股评员人数：',numPerson


    dfM = dfM.sort_values('Date')  # 按照日期进行排序
    print len(dfM['Text'].values), u'个总文档'
    print dfM['Date'].values[0],dfM['Date'].values[-1]
    dfM.to_csv('data4classification_OpenClose_oldLabel_length4_0117.csv',encoding='utf-8',index=False)


if __name__ == "__main__":
    t1 = time.clock()
    buildFeatures()
    t2 = time.clock()
    print 'elasped time:%s s' % (t2 - t1)