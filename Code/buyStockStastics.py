# dfsdfsds encoding: utf-8

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from datetime import date
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from scipy.sparse.csr import csr_matrix
import multiprocessing

from utility import outForSequenceData
from utility import str2date,addColumns,save_csv,load_csv
#from gensim.models.doc2vec import TaggedLineDocument
import time
import pickle
from math import sqrt

#SEED = 1428
#np.random.seed(SEED)

def calcTopN(dfData,aryTop,keyCorrect,keyConfi,asc=False):
    dfTotal = pandas.DataFrame()
    dfTotal[keyCorrect] = dfData[keyCorrect]
    dfTotal[keyConfi] = dfData[keyConfi]
    dfTotal = dfTotal.sort_values(keyConfi, ascending=asc)
    numTotal = len(dfTotal.values)

    aryN.append(numTotal)
    for i in aryN:
        if i > numTotal:
            print 'can not extract top%d, it is too large' % (i)
        else:
            dftopN = dfTotal[:i]
            numTopNCorrect = len(dftopN[dftopN[keyCorrect] == True].values)
            ratio = float(numTopNCorrect) / i
            print 'Top %d,%d,%.3f' % (i, numTopNCorrect, ratio)


#股票板块信息
openStockSector = open('StockSector.pkl', 'rb')
dictStockSector,setSectorName = pickle.load(openStockSector)
listSectorName = list(setSectorName)
#print listSectorName
openStockSector.close()

def initDicSector(setSectorName):
    dis,stk,cnf,cnfSum = dict(),dict(),dict(),dict()
    for i in setSectorName:
        dis[i]=0
        stk[i]=''
        cnf[i]=-1.0
        cnfSum[i] = 0.0
    return dis,stk,cnf,cnfSum

def getSectorInfo(sData,dfList,setSectorName):
    # 分别是所有板块的股票数目分布，各个板块中置信度最大的股票及置信度值，各个板块中股票置信度之和
    dicDist, dicMaxStock, dicMaxConfidence, dicSumCnf = initDicSector(setSectorName)

    for i in sData.index:
        stockID = dfList.loc[i, 'StockID'][2:]#去掉前缀，只保留六位数字
        confdc = dfList.loc[i, 'Confidence']
        sectorID = dictStockSector[stockID]  # 板块ID
        dicDist[sectorID] += 1
        # 找到各个板块中，置信度最高的股票
        if dicMaxStock[sectorID] == '':
            dicMaxStock[sectorID] = stockID
            dicMaxConfidence[sectorID] = confdc
        else:
            if confdc > dicMaxConfidence[sectorID]:
                dicMaxStock[sectorID] = stockID
                dicMaxConfidence[sectorID] = confdc
        dicSumCnf[sectorID] += confdc  # 当前板块中股票权重累加
    numSector = 0
    for i in dicDist:
        if dicDist[i] > 0:
            numSector += 1
    return dicDist,numSector,dicMaxStock,dicMaxConfidence,dicSumCnf

'''
df1 = pandas.read_csv('final_3.csv',encoding='utf-8')
dfAmount = pandas.DataFrame()
dfRatio = pandas.DataFrame()
dfReli = pandas.DataFrame()
dfAmount['PersonURL'] = df1['PersonURL']
dfRatio['PersonURL'] = df1['PersonURL']
dfReli['PersonURL'] = df1['PersonURL']
listAmount,listRatio,listReli = [],[],[]
fd = file( "tenCate.txt", "r" )

for line in fd.readlines():
    listAmount.append(line.strip())
    listRatio.append(line.strip()+'_P')
    listReli.append(line.strip() + '_R')
for i in listAmount:
    dfAmount[i] = df1[i]
for i in listRatio:
    dfRatio[i] = df1[i]
for i in listReli:
    dfReli[i] = df1[i]
dfAmount.to_csv('personBankuaiAmount.csv',encoding='utf-8',index=False)
dfRatio.to_csv('personBankuaiRatio.csv',encoding='utf-8',index=False)
dfReli.to_csv('personBankuaiReli.csv',encoding='utf-8',index=False)
quit()
'''

def funcDayIntraSecAverInterSecAver(sData, dfList,dt,numInDt, dicDist, numSector, dicMaxStock, dicMaxConfidence, dicSumCnf):
    for i in sData.index:
        stockID = dfList.loc[i,'StockID'][2:]
        sectorID = dictStockSector[stockID]#板块ID
        # 板块内及板块间均平均加权
        sData[i]=1.0/dicDist[sectorID]/numSector

def funcDayIntraSecConfiInterSecConfi(sData, dfList,dt,numInDt, dicDist, numSector, dicMaxStock, dicMaxConfidence, dicSumCnf):
    # 板块内按照置信度加权，板块间按照置信度加权
    # 首先计算出所有板块的置信度之和
    maxAllConfSum = 0.
    for k, v in dicSumCnf.items():
        if v >= 0:
            maxAllConfSum += v
    for i in sData.index:
        stockID = dfList.loc[i,'StockID'][2:]
        sectorID = dictStockSector[stockID]#板块ID
        confdc = dfList.loc[i, 'Confidence']
        #sData[i] = confdc / dicSumCnf[sectorID] * dicSumCnf[sectorID] / maxAllConfSum
        sData[i] = confdc / maxAllConfSum

def funcDayIntraSecConfiInterSecAver(sData, dfList,dt,numInDt, dicDist, numSector, dicMaxStock, dicMaxConfidence, dicSumCnf):
    for i in sData.index:
        stockID = dfList.loc[i,'StockID'][2:]
        sectorID = dictStockSector[stockID]#板块ID
        confdc = dfList.loc[i, 'Confidence']
        #板块内按照置信度加权，板块间平均加权
        sData[i] = confdc / dicSumCnf[sectorID] / numSector

def funcDayIntraSecTop1InterSecAver(sData, dfList,dt,numInDt, dicDist, numSector, dicMaxStock, dicMaxConfidence, dicSumCnf):
    # 板块内选择置信度最高的一只股票，板块间平均加权
    for i in sData.index:
        stockID = dfList.loc[i,'StockID'][2:]
        sectorID = dictStockSector[stockID]#板块ID
        confdc = dfList.loc[i, 'Confidence']
        if dicMaxStock[sectorID] != stockID:
            sData[i]=0
        else:
            sData[i] = 1.0 / numSector

def funcDayIntraSecTop1InterSecConfi(sData, dfList,dt,numInDt, dicDist, numSector, dicMaxStock, dicMaxConfidence, dicSumCnf):
    # 板块内选择置信度最高的一只股票，板块间按股票置信度加权
    # 首先计算dicMaxConfidence中所有板块最高置信度之和
    maxConfSum = 0.
    for k,v in dicMaxConfidence.items():
        if v >= 0:
            maxConfSum += v
    for i in sData.index:
        stockID = dfList.loc[i,'StockID'][2:]
        sectorID = dictStockSector[stockID]#板块ID
        confdc = dfList.loc[i, 'Confidence']
        if dicMaxStock[sectorID] != stockID:
            sData[i]=0
        else:
            sData[i] = confdc / maxConfSum

def funcDayAver(sData, dfList,dt,numInDt):
    w = 1.0 / numInDt
    for j in sData.index:
        sData[j] = w

def funcDayTopN(sData, dfList,dt,numInDt, maxC, maxIdx):
    idxRandom = np.random.randint(len(sData.index))
    iii = 0
    maxIdxRandom = -1
    noData = True
    for j in sData.index:
        confdc = dfList.loc[j, 'Confidence']

        if iii == idxRandom:
            maxIdxRandom = j
        iii += 1

        # 首先从maxC中选择一个最小的出来
        idxMin = -1
        mTag = 10000
        for i in range(len(maxC)):
            if mTag > maxC[i]:
                mTag = maxC[i]
                idxMin = i
        # 然后confdc与这个最小的数进行比较

        if confdc > mTag:
            maxC[idxMin] = confdc
            maxIdx[idxMin] = j
            noData = False
    if noData == True:
        sData[:] = 0.
        sData[maxIdxRandom] = 1.
    else:
        sumWeight = 0.
        numWeight = 0
        for j in range(len(maxC)):
            if maxC[j] > -1.:
                confdc = dfList.loc[maxIdx[j], 'Confidence']
                sumWeight += confdc
                numWeight += 1
        sData[:] = 0.
        for j in range(len(maxC)):
            if maxC[j] > -1.:
                confdc = dfList.loc[maxIdx[j], 'Confidence']
                sData[maxIdx[j]] = confdc / sumWeight #按置信度分配
                sData[maxIdx[j]] = 1.0 / numWeight #均匀分配

def funcDayTopOne(sData, dfList,dt,numInDt):
    maxC = [-1]
    maxIdx = [-1]
    funcDayTopN(sData, dfList, dt, numInDt, maxC, maxIdx)

def funcDayTopFive(sData, dfList,dt,numInDt):

    maxC = [-1,-1,-1,-1,-1]
    maxIdx = [-1, -1, -1, -1, -1]
    funcDayTopN(sData, dfList, dt, numInDt, maxC, maxIdx)

def funcDayTopTen(sData, dfList,dt,numInDt):

    maxC = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    maxIdx = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    funcDayTopN(sData, dfList, dt, numInDt, maxC, maxIdx)

def funcDayTop15(sData, dfList,dt,numInDt):

    maxC = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    maxIdx = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,-1,-1,-1,-1,-1]
    funcDayTopN(sData, dfList, dt, numInDt, maxC, maxIdx)

def funcDayTop20(sData, dfList,dt,numInDt):

    maxC = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    maxIdx = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    funcDayTopN(sData, dfList, dt, numInDt, maxC, maxIdx)



def funcDayTopSector(sData, dfList,dt,numInDt, dicDist, numSector, dicMaxStock, dicMaxConfidence, dicSumCnf):
    # 板块内按照置信度加权，板块间按照置信度加权
    # 首先计算出所有板块的置信度之和
    numValidSec = 0
    for k, v in dicSumCnf.items():
        if v > 0:
            numValidSec +=1
    maxC = []
    maxIdx = []
    for i in range(numValidSec):
        maxC.append(-1)
        maxIdx.append(-1)
    funcDayTopN(sData, dfList,dt,numInDt,maxC,maxIdx)


def calcDayWeight(dfList,weightKey,funcDay,isUseSectorInfo):
    dfList[weightKey] = np.zeros(len(dfList.values))
    dateHistogram = dfList['Date'].value_counts()
    listDateID = list(dateHistogram._index)
    listDateCount = list(dateHistogram.get_values())
    numDate = len(listDateID)
    for i in range(numDate):
        dt = listDateID[i]
        numInDt = listDateCount[i]

        sData = dfList.loc[dfList['Date'] == dt, weightKey]
        if isUseSectorInfo:
            dicDist, numSector, dicMaxStock, dicMaxConfidence, dicSumCnf = getSectorInfo(sData, dfBuyList, setSectorName)
            funcDay(sData,dfBuyList,dt,numInDt,dicDist, numSector, dicMaxStock, dicMaxConfidence, dicSumCnf)
        else:
            funcDay(sData, dfBuyList, dt, numInDt)
        dfList.loc[dfList['Date'] == dt, weightKey] = sData


def outProfit(dfList,weightKey,setErrorFile):
    moneyBuy, moneySell = 0.0, 0.0  # 买入卖出赚的钱
    iB, iS = 0,0
    numCorrectBuy, numWrongBuy, numCorrectSell, numWrongSell = 0,0,0,0
    #aa = dfList['Date'].values
    #for i in aa:
    #    print i
    dictDayProfit=dict()
    for i in dfList.index:
        dtDate = dfList.loc[i,'Date']
        if dtDate not in dictDayProfit:
            dictDayProfit[dtDate] = dict()
            dictDayProfit[dtDate]['buy'] = 0.
            dictDayProfit[dtDate]['sell'] = 0.
        fBuy = dfList.loc[i,'FileName']
        considerKaopu = dfList.loc[i,'Predict']
        senti = dfList.loc[i,'Label']
        v = dfList.loc[i, 'AfterOneDayTrendUnlimited_OpenClose']
        weight = dfList.loc[i,weightKey]
        if weight == 0:
            continue

        if fBuy in setErrorFile:
            senti = 1 - senti
            pass
        if considerKaopu == 0:
            senti = 1 - senti
            pass

        if senti == 1:
            moneyBuy += weight * v
            dictDayProfit[dtDate]['buy'] += weight * v
            iB += 1
            if v > 0:
                numCorrectBuy += 1
            else:
                numWrongBuy += 1
        elif senti == 0:  # and vP <= 0:
            moneySell -= weight * v
            dictDayProfit[dtDate]['sell'] -= weight * v
            iS += 1
            if v > 0:
                numWrongSell += 1
            else:
                numCorrectSell += 1
    profitBuy, profitSell = moneyBuy * 100,moneySell * 100
    profitSum = profitBuy + profitSell
    if numCorrectBuy + numWrongBuy == 0:
        accuBuy = 0.
    else:
        accuBuy = float(numCorrectBuy) / (numCorrectBuy + numWrongBuy)
    if numCorrectSell + numWrongSell==0:
        accuSell = 0.
    else:
        accuSell = float(numCorrectSell) / (numCorrectSell + numWrongSell)
    accuSum = float(numCorrectBuy + numCorrectSell) / (numCorrectBuy + numCorrectSell + numWrongBuy + numWrongSell)
    #print u'买入利润：', moneyBuy * 100, '%', u'卖出利润：', moneySell * 100, '%', u'最终利润：', (moneyBuy + moneySell) * 100, '%'
    #print u'买入', iB, u'次；卖出', iS, u'次'
    #print u'正确买入', numCorrectBuy, u'次；正确卖出', numCorrectSell, u'次'
    #print u'错误买入', numWrongBuy, u'次；错误卖出', numWrongSell, u'次'
    #print u'正确率', accuSum
    #print moneyBuy * 100, moneySell * 100, iB, iS, numCorrectBuy, numCorrectSell, numWrongBuy, numWrongSell
    #print '%.2f\n%.2f\n%.2f\n%d\n%d\n%d\n%.3f\n%.3f\n%.3f\n' \
    #      % (profitBuy,profitSell,profitSum,iB,iS,iB+iS,accuBuy,accuSell,accuSum)
    aryDay=[]
    aryDay2=[]
    for k in dictDayProfit:
        #print k,dictDayProfit[k]['buy'],dictDayProfit[k]['sell']
        aryDay.append((k,dictDayProfit[k]['buy']))
    aryDay.sort()
    dtTmp=0.
    for i in aryDay:
        dtTmp+=i[1]
        aryDay2.append((i[0],dtTmp*10000))
    for i in aryDay2:
        print i[0],i[1]
    quit()
    return (profitBuy,profitSell,profitSum,iB,iS,iB+iS,accuBuy,accuSell,accuSum)

def getErrorFile(fn):
    s = set()
    with open(fn,'r') as f:
        for i in f.readlines():
            s.add(i.strip())
    return s

def addDateSentiFromDF(df1,df2):
    cols = ['AfterOneDayTrendUnlimited_OpenClose']
    if 'Label' not in df1.columns:
        cols.append('Label') #如果是ARMA等时序预测数据，则自己提供Label
    if 'Date' not in df1.columns:
        cols.append('Date') #如果是ARMA等时序预测数据，则自己提供Date
    if 'StockID' not in df1.columns:
        cols.append('StockID') #如果是ARMA等时序预测数据，则自己提供StockID
    return addColumns(df1,df2,'FileName',cols)


#split_date = '2015-11-22' #这是小数据baseline的评估起始点
#split_year,split_month,split_day = 2015,11,22
split_date = '2016-03-23' #大数据
split_year,split_month,split_day = 2016,3,23


bARMAWeight=False
fnTmpF = load_csv('validation\\0117bkupFeatureARMA\\svm-big-rbf-OpenClose-noerror-0117-addFeatureARMA.csv')
fnFilter = fnTmpF['FileName'].values
#fnBuyList='validation\\TotalreturnResult_1.csv'
fnBuyList='validation\\0117bkupFeatureARMA\\svm-big-rbf-OpenClose-noerror-0117-addFeatureARMA.csv'
#fnBuyList='validation\\best 50-100-fm-big-OpenClose-noerror-0117-addFeatureARMA - No_FM.csv'

#fnBuyList='validation\\0117bkupFeatureARMA\\knn-big-OpenClose-noerror-0117-addFeatureARMA.csv'
#fnBuyList='validation\\TotalreturnResult_1.csv'

fndFMData='data\\dFM-big-oldLabel.csv'#包含情感倾向和涨跌信息
fnErrorSentiment='validation\\errorFileName-baselinebig.txt'

dfM=pandas.read_csv(fndFMData,encoding='utf-8')

#这是ARMA时序预测模型的结果
dfResults = pandas.read_csv('validation\\TotalreturnResult_1.csv', encoding='utf-8')
#dfResults = dfResults.sort_values('ReturnErr')#按照误差进行排序
#dfResults = dfResults.drop_duplicates('Date', keep='first')  # 对于在同一天的多条指示，保留置信度最大的那条
dfResults['Date'] = pandas.to_datetime(dfResults['Date'])
dfTest = dfResults[dfResults['Date']>=pandas.to_datetime(split_date)]

setErrorFile=set()
#if fnErrorSentiment != '':
#    setErrorFile = getErrorFile(fnErrorSentiment)

dfBuyListRaw=pandas.read_csv(fnBuyList,encoding='utf-8')
print len(dfBuyListRaw.values)
#如果是ARMA或GARCH,则需要过滤一下
#dfBuyListRaw = dfBuyListRaw[dfBuyListRaw['FileName'].isin(fnFilter)]
#save_csv(dfBuyListRaw,'ARMA-Part.csv')
#quit()
print len(dfBuyListRaw.values)


'''
colConfiARMA = []
colLableARMA = []
colKaopuARMA = np.ones(len(dfBuyListRaw.values))
for i in dfBuyListRaw.index:
    err = dfBuyListRaw.loc[i,'ReturnErr']
    predict = dfBuyListRaw.loc[i,'ReturnPredict']
    c = 1.0 / sqrt(err)
    l = 1.
    if predict <= 0:
        l = 0.
    colConfiARMA.append(c)
    colLableARMA.append(l)
dfBuyListRaw['Confidence'] = colConfiARMA
dfBuyListRaw['Label'] = colLableARMA
dfBuyListRaw['Predict'] = colKaopuARMA
save_csv(dfBuyListRaw,fnBuyList)
quit()
'''

dfBuyListRaw = addDateSentiFromDF(dfBuyListRaw,dfM)
dfARMALabel = pandas.DataFrame()
dfARMALabel['FileName'] = dfResults['FileName']
dfARMALabel['ARMALabel'] = dfResults['Label']
dfBuyListRaw = pandas.merge(dfBuyListRaw,dfARMALabel,how='left',on='FileName')

#u=0.9

#将ARMA中的置信度进行加权
if bARMAWeight:
    for i in range(len(dfBuyListRaw.values)):
        idx = dfBuyListRaw.index[i]
        fn = dfBuyListRaw.loc[idx, 'FileName']
        considerKaopu = dfBuyListRaw.loc[idx, 'Predict']
        senti = dfM[dfM['FileName'] == fn]['Label'].values[0]
        # print senti

        vP = dfTest[dfTest['FileName'] == fn]['ReturnPredict'].values[0]
        armaConf = dfTest[dfTest['FileName'] == fn]['ARMAConfidence'].values[0]
        clsConf =  dfBuyListRaw.loc[idx, 'Confidence']
        if fn in setErrorFile:
            senti = 1 - senti
        if considerKaopu == 0:
            senti = 1 - senti
        if senti == 1 and vP <= 0:
            dfBuyListRaw.loc[idx, 'Confidence'] -= armaConf / 3.5
            #dfBuyListRaw.loc[idx, 'Confidence'] = u * clsConf - (1.-u) * armaConf
        elif senti == 0 and vP > 0:
            dfBuyListRaw.loc[idx, 'Confidence'] -= armaConf / 3.5
            #dfBuyListRaw.loc[idx, 'Confidence'] = u * clsConf - (1. - u) * armaConf
        else:
            dfBuyListRaw.loc[idx, 'Confidence'] += armaConf / 3.5
            #dfBuyListRaw.loc[idx, 'Confidence'] = u * clsConf + (1. - u) * armaConf
        if dfBuyListRaw.loc[idx, 'Confidence'] < 0:
            considerKaopu = 1.0 - considerKaopu
            dfBuyListRaw.loc[idx, 'Predict'] = considerKaopu
            dfBuyListRaw.loc[idx, 'Result'] = not dfBuyListRaw.loc[idx, 'Result']
            dfBuyListRaw.loc[idx, 'Confidence'] = 0. - dfBuyListRaw.loc[idx, 'Confidence']
            #dfBuyListRaw.loc[idx, 'Confidence'] = 0.00001


aryN = [25,30,50,60, 100,  200, 500]
#calcTopN(dfBuyListRaw,aryN,'Result','Confidence',False)
#quit()
dfBuyListRaw=dfBuyListRaw.sort_values('Confidence',ascending=False) #按照置信度进行逆序排序
#dfBuyListRaw=dfBuyListRaw.drop_duplicates('Date',keep='first')#对于在同一天的多条指示，保留置信度最大的那条
dfBuyListRaw=dfBuyListRaw.sort_values('Date') #按日期顺序排序
dfBuyList = pandas.DataFrame()
dfBuyList[dfBuyListRaw.columns] = dfBuyListRaw[dfBuyListRaw.columns]
dfBuyList['Weight']=np.zeros_like(dfBuyListRaw['AfterOneDayTrendUnlimited_OpenClose'])
#dfBuyList = dfBuyList[dfBuyList['FileName'].isin(fnFilter)]

dateHistogram = dfBuyList['Date'].value_counts()
listDateID = list(dateHistogram._index)
listDateCount = list(dateHistogram.get_values())
numDate = len(listDateID)
print numDate, u'个日期'

numBuyList = len(dfBuyList.values)
print numBuyList,u'条股评'

weightTotalOne = np.ones(numBuyList)
dfBuyList['WtTotalOne'] = weightTotalOne
dfBuyList = dfBuyList.sort_values('Confidence',ascending=False)
numTop10 = int(0.1 * numBuyList)
weightTop10 = np.zeros(numBuyList)
weightTop10[:numTop10] = 1
dfBuyList['WtTop10'] = weightTop10
numTop1 = int(0.01 * numBuyList)
weightTop1 = np.zeros(numBuyList)
weightTop1[:numTop1] = 1
dfBuyList['WtTop1'] = weightTop1

numTop2p5 = int(0.025 * numBuyList)
weightTop2p5 = np.zeros(numBuyList)
weightTop2p5[:numTop2p5] = 1
dfBuyList['WtTop2p5'] = weightTop2p5

numTop5 = int(0.05 * numBuyList)
weightTop5 = np.zeros(numBuyList)
weightTop5[:numTop5] = 1
dfBuyList['WtTop5'] = weightTop5


aryAve = []
numLoop = 1
for loop in range(numLoop):
    aryExcel = []
    #print u'全量:'
    #aryExcel.append(outProfit(dfBuyList, 'WtTotalOne', setErrorFile))
    #print u'全量Top10%:'
    #aryExcel.append(outProfit(dfBuyList, 'WtTop10', setErrorFile))
    #print u'全量Top5%:'
    #aryExcel.append(outProfit(dfBuyList, 'WtTop5', setErrorFile))
    #print u'全量Top2.5%:'
    #aryExcel.append(outProfit(dfBuyList, 'WtTop2p5', setErrorFile))
    #print u'全量Top1%:'
    #aryExcel.append(outProfit(dfBuyList, 'WtTop1', setErrorFile))

    aryOutProfit = [
        #(u'按天平均:', 'WtDayAverage', funcDayAver, False)
        # (u'按天Top1:', 'WtDayTop1', funcDayTopOne, False)
        #(u'按天Top5:', 'WtDayTop5', funcDayTopFive, False)
        #, (u'按天Top10:', 'WtDayTop10', funcDayTopTen, False)
        #, (u'按天Top15:', 'WtDayTop15', funcDayTop15, False)
        #, (u'按天Top20:', 'WtDayTop20', funcDayTop20, False)
        #, (u'按天TopSector:', 'WtDayTopSector', funcDayTopSector, True)
        #, (u'按天板块内置信度加权板块间置信度加权:', 'funcDayIntraSecConfidenceInterSecConfidence', funcDayIntraSecConfiInterSecConfi, True)
        #, (u'按天板块内平均加权板块间平均加权:', 'WtDayIntraSectorAverInterSectorAver', funcDayIntraSecAverInterSecAver, True)
        #, (u'按天板块内置信度加权板块间平均加权:', 'WtDayIntraSectorConfidenceInterSectorAver', funcDayIntraSecConfiInterSecAver, True)
         (u'按天板块内Top1板块间平均加权:', 'funcDayIntraSecTop1InterSecAver', funcDayIntraSecTop1InterSecAver, True)
        #, (u'按天板块内Top1板块间置信度加权:', 'funcDayIntraSecTop1InterSecConfidence', funcDayIntraSecTop1InterSecConfi, True)
                    ]
    #aryOutProfit = [(u'按天Top1:', 'WtDayTop1', funcDayTopOne, False)]
    for i in aryOutProfit:
        print i[0]
        calcDayWeight(dfBuyList, i[1], i[2], i[3])
        aryExcel.append(outProfit(dfBuyList, i[1], setErrorFile))
    print 'output for Excel:'
    for i in range(len(aryExcel[0])):
        for j in range(len(aryExcel)):
            print aryExcel[j][i],
        print '\n',

    if len(aryAve) == 0:
        for kk in aryExcel:
            aryAve.append(list(kk))
    else:
        for i in range(len(aryExcel)):
            for j in range(len(aryExcel[0])):
                aryAve[i][j] += aryExcel[i][j]

for i in range(len(aryExcel)):
    for j in range(len(aryExcel[0])):
        d = float(aryAve[i][j])
        aryAve[i][j] = type(aryAve[i][j])(d / numLoop)

print '\n'
print '\n'

for i in range(len(aryAve[0])):
    for j in range(len(aryAve)):
        print aryAve[j][i],
    print '\n',