# dfsdfsds encoding: utf-8
# 2016-12-20特征中添加当天及前三天的销售额和销售量
# 将特征提取剥离开来，这里只做分类
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas
import pickle
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
from sklearn.feature_selection import chi2,f_classif
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from scipy.sparse.csr import csr_matrix
import multiprocessing
#from gensim.models import Doc2Vec
#from keras.preprocessing import sequence
#from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from sklearn.grid_search  import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
#from keras.layers import LSTM, SimpleRNN, GRU
#from keras.datasets import imdb
from utility import *
#from gensim.models.doc2vec import TaggedLineDocument
import time
from pyfm import pylibfm
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

SEED = 1428 #SVM LR
errorRatio = 0.#0.25
#SEED = 1027 #NB, kNN
#errorRatio = 0.3
np.random.seed(SEED)


def save_TestResults(model,fnoutput,X_test,y_test,fileName_Test):
    probY = model.predict_proba(X_test)
    clsY = model.predict_classes(X_test)
    #print(probY.shape, clsY.shape, y_test.shape)
    nSum = 0
    nCorect = 0
    dataR = []
    for i in range(probY.shape[0]):
        yReal = y_test[i]
        yPredict = clsY[i][0]
        yProb = probY[i][0]
        yy = abs(yProb - 0.5)
        strTF = 'False'
        if int(yReal) == int(yPredict):
            strTF = 'True'
        fn = fileName_Test[i]
        dataR.append((int(yReal), yPredict, strTF, yProb, yy, fn))
        nSum += 1
        if yReal == yPredict:
            nCorect += 1

    with open(fnoutput, 'w') as f:
        f.write('Real,Predict,Result,Prob,Confidence,FileName\n')
        for i in dataR:
            #print (i)
            f.write('%d,%d,%s,%f,%f,%s\n' \
                    % (i[0], i[1], i[2], i[3], i[4], i[5]))


def saveModelWeights(model,fn):
    numLayer = len(model.layers)

    output = open(fn, 'wb')
    pickle.dump(numLayer, output)
    lW = []
    for i in range(numLayer):
        lW.append(model.layers[i].get_weights())

    pickle.dump(lW, output)
    output.close()

def getDenseModel(input_features=85,l1=256,l2=256,l3=64,init='uniform',optimizer='adadelta'):
    model = Sequential()
    model.add(Dense(l1, input_dim=input_features, init=init, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(l2, init=init, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(l3, init=init, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model

def outputTopK_FM(filename,y_label,y_prob,y_test,fileName_test):
    with open(filename,'w') as f:
        colCorrect = []
        colConfidence = []
        predictions = y_prob
        predictLabel = y_label

        #f.write(u'预测值,实际值,正确,概率0,概率1,概率差,平面距离,距离绝对值\n')
        f.write('Predict,Real,Result,Confidence,FileName\n')
        for i in range(len(predictLabel)):
            yReal = y_test[i]
            fn = fileName_test[i]
            yPredict = predictLabel[i]
            yProb = predictions[i]
            conf = abs(yProb-0.5)
            sRight = 'False'
            if yPredict == yReal:
                sRight = 'True'
            colCorrect.append(sRight)
            colConfidence.append(conf)
            f.write('%d,%d,%s,%f,%s\n'\
                    %(yPredict, yReal, sRight, conf,fn))
            #print yPredict, yReal, yPredict == yReal, yProb, distance[i]
        #输出不同topN
        print 'Output TopN on',filename,'...'
        dfTotal = pandas.DataFrame()
        dfTotal['Correct'] = colCorrect
        dfTotal['Confidence'] = colConfidence
        dfTotal = dfTotal.sort_values('Confidence', ascending=False)
        numTotal = len(dfTotal.values)
        assert (numTotal == len(predictLabel))
        aryN = [10, 20, 50, 100, 200, 500, 1000]
        aryN.append(numTotal)
        for i in aryN:
            if i > numTotal:
                print 'can not extract top%d, it is too large'%(i)
            else:
                dftopN = dfTotal[:i]
                numTopNCorrect = len(dftopN[dftopN['Correct']=='True'].values)
                ratio = float(numTopNCorrect)/i
                print 'Top %d,%d,%.3f'%(i,numTopNCorrect,ratio)


def outputTopK(filename,c,X_test,y_test,fileName_test,hasDecionFunc=True):
    with open(filename,'w') as f:
        colCorrect = []
        colConfidence = []
        predictions = c.predict_proba(X_test)
        predictLabel = c.predict(X_test)
        if hasDecionFunc == True:
            distance = c.decision_function(X_test)
        else:
            distance = predictions
        #f.write(u'预测值,实际值,正确,概率0,概率1,概率差,平面距离,距离绝对值\n')
        f.write('Predict,Real,Result,Prob0,Proba,AbsProb,Dis,Confidence,FileName\n')
        for i in range(len(predictLabel)):
            yReal = y_test[i]
            fn = fileName_test[i]
            yPredict = predictLabel[i]
            yProb = predictions[i]
            absProb = abs(yProb[0]-yProb[1])
            if hasDecionFunc == True:
                dis = distance[i]
            else:
                dis = distance[i][0] - distance[i][1]
            absDis = abs(dis)
            sRight = 'False'
            if yPredict == yReal:
                sRight = 'True'
            colCorrect.append(sRight)
            colConfidence.append(absDis)
            f.write('%d,%d,%s,%f,%f,%f,%f,%f,%s\n'\
                    %(yPredict, yReal, sRight, yProb[0], yProb[1], absProb,dis,absDis,fn))
            #print yPredict, yReal, yPredict == yReal, yProb, distance[i]
        #输出不同topN
        print 'Output TopN on',filename,'...'
        dfTotal = pandas.DataFrame()
        dfTotal['Correct'] = colCorrect
        dfTotal['Confidence'] = colConfidence
        dfTotal = dfTotal.sort_values('Confidence', ascending=False)
        numTotal = len(dfTotal.values)
        assert (numTotal == len(predictLabel))
        aryN = [10, 20, 50, 100, 200, 500, 1000]
        aryN = [ 30,60, 100, 200, 500]
        aryN.append(numTotal)
        for i in aryN:
            if i > numTotal:
                print 'can not extract top%d, it is too large'%(i)
            else:
                dftopN = dfTotal[:i]
                numTopNCorrect = len(dftopN[dftopN['Correct']=='True'].values)
                ratio = float(numTopNCorrect)/i
                print 'Top %d,%d,%.3f'%(i,numTopNCorrect,ratio)

def addError(ary,trainRatio,errorRatio,fileNameAry):
    eAry=ary
    listErrorFileName=[]
    num_training = int(trainRatio * ary.shape[0])
    num_test = ary.shape[0] - num_training
    print num_training,u'个训练样本'
    #随机选择error_ratio比例的样本，调换其0/1值
    num_error=int(errorRatio*num_test)
    sidx = np.random.permutation(num_test)  # 随机数下标
    # print len(data["Label"].values),sidx
    for i in sidx[:num_error]:
        rV=ary[i + num_training]
        eAry[i + num_training]=1-rV
        listErrorFileName.append(fileNameAry[i + num_training])
    return eAry,listErrorFileName

def BinArray(ary):
    return ary
    for i in range(len(ary)):
        if ary[i]>0:
            ary[i]=1
        else:
            ary[i]=0
    return ary

def addStockPricePre25Features(X,dfM):
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
        #,'TotalTurnover'
        #, 'TotalTurnoverPre1'
        #, 'TotalTurnoverPre2'
        #, 'TotalTurnoverPre2'
        #,'TotalVolumeTraded'
        #, 'TotalVolumeTradedPre1'
        #, 'TotalVolumeTradedPre2'
        #, 'TotalVolumeTradedPre3'
    ]
    for f in listStockTrendFeatures:
        X = np.column_stack((X, BinArray(dfM[f].values)))

    return X

def addSAConfidenceFeatures(X,dfM):
    listSAConfidenceFeatures = [
        'SAConfi1','SAConfi2','SAConfi3','SAConfi4'
    ]
    for f in listSAConfidenceFeatures:
        X = np.column_stack((X, BinArray(dfM[f].values)))
    return X

def addStockPopSentiReliFeatures(X,dfM):
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
    for f in listStockPopSentiReliFeatures:
        X = np.column_stack((X, BinArray(dfM[f].values)))
    return X

def addARMAFeatures(X,dfM):
    X = np.column_stack((X, dfM['ReturnPredict'].values))
    X = np.column_stack((X, dfM['ReturnConfidence'].values))
    return X



def addPersonFeatures(X,dfM):
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
    for f in listPersonFeatures:
        dfMNoNan = dfM[f]
        #if f == 'SentiConfidence':
        #    dfMNoNan = dfMNoNan.fillna(1.)
        #elif f == 'SentiInconsistence':
        #    dfMNoNan = dfMNoNan.fillna(0.)
        X = np.column_stack((X, BinArray(dfMNoNan.values)))
    return X

def addPersonSectorFeatures(X,dfM):
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
    for f in listPersonSectorFeatures:
        X = np.column_stack((X, BinArray(dfM[f].values)))
    return X

def splitTrainTest(XData,YData,fileNameAry,trainRatio,numTrain=0):
    num_training = int(trainRatio * XData.shape[0])
    if numTrain > 0:
        num_training = numTrain
    num_test = XData.shape[0] - num_training

    print num_training, u'个训练数据'
    # 用随机分配的数据集进行训练与测试  (该方法得到的是ndarray，处理耗时，下面的方法用的是稀疏矩阵，处理块)

    # XArray = X.toarray()
    # X_trainML, X_testML, y_trainML, y_testML = train_test_split(
    # XArray, data["Label"].values, test_size=0.1, random_state=108)

    sidx = np.random.permutation(XData.shape[0])  # 随机数下标
    #sidx=range(XData.shape[0])

    rtrainidx = np.random.permutation(num_training)
    rtestidx = np.random.permutation(num_test)

    XTrain = XData[:num_training]
    XTest = XData[num_training:]
    YTrain = YData[:num_training]
    YTest = YData[num_training:]
    FNTrain = fileNameAry[:num_training]
    FNTest = fileNameAry[num_training:]

    X_train = XTrain[rtrainidx]
    X_test = XTest[rtestidx]
    y_train = YTrain[rtrainidx]
    y_test = YTest[rtestidx]
    fn_train = FNTrain[rtrainidx]
    fn_test = FNTest[rtestidx]



    #X_train = XData[sidx[:num_training]]
    #X_test = XData[sidx[num_training:]]
    #y_train = YData[sidx[:num_training]]
    #y_test = YData[sidx[num_training:]]

    print X_train.shape,y_train.shape
    print X_test.shape, y_test.shape
    return X_train,X_test,y_train,y_test,fn_train,fn_test

def saveFilteredCSV(fileNameOut,fileNameIn,model,fe):
    arySelect = model.get_support(True)
    setSelect = set()
    aryDics = fe.get_feature_names()

    for idx29 in arySelect:
        setSelect.add(aryDics[idx29])

    setFiltered = set(aryDics) - setSelect
    #for i in setFiltered:
    #    print i


    with open(fileNameIn,'r') as fIn:
        with open(fileNameOut,'w') as fOut:
            inti=0
            for oneLine in fIn.readlines():
                inti=inti+1
                if inti==1:
                    fOut.write(oneLine)
                else:
                    oneLine = oneLine.strip().decode('utf-8')
                    list1 = oneLine.split(',')
                    list2 = list1[3].split(' ')
                    listNew = []
                    for w in list2:
                        bFilter = False
                        for wS in setFiltered:
                            # print type(w.encode('utf-8')),type(oneLine)
                            if wS == w:
                                # print wS,w
                                bFilter = True
                        if bFilter == False:
                            listNew.append(w)
                    if listNew != []:
                        fOut.write('%s,%s,%s,%s\n'%(list1[0].encode('utf-8'),list1[1].encode('utf-8'),list1[2].encode('utf-8'),u' '.join(listNew).encode('utf-8')))

def wordvector(fileName):
    dicVector = {}
    with open(fileName, 'r') as f:
        for oneLine in f.readlines():
            oneLine = oneLine.strip()
            w = oneLine.split(' ')[0]# str类型

            v = oneLine.split(' ')[1].split(',')
            #print type(w), v
            dicVector[w] = [float(s) for s in v]
        return dicVector

def tfidfVector(tfidfArray,tfidfNames):
    tfidfArray=tfidfArray.toarray()
    dicWordVector = wordvector("data\\VectorDics400.txt")#
    adjustArray=[]

    for oneDoc in tfidfArray:
        oneDoc = list(oneDoc)

        idxWord = 0  #记录单词编号，与实际单词对应
        vDoc=[0 for i in range(len(dicWordVector.values()[0]))]

        for i in oneDoc:#i是一个单词的tfidf权重
            if i==0:
                pass
            else:
                #i与dicWordVector[tfidfNames[idxWord]]相乘
                w=tfidfNames[idxWord].encode('utf-8')

                if w in dicWordVector:
                    vw=dicWordVector[w]
                    vDoc = list(np.array(vDoc) + i * np.array(vw))
            idxWord = idxWord + 1
        adjustArray.append(vDoc)

    return preprocessing.normalize(np.array(adjustArray), norm='l2')

fnIn = 'data4classification_OpenClose_oldLabel.csv'
fnOutLR='lr-big-OpenClose-noerror-0117-addFeatureARMA.csv'
fnOutSVMRBF='svm-big-rbf-OpenClose-noerror-0117-addFeatureARMA.csv'
#fnOutSVMLinear='svm-big-linear-OpenClose.csv'
fnOutkNN='knn-big-OpenClose-noerror-0117-addFeatureARMA.csv'
fnOutNB='nb-big-OpenClose-noerror-0117-addFeatureARMA.csv'
fnOutFM='50-100-fm-big-OpenClose-noerror-0118.csv'
fm_matrix = 20
fm_iter = 2000

def do_things():
    #fileLSTMIput = open('sequenceData/openclose/StockDataset_8_Total_person_noerr_seq.pkl', 'rb')
    #X_trainTmp, y_trainTmp, X_testTmp, y_testTmp, fileName_TrainLSTM, fileName_TestLSTM = pickle.load(fileLSTMIput)
    #print 'text'
    #fileName_TrainLSTM=list(set(fileName_TrainLSTM))
    #fileNameTotal = list(fileName_TrainLSTM)
    #fileNameTotal[-1:-1] = list(fileName_TestLSTM)
    #print len(fileName_TrainLSTM),len(fileName_TestLSTM),len(fileNameTotal)
    #quit()
    dfM = pandas.read_csv(fnIn, encoding='utf-8')
    '''
    dicPerson = dict()
    for i in dfM.index:
        di = dfM.loc[i]
        p = di['PersonURL']
        if p not in dicPerson:
            dicPerson[p]=dict()
            dicPerson[p]['Reliability']=0
            dicPerson[p]['Unreliability'] = 0
        reli = di['Reliability']
        if reli == 1:
            dicPerson[p]['Reliability'] += 1
        else:
            dicPerson[p]['Unreliability'] += 1
    for p in dicPerson.keys():
        numRe = dicPerson[p]['Reliability']
        numUnr = dicPerson[p]['Unreliability']
        ratio = float(numRe)/(numRe + numUnr)
        print p,numRe,numUnr,ratio
    quit()
    numUse = 20455
    numTotal = len(dfM.values)
    sidxTotal = np.random.permutation(numTotal)  # 随机数下标

    dfM['sort'] = sidxTotal
    dfM=dfM.sort_values('sort')
    dfM = dfM[:numUse]
    '''


    #dfM = dfM[dfM['FileName'].isin(fileNameTotal)]
    dfSAConfidenceAll = pandas.read_csv('data\\total\\stopword\\svm-linear-confidence-All.csv', encoding='utf-8')
    dfSAConfidence = pandas.DataFrame()
    dfSAConfidence['FileName'] = dfSAConfidenceAll['FileName']
    dfSAConfidence['SAConfi1'] = dfSAConfidenceAll['SAConfi1']
    dfSAConfidence['SAConfi2'] = dfSAConfidenceAll['SAConfi2']
    dfSAConfidence['SAConfi3'] = dfSAConfidenceAll['SAConfi3']
    dfSAConfidence['SAConfi4'] = dfSAConfidenceAll['SAConfi4']
    dfM = pandas.merge(dfM,dfSAConfidence,how='left',on='FileName')

    dfARMATotal = pandas.read_csv('TotalreturnResult_1.csv', encoding='utf-8')
    dfARMA = pandas.DataFrame()
    dfARMA['FileName'] = dfARMATotal['FileName']
    colPredictARMA = []
    colConfidenceARMA = []
    for i in dfARMATotal.index:
        vPre = dfARMATotal.loc[i, 'ReturnPredict']
        vErr = dfARMATotal.loc[i, 'ReturnErr']
        #if vPre > 0:
        #    colPredictARMA.append(1)
        #else:
        #    colPredictARMA.append(0)
        colPredictARMA.append(vPre)
        colConfidenceARMA.append(1.0 / vErr)
    dfARMA['ReturnPredict'] = colPredictARMA
    dfARMA['ReturnConfidence'] = colConfidenceARMA

    # 将ARMA结果融合进特征
    dfM = pandas.merge(dfM, dfARMA, how='left', on='FileName')
    dfM=dfM.sort_values('Date')

    print len(dfM['Text'].values), u'个总文档'

    # convert to feature vector
    # feature_extraction = CountVectorizer(min_df=1, binary=True)
    feature_extraction = TfidfVectorizer()

    docD = dfM['Text'].values

    XTextFeature = feature_extraction.fit_transform(docD)

    tfidfNames = feature_extraction.get_feature_names()
    print len(tfidfNames), u'个唯一单词'

    # Xold=tfidfVector(Xold,tfidfNames) #考虑词向量

    # Xold = preprocessing.scale(Xold.toarray())
    # print type(Xold.toarray())
    # 增加靠谱还是不靠谱的特征，以及股票前n天的涨跌特征
    # quit()


    # saveFilteredCSV('featureselected.csv',fileIn,model,feature_extraction) #输出抽取特征之后的结果

    fileNameAry = dfM['FileName'].values  # 记录了数据集中的文件名，用于最后评估是追踪结果
    trainRatio = 0.9

    # X=dfM["Label"].values
    # 在情感态度的基础之上，对error_ratio的样本添加错误信息，模拟情感分类的精度
    # 这里是给测试集加的误差，训练集没有加
    listErrorFileName = []

    #X = XTextFeature
    #listErrorFileName=[]
    X, listErrorFileName = addError(dfM["Label"].values, trainRatio, errorRatio, fileNameAry)
    with open('errorFileName-baselinebig.txt','w') as f12:
        print 'output file names with incorrect sentiment:'
        for i in listErrorFileName:
            #print i
            f12.write('%s\n'%(i))

    X = addStockPricePre25Features(X,dfM)


    #print 'No SA No Person No ARMA'

    X = addPersonFeatures(X,dfM)  # 添加股评员特征

    X = addStockPopSentiReliFeatures(X,dfM)  # 添加股票热度情感等特征
    #X = X[:, 1:]
    #X = addSAConfidenceFeatures(X,dfM) #添加情感分析置信度特征
    X = preprocessing.scale(X)  # 股价涨跌趋势特征必须要scale
    # X = np.concatenate((X,XTextFeature.toarray()),axis=1)#把tfidf加到X中，跳过scale
    # X=XTextFeature#.toarray()preprocessing
    # X=csr_matrix(X)
    X = addARMAFeatures(X, dfM)
    #X = addPersonSectorFeatures(X,dfM)
    #X=X[:,1:]
    #X=np.column_stack((X, dfM['Label'].values))




    Yold = dfM['Reliability'].values

    # 可选：特征选择
    clf1 = LinearSVC(C=0.1,penalty='l1',dual=False).fit(X,Yold)
    clf2 = LogisticRegression(penalty="l1", C=0.1).fit(X,Yold)
    model=SelectFromModel(clf1,prefit=True)
    X=model.transform(X)  #当‘l2’时，剩下N个特征，精度和AUC达到最佳
    #X=SelectKBest(f_classif,60).fit_transform(X,Yold) #2711为文档频率大于等于4的特征数目
    print X.shape[1], u'个特征被SelectFromModel筛选出来'

    # X=Xold
    # quit()

    # split into training- and test set

    X_train, X_test, y_train, y_test, fileName_train, fileName_test = \
        splitTrainTest(X, Yold, fileNameAry, trainRatio)

    stime = time.clock()


    '''
    model = KerasClassifier(build_fn=getDenseModel, verbose=0)
    epochs = [100, 200, 500]
    batch_size = [64, 128, 256, 512]
    param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(X_train, y_train)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    '''
    '''
    modelDense=getDenseModel(X_train.shape[1])
    modelDense.fit(X_train, y_train,
              nb_epoch=2000,
              batch_size=512,validation_data=(X_test, y_test))
    score, acc = modelDense.evaluate(X_test, y_test,
                                batch_size=512)
    print('Test score:', score)
    print('Test accuracy:', acc)

    saveModelWeights(modelDense,'dense_weight-split.pkl')
    save_TestResults(modelDense,'dense-big-split.csv',X_test,y_test,fileName_test)
    '''
    # train classifier
    '''
    fm = pylibfm.FM(num_factors=fm_matrix, num_iter=fm_iter, verbose=True, task="classification", initial_learning_rate=0.0001,
                    learning_rate_schedule="optimal")
    from sklearn.feature_extraction import DictVectorizer
    v = DictVectorizer()
    X_dicvec=np.vstack((X_train, X_test))
    X_dicvec = vec2dicvec(X_dicvec)

    v.fit(X_dicvec)
    X_train = v.transform(vec2dicvec(X_train))
    X_test = v.transform(vec2dicvec(X_test))

    fm.fit(X_train, y_train)
    print 'finished FM fit'

    y_pre =  fm.predict(X_test)

    y_pre2,y_correct,yconfi = [],[],[]
    for i in range(len(y_pre)):
        yconfi.append(abs(0.5 - y_pre[i]))
        if y_pre[i] > 0.5:
            y_pre2.append(1)
            if y_test[i]==1:
                y_correct.append(True)
            else:
                y_correct.append(False)
        else:
            y_pre2.append(0)
            if y_test[i]==0:
                y_correct.append(True)
            else:
                y_correct.append(False)
    print 'real,preeict,correct,confidence,filename'
    for i in range(len(y_test)):
        print y_test[i],y_pre2[i],y_correct[i],yconfi[i],fileName_test[i]
    outputTopK_FM(fnOutFM, y_pre2,y_pre, y_test, fileName_test)
    quit()
    '''



    # LR
    '''
    t1 = time.clock()
    lrclf = LogisticRegression()
    lrclf.fit(X_train, y_train)
    lr_predict = lrclf.predict_proba(X_test)
    outputTopK(fnOutLR, lrclf, X_test, y_test, fileName_test)
    print u'LR的精度为', lrclf.score(X_test, y_test)
    print('ROC-AUC yields ' + str(roc_auc_score(y_test, lr_predict[:, 1])))
    t2 = time.clock()
    print 'LR elasped time:%s s' % (t2 - t1)

    # KNN
    knnclf = KNeighborsClassifier()  # default with k=5
    knnclf.fit(X_train, y_train)
    knn_predict = knnclf.predict_proba(X_test)

    print u'kNN的精度为', knnclf.score(X_test, y_test)
    print('KNN ' + str(roc_auc_score(y_test, knn_predict[:, 1])))
    outputTopK(fnOutkNN, knnclf, X_test, y_test, fileName_test,False)

    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_non_nega = min_max_scaler.fit_transform(X_train)
    X_test_non_nega = min_max_scaler.fit_transform(X_test)
    mnbclf = MultinomialNB()
    mnbclf.fit(X_train_non_nega, y_train)
    mnb_predict = mnbclf.predict_proba(X_test_non_nega)

    print u'NB的精度为', mnbclf.score(X_test_non_nega, y_test)
    print('Multinomial Naive Bayesian ' + str(roc_auc_score(y_test, mnb_predict[:, 1])))
    outputTopK(fnOutNB, mnbclf, X_test_non_nega, y_test, fileName_test,False)

    '''

    # SVM-RBF
    t1 = time.clock()
    clfRBF = SVC(C=0.7, probability=True, kernel='rbf')
    clfRBF.fit(X_train, y_train)

    # predict and evaluate predictions
    predictions = clfRBF.predict_proba(X_test)
    outputTopK(fnOutSVMRBF, clfRBF, X_test, y_test, fileName_test)
    print u'SVM rbf的精度为', clfRBF.score(X_test, y_test)
    print('ROC-AUC yields ' + str(roc_auc_score(y_test, predictions[:, 1])))
    t2 = time.clock()
    print 'SVM-RBF elasped time:%s s' % (t2 - t1)

    '''
    #eclf = VotingClassifier(estimators=[('svmrbf', clfRBF),('nb',mnbclf)], voting='soft', weights=[2, 1])
    eclf = VotingClassifier(estimators=[('svmrbf', clfRBF), ('lr', lrclf),('knn',knnclf)], voting='soft',weights=[3,1,1])
    eclf.fit(X_train, y_train)

    # predict and evaluate predictions
    predictions = eclf.predict_proba(X_test)

    print u'eclf rbf的精度为', eclf.score(X_test, y_test)
    print('ROC-AUC yields ' + str(roc_auc_score(y_test, predictions[:, 1])))
    outputTopK('ensemble.csv', eclf, X_test, y_test, fileName_test,False)
    '''

    #SVM-Linear
    '''
    t1 = time.clock()
    clf = SVC(C=0.7, probability=True, kernel='linear')
    clf.fit(X_train, y_train)

    # predict and evaluate predictions
    predictions = clf.predict_proba(X_test)
    outputTopK(fnOutSVMLinear, clf, X_test, y_test, fileName_test)
    print u'SVM linear的精度为', clf.score(X_test, y_test)
    print('ROC-AUC yields ' + str(roc_auc_score(y_test, predictions[:, 1])))
    t2 = time.clock()
    print 'SVM-Linear elasped time:%s s' % (t2 - t1)

    #print 'price stock'
    quit()
    '''



if __name__ == "__main__":

    do_things()