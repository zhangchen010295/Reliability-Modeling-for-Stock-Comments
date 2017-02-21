# dfsdfsds encoding: utf-8
# This script concatenates all news headlines of a day into one and uses the tf-idf scheme to extract a feature vector.
# An SVM with rbf kernel without optimization of hyperparameters is used as a classifier.
#该代码用于对情感分析精度的cross_validation,概率值钱topN%的精度

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
import time
from frameworks.SelfLearning import *
from frameworks.CPLELearning import CPLELearningModel
from sklearn.cross_validation import cross_val_score


#SEED = 4843
#np.random.seed(SEED)

def outputTopK(filename,c,X_test,y_test,fileName_test):
    with open(filename,'w') as f:
        predictions = c.predict_proba(X_test)
        predictLabel = c.predict(X_test)
        distance = c.decision_function(X_test)
        #f.write(u'预测值,实际值,正确,概率0,概率1,概率差,平面距离,距离绝对值\n')
        f.write('Predict,Real,Result,Prob0,Prob1,AbsProb,Dis,Confidence,FileName\n')
        for i in range(len(predictLabel)):
            yReal = y_test[i]
            fn = fileName_test[i]
            yPredict = predictLabel[i]
            yProb = predictions[i]
            absProb = abs(yProb[0]-yProb[1])
            dis = distance[i]
            absDis = abs(dis)
            sRight = 'False'
            if yPredict == yReal:
                sRight = 'True'
            f.write('%d,%d,%s,%f,%f,%f,%f,%f,%s\n'\
                    %(yPredict, yReal, sRight, yProb[0], yProb[1], absProb,dis,absDis,fn))
            #print yPredict, yReal, yPredict == yReal, yProb, distance[i]
    with open(filename+'stat.txt','w') as f1:
        dfData = pandas.read_csv(filename,encoding='utf-8')
        dfData = dfData.sort_values('Confidence',ascending=False)
        lenTotal = len(dfData.values)
        f1.write('Ratio,TopN,CorrectTopN,CorrectRatio\n')
        for i in range(1,11):
            ratio = float(i)/10.0
            topN = int(ratio * lenTotal)
            dfTop = dfData[:topN]
            dfTopCorrect = dfTop[dfTop['Result']==True]
            correctTopN = len(dfTopCorrect.values)
            accu = float(correctTopN)/topN
            f1.write('%.1f,%d,%d,%f\n'% (ratio, topN, correctTopN, accu))

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

# read data
fileIn = "data\\total\\stopword\\TotalComments13111.csv"
fileLable = 'data\\total\\stopword\\Union.csv'
dataTotal = pandas.read_csv(fileIn, encoding='utf-8')
#data=data.sort_values('Date')  这里注释掉，是因为数据本身是两个数据叠加而成的，用上半训练

dfLabel = pandas.read_csv(fileLable,encoding='utf-8')
dfLabelValid=dfLabel[dfLabel['Label'] != -1]
#dfLabel=dfLabel[dfLabel['OldOrNew'] == 'old']
dfLabelInValid=dfLabel[dfLabel['Label'] == -1]
listValidFile = list(dfLabelValid['FileName'].values)
listInValidFile = list(dfLabelInValid['FileName'].values)
print len(dfLabel.values),len(listValidFile),len(listInValidFile)

'''
dfConfidenceAll = pandas.read_csv('data\\total\\stopword\\svm-linear-confidence-All.csv',encoding='utf-8')
colsConfidence1 = []
colsConfidence2 = []
colsConfidence3 = []
colsConfidence4 = []
for i in dfConfidenceAll.index:
    conf = dfConfidenceAll.loc[i,'Confidence']
    if conf > 2:
        colsConfidence1.append(1)
        colsConfidence2.append(0)
        colsConfidence3.append(0)
        colsConfidence4.append(0)
    elif conf > 1:
        colsConfidence1.append(0)
        colsConfidence2.append(1)
        colsConfidence3.append(0)
        colsConfidence4.append(0)
    elif conf > 0.5:
        colsConfidence1.append(0)
        colsConfidence2.append(0)
        colsConfidence3.append(1)
        colsConfidence4.append(0)
    else:
        colsConfidence1.append(0)
        colsConfidence2.append(0)
        colsConfidence3.append(0)
        colsConfidence4.append(1)
dfConfidenceAll['SAConfi1']=colsConfidence1
dfConfidenceAll['SAConfi2']=colsConfidence2
dfConfidenceAll['SAConfi3']=colsConfidence3
dfConfidenceAll['SAConfi4']=colsConfidence4
dfConfidenceAll.to_csv('data\\total\\stopword\\svm-linear-confidence-All.csv',encoding='utf-8',index=False)
quit()
'''

data = dataTotal[dataTotal['FileName'].isin(listValidFile)]
#dataTest = dataTotal[dataTotal['FileName'].isin(listInValidFile)]
#data.to_csv('predicted.csv',encoding='utf-8',index=False)
#quit()

numValidData = len(data['Text'].values)
print numValidData,u'个带有效标记的文档'

data = data.drop('Label', axis=1)
dfL = pandas.DataFrame()
dfL['FileName'] = dfLabel['FileName']
#将人工标记的label替换SVM自动分类的
dfL['Label'] = dfLabel['Label']
data = pandas.merge(data,dfL,how='left',on='FileName')
data = data.sort_values('Date')  # 按照日期进行排序
#print data.columns

#data.to_csv('28101SA.csv',encoding='utf-8',index=False)
#quit()

# convert to feature vector
#feature_extraction = CountVectorizer(min_df=1, binary=True)
feature_extraction = TfidfVectorizer()

Xold = feature_extraction.fit_transform(data['Text'].values)
print len(feature_extraction.get_feature_names()),u'个唯一单词'
#Xold = preprocessing.scale(Xold.toarray())
#print type(Xold.toarray())
#增加靠谱还是不靠谱的特征，以及股票前n天的涨跌特征
#quit()
Yold=data["Label"].values

clf1 = LinearSVC(C=0.7,penalty='l2',dual=False).fit(Xold,Yold)

model=SelectFromModel(clf1,prefit=True)
X=model.transform(Xold)  #当‘l2’时，剩下N个特征，精度和AUC达到最佳
#saveFilteredCSV('featureselected.csv',fileIn,model,feature_extraction) #输出抽取特征之后的结果

#X=Xold  #不对特征进行筛选
print X.shape[1],u'个特征被SelectFromModel筛选出来'
#X=SelectKBest(chi2,2711).fit_transform(Xold,Yold) #2711为文档频率大于等于4的特征数目
#X=X[:13000]
#Yold=Yold[:13000]



# split into training- and test set
#TRAINING_END = date(2015, 12, 22)
#num_training = len(data[pandas.to_datetime(data["Date"]) <= TRAINING_END])


stime = time.clock()

#用随机分配的数据集进行训练与测试  (该方法得到的是ndarray，处理耗时，下面的方法用的是稀疏矩阵，处理块)

#XArray = X.toarray()
#X_trainML, X_testML, y_trainML, y_testML = train_test_split(
#XArray, data["Label"].values, test_size=0.1, random_state=108)



sidx = np.random.permutation(numValidData) #随机数下标

trainRatio=0.9
num_training = int(numValidData*trainRatio)

print num_training,u'个训练数据'

X_train = X[sidx[:num_training]]
X_test = X[sidx[num_training:]]
y_train = Yold[sidx[:num_training]]
y_test = Yold[sidx[num_training:]]
#X_train = X[:num_training]
#X_test = X[num_training:]
#y_train = Yold[:num_training]
#y_test = Yold[num_training:]
#y_unlabel = np.array(Yold)  #一定要重新新建一个数组，不然Yold的值会被覆盖
#y_unlabel[num_training:]=-1

#X_train, X_test, y_train, y_test = train_test_split(X, Yold, train_size=num_training)


#print X_trainML.shape,X_testML.shape,y_trainML.shape,y_testML.shape
#quit()

# train classifier
# LR
lrclf = LogisticRegression()
lrclf.fit(X_train, y_train)
lr_predict = lrclf.predict_proba(X_test)
lFileName = data['FileName'].values
fn_test = list(lFileName[sidx[num_training:]])
print len(fn_test)
outputTopK('lr-confidence.csv',lrclf,X_test,y_test,fn_test)

cvNum = 10

scores = cross_val_score(lrclf, X, Yold, cv=cvNum)
print 'lr-',cvNum,' fold valication',scores.max(),scores.min(),scores.mean()

print u'LR的精度为',lrclf.score(X_test,y_test)
print('LogisticRegression ' + str(roc_auc_score(y_test, lr_predict[:, 1])))


clf = SVC(C=0.7,probability=True, kernel='linear')
clf.fit(X_train, y_train)
# predict and evaluate predictions
predictions = clf.predict_proba(X_test)

print u'SVM-Linear的精度为',clf.score(X_test,y_test)
print('ROC-AUC yields ' + str(roc_auc_score(y_test, predictions[:, 1])))
outputTopK('svm-linear-confidence.csv',clf,X_test,y_test,fn_test)
scores = cross_val_score(clf, X, Yold, cv=cvNum)
print 'svm-linear-',cvNum,' fold valication',scores.max(),scores.min(),scores.mean()
quit()

rbfclf = SVC(C=1.0,probability=True, kernel='rbf')
rbfclf.fit(X_train, y_train)
# predict and evaluate predictions
predictions = rbfclf.predict_proba(X_test)

print u'SVM-RBF的精度为',rbfclf.score(X_test,y_test)
print('ROC-AUC yields ' + str(roc_auc_score(y_test, predictions[:, 1])))

quit()

ssmodel = SelfLearningModel(clf)
ssmodel.fit(X.toarray(), y_unlabel)
predictions = ssmodel.predict_proba(X_test.toarray())
yPredict=ssmodel.predict(X_test.toarray())
data['Label'][num_training:]=yPredict
data.to_csv('newData_SelfLearningModel.csv',encoding='utf-8',index=False)
print u'半监督SL SVM的精度为',ssmodel.score(X_test.toarray(),y_test)
#print(u'半监督SVMs ROC-AUC yields ' + str(roc_auc_score(y_test, predictions[:, 1])))
#print "CPLE semi-supervised log.reg. score", ssmodel.score(X.toarray(), Yold)

cplemodel = CPLELearningModel(clf)
cplemodel.fit(X.toarray(), y_unlabel)
predictions = cplemodel.predict_proba(X_test.toarray())
yPredict=cplemodel.predict(X_test.toarray())
data['Label'][num_training:]=yPredict
data.to_csv('newData_cplemodel.csv',encoding='utf-8',index=False)
print u'半监督cplemodel SVM的精度为',cplemodel.score(X_test.toarray(),y_test)

etime = time.clock()
print 'elasped time:%s s'%(etime-stime)

quit()
# LR
lrclf = LogisticRegression()
lrclf.fit(X_train, y_train)
lr_predict = lrclf.predict_proba(X_test)

print u'LR的精度为',lrclf.score(X_test,y_test)
print('LogisticRegression ' + str(roc_auc_score(y_test, lr_predict[:, 1])))



# KNN
knnclf = KNeighborsClassifier()  # default with k=5
knnclf.fit(X_train, y_train)
knn_predict = knnclf.predict_proba(X_test);

print u'kNN的精度为',knnclf.score(X_test,y_test)
print('KNN ' + str(roc_auc_score(y_test, knn_predict[:, 1])))

quit()
# create the Multinomial Naive Bayesian Classifier
mnbclf = MultinomialNB(alpha=0.01)
mnbclf.fit(X_train, y_train);
mnb_predict = mnbclf.predict_proba(X_test);

ppppp = mnbclf.predict(X_test)
idd = 0.0
for i in range(len(ppppp)):
    if ppppp[i]==y_test[i]:
        idd = idd + 1
print u'NB的精度为',idd/len(ppppp)
print('Multinomial Naive Bayesian ' + str(roc_auc_score(y_test, mnb_predict[:, 1])))
