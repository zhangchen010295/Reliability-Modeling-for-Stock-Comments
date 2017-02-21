# dfsdfsds encoding: utf-8
#按照Date,Label,Comments的csv格式来构建输入语料
import cPickle
import gzip
import os
import sys
import datetime
import nltk
import itertools
import numpy as np
import time

import numpy
import jieba
import extractStockComments
from gensim import corpora, models, similarities

tStart = time.clock()
listDir=[]
listCSVFile = []
#for root,subdirs,files in os.walk("tempAnnotation"):
#    for onefile in files:
#        listCSVFile.append(os.path.join(root, onefile))
for root,subdirs,files in os.walk("already"):
    for onedir in subdirs:
        listDir.append(os.path.join(root, onedir))
for oneRoot in listDir:
    for root, subdirs, files in os.walk(oneRoot):
        for oneFile in files:
            listCSVFile.append(os.path.join(root, oneFile))

totalCorpus = []
totalIO = []
totalUniqueCorpus = []
totalDate = []
totalLable = []
totalFileName = []

numDoc = 0
numT=0
for oneFile in listCSVFile:
    numT=numT+1
    if numT % 100 == 0:
        print u'Processing',numT
        #break
    fileName,sAnsDate, label, lcutstop = extractStockComments.extractStockComments(oneFile)
    #print sAnsDate
    #print label
    if lcutstop != [] and lcutstop != '' and lcutstop != None and len(lcutstop)>0:
        numDoc=numDoc+1
        totalIO.append(lcutstop)
        totalCorpus.extend(lcutstop)
        totalUniqueCorpus.extend(list(set(lcutstop)))
        totalDate.append(sAnsDate)
        totalLable.append(label)
        totalFileName.append(fileName)
print numT

#输出原始股评文件csv
with open('comments_raw.csv', 'w') as f:
    #日期,股评员URL,股票代号,
    # 股票前五天价格，股票前四天价格，股票前三天价格，股票前两天价格，股票前一天价格，
    # 股票当天价格,股票后一天价格，股票后两天价格，股票后三天价格，股票后四天价格，股票后五天价格，
    # 态度涨跌,短期是否靠谱,长期是否靠谱,短/长期,,,,,股评
    #f.write('Date,PersonURL,StockID,LabelSA,LabelShortCorrect,LabelLongCorrect,LabelRange,Text\n')
    f.write('FileName,Date,Label,Text\n')
    for i in range(len(totalIO)):
        fn= totalFileName[i]
        fn=os.path.basename(fn)
        d = totalDate[i]
        l = totalLable[i]
        t = totalIO[i]
        text=' '.join(t)
        if ',' in text:
            text=text.replace(',','')
        f.write('%s,%s,%s,%s\n' % (fn,d, l, text) )
print "Extract %d documents." % (numDoc)
#quit()
word_freq = nltk.FreqDist(totalUniqueCorpus)
print "Found %d unique words tokens." % len(word_freq.items())

totalV = word_freq.most_common(len(word_freq.items()))
MINDF = 4
idxV = 0
for iw in totalV:
    if iw[1]<MINDF:
        break
    idxV = idxV + 1
vocab = word_freq.most_common(idxV) #前idxV个单词的文档频率大于等于MINDF
for i in vocab:
    print i[0],i[1]

index_to_word = [x[0] for x in vocab]

word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(totalIO):
    totalIO[i] = [w if w in word_to_index else '' for w in sent]

for onedoc in totalIO:
    if '' in onedoc:
        onedoc.remove('')

#输出按文档频率筛选单词后的股评文件csv
with open('comments_filter.csv', 'w') as f:
    f.write('FileName,Date,Label,Text\n')
    for i in range(len(totalIO)):
        fn = totalFileName[i]
        fn = os.path.basename(fn)
        d = totalDate[i]
        l = totalLable[i]
        t = totalIO[i]
        if t != [] and t != '' and t != None and len(t)>0 and t[0] !='' and t[0] != ' ':
            sss=' '.join(t)
            if ',' in sss:
                sss = sss.replace(',', '')
            f.write('%s,%s,%s,%s\n' % (fn,d, l, sss) )

tEnd = time.clock()
print u'耗时：',tEnd-tStart