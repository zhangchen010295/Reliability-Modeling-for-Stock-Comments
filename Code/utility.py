# dfsdfsds encoding: utf-8
import pandas
import numpy as np
import datetime
from datetime import date

def addColumns(dfLeft,dfRight,colKey,colAry):
    dfData = pandas.DataFrame()
    dfData[colKey] = dfRight[colKey]
    for i in colAry:
        dfData[i] = dfRight[i]
    dfLeft = pandas.merge(dfLeft,dfData,how='left',on=colKey)
    return dfLeft

def load_csv(fn):
    return pandas.read_csv(fn,encoding='utf-8')
def save_csv(df,fn):
    df.to_csv(fn,encoding='utf-8',index=False)

def str2date(s):
    return datetime.datetime.strptime(s, "%Y-%m-%d").date()
def date2str(s):
    return s.strftime('%Y-%m-%d')

listOutputFeature=[
        'PersonURL',
        'StockID',
        'Date',
        'FileName',
        'Label',
        'Reliability',

        # 股价涨跌趋势特征
        'StockTrendPre1',
        'StockTrendPre2',
        'StockTrendPre3',
        'StockTrendPre4',
        'StockTrendPre5',
        'StockTrendPre6',
        'StockTrendPre7',
        'StockTrendPre8',
        'StockTrendPre9',
        'StockTrendPre10',
        'StockTrendPre11',
        'StockTrendPre12',
        'StockTrendPre13',
        'StockTrendPre14',
        'StockTrendPre15',
        'StockTrendPre16',
        'StockTrendPre17',
        'StockTrendPre18',
        'StockTrendPre19',
        'StockTrendPre20',
        'StockTrendPre21',
        'StockTrendPre22',
        'StockTrendPre23',
        'StockTrendPre24',
        'StockTrendPre25',

        # 股票特征
        'HisStockPopNum'
         ,'HisStockPopRatio'
        , '7DaysStockPopNum'
         ,'7DaysStockPopRatio'
        , 'CurDayNegNum'
        , 'CurDayPosNum'
        , '7DaysNegNum'
        , '7DaysPosNum'
        , '7DaysRight'
        , '7DaysWrong'
        , '7DaysRightNum'
        , '7DaysWrongNum'
         ,'7DaysRightRatio'
        , '7DaysProbNegWrong'
        , '7DaysProbNegRight'
        , '7DaysProbPosWrong'
        , '7DaysProbPosRight'

        # 股评员特征
        ,'PersonHisCommentCount'
        , 'PersonHisCommentCountPos'
        , 'PersonHisCommentCountNeg'
        , 'PersonHisCommentCountWrn'
        , 'PersonHisCommentCountRht'
        , 'PersonCurStockHisCommentCount'
        , 'PersonCurStockHisCommentCountPos'
        , 'PersonCurStockHisCommentCountNeg'
        , 'PersonCurStockHisCommentCountWrn'
        , 'PersonCurStockHisCommentCountRht'
        , 'PersonSectorHisCommentCount'
        , 'PersonSectorHisCommentCountPos'
        , 'PersonSectorHisCommentCountNeg'
        , 'PersonSectorHisCommentCountWrn'
        , 'PersonSectorHisCommentCountRht'
        , 'PersonSectorHisAmountRatio'
        , 'PersonSectorHisCorrectRatio'
        , 'PersonSectorHisWeight'
         , 'Person7DaysCommentCount',
        'Person7DaysCommentCountPos',
        'Person7DaysCommentCountNeg',
        'Person7DaysCommentCountWrn',
        'Person7DaysCommentCountRht',
        'PersonCurStock7DaysCommentCount',
        'PersonCurStock7DaysCommentCountPos',
        'PersonCurStock7DaysCommentCountNeg',
        'PersonCurStock7DaysCommentCountWrn',
        'PersonCurStock7DaysCommentCountRht'
        , 'Person30DaysCommentCount',
        'Person30DaysCommentCountPos',
        'Person30DaysCommentCountNeg',
        'Person30DaysCommentCountWrn',
        'Person30DaysCommentCountRht',
        'PersonCurStock30DaysCommentCount',
        'PersonCurStock30DaysCommentCountPos',
        'PersonCurStock30DaysCommentCountNeg',
        'PersonCurStock30DaysCommentCountWrn',
        'PersonCurStock30DaysCommentCountRht'
        , 'Person90DaysCommentCount',
        'Person90DaysCommentCountPos',
        'Person90DaysCommentCountNeg',
        'Person90DaysCommentCountWrn',
        'Person90DaysCommentCountRht',
        'PersonCurStock90DaysCommentCount',
        'PersonCurStock90DaysCommentCountPos',
        'PersonCurStock90DaysCommentCountNeg',
        'PersonCurStock90DaysCommentCountWrn',
        'PersonCurStock90DaysCommentCountRht',

        'PersonSector90DaysCommentCount',
        'PersonSector90DaysCommentCountPos',
        'PersonSector90DaysCommentCountNeg',
        'PersonSector90DaysCommentCountWrn',
        'PersonSector90DaysCommentCountRht',
        'PersonSector30DaysCommentCount',
        'PersonSector30DaysCommentCountPos',
        'PersonSector30DaysCommentCountNeg',
        'PersonSector30DaysCommentCountWrn',
        'PersonSector30DaysCommentCountRht',
        'PersonSector7DaysCommentCount',
        'PersonSector7DaysCommentCountPos',
        'PersonSector7DaysCommentCountNeg',
        'PersonSector7DaysCommentCountWrn',
        'PersonSector7DaysCommentCountRht',

        'PersonSector7DaysAmountRatio',
        'PersonSector7DaysCorrectRatio',
        'PersonSector7DaysWeight',
        'PersonSector30DaysAmountRatio',
        'PersonSector30DaysCorrectRatio',
        'PersonSector30DaysWeight',
        'PersonSector90DaysAmountRatio',
        'PersonSector90DaysCorrectRatio',
        'PersonSector90DaysWeight'

    ]

def getFullFeatures(dataFrame):
    d = pandas.DataFrame()

    for i in listOutputFeature:
        d[i] = dataFrame[i]

    return d

def outForSequenceData(fileName, dataFrame):
    d = pandas.DataFrame()

    for i in listOutputFeature:
        d[i] = dataFrame[i]

    d.to_csv(fileName,encoding='utf-8',index=False)

aryFeaturesScaled = [
'StockTrendPre1',
'StockTrendPre2',
'StockTrendPre3',
'StockTrendPre4',
'StockTrendPre5',
'StockTrendPre6',
'StockTrendPre7',
'StockTrendPre8',
'StockTrendPre9',
'StockTrendPre10',
'StockTrendPre11',
'StockTrendPre12',
'StockTrendPre13',
'StockTrendPre14',
'StockTrendPre15',
'StockTrendPre16',
'StockTrendPre17',
'StockTrendPre18',
'StockTrendPre19',
'StockTrendPre20',
'StockTrendPre21',
'StockTrendPre22',
'StockTrendPre23',
'StockTrendPre24',
'StockTrendPre25',
'HisStockPopNum',
'HisStockPopRatio',
'7DaysStockPopNum',
'7DaysStockPopRatio',
'CurDayNegNum',
'CurDayPosNum',
'7DaysNegNum',
'7DaysPosNum',
'7DaysRight',
'7DaysWrong',
'7DaysRightNum',
'7DaysWrongNum',
'7DaysRightRatio',
'7DaysProbNegWrong',
'7DaysProbNegRight',
'7DaysProbPosWrong',
'7DaysProbPosRight',
'PersonHisCommentCount',
'PersonHisCommentCountPos',
'PersonHisCommentCountNeg',
'PersonHisCommentCountWrn',
'PersonHisCommentCountRht',
'PersonCurStockHisCommentCount',
'PersonCurStockHisCommentCountPos',
'PersonCurStockHisCommentCountNeg',
'PersonCurStockHisCommentCountWrn',
'PersonCurStockHisCommentCountRht',
'Person7DaysCommentCount',
'Person7DaysCommentCountPos',
'Person7DaysCommentCountNeg',
'Person7DaysCommentCountWrn',
'Person7DaysCommentCountRht',
'PersonCurStock7DaysCommentCount',
'PersonCurStock7DaysCommentCountPos',
'PersonCurStock7DaysCommentCountNeg',
'PersonCurStock7DaysCommentCountWrn',
'PersonCurStock7DaysCommentCountRht',
'Person30DaysCommentCount',
'Person30DaysCommentCountPos',
'Person30DaysCommentCountNeg',
'Person30DaysCommentCountWrn',
'Person30DaysCommentCountRht',
'PersonCurStock30DaysCommentCount',
'PersonCurStock30DaysCommentCountPos',
'PersonCurStock30DaysCommentCountNeg',
'PersonCurStock30DaysCommentCountWrn',
'PersonCurStock30DaysCommentCountRht',
'Person90DaysCommentCount',
'Person90DaysCommentCountPos',
'Person90DaysCommentCountNeg',
'Person90DaysCommentCountWrn',
'Person90DaysCommentCountRht',
'PersonCurStock90DaysCommentCount',
'PersonCurStock90DaysCommentCountPos',
'PersonCurStock90DaysCommentCountNeg',
'PersonCurStock90DaysCommentCountWrn',
'PersonCurStock90DaysCommentCountRht'
]