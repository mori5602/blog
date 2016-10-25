# -*- coding: utf-8 -*-
'''
Code based on:
https://github.com/corrieelston/datalab/blob/master/FinancialTimeSeriesTensorFlow.ipynb
'''
from __future__ import print_function

import time
import datetime
import urllib2
from os import path
import os.path

import operator as op
from collections import namedtuple
import numpy as np
import pandas as pd
import jsm

#chainer用に追加
import chainer
from chainer import report, training, Chain, datasets, iterators, optimizers
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import tuple_dataset

#YQL用
import urllib
from urllib2 import urlopen
import json

#折り返し幅を広げる
pd.set_option('line_width', 1000)



DAYS_BACK = 30
FROM_YEAR = '1991'
US_EXCHANGES_DEFINE = [
    ['realGDAXI', '^GDAXI'], #ドイツ
    ['realHSI', '^HSI'], #香港
    #['N225', '^N225'],
    ['realSP500', '^GSPC'], #アメリカ
    ['realSSEC', '000001.SS'], #上海
    ['realIXIC', '^IXIC'],
    ['realDJI', '^DJI'],
    #['OIL','USO.MX'] # 原油ETF
]

JP_EXCHANGES_DEFINE = [
    ['realTOPIX', '1306'] #JPX400ETF
]

MONEY_EXCHANGES_DEFINE = [
    ['realUSDJPY', 'USDJPY.csv'] #ドル円為替
]

OFFLINE_EXCHANGES_DEFINE = [
#    ['TOPIX', 'TOPIX.csv']
]
EXCHANGES_DEFINE = US_EXCHANGES_DEFINE + JP_EXCHANGES_DEFINE + MONEY_EXCHANGES_DEFINE + OFFLINE_EXCHANGES_DEFINE

EXCHANGES_LABEL = [exchange[0] for exchange in EXCHANGES_DEFINE]

def setupDateURL(urlBase):
    now = datetime.date.today()
    return urlBase.replace('__FROM_YEAR__', str(now.year - 1))\
            .replace('__TO_MONTH__', str(now.month))\
            .replace('__TO_DAY__', str(now.day))\
            .replace('__TO_YEAR__', str(now.year))

'''
def fetchCSV(fileName, url):
    if path.isfile(fileName):
        print('fetch CSV for local: ' + fileName)
        with open(fileName) as f:
            return f.read()
    else:
        print('fetch CSV for url: ' + url)
        csv = urllib2.urlopen(url).read()
        with open(fileName, 'w') as f:
            f.write(csv)
        return csv
'''
#http://chart.finance.yahoo.com/table.csv?s=^GSPC&a=8&b=19&c=2016&d=9&e=19&f=2016&g=d&ignore=.csv

def fetchCSV(fileName, url):
    print('fetch CSV for url: ' + url)
    csv = urllib2.urlopen(url).read()
    with open(fileName, 'w') as f:
        f.write(csv)
    return csv


def fetchYahooComFinance(name, code):
    fileName = 'index_%s.csv' % name
    url = setupDateURL('http://chart.finance.yahoo.com/table.csv?s=%s&a=8&b=__TO_DAY__&c=__FROM_YEAR__&d=__TO_MONTH__&e=__TO_DAY__&f=__TO_YEAR__&g=d&ignore=.csv' % code)
    csv = fetchCSV(fileName, url)

def fetchYahooCoJpFinance(name, code):
    fileName = 'index_%s.csv' % name
    now = datetime.date.today()
    start_date = datetime.date(now.year-1, now.month, now.day)
    end_date = datetime.date(now.year, now.month , now.day )

    print('fetch CSV for Yahoo.co.jp')
    c = jsm.Quotes()
    stock_price = c.get_historical_prices(code, jsm.DAILY, start_date, end_date)
    stock_date_list = [each_day_data.date for each_day_data in stock_price]
    stock_date_str = [x.strftime('%Y-%m-%d') for x in stock_date_list]
    stock_open_list = [each_day_data.open for each_day_data in stock_price]
    stock_cloce_list = [each_day_data.close for each_day_data in stock_price]
    stock_high_list = [each_day_data.high for each_day_data in stock_price]
    stock_low_list = [each_day_data.low for each_day_data in stock_price]
    stock_volume_list = [each_day_data.volume for each_day_data in stock_price]
    stock_adj_close_list = [each_day_data._adj_close for each_day_data in stock_price]
    stock_list = {'Date':stock_date_str,'Open':stock_open_list,'Close':stock_cloce_list,'High':stock_high_list,'Low':stock_low_list,'Volume':stock_volume_list,'Adj Close':stock_adj_close_list}
    stock_list_df = pd.DataFrame(stock_list,columns=['Date','Open','High','Low','Close','Volume','Adj Close'])
    #stock_list_df = pd.DataFrame(stock_list, columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'])
    stock_list_df.to_csv(fileName)
    return fileName

def fetchYahooExchange(name, code):
    # YQLをAPI経由で利用します
    url = "https://query.yahooapis.com/v1/public/yql"
    params = {
        "q": 'select * from yahoo.finance.xchange where pair in ("USDJPY")',
        "format": "json",
        "env": "store://datatables.org/alltableswithkeys"
    }
    url += "?" + urllib.urlencode(params)
    res = urlopen(url)

    # 結果はJSON形式で受け取ることができます
    result = json.loads(res.read().decode('utf-8'))
    # pprint(result)
    """
    {'query': {'count': 1,
               'created': '2016-08-22T02:57:07Z',
               'lang': 'en-US',
               'results': {'rate': {'Ask': '100.6850',
                                    'Bid': '100.6380',
                                    'Date': '8/21/2016',
                                    'Name': 'USD/JPY',
                                    'Rate': '100.6380',
                                    'Time': '10:58pm',
                                    'id': 'USDJPY'}}}}
    """

    # その中から必要な情報（今回はUSD→JPYの為替レート）を取得します
    day = result["query"]["created"]
    rate = result["query"]["results"]["rate"]["Rate"]

    tstr = '2012-12-29 13:49:37'
    tdatetime = datetime.datetime.strptime(day, '%Y-%m-%dT%H:%M:%SZ')
    tdate = datetime.date(tdatetime.year, tdatetime.month, tdatetime.day - 1)

    exchange_df = pd.DataFrame(columns=['Open','High','Low','Close','Adj Close'],index=[tdate])
    #exchange_df['Data'] = day
    exchange_df['Close'] = rate
    exchange_df['Adj Close'] = rate
    #print('USD/JPY:', rate)

    df = pd.read_csv("index_USDJPY.csv",index_col='Date', parse_dates=True)
    df = df.append(exchange_df).copy()
    #df = pd.concat([df,exchange_df])
    #df = df.append(exchange_df)

    df.index = df.index.to_datetime('%Y-%m-%d')
    df.index.names = ['Date']
    #print(df.tail())
    df.to_csv("index_realUSDJPY.csv")


def fetchStockIndexes():
    '''株価指標のデータをダウンロードしファイルに保存
    '''

    #yahoo.comからダウンロード
    for exchange in US_EXCHANGES_DEFINE:
        fetchYahooComFinance(exchange[0], exchange[1])

    #yahoo.co.jpからダウンロード
    for exchange in JP_EXCHANGES_DEFINE:
        fetchYahooCoJpFinance(exchange[0],exchange[1])

    #YQLで為替情報取得
    for exchange in MONEY_EXCHANGES_DEFINE:
        fetchYahooExchange(exchange[0], exchange[1])


def load_exchange_dataframes():
    '''EXCHANGESに対応するCSVファイルをPandasのDataFrameとして読み込む。
    Returns:
        {EXCHANGES[n]: pd.DataFrame()}
    '''
    return {exchange: load_exchange_dataframe(exchange)
            for exchange in EXCHANGES_LABEL}


def load_exchange_dataframe(exchange):
    '''exchangeに対応するCSVファイルをPandasのDataFrameとして読み込む。
    Args:
        exchange: 指標名
    Returns:
        pd.DataFrame()
    '''
    return pd.read_csv('index_{}.csv'.format(exchange)).set_index('Date').sort_index()


def get_closing_data(dataframes):
    '''各指標の終値カラムをまとめて1つのDataFrameに詰める。
    Args:
        dataframes: {key: pd.DataFrame()}
    Returns:
        pd.DataFrame()
    '''
    #print(dataframes["USDJPY"].index)
    closing_data = pd.DataFrame(index=dataframes['realTOPIX'].index)
    for exchange, dataframe in dataframes.items():
        closing_data[exchange] = dataframe['Close'].copy()
    #closing_data.to_csv("closing_datav2.csv")
    closing_data = closing_data.fillna(method='ffill')
    return closing_data

def get_log_return_data(closing_data):
    '''各指標について、終値を1日前との比率の対数をとって正規化する。
    Args:
        closing_data: pd.DataFrame()
    Returns:
        pd.DataFrame()
    '''

    log_return_data = pd.DataFrame()
    for exchange in closing_data:
        # np.log(当日終値 / 前日終値) で前日からの変化率を算出
        # 前日よりも上がっていればプラス、下がっていればマイナスになる
        log_return_data[exchange] = np.log(closing_data[exchange]/closing_data[exchange].shift())

    return log_return_data


def build_training_data(log_return_data, target_exchange, max_days_back=DAYS_BACK):

    '''学習データを作る。分類クラスは、target_exchange指標の終値が前日に比べて上ったか下がったかの2つである。
    また全指標の終値の、当日から数えてmax_days_back日前までを含めて入力データとする。
    Args:
        log_return_data: pd.DataFrame()
        target_exchange: 学習目標とする指標名
        max_days_back: 何日前までの終値を学習データに含めるか
        use_subset (float): 短時間で動作を確認したい時用: log_return_dataのうち一部だけを学習データに含める
    Returns:
        pd.DataFrame()
    '''
    # 「上がる」「下がる」の結果を計算
    columns = []
    '''
    for colname, exchange, operator in iter_categories(target_exchange):
        columns.append(colname)
        # 全ての XXX_positive, XXX_negative を 0 に初期化
        log_return_data[colname] = 0
        # XXX_positive の場合は >=  0 の全てのインデックスを
        # XXX_negative の場合は < 0 の全てのインデックスを取得し、それらに 1 を設定する
        indices = operator(log_return_data[exchange], 0)
        log_return_data.ix[indices, colname] = 1
    '''
    #print (log_return_data.head())

    # 各指標のカラム名を追加
    for colname, _, _ in iter_exchange_days_back(target_exchange, max_days_back):
        columns.append(colname)
    #print(columns)
    '''
    columns には計算対象の positive, negative と各指標の日数分のラベルが含まれる
    例：[
        'SP500_positive',
        'SP500_negative',
        'DOW_0',
        'DOW_1',
        'DOW_2',
        'FTSE_0',
        'FTSE_1',
        'FTSE_2',
        'GDAXI_0',
        'GDAXI_1',
        'GDAXI_2',
        'HSI_0',
        'HSI_1',
        'HSI_2',
        'N225_0',
        'N225_1',
        'N225_2',
        'NASDAQ_0',
        'NASDAQ_1',
        'NASDAQ_2',
        'SP500_1',
        'SP500_2',
        'SP500_3',
        'SSEC_0',
        'SSEC_1',
        'SSEC_2'
    ]
    計算対象の SP500 だけ当日のデータを含めたらダメなので1〜3が入る
    '''

    # データ数をもとめる
    #max_index = len(log_return_data) - max_days_back + 1
    max_index = len(log_return_data)
    #if use_subset is not None:
    #    # データを少なくしたいとき
    #    max_index = int(max_index * use_subset)

    # 学習データを作る
    training_test_data = pd.DataFrame(columns=columns)


    for i in range(max_days_back , max_index):

        values = {}
        # 「上がる」「下がる」の答を入れる
        #for colname, _, _ in iter_categories(target_exchange):
        #    values[colname] = log_return_data[colname].ix[i]
        #values['label'] = log_return_data['label'].ix[i]

        # 学習データを入れる
        for colname, exchange, days_back in iter_exchange_days_back(target_exchange, max_days_back):
            values[colname] = log_return_data[exchange].ix[i - days_back]
        training_test_data = training_test_data.append(values, ignore_index=True)

    # 値上がりする（１）、値下がりする（０）の回答を入れる
    training_len = len(training_test_data)

    training_test_data = training_test_data.iloc[training_len - 1]
    training_test_data.to_csv('real_training_test_data.csv',index=False)

    return training_test_data


def iter_exchange_days_back(target_exchange, max_days_back):
    '''指標名、何日前のデータを読むか、カラム名を列挙する。
    '''
    for exchange in EXCHANGES_LABEL:
        # SP500 の結果を予測するのに SP500 の当日の値が含まれてはいけないので１日づらす
        start_days_back = 1 if exchange == target_exchange else 0
        #start_days_back = 1 # N225 で行う場合は全て前日の指標を使うようにする
        end_days_back = start_days_back + max_days_back
        for days_back in range(start_days_back, end_days_back):
            colname = '{}_{}'.format(exchange, days_back)
            yield colname, exchange, days_back

'''
def split_training_test_data(num_categories, training_test_data):
    #学習データをトレーニング用とテスト用に分割する。

    # 学習データ格納。先頭1列は回答データのため、スキップする。
    predictors_tf = training_test_data[training_test_data.columns[1:]].copy()

    # 回答データ格納。先頭1列のみ。
    classes_tf = training_test_data[training_test_data.columns[:1]].copy()

    #classes_tf["class"] = classes_tf["N225_positive"].apply(lambda x: 1 if x == 1 else 0 )
    #classes_tf = classes_tf["class"]

    # 学習用とテスト用のデータサイズを求める
    training_set_size = int(len(training_test_data) * 0.8)
    test_set_size = len(training_test_data) - training_set_size

    train_x = np.array(predictors_tf[:training_set_size])
    train_y = np.array(classes_tf[:training_set_size]).flatten()
    test_x = np.array(predictors_tf[training_set_size:])
    test_y = np.array(classes_tf[training_set_size:]).flatten()

    #chainerはfloat32しかダメ。変換する。
    train_x = train_x.astype(np.float32)
    train_y = train_y.astype(np.int32)
    test_x = test_x.astype(np.float32)
    test_y = test_y.astype(np.int32)


    # 古いデータ0.8を学習とし、新しいデータ0.2がテストとなる
    return train_x,train_y,test_x,test_y
'''

def feed_dict(env, test=False):
    '''学習/テストに使うデータを生成する。
    '''
    prefix = 'test' if test else 'training'
    predictors = getattr(env.dataset, '{}_predictors'.format(prefix))
    classes = getattr(env.dataset, '{}_classes'.format(prefix))
    return {
        env.feature_data: predictors.values,
        env.actual_classes: classes.values.reshape(len(classes.values), len(classes.columns))
    }


#ロードするモデルを定義
class MyModel(chainer.Chain):
    def __init__(self, n_in,n_units, n_out):
        super(MyModel, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(n_in, n_units),  # n_in -> n_units
            l2=L.Linear(n_units, n_units),  # n_units -> n_units
            l3=L.Linear(n_units, n_units),
            l4=L.Linear(n_units,n_units),
            l5=L.Linear(n_units,n_out)

        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.relu(self.l4(h3))

        #return self.l3(h2)
        #return self.l4(h3)
        return self.l5(h4)

def main(args):
    print('株価指標データをダウンロードしcsvファイルに保存')
    fetchStockIndexes()


    print('株価指標データを読み込む')
    all_data  = load_exchange_dataframes()
    #print(all_data['realTOPIX'])
    print('終値を取得')
    closing_data = get_closing_data(all_data)
    closing_data.to_csv("real_closing_data.csv")

    print('データを学習に使える形式に正規化')
    log_return_data = get_log_return_data(closing_data)
    '''
    label = []
    for key,row in all_data[args.target_exchange].iterrows():
        if row['Close'] - row['Open'] > 0:
            label.append(1)
        else:
            label.append(0)
    log_return_data['label'] = label
    #print (log_return_data.tail())
    '''

    print('答と学習データを作る')
    training_test_data = build_training_data(log_return_data, args.target_exchange)


    # 学習データ
    x = training_test_data

    model = L.Classifier(MyModel(x.shape[0], args.unit, 4))
    chainer.serializers.load_npz("stock.model", model)

    x = np.array(x,dtype=np.float32)
    x = [x]

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()
        x = chainer.cuda.to_gpu(x)
        #y = chainer.cuda.to_gpu(y)


    y = model.predictor(x)
    z = F.softmax(y)
    print(y.data)
    print(z.data)
    print(z.data.argmax())

    print("OK")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('target_exchange', choices=EXCHANGES_LABEL)
    #parser.add_argument('--steps', type=int, default=10000)
    #parser.add_argument('--checkin', type=int, default=1000)
    parser.add_argument('--use-subset', type=float, default=None)
    #parser.add_argument('--inspect', type=bool, default=False)
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))

    os.environ['PATH'] += ':/usr/local/cuda/bin'


    main(args)