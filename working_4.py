import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings("ignore")

# Farklı mağazalar için 3 aylık item-level sales tahmini.
# 5 yıllık bir veri setinde 10 farklı mağaza ve 50 farklı item var.
# Buna göre mağaza-item kırılımında 3 ay sonrasının tahminlerini vermemiz gerekiyor.

# Note : İstatistiksel zaman serilerinde yaklaşımımız şu şekilde olmalı; Yöntemler var ve bu yöntemlerin model dereceleri,
# diğer ifadesiyle hiperparametreleri var. Olası tüm hiperparametre kombinasyonlarını deneriz, en iyi versiyonlara sahip
# olan özelliklerle modellerimizi kurarız. Olaya nedensellik bağlamında değil de yüksek tahmin başarısı bağlamında yaklaşıyoruz.
# SARIMA modeli triple exponential smoothing methoduna benzemektedir.

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

train = pd.read_csv('/Users/cenancanbikmaz/PycharmProjects/DSMLBC-7/HAFTA_9/datasets/train.csv', parse_dates=['date'])
test = pd.read_csv('/Users/cenancanbikmaz/PycharmProjects/DSMLBC-7/HAFTA_9/datasets/test.csv', parse_dates=['date'])
sample_sub = pd.read_csv('/Users/cenancanbikmaz/PycharmProjects/DSMLBC-7/HAFTA_9/datasets/sample_submission.csv')
df = pd.concat([train, test], sort=False)
df.head()
#         date  store  item  sales  id
# 0 2013-01-01      1     1   13.0 NaN
# 1 2013-01-02      1     1   11.0 NaN
# 2 2013-01-03      1     1   14.0 NaN
# 3 2013-01-04      1     1   13.0 NaN
# 4 2013-01-05      1     1   10.0 NaN

df['date'].min(), df['date'].max()
# (Timestamp('2013-01-01 00:00:00'), Timestamp('2018-03-31 00:00:00'))

check_df(train)
# ##################### Shape #####################
# (913000, 4)
# ##################### Types #####################
# date     datetime64[ns]
# store             int64
# item              int64
# sales             int64
# dtype: object
# ##################### Head #####################
#         date  store  item  sales
# 0 2013-01-01      1     1     13
# 1 2013-01-02      1     1     11
# 2 2013-01-03      1     1     14
# 3 2013-01-04      1     1     13
# 4 2013-01-05      1     1     10
# ##################### Tail #####################
#              date  store  item  sales
# 912995 2017-12-27     10    50     63
# 912996 2017-12-28     10    50     59
# 912997 2017-12-29     10    50     74
# 912998 2017-12-30     10    50     62
# 912999 2017-12-31     10    50     82
# ##################### NA #####################
# date     0
# store    0
# item     0
# sales    0
# dtype: int64
# ##################### Quantiles #####################
#        0.00  0.05  0.50   0.95   0.99   1.00
# store   1.0   1.0   5.5   10.0   10.0   10.0
# item    1.0   3.0  25.5   48.0   50.0   50.0
# sales   0.0  16.0  47.0  107.0  135.0  231.0

check_df(test)
# ##################### Shape #####################
# (45000, 4)
# ##################### Types #####################
# id                int64
# date     datetime64[ns]
# store             int64
# item              int64
# dtype: object
# ##################### Head #####################
#    id       date  store  item
# 0   0 2018-01-01      1     1
# 1   1 2018-01-02      1     1
# 2   2 2018-01-03      1     1
# 3   3 2018-01-04      1     1
# 4   4 2018-01-05      1     1
# ##################### Tail #####################
#           id       date  store  item
# 44995  44995 2018-03-27     10    50
# 44996  44996 2018-03-28     10    50
# 44997  44997 2018-03-29     10    50
# 44998  44998 2018-03-30     10    50
# 44999  44999 2018-03-31     10    50
# ##################### NA #####################
# id       0
# date     0
# store    0
# item     0
# dtype: int64
# ##################### Quantiles #####################
#        0.00     0.05     0.50      0.95      0.99     1.00
# id      0.0  2249.95  22499.5  42749.05  44549.01  44999.0
# store   1.0     1.00      5.5     10.00     10.00     10.0
# item    1.0     3.00     25.5     48.00     50.00     50.0

check_df(df)
# ##################### Shape #####################
# (958000, 5)
# ##################### Types #####################
# date     datetime64[ns]
# store             int64
# item              int64
# sales           float64
# id              float64
# dtype: object
# ##################### Head #####################
#         date  store  item  sales  id
# 0 2013-01-01      1     1   13.0 NaN
# 1 2013-01-02      1     1   11.0 NaN
# 2 2013-01-03      1     1   14.0 NaN
# 3 2013-01-04      1     1   13.0 NaN
# 4 2013-01-05      1     1   10.0 NaN
# ##################### Tail #####################
#             date  store  item  sales       id
# 44995 2018-03-27     10    50    NaN  44995.0
# 44996 2018-03-28     10    50    NaN  44996.0
# 44997 2018-03-29     10    50    NaN  44997.0
# 44998 2018-03-30     10    50    NaN  44998.0
# 44999 2018-03-31     10    50    NaN  44999.0
# ##################### NA #####################
# date          0
# store         0
# item          0
# sales     45000
# id       913000
# dtype: int64
# ##################### Quantiles #####################
#        0.00     0.05     0.50      0.95      0.99     1.00
# store   1.0     1.00      5.5     10.00     10.00     10.0
# item    1.0     3.00     25.5     48.00     50.00     50.0
# sales   0.0    16.00     47.0    107.00    135.00    231.0
# id      0.0  2249.95  22499.5  42749.05  44549.01  44999.0

df['sales'].describe([0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99])
# count    913000.000000
# mean         52.250287
# std          28.801144
# min           0.000000
# 10%          20.000000
# 30%          33.000000
# 50%          47.000000
# 70%          64.000000
# 80%          76.000000
# 90%          93.000000
# 95%         107.000000
# 99%         135.000000
# max         231.000000

# Kaç tane mağaza var ?
df[["store"]].nunique()
# store    10

# Kaç tane item var ?
df[["item"]].nunique()
# item    50

#Her mağaza'da eşit sayıda mı eşsiz item var ?
df.groupby(["store"])["item"].nunique()
# store
# 1     50
# 2     50
# 3     50
# 4     50
# 5     50
# 6     50
# 7     50
# 8     50
# 9     50
# 10    50

# Her mağazada eşit sayıda mı satış var?
df.groupby(["store", "item"]).agg({"sales": ["sum"]})
#                sales
#                  sum
# store item
# 1     1      36468.0
#       2      97050.0
#       3      60638.0
#       4      36440.0
#       5      30335.0
#               ...
# 10    46    120601.0
#       47     45204.0
#       48    105570.0
#       49     60317.0
#       50    135192.0
# [500 rows x 1 columns]

# Mağaza-item kırılımında satış istatistikleri
df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})
#                sales
#                  sum       mean median        std
# store item
# 1     1      36468.0  19.971522   19.0   6.741022
#       2      97050.0  53.148959   52.0  15.005779
#       3      60638.0  33.208105   33.0  10.072529
#       4      36440.0  19.956188   20.0   6.640618
#       5      30335.0  16.612815   16.0   5.672102
#               ...        ...    ...        ...
# 10    46    120601.0  66.046550   65.0  18.114991
#       47     45204.0  24.755750   24.0   7.924820
#       48    105570.0  57.814896   57.0  15.898538
#       49     60317.0  33.032311   32.0  10.091610
#       50    135192.0  74.037240   73.0  19.937566
# [500 rows x 4 columns]

#######################
# Feature Engineering #
#######################

# gün
# hafta
# yıl
# hafta içi
# hafta sonu
# özel günler
# haftanın kaçıncı günü
# ayın kaçıncı günü
# yılın kaçıncı haftası
# yılın kaçıncı ayı
# yılın kaçıncı günü

df.head()
#         date  store  item  sales  id
# 0 2013-01-01      1     1   13.0 NaN
# 1 2013-01-02      1     1   11.0 NaN
# 2 2013-01-03      1     1   14.0 NaN
# 3 2013-01-04      1     1   13.0 NaN
# 4 2013-01-05      1     1   10.0 NaN

# Zaman odaklı bir özellik yapıyoruz. Örüntü yakalama açısıyla olaylara yaklaşmak gerekir.
def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_week'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df

df = create_date_features(df)

df.head()
#         date  store  item  sales  id  month  day_of_month  day_of_year  week_of_year  day_of_week  year  is_wknd  is_month_start  is_month_end
# 0 2013-01-01      1     1   13.0 NaN      1             1            1             1            1  2013        0               1             0
# 1 2013-01-02      1     1   11.0 NaN      1             2            2             1            2  2013        0               0             0
# 2 2013-01-03      1     1   14.0 NaN      1             3            3             1            3  2013        0               0             0
# 3 2013-01-04      1     1   13.0 NaN      1             4            4             1            4  2013        1               0             0
# 4 2013-01-05      1     1   10.0 NaN      1             5            5             1            5  2013        1               0             0


df.groupby(["store", "item", "month"]).agg({"sales": ["sum", "mean", "median", "std"]})
#                    sales
#                       sum       mean median        std
# store item month
# 1     1    1       2125.0  13.709677   13.0   4.397413
#            2       2063.0  14.631206   14.0   4.668146
#            3       2728.0  17.600000   17.0   4.545013
#            4       3118.0  20.786667   20.0   4.894301
#            5       3448.0  22.245161   22.0   6.564705
#                    ...        ...    ...        ...
# 10    50   8      13108.0  84.567742   85.0  15.676527
#            9      11831.0  78.873333   79.0  15.207423
#            10     11322.0  73.045161   72.0  14.209171
#            11     11549.0  76.993333   77.0  16.253651
#            12      8724.0  56.283871   56.0  11.782529
# [6000 rows x 4 columns]
# Aylara göre hangi mağazanın ne kadar satış yaptığı bilgisine erişmiş olduk.

################
# Random Noise #
################
# Veri setinin boyutu kadar bir normal dağılımlı bir gürültü seti oluşturuyoruz. Bu veri setinin boyutu kadar
# oluşturduğumuz rasgele değerlere oluşturacak olduğumuz yeni özelliklerin üzerine ekliyoruz. Yani rasgele
# gürültü oluşturuyoruz. Veriye gürültü eklemek modelin aşırı öğrenmesini engelliyor.
def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))

########################
# Lag/Shifted Features #
########################

# Geçmiş dönem satış sayılarına ilişkin özellikler üretiyoruz.

# Burada veri setini mağazaya, item'e ve tarihe göre sıralıyoruz.
df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)

df["sales"].head(10)
# 0    13.0
# 1    11.0
# 2    14.0
# 3    13.0
# 4    10.0
# 5    12.0
# 6    10.0
# 7     9.0
# 8    12.0
# 9     9.0

df["sales"].shift(1).values[0:10]
# array([nan, 13., 11., 14., 13., 10., 12., 10.,  9., 12.])

pd.DataFrame({"sales": df["sales"].values[0:10],
              "lag1": df["sales"].shift(1).values[0:10],
              "lag2": df["sales"].shift(2).values[0:10],
              "lag3": df["sales"].shift(3).values[0:10],
              "lag4": df["sales"].shift(4).values[0:10]})
#    sales  lag1  lag2  lag3  lag4
# 0   13.0   NaN   NaN   NaN   NaN
# 1   11.0  13.0   NaN   NaN   NaN
# 2   14.0  11.0  13.0   NaN   NaN
# 3   13.0  14.0  11.0  13.0   NaN
# 4   10.0  13.0  14.0  11.0  13.0
# 5   12.0  10.0  13.0  14.0  11.0
# 6   10.0  12.0  10.0  13.0  14.0
# 7    9.0  10.0  12.0  10.0  13.0
# 8   12.0   9.0  10.0  12.0  10.0
# 9    9.0  12.0   9.0  10.0  12.0

df.groupby(["store", "item"])["sales"].head()
# 0         13.0
# 1         11.0
# 2         14.0
# 3         13.0
# 4         10.0
#           ...
# 911174    33.0
# 911175    37.0
# 911176    46.0
# 911177    51.0
# 911178    41.0

df.groupby(["store", "item"])["sales"].transform(lambda x: x.shift(1))
# 0         NaN
# 1        13.0
# 2        11.0
# 3        14.0
# 4        13.0
#          ...
# 44995     NaN
# 44996     NaN
# 44997     NaN
# 44998     NaN
# 44999     NaN
# Name: sales, Length: 958000, dtype: float64
# Bütün veri setine 1 gecikme uygulamış olduk.

def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe
# Bu fonksiyonda dataframe'i verip istediğimiz gecikmeleri belirteceğiz. 15, 30, 90 günlük gecikmeleri belirtip
# bu gecikmeler için yeni özellikler türetip bu gecikmelerin içerisinde gezeceğiz ve üstüne gürültü ekleyerek
# veri setine uygulayacak.

df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])
# 3 aylık periyota denk gelecek olan mevsimselliği ilgili periyot ve katları olacak şekilde seçtik.

df.head()
#         date  store  item  sales  id  month  day_of_month  day_of_year  week_of_year  day_of_week  year  is_wknd  is_month_start  is_month_end  sales_lag_91  sales_lag_98  sales_lag_105  sales_lag_112  sales_lag_119  sales_lag_126  sales_lag_182  sales_lag_364  sales_lag_546  sales_lag_728
# 0 2013-01-01      1     1   13.0 NaN      1             1            1             1            1  2013        0               1             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN
# 1 2013-01-02      1     1   11.0 NaN      1             2            2             1            2  2013        0               0             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN
# 2 2013-01-03      1     1   14.0 NaN      1             3            3             1            3  2013        0               0             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN
# 3 2013-01-04      1     1   13.0 NaN      1             4            4             1            4  2013        1               0             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN
# 4 2013-01-05      1     1   10.0 NaN      1             5            5             1            5  2013        1               0             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN

#########################
# Rolling Mean Features #
#########################

# Hareketli ortalamalar trendi gösterir, geçmiş bilgiyi taşırlar.

df["sales"].head(10)
# 0    13.0
# 1    11.0
# 2    14.0
# 3    13.0
# 4    10.0
# 5    12.0
# 6    10.0
# 7     9.0
# 8    12.0
# 9     9.0

df["sales"].rolling(window=2).mean().values[0:10]
# array([ nan, 12. , 12.5, 13.5, 11.5, 11. , 11. ,  9.5, 10.5, 10.5])

pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].rolling(window=2).mean().values[0:10],
              "roll3": df["sales"].rolling(window=3).mean().values[0:10],
              "roll5": df["sales"].rolling(window=5).mean().values[0:10]})
#    sales  roll2      roll3  roll5
# 0   13.0    NaN        NaN    NaN
# 1   11.0   12.0        NaN    NaN
# 2   14.0   12.5  12.666667    NaN
# 3   13.0   13.5  12.666667    NaN
# 4   10.0   11.5  12.333333   12.2
# 5   12.0   11.0  11.666667   12.0
# 6   10.0   11.0  10.666667   11.8
# 7    9.0    9.5  10.333333   10.8
# 8   12.0   10.5  10.333333   10.6
# 9    9.0   10.5  10.000000   10.4
# Gecikmelerin ortalamasıdır.

pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
              "roll3": df["sales"].shift(1).rolling(window=3).mean().values[0:10],
              "roll5": df["sales"].shift(1).rolling(window=5).mean().values[0:10]})
#    sales  roll2      roll3  roll5
# 0   13.0    NaN        NaN    NaN
# 1   11.0    NaN        NaN    NaN
# 2   14.0   12.0        NaN    NaN
# 3   13.0   12.5  12.666667    NaN
# 4   10.0   13.5  12.666667    NaN
# 5   12.0   11.5  12.333333   12.2
# 6   10.0   11.0  11.666667   12.0
# 7    9.0   11.0  10.666667   11.8
# 8   12.0    9.5  10.333333   10.8
# 9    9.0   10.5  10.333333   10.6

# shitf uygulayarak geçmiş trendi yakalamaya çalışırken gözlem birimini almayıp ondan öncekini alarak uyguluyoruz.

def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

# Üretilen özellikleri veri setinin içerisine yerleştirmek üzere yazılmış bir fonksiyondur.
# store, item kırılımında sales değişkeninin 1 günlük shift'ini alıp ön tanımlı olarak girecek olduğumuz window'ların
# her birisi için hesaplama işlemini gerçekleştiriyoruz.

df = roll_mean_features(df, [365, 546])

df.head()
#         date  store  item  sales  id  month  day_of_month  day_of_year  week_of_year  day_of_week  year  is_wknd  is_month_start  is_month_end  sales_lag_91  sales_lag_98  sales_lag_105  sales_lag_112  sales_lag_119  sales_lag_126  sales_lag_182  sales_lag_364  sales_lag_546  sales_lag_728  sales_roll_mean_365  sales_roll_mean_546
# 0 2013-01-01      1     1   13.0 NaN      1             1            1             1            1  2013        0               1             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN                  NaN                  NaN
# 1 2013-01-02      1     1   11.0 NaN      1             2            2             1            2  2013        0               0             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN                  NaN                  NaN
# 2 2013-01-03      1     1   14.0 NaN      1             3            3             1            3  2013        0               0             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN                  NaN                  NaN
# 3 2013-01-04      1     1   13.0 NaN      1             4            4             1            4  2013        1               0             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN                  NaN                  NaN
# 4 2013-01-05      1     1   10.0 NaN      1             5            5             1            5  2013        1               0             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN                  NaN                  NaN

########################################
# Exponentially Weighted Mean Features #
########################################
# alpha değerleri burada ağırlıklı ortalamadır. 0.99 değeri verdiğimizde yakınındaki değerlere ağrılık verecek,
# 0.1 olduğunda ise uzağındaki değerlere ağırlık verecektir. Gecikme sayısını 1 verip buna göre üssel ortalamalarını aldık.
pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
              "ewm099": df["sales"].shift(1).ewm(alpha=0.99).mean().values[0:10],
              "ewm095": df["sales"].shift(1).ewm(alpha=0.95).mean().values[0:10],
              "ewm07": df["sales"].shift(1).ewm(alpha=0.7).mean().values[0:10],
              "ewm01": df["sales"].shift(1).ewm(alpha=0.1).mean().values[0:10]})
#    sales  roll2     ewm099     ewm095      ewm07      ewm01
# 0   13.0    NaN        NaN        NaN        NaN        NaN
# 1   11.0    NaN  13.000000  13.000000  13.000000  13.000000
# 2   14.0   12.0  11.019802  11.095238  11.461538  11.947368
# 3   13.0   12.5  13.970201  13.855107  13.287770  12.704797
# 4   10.0   13.5  13.009702  13.042750  13.084686  12.790637
# 5   12.0   11.5  10.030097  10.152137  10.920146  12.109179
# 6   10.0   11.0  11.980301  11.907607  11.676595  12.085878
# 7    9.0   11.0  10.019803  10.095380  10.502722  11.686057
# 8   12.0    9.5   9.010198   9.054769   9.450748  11.214433
# 9    9.0   10.5  11.970102  11.852738  11.235259  11.342672

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)

df.head()
#         date  store  item  sales  id  month  day_of_month  day_of_year  week_of_year  day_of_week  year  is_wknd  is_month_start  is_month_end  sales_lag_91  sales_lag_98  sales_lag_105  sales_lag_112  sales_lag_119  sales_lag_126  sales_lag_182  sales_lag_364  sales_lag_546  sales_lag_728  sales_roll_mean_365  sales_roll_mean_546  sales_ewm_alpha_095_lag_91  sales_ewm_alpha_095_lag_98  sales_ewm_alpha_095_lag_105  sales_ewm_alpha_095_lag_112  sales_ewm_alpha_095_lag_180  \
# 0 2013-01-01      1     1   13.0 NaN      1             1            1             1            1  2013        0               1             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN                  NaN                  NaN                         NaN                         NaN                          NaN                          NaN                          NaN
# 1 2013-01-02      1     1   11.0 NaN      1             2            2             1            2  2013        0               0             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN                  NaN                  NaN                         NaN                         NaN                          NaN                          NaN                          NaN
# 2 2013-01-03      1     1   14.0 NaN      1             3            3             1            3  2013        0               0             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN                  NaN                  NaN                         NaN                         NaN                          NaN                          NaN                          NaN
# 3 2013-01-04      1     1   13.0 NaN      1             4            4             1            4  2013        1               0             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN                  NaN                  NaN                         NaN                         NaN                          NaN                          NaN                          NaN
# 4 2013-01-05      1     1   10.0 NaN      1             5            5             1            5  2013        1               0             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN                  NaN                  NaN                         NaN                         NaN                          NaN                          NaN                          NaN
#    sales_ewm_alpha_095_lag_270  sales_ewm_alpha_095_lag_365  sales_ewm_alpha_095_lag_546  sales_ewm_alpha_095_lag_728  sales_ewm_alpha_09_lag_91  sales_ewm_alpha_09_lag_98  sales_ewm_alpha_09_lag_105  sales_ewm_alpha_09_lag_112  sales_ewm_alpha_09_lag_180  sales_ewm_alpha_09_lag_270  sales_ewm_alpha_09_lag_365  sales_ewm_alpha_09_lag_546  sales_ewm_alpha_09_lag_728  sales_ewm_alpha_08_lag_91  sales_ewm_alpha_08_lag_98  sales_ewm_alpha_08_lag_105  sales_ewm_alpha_08_lag_112  \
# 0                          NaN                          NaN                          NaN                          NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN
# 1                          NaN                          NaN                          NaN                          NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN
# 2                          NaN                          NaN                          NaN                          NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN
# 3                          NaN                          NaN                          NaN                          NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN
# 4                          NaN                          NaN                          NaN                          NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN
#    sales_ewm_alpha_08_lag_180  sales_ewm_alpha_08_lag_270  sales_ewm_alpha_08_lag_365  sales_ewm_alpha_08_lag_546  sales_ewm_alpha_08_lag_728  sales_ewm_alpha_07_lag_91  sales_ewm_alpha_07_lag_98  sales_ewm_alpha_07_lag_105  sales_ewm_alpha_07_lag_112  sales_ewm_alpha_07_lag_180  sales_ewm_alpha_07_lag_270  sales_ewm_alpha_07_lag_365  sales_ewm_alpha_07_lag_546  sales_ewm_alpha_07_lag_728  sales_ewm_alpha_05_lag_91  sales_ewm_alpha_05_lag_98  sales_ewm_alpha_05_lag_105  \
# 0                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN
# 1                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN
# 2                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN
# 3                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN
# 4                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN
#    sales_ewm_alpha_05_lag_112  sales_ewm_alpha_05_lag_180  sales_ewm_alpha_05_lag_270  sales_ewm_alpha_05_lag_365  sales_ewm_alpha_05_lag_546  sales_ewm_alpha_05_lag_728
# 0                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN
# 1                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN
# 2                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN
# 3                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN
# 4                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN

df.shape
# (958000, 71)

####################
# One-Hot Encoding #
####################

df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])

df.shape
# (958000, 146)

####################################
# Converting sales to log(1+sales) #
####################################

df['sales'] = np.log1p(df["sales"].values)

df.head()
#         date     sales  id  day_of_month  day_of_year  week_of_year  year  is_wknd  is_month_start  is_month_end  sales_lag_91  sales_lag_98  sales_lag_105  sales_lag_112  sales_lag_119  sales_lag_126  sales_lag_182  sales_lag_364  sales_lag_546  sales_lag_728  sales_roll_mean_365  sales_roll_mean_546  sales_ewm_alpha_095_lag_91  sales_ewm_alpha_095_lag_98  sales_ewm_alpha_095_lag_105  sales_ewm_alpha_095_lag_112  sales_ewm_alpha_095_lag_180  sales_ewm_alpha_095_lag_270  \
# 0 2013-01-01  2.639057 NaN             1            1             1  2013        0               1             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN                  NaN                  NaN                         NaN                         NaN                          NaN                          NaN                          NaN                          NaN
# 1 2013-01-02  2.484907 NaN             2            2             1  2013        0               0             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN                  NaN                  NaN                         NaN                         NaN                          NaN                          NaN                          NaN                          NaN
# 2 2013-01-03  2.708050 NaN             3            3             1  2013        0               0             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN                  NaN                  NaN                         NaN                         NaN                          NaN                          NaN                          NaN                          NaN
# 3 2013-01-04  2.639057 NaN             4            4             1  2013        1               0             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN                  NaN                  NaN                         NaN                         NaN                          NaN                          NaN                          NaN                          NaN
# 4 2013-01-05  2.397895 NaN             5            5             1  2013        1               0             0           NaN           NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN            NaN                  NaN                  NaN                         NaN                         NaN                          NaN                          NaN                          NaN                          NaN
#    sales_ewm_alpha_095_lag_365  sales_ewm_alpha_095_lag_546  sales_ewm_alpha_095_lag_728  sales_ewm_alpha_09_lag_91  sales_ewm_alpha_09_lag_98  sales_ewm_alpha_09_lag_105  sales_ewm_alpha_09_lag_112  sales_ewm_alpha_09_lag_180  sales_ewm_alpha_09_lag_270  sales_ewm_alpha_09_lag_365  sales_ewm_alpha_09_lag_546  sales_ewm_alpha_09_lag_728  sales_ewm_alpha_08_lag_91  sales_ewm_alpha_08_lag_98  sales_ewm_alpha_08_lag_105  sales_ewm_alpha_08_lag_112  sales_ewm_alpha_08_lag_180  \
# 0                          NaN                          NaN                          NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN
# 1                          NaN                          NaN                          NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN
# 2                          NaN                          NaN                          NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN
# 3                          NaN                          NaN                          NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN
# 4                          NaN                          NaN                          NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN
#    sales_ewm_alpha_08_lag_270  sales_ewm_alpha_08_lag_365  sales_ewm_alpha_08_lag_546  sales_ewm_alpha_08_lag_728  sales_ewm_alpha_07_lag_91  sales_ewm_alpha_07_lag_98  sales_ewm_alpha_07_lag_105  sales_ewm_alpha_07_lag_112  sales_ewm_alpha_07_lag_180  sales_ewm_alpha_07_lag_270  sales_ewm_alpha_07_lag_365  sales_ewm_alpha_07_lag_546  sales_ewm_alpha_07_lag_728  sales_ewm_alpha_05_lag_91  sales_ewm_alpha_05_lag_98  sales_ewm_alpha_05_lag_105  sales_ewm_alpha_05_lag_112  \
# 0                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN
# 1                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN
# 2                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN
# 3                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN
# 4                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                         NaN                        NaN                        NaN                         NaN                         NaN
#    sales_ewm_alpha_05_lag_180  sales_ewm_alpha_05_lag_270  sales_ewm_alpha_05_lag_365  sales_ewm_alpha_05_lag_546  sales_ewm_alpha_05_lag_728  store_1  store_2  store_3  store_4  store_5  store_6  store_7  store_8  store_9  store_10  item_1  item_2  item_3  item_4  item_5  item_6  item_7  item_8  item_9  item_10  item_11  item_12  item_13  item_14  item_15  item_16  item_17  item_18  item_19  item_20  item_21  item_22  item_23  item_24  item_25  item_26  item_27  item_28  item_29  item_30  \
# 0                         NaN                         NaN                         NaN                         NaN                         NaN        1        0        0        0        0        0        0        0        0         0       1       0       0       0       0       0       0       0       0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0
# 1                         NaN                         NaN                         NaN                         NaN                         NaN        1        0        0        0        0        0        0        0        0         0       1       0       0       0       0       0       0       0       0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0
# 2                         NaN                         NaN                         NaN                         NaN                         NaN        1        0        0        0        0        0        0        0        0         0       1       0       0       0       0       0       0       0       0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0
# 3                         NaN                         NaN                         NaN                         NaN                         NaN        1        0        0        0        0        0        0        0        0         0       1       0       0       0       0       0       0       0       0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0
# 4                         NaN                         NaN                         NaN                         NaN                         NaN        1        0        0        0        0        0        0        0        0         0       1       0       0       0       0       0       0       0       0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0
#    item_31  item_32  item_33  item_34  item_35  item_36  item_37  item_38  item_39  item_40  item_41  item_42  item_43  item_44  item_45  item_46  item_47  item_48  item_49  item_50  day_of_week_0  day_of_week_1  day_of_week_2  day_of_week_3  day_of_week_4  day_of_week_5  day_of_week_6  month_1  month_2  month_3  month_4  month_5  month_6  month_7  month_8  month_9  month_10  month_11  month_12
# 0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0              0              1              0              0              0              0              0        1        0        0        0        0        0        0        0        0         0         0         0
# 1        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0              0              0              1              0              0              0              0        1        0        0        0        0        0        0        0        0         0         0         0
# 2        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0              0              0              0              1              0              0              0        1        0        0        0        0        0        0        0        0         0         0         0
# 3        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0              0              0              0              0              1              0              0        1        0        0        0        0        0        0        0        0         0         0         0
# 4        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0        0              0              0              0              0              0              1              0        1        0        0        0        0        0        0        0        0         0         0         0

#########
# Model #
#########

########################
# Custom Cost Function #
########################

# LightGBM'i optimize ederken iterasyonlarda bakacak olduğumuz şey loss fonksiyondu.
# MSE: Mean Squared Error
# RMSE: Root Mena Squared Error
# MAE: Mean Absolute Error
# MAPE: Mean Absolute Percentage Error
# SMAPE: Symmetric Mean Absolute Percentage Error (Adjusted MAPE)

# Gerçek değerlerle, tahmin edilen değerleri kıyaslar, sonrasında bunların mutlak değerini alır ve bunları
# yüzdelik forma çevirir.
def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

# Daha öncesinde logaritmik bir dönüşüm yapmıştık. Bu logaritmik dönüşümü geri alacak şekilde MAPE fonksiyonuna
# bunu uygulayacağız. LightGBM'in anlayabileceği bir şekilde optimize ediyoruz.
def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

##############################
# Time-Based Validation Sets #
##############################

# Beklenen test tesi 2018'in ilk 3 ayı elimizde de 2017'nin son ayına kadarlık bir eğitim veri setimiz var.
# 2017'nin son üç ayı bizden beklenen örüntüyle örtüşür mü örtüşmez mi bilemiyoruz dolayısıyla aynı örüntüyü
# yakalayabileceğimiz başka bir noktaya gitmek istiyoruz. Buradaki tercihimiz doğal olarak 2017'nin ilk 3 ayını
# validasyon seti olarak ayırıyoruz.

test.head()
#    id       date  store  item
# 0   0 2018-01-01      1     1
# 1   1 2018-01-02      1     1
# 2   2 2018-01-03      1     1
# 3   3 2018-01-04      1     1
# 4   4 2018-01-05      1     1

test.tail()
#           id       date  store  item
# 44995  44995 2018-03-27     10    50
# 44996  44996 2018-03-28     10    50
# 44997  44997 2018-03-29     10    50
# 44998  44998 2018-03-30     10    50
# 44999  44999 2018-03-31     10    50

train["date"].min(), train["date"].max()
# (Timestamp('2013-01-01 00:00:00'), Timestamp('2017-12-31 00:00:00'))

test["date"].min(), test["date"].max()
# (Timestamp('2018-01-01 00:00:00'), Timestamp('2018-03-31 00:00:00'))

# 2017'nin başına kadar (2016'nın sonuna kadar) train seti.
train = df.loc[(df["date"] < "2017-01-01"), :]

# 2017'nin ilk 3 ayını validasyon seti yapıyoruz.
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]

# Bağımsız değişkenlerimizi cols adında bir değişkenin içine koyuyoruz. "date", "id", "sales", "year" değişkenleri
# ile bir işimiz yok.
cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

Y_train = train["sales"]
X_train = train[cols]

Y_val = val["sales"]
X_val = val[cols]

##################
# LightGBM Model #
##################

# "num_leaves" yaprak sayısını belirtir.
# "learning_rate" öğrenme oranıdır, shrinkage_rate, eta
# "feature_fraction" her iterasyonda gözlemlerin belirli bir oranda mı yoksa hepsini mi göz önünde bulunduralım parametresidir.
#  rf'nin random subspace özelliği. her iterasyonda rastgele göz önünde bulundurulacak değişken sayısı.
# "max_depth" maximum derinlik.
# "verbose" kaç adımda bir raporlama yapması gerektiğini belirtiyoruz.
# "num_boost_round" n estimators demektir, number of boosting iterations. En az 10000-15000 civarı yapmak lazım.
# "early_stopping_round" validasyon setindeki metrik belirli bir early_stopping_rounds'da ilerlemiyorsa yani
# hata düşmüyorsa modellemeyi durdur.
# "nthread" işlemcilerin kullanımı ile alakalı. -1 dediğimizde işlemcilerin hepsini kullanır.
# Hem train süresini kısaltır hem de overfit'e engel olur.
# LightGBM parameters

# metric mae: l1, absolute loss, mean_absolute_error, regression_l1
# l2, square loss, mean_squared_error, mse, regression_l2, regression
# rmse, root square loss, root_mean_squared_error, l2_root
# mape, MAPE loss, mean_absolute_percentage_error

lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200,
              'nthread': -1}

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
# Data bölümüne bağımsız değişkenleri label bölümüne bağımlı değişkeni, feature_name argümanına da bağımsız değişkenlerin
# isimlerini giriyoruz.

lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)
# Validasyon seti


# feval argümanı custom cost function'ı belirtir. Mean Absolute Error değerini yüzdelik bir hata oranı olarak göstermesini belirttik.
model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params["num_boost_round"],
                  early_stopping_rounds=lgb_params["early_stopping_rounds"],
                  feval=lgbm_smape,
                  verbose_eval=100)

# [LightGBM] [Warning] Auto-choosing row-wise multi-threading, the overhead of testing was 0.016832 seconds.
# You can set `force_row_wise=true` to remove the overhead.
# And if memory is not enough, you can set `force_col_wise=true`.
# Training until validation scores don't improve for 200 rounds
# [100]	training's l1: 0.172546	training's SMAPE: 17.5961	valid_1's l1: 0.171318	valid_1's SMAPE: 17.5098
# [200]	training's l1: 0.142145	training's SMAPE: 14.5582	valid_1's l1: 0.145245	valid_1's SMAPE: 14.8992
# [300]	training's l1: 0.136584	training's SMAPE: 14.0017	valid_1's l1: 0.140459	valid_1's SMAPE: 14.4192
# [400]	training's l1: 0.134453	training's SMAPE: 13.7893	valid_1's l1: 0.138927	valid_1's SMAPE: 14.2659
# [500]	training's l1: 0.133125	training's SMAPE: 13.657	valid_1's l1: 0.137614	valid_1's SMAPE: 14.1343
# [600]	training's l1: 0.132192	training's SMAPE: 13.5638	valid_1's l1: 0.136639	valid_1's SMAPE: 14.0362
# [700]	training's l1: 0.131487	training's SMAPE: 13.4933	valid_1's l1: 0.135878	valid_1's SMAPE: 13.9596
# Did not meet early stopping. Best iteration is:
# [1000]	training's l1: 0.130023	training's SMAPE: 13.3466	valid_1's l1: 0.134515	valid_1's SMAPE: 13.8224

# Note : LightGBM'in en önemli hiperparametre optimizasyonu num_boost_round'dur.

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

# Daha öncesinde modele sokulan değerlerin logaritması alınarak modele sokulmuştu. Şimdi bunların tersini alıyoruz.
smape(np.expm1(y_pred_val), np.expm1(Y_val))
# 13.822393276127935 hatamız var.

######################
# Feature Importance #
######################

def plot_lgb_importances(model, plot=False, num=10):

    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))

# 30 değişkeni göstermek için num parametresine 30 değerini girdik.
plot_lgb_importances(model, num=30)
#                          feature  split       gain
# 17           sales_roll_mean_546    920  54.288368
# 13                 sales_lag_364   1259  13.192058
# 16           sales_roll_mean_365    632   9.891184
# 60    sales_ewm_alpha_05_lag_365    364   4.960250
# 18    sales_ewm_alpha_095_lag_91     78   2.759355
# 1                    day_of_year    768   2.117360
# 54     sales_ewm_alpha_05_lag_91     84   1.866880
# 3                        is_wknd    229   1.215430
# 123                day_of_week_0    242   1.175220
# 141                     month_12    305   1.116339
# 2                   week_of_year    298   0.973359
# 36     sales_ewm_alpha_08_lag_91     14   0.906002
# 6                   sales_lag_91     89   0.845151
# 27     sales_ewm_alpha_09_lag_91     36   0.548153
# 7                   sales_lag_98     21   0.501067
# 62    sales_ewm_alpha_05_lag_728    391   0.388387
# 59    sales_ewm_alpha_05_lag_270    192   0.353259
# 53    sales_ewm_alpha_07_lag_728     72   0.350212
# 44    sales_ewm_alpha_08_lag_728     26   0.348790
# 51    sales_ewm_alpha_07_lag_365     51   0.234604
# 35    sales_ewm_alpha_09_lag_728     17   0.148732
# 12                 sales_lag_182    119   0.141630
# 136                      month_7    132   0.106572
# 19    sales_ewm_alpha_095_lag_98     10   0.095046
# 130                      month_1    102   0.093066
# 129                day_of_week_6     80   0.086785
# 45     sales_ewm_alpha_07_lag_91     16   0.085298
# 26   sales_ewm_alpha_095_lag_728      6   0.078847
# 126                day_of_week_3    112   0.061916
# 28     sales_ewm_alpha_09_lag_98      5   0.046419

# split argümanı değişkenin ağaç yöntemlerinde kaç bölme işleminde kullanıldığını göstermektedir. gain argümanı ise
# değişkenin modeli tahmin etmesinde ne kadar kazanç sağladığını göstermektedir.

plot_lgb_importances(model, num=30, plot=True)

lgb.plot_importance(model, max_num_features=20, figsize=(10, 10), importance_type="gain")
plt.show()

###############
# Final Model #
###############

# NA olmayan sales değerleri train setine karşılık gelir.
train = df.loc[~df.sales.isna()]

Y_train = train["sales"]
X_train = train[cols]

test = df.loc[df.sales.isna()]
X_test = test[cols]

# dataleakage
# tüm veride age 0-1 minmax dönüşümü
# train - test

lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)
# [LightGBM] [Warning] Auto-choosing row-wise multi-threading, the overhead of testing was 0.016028 seconds.
# You can set `force_row_wise=true` to remove the overhead.
# And if memory is not enough, you can set `force_col_wise=true`.

test_preds = model.predict(X_test, num_iteration=model.best_iteration)

submission_df = test.loc[:, ["id", "sales"]]
submission_df["sales"] = np.expm1(test_preds)
submission_df["id"] = submission_df.id.astype(int)
submission_df.to_csv("submission_demand.csv", index=False)
submission_df.head(20)
#     id      sales
# 0    0  11.784425
# 1    1  14.145350
# 2    2  13.632879
# 3    3  14.270368
# 4    4  17.758245
# 5    5  18.004468
# 6    6  19.996335
# 7    7  13.258884
# 8    8  14.792087
# 9    9  14.818561
# 10  10  15.148537
# 11  11  16.874620
# 12  12  15.801704
# 13  13  17.523760
# 14  14  12.808774
# 15  15  15.357608
# 16  16  13.822296
# 17  17  15.709829
# 18  18  17.256970
# 19  19  18.706011

