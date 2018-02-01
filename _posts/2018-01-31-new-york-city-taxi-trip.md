---
layout: post
title: "1. New York City Taxi Trip Duration"
subtitle: "캐글 무작정 따라해보기"
categories: study
tags: kaggle
comments: true
---

# New York City Taxi Trip Duration
---
분석 공부를 위해 캐글의 대회들 중 좋은 성적을 받았던 커널들을 따라해보려고 합니다.
<br/>
## 0. Competition Introduction
---
이 대회에서의 목적은 뉴욕에서의 택시 여행 기간을 예측하는 모델을 만드는 것으로서,

가장 성과측정치가 좋았던 사람을 뽑는 것보다는 통찰력 있고 사용 가능한 모델을 만드는 사람에게 보상을 지불하는 형태로 진행되었다.

성과측정치는 다음과 같다.

$$ \epsilon =\sqrt { \frac { 1 }{ n } \sum _{ i=1 }^{ n }{ { (log({ p }_{ i }+1)\quad -\quad log({ a }_{ i }+1)) }^{ 2 } } } $$

Where:

ϵ is the RMSLE value (score) <br/>
n is the total number of observations in the (public/private) data set, <br/>
$$$ {p}_{i} $$$ is your prediction of trip duration, and <br/>
$$$ {a}_{i} $$$ is the actual trip duration for ii.  <br/>
log(x) is the natural logarithm of x <br/>

> 이 분석은 캐글 대회 [New York City Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration)의 데이터를 이용하여 진행하였으며 <br/>
연습을 위해 Weiying Wang의 [A Practical Guide to NY Taxi Data (0.379)](https://www.kaggle.com/onlyshadow/a-practical-guide-to-ny-taxi-data-0-379) 커널을 참고하여 진행한 분석이다.


```python
# Library import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize']=(10, 18)
%matplotlib inline
from datetime import datetime
from datetime import date
import xgboost as xgb
from sklearn.cluster import MiniBatchKMeans
import seaborn as sns
import warnings
sns.set()
warnings.filterwarnings('ignore')
```

## 1. Data Preview
---


```python
train = pd.read_csv('Input/train.csv',
                    parse_dates=['pickup_datetime'])
test = pd.read_csv('Input/test.csv',
                   parse_dates=['pickup_datetime'])
train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>vendor_id</th>
      <th>pickup_datetime</th>
      <th>dropoff_datetime</th>
      <th>passenger_count</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>store_and_fwd_flag</th>
      <th>trip_duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id2875421</td>
      <td>2</td>
      <td>2016-03-14 17:24:55</td>
      <td>2016-03-14 17:32:30</td>
      <td>1</td>
      <td>-73.982155</td>
      <td>40.767937</td>
      <td>-73.964630</td>
      <td>40.765602</td>
      <td>N</td>
      <td>455</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id2377394</td>
      <td>1</td>
      <td>2016-06-12 00:43:35</td>
      <td>2016-06-12 00:54:38</td>
      <td>1</td>
      <td>-73.980415</td>
      <td>40.738564</td>
      <td>-73.999481</td>
      <td>40.731152</td>
      <td>N</td>
      <td>663</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id3858529</td>
      <td>2</td>
      <td>2016-01-19 11:35:24</td>
      <td>2016-01-19 12:10:48</td>
      <td>1</td>
      <td>-73.979027</td>
      <td>40.763939</td>
      <td>-74.005333</td>
      <td>40.710087</td>
      <td>N</td>
      <td>2124</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id3504673</td>
      <td>2</td>
      <td>2016-04-06 19:32:31</td>
      <td>2016-04-06 19:39:40</td>
      <td>1</td>
      <td>-74.010040</td>
      <td>40.719971</td>
      <td>-74.012268</td>
      <td>40.706718</td>
      <td>N</td>
      <td>429</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id2181028</td>
      <td>2</td>
      <td>2016-03-26 13:30:55</td>
      <td>2016-03-26 13:38:10</td>
      <td>1</td>
      <td>-73.973053</td>
      <td>40.793209</td>
      <td>-73.972923</td>
      <td>40.782520</td>
      <td>N</td>
      <td>435</td>
    </tr>
  </tbody>
</table>
</div>




```python
#dataDir = '../input/'
#train = pd.read_csv(dataDir + 'train.csv')
#test = pd.read_csv(dataDir + 'test.csv')
```


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1458644 entries, 0 to 1458643
    Data columns (total 11 columns):
    id                    1458644 non-null object
    vendor_id             1458644 non-null int64
    pickup_datetime       1458644 non-null datetime64[ns]
    dropoff_datetime      1458644 non-null object
    passenger_count       1458644 non-null int64
    pickup_longitude      1458644 non-null float64
    pickup_latitude       1458644 non-null float64
    dropoff_longitude     1458644 non-null float64
    dropoff_latitude      1458644 non-null float64
    store_and_fwd_flag    1458644 non-null object
    trip_duration         1458644 non-null int64
    dtypes: datetime64[ns](1), float64(4), int64(3), object(3)
    memory usage: 122.4+ MB


null값 없음. 11개 열과 1458644개 행


```python
for df in (train, test):
    df['year'] = df['pickup_datetime'].dt.year
    df['month'] = df['pickup_datetime'].dt.month
    df['day'] = df['pickup_datetime'].dt.day
    df['hour'] = df['pickup_datetime'].dt.hour
    df['minute'] = df['pickup_datetime'].dt.minute
    df['store_and_fwd_flag'] = 1 * (df['store_and_fwd_flag'].values == 'Y')
```


```python
test.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>vendor_id</th>
      <th>pickup_datetime</th>
      <th>passenger_count</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>store_and_fwd_flag</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>hour</th>
      <th>minute</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id3004672</td>
      <td>1</td>
      <td>2016-06-30 23:59:58</td>
      <td>1</td>
      <td>-73.988129</td>
      <td>40.732029</td>
      <td>-73.990173</td>
      <td>40.756680</td>
      <td>0</td>
      <td>2016</td>
      <td>6</td>
      <td>30</td>
      <td>23</td>
      <td>59</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id3505355</td>
      <td>1</td>
      <td>2016-06-30 23:59:53</td>
      <td>1</td>
      <td>-73.964203</td>
      <td>40.679993</td>
      <td>-73.959808</td>
      <td>40.655403</td>
      <td>0</td>
      <td>2016</td>
      <td>6</td>
      <td>30</td>
      <td>23</td>
      <td>59</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id1217141</td>
      <td>1</td>
      <td>2016-06-30 23:59:47</td>
      <td>1</td>
      <td>-73.997437</td>
      <td>40.737583</td>
      <td>-73.986160</td>
      <td>40.729523</td>
      <td>0</td>
      <td>2016</td>
      <td>6</td>
      <td>30</td>
      <td>23</td>
      <td>59</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id2150126</td>
      <td>2</td>
      <td>2016-06-30 23:59:41</td>
      <td>1</td>
      <td>-73.956070</td>
      <td>40.771900</td>
      <td>-73.986427</td>
      <td>40.730469</td>
      <td>0</td>
      <td>2016</td>
      <td>6</td>
      <td>30</td>
      <td>23</td>
      <td>59</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id1598245</td>
      <td>1</td>
      <td>2016-06-30 23:59:33</td>
      <td>1</td>
      <td>-73.970215</td>
      <td>40.761475</td>
      <td>-73.961510</td>
      <td>40.755890</td>
      <td>0</td>
      <td>2016</td>
      <td>6</td>
      <td>30</td>
      <td>23</td>
      <td>59</td>
    </tr>
  </tbody>
</table>
</div>



RMSLE를 사용하여 점수를 매길 것이기 때문에, 위의 성과측정치를 사용하여 실제 여행 기간을 변경한다.

$$ \epsilon =\sqrt { \frac { 1 }{ n } \sum _{ i=1 }^{ n }{ { (log({ p }_{ i }+1)\quad -\quad log({ a }_{ i }+1)) }^{ 2 } } } $$

<br/>

```python
train = train.assign(log_trip_duration = np.log(train.trip_duration+1))
```


```python
train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>vendor_id</th>
      <th>pickup_datetime</th>
      <th>dropoff_datetime</th>
      <th>passenger_count</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>store_and_fwd_flag</th>
      <th>trip_duration</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>hour</th>
      <th>minute</th>
      <th>log_trip_duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id2875421</td>
      <td>2</td>
      <td>2016-03-14 17:24:55</td>
      <td>2016-03-14 17:32:30</td>
      <td>1</td>
      <td>-73.982155</td>
      <td>40.767937</td>
      <td>-73.964630</td>
      <td>40.765602</td>
      <td>0</td>
      <td>455</td>
      <td>2016</td>
      <td>3</td>
      <td>14</td>
      <td>17</td>
      <td>24</td>
      <td>6.122493</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id2377394</td>
      <td>1</td>
      <td>2016-06-12 00:43:35</td>
      <td>2016-06-12 00:54:38</td>
      <td>1</td>
      <td>-73.980415</td>
      <td>40.738564</td>
      <td>-73.999481</td>
      <td>40.731152</td>
      <td>0</td>
      <td>663</td>
      <td>2016</td>
      <td>6</td>
      <td>12</td>
      <td>0</td>
      <td>43</td>
      <td>6.498282</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id3858529</td>
      <td>2</td>
      <td>2016-01-19 11:35:24</td>
      <td>2016-01-19 12:10:48</td>
      <td>1</td>
      <td>-73.979027</td>
      <td>40.763939</td>
      <td>-74.005333</td>
      <td>40.710087</td>
      <td>0</td>
      <td>2124</td>
      <td>2016</td>
      <td>1</td>
      <td>19</td>
      <td>11</td>
      <td>35</td>
      <td>7.661527</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id3504673</td>
      <td>2</td>
      <td>2016-04-06 19:32:31</td>
      <td>2016-04-06 19:39:40</td>
      <td>1</td>
      <td>-74.010040</td>
      <td>40.719971</td>
      <td>-74.012268</td>
      <td>40.706718</td>
      <td>0</td>
      <td>429</td>
      <td>2016</td>
      <td>4</td>
      <td>6</td>
      <td>19</td>
      <td>32</td>
      <td>6.063785</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id2181028</td>
      <td>2</td>
      <td>2016-03-26 13:30:55</td>
      <td>2016-03-26 13:38:10</td>
      <td>1</td>
      <td>-73.973053</td>
      <td>40.793209</td>
      <td>-73.972923</td>
      <td>40.782520</td>
      <td>0</td>
      <td>435</td>
      <td>2016</td>
      <td>3</td>
      <td>26</td>
      <td>13</td>
      <td>30</td>
      <td>6.077642</td>
    </tr>
  </tbody>
</table>
</div>



## 2. Features
---
참고한 커널의 의견에 따르면 중요한 Features는 다음과 같다.
1. the pickup time (rush hour should cause longer trip duration.)
2. the trip distance
3. the pickup location

## 2.1. Pickup Time and Weekend Features
---
자세한 내용은 코드를 통해 알아보자.


```python
from datetime import datetime
holiday1 = pd.read_csv('Input/NYC_2016Holidays.csv', sep=';')
```


```python
# holiday['Date'] = holiday['Date'].apply(lambda x: x + ' 2016')
# 이 커널의 경우 위와 같이 laambda 수식을 이용하여 코드로 만들었는데,
# 굳이 저런 식으로 만들 필요가 있을까 싶어서 변경하였다.
holiday['Date'] = holiday['Date'] + ' 2016'
```


```python
# strptime 함수를 통해 January 01 2016 과 같은 형식으로
# 되어있는 문자열을 데이터 타임으로 변경한다.
# '%B %d %Y'를 통해 현재 데이터가 어떤 형태로
# 날짜를 표현하고 있는지를 알려준다.
holidays = [datetime.strptime(holiday.loc[i, 'Date'],
            '%B %d %Y').date() for i in range(len(holiday))]
```


```python
time_train = pd.DataFrame(index = range(len(train)))
time_test = pd.DataFrame(index = range(len(test)))
```


```python
from datetime import date
def restday(yr, month, day, holidays):
    is_rest = [None]*len(yr)
    is_weekend = [None]*len(yr)
    i=0
    for yy, mm, dd in zip(yr, month, day):
        is_weekend[i] = date(yy, mm, dd).isoweekday() in (6,7)
        is_rest[i] = is_weekend[i] or date(yy, mm, dd) in holidays
        i+=1
    return is_rest, is_weekend
```


```python
rest_day, weekend = restday(train.year, train.month, train.day, holidays)
#time_train = time_train.assign(rest_day=rest_day)
#time_train = time_train.assign(weekend=weekend)
time_train['rest_day'] = rest_day
time_train['weekend'] = weekend
time_train['pickup_time'] = train.hour+train.minute/60
time_train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rest_day</th>
      <th>weekend</th>
      <th>pickup_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>17.400000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>True</td>
      <td>True</td>
      <td>0.716667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>11.583333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>19.533333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>True</td>
      <td>True</td>
      <td>13.500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
rest_day, weekend = restday(test.year, test.month, test.day, holidays)
#time_train = time_train.assign(rest_day=rest_day)
#time_train = time_train.assign(weekend=weekend)
time_test['rest_day'] = rest_day
time_test['weekend'] = weekend
time_test['pickup_time'] = test.hour+test.minute/60
time_test.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rest_day</th>
      <th>weekend</th>
      <th>pickup_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>23.983333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>23.983333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>23.983333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>23.983333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>23.983333</td>
    </tr>
  </tbody>
</table>
</div>



## 2.2. Distance Features
---
### 2.2.1. OSRM Features
---
이 커널에 따르면 GPS로부터 얻은 실제 pickup과 dropoff의 위치 차이가 아니라 travel distance가 더 관련성 있는 데이터라고 한다.
이 둘의 차이가 어떻게 다른지는 아직까지 감이 잡히지 않아서 코드를 통해 이유를 알아보자.
여하튼 그 데이터를 구하기가 어렵지만 Oscarleo가 데이터셋을 올려줬다고 해서 그 데이터를 활용해보자.


```python
fastrout1 = pd.read_csv('Input/fastest_routes_train_part_1.csv',
                usecols=['id', 'total_distance', 'total_travel_time',  
                         'number_of_steps','step_direction'])
fastrout2 = pd.read_csv('Input/fastest_routes_train_part_2.csv',
                usecols=['id', 'total_distance', 'total_travel_time',  
                         'number_of_steps','step_direction'])
fastrout = pd.concat((fastrout1, fastrout2))
fastrout.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>total_distance</th>
      <th>total_travel_time</th>
      <th>number_of_steps</th>
      <th>step_direction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id2875421</td>
      <td>2009.1</td>
      <td>164.9</td>
      <td>5</td>
      <td>left|straight|right|straight|arrive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id2377394</td>
      <td>2513.2</td>
      <td>332.0</td>
      <td>6</td>
      <td>none|right|left|right|left|arrive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id3504673</td>
      <td>1779.4</td>
      <td>235.8</td>
      <td>4</td>
      <td>left|left|right|arrive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id2181028</td>
      <td>1614.9</td>
      <td>140.1</td>
      <td>5</td>
      <td>right|left|right|left|arrive</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id0801584</td>
      <td>1393.5</td>
      <td>189.4</td>
      <td>5</td>
      <td>right|right|right|left|arrive</td>
    </tr>
  </tbody>
</table>
</div>




```python
# map 함수는 데이터 각각에 특정한 함수를 적용하는 것인데,
# lambda를 통해 즉석에서 함수를 만들어서 적용한다.
right_turn = []
left_turn = []
right_turn += list(map(lambda x:x.count('right')-
                x.count('slight right'), fastrout.step_direction))
left_turn += list(map(lambda x:x.count('left')-
                x.count('slight left'),fastrout.step_direction))

```


```python
osrm_data = fastrout[['id', 'total_distance', 'total_travel_time',
                      'number_of_steps']]
osrm_data['right_steps'] = right_turn
osrm_data['left_steps'] = left_turn
osrm_data.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>total_distance</th>
      <th>total_travel_time</th>
      <th>number_of_steps</th>
      <th>right_steps</th>
      <th>left_steps</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id2875421</td>
      <td>2009.1</td>
      <td>164.9</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id2377394</td>
      <td>2513.2</td>
      <td>332.0</td>
      <td>6</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id3504673</td>
      <td>1779.4</td>
      <td>235.8</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id2181028</td>
      <td>1614.9</td>
      <td>140.1</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id0801584</td>
      <td>1393.5</td>
      <td>189.4</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



OSRM 데이터의 열은 1458643개이며, 실제 데이터보다 1개 열이 적다.
그래서 이 데이터를 사용하기 위해서는
SQL의 join을 사용하여서 데이터를 접합시켜야 한다.


```python
osrm_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1458643 entries, 0 to 758642
    Data columns (total 6 columns):
    id                   1458643 non-null object
    total_distance       1458643 non-null float64
    total_travel_time    1458643 non-null float64
    number_of_steps      1458643 non-null int64
    right_steps          1458643 non-null int64
    left_steps           1458643 non-null int64
    dtypes: float64(2), int64(3), object(1)
    memory usage: 77.9+ MB



```python
train = train.join(osrm_data.set_index('id'), on='id')
train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>vendor_id</th>
      <th>pickup_datetime</th>
      <th>dropoff_datetime</th>
      <th>passenger_count</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>store_and_fwd_flag</th>
      <th>...</th>
      <th>month</th>
      <th>day</th>
      <th>hour</th>
      <th>minute</th>
      <th>log_trip_duration</th>
      <th>total_distance</th>
      <th>total_travel_time</th>
      <th>number_of_steps</th>
      <th>right_steps</th>
      <th>left_steps</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id2875421</td>
      <td>2</td>
      <td>2016-03-14 17:24:55</td>
      <td>2016-03-14 17:32:30</td>
      <td>1</td>
      <td>-73.982155</td>
      <td>40.767937</td>
      <td>-73.964630</td>
      <td>40.765602</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>14</td>
      <td>17</td>
      <td>24</td>
      <td>6.122493</td>
      <td>2009.1</td>
      <td>164.9</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id2377394</td>
      <td>1</td>
      <td>2016-06-12 00:43:35</td>
      <td>2016-06-12 00:54:38</td>
      <td>1</td>
      <td>-73.980415</td>
      <td>40.738564</td>
      <td>-73.999481</td>
      <td>40.731152</td>
      <td>0</td>
      <td>...</td>
      <td>6</td>
      <td>12</td>
      <td>0</td>
      <td>43</td>
      <td>6.498282</td>
      <td>2513.2</td>
      <td>332.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id3858529</td>
      <td>2</td>
      <td>2016-01-19 11:35:24</td>
      <td>2016-01-19 12:10:48</td>
      <td>1</td>
      <td>-73.979027</td>
      <td>40.763939</td>
      <td>-74.005333</td>
      <td>40.710087</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>19</td>
      <td>11</td>
      <td>35</td>
      <td>7.661527</td>
      <td>11060.8</td>
      <td>767.6</td>
      <td>16.0</td>
      <td>5.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id3504673</td>
      <td>2</td>
      <td>2016-04-06 19:32:31</td>
      <td>2016-04-06 19:39:40</td>
      <td>1</td>
      <td>-74.010040</td>
      <td>40.719971</td>
      <td>-74.012268</td>
      <td>40.706718</td>
      <td>0</td>
      <td>...</td>
      <td>4</td>
      <td>6</td>
      <td>19</td>
      <td>32</td>
      <td>6.063785</td>
      <td>1779.4</td>
      <td>235.8</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id2181028</td>
      <td>2</td>
      <td>2016-03-26 13:30:55</td>
      <td>2016-03-26 13:38:10</td>
      <td>1</td>
      <td>-73.973053</td>
      <td>40.793209</td>
      <td>-73.972923</td>
      <td>40.782520</td>
      <td>0</td>
      <td>...</td>
      <td>3</td>
      <td>26</td>
      <td>13</td>
      <td>30</td>
      <td>6.077642</td>
      <td>1614.9</td>
      <td>140.1</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



테스트 데이터에도 동일한 처리를 한다.


```python
osrm_test = pd.read_csv('Input/fastest_routes_test.csv')
right_turn= list(map(lambda x:x.count('right')-
                x.count('slight right'),osrm_test.step_direction))
left_turn = list(map(lambda x:x.count('left')-
                x.count('slight left'),osrm_test.step_direction))

osrm_test = osrm_test[['id','total_distance','total_travel_time',
                       'number_of_steps']]
osrm_test['right_steps'] = right_turn
osrm_test['left_steps'] = left_turn
osrm_test.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>total_distance</th>
      <th>total_travel_time</th>
      <th>number_of_steps</th>
      <th>right_steps</th>
      <th>left_steps</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id0771704</td>
      <td>1497.1</td>
      <td>200.2</td>
      <td>7</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id3274209</td>
      <td>1427.1</td>
      <td>141.5</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id2756455</td>
      <td>2312.3</td>
      <td>324.6</td>
      <td>9</td>
      <td>4</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
test = test.join(osrm_test.set_index('id'), on='id')
test.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>vendor_id</th>
      <th>pickup_datetime</th>
      <th>passenger_count</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>store_and_fwd_flag</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>hour</th>
      <th>minute</th>
      <th>total_distance</th>
      <th>total_travel_time</th>
      <th>number_of_steps</th>
      <th>right_steps</th>
      <th>left_steps</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id3004672</td>
      <td>1</td>
      <td>2016-06-30 23:59:58</td>
      <td>1</td>
      <td>-73.988129</td>
      <td>40.732029</td>
      <td>-73.990173</td>
      <td>40.756680</td>
      <td>0</td>
      <td>2016</td>
      <td>6</td>
      <td>30</td>
      <td>23</td>
      <td>59</td>
      <td>3795.9</td>
      <td>424.6</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id3505355</td>
      <td>1</td>
      <td>2016-06-30 23:59:53</td>
      <td>1</td>
      <td>-73.964203</td>
      <td>40.679993</td>
      <td>-73.959808</td>
      <td>40.655403</td>
      <td>0</td>
      <td>2016</td>
      <td>6</td>
      <td>30</td>
      <td>23</td>
      <td>59</td>
      <td>2904.5</td>
      <td>200.0</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id1217141</td>
      <td>1</td>
      <td>2016-06-30 23:59:47</td>
      <td>1</td>
      <td>-73.997437</td>
      <td>40.737583</td>
      <td>-73.986160</td>
      <td>40.729523</td>
      <td>0</td>
      <td>2016</td>
      <td>6</td>
      <td>30</td>
      <td>23</td>
      <td>59</td>
      <td>1499.5</td>
      <td>193.2</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id2150126</td>
      <td>2</td>
      <td>2016-06-30 23:59:41</td>
      <td>1</td>
      <td>-73.956070</td>
      <td>40.771900</td>
      <td>-73.986427</td>
      <td>40.730469</td>
      <td>0</td>
      <td>2016</td>
      <td>6</td>
      <td>30</td>
      <td>23</td>
      <td>59</td>
      <td>7023.9</td>
      <td>494.8</td>
      <td>11</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id1598245</td>
      <td>1</td>
      <td>2016-06-30 23:59:33</td>
      <td>1</td>
      <td>-73.970215</td>
      <td>40.761475</td>
      <td>-73.961510</td>
      <td>40.755890</td>
      <td>0</td>
      <td>2016</td>
      <td>6</td>
      <td>30</td>
      <td>23</td>
      <td>59</td>
      <td>1108.2</td>
      <td>103.2</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



### 2.2.2. Other Distance Features
---
세 가지의 다른 거리 계산법을 사용한다.
<br/>
1. Haversine distance: the direct distance of two GPS location, taking into account that the earth is round.
1. Manhattan distance: the usual L1 distance, here the haversine distance is used to calculate each coordinate of distance.
1. Bearing: The direction of the trip. Using radian as unit. (I must admit that I am not fully understand the formula. I have starring at it for a long time but can't come up anything. If anyone can help explain that will do me a big favor.)
<br/>
---
출처는 [beluga](https://www.kaggle.com/gaborfodor/from-eda-to-the-top-lb-0-367)


```python
def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))
```


```python
List_dist = []
for df in (train, test):
    lat1, lng1, lat2, lng2 = (df['pickup_latitude'].values, df['pickup_longitude'].values,
                              df['dropoff_latitude'].values,df['dropoff_longitude'].values)
    dist = pd.DataFrame(index=range(len(df)))
    dist = dist.assign(haversine_dist = haversine_array(lat1, lng1, lat2, lng2))
    dist = dist.assign(manhattan_dist = dummy_manhattan_distance(lat1, lng1, lat2, lng2))
    dist = dist.assign(bearing = bearing_array(lat1, lng1, lat2, lng2))
    List_dist.append(dist)
Other_dist_train,Other_dist_test = List_dist
Other_dist_train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>haversine_dist</th>
      <th>manhattan_dist</th>
      <th>bearing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.498521</td>
      <td>1.735433</td>
      <td>99.970196</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.805507</td>
      <td>2.430506</td>
      <td>-117.153768</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6.385098</td>
      <td>8.203575</td>
      <td>-159.680165</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.485498</td>
      <td>1.661331</td>
      <td>-172.737700</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.188588</td>
      <td>1.199457</td>
      <td>179.473585</td>
    </tr>
  </tbody>
</table>
</div>



## 2.3. Location Features: K-means Clustering
---


```python
coord_pickup = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,                  
                          test[['pickup_latitude', 'pickup_longitude']].values))
coord_dropoff = np.vstack((train[['dropoff_latitude', 'dropoff_longitude']].values,                  
                           test[['dropoff_latitude', 'dropoff_longitude']].values))
```


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1458644 entries, 0 to 1458643
    Data columns (total 23 columns):
    id                    1458644 non-null object
    vendor_id             1458644 non-null int64
    pickup_datetime       1458644 non-null datetime64[ns]
    dropoff_datetime      1458644 non-null object
    passenger_count       1458644 non-null int64
    pickup_longitude      1458644 non-null float64
    pickup_latitude       1458644 non-null float64
    dropoff_longitude     1458644 non-null float64
    dropoff_latitude      1458644 non-null float64
    store_and_fwd_flag    1458644 non-null int64
    trip_duration         1458644 non-null int64
    year                  1458644 non-null int64
    month                 1458644 non-null int64
    day                   1458644 non-null int64
    hour                  1458644 non-null int64
    minute                1458644 non-null int64
    log_trip_duration     1458644 non-null float64
    total_distance        1458643 non-null float64
    total_travel_time     1458643 non-null float64
    number_of_steps       1458643 non-null float64
    right_steps           1458643 non-null float64
    left_steps            1458643 non-null float64
    pickup_dropoff_loc    1458644 non-null int32
    dtypes: datetime64[ns](1), float64(10), int32(1), int64(9), object(2)
    memory usage: 250.4+ MB



```python
# null값 있는 1개 행 제거
train.dropna(inplace=True)
```


```python
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 9258 entries, 0 to 9257
    Data columns (total 24 columns):
    id                    9258 non-null object
    vendor_id             9258 non-null int64
    pickup_datetime       9258 non-null datetime64[ns]
    passenger_count       9258 non-null int64
    pickup_longitude      9258 non-null float64
    pickup_latitude       9258 non-null float64
    dropoff_longitude     9258 non-null float64
    dropoff_latitude      9258 non-null float64
    store_and_fwd_flag    9258 non-null int64
    year                  9258 non-null int64
    month                 9258 non-null int64
    day                   9258 non-null int64
    hour                  9258 non-null int64
    minute                9258 non-null int64
    total_distance        9258 non-null float64
    total_travel_time     9258 non-null float64
    number_of_steps       9258 non-null int64
    right_steps           9258 non-null int64
    left_steps            9258 non-null int64
    pickup_dropoff_loc    9258 non-null int32
    Temp.                 9258 non-null float64
    Precip                9258 non-null float64
    snow                  9258 non-null int64
    Visibility            9258 non-null float64
    dtypes: datetime64[ns](1), float64(9), int32(1), int64(12), object(1)
    memory usage: 1.7+ MB



```python
# null값 존재하는 1개 행 제거
test.dropna(inplace=True)
test.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>vendor_id</th>
      <th>pickup_datetime</th>
      <th>passenger_count</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>store_and_fwd_flag</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>hour</th>
      <th>minute</th>
      <th>total_distance</th>
      <th>total_travel_time</th>
      <th>number_of_steps</th>
      <th>right_steps</th>
      <th>left_steps</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id3004672</td>
      <td>1</td>
      <td>2016-06-30 23:59:58</td>
      <td>1</td>
      <td>-73.988129</td>
      <td>40.732029</td>
      <td>-73.990173</td>
      <td>40.756680</td>
      <td>0</td>
      <td>2016</td>
      <td>6</td>
      <td>30</td>
      <td>23</td>
      <td>59</td>
      <td>3795.9</td>
      <td>424.6</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id3505355</td>
      <td>1</td>
      <td>2016-06-30 23:59:53</td>
      <td>1</td>
      <td>-73.964203</td>
      <td>40.679993</td>
      <td>-73.959808</td>
      <td>40.655403</td>
      <td>0</td>
      <td>2016</td>
      <td>6</td>
      <td>30</td>
      <td>23</td>
      <td>59</td>
      <td>2904.5</td>
      <td>200.0</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id1217141</td>
      <td>1</td>
      <td>2016-06-30 23:59:47</td>
      <td>1</td>
      <td>-73.997437</td>
      <td>40.737583</td>
      <td>-73.986160</td>
      <td>40.729523</td>
      <td>0</td>
      <td>2016</td>
      <td>6</td>
      <td>30</td>
      <td>23</td>
      <td>59</td>
      <td>1499.5</td>
      <td>193.2</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id2150126</td>
      <td>2</td>
      <td>2016-06-30 23:59:41</td>
      <td>1</td>
      <td>-73.956070</td>
      <td>40.771900</td>
      <td>-73.986427</td>
      <td>40.730469</td>
      <td>0</td>
      <td>2016</td>
      <td>6</td>
      <td>30</td>
      <td>23</td>
      <td>59</td>
      <td>7023.9</td>
      <td>494.8</td>
      <td>11</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id1598245</td>
      <td>1</td>
      <td>2016-06-30 23:59:33</td>
      <td>1</td>
      <td>-73.970215</td>
      <td>40.761475</td>
      <td>-73.961510</td>
      <td>40.755890</td>
      <td>0</td>
      <td>2016</td>
      <td>6</td>
      <td>30</td>
      <td>23</td>
      <td>59</td>
      <td>1108.2</td>
      <td>103.2</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
coords = np.hstack((coord_pickup,coord_dropoff))
sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=10, batch_size=10000).fit(coords[sample_ind])
for df in (train, test):
    df.loc[:, 'pickup_dropoff_loc'] = kmeans.predict(df[['pickup_latitude', 'pickup_longitude',
                                                         'dropoff_latitude','dropoff_longitude']])
```


```python
kmean10_train = train[['pickup_dropoff_loc']]
kmean10_test = test[['pickup_dropoff_loc']]
```


```python
plt.figure(figsize=(16,16))
N = 500
for i in range(10):
    plt.subplot(4,3,i+1)
    tmp = train[train.pickup_dropoff_loc==i]
    drop = plt.scatter(tmp['dropoff_longitude'][:N], tmp['dropoff_latitude'][:N], s=10, lw=0, alpha=0.5,label='dropoff')
    pick = plt.scatter(tmp['pickup_longitude'][:N], tmp['pickup_latitude'][:N], s=10, lw=0, alpha=0.4,label='pickup')    
    plt.xlim([-74.05,-73.75]);plt.ylim([40.6,40.9])
    plt.legend(handles = [pick,drop])
    plt.title('clusters %d'%i)
```


![output_42_0](https://i.imgur.com/evA8yKl.png)

## 2.4. Weather Features
---


```python
weather = pd.read_csv('Input/KNYC_Metars.csv', parse_dates=['Time'])
weather.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>Temp.</th>
      <th>Windchill</th>
      <th>Heat Index</th>
      <th>Humidity</th>
      <th>Pressure</th>
      <th>Dew Point</th>
      <th>Visibility</th>
      <th>Wind Dir</th>
      <th>Wind Speed</th>
      <th>Gust Speed</th>
      <th>Precip</th>
      <th>Events</th>
      <th>Conditions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-12-31 02:00:00</td>
      <td>7.8</td>
      <td>7.1</td>
      <td>NaN</td>
      <td>0.89</td>
      <td>1017.0</td>
      <td>6.1</td>
      <td>8.0</td>
      <td>NNE</td>
      <td>5.6</td>
      <td>0.0</td>
      <td>0.8</td>
      <td>None</td>
      <td>Overcast</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-12-31 03:00:00</td>
      <td>7.2</td>
      <td>5.9</td>
      <td>NaN</td>
      <td>0.90</td>
      <td>1016.5</td>
      <td>5.6</td>
      <td>12.9</td>
      <td>Variable</td>
      <td>7.4</td>
      <td>0.0</td>
      <td>0.3</td>
      <td>None</td>
      <td>Overcast</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-12-31 04:00:00</td>
      <td>7.2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.90</td>
      <td>1016.7</td>
      <td>5.6</td>
      <td>12.9</td>
      <td>Calm</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>Overcast</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-12-31 05:00:00</td>
      <td>7.2</td>
      <td>5.9</td>
      <td>NaN</td>
      <td>0.86</td>
      <td>1015.9</td>
      <td>5.0</td>
      <td>14.5</td>
      <td>NW</td>
      <td>7.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>Overcast</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-12-31 06:00:00</td>
      <td>7.2</td>
      <td>6.4</td>
      <td>NaN</td>
      <td>0.90</td>
      <td>1016.2</td>
      <td>5.6</td>
      <td>11.3</td>
      <td>West</td>
      <td>5.6</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>None</td>
      <td>Overcast</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('The Events has values {}.'.format(str(weather.Events.unique())))
```

    The Events has values ['None' 'Rain' 'Snow' 'Fog\n\t,\nSnow' 'Fog' 'Fog\n\t,\nRain'].



```python
weather['snow'] = 1*(weather.Events=='Snow') + 1*(weather.Events=='Fog\n\t,\nSnow')
weather['year'] = weather['Time'].dt.year
weather['month'] = weather['Time'].dt.month
weather['day'] = weather['Time'].dt.day
weather['hour'] = weather['Time'].dt.hour
weather = weather[weather['year'] == 2016][['month','day','hour','Temp.','Precip','snow','Visibility']]
```


```python
weather.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>month</th>
      <th>day</th>
      <th>hour</th>
      <th>Temp.</th>
      <th>Precip</th>
      <th>snow</th>
      <th>Visibility</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>5.6</td>
      <td>0.0</td>
      <td>0</td>
      <td>16.1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>5.6</td>
      <td>0.0</td>
      <td>0</td>
      <td>16.1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>5.6</td>
      <td>0.0</td>
      <td>0</td>
      <td>16.1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>16.1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>16.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
train = pd.merge(train, weather, on = ['month', 'day', 'hour'],
                 how = 'left')
test = pd.merge(test, weather, on = ['month', 'day', 'hour'],
                 how = 'left')
```

## 3. Analysis of Features
---


```python
tmp = train
tmp = pd.concat([tmp, time_train], axis=1)
```


```python
fig = plt.figure(figsize=(18, 8))
sns.boxplot(x='hour', y='log_trip_duration', data=tmp);
```

![output_51_0](https://i.imgur.com/n9eSgRB.png)



```python
sns.violinplot(x='month', y='log_trip_duration', hue='rest_day',
               data=tmp, split=True, inner='quart');
```


![output_52_0](https://i.imgur.com/uXYGfMR.png)



```python
tmp.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>vendor_id</th>
      <th>pickup_datetime</th>
      <th>dropoff_datetime</th>
      <th>passenger_count</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>store_and_fwd_flag</th>
      <th>...</th>
      <th>right_steps</th>
      <th>left_steps</th>
      <th>pickup_dropoff_loc</th>
      <th>Temp.</th>
      <th>Precip</th>
      <th>snow</th>
      <th>Visibility</th>
      <th>rest_day</th>
      <th>weekend</th>
      <th>pickup_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id2875421</td>
      <td>2.0</td>
      <td>2016-03-14 17:24:55</td>
      <td>2016-03-14 17:32:30</td>
      <td>1.0</td>
      <td>-73.982155</td>
      <td>40.767937</td>
      <td>-73.964630</td>
      <td>40.765602</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>4.4</td>
      <td>0.3</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>False</td>
      <td>False</td>
      <td>17.400000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>id2377394</td>
      <td>1.0</td>
      <td>2016-06-12 00:43:35</td>
      <td>2016-06-12 00:54:38</td>
      <td>1.0</td>
      <td>-73.980415</td>
      <td>40.738564</td>
      <td>-73.999481</td>
      <td>40.731152</td>
      <td>0.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>28.9</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>16.1</td>
      <td>True</td>
      <td>True</td>
      <td>0.716667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id3858529</td>
      <td>2.0</td>
      <td>2016-01-19 11:35:24</td>
      <td>2016-01-19 12:10:48</td>
      <td>1.0</td>
      <td>-73.979027</td>
      <td>40.763939</td>
      <td>-74.005333</td>
      <td>40.710087</td>
      <td>0.0</td>
      <td>...</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>-6.7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>16.1</td>
      <td>False</td>
      <td>False</td>
      <td>11.583333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>id3504673</td>
      <td>2.0</td>
      <td>2016-04-06 19:32:31</td>
      <td>2016-04-06 19:39:40</td>
      <td>1.0</td>
      <td>-74.010040</td>
      <td>40.719971</td>
      <td>-74.012268</td>
      <td>40.706718</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>7.2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>16.1</td>
      <td>False</td>
      <td>False</td>
      <td>19.533333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>id2181028</td>
      <td>2.0</td>
      <td>2016-03-26 13:30:55</td>
      <td>2016-03-26 13:38:10</td>
      <td>1.0</td>
      <td>-73.973053</td>
      <td>40.793209</td>
      <td>-73.972923</td>
      <td>40.782520</td>
      <td>0.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>9.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>16.1</td>
      <td>True</td>
      <td>True</td>
      <td>13.500000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




```python
sns.violinplot(x="pickup_dropoff_loc", y="log_trip_duration",
               hue="rest_day",
               data=tmp,
               split=True,inner="quart");
```


![output_54_0](https://i.imgur.com/T0QUGWk.png)


## 4. XGB Model : the Prediction of trip duration
---


```python
testdf = test[['vendor_id','passenger_count','pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude','store_and_fwd_flag']]
```


```python
len(train)
```




    1458643



이후 아직 진행 중 입니다.
