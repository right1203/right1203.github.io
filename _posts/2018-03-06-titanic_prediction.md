---
layout: post
title: "Titaninc:Machine Learning from Disaster"
subtitle: "Kaggle Study"
categories: study
tags: kaggle
comments: true
---
> ## <span style="color:red">Exploration in Titanic</span>

# **Introduction**
* Titanic: Machine Learning from Disaster
* 주제 : Explore 단계를 스스로 생각해서 진행해보자
* **data description summary**
    * 타이타닉 호의 침몰 당시 승객 데이터를 이용하여 생존자를 예측
    * 891개의 트레이닝 데이터와 418개의 테스트 데이터를 미리 분리시켜 놓은 상태
    * 기존에 한 번 진행해봤던 분석이기 때문에 랜덤 포레스트를 이해하고 예측력을 높이는 것에 초점
* 이 커널은 [Titanic Data Science Solutions - Manav Sehgal](https://www.kaggle.com/startupsci/titanic-data-science-solutions)의 내용을 따라한 부분이 매우 많습니다.
* [SEMMA](https://en.wikipedia.org/wiki/SEMMA) 방법론에 따라 분석을 진행하였습니다.

## 1. <span style="color:red"> S</span>ample

---


```python
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# 데이터 불러오기
gender = pd.read_csv("../input/gender_submission.csv")
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
combine = [train, test]
```

이 대회의 경우 Sample단계에서 수행하는 데이터 수집, 데이터 추출과 분리 모두 미리 수행되어있기에

각 데이터들을 변수에 담는 것으로 넘어간다.

## 2. <span style="color:red"> E</span>xplore

---

> #### 변수 설명과 관계 단순 예측

<span style="color:red">PassengerId</span>   :  각 승객의 고유 번호로 의미 없음 - 모델링 시 제거

<span style="color:red">Survived</span>         :  생존 여부 (종속 변수)

<span style="color:red">Pclass</span>             :  티켓의 승선권 클래스 - 승객의 사회적, 경제적 지위를 알 수 있다. 클래스에 따라 좀 더 생존에 용이한 구조이지 않을까?
* 1st = Upper
* 2nd = Middle
* 3rd = Lower

<span style="color:red">Name</span>             :  이름 - 이름으로 딱히 의미가 있을까? 큰 의미 없을 듯.

<span style="color:red">Sex</span>                 :  성별 - 성별에 따라 먼저 구조될 수도, 생존에 용이할 수 있을 듯하다.

<span style="color:red">Age</span>                 :  나이 - 각 세대별(10대, 20대 또는 아이, 청소년, 성인 등)로 차이가 있을 수 있다.

<span style="color:red">SibSp</span>             :  동반한 Sibling(형제자매)와 Spouse(배우자)의 수 - 동반 여행자의 수에 따라 생존 차이가 있을 수도 있다.

<span style="color:red">Parch</span>              :  동반한 Parent(부모) Child(자식)의 수 - 위와 동일

<span style="color:red">Ticket</span>             :  티켓의 고유넘버 - 티켓 자체로 생존여부를 알 수는 없을 듯하다.

<span style="color:red">Fare</span>                :  티켓의 요금 - 비싼 요금의 티켓과 생존 여부에 상관이 있을 수도 있다. 하지만 Pclass가 이 변수를 categorical하게 만든 변수가 아닐까?

<span style="color:red">Cabin</span>             :  객실 번호 - 어떤 객실 번호 범위는 좀 더 갑판 등의 생존에 용이한 위치에 있을 수도 있을 듯하다.

<span style="color:red">Embarked</span>     :  승선한 항 - 큰 의미 없을 듯하다.
* C = Cherbourg
* Q = Queenstown
* S = Southampton


```python
# 데이터 구조 파악
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.6+ KB



```python
train.isnull().sum()
```




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64



> #### Training Data

891개의 데이터 행

**Age에 300여개 결측치** : 결측치가 많지 않고 나이에 따른 생존 여부와 상관있을 것으로 예상되기에 결측치를 예측하여 채워넣어야 할 듯하다.

**Cabin에 700여개의 결측치** : 객실 넘버가 생존 여부와 관련 있을 수도 있으나, 결측치가 너무 많아 사용하기 힘들어서 제거해야할 듯하다.

**Embarked에 2개의 결측치** : 2개의 경우 결측치가 있는 행을 제거하여도 무방하며 어떠한 값으로 채워도 큰 영향은 없을 것으로 예상


```python
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 11 columns):
    PassengerId    418 non-null int64
    Pclass         418 non-null int64
    Name           418 non-null object
    Sex            418 non-null object
    Age            332 non-null float64
    SibSp          418 non-null int64
    Parch          418 non-null int64
    Ticket         418 non-null object
    Fare           417 non-null float64
    Cabin          91 non-null object
    Embarked       418 non-null object
    dtypes: float64(2), int64(4), object(5)
    memory usage: 36.0+ KB



```python
test.isnull().sum()
```




    PassengerId      0
    Pclass           0
    Name             0
    Sex              0
    Age             86
    SibSp            0
    Parch            0
    Ticket           0
    Fare             1
    Cabin          327
    Embarked         0
    dtype: int64

> #### Test Data

418개의 데이터 행

**Age에 80여개 결측치**

**Fare에 1개 결측치** : 결측치 행을 제거하여도 무방하지만, 1개의 결측치이기 때문에 같은 Pclass의 데이터 값으로 채워넣어도 무방할 것으로 예쌍

**Cabin에 300여개 결측치**


```python
train.describe()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



Survived의 경우 평균이 0.38 (생존자보다 사망자가 많음)

나이는 0.42세부터 80세까지 존재하며 평균적으로 30세 수준

Fare가 0~512까지 존재 (0의 경우 무료 승객이나 선원, 또는 Null값일 수 있을 듯)


```python
#object describe
train.describe(include=["O"])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Sex</th>
      <th>Ticket</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891</td>
      <td>891</td>
      <td>891</td>
      <td>204</td>
      <td>889</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>891</td>
      <td>2</td>
      <td>681</td>
      <td>147</td>
      <td>3</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Chaffee, Mr. Herbert Fuller</td>
      <td>male</td>
      <td>CA. 2343</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>577</td>
      <td>7</td>
      <td>4</td>
      <td>644</td>
    </tr>
  </tbody>
</table>
</div>



Ticket은 891개의 데이터 중 681개가 고유값을 가지기 때문에 큰 의미를 가지기 힘들 것으로 예상.

Cabin 또한 200개의 객실 중 147개가 고유값. 따라서 결측치가 없었다 하더라도 큰 의미 없을 듯. **Cabin 열 제거해야할 듯**


```python
train.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



### 설명변수들의 분류에 따른 종속변수의 차이

---

위에서 단순 예측해본 변수들의 설명력을 실제 데이터를 이용하여 파악하여 본다.


```python
corr = train.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
fig, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            linewidths=.5, cbar_kws={"shrink": .6});
```


![output_18_0](https://i.imgur.com/3LvCG0p.png)



```python
train[['Pclass', 'Survived', 'Age']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Survived</th>
      <th>Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.629630</td>
      <td>38.233441</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.472826</td>
      <td>29.877630</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.242363</td>
      <td>25.140620</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.pivot_table('Survived', index='Sex', columns='Pclass')
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Pclass</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>female</th>
      <td>0.968085</td>
      <td>0.921053</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>male</th>
      <td>0.368852</td>
      <td>0.157407</td>
      <td>0.135447</td>
    </tr>
  </tbody>
</table>
</div>




```python
grid = sns.FacetGrid(train, size=5)
grid.map(sns.barplot, 'Pclass', 'Survived', palette='deep', order=[1,2,3]);
```


![output_21_0](https://i.imgur.com/hDFyTbh.png)



```python
train[['Sex', 'Survived']].groupby('Sex', as_index=False).mean().sort_values(by='Sex', ascending=False)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sex</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>male</td>
      <td>0.188908</td>
    </tr>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>0.742038</td>
    </tr>
  </tbody>
</table>
</div>




```python
grid = sns.FacetGrid(train, size=5)
grid.map(sns.barplot, 'Sex', 'Survived', order=['male','female'], palette='deep');
```


![output_23_0](https://i.imgur.com/HAgbc8x.png)


```python
grid = sns.FacetGrid(train, hue='Survived', size=3.5)
grid.map(plt.hist, 'Age', alpha=.5)
grid.add_legend();
```


![output_24_0](https://i.imgur.com/Y9fByfT.png)



```python
grid = sns.FacetGrid(train, col='SibSp', col_wrap=4, size = 3)
grid.map(sns.barplot, 'Sex', 'Survived', order=['male','female'], palette='deep');
```


![output_25_0](https://i.imgur.com/WuP5WxD.png)



```python
grid = sns.FacetGrid(train, col='Parch', col_wrap=4, size = 3)
grid.map(sns.barplot, 'Sex', 'Survived', order=['male','female'], palette='deep');
```


![output_26_0](https://i.imgur.com/rOvsV3A.png)



```python
grid = sns.FacetGrid(train, size = 5)
grid.map(sns.barplot, 'Ticket', 'Survived', palette='deep');
```

    /opt/conda/lib/python3.6/site-packages/seaborn/axisgrid.py:703: UserWarning: Using the barplot function without specifying `order` is likely to produce an incorrect plot.
      warnings.warn(warning)



![output_27_1](https://i.imgur.com/X69S87m.png)



```python
grid = sns.FacetGrid(train, hue='Survived', size=3.5)
grid.map(plt.hist, 'Fare', alpha=.5)
grid.add_legend();
```


![output_28_0](https://i.imgur.com/hClbEMY.png)



```python
grid = sns.FacetGrid(train, size = 5)
grid.map(sns.barplot, 'Cabin', 'Survived', palette='deep');
```

    /opt/conda/lib/python3.6/site-packages/seaborn/axisgrid.py:703: UserWarning: Using the barplot function without specifying `order` is likely to produce an incorrect plot.
      warnings.warn(warning)



![output_29_1](https://i.imgur.com/WKDn6RU.png)



```python
grid = sns.FacetGrid(train, col='Embarked', size = 3)
grid.map(sns.barplot, 'Sex', 'Survived', palette='deep', order=['female','male']);
```


![output_30_0](https://i.imgur.com/qvNg1O5.png)



```python
grid = sns.FacetGrid(train, col='Embarked')
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', order=[1,2,3], hue_order=['female','male'], palette='deep')
grid.add_legend();
```


![output_31_0](https://i.imgur.com/2XM6PzY.png)



```python
grid = sns.FacetGrid(train, row="Survived", col='Embarked', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', ci=None, palette='deep')
grid.add_legend()
```

    /opt/conda/lib/python3.6/site-packages/seaborn/axisgrid.py:703: UserWarning: Using the barplot function without specifying `order` is likely to produce an incorrect plot.
      warnings.warn(warning)





    <seaborn.axisgrid.FacetGrid at 0x7fc554845fd0>




![output_32_2](https://i.imgur.com/HtvjFap.png)



```python
grid = sns.FacetGrid(train, col="Pclass", hue="Survived", size=3)
grid.map(plt.hist, "Fare", alpha=.5)
plt.xlim(0,300)
grid.add_legend();
```


![output_33_0](https://i.imgur.com/51hYpqj.png)



```python
grid = sns.FacetGrid(train, col="Pclass", row="Embarked", hue="Survived", size=3)
grid.map(plt.hist, "Fare", alpha=.5)
plt.xlim(0,300)
plt.ylim(0,100)
grid.add_legend();
```


![output_34_0](https://i.imgur.com/53e9Ojt.png)



```python
grid = sns.FacetGrid(train, col="Pclass", hue="Survived", size=2.5, aspect=1.3)
grid.map(plt.hist, "Age", bins=20, alpha=0.5)
grid.add_legend();
```


![output_35_0](https://i.imgur.com/cYNZ76z.png)



```python
grid = sns.FacetGrid(train, col="Pclass", hue="Survived", size=3)
grid.map(plt.scatter, "Fare", "Age", alpha=.5)
plt.xlim(0,300)
grid.add_legend();
```


![output_36_0](https://i.imgur.com/y98NCPk.png)


## 3. <span style="color:red"> M</span>odify

---

Explore단계에서 알아본 결과,

Pclass, Sex, Age, SibSp, Parch, Fare, Embarked 변수의 경우 실제로 종속변수인 Survived에 영향을 미쳤다.

그리고 몇몇 변수들끼리 서로 강한 상관관계를 가지고 있어서 변수들 간의 결합(차원 축소)이 가능할 것으로 예상한다.

이를 확인하기 위해 가장 정확한 방법은 PCA 등의 데이터를 통한 다중공선성을 파악하는 것이 가장 좋겠지만, 여기서는 시간이 없기 때문에 직접 결합을 해보고 차이가 있는지 확인해보도록 하겠다.

그리고 Ticket과 Cabin 변수는 영향력을 찾기 어려웠기 때문에 제거할 것이며,

Age, Fare와 같은 numerical variable은 머신러닝을 위해 임의의 구간으로 잘라서 categorical 변수로 변환할 것이다.


```python
print('Ticket unique value  : {0}'.format(len(train['Ticket'].unique())))
print('Cabin unique value : {0}   Cabin null값 : {1}'
      .format(len(train['Cabin'].unique()), sum(train['Cabin'].isnull())))
```

    Ticket unique value  : 681
    Cabin unique value : 148   Cabin null값 : 687



```python
# 영향력이 없는 변수인 Ticket, Cabin 변수 삭제
train_df = train.drop(["Ticket", "Cabin"], axis=1)
test_df = test.drop(["Ticket", "Cabin"], axis=1)
combine = [train_df, test_df]

print("Before : ", train.shape, test.shape)
print("After  : ", train_df.shape, test_df.shape)
```

    Before :  (891, 12) (418, 11)
    After  :  (891, 10) (418, 9)



```python
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]*)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Sex</th>
      <th>female</th>
      <th>male</th>
    </tr>
    <tr>
      <th>Title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Capt</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Col</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Countess</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Don</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Dr</th>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Jonkheer</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Lady</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Major</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Master</th>
      <td>0</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Miss</th>
      <td>182</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mlle</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mme</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mr</th>
      <td>0</td>
      <td>517</td>
    </tr>
    <tr>
      <th>Mrs</th>
      <td>125</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Ms</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Rev</th>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Sir</th>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
rarelist = []
for a in set(train_df['Title']):
    if list(train_df['Title']).count(a)<10:
        rarelist.append(a)
rarelist
```




    ['Countess',
     'Rev',
     'Major',
     'Don',
     'Mlle',
     'Sir',
     'Ms',
     'Lady',
     'Dr',
     'Col',
     'Mme',
     'Capt',
     'Jonkheer']




```python
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')
    dataset['Title'] = dataset['Title'].replace(rarelist,'Rare')

train_df[['Title','Survived']].groupby(['Title'], as_index=False).mean()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Master</td>
      <td>0.575000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Miss</td>
      <td>0.702703</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mr</td>
      <td>0.156673</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mrs</td>
      <td>0.793651</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Rare</td>
      <td>0.347826</td>
    </tr>
  </tbody>
</table>
</div>




```python
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    dataset['Title'].astype(int)
train_df['Title'].head()
```




    0    1
    1    3
    2    2
    3    3
    4    1
    Name: Title, dtype: int64




```python
train_df = train_df.drop(['Name','PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df,test_df]
train_df.shape, test_df.shape
```




    ((891, 9), (418, 9))



위는 정규표현식을 사용하여 Name에서 Mr.와 같은 성별이나 결혼 유무 등을 나타내는 호칭을 뽑아내고,

그 단어에 따라 생존률에 차이가 있는지 확인하는 방법이다.

분석을 진행할 때 전혀 생각하지 못 했었는데, 내가 참고한 커널에서는 이러한 방법을 이용하여 내가 분석했다면 제거하였을 Name 변수를 활용하여 예측력을 높였다.


```python
for dataset in combine:
    dataset.Sex = dataset.Sex.map({'female':1,'male':0}).astype(int)

train_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
for dataset in combine:
    for i in range(2):
        for j in range(3):
            temp = (dataset.Sex == i) & (dataset['Pclass'] == j+1)
            dataset.loc[temp,'Age'] = dataset.loc[temp,'Age']. \
            where(pd.notnull(dataset[temp].Age),(dataset[temp].Age.median()/0.5 + 0.5) * 0.5)
    dataset.Age = dataset.Age.astype(int)

train_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>22</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>38</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>35</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>35</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby('AgeBand', as_index=False).mean().sort_values(by='AgeBand', ascending=True)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AgeBand</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(-0.08, 16.0]</td>
      <td>0.550000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(16.0, 32.0]</td>
      <td>0.337374</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(32.0, 48.0]</td>
      <td>0.412037</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(48.0, 64.0]</td>
      <td>0.434783</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(64.0, 80.0]</td>
      <td>0.090909</td>
    </tr>
  </tbody>
</table>
</div>




```python
for dataset in combine:
    dataset.loc[dataset.Age <= 16, 'Age'] = 0
    dataset.loc[(dataset.Age > 16) & (dataset.Age <= 32), 'Age'] = 1
    dataset.loc[(dataset.Age > 32) & (dataset.Age <= 48), 'Age'] = 2
    dataset.loc[(dataset.Age > 48) & (dataset.Age <= 64), 'Age'] = 3
    dataset.loc[dataset.Age > 64, 'Age'] = 4

train_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Title</th>
      <th>AgeBand</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>1</td>
      <td>(16.0, 32.0]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>3</td>
      <td>(32.0, 48.0]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>2</td>
      <td>(16.0, 32.0]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>3</td>
      <td>(32.0, 48.0]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>1</td>
      <td>(32.0, 48.0]</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df = train_df.drop('AgeBand', axis=1)
combine = [train_df, test_df]
train_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize','Survived']].groupby('FamilySize', as_index=False).mean().sort_values(by='Survived',ascending=False)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FamilySize</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.724138</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.578431</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.552795</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.303538</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0.136364</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>11</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
for dataset in combine:
    dataset['isAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'isAlone'] = 1

train_df[['isAlone', 'Survived']].groupby('isAlone', as_index = False).mean()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>isAlone</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.505650</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.303538</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df = train_df.drop(['Parch', 'SibSp'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp'], axis=1)
combine = [train_df, test_df]

train_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Title</th>
      <th>FamilySize</th>
      <th>isAlone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>7.2500</td>
      <td>S</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>71.2833</td>
      <td>C</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>7.9250</td>
      <td>S</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>53.1000</td>
      <td>S</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>8.0500</td>
      <td>S</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
for dataset in combine:
    dataset['Age*Pclass'] = dataset.Age * dataset.Pclass

train_df[['Age*Pclass', 'Survived']].groupby('Age*Pclass', as_index=False).mean().sort_values(by='Survived', ascending = False)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age*Pclass</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.728814</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.550000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.520408</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.415094</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.277487</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0.149425</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9</td>
      <td>0.111111</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>12</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df[['Age*Pclass','Survived']].groupby('Age*Pclass').count()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
    </tr>
    <tr>
      <th>Age*Pclass</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>59</td>
    </tr>
    <tr>
      <th>2</th>
      <td>196</td>
    </tr>
    <tr>
      <th>3</th>
      <td>382</td>
    </tr>
    <tr>
      <th>4</th>
      <td>53</td>
    </tr>
    <tr>
      <th>6</th>
      <td>87</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(train_df.Embarked.dropna().mode()[0])

train_df[['Embarked','Survived']].groupby('Embarked', as_index=False).mean().sort_values(by='Survived', ascending=False)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Embarked</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>C</td>
      <td>0.553571</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Q</td>
      <td>0.389610</td>
    </tr>
    <tr>
      <th>2</th>
      <td>S</td>
      <td>0.339009</td>
    </tr>
  </tbody>
</table>
</div>




```python
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

train_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Title</th>
      <th>FamilySize</th>
      <th>isAlone</th>
      <th>Age*Pclass</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>7.2500</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>71.2833</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>7.9250</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>53.1000</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>8.0500</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(train_df.Fare.isnull().sum(), test_df.Fare.isnull().sum())
```

    0 1



```python
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Title</th>
      <th>FamilySize</th>
      <th>isAlone</th>
      <th>Age*Pclass</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>7.8292</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>7.0000</td>
      <td>0</td>
      <td>3.0</td>
      <td>2</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>9.6875</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>8.6625</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>12.2875</td>
      <td>0</td>
      <td>3.0</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_df['Fareband'] = pd.qcut(train_df['Fare'], 4)
train_df[['Fareband','Survived']].groupby('Fareband').mean().sort_values(by='Survived', ascending=False)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
    </tr>
    <tr>
      <th>Fareband</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(31.0, 512.329]</th>
      <td>0.581081</td>
    </tr>
    <tr>
      <th>(14.454, 31.0]</th>
      <td>0.454955</td>
    </tr>
    <tr>
      <th>(7.91, 14.454]</th>
      <td>0.303571</td>
    </tr>
    <tr>
      <th>(-0.001, 7.91]</th>
      <td>0.197309</td>
    </tr>
  </tbody>
</table>
</div>




```python
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31.0), 'Fare'] = 2
    dataset.loc[ dataset['Fare'] > 31.0, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
train_df = train_df.drop('Fareband', axis=1)
combine = [train_df, test_df]

train_df.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Title</th>
      <th>FamilySize</th>
      <th>isAlone</th>
      <th>Age*Pclass</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_df.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Title</th>
      <th>FamilySize</th>
      <th>isAlone</th>
      <th>Age*Pclass</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3.0</td>
      <td>2</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>3.0</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>897</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>898</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>899</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>900</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>901</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1.0</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_df.isnull().sum()
```




    PassengerId    0
    Pclass         0
    Sex            0
    Age            0
    Fare           0
    Embarked       0
    Title          0
    FamilySize     0
    isAlone        0
    Age*Pclass     0
    dtype: int64



## 4. <span style="color:red"> M</span>odeling & <span style="color:red">A</span>ssess

---

알고리즘을 결정하기 전에 문제를 분류하자면

1. 종속변수, 즉 label이 있는 지도학습이다.
2. 종속변수가 categorical variable인 classfication 문제이다.
3. 독립변수들의 영향력은 앞서 나온 시각화나 상관관계 등을 통해 충분히 추정할 수 있다.
4. 독립변수들은 모두 categorical 변수이다.
5. 독립변수들은 모두 전처리되어있다.

우선 참고한 커널에서 진행한 대로 대표적인 지도학습의 분류 기법 알고리즘을 적용해보자.


```python
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df['Correlation'] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>Correlation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Sex</td>
      <td>2.215671</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Title</td>
      <td>0.476947</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fare</td>
      <td>0.255126</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Embarked</td>
      <td>0.211899</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Age*Pclass</td>
      <td>-0.157424</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Age</td>
      <td>-0.325558</td>
    </tr>
    <tr>
      <th>7</th>
      <td>isAlone</td>
      <td>-0.353892</td>
    </tr>
    <tr>
      <th>6</th>
      <td>FamilySize</td>
      <td>-0.458992</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Pclass</td>
      <td>-0.697766</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
```




    83.840000000000003




```python
# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
```




    84.180000000000007




```python
# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian
```




    80.359999999999999




```python
# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron
```

    /opt/conda/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:128: FutureWarning: max_iter and tol parameters have been added in <class 'sklearn.linear_model.perceptron.Perceptron'> in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      "and default tol will be 1e-3." % type(self), FutureWarning)





    81.140000000000001




```python
# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train,Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc
```




    81.480000000000004




```python
# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd
```

    /opt/conda/lib/python3.6/site-packages/sklearn/linear_model/stochastic_gradient.py:128: FutureWarning: max_iter and tol parameters have been added in <class 'sklearn.linear_model.stochastic_gradient.SGDClassifier'> in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      "and default tol will be 1e-3." % type(self), FutureWarning)





    75.980000000000004




```python
# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
```




    88.549999999999997




```python
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
```




    88.549999999999997




```python
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent', 'Linear SVC',
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Random Forest</td>
      <td>88.55</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Decision Tree</td>
      <td>88.55</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KNN</td>
      <td>84.18</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Support Vector Machines</td>
      <td>83.84</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Logistic Regression</td>
      <td>81.59</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Linear SVC</td>
      <td>81.48</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Perceptron</td>
      <td>81.14</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Naive Bayes</td>
      <td>80.36</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Stochastic Gradient Decent</td>
      <td>75.98</td>
    </tr>
  </tbody>
</table>
</div>



이런 문제의 경우 해석력은 이미 충분히 지녔으며 변수들은 결측치가 없는 등 전처리가 이루어져있기에 해석력보다는 가장 예측력이 높은 머신러닝 기법을 사용하면 되지 않을까?

라고 생각해봤지만 결과적으로 SVM이나 KNN같은 기법이 아니라 Decision Tree가 가장 성과 측정치가 높다.

모든 변수가 10이하의 categorical 변수이기 때문에 Decision Tree로 분류하는 것이 가장 좋은 것일까?

아직은 잘 모르겠다...

여기까지가 참고한 커널의 내용이다.

Decision Tree와 같이 가장 성과치가 높았던 Random Forest의 내용을 좀 더 알아보고 끝내도록 하자.

[Random Forest 기본 원리 @deepvalidation](https://medium.com/@deepvalidation/title-3b0e263605de)
