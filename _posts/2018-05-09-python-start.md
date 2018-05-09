---
layout: post
title: "Python to Data Analysis"
subtitle: "파이썬 기초"
categories: study
tags: lecture
comments: true
---

<!-- MDTOC maxdepth:3 firsth1:1 numbering:0 flatten:0 bullets:0 updateOnSave:1 -->

[파이썬 기초](#파이썬-기초)   
&emsp;[1. 파이썬이란?](#1-파이썬이란)   
&emsp;[2. 파이썬의 특징](#2-파이썬의-특징)   
&emsp;[3. 파이썬의 자료구조](#3-파이썬의-자료구조)   
&emsp;&emsp;[3.1 숫자형](#31-숫자형)   
&emsp;&emsp;[3.2 문자열](#32-문자열)   
&emsp;&emsp;[3.3 리스트](#33-리스트)   
&emsp;&emsp;[3.4 튜플](#34-튜플)   
&emsp;&emsp;[3.5 딕셔너리](#35-딕셔너리)   
&emsp;&emsp;[3.6 집합](#36-집합)   
&emsp;&emsp;[3.7 numpy](#37-numpy)   
&emsp;&emsp;[3.8 pandas](#38-pandas)   
&emsp;[4. 조건문과 반복문](#4-조건문과-반복문)   
&emsp;[5. 참고 및 출처](#5-참고-및-출처)   

<!-- /MDTOC -->


# 파이썬 기초

---

## 1. 파이썬이란?

1989년, 개발자 귀도 반 로섬이 크리스마스 주에 연구실 문을 닫아서 심심해서 만들기 시작한 언어.

인터프리터 언어이며 객체 지향 프로그래밍을 추구한다.

---

## 2. 파이썬의 특징

1. 아름다운 것이 추한 것보다 낫다.

> 다른 언어들은 각자의 코딩 스타일에 맞춰 코드를 작성하지만 파이썬은 가장 아름답다고 생각되는 스타일(강제 개행 등)로 강제한다.

2. 명시적인 것이 암시적인 것보다 낫다.

> 가독성을 중요하게 생각하여, 모든 코드는 의사코드와 비슷한 형태로 작성된다.

3. 간결한 것이 복잡한 것보다 낫다.

> 모든 동작은 간결한 명령문으로 동작하며 기저의 다양한 동작은 파이썬이 대신하여 처리해준다.

**하지만 이 특성들로 인하여 파이썬은 다른 low level Language(C언어, JAVA)에 비해 훨씬 느리다.**

---

## 3. 파이썬의 자료구조

---

### 3.1 숫자형


```python
# 변수 선언

a = 15
print("a의 값은 {}이고, a의 타입은 {}입니다.".format(a,type(a)))

b = 15.3
print("b의 값은 {0}이고, b의 타입은 {1}입니다.".format(b,type(b)))

c = a + b
print("c의 값은 {1}이고, c의 타입은 {0}입니다.".format(type(c),c))

print()

print(int(b))
print(float(a))
```

    a의 값은 15이고, a의 타입은 <class 'int'>입니다.
    b의 값은 15.3이고, b의 타입은 <class 'float'>입니다.
    c의 값은 30.3이고, c의 타입은 <class 'float'>입니다.

    15
    15.0


변수의 타입은 따로 선언이 필요 없으며 넣은 데이터에 따라 자동적으로 타입을 부여한다.


```python
# 숫자형의 타입

a = 13 # 정수형
print(type(a))
a = 12.1423 # 실수형
print(type(a))
a = 0o123 # 8진수
print(a)
a = 0xFF2 # 16진수
print(a)
```

    <class 'int'>
    <class 'float'>
    83
    4082


한 변수에도 다른 데이터 타입을 선언할 수 있다.


```python
# 숫자형의 연산

a = 5
b = 3
print(a + b) # 더하기
print(a * b) # 빼기
print(a / b) # 나누기
print(a % b) # 나머지
print(a // b) # 나머지를 버린 나누기
print(a ** b) # 제곱
```

    8
    15
    1.6666666666666667
    2
    1
    125


---

### 3.2 문자열


```python
# 문자열의 연산 (다른 언어에서는 대부분 불가능)
# 파이썬에서는 모든 문자는 문자열 자료구조

head = "Python is"
tail = " fun"
print(head + tail) # 출력값은 아니며 중복하여 화면에 나타낼 수 있음
print(head * 2)
head + tail # 출력값 자체이며 마지막 출력값만 저장되어 화면에 나타남
head * 2
```

    Python is fun
    Python isPython is





    'Python isPython is'




```python
# 문자열 연산 응용

print("-" * 50)
print(head + tail)
print("-" * 50)
```

    --------------------------------------------------
    Python is fun
    --------------------------------------------------



```python
# 문자열 슬라이싱

py = "0123456789"
print(len(py)) # 길이 출력
print(py[0]) # 파이썬은 0부터 시작, 길이가 13일 경우 0부터 12까지 인덱스 존재
print(py[2:8]) # 2번째 "이상" 8번째 "미만"까지 출력
print(py[-5:-1]) # 뒤에서 5번째 이상 뒤에서 1번째 미만까지 출력
print(py[4:-1]) # 4번째 이상 뒤에서 1번째 미만까지 출력
print(py[4:]) # 4번째 이상 모두 출력
```

    10
    0
    234567
    5678
    45678
    456789



```python
# 슬라이싱 응용

temp = "Pithon"

# Python으로 변경하고 싶으면?

temp = temp[0] + "y" + temp[2:]
temp
```




    'Python'



---

### 3.3 리스트


```python
# 리스트 선언

a = [] # a = list() 동일
print(a)

a = [1,"a","3",[5,3,[2],4]] # 리스트의 내부에는 모든 자료구조가 다 들어갈 수 있다.
print(a)
```

    []
    [1, 'a', '3', [5, 3, [2], 4]]



```python
# 리스트 슬라이싱(문자열과 동일)

print(a[0]) # 단일 슬라이싱
print(a[0:3]) # 다중 슬라이싱
print(a[3]) # list 안의 list
print(a[3][2]) # list 안의 list 안의 list
print(a[3][2][0]) # ...
```

    1
    [1, 'a', '3']
    [5, 3, [2], 4]
    [2]
    2



```python
# 리스트에서의 연산

a = [1,2,3]
b = [4,5,6]
print(a+b) # 더하기
print(a*2) # 곱하기

a[1] = [10] # 리스트 단일 값 수정
print(a)
a[1:2] = [11,12] # 리스트 다중 값 수정
print(a)

# a[1]과 a[1:2]는 동일한 범위 인덱싱이지만 결과가 다르다.
# a[1]은 리스트의 첫 번째 요소 값만을 반환하지만, a[2]은 리스트를 반환
```

    [1, 2, 3, 4, 5, 6]
    [1, 2, 3, 1, 2, 3]
    [1, [10], 3]
    [1, 11, 12, 3]



```python
# 리스트 내부 함수들
# 객체 지향 프로그래밍의 특징

a = [3,2]

a.append(1) # 추가
print("append = ", a)

a.sort() # 정렬
print("sort = ", a)

a.sort?
# 역순으로 정렬하려면 어떻게 해야할까?

a.append(2)
a.reverse() # 뒤집기
print("reverse = ", a)

print("index = ", a.index(3), a.index(2), a.index(1)) # 값의 위치 찾기
# 중복된 값이 있으면 가장 앞에 있는 값만 찾아준다.

a.insert(2,15) # 원하는 위치에 삽입
print("insert = ", a)

print("2의 count = ", a.count(2)) # 리스트 내에서 특정한 값의 개수 세기

a.remove(2) # 값을 찾아서 지우기, 중복된 값이 있으면 가장 앞에 있는 값만 삭제
print("remove = ", a)

print("pop = ", a.pop()) # 마지막 요소의 값을 삭제하고 반환한다.
print("pop = ", a.pop(1)) # 두 번째 요소의 값을 삭제하고 반환한다.
print(a)

b = [5,2]
a.extend(b) # 리스트 더하기. a += b 와 동일함
print("extend = ", a)

del a[1] # 특정한 위치의 값 삭제. pop과 동일하지만 값을 반환하지 않는다는 차이 존재
print(a)
```

    append =  [3, 2, 1]
    sort =  [1, 2, 3]
    reverse =  [2, 3, 2, 1]
    index =  1 0 3
    insert =  [2, 3, 15, 2, 1]
    2의 count =  2
    remove =  [3, 15, 2, 1]
    pop =  1
    pop =  15
    [3, 2]
    extend =  [3, 2, 5, 2]
    [3, 5, 2]


---

### 3.4 튜플

튜플은 리스트와 동일하지만, 처음 선언 이후 수정 불가
연산 속도가 리스트보다 훨씬 빠르며 고정된 값을 이용할 때, 값을 실수로 변경하면 안 될때 많이 사용한다.


```python
# 튜플의 선언

a = () # a = tuple()
print(type(a))

temp = [3,2,1]
a = tuple(temp)
print(a) # 변수 자체를 덮어씌우는 것은 가능하지만 리스트의 연산, 함수들 사용 불가
```

    <class 'tuple'>
    (3, 2, 1)


---

### 3.5 딕셔너리

사전과 같이 {key:value} 형식으로 특정한 key를 사용하여 value 값을 반환받을 수 있다.


```python
# 딕셔너리 선언

a = {"a":1, 2:[1,2,3], "c":"d", "boil":100} # a.dict()
print(type(a))
print(a)
print(a["boil"])
print(a[2])

# 딕셔너리 요소 추가
a["abcd"] = 1234
print(a)
```

    <class 'dict'>
    {'a': 1, 2: [1, 2, 3], 'c': 'd', 'boil': 100}
    100
    [1, 2, 3]
    {'a': 1, 2: [1, 2, 3], 'c': 'd', 'boil': 100, 'abcd': 1234}



```python
# 딕셔너리 함수들

print(a.keys()) # key 값의 요소들 반환
print(a.values()) # value 값의 요소들 반환
print(a.items()) # key와 value의 매칭을 튜플로 각각 반환함
'boil' in a # 'boil'이라는 key 값이 a 딕셔너리 변수 안에 존재하는지 확인
```

    dict_keys(['a', 2, 'c', 'boil', 'abcd'])
    dict_values([1, [1, 2, 3], 'd', 100, 1234])
    dict_items([('a', 1), (2, [1, 2, 3]), ('c', 'd'), ('boil', 100), ('abcd', 1234)])





    True



---

### 3.6 집합

중복된 값을 넣을 수 없으며 값에 순서가 없다.


```python
# 집합 선언

a = {3,3,3,6,7,8} # a.set()
print(type(a))
print(a)
```

    <class 'set'>
    {8, 3, 6, 7}


---

### 3.7 numpy

데이터 분석을 위한 모듈이며 해당 모듈을 설치하여 불러와야한다.

numpy는 과학 계산을 위한 라이브러리로서 다차원 배열을 처리하는데 필요한 여러 유용한 기능을 제공하고 있다.

numpy는 계산 속도를 향상시키기 위해 C 등의 저급 언어를 사용하여 연산하기 때문에 기존 파이썬 연산보다 훨씬 빠르며 벡터 연산을 스칼라 연산처럼 사용 가능하다.


```python
# $ pip install numpy  -  numpy를 설치하기 위한 명령어

import numpy as np # numpy 모듈을 np라는 명령어로 불러와라.

list1 = [1, 2, 3, 4]
a = np.array(list1)
print(a.shape, end="\n\n") # (4, )

b_li = [[1,2,3],[4,5,6],[7,8,9]] # list형태
print(b_li, end="\n\n")

b = np.array(b_li, dtype='int64')
print(b) # array 형태
print(b.shape) # (3, 3)
print(b[1,1])  # 5
print()

print(b.dtype) # 데이터 타입 확인
b = np.array(b, dtype=np.float32)
print(b.dtype)
```

    (4,)

    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    (3, 3)
    5

    int64
    float32



```python
a = np.array(range(20)).reshape((4,5))
print(a)
# 범위 0~19까지의 수를 array 형태로 만드는데, 4행 5열의 형태로 만들어라
```

    [[ 0  1  2  3  4]
     [ 5  6  7  8  9]
     [10 11 12 13 14]
     [15 16 17 18 19]]



```python
lst = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
arr = np.array(lst)

# 슬라이스
a = arr[0:2, 0:2]
print(a)

a = arr[1:, 1:]
print(a)
```

    [[1 2]
     [4 5]]
    [[5 6]
     [8 9]]



```python
# 벡터 연산 ★★★★★

a = np.arange(0,10)
print(a)
b = np.arange(10,20)
print(b)
print("더하기 = ", a + b)
print("곱하기 = ", a * 2)
print("제곱근 = ", np.sqrt(a))
print("비교연산 = ", a < b)
```

    [0 1 2 3 4 5 6 7 8 9]
    [10 11 12 13 14 15 16 17 18 19]
    더하기 =  [10 12 14 16 18 20 22 24 26 28]
    곱하기 =  [ 0  2  4  6  8 10 12 14 16 18]
    제곱근 =  [ 0.          1.          1.41421356  1.73205081  2.          2.23606798
      2.44948974  2.64575131  2.82842712  3.        ]
    비교연산 =  [ True  True  True  True  True  True  True  True  True  True]



```python
arr = np.arange(10)
print(arr) # [0 1 2 3 4 5 6 7 8 9]
print(arr[5]) # 5
print(arr[5:8]) # [5 6 7]
arr[5:8] = 12 # [0 1 2 3 4 12 12 12 8 9]
#5이상 8미만의 인덱스를 모두 12로 변경!
print(arr)
```

    [0 1 2 3 4 5 6 7 8 9]
    5
    [5 6 7]
    [ 0  1  2  3  4 12 12 12  8  9]



```python
# 리스트와 numpy의 array 자료구조 속도 비교

a = range(10000000)
%timeit sum(a)
b = np.array(a)
%timeit np.sum(b)
```

    196 ms ± 9.15 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
    7.81 ms ± 104 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)



```python
%%timeit
for i in range(10000000):
    a[i] + 2
```

    827 ms ± 87.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)



```python
%timeit b + 2
```

    27 ms ± 1.79 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)


numpy 내부 함수는 엄청나게 많으며 하나씩 다룰 수는 없기 때문에 필요한 기능들을 구글링하여 찾는 것이 효율적이다.

모든 함수는 np.(tab)을 누르게 되면 사용 가능한 모든 함수가 뜬다.

또한 np.arrange? 와 같은 help 함수를 사용하여 참고하는 것이 필수적이다.

---

### 3.8 pandas

NumPy 기반에서 개발되어 NumPy를 사용하는 애플리케이션에서 쉽게 사용할 수 있다.

기본적으로 csv(엑셀 등) 파일을 다루기 위한 데이터 분석 모듈.

이차원 행렬을 가장 보기 좋게 그려줄 뿐만 아니라 수정, 제거, 결측치 채우기 등 많은 도구를 제공한다.


```python
# Series
# Series는 일련의 객체를 담을 수 있는 1차원 배열 같은 자료 구조
# (어떤 NumPy 자료형이라도 담을 수 있다.)

import pandas as pd # pandas 모듈을 pd라는 이름으로 불러온다.
a = pd.Series([-4,3,2,-13], index=['a','b','c','d'])
a
```




    a    -4
    b     3
    c     2
    d   -13
    dtype: int64




```python
#Series 데이터 접근

print(a.values)
print(a.index)
```

    [ -4   3   2 -13]
    Index(['a', 'b', 'c', 'd'], dtype='object')



```python
# Series 특정 값 가져오기
print(a[a>0])
```

    b    3.0
    c    2.0
    dtype: float64



```python
a['a'] = np.NAN # Not a Number, 즉 null 값으로 변경
print(pd.isnull(a)) # pandas에 내장되어 있는 함수를 이용하여 a의 null값 확인
print()
print(a.isnull()) # Series 객체에 상속되는 함수를 이용하여 a의 null값 확인
```

    a     True
    b    False
    c    False
    d    False
    dtype: bool

    a     True
    b    False
    c    False
    d    False
    dtype: bool



```python
# DataFrame ★★★★★

data = {'state' : ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year' : [2000, 2001, 2002, 2001, 2002],
        'pop' : [1.5, 1.7, 3.6, 2.4, 2.9]}

df = pd.DataFrame(data)
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pop</th>
      <th>state</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.5</td>
      <td>Ohio</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.7</td>
      <td>Ohio</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.6</td>
      <td>Ohio</td>
      <td>2002</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.4</td>
      <td>Nevada</td>
      <td>2001</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.9</td>
      <td>Nevada</td>
      <td>2002</td>
    </tr>
  </tbody>
</table>
</div>




```python
from pandas import DataFrame #pandas 모듈의 DataFrame 클래스를 가져온다.
# 이 방법을 통해 pd.DataFrame이 아닌 DataFrame으로 DataFrame을 만들 수도 있다.
DataFrame(data, columns=['year', 'state', 'pop'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>state</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2000</td>
      <td>Ohio</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2001</td>
      <td>Ohio</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2002</td>
      <td>Ohio</td>
      <td>3.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2001</td>
      <td>Nevada</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2002</td>
      <td>Nevada</td>
      <td>2.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                index=['일','이','삼','사','오']) # 한글도 가능은 하지만...
df2[df2['year']>=2001] # year 열이 2001년 이상인 행들만 불러온다.
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>state</th>
      <th>pop</th>
      <th>debt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>이</th>
      <td>2001</td>
      <td>Ohio</td>
      <td>1.7</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>삼</th>
      <td>2002</td>
      <td>Ohio</td>
      <td>3.6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>사</th>
      <td>2001</td>
      <td>Nevada</td>
      <td>2.4</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>오</th>
      <td>2002</td>
      <td>Nevada</td>
      <td>2.9</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df['year'][0:3], end='\n\n') # year 열의 0이상 3미만 행을 불러온다.

print(df[['year','state']],[[0,2]])
# 특정한 열 또는 행만 불러오고 싶을 때 리스트로 묶어준다.
```

    0    2000
    1    2001
    2    2002
    Name: year, dtype: int64

       year   state
    0  2000    Ohio
    1  2001    Ohio
    2  2002    Ohio
    3  2001  Nevada
    4  2002  Nevada [[0, 2]]



```python
# DataFrame 인덱싱

print(df.iloc[:3,1:]) # integer-location 숫자를 이용하여 행, 열 순서로 인덱싱
print()

print(df.loc[df['year']>2000, 'state']) # 문자 또는 진리값(bool)을 이용하여 인덱싱
print()

print(df.ix[:3,'state']) # 숫자, 문자, 진리값 모두 혼합하여 사용 가능
```

      state  year
    0  Ohio  2000
    1  Ohio  2001
    2  Ohio  2002

    1      Ohio
    2      Ohio
    3    Nevada
    4    Nevada
    Name: state, dtype: object

    0      Ohio
    1      Ohio
    2      Ohio
    3    Nevada
    Name: state, dtype: object


---

## 4. 조건문과 반복문


```python
# if 조건문

GoDong = '학교'
JoDong = '예비군'

if GoDong=='학교' and JoDong=='학교':
    print('만남')
elif GoDong=='예비군' and JoDong=='예비군':
    print('만날 수도 있다')
else:
    print('만나지 못한 두 사람')
```

    못 만남



```python
# while 반복문

i=0
while i<10:
    print(' ' * (10-i), end='')
    print('*' * (2*i-1))
    i+=1
```


             *
            ***
           *****
          *******
         *********
        ***********
       *************
      ***************
     *****************



```python
# for 반복문

names = ['고동','승현','혜원','주영','혜령','영훈','조동','민영']
couple = []
for name in names:
    if name.endswith('동'):
        couple.append(name)
couple
```




    ['고동', '조동']



---

## 5. 참고 및 출처

numpy와 pandas는 데이터 분석에 있어서 가장 중요한 것들이면서도 많은 기능들이 존재한다.

한 시간에 모든 것을 배울 수는 없기 때문에 필수적이면서도 간단한 것들 위주로 정리하였지만, 필요에 따라서 구글링을 하면서 배우기를 권장한다.

또한 이외에도 matplotlib와 scikit-learn, scipy 정도는 데이터 분석에 있어서 필수적인 라이브러리들이기 때문에 공부해두는 것도 좋다.

정리하는 데에 참고한 사이트들을 보며 공부하는 것도 추천한다.

https://wikidocs.net/book/1 (점프투파이썬, 데이터 분석이 아닌 기본적인 파이썬 사용법과 정규표현식 등을 설명)

https://github.com/wesm/pydata-book (numpy, pandas, matplotlib, sklearn 등 책 예제를 올려둔 사이트)

https://datascienceschool.net/view-notebook/d0b1637803754bb083b5722c9f2209d0/
(matplotlib를 공부하기 좋은 링크)

https://github.com/brenden17/blog/blob/master/post/ms.scikit-learn.v.md
(sklearn 기초 예제를 통해 배울 수 있는 링크)

---
