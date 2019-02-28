---
layout: post
title: "[게임데이터 분석 #1] League Of Legends(lol) 바텀 듀오 티어 계산 "
subtitle: "원딜과 바텀 듀오, 누가 좋을까?"
categories: study
tags: project
comments: true
---

# 분석의 목적

---

친구랑 같이 롤을 즐길 때에는 봇 듀오로 같이 가는 경우가 많다.

하지만 나는 일반 게임이나 랭크 게임에서는 오로지 탑만 가는 진정한 탑 솔로이므로 원딜과 서포터의 어떤 조합이 좋은지 잘 모른다.

그래서 이번 분석에서 어떤 원딜, 서폿 조합이 가장 좋은지 데이터를 통해 알아내고자 한다.

이외에도 각자의 플레이 스타일에 맞는 아이템 추천, 탑 또는 미드와 정글의 조합, 5인 팀 게임의 조합 등과 같이 해보고 싶은 분석은 많지만 이번 분석에서는 **원딜과 서폿의 조합 티어를 밝혀내는 것** 을 분석의 목적으로 한다.

> 계산 과정이 오래 걸리는 코드는 첫 계산 이후 주석 처리하고 저장 후 불러오는 방식으로 사용하였습니다.

## 1. 데이터 저장

---



```python
# 패키지 불러오기

import pickle # 리스트 안의 데이터프레임 형태 저장
import requests # api 요청
import json
import pandas as pd
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
import time

from skimage import io # 미니맵 처리
from sklearn.preprocessing import MinMaxScaler

%matplotlib inline
sns.set()
```

### 1-1. Champs
Google에 lol Static Data이라고 치면 페이지가 나온다.

여기서 이미지, 챔피언 등의 Static Data를 받아볼 수 있다.

버전은 동일한 페이지의 versions 항목을 참고하여 최신버전으로 받아오자.


```python
# 버전 확인 및 API Key 갱신
# 사용한 데이터는 9.3.261버전 데이터

api_key = 'Key' # Key를 갱신하여야 한다
r = requests.get('https://ddragon.leagueoflegends.com/api/versions.json') # version data 확인
current_version = r.json()[0] # 가장 최신 버전 확인
current_version
```




    '9.4.1'




```python
# Champs 데이터 받아오기

# requests 함수로 chapms 데이터 받아오기 (가장 최신 버전)
r = requests.get('http://ddragon.leagueoflegends.com/cdn/{}/data/ko_KR/champion.json'.format(current_version))
parsed_data = r.json() # 파싱
info_df = pd.DataFrame(parsed_data)
info_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>type</th>
      <th>format</th>
      <th>version</th>
      <th>data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Aatrox</th>
      <td>champion</td>
      <td>standAloneComplex</td>
      <td>9.4.1</td>
      <td>{'version': '9.4.1', 'id': 'Aatrox', 'key': '2...</td>
    </tr>
    <tr>
      <th>Ahri</th>
      <td>champion</td>
      <td>standAloneComplex</td>
      <td>9.4.1</td>
      <td>{'version': '9.4.1', 'id': 'Ahri', 'key': '103...</td>
    </tr>
    <tr>
      <th>Akali</th>
      <td>champion</td>
      <td>standAloneComplex</td>
      <td>9.4.1</td>
      <td>{'version': '9.4.1', 'id': 'Akali', 'key': '84...</td>
    </tr>
    <tr>
      <th>Alistar</th>
      <td>champion</td>
      <td>standAloneComplex</td>
      <td>9.4.1</td>
      <td>{'version': '9.4.1', 'id': 'Alistar', 'key': '...</td>
    </tr>
    <tr>
      <th>Amumu</th>
      <td>champion</td>
      <td>standAloneComplex</td>
      <td>9.4.1</td>
      <td>{'version': '9.4.1', 'id': 'Amumu', 'key': '32...</td>
    </tr>
  </tbody>
</table>
</div>



type, format, version 모두 단일값 -> 제거

data 안의 내용을 데이터프레임으로 변한해야한다.


```python
# champ_info_df의 data 값들을 데이터프레임으로 변환

# 데이터의 각 행을 시리즈 형태로 변환하여 딕셔너리에 추가
champ_dic = {}
for i, champ in enumerate(info_df.data):
    champ_dic[i] = pd.Series(champ)

# 데이터 프레임 변환 후 Transpose
champ_df = pd.DataFrame(champ_dic).T

# output : 챔피언 데이터 안에도 info와 stats가 딕셔너리 형태임
# 이 데이터들을 데이터프레임으로 변환하여 각 챔피언에 대한 변수로 추가해야 한다.

# champ_df의 info, stats의 데이터를 변수로 추가
champ_info_df = pd.DataFrame(dict(champ_df['info'])).T
champ_stats_df = pd.DataFrame(dict(champ_df['stats'])).T

# 데이터 합치기
champ_df = pd.concat([champ_df, champ_info_df], axis=1)
champ_df = pd.concat([champ_df, champ_stats_df], axis=1)
# 이번 분석에서 필요없는 데이터 제거
champ_df = champ_df.drop(['version', 'image', 'info', 'stats'], axis=1)
champ_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 143 entries, 0 to 142
    Data columns (total 31 columns):
    id                      143 non-null object
    key                     143 non-null object
    name                    143 non-null object
    title                   143 non-null object
    blurb                   143 non-null object
    tags                    143 non-null object
    partype                 143 non-null object
    attack                  143 non-null int64
    defense                 143 non-null int64
    difficulty              143 non-null int64
    magic                   143 non-null int64
    armor                   143 non-null float64
    armorperlevel           143 non-null float64
    attackdamage            143 non-null float64
    attackdamageperlevel    143 non-null float64
    attackrange             143 non-null float64
    attackspeed             143 non-null float64
    attackspeedperlevel     143 non-null float64
    crit                    143 non-null float64
    critperlevel            143 non-null float64
    hp                      143 non-null float64
    hpperlevel              143 non-null float64
    hpregen                 143 non-null float64
    hpregenperlevel         143 non-null float64
    movespeed               143 non-null float64
    mp                      143 non-null float64
    mpperlevel              143 non-null float64
    mpregen                 143 non-null float64
    mpregenperlevel         143 non-null float64
    spellblock              143 non-null float64
    spellblockperlevel      143 non-null float64
    dtypes: float64(20), int64(4), object(7)
    memory usage: 40.8+ KB


### 1-2. League

최신 게임들을 받아와서 데이터를 분석해야하는데, 최근 게임 자체를 다 받아오는 API는 없는 듯하다.

그래서 그랜드마스터, 마스터 등급 리그의 소환사 데이터를 받아와서, 각 소환사 ID의 데이터를 받아오자.

아무래도 티어가 높을수록 데이터의 신뢰도는 높을 것이다.


```python
# API에 Key를 매일 갱신하여 받아와야한다.
# API의 요청 제한은 2분에 100개인 듯하다. 언제 다 받아서 분석하지...

# api 요청

api_url_grandmaster = 'https://kr.api.riotgames.com/lol/league/v4/grandmasterleagues/by-queue/RANKED_SOLO_5x5?api_key=' + api_key
api_url_master = 'https://kr.api.riotgames.com/lol/league/v4/masterleagues/by-queue/RANKED_SOLO_5x5?api_key=' + api_key
# 그랜드마스터, 마스터 데이터 가져와서 데이터프레임으로 변환
r = requests.get(api_url_grandmaster)
league_df = pd.DataFrame(r.json())
r = requests.get(api_url_master)
league_df = pd.concat([league_df, pd.DataFrame(r.json())], axis=0)

# entries 데이터프레임으로 변환하여 추가
league_df.reset_index(inplace=True)
league_entries_df = pd.DataFrame(dict(league_df['entries'])).T
league_df = pd.concat([league_df, league_entries_df], axis=1)

league_df = league_df.drop(['index', 'queue', 'name', 'leagueId', 'entries', 'rank'], axis=1)
league_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1592 entries, 0 to 1591
    Data columns (total 10 columns):
    tier            1592 non-null object
    freshBlood      1592 non-null object
    hotStreak       1592 non-null object
    inactive        1592 non-null object
    leaguePoints    1592 non-null object
    losses          1592 non-null object
    summonerId      1592 non-null object
    summonerName    1592 non-null object
    veteran         1592 non-null object
    wins            1592 non-null object
    dtypes: object(10)
    memory usage: 124.5+ KB

### 1-3. Summoner

Summoner 데이터에 있는 AccountId를 받아와야 매치 데이터를 받아올 수 있다.


```python
# summonerId를 이용하여 accountId를 모두 받아와야 한다. 요청 제한으로 인해 오래 걸리므로 파일로 저장하여 관리
# API 요청 제한 : 1초에 20, 2분에 100

# league_df['account_id'] = np.nan # account_id 초기화
# for i, summoner_id in enumerate(league_df['summonerId']):
#     # 각 소환사의 SummonerId와 API Key를 포함한 url을 만들고, Summoner API에서 AccountId를 가져와 채워넣는다.
#     api_url = 'https://kr.api.riotgames.com/lol/summoner/v4/summoners/' + summoner_id + '?api_key=' + api_key
#     r = requests.get(api_url)
#     while r.status_code!=200: # 요청 제한 또는 오류로 인해 정상적으로 받아오지 않는 상태라면, 5초 간 시간을 지연
#         time.sleep(5)
#         r = requests.get(api_url)
#     account_id = r.json()['accountId']
#     league_df.iloc[i, -1] = account_id

# league_df.to_csv('LeagueData.csv')

league_df = pd.read_csv('LeagueData.csv',index_col=0)
league_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tier</th>
      <th>freshBlood</th>
      <th>hotStreak</th>
      <th>inactive</th>
      <th>leaguePoints</th>
      <th>losses</th>
      <th>summonerId</th>
      <th>summonerName</th>
      <th>veteran</th>
      <th>wins</th>
      <th>account_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GRANDMASTER</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>370</td>
      <td>79</td>
      <td>소환사 아이디</td>
      <td>닉네임</td>
      <td>False</td>
      <td>61</td>
      <td>계정 아이디</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GRANDMASTER</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>431</td>
      <td>25</td>
      <td>소환사 아이디</td>
      <td>닉네임</td>
      <td>False</td>
      <td>55</td>
      <td>계정 아이디</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GRANDMASTER</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>271</td>
      <td>27</td>
      <td>소환사 아이디</td>
      <td>닉네임</td>
      <td>False</td>
      <td>35</td>
      <td>계정 아이디</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GRANDMASTER</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>260</td>
      <td>61</td>
      <td>소환사 아이디</td>
      <td>닉네임</td>
      <td>False</td>
      <td>73</td>
      <td>계정 아이디</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GRANDMASTER</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>224</td>
      <td>63</td>
      <td>소환사 아이디</td>
      <td>닉네임</td>
      <td>False</td>
      <td>111</td>
      <td>계정 아이디</td>
    </tr>
  </tbody>
</table>
</div>



## 1-4. Match

각 소환사의 AcountId로 매치 데이터를 받아오자.

매치 데이터에서도 소환사ID로 받는 데이터와 게임ID로 받는 데이터가 다르다.

이번 분석에서 필요한 데이터는 게임ID로 받는 데이터가 필요하기 때문에

소환사 ID로 데이터를 받고, 그 데이터에서 게임ID를 얻어 데이터를 받는, 두 번의 작업을 거쳐야 한다.

#### a. Match List


```python
# Match info 데이터 받기 (gameId를 얻기 위해서)

# season = str(13)

# match_info_df = pd.DataFrame()
# for account_id in league_df['account_id']:
#     api_url = 'https://kr.api.riotgames.com/lol/match/v4/matchlists/by-account/' + account_id + \
#                   '?season=' + season + '&api_key=' + api_key
#     r = requests.get(api_url)
#     while r.status_code!=200: # 요청 제한 또는 오류로 인해 정상적으로 받아오지 않는 상태라면, 5초 간 시간을 지연
#         time.sleep(5)
#         r = requests.get(api_url)
#     match_info_df = pd.concat([match_info_df, pd.DataFrame(r.json()['matches'])])
# match_info_df.to_csv('MatchInfoData.csv')

# 저장된 데이터 받아오기
match_info_df = pd.read_csv('MatchInfoData.csv', index_col=0)
match_info_df.reset_index(inplace=True)
```

#### b. Match


```python
# Match 데이터 받기 (gameId를 통해 경기의 승패, 팀원과 같은 정보가 담겨있다.)

# match_info_df = match_info_df.drop_duplicates('gameId')

# match_df = pd.DataFrame()
# for game_id in match_info_df['gameId']: # 이전의 매치에 대한 정보 데이터에서 게임 아이디를 가져온다
#     api_url = 'https://kr.api.riotgames.com/lol/match/v4/matches/' + str(game_id) + '?api_key=' + api_key
#     r = requests.get(api_url)
#     while r.status_code!=200: # 요청 제한 또는 오류로 인해 정상적으로 받아오지 않는 상태라면, 5초 간 시간을 지연
#         time.sleep(5)
#         r = requests.get(api_url)
#     r_json = r.json()
#     temp_df = pd.DataFrame(list(r_json.values()), index=list(r_json.keys())).T # 게임 아이디에 대한 매치 데이터를 받아서 추가
#     match_df = pd.concat([match_df, temp_df])

# match_df.to_csv('MatchData.csv') # 파일로 저장

match_df = pd.read_csv('MatchData.csv', index_col=0)

# 정확한 통계를 위해 가장 최신의 버전과 클래식 게임에 대한 데이터만 가져오자
match_df = match_df.loc[(match_df['gameVersion']=='9.3.261.9578') &
             (match_df['gameMode']=='CLASSIC'), :]

# 그 중에서도 이번 분석에서는 소환사의 협곡 솔로 랭크와 팀 랭크 게임만 사용한다.
select_indices = (match_df['queueId']==420) | (match_df['queueId']==440)
match_df = match_df.loc[select_indices, :].reset_index(drop=True)

# DataFrame 내의 리스트들이 파일로 저장되었다가 불러지는 과정에서 문자로 인식됨
for column in ['teams', 'participants', 'participantIdentities']:
    match_df[column] = match_df[column].map(lambda v: eval(v)) # 각 값에 대해 eval 함수를 적용

match_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gameId</th>
      <th>platformId</th>
      <th>gameCreation</th>
      <th>gameDuration</th>
      <th>queueId</th>
      <th>mapId</th>
      <th>seasonId</th>
      <th>gameVersion</th>
      <th>gameMode</th>
      <th>gameType</th>
      <th>teams</th>
      <th>participants</th>
      <th>participantIdentities</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3545799154</td>
      <td>KR</td>
      <td>1550234618434</td>
      <td>1101</td>
      <td>420</td>
      <td>11</td>
      <td>13</td>
      <td>9.3.261.9578</td>
      <td>CLASSIC</td>
      <td>MATCHED_GAME</td>
      <td>[{'teamId': 100, 'win': 'Win', 'firstBlood': T...</td>
      <td>[{'participantId': 1, 'teamId': 100, 'champion...</td>
      <td>[{'participantId': 1, 'player': {'platformId':...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3545778231</td>
      <td>KR</td>
      <td>1550232922083</td>
      <td>1358</td>
      <td>420</td>
      <td>11</td>
      <td>13</td>
      <td>9.3.261.9578</td>
      <td>CLASSIC</td>
      <td>MATCHED_GAME</td>
      <td>[{'teamId': 100, 'win': 'Fail', 'firstBlood': ...</td>
      <td>[{'participantId': 1, 'teamId': 100, 'champion...</td>
      <td>[{'participantId': 1, 'player': {'platformId':...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3545729791</td>
      <td>KR</td>
      <td>1550231302002</td>
      <td>1261</td>
      <td>420</td>
      <td>11</td>
      <td>13</td>
      <td>9.3.261.9578</td>
      <td>CLASSIC</td>
      <td>MATCHED_GAME</td>
      <td>[{'teamId': 100, 'win': 'Win', 'firstBlood': F...</td>
      <td>[{'participantId': 1, 'teamId': 100, 'champion...</td>
      <td>[{'participantId': 1, 'player': {'platformId':...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3542888104</td>
      <td>KR</td>
      <td>1550065105427</td>
      <td>944</td>
      <td>420</td>
      <td>11</td>
      <td>13</td>
      <td>9.3.261.9578</td>
      <td>CLASSIC</td>
      <td>MATCHED_GAME</td>
      <td>[{'teamId': 100, 'win': 'Win', 'firstBlood': T...</td>
      <td>[{'participantId': 1, 'teamId': 100, 'champion...</td>
      <td>[{'participantId': 1, 'player': {'platformId':...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3542796242</td>
      <td>KR</td>
      <td>1550060832419</td>
      <td>2070</td>
      <td>420</td>
      <td>11</td>
      <td>13</td>
      <td>9.3.261.9578</td>
      <td>CLASSIC</td>
      <td>MATCHED_GAME</td>
      <td>[{'teamId': 100, 'win': 'Fail', 'firstBlood': ...</td>
      <td>[{'participantId': 1, 'teamId': 100, 'champion...</td>
      <td>[{'participantId': 1, 'player': {'platformId':...</td>
    </tr>
  </tbody>
</table>
</div>



큰일났다.

새벽 1시에 API Key가 만료되는데, 깜빡하고 잠들었는데 일어나보니 블랙리스트에 올라서 Key 갱신이 불가능하다...

아마도 Key도 만료된 주제에 10시간 동안 5초 마다 데이터를 요청해서 많이 화난 것 같다.

더 이상의 데이터를 받아올 수 없다... 이미 받은 11,958개의 데이터 만으로 분석을 끝내야한다.

---

하루가 지나니 갱신 가능해서 Key를 재발급 받았다. 하지만 이미 가공한 데이터로 진행하려고 한다.

#### c. Match Team


```python
## 매치 데이터에서 teams, participants, participantIdentities가 최종적으로 원하는 데이터이다.

# match_teams_df = pd.DataFrame()
# for i in range(len(match_df)):
#     temp_df = pd.DataFrame(match_df['teams'].iloc[i]) # teams 데이터를 2행 짜리 데이터프레임으로 변환
#     temp_df['gameId'] = match_df['gameId'].iloc[i] # teams 데이터에 각 게임의 gameId 추가 (2행 마다 같은 값)
#     # teams 데이터에 있는 bans 데이터를 5개의 변수로 추가한다
#     ban_dict = {i: pd.DataFrame(temp_df['bans'][i]).iloc[:, 0] for i in range(2)} # 각 팀의 밴픽을 저장
#     temp_ban = pd.DataFrame(ban_dict).T
#     temp_ban.columns = [f'ban{i}' for i in range(1, 6)] # 열 이름 변경
#     temp_df = pd.concat([temp_df, temp_ban], axis=1)

#     match_teams_df = pd.concat([match_teams_df, temp_df])

# match_teams_df.to_csv('MatchTeamsData.csv')

match_teams_df = pd.read_csv('MatchTeamsData.csv', index_col=0).reset_index()
match_teams_df.drop('bans', axis=1, inplace=True)
```

#### d. Match Participants

Match 데이터에 있는 Participants 데이터를 데이터프레임으로 변환하려고 하는데,

lane과 role이 틀리는 문제가 발생한다. 그것도 한 두 개가 틀리는게 아니라 엄청 틀린다.

잘못되어 있는 정보는 대부분 None으로 되어있지만, 적혀있으면서도 틀리는 경우도 종종 보인다.

다른 API도 모두 살펴봤지만 어디에도 제대로 lane이나 role이 기재되어 있는 경우가 안보인다.

그런 데이터에도 op.gg에서는 제대로 인식이 되어 있는데, 어떻게 인식하는 것인지 궁금하다. 근데 op.gg에서도 가끔 틀리는 경우가 보이긴 한다.

자체적으로 데이터의 패턴에서 계산해내는 것인가? 한 번 생각해보자면

1. 강타를 들면 정글러다.

    * 강타를 들고 라인전을 하는 캐릭터는 어떻게 판단하는지? -> 강타가 두 명 이상일 때에 timestamp에 찍히는 위치로 판단 가능하지 않을까? 그리고 타임스탬프에 찍힌 정글 몬스터 잡은 수가 가장 확실할 것 같다.
    * 강타를 들지 않고 정글을 하는 경우는 어떻게 판단하는지? -> 이런 경우에는 적어도 강타가 한 명도 없지 않을까? 강타를 들지 않고 정글을 도는 경우도 거의 없을 건데 동시에 강타 들고 라인전을 가는 경우는 극히 드물 것이고, 이럴 때는 정글 몬스터 잡은 수로 판단하자.


2. CS가 가장 적거나 서포터 아이템을 가지고 있으면 서포터다.

    * 서포터 아이템을 가지고 라인에 가는 단식 메타는 어떻게 할지? -> 타임스탬프에 찍힌 위치? 골드 수급 상황? cs 개수가 가장 유효할 것 같다, 일단 다행히도 9.3에서부터 단식메타가 망했다.


3. 원딜, 미드, 탑이 애매하다. 잘 쓰이는 스펠과 챔피언은 있지만 항상 그렇지는 않다. 챔피언, 스펠, 타임스탬프의 위치, 아이템 등 모든 요소를 조금씩 가중치를 주고 곱해서 점수를 부여하는 것이 좋을 것 같다.

> 더해서 이름이 틀리는 경우도 있는데 이 경우는 아마 닉네임을 변환해서 그런 것이 아닐까 싶다 -> 하지만 op.gg에는 제대로 된 정보가 기재되어있는데, 그렇다면 닉네임 변환되는 것을 인식할 수 있는 방법이 있나? (새로 등록되는 이름이지만 과거의 데이터가 존재하는 이름에 대해서 데이터베이스에 저장된 과거 데이터 값과 비교해서 가장 유사도가 높은 데이터의 이름을 변경하는건가?)

---

타임스탬프를 더 자세히 봤는데, 무엇보다 타임스탬프에 찍힌 위치가 가장 확실할 것 같다.

그리고 현재 타임스탬프를 전부 봐서 서포터 아이템 유무를 판단하기가 쉽지 않다.

그래서 규칙을 하나 정했다.

규칙 (순서대로 적용)

1. 강타를 든 경우 정글이다.
2. 강타가 두 명 이상일 경우, 정글은 라인을 보고 결정한다.
3. 정글을 제외하고 CS가 가장 적으면서 위치 상 바텀에 가장 많이 있으면 서포터다.
4. 서포터를 제외하고 각 라인에 있는 수가 가장 많으면 그 라인(탑, 정글, 미드, 봇 캐리)이다.

#### Timestamp - position


```python
# 각 타임라인에 찍힌 위치 정보가 필요한데, match-timelines 데이터에 모여있다.
# 그래서 이 데이터를 가져와야한다.

# match_timeline_list = []
# for game_id in tqdm(match_df['gameId']): # 각 게임 아이디마다 요청
#     api_url = f'https://kr.api.riotgames.com/lol/match/v4/timelines/by-match/{game_id}?api_key={api_key}'
#     r = requests.get(api_url)
#     while r.status_code!=200: # 요청 제한 또는 오류로 인해 정상적으로 받아오지 않는 상태라면, 3초 간 시간을 지연
#         time.sleep(3)
#         r = requests.get(api_url)
#     temp_match = pd.DataFrame(list(r.json().values())[0]) # 전체 데이터 저장 (데이터 값에 딕셔너리 형태로 필요한 정보가 저장)
#     temp_timeline = pd.DataFrame()
#     len_timeline = temp_match.shape[0]
#     for i in range(len_timeline): # 각 게임의 타임라인이 모두 다르기 때문 (1분 가량마다 타임라인이 찍힌다)
#         temp_current_timeline = pd.DataFrame(temp_match['participantFrames'].iloc[i]).T
#         if i != (len_timeline-1):
#             temp_position = pd.DataFrame(list(temp_current_timeline['position'].values), index=temp_current_timeline.index)
#             temp_current_timeline = pd.concat([temp_current_timeline, temp_position], axis=1)
#             temp_current_timeline.drop('position', axis=1, inplace=True)
#         temp_current_timeline['timestamp'] = temp_match['timestamp'][i]
#         temp_timeline = pd.concat([temp_timeline, temp_current_timeline], axis=0, sort=False)
#     match_timeline_list.append(temp_timeline)

# f = open('MatchTimelineData.pickle', 'wb') # 리스트 안의 데이터프레임 형태이므로 바이너리 코드로 저장하기 위함임
# pickle.dump(match_timeline_list, f)
# f.close()

# 블랙리스트 되서 또 11788개의 데이터만 받아왔음
match_df = match_df.iloc[:11788, :].copy()
f = open('MatchTimelineData.pickle', 'rb')
match_timeline_list = pickle.load(f)
```

블랙리스트에 또 걸려서 200개 가량의 데이터를 못 받았고, 11,788개의 데이터만 받아왔다.


```python
# 강타 유무를 판단하기 위해 spell api의 정보를 받아와서 소환사의 협곡 맵에서 사용하는 스펠만 가져온다.

spell_api = 'http://ddragon.leagueoflegends.com/cdn/9.3.1/data/ko_KR/summoner.json'
r = requests.get(spell_api)
spell_info_df = pd.DataFrame(r.json())
spell = {}
for i in range(len(spell_info_df)):
    cur_spell = spell_info_df['data'].iloc[i]
    if 'CLASSIC' in cur_spell['modes']:
        spell[int(cur_spell['key'])] = cur_spell['name']
spell = sorted(spell.items(), key=lambda t : t[0])
```

맵 데이터는 512*512 형태의 이미지인데, 찍힌 좌표는 15000 가량의 범위이다. 그렇기 때문에 Scale이 필요하다.


```python
def MapScaler(data, x_range=(-120, 14870), y_range=(-120, 14980)): # x, y의 범위
    x = data['x'].astype('float64').values.reshape(-1, 1)
    y = data['y'].astype('float64').values.reshape(-1, 1)
    x_range = np.array(x_range).astype('float64').reshape(-1, 1)
    y_range = np.array(y_range).astype('float64').reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 512)) # 0~512로 변환
    scaler.fit(x_range)
    x = scaler.transform(x)
    scaler.fit(y_range)
    y = scaler.transform(y)
    return x, 512 - y

# 이미지는 왼쪽 위가 0인데, 지도는 왼쪽 아래가 0이다.
# 그래서 좌표만 뒤집어서 출력을 시키고 싶었는데, 찾기 힘들어서 임의로 y값을 뒤집었다.
```


```python
cur_player = cur_timeline.loc['10'].copy()
x = cur_player['x']
y = cur_player['y']
max_lane, lane = LanePredict(cur_player, support_index==str(j), jungle_index)

maplink = 'https://s3-us-west-1.amazonaws.com/riot-developer-portal/docs/map11.png'
image = io.imread(maplink)
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
ax.scatter(x, y, c='black', s=50)
ax.axis('off')
ax.imshow(image)
plt.show()
```


![1](/assets/post-image/2019-02-28-lol-duo-tier/1.png)

실제 데이터의 탑 라이너 동선이다.

```python
x1 = np.random.uniform(20, 60, 1000)
y1 = np.random.uniform(30, 220, 1000)

x2 = np.random.uniform(20, 150, 1500)
y2 = np.random.uniform(20, 160, 1500)

x3 = np.random.uniform(30, 200, 1000)
y3 = np.random.uniform(20, 60, 1000)
maplink = 'https://s3-us-west-1.amazonaws.com/riot-developer-portal/docs/map11.png'
image = io.imread(maplink)
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
ax.scatter(x1, y1, c='black', s=50)
ax.scatter(x2, y2, c='black', s=50)
ax.scatter(x3, y3, c='black', s=50)
ax.axis('off')
ax.imshow(image)
plt.show()
```


![2](/assets/post-image/2019-02-28-lol-duo-tier/2.png)


미니맵에 찍어보면서 찾은 각 라인의 좌표 범위이다.

위의 탑 경우 탑 8점, etc 2점, 정글 2점으로 탑으로 라인이 계산된다.

etc는 각 본진으로, 어디에도 속하지 않는 범위이며 여기 어디에도 속하지 않으면 정글이다.

---

* top_range : (20 ~  60, 30 ~ 220), (20 ~ 150, 20 ~ 160), (30 ~ 200, 20 ~ 60)
* mid_range : (195 ~ 265, 250 ~ 310), (220 ~ 295, 220 ~ 290), (250 ~ 320, 200 ~ 260), (290 ~ 350, 160 ~ 215), (160 ~ 220, 290 ~ 340)
* bot_range : (310 ~ 460, 435 ~ 485), (400 ~ 490, 385 ~ 480), (440 ~ 490, 310 ~ 455)
* etc_range : (0 ~ 170, 340 ~ 512), (340 ~ 512, 0 ~ 170)


```python
def SupJugPredict(data, player_spells, i): # 서폿과 정글을 판단하기 위한 함수
    if i==1: # 블루팀
        final_timeline = data.iloc[-10:-5].copy()
        spells = player_spells[:5]
    if i==6: # 레드팀
        final_timeline = data.iloc[-5:].copy()
        spells = player_spells[5:10]
    smite_count = (spells == 11).sum()
    # 각 팀의 스마이트 개수 (스마이트가 2명 이상이 보유하면, 라인의 위치로 판단하기 위함)
    if smite_count == 0: jungle_index = final_timeline['jungleMinionsKilled'].idxmax()
    elif  smite_count > 1: jungle_index = False
    else: jungle_index = np.where(spells == 11)[0][0] + i # 강타를 들고있는 플레이어 번호

    final_timeline = final_timeline.drop(index=final_timeline['jungleMinionsKilled'].idxmax()) # 정글 제외
    return final_timeline['minionsKilled'].idxmin(), str(jungle_index) # 정글을 제외하고 cs가 가장 적은 사람이 서포터

def LanePredict(data, support_bool=False, jungle_bool=True):
    lane = {'TOP': 0, 'MID': 0, 'BOT_CARRY': 0, 'JUNGLE': 0}
    etc = 0
    for xi, yi in zip(data['x'], data['y']):
        if (xi > 20) & (xi < 60) & (yi > 30) & (yi < 220): lane['TOP'] += 1
        elif (xi > 20) & (xi < 150) & (yi > 20) & (yi < 160): lane['TOP'] += 1
        elif (xi > 30) & (xi < 200) & (yi > 20) & (yi < 60): lane['TOP'] += 1
        elif (xi > 195) & (xi < 265) & (yi > 250) & (yi < 310): lane['MID'] += 1
        elif (xi > 220) & (xi < 295) & (yi > 220) & (yi < 290): lane['MID'] += 1
        elif (xi > 250) & (xi < 320) & (yi > 200) & (yi < 260): lane['MID'] += 1
        elif (xi > 290) & (xi < 350) & (yi > 160) & (yi < 215): lane['MID'] += 1
        elif (xi > 160) & (xi < 220) & (yi > 290) & (yi < 340): lane['MID'] += 1
        elif (xi > 310) & (xi < 460) & (yi > 435) & (yi < 485): lane['BOT_CARRY'] += 1
        elif (xi > 400) & (xi < 490) & (yi > 385) & (yi < 480): lane['BOT_CARRY'] += 1
        elif (xi > 440) & (xi < 490) & (yi > 310) & (yi < 455): lane['BOT_CARRY'] += 1
        elif (xi > 0) & (xi < 170) & (yi > 340) & (yi < 512): etc += 1
        elif (xi > 340) & (xi < 512) & (yi > 0) & (yi < 170): etc += 1
        else: lane['JUNGLE'] += 1
    if jungle_bool:
        del lane['JUNGLE']
    # 예측된 서포터 번호가 봇에 가장 오래 있었으면 서포터로 확정, 아니면 라인으로 판단
    if support_bool & (max(lane, key=lane.get) == 'BOT_CARRY'): return 'BOT_SUPPORT', lane
    return max(lane, key=lane.get), lane
```


```python
# 라인 계산

# lane_calculated = pd.DataFrame()
# for k in range(len(match_timeline_list)):
#     if match_df['gameDuration'][k] < 600: continue

#     cur_timeline = match_timeline_list[k].copy()
#     cur_timeline['jungleMinionsKilled'] = cur_timeline['jungleMinionsKilled'].astype('float64')
#     cur_timeline['minionsKilled'] = cur_timeline['minionsKilled'].astype('float64')
#     # 타임스탬프는 op.gg가 나타내는 아이템 타임스탬프와 비교 결과, 타임스탬프 값의 1000 단위가 1초인 것을 파악함
#     cur_timeline['timestamp'] = cur_timeline['timestamp'] / (1000*60)  # 타임스탬프 값을 분 단위로 변환
#     condition = (cur_timeline['timestamp'] > 2) & \  # 2분 이하 : 미니언이 라인 도착 전, 시야 확보와 정글 리쉬
#                 (cur_timeline['timestamp'] < 15)     # 15분 이상 : 라인전을 끝내고 다른 라인으로 이동할 수 있음
#     cur_timeline = cur_timeline.loc[condition].copy()
#     cur_timeline['x'], cur_timeline['y'] = MapScaler(cur_timeline)
#     player_spells = [(data['spell1Id'], data['spell2Id'])for data in match_df['participants'][k]] # 스펠 확인
#     player_spells = np.array(player_spells)

#     lane = {}
#     for i in range(1, 11, 5):  # 라인 계산
#         support_index, jungle_index = SupJugPredict(cur_timeline, player_spells, i)
#         for j in range(i, i+5):
#             if str(j) == jungle_index:
#                 lane[f'lane{j}'] = 'JUNGLE'
#                 continue
#             cur_player = cur_timeline.loc[str(j)].copy()
#             lane[f'lane{j}'], _ = LanePredict(cur_player, support_index==str(j), jungle_index)
#     lane['gameId'] = match_df['gameId'][k]
#     lane = pd.Series(lane)
#     if lane.value_counts().max() > 2: # 각 게임에 한 라인이 2명 이상이면 계산 착오로 판단하여 데이터 삭제
#         print(f"{k}번째 계산이 잘못되었습니다.")
#         continue
#     lane_calculated = pd.concat([lane_calculated, pd.Series(lane)], axis=1, sort=False)

# match_df = pd.concat([match_df, lane_calculated.T.reset_index(drop=True).drop('k', axis=1)], join='inner', axis=1)

# lane_calculated = lane_calculated.T.reset_index(drop=True).drop('k', axis=1)
# lane_calculated.to_csv('LaneCalculated.csv')

lane_calculated = pd.read_csv('LaneCalculated.csv', index_col=0)
match_df = pd.merge(match_df, lane_calculated, on='gameId')
```

11,788 개의 데이터 중 확실한 계산 실수는 450개였으며 11,338개의 데이터가 남았음.

또 실수가 있겠지만, 이후에 100개 이하의 원딜과 서포터 조합은 제외할 것으로 상관없을 것이라 판단.


```python
player_champs = [] # 각 플레이어의 챔피언 정보를 가져온다.
for i in range(len(match_df)): # 10명의 챔피언을 딕셔너리 형태로 변환하여 저장
    cur_champs = {f'champ{j+1}': match_df['participants'][i][j]['championId'] for j in range(10)}
    cur_champs['gameId'] = match_df['gameId'][i]
    player_champs.append(cur_champs)

player_champs = pd.DataFrame(player_champs, columns=cur_champs.keys())
```


```python
# use_cols = ['kills', 'deaths', 'assists', 'largestKillingSpree', 'largestMultiKill',
#             'longestTimeSpentLiving', 'totalDamageDealtToChampions', 'totalHeal', 'totalDamageTaken',
#             'goldEarned', 'turretKills', 'totalMinionsKilled', 'visionScore',
#             'firstBloodKill', 'firstBloodAssist', 'timeCCingOthers']
# # 일단 stats 정보에서 남길 데이터들을 선정하여 가져온다.

# stats_df = pd.DataFrame()
# for i in range(len(match_df)):
#     temp = pd.DataFrame()
#     for col in use_cols:
#         cur_values = {f'{col}{j+1}': match_df['participants'][i][j]['stats'][col] for j in range(10)}
#         temp = pd.concat([temp, pd.Series(cur_values)], axis=0, sort=False)
#     stats_df = pd.concat([stats_df, temp], axis=1, sort=False)
# stats_df = stats_df.T.reset_index(drop=True)
# stats_df['gameId'] = match_df['gameId']
# stats_df.to_csv('statsData.csv')

stats_df = pd.read_csv('statsData.csv', index_col=0)
stats_df = pd.merge(stats_df, player_champs)
```


```python
# 원하는 것은 원딜과 서포터의 조합일 때의 데이터이므로 팀 정보는 의미없다.
# 그러므로 블루팀과 레드팀으로 나눠서 정보를 가공한 후, 합친다.

def DropColumns(data, col, i): # 각 팀에 맞게 상대팀의 필요없는 데이터는 삭제하기 위함
    for k in range(1, 6):
        data[i].drop(f'{col}{i*5+k}', axis=1, inplace=True)
        if i==0: data[i].rename(columns={f'{col}{5+k}':f'{col}{k}'}, inplace=True)
    return

use_cols = ['lane', 'champ', 'kills', 'deaths', 'assists', 'largestKillingSpree', 'largestMultiKill',
            'longestTimeSpentLiving', 'totalDamageDealtToChampions', 'totalHeal', 'totalDamageTaken',
            'goldEarned', 'turretKills', 'totalMinionsKilled', 'visionScore',
            'firstBloodKill', 'firstBloodAssist', 'timeCCingOthers']

# 팀 나누기
blue_teams_df = pd.merge(match_teams_df[match_teams_df['teamId'] == 100], lane_calculated, on='gameId')
blue_teams_df = pd.merge(blue_teams_df, stats_df, on='gameId')
red_teams_df = pd.merge(match_teams_df[match_teams_df['teamId'] == 200], lane_calculated, on='gameId')
red_teams_df = pd.merge(red_teams_df, stats_df, on='gameId')

teams = [red_teams_df, blue_teams_df] # 팀을 리스트로 만들어 포인터를 이용하여 데이터를 변환한다.
for i in range(2):
    for col in use_cols:
        DropColumns(teams, col, i)

teams_df = pd.concat([blue_teams_df, red_teams_df]).reset_index(drop=True)
teans_df.drop('index', axis=1)
```


```python
# 가지고 있는 전체 플레이어에 대한 데이터를 원딜과 서포터에 대한 데이터로 변환한다.

# def ChampTrans(x):  # 챔피언 key를 이름으로 변환하는 함수
#     if x!=-1: return champ_df.loc[champ_df['key'] == str(x), 'name'].values[0]
#     return

# for i in range(1, 6):
#     teams_df[f'ban{i}'] = teams_df[f'ban{i}'].map(lambda x: ChampTrans(x)) # map 함수로 챔피언 모두 변환
#     teams_df[f'champ{i}'] = teams_df[f'champ{i}'].map(lambda x: ChampTrans(x))
#     for col in use_cols:
#         if i == 1:
#             teams_df[f'CARRY_{col}'] = np.nan
#             teams_df[f'SUPPORT_{col}'] = np.nan
#         temp_index = np.where(teams_df[f'lane{i}'] == 'BOT_CARRY')
#         teams_df[f'CARRY_{col}'].iloc[temp_index] = teams_df[f'{col}{i}'].iloc[temp_index]
#         temp_index = np.where(teams_df[f'lane{i}'] == 'BOT_SUPPORT')
#         teams_df[f'SUPPORT_{col}'].iloc[temp_index] = teams_df[f'{col}{i}'].iloc[temp_index]
# teams_df = teams_df.fillna('NONE')
# duo_df = teams_df.copy()
# duo_df.to_csv('DuoData.csv')

duo_df = pd.read_csv('DuoData.csv', index_col=0)
duo_df = duo_df.drop('index', axis=1)
duo_df['win'] = duo_df['win'].map({'Win':1, 'Fail':0}) # win을 1, fail을 0으로 변환

duo_df = pd.concat([duo_df.iloc[:, :21], duo_df.iloc[:, 111:]], axis=1, sort=False)
```

## 계산

이제 스탯 정보에서 의미를 창출하여 바텀 듀오의 스코어를 산정해야 한다.

op.gg에서는 골드, 받은 데미지, 입힌 데미지, 밴픽, kda 등을 모두 참고한다고 한다.

일단 stats에 있는 많은 데이터를 모두 가져와서 나름의 계산을 해봤지만, 가중치를 부여하기 힘들어서 기존 수식을 이용하고자 했다.

그래서 https://www.metasrc.com 사이트에서 밴, 픽, 승률, kda와 점수 데이터를 받아와서 회귀 분석을 통해 계수를 예측했다.

예측한 계수에서 조금 변경하여 Score 수식을 만들었는데, 그 수식은 아래와 같다.

Score = winRate*2 + pickRate*2 + banRate*0.1 + kdaMean*1 - 40

---

여기서 winRate는 기존 1.66 정도의 계수에서 다른 밴, 픽, kda 데이터가 단일 라인보다 부정확한 것을 감안하여 승률에 큰 가중치를 부여했다.

pickRate는 기존과 거의 동일하게, banRate는 기존 0.5 정도에서 0.1로 낮췄다.

데이터가 적어서 그런지, 아니면 천상계라서 그런지 쓰레쉬와 같은 몇 캐릭터 밴률이 게임 당 60%에 달한다.

물론, 이는 한 게임에서 두 개의 밴을 받을 수 있는 것이 크겠지만 그렇다고 이것을 하나의 밴으로 변환하는 것도 좋진 않을 것 같다.

가장 최선은 두 번째 밴은 가중치를 부여하여 적용하는 것이겠지만, 현재는 가중치를 부여하지 않고 하였다.

여튼, 특정 캐릭터의 밴률이 매우 높아 너무 큰 가중치가 부여되지 않도록 0.1로 낮췄으며 kda는 동일하게 1로 부여하였다.

절편(intercept)은 기존과 비슷하게 40으로 설정하였다.

OP.GG에서는 다른 지표들을 이용하여 어떻게 수식을 만들었는지 궁금하다.


```python
# 밴 확률 계산, 원딜과 서포터의 동시 밴은 너무 경우의 수가 적어서 각각의 밴률 평균으로 계산
def BanRate(row, carry, support):  
    carry_ban = 0
    support_ban = 0
    for i in range(1, 6):
        if row[f'ban{i}'] == carry: carry_ban += 1
        if row[f'ban{i}'] == support: support_ban += 1
    return (carry_ban + support_ban) / 2

def KdaCalc(row): # kda 계산 (데스가 0이면 1.2를 곱하는 방식으로 가중치 부여)
    if row['CARRY_deaths'] != 0:
        carry = (row['CARRY_kills'] + row['CARRY_assists']) / row['CARRY_deaths']
    else: carry = (row['CARRY_kills'] + row['CARRY_assists']) * 1.2
    if row['SUPPORT_deaths'] != 0:
        support = (row['SUPPORT_kills'] + row['SUPPORT_assists']) / row['SUPPORT_deaths']
    else: support = (row['SUPPORT_kills'] + row['SUPPORT_assists']) * 1.2
    return (carry+support)/2

match_count = duo_df['gameId'].count() / 2 # 한 게임당 두 행으로 구성되어 있으므로 2로 나눈다.
duo_stats = pd.DataFrame()
for carry in duo_df['CARRY_champ'].unique(): # 각 원딜과 서포터마다 계산
    for support in duo_df['SUPPORT_champ'].unique():
        condition =  (duo_df['CARRY_champ'] == carry) & \
                     (duo_df['SUPPORT_champ'] == support)
        cond_count = condition.sum()
        if cond_count < 100: continue # 바텀 듀오 데이터가 100이하면 스킵

        stat = {}
        stat['count'] = cond_count
        stat['duoChamps'] = f'{carry}&{support}'
        stat['banRate'] = duo_df.apply(lambda x: BanRate(x, carry, support), axis=1).sum() / match_count
        stat['pickRate'] = duo_df.loc[condition, 'gameId'].count() / match_count
        stat['winRate'] = duo_df.loc[condition, 'win'].sum() / cond_count
        stat['kdaMean'] = duo_df[condition].apply(KdaCalc, axis=1).mean()
        stat['largestKillingSpreeWeightMean'] = ((duo_df[condition]['CARRY_largestKillingSpree']*2 + \
                                                  duo_df[condition]['CARRY_largestKillingSpree']) / 3).mean()
        stat['largestMultiKillWeightMean'] = ((duo_df[condition]['CARRY_largestMultiKill']*2 + \
                                               duo_df[condition]['SUPPORT_largestMultiKill']) / 3).mean()
        stat['longestTimeSpentLivingMean'] = ((duo_df[condition]['CARRY_longestTimeSpentLiving'] + \
                                              duo_df[condition]['SUPPORT_longestTimeSpentLiving']) / 2).mean()
        stat['totalDamageDealtToChampionsWeightMean'] = ((duo_df[condition]['CARRY_totalDamageDealtToChampions']*2 + \
                                                          duo_df[condition]['SUPPORT_totalDamageDealtToChampions']) / 3).mean()
        stat['totalHealWeightMean'] = ((duo_df[condition]['CARRY_totalHeal'] + \
                                        duo_df[condition]['SUPPORT_totalHeal']*2) / 3).mean()
        stat['SUPPORT_totalDamageTaken'] = duo_df[condition]['SUPPORT_totalDamageTaken'].mean()
        stat['goldEarnedMean'] = ((duo_df[condition]['CARRY_goldEarned'] + \
                                  duo_df[condition]['SUPPORT_goldEarned']) / 2).mean()
        stat['turretKillsMean'] = ((duo_df[condition]['CARRY_turretKills'] + \
                                    duo_df[condition]['SUPPORT_turretKills']) / 2).mean()        
        stat['totalMinionsKilledWeightMean'] = ((duo_df[condition]['CARRY_totalMinionsKilled']*2 + \
                                                 duo_df[condition]['SUPPORT_totalMinionsKilled']) / 3).mean()
        stat['visionScoreWeightMean'] = ((duo_df[condition]['CARRY_visionScore'] + \
                                          duo_df[condition]['SUPPORT_visionScore']*2) / 3).mean()
        stat['firstBloodKillAssistMean'] = ((duo_df[condition]['CARRY_firstBloodKill'] + \
                                        duo_df[condition]['CARRY_firstBloodAssist'] + \
                                        duo_df[condition]['SUPPORT_firstBloodKill'] + \
                                        duo_df[condition]['SUPPORT_firstBloodAssist']) / 4).mean()
        stat['timeCCingOthersWeightMean'] = ((duo_df[condition]['CARRY_timeCCingOthers'] + \
                                        duo_df[condition]['SUPPORT_timeCCingOthers']*2) / 3).mean()

        duo_stats = pd.concat([duo_stats, pd.Series(stat)], axis=1, sort=False)

duo_stats = duo_stats.T.reset_index(drop=True)

duo_stats['banRate'] = duo_stats['banRate'] * 100
duo_stats['pickRate'] = duo_stats['pickRate'] * 100
duo_stats['winRate'] = duo_stats['winRate'] * 100
duo_stats['Score'] = duo_stats['winRate']*2 + duo_stats['pickRate']*2 + duo_stats['banRate']*0.1 + duo_stats['kdaMean']*1 - 40
```


```python
duo_score = pd.concat([duo_stats.iloc[:, :6], duo_stats.iloc[:, -1:]], axis=1, sort=False)
duo_score = duo_score.sort_values(by='Score', ascending=False)
duo_score['Tier'] = 0
duo_score.loc[duo_score['Score'] < 65, 'Tier'] = 4
duo_score.loc[duo_score['Score'] >= 65, 'Tier'] = 3
duo_score.loc[duo_score['Score'] >= 75, 'Tier'] = 2
duo_score.loc[duo_score['Score'] >= 85, 'Tier'] = 1
duo_score
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>duoChamps</th>
      <th>banRate</th>
      <th>pickRate</th>
      <th>winRate</th>
      <th>kdaMean</th>
      <th>Score</th>
      <th>Tier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>51</th>
      <td>104</td>
      <td>야스오&amp;알리스타</td>
      <td>42.4149</td>
      <td>0.917269</td>
      <td>60.5769</td>
      <td>4.61059</td>
      <td>91.8405</td>
      <td>1</td>
    </tr>
    <tr>
      <th>46</th>
      <td>109</td>
      <td>루시안&amp;피들스틱</td>
      <td>55.1685</td>
      <td>0.961369</td>
      <td>59.633</td>
      <td>4.05884</td>
      <td>90.7645</td>
      <td>1</td>
    </tr>
    <tr>
      <th>43</th>
      <td>266</td>
      <td>루시안&amp;브라움</td>
      <td>50.9922</td>
      <td>2.34609</td>
      <td>57.1429</td>
      <td>5.16097</td>
      <td>89.2381</td>
      <td>1</td>
    </tr>
    <tr>
      <th>47</th>
      <td>150</td>
      <td>루시안&amp;나미</td>
      <td>51.2171</td>
      <td>1.32298</td>
      <td>58</td>
      <td>4.90897</td>
      <td>88.6767</td>
      <td>1</td>
    </tr>
    <tr>
      <th>35</th>
      <td>290</td>
      <td>칼리스타&amp;쓰레쉬</td>
      <td>30.4154</td>
      <td>2.55777</td>
      <td>57.5862</td>
      <td>4.93554</td>
      <td>88.265</td>
      <td>1</td>
    </tr>
    <tr>
      <th>41</th>
      <td>312</td>
      <td>루시안&amp;쓰레쉬</td>
      <td>80.9182</td>
      <td>2.75181</td>
      <td>53.8462</td>
      <td>4.92703</td>
      <td>86.2148</td>
      <td>1</td>
    </tr>
    <tr>
      <th>37</th>
      <td>180</td>
      <td>베인&amp;알리스타</td>
      <td>8.75816</td>
      <td>1.58758</td>
      <td>58.8889</td>
      <td>3.90254</td>
      <td>85.7313</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>116</td>
      <td>시비르&amp;카르마</td>
      <td>2.35491</td>
      <td>1.02311</td>
      <td>58.6207</td>
      <td>4.61989</td>
      <td>84.143</td>
      <td>2</td>
    </tr>
    <tr>
      <th>24</th>
      <td>551</td>
      <td>카이사&amp;쓰레쉬</td>
      <td>32.1132</td>
      <td>4.85976</td>
      <td>52.4501</td>
      <td>4.83784</td>
      <td>82.6689</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>144</td>
      <td>시비르&amp;소라카</td>
      <td>3.05609</td>
      <td>1.27007</td>
      <td>57.6389</td>
      <td>4.26753</td>
      <td>82.391</td>
      <td>2</td>
    </tr>
    <tr>
      <th>0</th>
      <td>279</td>
      <td>시비르&amp;쓰레쉬</td>
      <td>30.9843</td>
      <td>2.46075</td>
      <td>54.1219</td>
      <td>4.87986</td>
      <td>81.1435</td>
      <td>2</td>
    </tr>
    <tr>
      <th>13</th>
      <td>291</td>
      <td>이즈리얼&amp;카르마</td>
      <td>7.65126</td>
      <td>2.56659</td>
      <td>55.6701</td>
      <td>3.81914</td>
      <td>81.0577</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>242</td>
      <td>시비르&amp;알리스타</td>
      <td>6.12542</td>
      <td>2.13442</td>
      <td>55.7851</td>
      <td>4.3937</td>
      <td>80.8453</td>
      <td>2</td>
    </tr>
    <tr>
      <th>53</th>
      <td>156</td>
      <td>드레이븐&amp;쓰레쉬</td>
      <td>32.4175</td>
      <td>1.3759</td>
      <td>55.1282</td>
      <td>4.50432</td>
      <td>80.7543</td>
      <td>2</td>
    </tr>
    <tr>
      <th>42</th>
      <td>190</td>
      <td>루시안&amp;알리스타</td>
      <td>56.0593</td>
      <td>1.67578</td>
      <td>52.6316</td>
      <td>4.34786</td>
      <td>78.5685</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>127</td>
      <td>시비르&amp;피들스틱</td>
      <td>5.23461</td>
      <td>1.12013</td>
      <td>55.9055</td>
      <td>3.35173</td>
      <td>77.9265</td>
      <td>2</td>
    </tr>
    <tr>
      <th>49</th>
      <td>279</td>
      <td>케이틀린&amp;쓰레쉬</td>
      <td>30.9093</td>
      <td>2.46075</td>
      <td>52.6882</td>
      <td>4.24528</td>
      <td>77.6341</td>
      <td>2</td>
    </tr>
    <tr>
      <th>39</th>
      <td>193</td>
      <td>베인&amp;나미</td>
      <td>3.91603</td>
      <td>1.70224</td>
      <td>54.9223</td>
      <td>3.83182</td>
      <td>77.4725</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>412</td>
      <td>이즈리얼&amp;소라카</td>
      <td>8.35244</td>
      <td>3.6338</td>
      <td>52.4272</td>
      <td>3.69478</td>
      <td>76.652</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>207</td>
      <td>이즈리얼&amp;탐 켄치</td>
      <td>6.58846</td>
      <td>1.82572</td>
      <td>53.6232</td>
      <td>3.56866</td>
      <td>75.1253</td>
      <td>2</td>
    </tr>
    <tr>
      <th>44</th>
      <td>165</td>
      <td>루시안&amp;파이크</td>
      <td>60.5618</td>
      <td>1.45528</td>
      <td>50.9091</td>
      <td>4.04977</td>
      <td>74.8347</td>
      <td>3</td>
    </tr>
    <tr>
      <th>54</th>
      <td>136</td>
      <td>징크스&amp;쓰레쉬</td>
      <td>30.0582</td>
      <td>1.19951</td>
      <td>52.2059</td>
      <td>4.70116</td>
      <td>74.5178</td>
      <td>3</td>
    </tr>
    <tr>
      <th>23</th>
      <td>118</td>
      <td>이즈리얼&amp;질리언</td>
      <td>6.76486</td>
      <td>1.04075</td>
      <td>53.3898</td>
      <td>3.65563</td>
      <td>73.1933</td>
      <td>3</td>
    </tr>
    <tr>
      <th>17</th>
      <td>122</td>
      <td>이즈리얼&amp;나미</td>
      <td>6.57964</td>
      <td>1.07603</td>
      <td>52.459</td>
      <td>4.02758</td>
      <td>71.7556</td>
      <td>3</td>
    </tr>
    <tr>
      <th>36</th>
      <td>265</td>
      <td>베인&amp;쓰레쉬</td>
      <td>33.617</td>
      <td>2.33727</td>
      <td>49.434</td>
      <td>4.13371</td>
      <td>71.0379</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>521</td>
      <td>이즈리얼&amp;쓰레쉬</td>
      <td>36.2806</td>
      <td>4.59517</td>
      <td>47.2169</td>
      <td>3.67232</td>
      <td>70.9245</td>
      <td>3</td>
    </tr>
    <tr>
      <th>28</th>
      <td>251</td>
      <td>카이사&amp;파이크</td>
      <td>11.7569</td>
      <td>2.21379</td>
      <td>50.1992</td>
      <td>4.28129</td>
      <td>70.283</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>426</td>
      <td>이즈리얼&amp;알리스타</td>
      <td>11.4218</td>
      <td>3.75728</td>
      <td>48.8263</td>
      <td>3.92772</td>
      <td>70.237</td>
      <td>3</td>
    </tr>
    <tr>
      <th>25</th>
      <td>479</td>
      <td>카이사&amp;알리스타</td>
      <td>7.25437</td>
      <td>4.22473</td>
      <td>48.4342</td>
      <td>3.99136</td>
      <td>70.0347</td>
      <td>3</td>
    </tr>
    <tr>
      <th>20</th>
      <td>175</td>
      <td>이즈리얼&amp;잔나</td>
      <td>6.44294</td>
      <td>1.54348</td>
      <td>50.8571</td>
      <td>3.68422</td>
      <td>69.1298</td>
      <td>3</td>
    </tr>
    <tr>
      <th>30</th>
      <td>131</td>
      <td>카이사&amp;그라가스</td>
      <td>3.1046</td>
      <td>1.15541</td>
      <td>51.145</td>
      <td>3.93526</td>
      <td>68.8466</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>115</td>
      <td>시비르&amp;모르가나</td>
      <td>15.0556</td>
      <td>1.01429</td>
      <td>50.4348</td>
      <td>4.26275</td>
      <td>68.6664</td>
      <td>3</td>
    </tr>
    <tr>
      <th>38</th>
      <td>169</td>
      <td>베인&amp;파이크</td>
      <td>13.2607</td>
      <td>1.49056</td>
      <td>50.2959</td>
      <td>3.5545</td>
      <td>68.4534</td>
      <td>3</td>
    </tr>
    <tr>
      <th>45</th>
      <td>119</td>
      <td>루시안&amp;모르가나</td>
      <td>64.9894</td>
      <td>1.04957</td>
      <td>47.8992</td>
      <td>3.81005</td>
      <td>68.2064</td>
      <td>3</td>
    </tr>
    <tr>
      <th>31</th>
      <td>157</td>
      <td>카이사&amp;피들스틱</td>
      <td>6.36356</td>
      <td>1.38472</td>
      <td>50.3185</td>
      <td>3.39657</td>
      <td>67.4393</td>
      <td>3</td>
    </tr>
    <tr>
      <th>16</th>
      <td>276</td>
      <td>이즈리얼&amp;피들스틱</td>
      <td>10.531</td>
      <td>2.43429</td>
      <td>48.913</td>
      <td>3.19483</td>
      <td>66.9426</td>
      <td>3</td>
    </tr>
    <tr>
      <th>15</th>
      <td>130</td>
      <td>이즈리얼&amp;그라가스</td>
      <td>7.27201</td>
      <td>1.14659</td>
      <td>50</td>
      <td>3.45925</td>
      <td>66.4796</td>
      <td>3</td>
    </tr>
    <tr>
      <th>32</th>
      <td>103</td>
      <td>카이사&amp;블리츠크랭크</td>
      <td>3.22367</td>
      <td>0.908449</td>
      <td>49.5146</td>
      <td>4.0504</td>
      <td>65.2188</td>
      <td>3</td>
    </tr>
    <tr>
      <th>48</th>
      <td>133</td>
      <td>루시안&amp;소라카</td>
      <td>52.9899</td>
      <td>1.17305</td>
      <td>46.6165</td>
      <td>4.10975</td>
      <td>64.9879</td>
      <td>4</td>
    </tr>
    <tr>
      <th>12</th>
      <td>463</td>
      <td>이즈리얼&amp;파이크</td>
      <td>15.9243</td>
      <td>4.08361</td>
      <td>45.3564</td>
      <td>3.4853</td>
      <td>63.9577</td>
      <td>4</td>
    </tr>
    <tr>
      <th>40</th>
      <td>149</td>
      <td>베인&amp;소라카</td>
      <td>5.68883</td>
      <td>1.31416</td>
      <td>48.3221</td>
      <td>3.9012</td>
      <td>63.7427</td>
      <td>4</td>
    </tr>
    <tr>
      <th>22</th>
      <td>115</td>
      <td>이즈리얼&amp;갈리오</td>
      <td>8.18928</td>
      <td>1.01429</td>
      <td>48.6957</td>
      <td>3.49081</td>
      <td>63.7296</td>
      <td>4</td>
    </tr>
    <tr>
      <th>50</th>
      <td>383</td>
      <td>케이틀린&amp;모르가나</td>
      <td>14.9806</td>
      <td>3.37802</td>
      <td>45.6919</td>
      <td>4.02377</td>
      <td>63.6617</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>198</td>
      <td>시비르&amp;파이크</td>
      <td>10.628</td>
      <td>1.74634</td>
      <td>47.4747</td>
      <td>4.07637</td>
      <td>63.5813</td>
      <td>4</td>
    </tr>
    <tr>
      <th>34</th>
      <td>169</td>
      <td>카이사&amp;소라카</td>
      <td>4.18504</td>
      <td>1.49056</td>
      <td>47.3373</td>
      <td>4.09941</td>
      <td>62.1736</td>
      <td>4</td>
    </tr>
    <tr>
      <th>27</th>
      <td>113</td>
      <td>카이사&amp;레오나</td>
      <td>2.68566</td>
      <td>0.996648</td>
      <td>47.7876</td>
      <td>3.36027</td>
      <td>61.1974</td>
      <td>4</td>
    </tr>
    <tr>
      <th>10</th>
      <td>187</td>
      <td>이즈리얼&amp;브라움</td>
      <td>6.35474</td>
      <td>1.64932</td>
      <td>45.9893</td>
      <td>3.54002</td>
      <td>59.4527</td>
      <td>4</td>
    </tr>
    <tr>
      <th>14</th>
      <td>263</td>
      <td>이즈리얼&amp;모르가나</td>
      <td>20.3519</td>
      <td>2.31963</td>
      <td>44.1065</td>
      <td>3.40599</td>
      <td>58.2934</td>
      <td>4</td>
    </tr>
    <tr>
      <th>33</th>
      <td>115</td>
      <td>카이사&amp;노틸러스</td>
      <td>2.2182</td>
      <td>1.01429</td>
      <td>45.2174</td>
      <td>3.26445</td>
      <td>55.9496</td>
      <td>4</td>
    </tr>
    <tr>
      <th>21</th>
      <td>156</td>
      <td>이즈리얼&amp;바드</td>
      <td>6.50026</td>
      <td>1.3759</td>
      <td>44.2308</td>
      <td>3.28781</td>
      <td>55.1512</td>
      <td>4</td>
    </tr>
    <tr>
      <th>11</th>
      <td>120</td>
      <td>이즈리얼&amp;레오나</td>
      <td>6.85306</td>
      <td>1.05839</td>
      <td>44.1667</td>
      <td>3.36044</td>
      <td>54.4959</td>
      <td>4</td>
    </tr>
    <tr>
      <th>52</th>
      <td>172</td>
      <td>자야&amp;라칸</td>
      <td>0.401305</td>
      <td>1.51702</td>
      <td>43.6047</td>
      <td>3.81218</td>
      <td>54.0957</td>
      <td>4</td>
    </tr>
    <tr>
      <th>26</th>
      <td>115</td>
      <td>카이사&amp;브라움</td>
      <td>2.18733</td>
      <td>1.01429</td>
      <td>42.6087</td>
      <td>3.21602</td>
      <td>50.6807</td>
      <td>4</td>
    </tr>
    <tr>
      <th>29</th>
      <td>169</td>
      <td>카이사&amp;모르가나</td>
      <td>16.1845</td>
      <td>1.49056</td>
      <td>40.8284</td>
      <td>3.78084</td>
      <td>50.0372</td>
      <td>4</td>
    </tr>
    <tr>
      <th>18</th>
      <td>134</td>
      <td>이즈리얼&amp;블리츠크랭크</td>
      <td>7.39107</td>
      <td>1.18187</td>
      <td>38.806</td>
      <td>3.32509</td>
      <td>44.0399</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



## 결론

---

아주 의외의 결과가 나와서 계산이 잘못된 것은 아닌지 처음부터 검증해봤지만, 계산상 실수는 없는 듯하다.

단지, 게임 수가 100판 밖에 안 되는 것이 문제인 듯하다.

하지만 현재로서는 시간이 부족하여 2분 당 100개의 데이터밖에 받아오지 못 하며 계속해서 블랙리스트에 오르기 때문에 더 이상은 불가능하다.

하지만 데이터가 크게 잘못되지는 않은 것 같다.

op.gg에서 보여주는 야스오의 티어는 4이지만, 알리스타와 매우 궁합이 좋다.

실제로도 인게임에서 저 조합을 만나면 매우 힘든 것이 사실이므로 크게 잘못된 데이터는 아니라고 생각한다.

또한 루시안은 현재 최강의 OP 캐릭터이며 피들스틱과 브라움, 나미와 매우 조합이 좋은 것도 사실이다.

다만, 브라움이 4티어인데 비해 3위를 차지하고 있는 것은 의외이다.

나름의 만족하는 결과를 얻었고, 짧은 시간이었지만 정말 재미있었다.

다만 공모전 기한과 겹치지 않아서 좀 더 시간이 있고 데이터가 많았으면, 그리고 기존에 연구하시던 분의 노하우를 배운다면 더 의미있는 분석이 되지 않았을까 아쉬움이 든다.

> 2019-02-20 ~ 2019-02-25


```python
# https://www.metasrc.com 데이터 (무단 추출 죄송합니다. 문제 시, 삭제하겠습니다.)
# 데이터는 하나의 값만 제외하고 삭제하였습니다.
tier = pd.DataFrame([[48.08, 53.86, 70.17, 1.59, 0.75, 2.01]],
                     columns=['Score', 'winRate', 'roleRate', 'pickRate', 'banRate' , 'kdaRatio'])

X = tier.drop(['Score', 'roleRate'], axis=1)
y = np.array(tier['Score'])

from sklearn.linear_model import LinearRegression

linear = LinearRegression(fit_intercept=True)
linear.fit(X, y)
linear.coef_
```




    array([1.66528205, 1.98733107, 0.42611145, 1.04606604])
