---
layout: post
title: "[게임데이터 분석 #2] BattleGround(배틀그라운드) 프로들의 이동 패턴 분석"
subtitle: "프로들은 어떤 행동을 할까?"
categories: study
tags: project
comments: true
---

### 분석 배경

이전에 입사 사전과제로 분석했던 내용인데, 원하는 만큼의 퀄리티가 나오진 않았습니다.

천 만 행이 넘는 큰 JSON 파일을 분석해본 경험도 처음이었고, 배틀그라운드에 대한 기본적인 지식도 부족했던 것 같네요.

특히 일반 유저들과 프로 선수들의 경기가 매우 큰 차이가 있다는 것을 알고, 엎고 다시 진행했던 것이 시간을 많이 날려먹었습니다.

총 분석 기간은 8일 정도였고, 부족한 분석이었지만 올려둡니다.


```python
# 패키지 불러오기

# api 요청
import requests
import json
# 데이터 자료형 및 분석도구
import pandas as pd
import numpy as np
# 시각화 패키지
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import patheffects

%matplotlib inline
sns.set()
# 스케일링
from sklearn.preprocessing import MinMaxScaler
# 시간
import time
from datetime
# 진행 사항 확인
from tqdm import tqdm
# PUBG 분석 도구
from chicken_dinner.pubgapi import PUBG
from chicken_dinner.constants import COLORS
from chicken_dinner.constants import map_dimensions
```


```python
# api key 설정 및 데이터 요청
api_key = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJqdGkiOiI2ZGNjNzNhMC0yMDk5LTAxMzctYjNjMi0wMmI4NjZkNzliOGIiLCJpc3MiOiJnYW1lbG9ja2VyIiwiaWF0IjoxNTUxNjk2NzA2LCJwdWIiOiJibHVlaG9sZSIsInRpdGxlIjoicHViZyIsImFwcCI6InB1YmctYmVzdC1wbGF5In0.Oz7GmLsWF7038XIO4vKd5sivhLreOnizxTNcARAEFQs'
headers = {'accept': 'application/vnd.api+json',
           'Authorization': f'Bearer {api_key}'}

url = "https://api.pubg.com/tournaments/kr-pkl18"
r = requests.get(url, headers=headers)
league_json = r.json()
league_json
```




    {'data': {'type': 'tournament',
      'id': 'kr-pkl18',
      'relationships': {'matches': {'data': [{'type': 'match',
          'id': '277295d2-e148-4749-92e8-7c00f3a23219'},
         {'type': 'match', 'id': '7cac694a-575f-46ad-8101-e3226cfbaf10'},
         {'type': 'match', 'id': 'ab7c47b7-5d1f-465f-8843-44b552809881'},
         {'type': 'match', 'id': '7428419d-81ec-45a9-ba4b-8a30f809e226'},
         {'type': 'match', 'id': 'f169b923-612f-42f5-a45b-7b2e4221d92d'},
         {'type': 'match', 'id': '08e38587-85f1-4730-b073-b6ea5e6daf66'},
         {'type': 'match', 'id': '59dc37a9-aee1-4bdb-8326-f46335f0033f'},
         {'type': 'match', 'id': '77f61312-b6d6-4357-b163-f979a35f20fd'},
         {'type': 'match', 'id': '96528262-a656-4480-86ef-90746bdae198'},
         {'type': 'match', 'id': '70526dba-9f30-4fbc-8b23-90d1d0bac580'},
         {'type': 'match', 'id': '3bb2015a-8588-4ed5-b7fd-7d9121683f3b'},
         {'type': 'match', 'id': '7f5d7c31-92d9-4e9c-85a4-84d491b2a0e6'},
         {'type': 'match', 'id': 'b5add322-a6b4-4617-8e40-ee5188bbb894'},
         {'type': 'match', 'id': 'e3eee3ef-5e00-45f4-9b0c-a39c8ced541a'},
         {'type': 'match', 'id': '23ebd02c-065d-4bb3-91c2-058e16c39bb3'},
         {'type': 'match', 'id': '7fbb7ec7-bcfb-4f4b-98a7-369b588b2ce9'},
         {'type': 'match', 'id': 'f2f08322-2b22-406e-b504-3b873274d618'},
         {'type': 'match', 'id': 'b63f2031-c72b-4a36-a7bf-783fe6ae525a'},
         {'type': 'match', 'id': '7eb3b00b-8321-4605-8991-56835da566ae'},
         {'type': 'match', 'id': '1ba83b28-5d8a-4042-a824-45f694a25059'},
         {'type': 'match', 'id': '240fd3df-f410-4a61-8172-1c078a0456a1'},
         {'type': 'match', 'id': '8ce54d9f-1c6b-44f8-bd21-20c314bafc2c'},
         {'type': 'match', 'id': '318052e5-da5e-41ee-8a71-40aabf75a553'},
         {'type': 'match', 'id': '932da511-6e11-4e7d-8f01-88f9bb208e15'},
         {'type': 'match', 'id': '8e6c3701-3cb5-4d0e-96ef-aedaa55b8813'},
         {'type': 'match', 'id': '1e201544-4d59-4c36-a259-d2529d4c0b6b'},
         {'type': 'match', 'id': '0e4e7aaa-3b2c-4048-8318-b160e9939769'},
         {'type': 'match', 'id': '6ff3c0f5-483d-4c09-986d-f021cf09fd2e'},
         {'type': 'match', 'id': 'af2e8a26-419a-40e4-b7ae-9bad7789a487'},
         {'type': 'match', 'id': 'e727d9cf-fce0-4fff-a548-ec8b12395705'},
         {'type': 'match', 'id': '31e91af9-4f2b-45c3-a033-22accc34a212'},
         {'type': 'match', 'id': 'bbdfd6ed-bb52-46fc-9d7e-af791fbfbac1'},
         {'type': 'match', 'id': '94027cf7-38ee-4dcb-b85e-13d5d92e4280'},
         {'type': 'match', 'id': '5b7c365c-66e2-453c-a0dd-a1062d8c6882'},
         {'type': 'match', 'id': 'c4873067-9521-4bce-9e19-5d7678c76ef2'},
         {'type': 'match', 'id': '95a3976c-3487-4f05-8266-6c6745cfa58f'},
         {'type': 'match', 'id': 'e83b2eea-e404-42ad-ab23-e71bddb23eb0'},
         {'type': 'match', 'id': '41380e68-bd56-471a-9cdf-d9766d06a76f'},
         {'type': 'match', 'id': '50d55caa-1797-49e4-bc5d-27957e85fc82'},
         {'type': 'match', 'id': '4cf4dc9e-8d82-4801-ad13-688f12d123e9'},
         {'type': 'match', 'id': '1074498c-14ea-4bf8-a389-424eee3d6e3b'},
         {'type': 'match', 'id': '0b65639d-103c-4c60-96f4-7b9ada93e18a'},
         {'type': 'match', 'id': '45821922-204d-47bf-aecb-e66920810391'},
         {'type': 'match', 'id': 'fb3a2185-5a49-4f77-b10d-5a0ea3976dc6'},
         {'type': 'match', 'id': 'f2219ec6-a9af-40a8-a728-5994a81078de'},
         {'type': 'match', 'id': '57cf15be-0a8e-47d6-9cd5-16a871ade68d'},
         {'type': 'match', 'id': '3ace1250-c286-44b1-b89e-730501b71cea'},
         {'type': 'match', 'id': 'e306d511-fa81-4eae-9899-51fb40c059ae'},
         {'type': 'match', 'id': '6678fe52-21a2-498a-9c5d-e9eeb392a26f'},
         {'type': 'match', 'id': '1dd5446b-3bcc-43e5-8a77-5ec12fbe041e'},
         {'type': 'match', 'id': '87c5e8f6-5fa0-4b38-953f-ab61c2d166dd'},
         {'type': 'match', 'id': 'e9a7870d-381f-4533-a4f5-ea8ac328a594'},
         {'type': 'match', 'id': '97079dba-96fd-49d8-a3d8-8f56882f9427'},
         {'type': 'match', 'id': '1a193d4e-f09f-4c37-8ccb-b479947c3ce3'},
         {'type': 'match', 'id': 'cc720659-a822-4c52-9f3f-95cb41b05264'},
         {'type': 'match', 'id': '40e548ae-8b7e-4851-b789-b8cc6d9fdf00'},
         {'type': 'match', 'id': '68d7400f-9885-4fb4-a57b-38bea3bfd389'},
         {'type': 'match', 'id': 'd8b968c1-eda0-4882-99d9-04d7000c1176'},
         {'type': 'match', 'id': '0f3cd866-3c61-4ffc-8857-2c15b58e0de7'},
         {'type': 'match', 'id': 'b5f35c26-3d8f-46a4-bc51-d1fbc4717b39'},
         {'type': 'match', 'id': '0b3e9677-ab01-4306-9da9-78f1ee68c0ad'},
         {'type': 'match', 'id': '721cda05-b31f-4695-80f1-dae5fcd1ea16'},
         {'type': 'match', 'id': '70a38158-e112-4cb1-adbc-565341a8b0ff'},
         {'type': 'match', 'id': 'e18c86a3-a08c-40e8-9cec-4c7274e1722c'},
         {'type': 'match', 'id': '31dfa5d1-8f04-4d8a-9e9a-bd7eb6dd41a5'},
         {'type': 'match', 'id': 'd0a4e621-d13e-4c04-9fe7-9a2dae50999e'},
         {'type': 'match', 'id': '83d2a8fc-5f6d-4bc7-b1e6-9fe06a14b66c'},
         {'type': 'match', 'id': 'db731b43-8ecb-40ba-bf82-aa16ed06e037'},
         {'type': 'match', 'id': 'd3b15c5c-cafa-4a80-8cec-722fe6401813'},
         {'type': 'match', 'id': '770e784c-5a03-4851-af13-13d1abdd90fe'},
         {'type': 'match', 'id': '7a84a088-574f-40a5-9dc6-746b82cbc819'},
         {'type': 'match', 'id': '1494c9aa-9917-4f13-b36f-2b3b7e8290fe'},
         {'type': 'match', 'id': 'd9672e5a-5364-4a03-a8b0-9731d6c76152'},
         {'type': 'match', 'id': '4107487e-c7eb-409d-8bbe-53e37773d28e'},
         {'type': 'match', 'id': '25f72d2d-d361-4f59-ac8d-57b18b92e15b'},
         {'type': 'match', 'id': '269eb667-dddd-4aec-8dbb-ebb9221ce5dd'},
         {'type': 'match', 'id': '73648bdf-69b4-4836-844d-4a20fece5c29'},
         {'type': 'match', 'id': '5ebeb8ef-4890-4a07-bf06-be2969a4eaf4'},
         {'type': 'match', 'id': 'c4bb04f2-d07c-4c68-a972-ffb25eaee00f'},
         {'type': 'match', 'id': 'fae38a34-3ccf-4d86-96e3-d913c4088b20'},
         {'type': 'match', 'id': '1dde4b76-a74e-4456-b3ee-c044468b29e8'},
         {'type': 'match', 'id': 'd4dcc5f1-64b2-4b49-84f1-895aaec42929'},
         {'type': 'match', 'id': 'aebf2289-0fb2-42b1-8130-dc813eae8256'},
         {'type': 'match', 'id': 'a466ff36-c052-4242-92a1-1c07929ffd32'},
         {'type': 'match', 'id': '47d1faaa-fdae-4f31-bec0-ebe13b144296'},
         {'type': 'match', 'id': '57c66676-7282-4840-8b27-44d50313c8aa'},
         {'type': 'match', 'id': '6241c305-35da-4509-8d6c-72268865d914'},
         {'type': 'match', 'id': '5d975312-d771-4d7f-9de3-039d2a182ef8'},
         {'type': 'match', 'id': '04ed8681-1b8a-4aa6-bee6-4a0360ad24ea'},
         {'type': 'match', 'id': '8e48a620-36b1-41f5-9bc5-dd35cfc5ebf0'},
         {'type': 'match', 'id': '0c92a712-45b2-4bb6-af9d-2ae4813eb5fc'},
         {'type': 'match', 'id': '547ab699-691c-4c2a-8cf1-9a974dd42d02'},
         {'type': 'match', 'id': 'a1ed9774-a64a-4f48-ad42-3247c6fbdb1e'},
         {'type': 'match', 'id': 'b3372420-9b56-421c-b67c-4e83710ba0ce'},
         {'type': 'match', 'id': '11ec1322-fc32-47b6-a5dd-c4f10c519902'},
         {'type': 'match', 'id': '50235518-fea9-4c0d-b615-9499f43a22fa'},
         {'type': 'match', 'id': 'd959f7c3-1b29-4fe6-8934-15994e774e8a'},
         {'type': 'match', 'id': '2cfc5208-c11c-4ad8-8fa9-5ccd2f7016f1'},
         {'type': 'match', 'id': '2f97533b-34ff-4e96-a574-aafbd638a390'},
         {'type': 'match', 'id': '92df5f04-84df-4779-9546-e27935194e8d'},
         {'type': 'match', 'id': 'ab3e3de2-e645-4aac-a055-959dbf7cd8ee'},
         {'type': 'match', 'id': 'b503a7e9-9b14-437b-b4e7-a98c1b5d91e0'},
         {'type': 'match', 'id': 'd5cc2593-d715-4a8e-a7ea-7e5379fa1ea2'},
         {'type': 'match', 'id': '93d25bf4-46c3-42e6-88e6-27ec91242647'},
         {'type': 'match', 'id': 'b335a99f-44b0-47fc-b781-2172eece5593'},
         {'type': 'match', 'id': 'c3cfe2f2-8746-4faf-ba70-828cf6e19668'},
         {'type': 'match', 'id': '6188d175-54fa-4d0d-a1d9-932b2fac23ab'}]}}},
     'included': [{'type': 'match',
       'id': '59dc37a9-aee1-4bdb-8326-f46335f0033f',
       'attributes': {'createdAt': '2019-02-07T05:49:39Z'}},
      {'type': 'match',
       'id': 'b5add322-a6b4-4617-8e40-ee5188bbb894',
       'attributes': {'createdAt': '2019-01-04T12:24:53Z'}},
      {'type': 'match',
       'id': '7eb3b00b-8321-4605-8991-56835da566ae',
       'attributes': {'createdAt': '2018-12-01T11:09:27Z'}},
      {'type': 'match',
       'id': '318052e5-da5e-41ee-8a71-40aabf75a553',
       'attributes': {'createdAt': '2018-11-23T11:54:53Z'}},
      {'type': 'match',
       'id': 'af2e8a26-419a-40e4-b7ae-9bad7789a487',
       'attributes': {'createdAt': '2018-11-21T10:42:51Z'}},
      {'type': 'match',
       'id': '6188d175-54fa-4d0d-a1d9-932b2fac23ab',
       'attributes': {'createdAt': '2018-10-01T09:13:33Z'}},
      {'type': 'match',
       'id': '932da511-6e11-4e7d-8f01-88f9bb208e15',
       'attributes': {'createdAt': '2018-11-23T11:04:39Z'}},
      {'type': 'match',
       'id': '83d2a8fc-5f6d-4bc7-b1e6-9fe06a14b66c',
       'attributes': {'createdAt': '2018-10-24T09:12:00Z'}},
      {'type': 'match',
       'id': 'b335a99f-44b0-47fc-b781-2172eece5593',
       'attributes': {'createdAt': '2018-10-01T10:47:39Z'}},
      {'type': 'match',
       'id': '5b7c365c-66e2-453c-a0dd-a1062d8c6882',
       'attributes': {'createdAt': '2018-11-19T10:33:02Z'}},
      {'type': 'match',
       'id': '3ace1250-c286-44b1-b89e-730501b71cea',
       'attributes': {'createdAt': '2018-11-05T09:07:29Z'}},
      {'type': 'match',
       'id': '70a38158-e112-4cb1-adbc-565341a8b0ff',
       'attributes': {'createdAt': '2018-10-26T09:08:53Z'}},
      {'type': 'match',
       'id': 'a466ff36-c052-4242-92a1-1c07929ffd32',
       'attributes': {'createdAt': '2018-10-12T11:26:03Z'}},
      {'type': 'match',
       'id': '2f97533b-34ff-4e96-a574-aafbd638a390',
       'attributes': {'createdAt': '2018-10-05T09:08:49Z'}},
      {'type': 'match',
       'id': '50d55caa-1797-49e4-bc5d-27957e85fc82',
       'attributes': {'createdAt': '2018-11-09T09:07:11Z'}},
      {'type': 'match',
       'id': '0b3e9677-ab01-4306-9da9-78f1ee68c0ad',
       'attributes': {'createdAt': '2018-10-26T10:58:02Z'}},
      {'type': 'match',
       'id': 'aebf2289-0fb2-42b1-8130-dc813eae8256',
       'attributes': {'createdAt': '2018-10-15T09:13:42Z'}},
      {'type': 'match',
       'id': '92df5f04-84df-4779-9546-e27935194e8d',
       'attributes': {'createdAt': '2018-10-03T11:32:52Z'}},
      {'type': 'match',
       'id': 'b63f2031-c72b-4a36-a7bf-783fe6ae525a',
       'attributes': {'createdAt': '2018-12-01T11:55:29Z'}},
      {'type': 'match',
       'id': '95a3976c-3487-4f05-8266-6c6745cfa58f',
       'attributes': {'createdAt': '2018-11-09T11:36:38Z'}},
      {'type': 'match',
       'id': '31dfa5d1-8f04-4d8a-9e9a-bd7eb6dd41a5',
       'attributes': {'createdAt': '2018-10-24T10:38:09Z'}},
      {'type': 'match',
       'id': '08e38587-85f1-4730-b073-b6ea5e6daf66',
       'attributes': {'createdAt': '2019-02-07T06:09:17Z'}},
      {'type': 'match',
       'id': '1074498c-14ea-4bf8-a389-424eee3d6e3b',
       'attributes': {'createdAt': '2018-11-07T10:32:02Z'}},
      {'type': 'match',
       'id': '97079dba-96fd-49d8-a3d8-8f56882f9427',
       'attributes': {'createdAt': '2018-10-31T10:51:41Z'}},
      {'type': 'match',
       'id': '11ec1322-fc32-47b6-a5dd-c4f10c519902',
       'attributes': {'createdAt': '2018-10-08T09:08:04Z'}},
      {'type': 'match',
       'id': '1ba83b28-5d8a-4042-a824-45f694a25059',
       'attributes': {'createdAt': '2018-12-01T10:18:49Z'}},
      {'type': 'match',
       'id': '68d7400f-9885-4fb4-a57b-38bea3bfd389',
       'attributes': {'createdAt': '2018-10-29T10:45:48Z'}},
      {'type': 'match',
       'id': '47d1faaa-fdae-4f31-bec0-ebe13b144296',
       'attributes': {'createdAt': '2018-10-12T10:45:31Z'}},
      {'type': 'match',
       'id': '721cda05-b31f-4695-80f1-dae5fcd1ea16',
       'attributes': {'createdAt': '2018-10-26T09:56:40Z'}},
      {'type': 'match',
       'id': '0c92a712-45b2-4bb6-af9d-2ae4813eb5fc',
       'attributes': {'createdAt': '2018-10-10T09:09:56Z'}},
      {'type': 'match',
       'id': '2cfc5208-c11c-4ad8-8fa9-5ccd2f7016f1',
       'attributes': {'createdAt': '2018-10-05T09:51:34Z'}},
      {'type': 'match',
       'id': '7428419d-81ec-45a9-ba4b-8a30f809e226',
       'attributes': {'createdAt': '2019-02-08T10:19:17Z'}},
      {'type': 'match',
       'id': '770e784c-5a03-4851-af13-13d1abdd90fe',
       'attributes': {'createdAt': '2018-10-22T09:46:30Z'}},
      {'type': 'match',
       'id': '5d975312-d771-4d7f-9de3-039d2a182ef8',
       'attributes': {'createdAt': '2018-10-10T11:31:53Z'}},
      {'type': 'match',
       'id': '04ed8681-1b8a-4aa6-bee6-4a0360ad24ea',
       'attributes': {'createdAt': '2018-10-10T10:49:24Z'}},
      {'type': 'match',
       'id': '0f3cd866-3c61-4ffc-8857-2c15b58e0de7',
       'attributes': {'createdAt': '2018-10-29T09:13:08Z'}},
      {'type': 'match',
       'id': 'e18c86a3-a08c-40e8-9cec-4c7274e1722c',
       'attributes': {'createdAt': '2018-10-24T11:18:20Z'}},
      {'type': 'match',
       'id': '1a193d4e-f09f-4c37-8ccb-b479947c3ce3',
       'attributes': {'createdAt': '2018-10-31T10:01:07Z'}},
      {'type': 'match',
       'id': 'db731b43-8ecb-40ba-bf82-aa16ed06e037',
       'attributes': {'createdAt': '2018-10-22T11:41:14Z'}},
      {'type': 'match',
       'id': '547ab699-691c-4c2a-8cf1-9a974dd42d02',
       'attributes': {'createdAt': '2018-10-08T11:19:49Z'}},
      {'type': 'match',
       'id': 'd959f7c3-1b29-4fe6-8934-15994e774e8a',
       'attributes': {'createdAt': '2018-10-05T10:37:03Z'}},
      {'type': 'match',
       'id': '7fbb7ec7-bcfb-4f4b-98a7-369b588b2ce9',
       'attributes': {'createdAt': '2019-01-04T10:18:46Z'}},
      {'type': 'match',
       'id': '57cf15be-0a8e-47d6-9cd5-16a871ade68d',
       'attributes': {'createdAt': '2018-11-05T09:55:18Z'}},
      {'type': 'match',
       'id': '7cac694a-575f-46ad-8101-e3226cfbaf10',
       'attributes': {'createdAt': '2019-02-10T07:51:10Z'}},
      {'type': 'match',
       'id': '8ce54d9f-1c6b-44f8-bd21-20c314bafc2c',
       'attributes': {'createdAt': '2018-12-01T08:29:02Z'}},
      {'type': 'match',
       'id': '1e201544-4d59-4c36-a259-d2529d4c0b6b',
       'attributes': {'createdAt': '2018-11-23T09:27:22Z'}},
      {'type': 'match',
       'id': 'e9a7870d-381f-4533-a4f5-ea8ac328a594',
       'attributes': {'createdAt': '2018-10-31T11:40:56Z'}},
      {'type': 'match',
       'id': 'c3cfe2f2-8746-4faf-ba70-828cf6e19668',
       'attributes': {'createdAt': '2018-10-01T09:56:38Z'}},
      {'type': 'match',
       'id': '240fd3df-f410-4a61-8172-1c078a0456a1',
       'attributes': {'createdAt': '2018-12-01T09:31:08Z'}},
      {'type': 'match',
       'id': '31e91af9-4f2b-45c3-a033-22accc34a212',
       'attributes': {'createdAt': '2018-11-21T09:12:03Z'}},
      {'type': 'match',
       'id': '8e48a620-36b1-41f5-9bc5-dd35cfc5ebf0',
       'attributes': {'createdAt': '2018-10-10T10:05:31Z'}},
      {'type': 'match',
       'id': 'd5cc2593-d715-4a8e-a7ea-7e5379fa1ea2',
       'attributes': {'createdAt': '2018-10-03T09:12:23Z'}},
      {'type': 'match',
       'id': '0e4e7aaa-3b2c-4048-8318-b160e9939769',
       'attributes': {'createdAt': '2018-11-23T09:19:39Z'}},
      {'type': 'match',
       'id': 'e306d511-fa81-4eae-9899-51fb40c059ae',
       'attributes': {'createdAt': '2018-11-02T11:36:14Z'}},
      {'type': 'match',
       'id': '5ebeb8ef-4890-4a07-bf06-be2969a4eaf4',
       'attributes': {'createdAt': '2018-10-17T09:58:32Z'}},
      {'type': 'match',
       'id': 'c4bb04f2-d07c-4c68-a972-ffb25eaee00f',
       'attributes': {'createdAt': '2018-10-17T09:13:42Z'}},
      {'type': 'match',
       'id': 'ab7c47b7-5d1f-465f-8843-44b552809881',
       'attributes': {'createdAt': '2019-02-10T07:11:22Z'}},
      {'type': 'match',
       'id': 'ab3e3de2-e645-4aac-a055-959dbf7cd8ee',
       'attributes': {'createdAt': '2018-10-03T10:42:23Z'}},
      {'type': 'match',
       'id': 'bbdfd6ed-bb52-46fc-9d7e-af791fbfbac1',
       'attributes': {'createdAt': '2018-11-19T12:01:03Z'}},
      {'type': 'match',
       'id': 'd8b968c1-eda0-4882-99d9-04d7000c1176',
       'attributes': {'createdAt': '2018-10-29T10:00:08Z'}},
      {'type': 'match',
       'id': '7a84a088-574f-40a5-9dc6-746b82cbc819',
       'attributes': {'createdAt': '2018-10-22T09:05:54Z'}},
      {'type': 'match',
       'id': '1dd5446b-3bcc-43e5-8a77-5ec12fbe041e',
       'attributes': {'createdAt': '2018-11-02T09:54:11Z'}},
      {'type': 'match',
       'id': 'cc720659-a822-4c52-9f3f-95cb41b05264',
       'attributes': {'createdAt': '2018-10-31T09:12:52Z'}},
      {'type': 'match',
       'id': '40e548ae-8b7e-4851-b789-b8cc6d9fdf00',
       'attributes': {'createdAt': '2018-10-29T11:35:08Z'}},
      {'type': 'match',
       'id': '96528262-a656-4480-86ef-90746bdae198',
       'attributes': {'createdAt': '2019-01-05T11:43:35Z'}},
      {'type': 'match',
       'id': '0b65639d-103c-4c60-96f4-7b9ada93e18a',
       'attributes': {'createdAt': '2018-11-07T09:46:28Z'}},
      {'type': 'match',
       'id': '45821922-204d-47bf-aecb-e66920810391',
       'attributes': {'createdAt': '2018-11-07T09:05:53Z'}},
      {'type': 'match',
       'id': '6678fe52-21a2-498a-9c5d-e9eeb392a26f',
       'attributes': {'createdAt': '2018-11-02T10:45:34Z'}},
      {'type': 'match',
       'id': '1494c9aa-9917-4f13-b36f-2b3b7e8290fe',
       'attributes': {'createdAt': '2018-10-19T11:30:58Z'}},
      {'type': 'match',
       'id': '93d25bf4-46c3-42e6-88e6-27ec91242647',
       'attributes': {'createdAt': '2018-10-01T11:35:35Z'}},
      {'type': 'match',
       'id': '77f61312-b6d6-4357-b163-f979a35f20fd',
       'attributes': {'createdAt': '2019-01-05T12:25:28Z'}},
      {'type': 'match',
       'id': '70526dba-9f30-4fbc-8b23-90d1d0bac580',
       'attributes': {'createdAt': '2019-01-05T11:02:20Z'}},
      {'type': 'match',
       'id': 'd0a4e621-d13e-4c04-9fe7-9a2dae50999e',
       'attributes': {'createdAt': '2018-10-24T09:52:55Z'}},
      {'type': 'match',
       'id': 'd3b15c5c-cafa-4a80-8cec-722fe6401813',
       'attributes': {'createdAt': '2018-10-22T11:01:22Z'}},
      {'type': 'match',
       'id': '269eb667-dddd-4aec-8dbb-ebb9221ce5dd',
       'attributes': {'createdAt': '2018-10-17T11:43:38Z'}},
      {'type': 'match',
       'id': 'f169b923-612f-42f5-a45b-7b2e4221d92d',
       'attributes': {'createdAt': '2019-02-07T06:36:32Z'}},
      {'type': 'match',
       'id': 'e3eee3ef-5e00-45f4-9b0c-a39c8ced541a',
       'attributes': {'createdAt': '2019-01-04T11:45:06Z'}},
      {'type': 'match',
       'id': '8e6c3701-3cb5-4d0e-96ef-aedaa55b8813',
       'attributes': {'createdAt': '2018-11-23T10:15:56Z'}},
      {'type': 'match',
       'id': '94027cf7-38ee-4dcb-b85e-13d5d92e4280',
       'attributes': {'createdAt': '2018-11-19T11:16:08Z'}},
      {'type': 'match',
       'id': 'fb3a2185-5a49-4f77-b10d-5a0ea3976dc6',
       'attributes': {'createdAt': '2018-11-05T11:36:56Z'}},
      {'type': 'match',
       'id': '87c5e8f6-5fa0-4b38-953f-ab61c2d166dd',
       'attributes': {'createdAt': '2018-11-02T09:08:21Z'}},
      {'type': 'match',
       'id': '23ebd02c-065d-4bb3-91c2-058e16c39bb3',
       'attributes': {'createdAt': '2019-01-04T11:02:52Z'}},
      {'type': 'match',
       'id': 'b5f35c26-3d8f-46a4-bc51-d1fbc4717b39',
       'attributes': {'createdAt': '2018-10-26T11:39:43Z'}},
      {'type': 'match',
       'id': '6241c305-35da-4509-8d6c-72268865d914',
       'attributes': {'createdAt': '2018-10-12T09:07:59Z'}},
      {'type': 'match',
       'id': 'b503a7e9-9b14-437b-b4e7-a98c1b5d91e0',
       'attributes': {'createdAt': '2018-10-03T09:54:25Z'}},
      {'type': 'match',
       'id': '277295d2-e148-4749-92e8-7c00f3a23219',
       'attributes': {'createdAt': '2019-02-11T06:25:33Z'}},
      {'type': 'match',
       'id': '7f5d7c31-92d9-4e9c-85a4-84d491b2a0e6',
       'attributes': {'createdAt': '2019-01-05T09:32:36Z'}},
      {'type': 'match',
       'id': '73648bdf-69b4-4836-844d-4a20fece5c29',
       'attributes': {'createdAt': '2018-10-17T10:47:06Z'}},
      {'type': 'match',
       'id': 'fae38a34-3ccf-4d86-96e3-d913c4088b20',
       'attributes': {'createdAt': '2018-10-15T11:38:49Z'}},
      {'type': 'match',
       'id': 'b3372420-9b56-421c-b67c-4e83710ba0ce',
       'attributes': {'createdAt': '2018-10-08T09:52:18Z'}},
      {'type': 'match',
       'id': 'f2f08322-2b22-406e-b504-3b873274d618',
       'attributes': {'createdAt': '2019-01-04T09:32:43Z'}},
      {'type': 'match',
       'id': 'f2219ec6-a9af-40a8-a728-5994a81078de',
       'attributes': {'createdAt': '2018-11-05T10:55:42Z'}},
      {'type': 'match',
       'id': '4107487e-c7eb-409d-8bbe-53e37773d28e',
       'attributes': {'createdAt': '2018-10-19T09:52:13Z'}},
      {'type': 'match',
       'id': 'd4dcc5f1-64b2-4b49-84f1-895aaec42929',
       'attributes': {'createdAt': '2018-10-15T09:59:10Z'}},
      {'type': 'match',
       'id': '41380e68-bd56-471a-9cdf-d9766d06a76f',
       'attributes': {'createdAt': '2018-11-09T09:53:15Z'}},
      {'type': 'match',
       'id': '1dde4b76-a74e-4456-b3ee-c044468b29e8',
       'attributes': {'createdAt': '2018-10-15T10:46:57Z'}},
      {'type': 'match',
       'id': '3bb2015a-8588-4ed5-b7fd-7d9121683f3b',
       'attributes': {'createdAt': '2019-01-05T10:18:04Z'}},
      {'type': 'match',
       'id': 'c4873067-9521-4bce-9e19-5d7678c76ef2',
       'attributes': {'createdAt': '2018-11-19T09:21:48Z'}},
      {'type': 'match',
       'id': '4cf4dc9e-8d82-4801-ad13-688f12d123e9',
       'attributes': {'createdAt': '2018-11-07T11:11:41Z'}},
      {'type': 'match',
       'id': 'd9672e5a-5364-4a03-a8b0-9731d6c76152',
       'attributes': {'createdAt': '2018-10-19T10:39:18Z'}},
      {'type': 'match',
       'id': '25f72d2d-d361-4f59-ac8d-57b18b92e15b',
       'attributes': {'createdAt': '2018-10-19T09:08:43Z'}},
      {'type': 'match',
       'id': '57c66676-7282-4840-8b27-44d50313c8aa',
       'attributes': {'createdAt': '2018-10-12T09:51:24Z'}},
      {'type': 'match',
       'id': 'a1ed9774-a64a-4f48-ad42-3247c6fbdb1e',
       'attributes': {'createdAt': '2018-10-08T10:39:00Z'}},
      {'type': 'match',
       'id': '6ff3c0f5-483d-4c09-986d-f021cf09fd2e',
       'attributes': {'createdAt': '2018-11-21T11:33:42Z'}},
      {'type': 'match',
       'id': 'e727d9cf-fce0-4fff-a548-ec8b12395705',
       'attributes': {'createdAt': '2018-11-21T09:56:50Z'}},
      {'type': 'match',
       'id': 'e83b2eea-e404-42ad-ab23-e71bddb23eb0',
       'attributes': {'createdAt': '2018-11-09T10:55:53Z'}},
      {'type': 'match',
       'id': '50235518-fea9-4c0d-b615-9499f43a22fa',
       'attributes': {'createdAt': '2018-10-05T11:24:36Z'}}],
     'links': {'self': 'https://api.pubg.com/tournaments/kr-pkl18'},
     'meta': {}}




```python
# kr-pkl18 데이터 matchId 저장

matchId_dict = {match['attributes']['createdAt']: match['id'] for match in league_json['included']}
matchId_df = pd.DataFrame(sorted(matchId_dict.items(), key=lambda x: x[0]),
                              columns=['createdAt', 'matchId'])
# Erangel 맵 데이터만 사용
erangelMatch_df = match_df[match_df['mapName'] == 'Erangel_Main'].loc[:90]
# Rangers 팀이 참가한 데이터만 사용
pubg = PUBG(api_key, shard='tournament')
matchId_rangers = []
for matchId in tqdm(erangelMatch_df['matchId']):
    current_match = pubg.match(matchId)
    telemetry = current_match.get_telemetry()
    telemetry.player_names()
    findCount = (pd.Series(telemetry.player_names()).str.find('Rangers') > -1).sum()
    if findCount > 0:
        matchId_rangers.append(matchId)
```


```python
# 경기 후의 stats 정보를 데이터프레임 안의 데이터프레임으로 저장
# 비효율적. 이번 분석에서는 stats 정보를 이용하지 않았음

# apply lambda를 위한 함수 (시리즈 변환)
def ParticipantsMerge(row):
    participants = pd.Series(row['attributes']['stats'])
    participants['id'] = row['id']
    return participants

headers = {'accept': 'application/vnd.api+json'}

match_df = pd.DataFrame()
participants_df = pd.DataFrame()
for matchId in matchId_rangers:
    url = f'https://api.pubg.com/shards/tournament/matches/{matchId}'
    r = requests.get(url, headers=headers)
    while r.status_code != 200:
        time.sleep(3)
        r = requests.get(url, headers=headers)
    data = r.json()
    stats_series = pd.Series(data['data']['attributes'])
    stats_series['matchId'] = matchId

    included = pd.DataFrame(data['included'])
    rosters = included[included['type'] == 'roster']
    rosters = rosters.apply(lambda row: [row['attributes']['stats']['rank'],
                                         row['attributes']['stats']['teamId'],
                                         row['attributes']['won'],
                                         pd.DataFrame(row['relationships']['participants']['data']),
                                         row['id']], axis=1)
    rosters = pd.DataFrame(list(rosters), columns=['rank', 'teamId', 'won', 'participants', 'id'])

    participants = included[included['type'] == 'participant']
    participants = participants.apply(lambda row: ParticipantsMerge(row), axis=1)
    participants = participants.reset_index(drop=True)
    participants = participants.drop(['killPoints', 'killPointsDelta', 'lastKillPoints', 'lastWinPoints',
                                      'mostDamage', 'rankPoints', 'winPoints', 'winPointsDelta'],
                                      axis=1, errors='ignore')

    rosters['participants'] = rosters['participants'].map(lambda x: x.merge(participants))

    stats_series['rosters'] = rosters

    participants_df = participants_df.append(participants, ignore_index=True, sort=False)
    match_df = match_df.append(stats_series, ignore_index=True, sort=False)

    match_df = match_df.drop(['stats', 'titleId', 'shardId', 'tags', 'gameMode', 'isCustomMatch'], axis=1)
```


```python
# 교전 횟수, 첫 차량 탑승, 화이트&블루존과의 상대적 거리 구하기

attack_df = pd.DataFrame()
teamCount = {}
whiteCenter_df = pd.DataFrame()
blueBorder_df = pd.DataFrame()
playerMove_df = pd.DataFrame()
firstVehicle_df = pd.DataFrame()

for matchId in matchId_rangers:
    current_match = pubg.match(matchId) #chicken_dinner 라이브러리 이용
    telemetry = current_match.get_telemetry()
    positions = telemetry.player_positions()
    circles = telemetry.circle_positions()
    players = np.array(telemetry.player_names())
    rosters = players[np.where(pd.Series(players).str.find('Rangers') > -1)[0]]  # {Team}의 존 상대적 거리

    whiteCenter_means = {}
    blueBorder_means = {}
    playerMove_means = {}

    for player in rosters:
        curpos = np.array(positions[player])
        while len(circles['white']) < len(curpos): curpos = curpos[:-1]
        length = len(curpos)
        start = np.where(curpos[:, 3] < 30000)[0][0]

        curpos = curpos[start:]
        whites = np.array(circles['white'])[start:length]
        blues = np.array(circles['blue'])[start:length]
        white_xDiff = (whites[:, 1] - curpos[:, 1]); white_yDiff = (whites[:, 2] - curpos[:, 2])
        blue_xDiff = (blues[:, 1] - curpos[:, 1]); blue_yDiff = (blues[:, 2] - curpos[:, 2])

        phases = np.where(whites[1:, 4] - whites[:-1, 4] < 0)[0] + 1 # 단계 인덱싱 구하기
        phases = np.append(phases, len(whites))

        white_means = []
        blue_means = []
        moves = []
        pre = 0
        for phase in phases: #단계마다 공식 계산
            cur_white_xDiff = white_xDiff[pre:phase]; cur_white_yDiff = white_yDiff[pre:phase]
            cur_blue_xDiff = blue_xDiff[pre:phase]; cur_blue_yDiff = blue_yDiff[pre:phase]

            whiteCenter_diff = np.sqrt(np.square(cur_white_xDiff) + np.square(cur_white_yDiff)) / whites[pre:phase, 4]
            blueBorder_diff = (blues[pre:phase, 4] - np.sqrt(np.square(cur_blue_xDiff) + np.square(cur_blue_yDiff))) \
                                    / blues[pre:phase, 4]

            white_means.append(whiteCenter_diff.mean())
            blue_means.append(blueBorder_diff.mean())
            moves.append((whiteCenter_diff[1:] - whiteCenter_diff[:-1]).mean())
            pre = phase

        whiteCenter_means[player] = white_means
        blueBorder_means[player] = blue_means
        playerMove_means[player] = moves

    whiteCenter_df = pd.concat(
                        [whiteCenter_df, pd.DataFrame.from_dict(whiteCenter_means, orient='index').T.mean(axis=1)],
                        axis=1, sort=False)
    blueBorder_df = pd.concat(
                        [blueBorder_df, pd.DataFrame.from_dict(blueBorder_means, orient='index').T.mean(axis=1)],
                        axis=1, sort=False)
    playerMove_df = pd.concat(
                        [playerMove_df, pd.DataFrame.from_dict(playerMove_means, orient='index').T.mean(axis=1)],
                        axis=1, sort=False)

    # 게임 시작 시간
    startTime = pd.to_timedelta(telemetry.started()[telemetry.started().find('T')+1:-1])
    endTime = telemetry.events[-1].timestamp # 마지막 이벤트 시간
    endTime = (pd.to_timedelta(endTime[endTime.find('T')+1:-1]) - startTime).total_seconds()
    circles = telemetry.circle_positions()
    whites = np.array(circles['white'])
    phases = np.where(whites[1:, 4] - whites[:-1, 4] != 0)[0] + 1
    phaseTimes = whites[phases, 0]
    phaseTimes = np.append(phaseTimes, endTime)

    attackLog = telemetry.filter_by('log_player_attack') # 교전 (공격한 경우) 데이터
    attackData = [(log['attacker']['name'],
                   (pd.to_timedelta(log.timestamp[log.timestamp.find('T')+1:-1]) - startTime).total_seconds())
                    for log in attackLog if pd.to_timedelta(log.timestamp[log.timestamp.find('T')+1:-1]) > startTime]
    attackData = pd.DataFrame(attackData, columns=['name', 'time'])
    attackData['teamName'] = attackData['name'].str.extract(r'([0-9A-Za-z]+)\_') # 팀명 추출
    attackData['phase'] = np.nan
    for i in range(len(phaseTimes)-1):
        attackData.loc[(attackData['time'] < phaseTimes[i+1]) & (attackData['time'] > phaseTimes[i]), 'phase'] = i+1
    attack_df = pd.concat([attack_df, attackData], axis=0)

    for team in attackData['teamName'].unique():
        try:
            teamCount[team] += 1
        except KeyError:
            teamCount[team] = 1

    vehicles = telemetry.filter_by('log_vehicle_ride') # 차량 탑승 데이터
    firstVehicle = {}
    used_teamId = []
    for vehicle in vehicles: # 팀에서 첫 차량 탑승 경우만 구하기
        if vehicle['vehicle']['vehicle_type'] != 'WheeledVehicle' or \
            vehicle['character']['name'] in firstVehicle.keys() or \
            vehicle['character']['name'][:vehicle['character']['name'].find('_')] in used_teamId: continue
        firstVehicle[vehicle['character']['name'][:vehicle['character']['name'].find('_')]] = \
            ((pd.to_timedelta(vehicle.timestamp[vehicle.timestamp.find('T')+1:-1]) - startTime).total_seconds(), \
            vehicle['character']['location']['x'], mapy - vehicle['character']['location']['y'])
        used_teamId.append(vehicle['character']['name'][:vehicle['character']['name'].find('_')])
    firstVehicle_df = pd.concat([firstVehicle_df, pd.DataFrame(firstVehicle)], axis=1, sort=False)

firstVehicle_df = firstVehicle_df.T
firstVehicle_df.columns = ['time', 'x', 'y']
firstVehicle_df['teamName'] = firstVehicle_df.index
firstVehicle_team = pd.concat([firstVehicle_df[['teamName', 'time']].groupby('teamName').mean(),
                      firstVehicle_df[['teamName', 'time']].groupby('teamName').count()], axis=1, sort=False)
firstVehicle_team.columns = ['time', 'count']
firstVehicle_team = firstVehicle_team[firstVehicle_team['count'] > 10]

teamCount = pd.DataFrame(pd.Series(teamCount))
teamCount.columns = ['name']
teamCount.index.name = 'teamName'

teamAttack_df = attack_df[['teamName', 'phase', 'name']].groupby(['teamName', 'phase']).count()
teamAttack_df['countMean'] = (teamAttack_df / teamCount).round()
teamAttack_df = teamAttack_df.drop('name', axis=1)
teamAttack_df['phasePercent'] = (teamAttack_df / teamAttack_df.sum(level=0)).round(4) * 100
teamAttack_df.loc[['Rangers', 'Hunters', 'EntusA', 'EntusF'], :]
```

    0
    1
    2


    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:50: RuntimeWarning: Mean of empty slice.
    /usr/local/lib/python3.6/dist-packages/numpy/core/_methods.py:85: RuntimeWarning: invalid value encountered in double_scalars
      ret = ret.dtype.type(ret / rcount)


    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    15
    16
    17
    18
    19
    20
    21
    22
    23
    24
    25
    26
    27
    28
    29
    30
    31
    32
    33
    34
    35
    36
    37
    38
    39
    40
    41
    42





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
      <th></th>
      <th>countMean</th>
      <th>phasePercent</th>
    </tr>
    <tr>
      <th>teamName</th>
      <th>phase</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="8" valign="top">EntusA</th>
      <th>1.0</th>
      <td>40.0</td>
      <td>17.02</td>
    </tr>
    <tr>
      <th>2.0</th>
      <td>58.0</td>
      <td>24.68</td>
    </tr>
    <tr>
      <th>3.0</th>
      <td>44.0</td>
      <td>18.72</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>49.0</td>
      <td>20.85</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>29.0</td>
      <td>12.34</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>10.0</td>
      <td>4.26</td>
    </tr>
    <tr>
      <th>7.0</th>
      <td>2.0</td>
      <td>0.85</td>
    </tr>
    <tr>
      <th>9.0</th>
      <td>3.0</td>
      <td>1.28</td>
    </tr>
    <tr>
      <th rowspan="9" valign="top">EntusF</th>
      <th>1.0</th>
      <td>29.0</td>
      <td>11.65</td>
    </tr>
    <tr>
      <th>2.0</th>
      <td>48.0</td>
      <td>19.28</td>
    </tr>
    <tr>
      <th>3.0</th>
      <td>24.0</td>
      <td>9.64</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>49.0</td>
      <td>19.68</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>34.0</td>
      <td>13.65</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>41.0</td>
      <td>16.47</td>
    </tr>
    <tr>
      <th>7.0</th>
      <td>17.0</td>
      <td>6.83</td>
    </tr>
    <tr>
      <th>8.0</th>
      <td>6.0</td>
      <td>2.41</td>
    </tr>
    <tr>
      <th>9.0</th>
      <td>1.0</td>
      <td>0.40</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Hunters</th>
      <th>1.0</th>
      <td>15.0</td>
      <td>5.03</td>
    </tr>
    <tr>
      <th>2.0</th>
      <td>24.0</td>
      <td>8.05</td>
    </tr>
    <tr>
      <th>3.0</th>
      <td>38.0</td>
      <td>12.75</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>63.0</td>
      <td>21.14</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>60.0</td>
      <td>20.13</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>49.0</td>
      <td>16.44</td>
    </tr>
    <tr>
      <th>7.0</th>
      <td>42.0</td>
      <td>14.09</td>
    </tr>
    <tr>
      <th>8.0</th>
      <td>7.0</td>
      <td>2.35</td>
    </tr>
    <tr>
      <th rowspan="9" valign="top">Rangers</th>
      <th>1.0</th>
      <td>16.0</td>
      <td>6.04</td>
    </tr>
    <tr>
      <th>2.0</th>
      <td>34.0</td>
      <td>12.83</td>
    </tr>
    <tr>
      <th>3.0</th>
      <td>43.0</td>
      <td>16.23</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>64.0</td>
      <td>24.15</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>41.0</td>
      <td>15.47</td>
    </tr>
    <tr>
      <th>6.0</th>
      <td>37.0</td>
      <td>13.96</td>
    </tr>
    <tr>
      <th>7.0</th>
      <td>21.0</td>
      <td>7.92</td>
    </tr>
    <tr>
      <th>8.0</th>
      <td>8.0</td>
      <td>3.02</td>
    </tr>
    <tr>
      <th>9.0</th>
      <td>1.0</td>
      <td>0.38</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 첫 차량 탑승 지역 시각화

fig = plt.figure(figsize=(10, 10), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")
img_path = '/home/idea_demo/LPoint/KYH/BattleGround/Maps/Erangel_Main_High_Res.jpg'
img = mpimg.imread(img_path)
ax.imshow(img, extent=[0, mapx, 0, mapy])
xy = np.vstack([firstVehicle_df.x,firstVehicle_df.y])
z = gaussian_kde(xy)(xy) # 데이터 몰려있는 지역 확인
ax.scatter(firstVehicle_df.x, firstVehicle_df.y,
           marker="o", c=z, edgecolor="k", s=45, linewidths=0.8, zorder=20)
```




    <matplotlib.collections.PathCollection at 0x7fcc2ede4d68>




![1](/assets/post-image/2019-06-30-battleground-pro-pattern/1.png)



```python
# 경기 이동 경로 시각화

current_match = pubg.match(matchId_rangers[2]) #1주차 4라운드 경기
telemetry = current_match.get_telemetry()
positions = telemetry.player_positions()
circles = telemetry.circle_positions()
whites = np.array(circles['white'])
whites[:, 2] = mapy - whites[:, 2]
phases = np.where(whites[1:, 4] - whites[:-1, 4] != 0)[0] + 1

fig = plt.figure(figsize=(10, 10), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")
img_path = '/home/idea_demo/LPoint/KYH/BattleGround/Maps/Erangel_Main_High_Res.jpg'
img = mpimg.imread(img_path)
ax.imshow(img, extent=[0, mapx, 0, mapy])
for phase in phases:
    white_circle = plt.Circle((whites[phase][1], whites[phase][2]), whites[phase][4],
                                  edgecolor="w", linewidth=0.7, fill=False, zorder=5)
    ax.add_patch(white_circle)

startTime = pd.to_timedelta(telemetry.started()[telemetry.started().find('T')+1:-1])
unequips = telemetry.filter_by('log_item_unequip')
landing_locations = {unequip['character']['name']:
                        (unequip['character']['location']['x'], mapy - unequip['character']['location']['y'],
                        (pd.to_timedelta(unequip.timestamp[unequip.timestamp.find('T')+1:-1]) - startTime).total_seconds(),
                        unequip['character']['team_id'])
                        for unequip in unequips if unequip['item']['item_id'] == 'Item_Back_B_01_StartParachutePack_C'}
landing_locations = pd.DataFrame(landing_locations).T.reset_index()
landing_locations.columns = ['name', 'x', 'y', 'time', 'teamId']
landing_locations['teamId'] = landing_locations['teamId'].astype('int64')
landing_locations['teamName'] = landing_locations['name'].str.extract(r'([0-9A-Za-z]+)\_')

COLORS = {'ACTOZ': 'b', 'Rangers': 'r', 'GCBusan': 'c', 'ZDG': 'g'}
for player in positions.keys():
    if 'ZDG' not in player and 'Rangers' not in player and \
       'GCBusan' not in player and 'ACTOZ' not in player: continue
    curpos = np.array(positions[player])
    curpos[:, 2] = mapy - curpos[:, 2]
    curlanding = landing_locations[landing_locations['name'] == player]
    curpos = curpos[curpos[:, 0] > curlanding['time'].values[0]]
    ax.plot(curpos[:, 1], curpos[:, 2], '--', c=COLORS[curlanding['teamName'].values[0]], linewidth=2, zorder=20)
```


![2](/assets/post-image/2019-06-30-battleground-pro-pattern/2.png)



```python
# 경기 비행기라인 및 낙하산 거리 계산

current_match = pubg.match(matchId_rangers[2])
telemetry = current_match.get_telemetry()
positions = telemetry.player_positions()
circles = telemetry.circle_positions()
whites = np.array(circles['white'])
whites[:, 2] = mapy - whites[:, 2]
curpos = np.array(positions['Rangers_suk'])
curpos[:, 2] = mapy - curpos[:, 2]

unequips = telemetry.filter_by('log_item_unequip') # 아이템 제거한 이벤트
landing_locations = {unequip['character']['name']:
                        (unequip['character']['location']['x'], mapy - unequip['character']['location']['y'],
                         unequip['character']['team_id'])
                        for unequip in unequips if unequip['item']['item_id'] == 'Item_Back_B_01_StartParachutePack_C'} # 낙하산 제거
landing_locations = pd.DataFrame(landing_locations).T.reset_index()
landing_locations.columns = ['name', 'x', 'y', 'teamId']
landing_locations['teamId'] = landing_locations['teamId'].astype('int64')

map_id = telemetry.map_id()
mapx, mapy = map_dimensions[map_id]
#비행기 라인 기울기 및 절편 계산 (수송선 내에 있는 경우의 플레이어 데이터 이용)
slope1 = (curpos[0][2] - curpos[1][2]) / (curpos[0][1] - curpos[1][1])
beta1 = curpos[1][2] - curpos[1][1]*slope1
x = np.linspace(0, mapx, 100)
y = slope1*x + beta1
x = np.delete(x, np.where(y > mapy))
y = np.delete(y, np.where(y > mapy))
np.random.shuffle(COLORS)

map_range = (0, mapy)
map_range = np.array(map_range).astype('float64').reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 8))
scaler.fit(map_range)
# 낙하산 거리 계산 (선과 직선 사이의 수직선 거리 계산 공식 이용)
landing_locations['chuteDist'] = np.abs(slope1*landing_locations.x - landing_locations.y + beta1) / np.sqrt(slope1*slope1 + 1)
landing_locations['chuteDist'] = scaler.transform(landing_locations['chuteDist'].values.reshape(-1, 1))
landing_locations['teamName'] = landing_locations['name'].str.extract(r'([0-9A-Za-z]+)\_')
team_dists = landing_locations.groupby('teamName').mean()
team_dists['chuteDist'] = np.round(team_dists['chuteDist'], 2)

fig = plt.figure(figsize=(10, 10), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")
img_path = '/home/idea_demo/LPoint/KYH/BattleGround/Maps/Erangel_Main_High_Res.jpg'
img = mpimg.imread(img_path)
ax.imshow(img, extent=[0, mapx, 0, mapy])
# 화이트존 그리기
white_circle = plt.Circle((whites[1][1], whites[1][2]), whites[1][4], edgecolor="w", linewidth=2, fill=False, zorder=5)
ax.add_patch(white_circle)
ax.plot(x, y, 'r--', linewidth=3, zorder=20) # 비행기 라인

used_teamId = []
for index, row in landing_locations.iterrows():
    # 직교하는 선 구하기
    slope2 = -1 / slope1
    beta2 = row.y - row.x*slope2
    x2 = -(beta1-beta2)/(slope1-slope2)
    y2 = x2*slope1 + beta1

    if not row['teamId'] in used_teamId:  # 각 팀의 첫 번째만 글자 그리기
        teamName = row["name"][:row['name'].find('_')]
        label = ax.text(team_dists['x'][teamName], team_dists['y'][teamName] + np.random.randint(-10000, 10000),
                        '{}: {}Km'.format(teamName, team_dists['chuteDist'][teamName]),
                        color=COLORS[row['teamId']], size=10, zorder=22)
        label.set_path_effects([patheffects.withStroke(linewidth=1.5, foreground='k')])
        used_teamId.append(row['teamId'])
    ax.plot([x2, row.x], [y2, row.y], c=COLORS[row['teamId']], linestyle='--', linewidth=0.5, zorder=15) # 선수와 비행기 간의 선
    ax.scatter(row.x, row.y, marker="o", c=COLORS[row['teamId']], edgecolor="k", s=60, linewidths=1, zorder=20); # 선수
```


![3](/assets/post-image/2019-06-30-battleground-pro-pattern/3.png)



```python
# 시즌 전체의 낙하산 거리 계산

fig = plt.figure(figsize=(10, 10), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")
img = mpimg.imread(img_path)
ax.imshow(img, extent=[0, mapx, 0, mapy])

chute_dists = []
total_dists = pd.DataFrame()
whiteBetween_dists = pd.DataFrame()
for matchId in matchId_rangers:
    current_match = pubg.match(matchId)
    telemetry = current_match.get_telemetry()
    positions = telemetry.player_positions()
    curpos = np.array(positions[list(positions)[0]])
    curpos[:, 2] = mapy - curpos[:, 2]

    unequips = telemetry.filter_by('log_item_unequip')
    landing_locations = {unequip['character']['name']:
                            (unequip['character']['location']['x'], mapy - unequip['character']['location']['y'],
                             unequip['character']['team_id'])
                            for unequip in unequips if (unequip['item']['item_id'] == 'Item_Back_B_01_StartParachutePack_C')}
    landing_locations = pd.DataFrame(landing_locations).T.reset_index()
    landing_locations.columns = ['name', 'x', 'y', 'teamId']
    landing_locations['teamId'] = landing_locations['teamId'].astype('int64')

    slope1 = (curpos[0][2] - curpos[1][2]) / (curpos[0][1] - curpos[1][1])
    beta1 = curpos[1][2] - curpos[1][1]*slope1
    landing_locations['chuteDist'] = np.abs(slope1*landing_locations.x - landing_locations.y + beta1) / np.sqrt(slope1*slope1 + 1)
    landing_locations['chuteDist'] = scaler.transform(landing_locations['chuteDist'].values.reshape(-1, 1))
    landing_locations['teamName'] = landing_locations['name'].str.extract(r'([0-9A-Za-z]+)\_')
    team_dists = landing_locations.groupby('teamName').mean()
    team_dists['chuteDist'] = np.round(team_dists['chuteDist'], 2)
    total_dists = pd.concat([total_dists, team_dists], axis=0, sort=False)

    rangers = landing_locations[landing_locations['teamName'] == 'Rangers']
    hunters = landing_locations[landing_locations['teamName'] == 'Hunters']
    actoz = landing_locations[landing_locations['teamName'] == 'ZDG']
    entusf = landing_locations[landing_locations['teamName'] == 'EntusF']
    ax.scatter(rangers.x, rangers.y, marker="o", c='r', edgecolor="k", s=60, linewidths=1, zorder=21); # 시즌 전체의 착륙 위치
    ax.scatter(hunters.x, hunters.y, marker="o", c='c', edgecolor="k", s=60, linewidths=1, zorder=20);
    ax.scatter(actoz.x, actoz.y, marker="o", c='g', edgecolor="k", s=60, linewidths=1, zorder=19);
    ax.scatter(entusf.x, entusf.y, marker="o", c='b', edgecolor="k", s=60, linewidths=1, zorder=18);

total_dists['teamName'] = total_dists.index
total_dists = total_dists.reset_index(drop=True)
```


![4](/assets/post-image/2019-06-30-battleground-pro-pattern/4.png)



```python
total_dists[['teamName', 'chuteDist']].groupby('teamName').mean()
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
      <th>chuteDist</th>
    </tr>
    <tr>
      <th>teamName</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>247</th>
      <td>1.177778</td>
    </tr>
    <tr>
      <th>ACTOZ</th>
      <td>0.992791</td>
    </tr>
    <tr>
      <th>AFA</th>
      <td>1.127353</td>
    </tr>
    <tr>
      <th>AFF</th>
      <td>1.003548</td>
    </tr>
    <tr>
      <th>AGON</th>
      <td>0.818462</td>
    </tr>
    <tr>
      <th>APK</th>
      <td>1.090556</td>
    </tr>
    <tr>
      <th>BPT</th>
      <td>1.020000</td>
    </tr>
    <tr>
      <th>BSG</th>
      <td>1.076667</td>
    </tr>
    <tr>
      <th>C9</th>
      <td>0.978889</td>
    </tr>
    <tr>
      <th>DPG</th>
      <td>1.131667</td>
    </tr>
    <tr>
      <th>DPGA</th>
      <td>1.073333</td>
    </tr>
    <tr>
      <th>DTN</th>
      <td>1.208333</td>
    </tr>
    <tr>
      <th>EM</th>
      <td>1.126667</td>
    </tr>
    <tr>
      <th>EntusA</th>
      <td>0.966667</td>
    </tr>
    <tr>
      <th>EntusF</th>
      <td>0.534000</td>
    </tr>
    <tr>
      <th>GCBusan</th>
      <td>1.011613</td>
    </tr>
    <tr>
      <th>GEN</th>
      <td>0.939091</td>
    </tr>
    <tr>
      <th>Hunters</th>
      <td>1.118636</td>
    </tr>
    <tr>
      <th>KDG</th>
      <td>1.152222</td>
    </tr>
    <tr>
      <th>KDR</th>
      <td>0.693125</td>
    </tr>
    <tr>
      <th>MVP</th>
      <td>0.955333</td>
    </tr>
    <tr>
      <th>MVPL</th>
      <td>1.200625</td>
    </tr>
    <tr>
      <th>Maxtill</th>
      <td>1.124516</td>
    </tr>
    <tr>
      <th>NW</th>
      <td>0.687778</td>
    </tr>
    <tr>
      <th>Quadro</th>
      <td>0.925200</td>
    </tr>
    <tr>
      <th>ROGS</th>
      <td>1.010417</td>
    </tr>
    <tr>
      <th>ROX</th>
      <td>0.890000</td>
    </tr>
    <tr>
      <th>Rangers</th>
      <td>0.930000</td>
    </tr>
    <tr>
      <th>RoccatA</th>
      <td>0.997600</td>
    </tr>
    <tr>
      <th>RoccatI</th>
      <td>0.919583</td>
    </tr>
    <tr>
      <th>SKT</th>
      <td>1.188000</td>
    </tr>
    <tr>
      <th>WEGIRLS</th>
      <td>0.817500</td>
    </tr>
    <tr>
      <th>WeGirls</th>
      <td>1.140000</td>
    </tr>
    <tr>
      <th>ZDG</th>
      <td>0.885000</td>
    </tr>
    <tr>
      <th>danawa</th>
      <td>1.028333</td>
    </tr>
  </tbody>
</table>
</div>




```python
# chicken_dinner 데이터의 애니메이션 그리는 함수 코드
# mp4 파일 저장과 설정을 위해 일부 수정하였음

"""Function for generating playback animations."""
import logging
import os
import random

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib import rc
from matplotlib.animation import FuncAnimation

from chicken_dinner.constants import COLORS
from chicken_dinner.constants import map_dimensions


rc("animation", embed_limit=100)


MAP_ASSET_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.realpath('./KYH/BattleGround')
        )
    ),
    "Maps"
)


def create_playback_animation(
        telemetry,
        filename="playback.html",
        labels=True,
        disable_labels_after=None,
        label_players=None,
        dead_players=True,
        dead_player_labels=False,
        zoom=False,
        zoom_edge_buffer=0.5,
        use_hi_res=False,
        use_no_text=False,
        color_teams=True,
        highlight_teams=None,
        highlight_players=None,
        highlight_color="#FFFF00",
        highlight_winner=False,
        label_highlights=True,
        care_packages=True,
        damage=True,
        end_frames=20,
        size=10,
        dpi=200,
        interpolate=True,
        interval=1,
        fps=30,
):
    """Create a playback animation from telemetry data.
    Using matplotlib's animation library, create an HTML5 animation saved to
    disk relying on external ``ffmpeg`` library to create the video.
    To view the animation, open the resulting file in your browser.
    :param telemetry: an Telemetry instance
    :param filename: a file to generate for the animation (default
        "playback.html")
    :param bool labels: whether to label players by name
    :param int disable_labels_after: if passed, turns off player labels
        after number of seconds elapsed in game
    :param list label_players: a list of strings of player names that
        should be labeled
    :param bool dead_players: whether to mark dead players
    :param list dead_player_labels: a list of strings of players that
        should be labeled when dead
    :param bool zoom: whether to zoom with the circles through the playback
    :param float zoom_edge_buffer: how much to buffer the blue circle edge
        when zooming
    :param bool use_hi_res: whether to use the hi-res image, best to be set
        to True when using zoom
    :param bool use_no_text: whether to use the image with no text for
        town/location names
    :param bool color_teams: whether to color code different teams
    :param list highlight_teams: a list of strings of player names whose
        teams should be highlighted
    :param list highlight_players: a list of strings of player names who
        should be highlighted
    :param str highlight_color: a color to use for highlights
    :param bool highlight_winner: whether to highlight the winner(s)
    :param bool label_highlights: whether to label the highlights
    :param bool care_packages: whether to show care packages
    :param bool damage: whether to show PvP damage
    :param int end_frames: the number of extra end frames after game has
        been completed
    :param int size: the size of the resulting animation frame
    :param int dpi: the dpi to use when processing the animation
    :param bool interpolate: use linear interpolation to get frames with
        second-interval granularity
    :param int interval: interval between gameplay frames in seconds
    :param int fps: the frames per second for the animation
    """

    # Extract data
    positions = telemetry.player_positions()
    circles = telemetry.circle_positions()
    rankings = telemetry.rankings()
    winner = telemetry.winner()
    killed = telemetry.killed()
    rosters = telemetry.rosters()
    damages = telemetry.player_damages()
    package_spawns = telemetry.care_package_positions(land=False)
    package_lands = telemetry.care_package_positions(land=True)
    map_id = telemetry.map_id()
    mapx, mapy = map_dimensions[map_id]
    all_times = []
    for player, pos in positions.items():
        for p in pos:
            all_times.append(int(p[0]))
    all_times = sorted(list(set(all_times)))

    if label_players is None:
        label_players = []

    if highlight_players is None:
        highlight_players = []

    if highlight_winner:
        for player in winner:
            highlight_players.append(player)
        highlight_players = list(set(highlight_players))

    if highlight_teams is not None:
        for team_player in highlight_teams:
            for team_id, roster in rosters.items():
                if team_player in roster:
                    for player in roster:
                        highlight_players.append(player)
                    break

        highlight_players = list(set(highlight_players))

    if label_highlights:
        for player in highlight_players:
            label_players.append(player)

    label_players = list(set(label_players))

    team_colors = None
    if color_teams:
        # Randomly select colors from the pre-defined palette
        colors = COLORS
        idx = list(range(len(colors)))
        random.shuffle(idx)
        team_colors = {}
        count = 0
        for team_id, roster in rosters.items():
            for player in roster:
                team_colors[player] = colors[idx[count]]
            count += 1

    # Get the max "frame number"
    maxlength = 0
    for player, pos in positions.items():
        try:
            if pos[-1][0] > maxlength:
                maxlength = pos[-1][0]
        except IndexError:
            continue

    if interpolate:
        maxlength = max(all_times)
    else:
        maxlength = max([maxlength, len(circles)])

    # Initialize the plot and artist objects
    fig = plt.figure(frameon=False, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    if use_no_text:
        no_text = "_No_Text"
    else:
        no_text = ""

    # Sahnok (Savage_Main) high res is png for some reason; rest are jpg
    # Also Vikendi (DihorOtok_Main) is all png
    if use_hi_res:
        if map_id == "Savage_Main":
            if use_no_text:
                map_image = map_id + no_text + "_High_Res.jpg"
            else:
                map_image = map_id + no_text + "_High_Res.png"
        elif map_id == "DihorOtok_Main":
            map_image = map_id + no_text + "_High_Res.png"
        else:
            map_image = map_id + no_text + "_High_Res.jpg"
    else:
        if map_id == "DihorOtok_Main":
            map_image = map_id + no_text + "_High_Res.png"
        else:
            map_image = map_id + no_text + "_High_Res.jpg"
    img_path = os.path.join(MAP_ASSET_PATH, map_image)
    try:
        img = mpimg.imread(img_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            "High resolution images not included in package.\n"
            "Download images from https://github.com/pubg/api-assets/tree/master/Assets/Maps\n"
            "and place in folder: " + MAP_ASSET_PATH
        )
    ax.imshow(img, extent=[0, mapx, 0, mapy])

    players = ax.scatter(-10000, -10000, marker="o", c="w", edgecolor="k", s=60, linewidths=1, zorder=20)
    deaths = ax.scatter(-10000, -10000, marker="X", c="r", edgecolor="k", s=60, linewidths=1, alpha=0.5, zorder=10)

    highlights = ax.scatter(-10000, -10000, marker="*", c=highlight_color, edgecolor="k", s=180, linewidths=1, zorder=25)
    highlights_deaths = ax.scatter(-10000, -10000, marker="X", c=highlight_color, edgecolor="k", s=60, linewidths=1, zorder=15)

    if labels:
        if label_players is not None:
            name_labels = {
                player_name: ax.text(0, 0, player_name, size=8, zorder=19)
                for player_name in positions if player_name in label_players
            }
        else:
            name_labels = {
                player_name: ax.text(0, 0, player_name, size=8, zorder=19)
                for player_name in positions
            }
        for label in name_labels.values():
            label.set_path_effects([patheffects.withStroke(linewidth=2, foreground="w")])

    blue_circle = plt.Circle((0, 0), 0, edgecolor="b", linewidth=2, fill=False, zorder=5)
    white_circle = plt.Circle((0, 0), 0, edgecolor="w", linewidth=2, fill=False, zorder=6)
    red_circle = plt.Circle((0, 0), 0, color="r", edgecolor=None, lw=0, fill=True, alpha=0.3, zorder=7)

    care_package_spawns, = ax.plot(-10000, -10000, marker="s", c="w", markerfacecoloralt="w", fillstyle="bottom", mec="k", markeredgewidth=0.5, markersize=10, lw=0, zorder=8)
    care_package_lands, = ax.plot(-10000, -10000, marker="s", c="r", markerfacecoloralt="b", fillstyle="bottom", mec="k", markeredgewidth=0.5, markersize=10, lw=0, zorder=9)

    damage_slots = 50
    damage_lines = []
    for k in range(damage_slots):
        dline, = ax.plot(-10000, -10000, marker="x", c="r", mec="r", markeredgewidth=5, markevery=-1, markersize=10, lw=2, alpha=0.5, zorder=50)
        damage_lines.append(dline)

    ax.add_patch(blue_circle)
    ax.add_patch(white_circle)
    ax.add_patch(red_circle)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.set_size_inches((size, size))

    ax.set_xlim([0, mapx])
    ax.set_ylim([0, mapy])

    # Frame init function
    def init():
        if labels:
            if highlight_players or highlight_teams:
                updates = players, deaths, highlights, highlights_deaths, blue_circle, red_circle, white_circle, *tuple(name_labels.values())
            else:
                updates = players, deaths, blue_circle, red_circle, white_circle, *tuple(name_labels.values())
        else:
            if highlight_players or highlight_teams:
                updates = players, deaths, highlights, highlights_deaths, blue_circle, red_circle, white_circle
            else:
                updates = players, deaths, blue_circle, red_circle, white_circle
        if care_packages:
            updates = *updates, care_package_lands, care_package_spawns
        if damage:
            updates = *updates, *damage_lines
        return updates

    def interpolate_coords(t, coords, tidx, vidx, step=False):
        inter = False
        for idx, coord in enumerate(coords):
            if coord[tidx] > t:
                inter = True
                break

        if not inter:
            return coords[-1][vidx]

        if idx == 0:
            return coords[0][vidx]
        else:
            v0 = coords[idx - 1][vidx]
            t0 = coords[idx - 1][tidx]

        v1 = coords[idx][vidx]
        t1 = coords[idx][tidx]

        if step:
            return v1
        else:
            return v0 + (t - t0) * (v1 - v0) / (t1 - t0)

    # Frame update function
    def update(frame):
        logging.info("Processing frame {frame}".format(frame=frame))
        try:
            if interpolate:
                blue_circle.center = (
                    interpolate_coords(frame, circles["blue"], 0, 1),
                    mapy - interpolate_coords(frame, circles["blue"], 0, 2))
                red_circle.center = (
                    interpolate_coords(frame, circles["red"], 0, 1, True),
                    mapy - interpolate_coords(frame, circles["red"], 0, 2, True))
                white_circle.center = (
                    interpolate_coords(frame, circles["white"], 0, 1, True),
                    mapy - interpolate_coords(frame, circles["white"], 0, 2, True))

                blue_circle.set_radius(
                    interpolate_coords(frame, circles["blue"], 0, 4))
                red_circle.set_radius(
                    interpolate_coords(frame, circles["red"], 0, 4, True))
                white_circle.set_radius(
                    interpolate_coords(frame, circles["white"], 0, 4, True))
            else:
                blue_circle.center = circles["blue"][frame][1], mapy - circles["blue"][frame][2]
                red_circle.center = circles["red"][frame][1], mapy - circles["red"][frame][2]
                white_circle.center = circles["white"][frame][1], mapy - circles["white"][frame][2]

                blue_circle.set_radius(circles["blue"][frame][4])
                red_circle.set_radius(circles["red"][frame][4])
                white_circle.set_radius(circles["white"][frame][4])
        except IndexError:
            pass

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xwidth = xlim[1] - xlim[0]
        ywidth = ylim[1] - ylim[0]

        if zoom:
            try:
                if interpolate:
                    margin_offset = (1 + zoom_edge_buffer) * interpolate_coords(frame, circles["blue"], 0, 4)
                    xmin = max([0, interpolate_coords(frame, circles["blue"], 0, 1) - margin_offset])
                    xmax = min([mapx, interpolate_coords(frame, circles["blue"], 0, 1) + margin_offset])
                    ymin = max([0, mapy - interpolate_coords(frame, circles["blue"], 0, 2) - margin_offset])
                    ymax = min([mapy, mapy - interpolate_coords(frame, circles["blue"], 0, 2) + margin_offset])
                else:
                    margin_offset = (1 + zoom_edge_buffer) * circles["blue"][frame][4]
                    xmin = max([0, circles["blue"][frame][1] - margin_offset])
                    xmax = min([mapx, circles["blue"][frame][1] + margin_offset])
                    ymin = max([0, mapy - circles["blue"][frame][2] - margin_offset])
                    ymax = min([mapy, mapy - circles["blue"][frame][2] + margin_offset])

                # ensure full space taken by map
                if xmax - xmin >= ymax - ymin:
                    if ymin == 0:
                        ymax = ymin + (xmax - xmin)
                    elif ymax == mapy:
                        ymin = ymax - (xmax - xmin)
                else:
                    if xmin == 0:
                        xmax = xmin + (ymax - ymin)
                    elif xmax == mapx:
                        xmin = xmax - (ymax - ymin)

                ax.set_xlim([xmin, xmax])
                ax.set_ylim([ymin, ymax])

                xwidth = xmax - xmin
                ywidth = ymax - ymin
            except IndexError:
                pass

        positions_x = []
        positions_y = []
        highlights_x = []
        highlights_y = []
        deaths_x = []
        deaths_y = []
        highlights_deaths_x = []
        highlights_deaths_y = []
        care_package_lands_x = []
        care_package_lands_y = []
        care_package_spawns_x = []
        care_package_spawns_y = []

        if color_teams:
            marker_colors = []
            death_marker_colors = []
        else:
            marker_colors = "w"
            death_marker_colors = "r"

        t = 0
        damage_count = 0
        for player, pos in positions.items():
            try:
                player_max = pos[-1][0]
                # This ensures the alive winner(s) stay on the map at the end.
                if frame >= player_max and player not in winner:
                    raise IndexError
                elif frame >= player_max and player not in killed:
                    fidx = frame if interpolate else -1
                else:
                    fidx = frame

                if interpolate:
                    t = max([t, fidx])
                else:
                    t = max([t, pos[fidx][0]])

                for package in package_spawns:
                    if package[0] < t and package[0] > t - 60:
                        care_package_spawns_x.append(package[1])
                        care_package_spawns_y.append(mapy - package[2])
                for package in package_lands:
                    if package[0] < t:
                        care_package_lands_x.append(package[1])
                        care_package_lands_y.append(mapy - package[2])

                # Update player positions
                if interpolate:
                    if fidx >= pos[-1][0] and player in killed:
                        raise IndexError
                    x = interpolate_coords(fidx, pos, 0, 1)
                    y = mapy - interpolate_coords(fidx, pos, 0, 2)
                else:
                    x = pos[fidx][1]
                    y = mapy - pos[fidx][2]

                # Update player highlights
                if player in highlight_players:
                    highlights_x.append(x)
                    highlights_y.append(y)
                else:
                    positions_x.append(x)
                    positions_y.append(y)
                    # Set colors
                    if color_teams:
                        marker_colors.append(team_colors[player])

                # Update labels
                if labels and player in label_players:
                    if disable_labels_after is not None and frame >= disable_labels_after:
                        name_labels[player].set_position((-100000, -100000))
                    else:
                        name_labels[player].set_position((x + 10000 * xwidth / mapx, y - 10000 * ywidth / mapy))

                # Update player damages
                if damage:
                    try:
                        for attack in damages[player]:
                            damage_frame = int(attack[0])
                            if damage_frame >= fidx + interval:
                                break
                            elif damage_frame >= fidx and damage_frame < fidx + interval:
                                damage_line_x = [attack[1], attack[4]]
                                damage_line_y = [mapy - attack[2], mapy - attack[5]]
                                damage_lines[damage_count].set_data(damage_line_x, damage_line_y)
                                damage_count += 1
                    except KeyError:
                        pass

            except IndexError as exc:
                # Sometimes players have no positions
                if len(pos) == 0:
                    pos = [(1, -10000, -10000, -10000)]

                # Set death markers
                if player in highlight_players:
                    highlights_deaths_x.append(pos[-1][1])
                    highlights_deaths_y.append(mapy - pos[-1][2])
                else:
                    deaths_x.append(pos[-1][1])
                    deaths_y.append(mapy - pos[-1][2])

                    # Set death marker colors
                    if color_teams:
                        death_marker_colors.append(team_colors[player])

                # Draw dead players names
                if labels and dead_player_labels and player in label_players:
                    name_labels[player].set_position((pos[-1][1] + 10000 * xwidth / mapx, mapy - pos[-1][2] - 10000 * ywidth / mapy))
                    name_labels[player].set_path_effects([patheffects.withStroke(linewidth=1, foreground="gray")])
                # Offscreen if labels are off
                elif labels and player in label_players:
                    name_labels[player].set_position((-100000, -100000))

        player_offsets = [(x, y) for x, y in zip(positions_x, positions_y)]
        if len(player_offsets) > 0:
            players.set_offsets(player_offsets)
        else:
            players.set_offsets([(-100000, -100000)])

        if color_teams:
            players.set_facecolors(marker_colors)

        death_offsets = [(x, y) for x, y in zip(deaths_x, deaths_y)]
        if len(death_offsets) > 0:
            deaths.set_offsets(death_offsets)
        if color_teams:
            deaths.set_facecolors(death_marker_colors)

        if highlight_players is not None:
            highlight_offsets = [(x, y) for x, y in zip(highlights_x, highlights_y)]
            if len(highlight_offsets) > 0:
                highlights.set_offsets(highlight_offsets)
            else:
                highlights.set_offsets([(-100000, -100000)])

            highlight_death_offsets = [(x, y) for x, y in zip(highlights_deaths_x, highlights_deaths_y)]
            if len(highlight_death_offsets) > 0:
                highlights_deaths.set_offsets(highlight_death_offsets)

        if len(care_package_lands_x) > 0:
            care_package_lands.set_data(care_package_lands_x, care_package_lands_y)

        if len(care_package_spawns_x) > 0:
            care_package_spawns.set_data(care_package_spawns_x, care_package_spawns_y)

        # Remove the remaining slots
        for k in range(damage_count, damage_slots):
            damage_lines[k].set_data([], [])

        if labels:
            if highlight_players or highlight_teams:
                updates = players, deaths, highlights, highlights_deaths, blue_circle, red_circle, white_circle, *tuple(name_labels.values())
            else:
                updates = players, deaths, blue_circle, red_circle, white_circle, *tuple(name_labels.values())
        else:
            if highlight_players or highlight_teams:
                updates = players, deaths, highlights, highlights_deaths, blue_circle, red_circle, white_circle
            else:
                updates = players, deaths, blue_circle, red_circle, white_circle
        if care_packages:
            updates = *updates, care_package_lands, care_package_spawns
        if damage:
            updates = *updates, *damage_lines
        return updates

    # Create the animation
    animation = FuncAnimation(
        fig, update,
        frames=range(0, maxlength + end_frames, interval),
        interval=int(1000 / fps),
        init_func=init, blit=True,
    )

    # Write the html5 to buffer
    h5 = animation.save('Battle.mp4', fps=fps, dpi=dpi)



    return True
```


```python
# create_playback_animation(telemetry, highlight_teams='Rangers')
```
