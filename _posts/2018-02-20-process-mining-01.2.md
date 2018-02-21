---
layout: post
title: "Process Mining 02"
subtitle: "Process Mining - Coursera"
categories: study
tags: lecture
comments: true
---
> ## <span style="color:red">Decision Trees</span>

<br/>

1.4 Learning Decision Trees
---
---

[Decision Tree](https://en.wikipedia.org/wiki/Decision_tree) 는 의사결정 나무, 결정 트리로 불리는 supervised learning 기법이다.

flow-chart와 유사한 그래프를 이용하여 의사결정을 도와주는 도구로써 예측 성능은 일반적으로 그리 뛰어나지 않은 편이지만, 해석력이 매우 뛰어나다.

그래서 자료 자체를 시각화하여 보여주거나 머신 러닝의 결과를 해석하는 데에 사용하는 경우도 많다고 한다.

Decision Tree의 기본적인 아이디어는 가장 불확실성을 낮추는 방향으로 집합을 쪼개는 것이다.

![캡처](https://i.imgur.com/RBAXmJT.png "강의록")

초록 공 3개와 빨간 공 3개가 모여있다.

여기서 랜덤으로 하나를 뽑게 된다면 초록 공이 나올 확률과 빨간 공이 나올 확률이 각 50%로써 가장 불확실성이 높은 상황이다.

하지만 이것을 그림과 같이 예측 변수에 의해 빨간 공 2개만 있는 집합으로 줄일 수 있다면, 그 집합의 불확실성은 0이 된다.

이처럼 Decision Tree는 예측 변수의 모든 상황을 고려하며 전체의 불확실성이 가장 낮은 방향으로 움직인다.

이 불확실성을 측정하는 방법으로 entropy, Gini index 등을 사용하는데 여기서는 entropy를 이용하여 설명한다.

![캡처](https://i.imgur.com/Dg79p1d.png "강의록")

이 공식을 이용하여 계산하면

![캡처](https://i.imgur.com/hT9FikO.png "강의록")

다음과 같이 계산할 수 있다.

![캡처](https://i.imgur.com/jToYrHH.png "강의록")

각 엔트로피의 합을 구하여 전체 엔트로피를 구한다.

![캡처](https://i.imgur.com/AB9ygaz.png "강의록")

의사결정 나무의 노드를 늘려갈수록 전체 엔트로피가 줄어드는 것을 볼 수 있다.

엔트로피가 줄어드는 만큼을 information gain, 엔트로피가 오히려 늘어나는 것을 information loss라고 표현한다.

여기서는 E=1에서 E=0.54가 되므로
**0.46만큼의 information gain** 이 일어났다.

1.5 Applying Decision Trees
---
---

**의사결정나무가 끝 없이 전개된다면?**

의사결정나무가 제한 없이 뻗어나간다면 무수한 노드가 생겨 결국 전체 엔트로피가 0이 될 것이다.

> 엔트로피가 0이 되어 불확실성이 0이 되면 좋은 거 아닌가요?

엔트로피가 0이 된다는 것이 완벽하게 예측할 수 있게 되었다는 뜻이 아니다.

그저 훈련 데이터를 완벽하게 외워서 특정 상황에 동일한 답이 돌아올 뿐이다.

이런 경우를 ***overfitting*** 이라고 하며
훈련 데이터가 아닌 새로운 instance가 들어오게 되면 엉뚱한 대답을 할 가능성이 높아서 정확도가 떨어지게 된다.

반대로 overfitting을 피하기 위해 훈련을 너무 적게 하게 되면
> 남자이면 키가 50cm 이상이다.

와 같은 너무나도 당연한 대답을 하게되는데
이런 경우를 ***underfitting*** 이라고 한다.

이런 경우를 방지하기 위해서

* minimum information gain
* minimum depth
* maximum depth
* recursive partitioning : 모든 변수의 조합을 비교하며 최대 정보 획득(information gain)이 일어나는 지점을 선택
* Post pruning : 나무를 최대로 성장시킨 뒤 비용 복잡도(노드의 수가 많을수록 비용을 부과)를 이용하여 필요없는 노드들을 제거

등의 기법들을 이용한다.

이 Decision Tree 알고리즘은 대표적으로 C4.5, CART, CHAID 등이 있으며

그 중 CART의 경우 **Classification And Regression Tree** 인데,
Decision Tree를 이용하여 분류 문제와 회귀 문제, 즉 종속변수가 categorical, numerical인 문제 모두를 해결할 수 있다는 것을 알 수 있다.

더 자세한 사항은 다른 문서를 참고하기 바란다.

1.6 Association Rule Learning
---
---

작성 중...
