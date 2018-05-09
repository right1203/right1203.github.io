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

또한 Decision Tree는 Robust한 특성을 가지고 있어서 정규화나 결측치 처리 등의 전처리 과정을 할 필요가 없이 작동하는 장점을 가지고 있다.

그래서 자료 자체를 시각화하여 보여주거나 머신 러닝의 결과를 해석하는 데에 사용하는 경우도 많다.

그리고 아주 손쉽게 알고리즘을 적용가능한데 반해 각 변수의 영향력을 그런대로 잘 측정하여 feature selection으로서의 수단으로도 사용 가능하다.

Decision Tree가 왜 이런 특성을 가지고 있는지 알아보자.

> **Decision Tree의 기본적인 아이디어는 가장 불확실성을 낮추는 방향으로 집합을 쪼개는 것**

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

![캡처](https://i.imgur.com/dQX2064.png)
(http://www.learnbymarketing.com/tutorials/rpart-decision-trees-in-r/)

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
* recursive partitioning : 모든 변수의 조합을 비교하며 최대 정보 획득(information gain)이 일어나는 지점을 선택한다.
* pruning : 나무를 최대로 성장시킨 뒤 비용 복잡도(노드의 수가 많을수록 비용을 부과)를 이용하여 필요없는 노드들을 제거한다.

등의 기법들을 이용할 수 있다.

이 Decision Tree 알고리즘은 대표적으로 C4.5, CART, CHAID 등이 있으며

그 중 CART의 경우 **Classification And Regression Tree** 인데,

Decision Tree를 이용하여 분류 문제와 회귀 문제, 즉 종속변수가 categorical, numerical인 문제 모두를 해결할 수 있다는 것을 알 수 있다.

더 자세한 사항은 다른 문서를 참고하기 바란다.

1.6 Association Rule Learning
---
---

Association Rule이란 데이터 간의 연관 규칙을 찾는 기법이다.
종속 변수는 따로 없으며 연관성이 높은 item들을 하나의 그룹으로 묶는 unsupervised learning 기법이다.

연관 규칙은 뉴스나 실생활에서 자주 접할 수 있는데,
어느 편의점에서 맥주를 사는 고객이 만두를 같이 사갈 확률이 높다는 분석을 통해 맥주와 만두 결합상품을 판매한 적이 있다.

Association Rule의 아이디어는 세 가지 개념으로 나타난다.

#### `Support`

우선 모든 Association Rule의 형태는
{X} => {Y} 와 같다.

![캡처](https://i.imgur.com/oYdDr0I.png)

$$ support(\{ X\} \Rightarrow \{ Y\} )= \frac { { N }_{ X\cup Y } }{ N }  $$

support는 전체 데이터 중에서 X와 Y가 얼마나 동시에 일어났는가?
즉, 로그에서 X와 Y라는 아이템이 얼마나 자주 발생되는지를 측정해준다.

0~1의 값을 가지며
값이 높을수록 이 규칙을 support해준다.

#### `Confidence`

![캡처](https://i.imgur.com/J3Thm9u.png)

confidence는 X가 발생한 것 중에서 얼마나 Y가 같이 발생하였는가?
즉, 연관성 규칙이 X 인스턴스 중에서는 얼마나 유효한지를 알 수 있다.
반대로 Y => X라면 Y에서 X로의 순차적인 연관성을 알 수 있다.

0~1의 값을 가지며
값이 높을수록 이 규칙은 신뢰성이 높다.

#### `Lift`

![캡처](https://i.imgur.com/oCPw38S.png)

lift는 둘의 연관성이 긍정적인 방향인지, 부정적인 방향인지, 또는 관계가 없는지를 알 수 있다.

lift>1이라면 두 item은 긍정적인 연관성을 지닌다.
마치 맥주를 사는 사람이 만두를 같이 산다는 것처럼 말이다.

반대로 lift<1이라면 두 item은 부정적인 연관성을 지녀서 X를 사면 Y를 구매하지 않는 경향이 강해진다.

또한 lift가 1에 근접하면 두 item은 독립적이어서 연관성을 가지지 않는다.

![캡처](https://i.imgur.com/ksyiDLt.png)

위 예제에서 {Dommelsch} => {Pampers}의 경우
> support : 51/100 = 0.51
confidence : 51/51 = 1
lift : (51\*100)/(51\*91) = 1.1

X와 Y는 전체 데이터에서 51%나 동시에 일어나며 그 경우 모두 이 연관성 규칙을 만족한다. 그리고 lift>1이므로 긍정적인 연관성을 띄고 있음을 알 수 있다.

Association rule은 간단한 계산을 통해 매우 직관적인 해석이 가능하지만, 데이터의 양이 많아지면

`computation ploblem` : 아주 많은 빈도 데이터들을 모두 살펴봐야한다.
`interprtation ploblem` : 너무 많은 결과 규칙들이 혼란을 주게 된다.

의 두 가지 문제점이 발생한다.
이러한 문제점을 해결하기 위해 minimum support,  minimum confidence 등의 `threshold`를 적절히 설정하거나,
연산 수를 줄이기 위해 [Apriori Algorithm](https://en.wikipedia.org/wiki/Apriori_algorithm), DHP algorithm 등의 전략을 사용할 수 있다.

<br/>

> ## <span style="color:red"> Clustering and Association Rule Learning</span>

<br/>

1.7 Cluster Analysis
---
---

k-means clustering
클러스터링 기법의 전체적인 방법

1.8 Evaluating Mining Results
---
---

confusion matrix
accuracy, error, precision, recall, f1-score
cross validation
