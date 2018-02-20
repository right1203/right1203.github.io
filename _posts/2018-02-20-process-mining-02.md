---
layout: post
title: "Process Mining 02"
subtitle: "Process Mining - Coursera"
categories: study
tags: lecture
comments: true
---
> ## Decision Trees

---

1.4 Learning Decision Trees
---
---

decision tree가 어떤건지
엔트로피와 불확실성, Gini index
전체 엔트로피의 합이 가장 줄어드는 방향으로 결정
information gain, information loss
Post pruning, recursive partitioning

1.5 Applying Decision Trees
---
---

엔트로피 계산
RapidMiner
overfitting, underfitting

1.6 Association Rule Learning
---
---

support : 전체 데이터에서 X와 Y가 얼마나 같이 일어났는가? - 이 규칙이 X, Y를 얼마나 support 하는가?
confidence : X와 Y가 일어났을 때 X가 얼마나 발생하였는가? 이 규칙이 얼마나 유효한가?
lift : 둘의 연관성이 긍정적인가? 부정적인가? 얼마나 그러한가?
2가지 문제점 : 아주 많은 빈도 데이터들을 모두 살펴봐야함(computation ploblem). 너무 많은 결과 규칙들이 혼란을 줌(interprtation ploblem).
그래서 threshold를 적절히 설정하여야함
sequence mining, episode mining

> Clustering and Association Rule Learning

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


1. 둘둘 둘셋 같이 커널
2. 코드의 기능적 설명이 아니라 코드의 역할 (왜썼나)
3. (토론포함)상한 30분
4. 순서 바꿔가면서 하기 (앞사람이 설명량이 당연 많음)
5. 모델의 알고리즘이 아니라 왜, 적용성에 대한 토의
