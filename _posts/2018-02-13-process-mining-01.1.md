---
layout: post
title: "프로세스  01"
subtitle: "Process Mining - Coursera"
categories: study
tags: lecture
comments: true
---
> ## Introduction

개인적으로 진행하는 스터디에서도, 학교에서도 여러가지 경로를 통해 접해보면서 **프로세스 마이닝을 이벤트 로그를 이용하여 프로세스 모델을 규명하고 문제점을 찾아서 해결하는 것** 정도로만 생각해왔다.

하지만 지속적으로 프로세스 마이닝에 대한 이야기를 나눌 때 용어나 개념이 이해가 가지 않는 부분이 많아서 좀 더 자세히 공부해보고자 한다.

구글링을 해보니 Coursera에서 Wil van der Aalst라는 네덜란드 출신의 Eindhoven 대학 교수님이 강의하신 [Process Mining: Data Science in Action](https://www.coursera.org/learn/process-mining/)이 유명한 것 같아 공부하면서 블로그에 정리해볼 것이다.


> ## Data and Process Mining

---

1.1 Data Science and Big data
-----------------------------

---

> ### **Data is new oil**

Moore's law에 따르면 반도체 집적회로의 성능이 24개월마다 2배로 증가한다고 한다.

![Moore's law](https://upload.wikimedia.org/wikipedia/commons/f/f7/Transistor_Count_and_Moore%27s_Law_-_2011_Estonian.png "Wikipedia")

만약 Moore's law가 다른 영역에 적용된다면 어떻게 될까?

1.	40년 전, 네덜란드의 Eindhoven에서 Amsterdam까지 가는데 기차로 1.5시간이 걸렸다고 한다. 그렇다면 40년이 지난 지금은 무어의 법칙이 성립한다면 얼마가 걸리는가?

> 답은 1.5 * 60(분) * 60(초) * $ {2}^{20} $ = 0.00515 seconds 이다.

이것이 실제로 일어난다면 정말 혁명적일 것이다.

부산에서 서울까지의 거리를 단 1초 정도만에 주파할 수 있다는 것이다.

물론 이 무어의 법칙이 반도체 집적회로가 아닌 다른 곳에서 적용되는 것은 아니다.

하지만 다른 곳에 적용되면 혁명과도 같은 법칙이 집적회로에서는 얼마전까지만 하더라도 실제로 일어나고 있었다. (현재로서는 발열이나 경제성 등 많은 이유로 무어의 법칙에 한계가 발생한다.)

이런 데이터를 담을 수 있는 그릇의 기하급수적인 발전과 함께 IoT, Cloud Computing 등의 발전으로 데이터의 양 또한 기하급수적으로 늘어나며 ***BIG DATA*** 라는 용어도 4차 산업혁명과 함께 큰 이슈가 되고 있다.

그에 따라 이 강의에서는 ***Data is new oil*** 라고 한다.

> ### 4V's of Big Data

빅데이터의 정의로서 많이 사용하는 용어가 3V 또는 4V인데 다음과 같다. 1. VOLUME : Data Size 2. VELOCITY : Speed of Change 3. VARIETY : Different forms of Data Sources 4. VERACITY : Uncertainty of Data

이에 대한 자세한 내용은 다른 곳을 참고하기 바란다.


> ### What is the best that can happen?

그래서 빅데이터를 통해 결국 무엇을 할 수 있을까?

여기서는 환자가 얼마나 기다려야 하고 왜 기다려야 하는지, 의사가 가이드라인을 잘 지키고 있는지, 내일은 얼마나 많은 의료진이 필요할지, 비용을 줄일 수 있을지 예측하는 데 사용할 수 있다고 한다.

물론 이 뿐만 아니라 많은 영역에서 예측과 최적화, 자동화 등을 할 수 있지만 여기서는 ***Process Mining*** 이 주제이므로

Event Data를 이용하여 현재의 프로세스를 이해하고 개선하는 Process-centric view on data science가 주된 강의 내용이 될 것이다.

뒤의 내용들은 개론과 달리 조금 더 개략적이고 덜 세세하게 공부한 내용을 기억나는 대로 쓸 생각이기에, 초보자의 기억에 의존한 내용이므로 절대... 절대로 정답이라고 생각하지 말고 참고만 하길 바란다.

>글을 작성하는 중에 발견하였는데, <https://1ambda.github.io/data-analysis/process-mining-1/> 에서 1ambda라는 분께서 아주 잘 정리를 해뒀다. 나 또한 이 글을 참고하여 이해가 안 가는 부분을 공부할 예정이기에 이쪽을 참고하는 것도 좋을 것 같다.


1.2 Different Types of Process Mining
-------------------------------------

---

![캡처](https://i.imgur.com/opzTD56.png "강의 자료")


프로세스 마이닝이란 기존의 프로세스 모델 분석과 머신러닝과 같은 데이터 지향적인 모델 분석의 갭을 이어주는 다리와 같다.

data-oriented analysis는 데이터를 이용하여 최적의 모델을 발견하는 데에 집중하였다.

하지만 그 모델이 실제 세계에 적용이 되었을 때, 어떠한 문제점이 발생하는지, 왜 발생하며 어떻게 줄일 수 있는지에 대한 문제를 해결해줄 수 없었다.

그래서 프로세스 마이닝은
* event logs를 이용하여 실제 적용되고 있는 프로세스 모델을 추론하고 **(Play-In)** ,
* 모델을 이용하여 다양한 시나리오를 생성하여 **(Play-Out)**
* 실제 세계에 적용하였을 때의 적합성과 개선 가능 요소 등을 파악하는 것 **(Replay)** 에 초점을 맞춘다.

![캡처](https://i.imgur.com/lPCOsww.jpg "slideshare.net")
출처 : [Wil van der Aalst 슬라이드](https://www.slideshare.net/wvdaalst/process-mining-chapter05processdiscovery)

이 과정을 도식화한 그림은 다음과 같다.

![캡처](https://i.imgur.com/yEFhFMJ.png)

이 과정들에서 무조건적인 순서는 없지 않을까?
모델을 추론하고 시나리오를 생성하여 Replay 해볼 수도 있을 것이고,
이미 존재하는 모델에 Replay를 하여 실제 사례들이 모델에 적합하게 행동하는지, 아니면 에러나 병목현상 등이 발생하고 있는지를 볼 수도 있을 것이다.

1.3 How Process Mining Relates to Data Mining
---

---

ProM 이라는 툴을 이용하여 프로세스 마이닝을 시각화한 결과이다.

![캡처](https://i.imgur.com/9x2ya9u.png)

빨간 도형과 선들이 Play-In을 통해 추론한 모델이고,
하얀 원들이 실제 케이스의 동작을 나타내며 계속 움직인다.

이런 모델의 추론과 동작을 어떻게 표현할 수 있을까?


이 하얀 원들의 움직임은 event logs에 남아있는 **변수(Variables)** 를 통해 발견한다.

![캡처](https://i.imgur.com/sS7wTki.png)

event logs의 변수들은 **case id, activity name, timestamp** 의 필수적인 요소들과 other data들을 가지는 경우가 많다.

* case id는 위 그림과 같이 event의 행위자(resource)를 나타낸다.
* activity name은 어떠한 event가 일어났는지를 나타낸다.
* timestamp는 event가 언제 일어났는지 나타내며 하나일 수도 있지만, 시작과 끝을 나타내는 두 개의 timestamp가 있을 수도 있다.


그런데 이 변수들이 단순한 번호인지, 아니면 순서를 나타내는 것인지, 날짜를 나타내는 것인지 컴퓨터는 어떻게 판단을 할 수 있는가?

이것들을 컴퓨터가 구별할 수 있도록 변수들을 각 특성별로 나눌 수 있다.

* Variables
  * categorical
    * ordinal
    * nominal
  * numerical

또한 이 변수들의 특성이나 목적 등에 따라 Supervised Learning(분류인지, 회귀인지로 나눌 수 있음), Unsupervised Learning으로 나뉘는데

변수들이나 머신러닝 기법들이 여기서의 주제가 아니므로 설명은 생략한다.

> **그래서 프로세스 마이닝과 데이터 마이닝의 차이가 뭔데?**

여기서 말하는 유사점과 차이점은 이렇다.

* Both start from data.
  데이터에서 시작하는 것은 같다.
* Data mining techniques are typically not process-centric.
  프로세스 중심이냐의 차이
* Topics such as process discovery, conformance checking, and bottleneck analysis are not addressed by traditional data mining techniques.
  데이터 마이닝에서는 프로세스 발견, 적합성 판단, 병목현상 분석 등을 다루지 않았음.
* End-to-end process models and concurrency are essential for process mining.
  처음부터 끝까지의 과정 모델과 동시 실행이 프로세스 마이닝에서는 중요.
* Process mining assumes event logs where events have timestamps and refer to cases(process instances).
  프로세스 마아닝의 이벤트 로그에는 타임스탬프와 cases, 프로세스의 수행자가 존재.
* Process mining and data mining need to be combined for more advanced questions.
  프로세스 마이닝과 데이터 마이닝은 합쳐져야 한다.

다음은 Decision Trees를 배워보자.
