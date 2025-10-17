---
title: Recurrent Neural Networks(RNN)
description: 본 글은 cs224n를 정리하였습니다. 
slug: llm
date: 2025-06-21 16:53:00+0900
categories:
    - LLM
tags:
    - LLM
    - AI
weight: 1  # You can add weight to some posts to override the default sorting (date descending)
---

### Core Idea

[##_Image|kage@cd3kQx/btsOL55YG2J/AAAAAAAAAAAAAAAAAAAAANYIGed86TNJRULXuEeGhsPDL1fSZKTEgFCyVcJT_nnM/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=xtXMYU30gOU9Dw%2FwaYztfGNHw2E%3D|CDM|1.3|{"originWidth":620,"originHeight":302,"style":"alignCenter"}_##]

같은 가중치 W를 반복해서 적용한다!!!!

각 입력에 대해서 hidden state에 같은 W를 곱한것을 같이 처리하는 것을 볼 수 있다. (이전에는 x1은 w1, x2는 w2랑 곱해져왔음) 

### RNN의 전체 구조 

[##_Image|kage@nAlnd/btsOKCKGYh2/AAAAAAAAAAAAAAAAAAAAAI_fpqbVrSvAnP94hHCY_7jcCxy1K1hw80WqHlf7PRzI/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=agopE1K1jweMPlppzIrc%2BVSfsUY%3D|CDM|1.3|{"originWidth":761,"originHeight":525,"style":"alignCenter"}_##]

#### 장점

-   어떤 길이의 input이든 처리 가능
-   step t에 대한 계산은 많은 뒤의 step의 정보를 활용한다.(이전 정보를 활용한다!) 
-   input 크기에 따라 Model size(Wh, We)가 증가하지 않는다. 
-   모든 시간 step t에 같은 W를 적용하기 때문에 위치에 상관없이 공정하게 처리된다. 

#### 단점 

-   Recurrent computation이 너무 느려 -> 병렬 처리가 안 된다!! 
-   완전 이전 step의 정보를 얻는게 좀 어려움... -> 정보 소실 문제! (gradient vanishing)

### Training 

-   Input : 단어들의 sequence
-   Ouput : 모든 각 step t의 distribution yt
-   Loss : step t에 대해서, 예측 확률 분포 yt와 진짜 다음 단어 yt 간의 cross-entoropy(불일치를 측정하는 함수)를 계산 
-   그 Loss를 모두 더해서 평균을 낸다.  

**\* cross-entropy?**

예측한 확률 분포yt와 실제 정답(label)의 분포 yt 사이의 불일치를 측정, 모델에 확신을 가지고 맞췄는지를 평가하는 Loss 함수! 

따라서 정답을 맞춰도 확신이 없으면 Loss가 커진다... 

[##_Image|kage@bu6ZvL/btsOL8OXDOx/AAAAAAAAAAAAAAAAAAAAANYGFDr1VnpHth2STLiZISwfe_wqt8My-valyxnwKa7Y/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=OBFojuUdiS4Ro%2FJuzjm6fOXkAGA%3D|CDM|1.3|{"originWidth":616,"originHeight":255,"style":"alignCenter"}_##][##_Image|kage@qz5G4/btsOK40UBYS/AAAAAAAAAAAAAAAAAAAAAPgHM10-ONI6fduzzG8qC55ZS0yeKIB5r3a_e-eiFBq3/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=Ln447zxoSPmQ9%2B6wXoRP9D%2Fvt%2BM%3D|CDM|1.3|{"originWidth":839,"originHeight":522,"style":"alignCenter","width":681,"height":424}_##]

1\. x1이 들어와서 y1을 예측

2\. 이때 Loss는 실제 true값(y2) 와의 nevative log prob을 계산한다. 

3\. time step t마다 계산 후에 

4\. Loss를 계산 후 평균 

### 문제점

전체 corpus(문장, document)에 대해서 loss와 grandient를 한번에 계산하는 것은 너무 비쌈.. 메모리 문제.. 

그래서 

Stochastic Gradient Descent 기억하니??? 이러한 작은 data chunk에 대해서 loss와 gradient를 계산하고 update..

이걸 적용해보겠다... 

그래서 한 문장에 대해서 loss를 계산하고(원래는 여러 문장 batch에 대해서 계산했었음) graeidnt랑 weight를 업데이트 해서.. 이제 다른 새로운 한 문장에 대해서 반복한다~  

### Backpropagation for RNNs (W parameter를 훈련 시키기 위함!) 

[##_Image|kage@mg3iq/btsOK3gBiFm/AAAAAAAAAAAAAAAAAAAAAOoIiVeMS4EAFLxLbEQdFCnjBPX6LasLXf1C3cmF95GE/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=pAazc3X5vbWcsh65DO1koqUo68s%3D|CDM|1.3|{"originWidth":756,"originHeight":419,"style":"alignCenter","width":612,"height":339}_##]

가중치 matrix Wh에 대해서 Jt의 미분은 무엇인가? 

여러 위치에서 반복적으로 사용된 같은 가중치에 대한 gradient는 각각의 사용 위치에서 나온 gradient들을 모두 더한것! 

즉, 각 timestep에서 loss에 기여한 W에 gradient가 따로 생김! 이 W에 대해 업데이트 하려면 모든 timestep에서 나온 gradient를 한번에 더해야해! 

[##_Image|kage@3lKBN/btsOLfnQEAb/AAAAAAAAAAAAAAAAAAAAAPjbKuP46Dg-E-VCUPX_CFF7mHCsyRNM6Q1Is8H4ZAvF/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=u%2BO2hvqDSwJXiSFX%2Bq0GuR%2FeUeU%3D|CDM|1.3|{"originWidth":872,"originHeight":459,"style":"alignCenter"}_##]

timestep를 따라 거꾸로 역전파하면서(t, t-1, .. 0) 각 시점마다의 gradient를 계산해서 전부 더한다. 

따라서 backpropatgation through time (BPTT) 

그니까 시간을 거꾸로 돌면서...! 계산한다! 

마치 n-gram LM같음! 반복되는 sampling을 통해서 text를 생성하는~~

### Problems with RNNs: Vanishing and Exploding Gradients

그러면 이제 각 timestep t에 대해서 Wh에 대한 J4의 gradient를 계산해야 함! 

이때 시점이 1일 때 J4/Wh

[##_Image|kage@c6Mtp5/btsOMqIBmyj/AAAAAAAAAAAAAAAAAAAAAKBM8pAVxyGZw73TMjsAJ0pmdqhp4Zt-xEYITKJG6djN/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=3BiU%2FM%2BvC7sZD6aur%2BHCTsfdFps%3D|CDM|1.3|{"originWidth":595,"originHeight":350,"style":"alignCenter"}_##]

우리가 4번째 Loss를 계산하고 이것을 첫 번째 W(hidden state)에 대한 가중치를 계산해야 함! 

[##_Image|kage@c41Eec/btsOKq4vd6Y/AAAAAAAAAAAAAAAAAAAAAJdDrLhfcBcbPxKpllFz0izw43toU0HB8hVXWQxP56yA/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=e7hJ%2BvM10%2FBDDhCsVZL1ldD2sT0%3D|CDM|1.3|{"originWidth":799,"originHeight":511,"style":"alignCenter","width":458,"height":293}_##]

이를 위해서는 chain rule에 따라 위와 같이 계산된다.. 

그런데... 곱해지는 저 값들이 작으면 어떻게 되니? 

**Vanishing gradient problem**

backpropagation이 더 멀리 진행될 수록 gradient signal이 더 작이지고 작아지는 현상

[##_Image|kage@cSSAqq/btsOLwJyfLB/AAAAAAAAAAAAAAAAAAAAANx0D3CcClzEqW80cTkfeIK2NffZKwAtp-HYRQzx45o5/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=MM5pOJ%2Fsdg7lqRrGlRThWq9NhhQ%3D|CDM|1.3|{"originWidth":770,"originHeight":409,"style":"alignCenter"}_##]

그러니까... ht 함수에는 wh 라는 우리가 계산해야 할 가중치가 있고 또 그 안의 ht-1 함수에는 우리가 계산해야 할 가중치가 있다. chain으로 곱채지는데... 곱해지는 값이 작은 수면.. 작은 수 끼리 계속 곱해진다... 따라서 지수적으로 작아진다는 문제가 발생

[##_Image|kage@UOpMn/btsOLdDCn8y/AAAAAAAAAAAAAAAAAAAAAJ1SE3Q95f-Db_u_s_neu4yQdaqc7_iRYuoKmdU0ES1p/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=pPPO1mCmNROOkzinLjcn9Q5Z7Bs%3D|CDM|1.3|{"originWidth":916,"originHeight":455,"style":"alignCenter"}_##]

그래서 문제는... backpropagation을 할 때 timestep이 올라갈 수록 점점 그 값이 작아져서 .. 나랑 가까운 시점의 gradient는 크고 나랑 먼 시점의 gradient는 기억하지 못한다는 단점이 있다. 즉, 짧은 기억만 가능하다!!! 긴 문장 처리 못 함!! 

즉 어떤 t 시점의 Loss와 W를 계산하려면 i 시점의 gradient 즉, Loss t / hi를 계산해야 하고 이때 i가 t보다 너무 멀면 아예 값이 작아져서 영향을 못 준다는 거구나!!!

그래서... 

긴 문장이 주어지고 빈칸을 주어지는 문제를 맞출 때...! 

When she tried to print her tickets, she found that the printer was out of toner. She went to the stationery store to buy more toner. It was very overpriced. After installing the toner into the printer, she finally printed her \_\_\_\_\_\_\_\_

여기서 답은 ticket 즉, 맨 앞의 7번째 단어를 보고 맞춰야 하는데.. 너무 멀어서 vanishing 되기 때문에.. 못 맞춤.. 

exploding 문제는 걍.. 아예 다른 값이 나올 수 있다는 뜻!!! NaN

이 문제는 

-   Gradient가 일정 threshold보다 크면 잘라주는 작업을 하게 됨! clipping!!

### Vanishing 문제를 어떻게 해결해야 할까? 

1\. **기억을 따로 보관하고 더해가는 RNN은 어때?**"

\-> LSTM

2\. **정보가 더 직접적으로 흘러가게 (linear pass-through)** 하는 구조

\-> **Attention**이나 **Residual Connection**

### 추가 질문

Q. 여기서 embedding은 word2vector와 같은 pretrained embedding model을 통해 를 통해서 embedding 되는건가?

No! 그냥 가중치 W에 의해 vector화 해주는거고 pretraining 단계에서 embedding은 자동으로 함께 학습된다. (즉 end-to-end로 같이 학습) 

Q. RNN은 LM인가?

아님!! RNN은 모델 아키텍쳐이고 LM은 학습 목적이자 Task!!! 

즉 RNN은 시퀀스를 처리하는 신경망 구조이고 

LM은 텍스트에서 다음 단어 예측 같은 Task를 수행하는 모델 

-   너는 **RNN을 써서 Language Model을 만들 수 있어**  
    → 예: RNN-based Language Model (2015년 이전에 많이 사용)
-   하지만 **RNN이 항상 Language Model을 의미하는 건 아니야**  
    → 예: RNN으로 음악 생성, 시계열 예측, 번역 등 **비-LM task**도 가능함
-   그리고 **Language Model이 꼭 RNN을 써야 하는 것도 아냐**  
    → GPT 같은 모델은 **Transformer 기반 LM**
