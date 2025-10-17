---
title: Transformer - Encoder
description: 본 글은 cs224n를 정리하였습니다. 
slug: llm
date: 2025-06-22 15:45:00+0900
categories:
    - LLM
tags:
    - LLM
    - AI
weight: 1  # You can add weight to some posts to override the default sorting (date descending)
---

[##_Image|kage@ts3IR/btsOMTqkBt1/AAAAAAAAAAAAAAAAAAAAAO5eCa8c7NYTnezb7sEzqKBacKSPresrBDlEMbGfNoLD/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=W679xjbXXwiqbkktqAK99VZUDmE%3D|CDM|1.3|{"originWidth":602,"originHeight":667,"style":"alignCenter","width":425,"height":471}_##]

Transformer Decoder는 **language model처럼 단방향(unidirectional) context만을 사용하도록 contraint(제한)**한다. 

\-> 즉 과거(왼쪽) 정보만 보고 미래(오른쪽)는 보지 않는다. 

\-> 이는 Langue model처럼 다음에 올 단어를 예측하는 task

그러면 우리는 **bidirectional context를 원하면 양방향 RNN처럼 하면 될까?** 

그럴 때 사용하는 것이!!!!! **Transformer Encoder**이다!

\-> 이는 문장 이해, 분류에 적합한 task 

**Decoder와의 유일한 차이점은 self-attention에서 masking을 제거한다는 것..! (즉, 모든 단어를 동시에 본다!)** 

### **The transformer Encoder-Decoder**

[##_Image|kage@EsUBF/btsOMe2P77d/AAAAAAAAAAAAAAAAAAAAAMMZMPLAgERAICX1MeRLDaUvqrufJ3FCTwJ4Acj8HCpg/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=lSVxZmTQSeho1FJ6UDsGGzTTBM0%3D|CDM|1.3|{"originWidth":648,"originHeight":694,"style":"alignCenter","width":381,"height":408}_##]

Machine translation을 회기하면,, 

우리는 bidirectional model에서 source sentence를 처리했고, unidirectional model에서 target을 생성했다!  

seq2seq format의 작업에서는... 이렇게 Transformer Encoder-Decoder 구조를 사용한다!!!!!

-   **Encoder**  
    일반적인 Transformer Encoder를 그대로 사용
-   **Decoder**  
    Encoder의 출력에 대해 **cross-attention**을 수행할 수 있도록 수정된다. Decoder는 **self-attention**으로 자기 자신만 보는게 아니라, Encoder가 만든 input 문장의 context를 참고해서 번역을 생성 

### **Cross-attention (details)**

[##_Image|kage@c0gxt1/btsOMku0Kfk/AAAAAAAAAAAAAAAAAAAAABG7eNvJgk9_dlDwwjsrY0PNbkeQG-hXvdGsIMMsGfdq/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=Wz7T1TPA5m%2BSrkRUCGrk21%2FpPg4%3D|CDM|1.3|{"originWidth":597,"originHeight":482,"style":"alignCenter","width":382,"height":308}_##]

**self-attention은 같은 source에서 keys, queries, values가 만들어질 때를 말한다.**

**즉, Encoder 내부, Decoder 내부** 

Decoder에서는 self-attention도 있지만, 다른 source를 바라보는 attention도 있다!(저번 시간에 배움) 그게 바로 **cross-attention**

-   **h1, ... , hn**: Transformer **encoder** 로 부터 얻은 **output vector(hidden state)** 𝑥𝑖∈ℝ𝑑
-   **z1, ... , zn**: Transformer **decoder**로 부터 얻은 **input vector(즉, 이미 생성된 단어들의 hidden state)** 𝑧𝑖∈ℝ𝑑

여기서 keys와 values는 encoder의 출력 hi에서 계산

즉 Encoder는 일종의 memory 같은 느낌 

𝑘𝑖=𝐾ℎ𝑖, 𝑣𝑖=𝑉ℎ𝑖

queries는 decoder로 현재 입력 zi에서 가져온 것 

𝑞𝑖=𝑄𝑧𝑖.

그니까 정리하자면... query는 decoder의 input 우리가 집중해서 보고 싶은 것들은 encoder의 input이고, key와 value에 해당 

즉 Decoder가 Encoder의 출력에 대해 attention 을 날리는 것! **참조하며 문장을 생성할 수 있게!!**

### Cross-attention이 계산되는 과정 

[##_Image|kage@leV61/btsOLl2zPW4/AAAAAAAAAAAAAAAAAAAAAIA7cXyO2OQFSby4v8-pcRp2XmxzWFnaRyUMSH9ihgPv/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=V2xFp9KXOuxlWZ8yLjJQZxeZ4EI%3D|CDM|1.3|{"originWidth":309,"originHeight":88,"style":"alignCenter"}_##]

H는 encoder vector의 concatenaton! 

Z는 dcoder vector의 concatenation!

여기서 T는 encoder의 길이 또는 decoder의 길이 중 하나일 수 있는데, **문맥상 둘 다 T로 표기한 거지 실제로는 다를 수 있어.**

-   예를 들어: 영어 입력(5단어) → 프랑스어 출력(7단어)이라면
    -   Encoder: T₁ = 5, Decoder: T₂ = 7

[##_Image|kage@cgCHrb/btsOMFyVJLj/AAAAAAAAAAAAAAAAAAAAAAfCTekS-Os5iVz3UuE7abPQNnIuvyBsZc7hVMY0M0_8/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=YXRMdc7KiYsDXz9NJeiVHIeoqbY%3D|CDM|1.3|{"originWidth":470,"originHeight":43,"style":"alignCenter"}_##]

decoder vector에 Query를 곱하고 

encoder vector에 Key를 곱한 후 

그 둘을 계산하여 Attention score를 계산 이를 softmax로 통과하여 attention distribution 즉, 가중치를 만들어 내고 

이를 encoder vector의 value들에 곱한다!! 

[##_Image|kage@bWv0Qu/btsOMcjF5EO/AAAAAAAAAAAAAAAAAAAAAAhMLglNh1dZVXt4HF_o3G49dac-4pEB-YMPtuY9VRzM/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=zYlrTVAYTUG9f8kFM%2FnQL5qPDZw%3D|CDM|1.3|{"originWidth":717,"originHeight":373,"style":"alignCenter","width":655,"height":341}_##]

### 성능을 봅시다. 

**Machine Translation** 

[##_Image|kage@dppGcu/btsOM4d8Xmw/AAAAAAAAAAAAAAAAAAAAACowHO3mblz7Wo1-iI1Hmsj7koe4pZuJDAQPs4JihY85/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=q2ltKQI0%2BhF8IirfD2IfJfixqIg%3D|CDM|1.3|{"originWidth":904,"originHeight":324,"style":"alignCenter"}_##]

**document generation**

[##_Image|kage@cH560F/btsOLFmjPYx/AAAAAAAAAAAAAAAAAAAAAOP0FZ61m9K6cZ_N3KXJbor5XqMzgrXt2snrP6Ye7Nwv/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=j8JxrmMrQuHJFhW3LlvSN80Bt5U%3D|CDM|1.3|{"originWidth":1068,"originHeight":399,"style":"alignCenter"}_##]

Transformer는 병렬처리가 잘 되기 때문에 pre-training으로 효율적으로 수행할 수 있게 되었고, 사실상 NLP의 표준이 되었다. 

### Transformer의 한계 

**1\. Training instabilities (Pre vs Post norm)**

training이 불안정함! 특히 LayerNorm을 어디에 넣느냐에 따라 학습 안정성이 달라짐

-   **Pre-norm**: LayerNorm을 **Residual 전에** 적용 (요즘 이 방식이 더 안정적임)
-   **Post-norm**: Residual **후에** 적용 (초기 Transformer 논문 방식)

[##_Image|kage@wZUBY/btsOMjJFANk/AAAAAAAAAAAAAAAAAAAAACaRrxSwZ-2TCwVSz2LENOjyyg5tRODrIAzZ62b7-54O/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=rZUnNPBpCdLtNZdbLYrXb%2BJjxOY%3D|CDM|1.3|{"originWidth":986,"originHeight":543,"style":"alignCenter","width":566,"height":312}_##]

**2\. Quadratic comput in self-attention** 

self-attention 은 계산량이 sequence 길이의 제곱에 비례함 

왜냐하면

모든 token쌍마다 attention을 꼐산하기 때문에 O(n^2) 임

근데 왜 써??? Transformer가 커질 수록 전체 계산에서 self-attention이 차지하는 비중이 점점 줄어든다. 즉 self-attention 계산은 여전히 느리지만, 전체 모델 계산 중 일부일 뿐 

\- 게산을 줄이려고 만든 **저렴한(효율적인) Self-Attention 방법들**은

→ **모델이 커질수록 성능이 별로 안 좋음.**  
→ 그래서 실제 대형 모델에는 잘 안 쓰인다.

**\- Systems optimizations work well (FlashAttention – Jun 2022)**

→ 그래서 **알고리즘을 바꾸는 대신**,  
→ **기존 Attention을 더 빠르게 구현한 최적화 기술**이 많이 쓰임.

### 잠.시.만 Encoder-Decoder 구조에서 Decoder의 입력이 Encoder의 output이 아니었어???

> ✅ **Transformer의 Decoder는 Encoder의 출력을 "직접 입력으로 받는 게 아니라", Cross-Attention을 통해 "참조"만 해.**

즉:  
**Decoder의 입력은 "생성 중인 문장"이고**,  
**Encoder의 출력은 "참조 대상"일 뿐이야.**

---

## 🔍 정리해보자:

### Transformer Encoder-Decoder 구조에서의 역할

컴포넌트입력처리출력

| **Encoder** | 전체 source 문장 (ex: 영어) | Full self-attention | 문장 전체의 의미 벡터 |
| --- | --- | --- | --- |
| **Decoder** | 지금까지 생성된 target 문장 (ex: 불어 일부) | Masked self-attention + Cross-attention | 다음 단어 예측 |

---

## 📌 중요한 구분

### ❌ Decoder의 입력이 "Encoder 출력"이라는 오해

-   ❌ Decoder는 Encoder의 출력 벡터들을 input으로 "넣는" 게 아님
-   ❌ Decoder는 Encoder의 출력을 문장처럼 "feeding"하지 않음

---

### ✅ 실제 구조는 이렇게 생겼어:

#### Decoder 내부에는 **두 개의 Attention Layer**가 있음:

1.  **Masked Self-Attention**  
    → 지금까지 생성한 단어들끼리만 참고  
    → 미래 단어는 가려져 있음
2.  **Cross-Attention**  
    → Encoder의 출력 전체를 쳐다보면서 필요한 정보 참조  
    → 즉, **Decoder의 Query가 Encoder의 Key/Value를 "attend"하는 구조**

---

## 🎯 다시 말해 요약하면

> **Decoder는 Encoder의 출력을 input처럼 받는 게 아니라, cross-attention의 대상(memory bank)으로 활용한다.**  
> Decoder의 **실제 입력은 지금까지 생성한 target token들**이야.

---

## 💡 시각적으로 그려보면:

css

복사편집

\[입력 문장 (source)\] → \[Encoder\] → context representations ↑ \[Decoder ← 이전 단어들\] → Masked self-attention ↓ Cross-attention to Encoder output ↓ 다음 단어 예측 (one token at a time)