---
title: Machine Translation (MT)의 발전
description: 본 글은 cs224n를 정리하였습니다. 
slug: llm
date: 2025-06-21 18:13:00+0900
categories:
    - LLM
tags:
    - LLM
    - AI
weight: 1  # You can add weight to some posts to override the default sorting (date descending)
---

sentence x를 source language에서 target lanuge인 sentence y로 번역하는 task

[##_Image|kage@pGqje/btsOLcLteEK/AAAAAAAAAAAAAAAAAAAAAJ8N5fEEcNKATTcOrT6v8Z2T9AlfGSSn9AL_69fY9USs/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=eajn9PkUyAUTu8iHDsP0Rqep3SQ%3D|CDM|1.3|{"originWidth":575,"originHeight":248,"style":"alignCenter"}_##]

### Statistical Machine Translation 

**Core Idea**

data로부터 probabilistic model을 학습하자!!

[##_Image|kage@bshULM/btsOL9UEEqN/AAAAAAAAAAAAAAAAAAAAAIp1l1JOWvn-gSVfpn_N1HdbHlVlG2rHxsPH8wbZkC2g/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=rmHlRNvtGTV4s%2B%2BZPCEf9yo8ZuM%3D|CDM|1.3|{"originWidth":710,"originHeight":380,"style":"alignCenter"}_##]

프랑스어 sentence x가 주어졌을 때, sentence y를 찾아보자!!!

즉 2개의 component로 각각 분리해서 학습시킨다! 

-   **Translation Model**word나 phrases를 어떻게 번역해야 하는지 모델링, parallel data로부터 학습됨
-   **Language Model**  
    좋은 English를 어떻게 써야 하는지 모델링, monologual data로부터 학습됨

그러나... 극도로 복잡하고 중요한 detail들이 너무 많다... 

갑자기...등장한 것이 있으니..

### Neural Machine Translation (NMT)

구글 번역도 원래 SMT system이었는데..! NMT system으로 바뀌면서 빠르게 발전됨!!

[##_Image|kage@KztoA/btsOLjKAPJO/AAAAAAAAAAAAAAAAAAAAAKSVFfP_V1F1CGPM8qs4CtkTkr9qolVRwRl448JVY1b2/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=LYQiQbO%2BxDXQc%2BSMzU2ImHKugGo%3D|CDM|1.3|{"originWidth":748,"originHeight":436,"style":"alignCenter","width":640,"height":373}_##]

sequence-to-sequence model 이다!!

-   **Encoder RNN  
    **Encoder RNN은 source sentence의 encoding을 생성  
    Encoding 결과는 Decoder의 초기 hidden state로 제공 
-   **Decoder RNN  
    **encoding을 조건으로 하여 target sentence를 생성하는 Language Model 

즉 Encoder는 input을 취하고 neural representaion을 생성, Decoder는 neural representation을 기반으로 output을 생성

이는 seq2seq model 이라고 부른다. 

많은 NLP task에서는 seq-to-seq로 표현될 수 있음!

-   Summarization (long text → short text)
-   Dialogue (previous utterances → next utterance)
-   Parsing (input text → output parse as sequence)
-   Code generation (natural language → Python code)

### NMT

[##_Image|kage@el1MRG/btsOLt0scSI/AAAAAAAAAAAAAAAAAAAAAPIidx9a-aWVJQotstRic7e-OxaafFnqpJygAm0-iGFH/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=aj7Nv2EZj29i%2B5XsCKiWclilJMs%3D|CDM|1.3|{"originWidth":654,"originHeight":165,"style":"alignCenter"}_##]

seq2seq 모델은 **Conditional Laguage Model**의 한 예시임!

Language model인 이유는 decoder가 target sentece y의 다음 word를 예측하기 때문이고 

Conditional인 이유는 predction이 이전 단어들에 더해 source sentece에 의존하기 때문! 

### 어떻게 train?

[##_Image|kage@4Twsx/btsOMmM2cZG/AAAAAAAAAAAAAAAAAAAAAKAe9xc8Q4WOYWbkWdvIrtxE287Km-HlwKyzNev4TxfS/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=oNJVT4MAY3sM9mEKBHnEseFO0mY%3D|CDM|1.3|{"originWidth":774,"originHeight":453,"style":"alignCenter"}_##]

즉 loss는 decoder에서 계산되고... 하나의 systemdㅡ로서 end-to-end로 Backpropagation함! 

### 문제점: issues with recurrent models

#### Linear interaction distance 

O(sequence length) 단계의 거리의 word pairs간 상호작용한다는 뜻은...

긴 거리의 dependencies은 학습하기 어렵다..(grdient 소실 문제 여전히...)

→ 예: "The **dog** that chased the cat **ran**" → "dog"과 "ran"은 멀리 떨어져 있음

RNN은 그냥 왼쪽부터 오른쪽으로 하나씩 처리하기 때문에 문장 구조에 Linear order가 장제로 박혀있다.

문장을 단순한 순서로만 생각해서는 안 된다..!  

"The dog **that chased the cat** ran away."  
→ 주어는 "The dog", 동사는 "ran", 근데 둘 사이에 관계 있는 단어들이 멀리 떨어져 있음

즉 이를 순서대로 처리하는 모델은 구문 구조나 의미 관계를 잘 파악하지 못한다. 

#### Lack of parallelizability

Unparallelizable하게 작동하기 때문에 forward와 backward 연산은  O(seq length)를 가진다. 

GPU 병렬 처리에 잘 안 맞고..(이전 hidden state를 계산해야 다음 hidden state를 계산할 수 있기 때문!!) 

큰 데이터 학습이 힘들다!!