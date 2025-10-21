---
title: Pretraining
description: 본 글은 cs224n를 정리하였습니다. 
slug: nlp
date: 2025-07-06 19:14:00+0900
categories:
    - NLP
tags:
    - NLP
    - Deep Learning
weight: 1  # You can add weight to some posts to override the default sorting (date descending)
---
### 1\. Subword modeling 간단하게 알아보기 

[##_Image|kage@bO6314/btsO5L1bTW2/AAAAAAAAAAAAAAAAAAAAAGRPfPUE5tQq1W_Qa92a8ZA-rJdGPLmE4vFSLj_KMfxw/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=xr0cD%2FJl2O0ronELf5POn%2FTFJqs%3D|CDM|1.3|{"originWidth":1046,"originHeight":319,"style":"alignCenter"}_##]

먼저, 우리의 language의 vocabulary를 만들 때 다음과 가정을 한다고 하자. 

training set으로 부터 만들어진, 수 천개의 words로 이루어진 fixed vocab가 있고,

test time에 처음 본 모든 word들에 대해서 single UNK로 mapping 한다고 가정한다. 

이 상황에서, subword의 modeling은 word level보다 더 작은 structure (Parts of words, characters, bytes) 를 고려하는 다양한 method 방법들을 포함한다! 

현대 NLP에서는 subword tokens으로 구성된 vocabulary를 학습하는 방식이 지배적이다. 

training과 testing 시 각 word는 vocabulary에 알려진 subword들의 sequence로 분할된다. 

#### The byte-pair encoding(BPE) algorithm 

이때, BPE는 subword vocabulary를 정의하기 위한 effective strategy 라고 할 수 있다!! 

1\. init vocabulary : 모든 문자를 각 character들과 단어 끝(end-of-word) symobl만 포함

2\. text의 corpus를 보면서 가장 자주 함께 등장하는 character 쌍 "a, b"를 찾아서 "ab"를 subword로 추가 

3\. 해당 charcter pair를 new subword로 replace, 이때 vocab size로 도달할 때까지 반복!!! 

근데 이 방식은.. 초기 machine translation과 같은 NLP에만 사용되었음!! 

현재는 WordPiece(확률 based) 와 같은 method가 pretrained model에서 사용되고 있음!!1 

\- WordPiece?? byte coding 자세한 건

[##_Image|kage@bgcmpl/btsO6rA7437/AAAAAAAAAAAAAAAAAAAAAN1b4Om0TbnQVPR2anMt7BwNRYiZ1lLCWBr69jOQT1YB/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=zlcbvOh1sPsfw%2FXHI3cJtTFwdCU%3D|CDM|1.3|{"originWidth":738,"originHeight":223,"style":"alignCenter"}_##]

대부분의 word들은 subword vocabulary의 일부에 포함되겠지만,, 드문 word들은 component로 분리된다...

아주 안 좋은 경우에는 word가 아주 많은 subwords로 분리될 수도 있다는 사실... 

### 2\. Motivating model pretraining from word embeddings

그래서..! 

### 4\. Model pretraining three ways

4-1. Decoders

4-2. Encoders

4-3. Encoder-Decoders

### 5\. Interlude: what do we think pretraining is teaching?

### 6\. Very Large models and in-context learning