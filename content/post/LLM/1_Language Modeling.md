---
title: Language Modeling
description: 본 글은 cs224n를 정리하였습니다. 
slug: nlp
date: 2025-06-21 14:10:00+0900
categories:
    - NLP
tags:
    - NLP
    - Deep Learning
weight: 1  # You can add weight to some posts to override the default sorting (date descending)
---

## Language Modeling의 정의  

_다음에 올 단어를 예측하는 task를 수행한다_

EX) the students opened their \_\_\_\_ -> (books, laptops, exams, minds)

_주어진 단어들의 sequence(== 문장)_

_x1, x2, ,,,, xt_

_다음 단어 xt+1의 probability distribution를 계산하는 것_ 

_x+1은 vocabulary V= {w1, ... , wV}의 아무 단어를 의미한다._ 

_위와 같은 작업을 수행하는 것이 Language Model이다._ 

쉽게 생각해서 단더 조각의 probabiltiy를 할당해주는 system이라고 생각하자! 

[##_Image|kage@bEu4UG/btsOMbx4VHd/AAAAAAAAAAAAAAAAAAAAALAuedCmtWhEmgVvyRDKLlYNoK0C_VgaLV7x_pUgvez8/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=c1L%2Ft7T%2BP3fKyowIDA7ity6N4xU%3D|CDM|1.3|{"originWidth":628,"originHeight":179,"style":"alignCenter"}_##]

그러니까 즉, 어떤 문장이 생성되기 위해서는 첫 번째 단어가 올 확률 x 첫 번째 단어가 있을 때 2번째 단어가 올 확률 x ...

쉽게 생각하면 input은 그동안 생성한 글의 확률, output은 그 글의 다음 단어의 확률 

## Language Modeling을 왜 신경써야 할까? 

Language Modeling는 **benchmark task**이다!

이는 우리의 progress가 잘 예측하는지 측정하는데 도움이 된다. 

또한 많은 NLP(Natural Language Process)의 subcomponent라고 보면 된다.

왜냐하면 **text 생성 이나 text의 확률을 예측**하는 일에서 매우 중요하다ㅏ. 

•  Predictive typing  
• Speech recognition  
• Handwriting recognition  
• Spelling/grammar correction  
• Authorship identification  
• Machine translation  
• Summarization  
• Dialogue

결론

NLP의 모든 기술은 LM을 기반으로 다시 설계된 것.

옛날에는 NLP 각 task마다 model이 따로 존재(감정 분석: SVM, 번역: RNN, 요약 Seq2Seq...)

요즘은 대규모 언어 모델(GPT, Bert) 하나로 모든 걸 해결! 

## Next word prediction 

[##_Image|kage@B880X/btsOK56w4m9/AAAAAAAAAAAAAAAAAAAAAGi7MTKKVksaHyg8oGF0rKmbkx07-XeOS6WYhS79W1rJ/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=ZvaUY0HkP726Ec1gIbbi2I5wAXM%3D|CDM|1.3|{"originWidth":876,"originHeight":272,"style":"alignCenter","width":699,"height":217}_##]

이런식으로 문장 안의 blank를 두어서 예측을 하도록 학습시켰다면... 

[##_Image|kage@brGrYk/btsOMbrhW2U/AAAAAAAAAAAAAAAAAAAAAJVZ7cxHL1j8l05ObJ55ytt-g0YBZVYRgco6NaEaiMlP/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=I736lUxcPUJPfuxX3GzgIH38w3U%3D|CDM|1.3|{"originWidth":768,"originHeight":364,"style":"alignCenter","width":684,"height":324,"caption":"Context와 example 하나를 보여주고 질문에 답을 하게 한다."}_##]

요즘은 이렇게 GPT 같은 모델들이 **in-context learning** 으로 작업을 수행하게 한다. 

즉, 몇 개의 example을 주어지면 그 작업을 수행하게 한다. 

fine-tuning(훈련)하지 않고 propmt 상에서 example 몇 개(한 개 혹은 그 이상)만 보여주어서 새로운 작업을 수행하게 한다. 

## 예전에는 어떻게 Language Model을 학습시켰을까?

**n-gram Language Model!**

**Markov assumption**

현재 상태는 오직 직전 상태에 의존한다. 과거의 모든 정보는 직전 상태(n이면 n-1개까지의 과거)에 요약되어 있다! 

자연어 처리에서는...   

[##_Image|kage@buCTf6/btsOLCCYRPY/AAAAAAAAAAAAAAAAAAAAAEJfukJ_v1O9xuY3QzTuz7i_7f-TsttY1Bv7BWikBI8k/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=xhloXgei4u44u82gqR%2B99W%2FZtIk%3D|CDM|1.3|{"originWidth":762,"originHeight":114,"style":"alignCenter"}_##]

xt+1(현재 단어)은 이전의 n-1 개의 단어에(과거의 정보) 의존한다.

[##_Image|kage@cpa9dj/btsOKBkD6Gh/AAAAAAAAAAAAAAAAAAAAAE1v60qR30vxs8aj35g8PCBZX72lX-Cy40UCZ6SEp8ps/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=qIY0LUvEpKo1VTag8E7SzXTM%2Feo%3D|CDM|1.3|{"originWidth":614,"originHeight":187,"style":"alignCenter"}_##]

만약 4-gram Language Model이라고 하면 3개의 단어를 확인 

이를 직접 statistical approximation 하게 count 한다. 

[##_Image|kage@MjTJo/btsOLvqmrvI/AAAAAAAAAAAAAAAAAAAAAPGTzzs7bDTPGiHL-aOJuJUES5godmZjlxljy8xrcZvy/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=9ArLFBmwn2JsNsaf4ukucZ4IMwA%3D|CDM|1.3|{"originWidth":730,"originHeight":202,"style":"alignCenter"}_##]

즉 corpus에서 3개의 단어가 1000개 나왔을 때 다음 단어를 count해서 학습 

하지만 proctor 라는 context(문맥)을 아예 무시해도 될까??

### 문제 1

만약 data에 "students opened their w" 이 아예 나타나지 않으면? 

\-> 작은 입실론을 추가해준다. 

### 문제 2

"students opened their" 가 아예 없으면?

\-> backoff로 "opened their" 로 대신해라!  

### 문제 3

그리고 그 모든 확률을 다 count 해야 하는데..? 

\-> storage 문제 발생 

사실 n=5이상이면 문제가 발생해서... 잘 안 쓴다. 

근데...

incohrent하다!! 즉 앞뒤가 안 맞는 문장이 생성된다. 

적어도 3개 이상의 단어를 고려해야 하는데... n이 커지면 위와 같은 문제가 발생하고... 어뜩하지 

## Neural Language Model? 

즉 확률을 모델링하는 것이 아니라... Deep Learning(신경망)을 이용해서 확률을 모델링하자! 

[##_Image|kage@UZADO/btsOKDicr3a/AAAAAAAAAAAAAAAAAAAAAKs73WHd8Jz4kU62qvLWkSTEREbXI-zd3iVBIIyhRW8t/img.png?credential=yqXZFxpELC7KVnFOS48ylbz2pIh7yKj8&amp;expires=1761922799&amp;allow_ip=&amp;allow_referer=&amp;signature=bLSWe3euV220bNV8vjgc99woGa4%3D|CDM|1.3|{"originWidth":721,"originHeight":495,"style":"alignCenter"}_##]

즉 위와 같이 vector를 만들어서 embedding을 해주고 hidden layer를 통해 계산. 

따라서 n-gram LM의 문제를 해결한다! 

관찰되어야 할 n-gram을 모두 저장할 필요가 없고 

정보가 존재하는지 아닌지도 문제 (sparsity problem) 도 없어진다!

그러나... 다음과 같은 한계가 존재하는데....

-   Fixed window도 너무 작다. 
-   window size를 키우면 W(가중치)도 더 커진다!!!
-   즉, 매우 긴 sequnce를 이해할 수 없다!! (문장 앞부분의 개념이 뒷부분 해석에 중요할 수도 있으니!)
-   x1과 x2는 곱해지는 가중치가 다르다! (즉, 입력 위치에 따라 처리 방식이 달라진다) 이런 경우 입력 위치가 다르면 같은 단어라도 다르게 처리될 수도..(즉 dog라는 의미는 앞이든 뒤이든 같아야 함!!) 

어떤 길이의 input이든 처리할 수 있는 neural architecture가 필요하다!!! 

이제... RNN 을 배울 때다!!