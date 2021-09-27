### 🤔 지난주 과제 
* 파이썬 라이브러리를 활용한 머신러닝 구매
* Kaggle 승급하기
* Jupyter, colab 노트북 생성
* 파이썬 라이브러리를 활용한 머신러닝 github clone
* 백준, 프로그래머스 가입
<br>
<br>

# ✏ 1주차 우리가 한 것은..  
9월 14일 ML은 1주차로 "Numpy와 Pandas"에 대해 알아보는 시간을 가졌습니다!   
3시간.. 동안 ppt와 코드를 보면서 발표를 듣고 마지막에 복습 겸 퀴즈를 풀며 스터디를 마무리했습니다! 
<br>
<br>

 ## 1. 서론  
> Numpy와 Pandas에 대해 알아보기 전에 파이썬 머신러닝 생태계를 구성하는 주요 패키지에 대해서 알아보았습니다! 
<img width="900"  src="https://user-images.githubusercontent.com/63052097/134831844-565fe313-9a27-4941-b22c-8dae8d984c2d.png" style="margin-top: 10px;">
<br>  
- **머신러닝 패키지** : **scikit-learn**    
  
▫ 오픈소스 머신러닝 라이브러리   
▫ Numpy, Scipy, Matplotlib 기반으로 생성    
   **== *머신러닝의 주요 알고리즘/모듈(분류/회귀/클러스터링 등)이 구현되어 있는 라이브러리  (단, 딥러닝 모듈 없음! )***    
   cf) 딥러닝 패키지 : TensorFlow,  Keras, PyTorch  
    [00. 사이킷런(scikit-learn)이란](https://deepflowest.tistory.com/64)   
    [[python] scikit-learn이란](https://engineer-mole.tistory.com/16)

- **배열/선형대수/통계 패키지**    
 ▫  **Numpy** : 제일 중요! / **배열과 선형대수를 위한** 패키지 / Numpy를 기반으로 해서 많은 머신러닝 패키지들이 만들어져 있음!    
 ▫  **SciPy :** 파이썬의 대표적인 **통계 패키지** / 희소행렬과 같은 자연과학에서 많이 사용되는 유틸리티가 담겨있음!

- **데이터 핸들링**   
▫  **pandas** :  2차원 데이터를 핸들링 == 행과 열로 이루어진 정형 데이터를 가공, 변환   
*cf) Numpy도 데이터 핸들링 가능 → but 저차원 API라 pandas에 비해 개발 생산성이 떨어짐*

- **데이터 시각화**    
▫ **matplotlib :** 데이터의 직관적 이해, 시각적인 표현을 위한 라이브러리 / **많은 다른 시각화 솔루션들이 matplotlib을 기반으로 만들어짐!(seaborn도 마찬가지)**    

- **대화형 파이썬 툴**    
▫  노트필기 하듯이 코드 + 필기할 수 있는 툴 / **코드를 분할해서 실행이 가능!!**   

<br>
<hr>
<br>

## 2. 본론   
> 저희는 다음과 같은 내용을 배웠는데요(3시간인 이유가 있습니다^^ 엉엉) 간단하게 요약해보도록 할게요!     
> 1. Numpy와 Pandas는 왜 중요한가? 
> 2. Numpy 넘파이  
>   - 넘파이 배열
>   - 넘파이 배열 - ndarray와 관련된 함수들 
> 3. Pandas 판다스 
>   - 판다스의 주요 구성요소 - DataFrame, Series, Index
>   - 기본 API들
>   - DataFrame ←→ 리스트, 딕셔너리, 넘파이 ndarray 
>   - DataFrame 데이터 삭제 
>   - 데이터 셀렉션 및 필터링 
>   - 판다스 Aggregation 함수와 group by
>   - 결손 데이터 처리 
>   - 판다스 apply, map
   
<br>  

### 1️⃣ Numpy와  Pandas 왜 중요한가?!1️
<img width=800 src="https://user-images.githubusercontent.com/63052097/134837501-b17fca92-af87-41ff-ba34-8a420fc8eea9.png">   
제일 먼저 넘파이와 판다스가 많고 많은 라이브러리 중에 왜 중요한지 자세하게 알아보는 시간을 가졌습니다!    

<br>   
  **첫째** 우리는 이미 기존에 있는 알고리즘을 이용해서 주어진 알고리즘이 결과를 잘 도출해낼 수 있도록 주어진 데이터를 적절하게 추출/가공/변환을 하는 것을 해야 합니다!   
 이때 데이터 처리를 할 때 대부분 넘파이와 판다스 라이브러리를 사용하기 때문에 중요한 것이겠지요?!
<br>    
  **둘째**  머신러닝 알고리즘의 모듈들이 모인 사이킷런이 넘파이 기반에서 작성됐기 때문에 넘파이를 잘 이해해야 합니다!   
 특히 뒤에서 알아보겠지만 넘파이 배열 꼭 중요한 부분이니까 기억해두는 것이 좋아요!
 
<br>

### 2️⃣ Numpy
* 넘파이란?
![](https://user-images.githubusercontent.com/63052097/134839420-81625dd7-4a2c-4fca-8156-589d1c5eca53.png)
넘파이는 파이썬으로 과학 계산을 할 때 필요한 패키지이자 라이브러리 이구요!! **다차원 배열을 위한 다양한 기능 선형대수 연산**을 하는데 다양한 함수가 제공됩니다!    

<br>
<br>

* 넘파이 배열     
 ** 1) scikit-learn에서 기본 데이터 구조!**     
: 넘파이 배열 형태의 데이터를 입력으로 받음 → 우리가 사용할 데이터들은 모두 Numpy 배열로 변환되어야 함    

cf) 파이썬은 자체적으로 배열 자료형을 제공하지 않기 때문에 배열을 사용하기 위해서는 넘파이를 사용

* 넘파이 배열 - ndarray와 관련된 함수들

### 3️⃣ Pandas 
* 판다스의 주요 구성요소 - DataFrame, Series, Index
* 기본 API들
* DataFrame ←→ 리스트, 딕셔너리, 넘파이 ndarray 
* DataFrame 데이터 삭제 
* 데이터 셀렉션 및 필터링 
* 판다스 Aggregation 함수와 group by
* 결손 데이터 처리 
* 판다스 apply, map


