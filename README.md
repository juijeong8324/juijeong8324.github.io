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
- **머신러닝 패키지** 
**scikit-learn**   
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
▫ **matplotlib :** 데이터의 직관적 이해, 시각적인 표현을 위한 라이브러리 ****/ 많은 ****다른 시각화 솔루션들이 matplotlib을 기반으로 만들어짐!(seaborn도 마찬가지)    

- **대화형 파이썬 툴**    
▫  노트필기 하듯이 코드 + 필기할 수 있는 툴 / **코드를 분할해서 실행이 가능!!**   

<br>
<hr>
<br>

## 2. 본론   
> 크흡 저희가 왜 3시간이 걸렸는지 목차를 보시면 알겁니다!    
> 1. Numpy와 Pandas는 왜 중요한가? 
> 2. Numpy 넘파이  
> 3. 넘파이 배열
> 4. 넘파이 배열 - ndarray와 관련된 함수들 
> 5. Pandas 판다스 
> 6. 판다스의 주요 구성요소 - DataFrame, Series, Index
> 7. 기본 API들
> 8. DataFrame ←→ 리스트, 딕셔너리, 넘파이 ndarray 
> 9. DataFrame 데이터 삭제 
> 10. 데이터 셀렉션 및 필터링 
> 11. 판다스 Aggregation 함수와 group by
> 12. 결손 데이터 처리 
> 13. 판다스 apply, map
   
<br>    
### 1️⃣ Numpy와  Pandas 왜 중요한가?!

