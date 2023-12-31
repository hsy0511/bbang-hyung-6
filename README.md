# bbang-hyung-6
# 머신러닝 실전 스킬
- 교차 검증 방법
- 데이터 증강 기법
- 앙상블
# 교차 검증 방법
Cross validation

데이터가 부족할 때 자주 사용하는 방법으로 데이터를 여러 개로 나누어 각 학습마다 다른 검증 데이터셋을 사용하는 방법
## K-fold Cross Validation
k = 5

파란색 부분만이 검증 데이터 

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/4888608a-9305-46f7-9a76-b5b2aa5752c0)

## 유방암 데이터셋을 이용한 실습
```python
from sklearn.datasets import load_breast_cancer
import pandas as pd
// 유방암 데이터셋과 판다스 패키지를 가져온다.

data = load_breast_cancer()
// data에 유방암 데이터셋을 넣는다.

df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']
// 판다스 데이터 프레임으로 바꾼 후 data에는 유방암 데이터셋에 있는 data를 넣고, columns에는 feature_names를 넣고, target에는 data에 target를 넣어준다

df.head()
```
타겟 0 : 양성, 1 : 악성

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/6baeec21-b909-4a36-8217-7beb06392902)

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/a302ab39-8474-46dc-bb5b-116a2f2a6f6c)

## 전처리
```python
from sklearn.preprocessing import StandardScaler
// 표준화 패키지를 가져온다.

scaler = StandardScaler()
// scaler에 표준화 패키지 객체를 생성한다.

scaled = scaler.fit_transform(df.drop(columns=['target']))
// scaled에 target 데이터를 삭제하고 표준화 형태로 변환한 데이터 값을 넣어준다.

scaled[0]
```

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/b4d9bc94-4386-49ba-9b21-6b4a590b6435)

## 데이터셋 분할
```python
from sklearn.model_selection import train_test_split
// train_test_split 패키지를 가져온다.

x_train, x_val, y_train, y_val = train_test_split(scaled, df['target'], test_size=0.2, random_state=2020)
// 훈련 데이터와 검증 데이터로 데이터를 분할한다.

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
// 데이터가 어떻게 들어가 있는지 확인한다.
```

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/5ddb4586-9ab2-4bd8-85ce-d0f5380330f3)

## 학습, 검증
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
// 로지스틱 회귀 모델과 accuracy_score 정확도 패키지를 가져온다.

model = LogisticRegression()
// 모델 객체 생성

model.fit(x_train, y_train)
// 모델 학습

y_pred = model.predict(x_val)
// 검증 데이터로 정답값 예측

accuracy_score(y_val, y_pred)
// 정답값과 예측값을 비교하여 정확도를 확인한다.
```

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/d269ce83-a510-4086-b167-a21593191456)

## 교차 검증을 위한 데이터 분할
```python
from sklearn.model_selection import KFold
// KFold 데이를 나누는 패키지를 가져온다.

cv = KFold(n_splits=5)
// 데이터를 5개로 나누어주는 객체를 생성한다.

for i, (train_indices, val_indices) in enumerate(cv.split(scaled)):
  print('i =', i + 1)
  print('train_indices', train_indices)
  print('val_indices', val_indices)
// 트레이닝할 인덱스와 검증할 인덱스를 뽑는다.
```

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/dfebd04f-acb9-439a-a0cc-34f086cb1619)

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/8c4a4b77-82b2-44cd-9abd-4a2c06d8607c)

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/44e44b9a-2628-4a8b-ad79-5139ba68e22f)

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/1f5d4ed8-59a9-4f71-b07f-1453f892c627)

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/0ec923de-0973-487e-991b-20739aa89d85)

## 교차 검증 (진부한 방법)
```python
model = LogisticRegression()
// 로지스틱 회귀 객체 생성

cv = KFold(n_splits=5)
// 데이터를 5개로 나누어주는 객체를 생성한다.

accs = []

for train_indices, val_indices in cv.split(scaled):
  x_train = scaled[train_indices]
  y_train = df.loc[train_indices]['target']
// x트레인에 데이터는 전처리한 데이터의 트레인 인덱스를 넣고 y트레인은 트레인 인덱스의 타겟만 넣는다.

  x_val = scaled[val_indices]
  y_val = df.loc[val_indices]['target']
// x벨리데이션은 벨리데이션 인덱스를 넣고, y벨리데이션은 델리데이션 인덱스의 타겟만 넣는다.

  model.fit(x_train, y_train)
// 모델 학습
  y_pred = model.predict(x_val)
// 정답값예측
  accs.append(accuracy_score(y_val, y_pred))
// 정답값과 예측값을 비교하여 정확도를 확인한다.
accs
```

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/7d5985c7-6617-44dd-af1a-6a2a1ffd9761)

## 교차 검증 (간단한 방법)
```python
from sklearn.model_selection import cross_val_score
// cross_val_score 패키지를 가져온다.

model = LogisticRegression()
// 로지스틱 회귀 모델 객체 생성

cv = KFold(n_splits=5)
// 데이터를 5개로 나누어주는 객체를 생성한다.

accs = cross_val_score(model, scaled, df['target'], cv=cv)
// cross_val_score 안에 모델, 학습시킬 데이터, 라벨, 5개로 나누어준 데이터를 넣어주면 교차 검증이된다.

accs
```

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/90b93b56-d236-4461-baf9-a42cc43e0afb)

```python
from sklearn.svm import SVC
// svc 모델 패키지를 가져온다
model = SVC()
// svc 모델 객체 생성
cv = KFold(n_splits=5)
// 데이터를 5개로 나누어주는 객체를 생성한다.
accs = cross_val_score(model, scaled, df['target'], cv=cv)
// cross_val_score 함수를 사용해서 교차 검증을 한다.
accs
```

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/4407970d-bf09-47f2-98b8-0726c4af7a3e)

## 생각해 볼 문제
교차 검증을 사용하지 않았을 때의 결과와 비교해서 더 나아졌다고 말할 수 있을까

# 데이터 증강 기법
Data augmentation (증강)

과대적합을 해결하고 정확도를 높이기 위해 데이터의 양을 증가시키는 방법

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/31fb2241-42dd-4212-8fda-b6cd8bab11a0)

## MNIST 데이터셋을 이용한 실습
```python
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
// digits 데이터셋, 정규화, matplotlib 패키지를 가져온다.

digits = load_digits()
// 데이터셋을 넣는다.

data = digits['data']
target = digits['target']
// 데이터와 타겟을 넣는다.

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)
// 데이터를 정규화한 값으로 변환시켜 넣는다.

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(16, 8))
// 가로 16 세로 8 사이즈에 2행 5열 차트를 만든다.
 
for i, ax in enumerate(axes.flatten()):
  ax.imshow(scaled[i].reshape((8, 8)))
  ax.set_title(target[i])
// 라벨을 데이터셋 로드를 했다. 
```

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/af876edc-c1bc-4cea-9e50-0fe7018a588a)

## 데이터 분할
데이터를 분할하는 이유는 train 세트만 imgaug를 할것이다. 왜냐하면 검증 세트를 건들지 않아야 정확하게 검증을 할 수 있기 때문이다.

```python
from sklearn.model_selection import train_test_split
// train_test_split 패키지를 가져온다.

x_train, x_val, y_train, y_val = train_test_split(scaled, target, test_size=0.2, random_state=2021)
// 훈련 데이터와 검증 데이터로 분할한다.

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
// 데이터가 어떻게 들어가 있는지 확인한다.
```

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/c19015b4-0474-47df-91d5-b135967d9b2c)

## 학습, 검증
```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
// svc모델과 accuracy_score 정확도 패키지를 가져온다.

model = SVC()
// 모델 객체 생성

model.fit(x_train, y_train)
// 모델 학습

y_pred = model.predict(x_val)
// 검증데이터를 이용한 정답값 예측

accuracy_score(y_val, y_pred) * 100
// 정답값과 예측값을 비교하여 정확도 확인
```

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/cb930a2b-fa99-47ee-a2ab-24e1404f3669)

## imgaug 패키지
Image augmentation 을 쉽게 적용할 수 있도록 하는 도와주는 패키지

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/56f6a133-f106-449a-8883-f68f51ae8ed9)

## 증강 전 데이터 출력
```python
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(16, 8))

for i, ax in enumerate(axes.flatten()):
  ax.imshow(x_train[i].reshape((8, 8)))
  ax.set_title(y_train[i])
```

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/d9f19c40-b5c4-4474-bad7-37b464ded5dd)

## 증강 후 데이터 출력

imgaug의 Sequential은 이미지를 입력으로 받는다 (reshape 필요)

```python
import imgaug.augmenters as iaa
// imgaug 패키지를 가져온다.

seq = iaa.Sequential([
  iaa.Affine(
    translate_px={'x': (-1, 1), 'y': (-1, 1)},
    rotate=(-15, 15)
  ),
  iaa.GaussianBlur(sigma=(0, 0.5))
])
// sequential 클래스를 사용하고, affine translate_px를 한다.
// 픽셀을 좌우로 -1부터 좌측으로 1 우측으로 1픽셀씩 랜덤으로 이동 시킬거고 위아래로 -1에서 1픽셀 이동 시킬거고 회전을 -15도에서부터 15까지 랜덤으로 줄것이다.
// 뿌옇게하는 정도를 0부터 0.5까지 랜덤으로 나타나게 한다.

x_train_aug = seq(images=x_train.reshape((-1, 8, 8)))
// 시퀀셜 오브젝트에 이미지를 넣는데 트레인 이미지만 넣고 이미지 어그라는 패키지는 이미지 형태로 받아야 되기 떄문에 8x8로 만들어서 넣어야한다.

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(16, 8))

for i, ax in enumerate(axes.flatten()):
  ax.imshow(x_train_aug[i])
  ax.set_title(y_train[i])
```

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/5d31e241-8ae7-43c1-80a3-e03667ec4459)

## 증강 전 데이터와 증강 후 데이터 합치기
```python
import numpy as np
// numpy 패키지를 가져온다.

x_train_merged = np.concatenate([x_train, x_train_aug.reshape((-1, 64))], axis=0)
// np.concatenate 함수를 사용하여 0번축으로 원래 x트레인 데이터와 어그멘테이션된 x트레인을 합친다.
y_train_merged = np.concatenate([y_train, y_train], axis=0)
// np.concatenate 함수를 사용하여 0번축으로 y트레인과 y트레인을 합친다.
// y트레인은 라벨 값이기 때문에 어차피 어그멘테이션해도 데이터가 안바뀌기 때문에 y트레인만 합쳐도된다.

print(x_train_merged.shape, y_train_merged.shape)
// 데이터가 어떻게 들어가 있는지 확인한다.
```
2배로 커진 데이터 셋을 확인할 수 있다.

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/f4f40503-10ae-4e5e-89c2-1e964a07860b)

## 학습, 검증
```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
// svc모델과 accuracy_score 정확도 패키지를 가져온다.

model = SVC()
// 모델 객체 생성

model.fit(x_train_merged, y_train_merged)
// 모델 훈련

y_pred = model.predict(x_val)
// 검증데이터로 정답값을 예측한다.

accuracy_score(y_val, y_pred) * 100
// 정답값과 예측값을 비교하여 정확도를 확인한다.
```

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/cb578401-f74e-4d1e-93b4-952a5808b523)

## 데이터 증강 +2
정확도가 안좋아졌다구요? 왜 그럴까요?

한 번 더 증강하면 좋아질까요?

```python
seq = iaa.Sequential([
  iaa.Affine(
    translate_px={'x': (-1, 1), 'y': (-1, 1)},
    rotate=(-15, 15)
  ),
  iaa.GaussianBlur(sigma=(0, 0.5))
])

x_train_aug = seq(images=x_train.reshape((-1, 8, 8)))

x_train_merged = np.concatenate([x_train_merged, x_train_aug.reshape((-1, 64))], axis=0)
y_train_merged = np.concatenate([y_train_merged, y_train], axis=0)

print(x_train_merged.shape, y_train_merged.shape)
```

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/d631a2db-d513-47e3-9684-47474fa6d17e)

## 학습, 검증
```python
model = SVC()
// 모델 객체 생성

model.fit(x_train_merged, y_train_merged)
// 모델 훈련

y_pred = model.predict(x_val)
// 검증데이터로 정답값 예측

accuracy_score(y_val, y_pred) * 100
// 정답값과 예측값을 비교하여 정확도를 확인한다.
```

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/907967bc-fa18-4d1c-8fab-bf85a28bcd24)

# 앙상블
여러 개의 모델을 이용해 최적의 답을 찾아내는 기법
- 최소 2% 이상의 성능 향상 효과를 볼 수 있다.
- 적절한 Hyperparameter 튜닝이 필수
- (여러개의 모델을 사용하니까) 일반적으로 학습 시간이 오래 걸린다

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/97ecf929-a28d-4f2d-8819-80df09e0fd0c)

## Voting
### 보스턴 집 값 데이터셋을 사용한 실습
```python
from sklearn.datasets import load_boston
import pandas as pd
// 보스턴 데이터셋과 판다스 패키지를 가져온다.

data = load_boston()
// 데이터에 보스턴 데이터셋을 넣어준다.

df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']
// data는 보스턴 데이터셋에 data를 넣고, columns에는 feature_names를 target은 target을 넣는다.

df.head()
```

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/738a6667-e092-40a8-abdf-78e6fbefbd98)

### 데이터셋 분할
```python
from sklearn.model_selection import train_test_split
// train_test_split 패키지를 가져온다.

x_train, x_val, y_train, y_val = train_test_split(df.drop(columns=['target']), df['target'], test_size=0.2, random_state=2021)
// 트레인 데이터와 검증 데이터로 분할한다.

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
// 데이터가 어떻게 들어가 있는지 확인한다.
```

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/131c16df-fc06-43ac-9c48-93ed3a5d5d46)

### 파이프라인
Pipeline

데이터 전처리 과정와 모델의 학습 과정을 합쳐 하나의 파이프라인으로 만들 수 있어요.

1. 표준화
2. 학습

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/6007bfb4-a5f9-4920-9de4-812975cf0c79)

#### 선형 회귀 파이프라인 예제
```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
// 선형회귀, 표준화, 파이프라인, mean_absolute_error 패키지를 가져온다.


lr = make_pipeline(
    StandardScaler(),
    LinearRegression()
)
// 표준화를 하고 선형회귀를 하는 파이프라인을 lr 변수에 저장한다.

lr.fit(x_train, y_train)
// 자동으로 훈련시켜준다.

lr_y_pred = lr.predict(x_val)
// 자동으로 예측해준다.

lr_mae = mean_absolute_error(y_val, lr_y_pred)
// 회귀문제를 풀기 때문에 mean_absolute_error나 mean_squared_error를 사용하는데 여기서는 mean_absolute_error를 사용하여 검증한다.

lr_mae
```

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/881cc6cf-e1c2-4613-8b0d-79f8209c1fa5)

### Ridge Regression
https://scikit-learn.org/stable/modules/linear_model.html
```python
from sklearn.linear_model import Ridge
// Ridge 모델 패키지를 가져온다.

ridge = Ridge()
// 모델 객체 생성

ridge.fit(x_train, y_train)
ridge_y_pred = ridge.predict(x_val)
// 훈련, 값 예측

ridge_mae = mean_absolute_error(y_val, ridge_y_pred)
// 검증
ridge_mae
``` 

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/58a1a2f8-4647-4499-b476-05b0e1cbf56a)

### Lassso
```python
from sklearn.linear_model import Lasso
// Lasso 모델 패키지를 가져온다.
lasso = make_pipeline(
    StandardScaler(),
    Lasso()
)
// 파이프라인에 표준화, 라쏘를 순서대로 넣어본다.

lasso.fit(x_train, y_train)
lasso_y_pred = lasso.predict(x_val)
// 자동으로 훈련과 예측을 해준다.

lasso_mae = mean_absolute_error(y_val, lasso_y_pred)
// 검증
lasso_mae
```

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/35775866-4a12-4cb8-93aa-354ad7a8c5a8)

### Elastic-Net
```python
from sklearn.linear_model import ElasticNet
// ElasticNet 패키지를 가져온다.

en = make_pipeline(
    StandardScaler(),
    ElasticNet()
)
// 파이프라인에 표준화, ElasticNet을 순서대로 넣어준다.

en.fit(x_train, y_train)
en_y_pred = en.predict(x_val)
// 자동으로 훈련과 예측을 해준다.

en_mae = mean_absolute_error(y_val, en_y_pred)
// 검증
en_mae
```

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/7cbf034a-de07-4eac-b399-9d5630435e8e)

### Voting
투표를 통해 결정하는 방법. 회귀 모델에서는 각 모델 예측값의 평균을 낸다

```python
(lr_y_pred + ridge_y_pred + lasso_y_pred + en_y_pred) / 4
// 예측값을 다 더해서 나눈다.
```

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/af4566e5-60d3-4b5f-a289-088e04ff321e)

### Voting MAE
```python
mean_absolute_error(y_val, (lr_y_pred + ridge_y_pred + lasso_y_pred + en_y_pred) / 4)
// 평균을 예측한 값에 정답값하고 mean_absolute_error를 사용하여 계산한다.
```

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/0958f87e-dae6-4411-88e6-337a32a47d52)

### voting (간단한 방법)
```python
from sklearn.ensemble import VotingRegressor
// VotingRegressor 패키지를 가져온다.

models = [
    ('lr', lr),
    ('ridge', ridge),
    ('lasso', lasso),
    ('en', en)
]
// lr, ridge, lasso, en 회귀들을 models에 넣어준다.

vr = VotingRegressor(models)
// VotingRegressor에 models를 넣어준다.

vr.fit(x_train, y_train)
// VotingRegressor를 훈련시킨다.

y_pred = vr.predict(x_val)
// VotingRegressor를 예측한 값을 저장한다.

mean_absolute_error(y_val, y_pred)
// mean_absolute_error로 계산한다. 
```

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/7006e58d-9af1-4c7f-b677-1969803b85b9)

### MNIST 데이터셋을 사용한 실습 (보스턴 데이터셋은 실수기 때문에 회귀로 풀었지만 MNIST 데이터셋은 10개 숫자를 구분하는 분류문제기 때문에 분류에서 앙상블을 사용하는 것을 보여주기 위해서 실습한다.)
```python
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_spli
// load_digits 데이터셋과 정규화패키지와 train_test_split 패키지를 가져온다.

digits = load_digits()
// digits 데이터셋을 넣는다.

data = digits['data']
target = digits['target']
// data는 digtits 데이터를 넣는다.

scaler = MinMaxScaler()
// scaler에 정규화 패키지 객체 생성

scaled = scaler.fit_transform(data)
// scaled에 정규화 형태로 바꾼 데이터를 넣는다.

x_train, x_val, y_train, y_val = train_test_split(scaled, target, test_size=0.2, random_state=2021)
// 트레인 데이터와 검증 데이터로 데이터를 분할한다.

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
// 데이터가 어떻게 들어가 있는지 확인한다.
```

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/83618772-a0ad-434c-9494-a72aad03c463)

### Voting (Hard)
```python
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
// VotingClassifier, SVC, KNeighborsClassifier, DecisionTreeClassifier, accuracy_score 패키지를 가져온다.

models = [
    ('svc', SVC()), 
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('dt', DecisionTreeClassifier())
]
// SVC, knn, dt를 models에 다 넣어준다.

vc = VotingClassifier(models, voting='hard')
// hard voting을 한다.

vc.fit(x_train, y_train)
// VotingClassifier 학습

y_pred = vc.predict(x_val)
// VotingClassifier 예측

accuracy_score(y_val, y_pred) * 100
// 정확도 확인
```

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/c65ad5dc-0849-4591-bc3d-1be95f40eff1)

### 분류 문제에서의 Voting 방법

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/dfadf6a1-d4a0-4718-bc2d-50a90a316a0d)

### Voting (Soft)
```python
models = [
    ('svc', SVC(probability=True)),
// SVC를 그냥 쓰면 확률값을 반환하지 않기 때문에 probability=True를 사용한다.
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('dt', DecisionTreeClassifier())
]
// SVC, knn, dt를 models에 다 넣어준다.

vc = VotingClassifier(models, voting='soft')
// soft voting을 한다.

vc.fit(x_train, y_train)
// VotingClassifier 학습

y_pred = vc.predict(x_val)
// VotingClassifier 예측

accuracy_score(y_val, y_pred) * 100
// 정확도 확인
```

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/d242ab85-84a5-4f0c-bd5f-292f7cf145fe)

## Bagging
Bootstrap AGGregatING

- Voting은 여러 알고리즘을 조합하여 사용
- Bagging은 한 알고리즘을 데이터를 분할하여 사용

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/318b4bfc-1ef1-4b6f-9f09-abadbeae1805)

### Random Forest
- Decision Tree 기반 가장 유명한 Bagging 앙상블 모델
- (대충써도) 성능이 좋다

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/5cd8a2ec-82ce-4ab3-bbdb-fed15b88a930)

```python
from sklearn.ensemble import RandomForestClassifier
// RandomForestClassifier 패키지를 가져온다.

model = RandomForestClassifier(n_estimators=100)
// decision tree가 100개인 RandomForestClassifier 객체 생성

model.fit(x_train, y_train)
// 모델 훈련

y_pred = model.predict(x_val)
// 정답값 예측

accuracy_score(y_pred, y_val) * 100
// 정확도 확인
```

![image](https://github.com/hsy0511/bbang-hyung-6/assets/104752580/98ff7430-8819-4458-bb93-c7ae3be107a5)

## 앙상블 더 알아보기
- Boosting : https://teddylee777.github.io/machine-learning/ensemble%EA%B8%B0%EB%B2%95%EC%97%90-%EB%8C%80%ED%95%9C-%EC%9D%B4%ED%95%B4%EC%99%80-%EC%A2%85%EB%A5%98-3

