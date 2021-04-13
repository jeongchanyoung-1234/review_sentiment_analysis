## Log
### v.1
  - trainer 모듈에서 accuracy가 항상 1.0으로 출력되던 오류를 수정하였습니다.
  - data_loader 모듈에서 label field의 <unk> token으로 인해 loss가 계산되지 않던 오류를 수정하였습니다.
### v.1.1
  - RNN 모델 train paramter 추가
### v.1.2
  - CNN 기반 모델을 추가하였습니다.
  - template이 각 epoch의 best loss를 제대로 반영하도록 trainer.py를 수정하였습니다.
### v.1.3
  - Voting 기능을 추가하였습니다.



## 사용법
```
python train.py --model_fn modle.pth --model_name rnn --file_path ./review.txt --batch_size 128 --n_epochs 500 --train_ratio .8 --emb_dim 64
```


## Parameter
### 1. model_fn(필수)
- 각 epoch의 종료 시 best model을 저장합니다. 그 때 사용될 모델 파일의 이름을 지정합니다. pytorch model의 state_dict를 기억하므로 pth파일 형식을 사용합니다.
### 2. rnn
- rnn 아키텍처로 모델링합니다.
### 3. cnn
- cnn 아키텍처로 모델링합니다.
- rnn과 cnn 모두 사용 시 예측에 두 모델의 결과값을 평균내어 사용할 수 있습니다.  
### 4. file_path(필수)
- 훈련에 사용할 텍스트 데이터를 지정합니다. tsv 파일 형식으로 저장되어 첫 번째 컬럼은 라벨, 두 번째 컬럼은 본문이어야 합니다.
### 5. gpu_id
- 연산장치를 결정합니다. 자동으로 적절한 디바이스를 사용합니다.
### 6. verbose
- 출력할 정보의 양을 결정합니다. 0: 출력하지 않음 / 1: 배치의 루프마다 결과 출력 / 2: 에포크마다 결과 출력
### 7. batch_size
- 훈련에 사용할 배치의 크기를 지정합니다. 디폴트는 64입니다.
### 8. n_epochs
- 훈련 에포크를 지정합니다. 디폴트는 5입니다.
### 9. train_ratio
- 훈련에 사용될 데이터의 비율을 지정합니다. 나머지는 검증에 사용됩니다. 디폴트는 0.8입니다.
### 10. dropout
- rnn 혹은 cnn 계층에 사용될 dropout의 비율을 지정합니다. 디폴트는 0.2입니다.
### 11. hidden_size(rnn)
- rnn레이어의 은닉상태의 크기를 지정합니다. 디폴트는 32입니다.
### 12. n_layers(rnn)
- rnn레이어의 층 개수를 지정합니다. 디폴트는 3(multi-layered)입니다.

### 13. max_length(rnn)
- 입력 시퀀스의 최대 길이를 지정합니다. 디폴트는 256입니다.
### 14. window_size(cnn)
- cnn레이어의 윈도우 크기를 지정합니다. cnn층위에서 각각 다른 크기의 윈도우를 사용하여 단어의 출현 패턴을 파악할 수 있습니다. 디폴트는 [2, 3, 4]입니다.
### 15. n_filters(cnn)
- cnn레이어의 커널 개수를 지정합니다. 디폴트는 32입니다.
### 16. use_batchnorm(cnn)
- 배치 정규화 사용 여부를 결정합니다. 디폴트는 True이며 False로 지정 시 dropout을 사용합니다.
### 17. use_padding(cnn)
- 현재 지원하지 않습니다.

## 결과 예시
```
positive	생각 보다 밝 아요 ㅎㅎ
negative	쓸 대 가 없 네요
positive	깔 금 해요 . 가벼워 요 . 설치 가 쉬워요 . 타 사이트 에 비해 가격 도 저렴 하 답니다 .
positive	크기 나 두께 가 딱 제 가 원 하 던 사이즈 네요 . 책상 의자 가 너무 딱딱 해서 쿠션 감 좋 은 방석 이 필요 하 던 차 에 좋 은 제품 만났 네요 . 냄새 얘기 하 시 는 분 도 더러 있 던데 별로 냄새 안 나 요 .
positive	빠르 고 괜찬 습니다 .
positive	유통 기한 도 넉넉 하 고 좋 아요
positive	좋 은 가격 에 좋 은 상품 잘 쓰 겠 습니다 .
negative	사이트 에서 늘 생리대 사 서 쓰 는데 오늘 처럼 이렇게 비닐 에 포장 되 어 받 아 본 건 처음 입니다 . 위생 용품 이 고 자체 도 비닐 포장 이 건만 소형 박스 에 라도 넣 어 보내 주 시 지 . ..
negative	연결 부분 이 많이 티 가 납니다 . 재질 구김 도 좀 있 습니다 .
positive	애기 태열 때문 에 구매 해서 잘 쓰 고 있 습니다 .
positive	항상 쓰 던 거 라 만족 합니다 . 항상 쓰 던 거 라 만족 합니다 .
positive	자 ~~~ 알 쓰 겠 습니다 .
positive	등기 구 매번 깜빡이 고 갈 고 그랬 는데 . . 이거 하 면 안 그러 겠 죠 . .
positive	케이스 잘 받 았 습니다
positive	빠른 배송 에 잘 씻기 고 좋 아요 ^.^
positive	저렴 한 금액 으로 잘 삿 습니다
```

