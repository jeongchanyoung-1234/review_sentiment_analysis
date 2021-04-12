## Log
### v.1
  - trainer 모듈에서 accuracy가 항상 1.0으로 출력되던 오류를 수정하였습니다.
  - data_loader 모듈에서 label field의 <unk> token으로 인해 loss가 계산되지 않던 오류를 수정하였습니다.
### v.1.1
  - RNN 모델 train paramter 추가
### v.1.2
  - CNN 기반 모델을 추가하였습니다.
  - template이 각 epoch의 best loss를 제대로 반영하도록 trainer.py를 수정하였습니다.


## 사용법
```
python train.py --model_fn modle.pth --model_name rnn --file_path ./review.txt --batch_size 128 --n_epochs 500 --train_ratio .8 --emb_dim 64
```


## Parameter
### 1. model_fn(필수)
- 각 epoch의 종료 시 best model을 저장합니다. 그 때 사용될 모델 파일의 이름을 지정합니다. pytorch model의 state_dict를 기억하므로 pth파일 형식을 사용합니다.
### 2. model_name(필수)
- 사용할 모델을 지정합니다. 현재 'cnn'과 'rnn(LSTM)'을 지원합니다.    
### 3. file_path(필수)
- 훈련에 사용할 텍스트 데이터를 지정합니다. tsv 파일 형식으로 저장되어 첫 번째 컬럼은 라벨, 두 번째 컬럼은 본문이어야 합니다.
### 4. gpu_id
- 연산장치를 결정합니다. 자동으로 적절한 디바이스를 사용합니다.
### verbose
- 출력할 정보의 양을 결정합니다. 0: 출력하지 않음 / 1: 배치의 루프마다 결과 출력 / 2: 에포크마다 결과 출력
### 5. batch_size
- 훈련에 사용할 배치의 크기를 지정합니다. 디폴트는 64입니다.
### 6. n_epochs
- 훈련 에포크를 지정합니다. 디폴트는 5입니다.
### 7. train_ratio
- 훈련에 사용될 데이터의 비율을 지정합니다. 나머지는 검증에 사용됩니다. 디폴트는 0.8입니다.
### 8. dropout
- rnn 혹은 cnn 계층에 사용될 dropout의 비율을 지정합니다. 디폴트는 0.2입니다.
### 9. hidden_size(rnn)
- rnn레이어의 은닉상태의 크기를 지정합니다. 디폴트는 32입니다.
### 10. n_layers(rnn)
- rnn레이어의 층 개수를 지정합니다. 디폴트는 3(multi-layered)입니다.
### 11. window_size(cnn)
- cnn레이어의 윈도우 크기를 지정합니다. cnn층위에서 각각 다른 크기의 윈도우를 사용하여 단어의 출현 패턴을 파악할 수 있습니다. 디폴트는 [2, 3, 4]입니다.
### 12. n_filters(cnn)
- cnn레이어의 커널 개수를 지정합니다. 디폴트는 32입니다.
### 13. use_batchnorm(cnn)
- 배치 정규화 사용 여부를 결정합니다. 디폴트는 True이며 False로 지정 시 dropout을 사용합니다.
### 14. use_padding(cnn)
- 현재 지원하지 않습니다.
