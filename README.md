## Log
### v.1
  - trainer 모듈에서 accuracy가 항상 1.0으로 출력되던 오류를 수정하였습니다.
  - data_loader 모듈에서 label field의 <unk> token으로 인해 loss가 계산되지 않던 오류를 수정하였습니다.
### v.1.1
  - RNN 모델 train paramter 추가
### v.1.2
  - CNN 기반 모델을 추가하였습니다.
  - template이 각 epoch의 best loss를 제대로 반영하도록 trainer.py를 수정하였습니다.
