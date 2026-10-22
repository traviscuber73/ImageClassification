# 새 PyTorch 노트북 시작하기

프로젝트 위치: C:\Users\박정원\ImageClassification

1. 예전 학습 셀이 계속 실행 중이면 그 노트북에서 Kernel → Shut Down Kernel로 종료합니다. 저장하지 않은 모델 상태는 사라집니다.
2. 위 프로젝트 폴더에서 START_PYTORCH.cmd를 더블클릭합니다.
3. 브라우저에서 01_PyTorch_Intel_GPU.ipynb를 엽니다. 시작 파일은 이 새 노트북을 직접 열도록 지정되어 있습니다.
4. 커널은 Python 3.14 (Intel GPU)를 선택합니다.
5. 코드 셀을 하나씩 Shift+Enter로 실행합니다. 처음에는 Run All을 누르지 않습니다.

PyTorch 재설치는 필요하지 않습니다. 현재 가상환경의 2.14.1+xpu와 기존 happy/sad 이미지를 사용합니다.
이전처럼 오래 기다려도 아무 표시가 없는 상황을 줄이도록 첫 묶음의 처리 단계와 이후 각 묶음의 시간을 출력합니다.

새 시작 설정은 128×128 이미지, 배치 크기 8, 3에포크입니다.
이전 코드보다 작은 CNN을 사용합니다. 정확한 학습 시간과 성능은 실행 결과로 확인합니다.
각 에포크 완료 후 모델이 models\pytorch_restart_실행시각\imageclassifier.pt에 자동 저장됩니다.

진행이 멈추면 마지막으로 출력된 단계 또는 오류를 확인하세요.
GPU 인식 성공만으로 모든 GPU 연산의 성공이 확인되는 것은 아닙니다.
새 노트북에는 GPU 실행을 확인하는 작은 계산 셀이 포함되어 있습니다.

명령 프롬프트에서 직접 열고 싶다면:
```bat
cd /d "C:\Users\박정원\ImageClassification"
.\.venv-torch-xpu\Scripts\python.exe -m jupyterlab "01_PyTorch_Intel_GPU.ipynb"
```

