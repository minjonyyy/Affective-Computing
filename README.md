# Affective Computing Project - VOLO

## 📌 프로젝트 개요

-   **팀명**: VOLO
-   **구성원**: 박태정, 한승훈, 이민정
-   **주제**: 웹캠 기반 표정 인식을 활용한 감성 컴퓨팅 서비스

본 프로젝트는 비대면 상황에서 사용자의 얼굴 표정을 인식하여 감정을 분류하고, 실시간으로 이모지 및 통계를 제공하는 시스템입니다. 

카메라를 켜지 않은 상태에서도 감정을 전달할 수 있는 기능을 지원하여 원활한 소통을 돕습니다.

------------------------------------------------------------------------

## 🚩 문제 정의

-   비대면 소통에서 상대방의 감정을 직관적으로 알기 어려움
-   카메라를 켜지 않고도 자동으로 감정을 표현하고 싶은 사용자 니즈

------------------------------------------------------------------------

## 🎯 목표

-   비대면 회의, 온라인 강의, 방송, 인터뷰 등에서 원활한 감정 전달
-   표정 기반 이모지 변환과 세션 종료 후 감정 통계 제공

------------------------------------------------------------------------

## 👥 대상 사용자

-   비대면 소통을 자주 하는 일반 사용자
-   금융 영업, 비대면 강의, 진료, 발표, 방송 등

------------------------------------------------------------------------

## 🛠 주요 기능

-   **기본 기능**
    -   4가지 주요 감정 인식 (Happiness, Sadness, Surprise, Anger)
    -   실시간 이모지 변환 및 브라우저 스트리밍
-   **부가 기능**
    -   감정 통계 제공 (세션 종료 시 감정별 누적 시간/횟수 표시)

------------------------------------------------------------------------

## ⚙️ 기술 스택

### 사용 언어

-   **Python**: 서버, 비디오 처리, 모델 추론
-   **HTML / Jinja2**: 템플릿 렌더링
-   **JavaScript**: 클라이언트 상호작용, 녹화 제어
-   **TypeScript (vendored)**: `socket.io-client` 소스 포함 (직접
    빌드에는 사용되지 않음)

### 핵심 스택 및 라이브러리

-   **웹 프레임워크**: Flask
-   **컴퓨터 비전**: OpenCV (`cv2`)
-   **포즈/랜드마크 추출**: MediaPipe Face Mesh
-   **추론 엔진**: TensorFlow Lite Interpreter (`.tflite` 모델)
-   **수치 연산**: NumPy
-   **템플릿 엔진**: Jinja2 (`render_template`)
-   **스트리밍**: Flask Response 기반 MJPEG
    (`multipart/x-mixed-replace`)
-   **정적 리소스**: PNG 이모지, `static/recorder.js`

------------------------------------------------------------------------

## 📂 디렉터리 구조

    src/
     ├─ flask_opencv/
     │   ├─ server.py              # Flask 서버 구동, /, /video_feed 라우트
     │   ├─ camera.py              # 웹캠 프레임 획득/처리 유틸
     │   ├─ templates/index.html   # 스트리밍 <img> + 감정 카운트 표시
     │   └─ static/recorder.js     # 녹화 제어 스크립트 (/record_status 필요)
     │
     ├─ flask_opencv(socket)/      # 소켓 기반 변형 (socket.io-client 소스 포함)
     │   └─ static/lib/            # vendored socket.io-client
     │
     ├─ Facial-emotion-recognition/
     │   └─ 학습/추론 관련 스크립트
     │
     ├─ model/keypoint_classifier/
     │   ├─ model.tflite           # TensorFlow Lite 모델
     │   ├─ labels.csv             # 감정 라벨 정의
     │   └─ wrapper.py             # 추론 래퍼
     │
     └─ emojis/                    # 감정별 이모지 이미지 리소스

------------------------------------------------------------------------

## 🔄 동작 흐름

1.  브라우저가 `/video_feed` 요청
2.  Flask 서버에서 웹캠 프레임 캡처 → RGB 변환
3.  MediaPipe Face Mesh로 얼굴 랜드마크 추출
4.  좌표 정규화 후 TFLite 모델(KeyPointClassifier)로 감정 분류
5.  감정별 카운트 누적 및 결과 오버레이
6.  MJPEG로 실시간 스트리밍 → 브라우저에서 `<img src="/video_feed">`로
    표시
7.  세션 종료 시 감정별 통계(Happiness, Sadness, Surprise, Anger) 제공

------------------------------------------------------------------------


## 🔄 구현 방식
1. **웹캠 프레임 캡처**  
   - OpenCV로 프레임 획득  
   - 좌우 반전 및 RGB 변환  

2. **랜드마크 추출**  
   - MediaPipe Face Mesh로 얼굴 특징점 좌표 검출  
   - 좌표 정규화 및 상대좌표 변환  

3. **전처리**  
   - 랜드마크 좌표를 벡터로 변환  
   - 필요 없는 영역 제거, 특징점 좌표만 추출  

4. **모델 추론**  
   - TensorFlow Lite 모델(`.tflite`)로 감정 분류 수행  
   - 예측된 감정을 Angry / Happy / Sad / Surprise 중 하나로 결정  

5. **결과 반영**  
   - 추론 결과에 따라 이모지 오버레이  
   - 감정별 카운트 서버 메모리에 누적  

6. **실시간 스트리밍**  
   - Flask Response로 MJPEG 스트리밍 제공  
   - 브라우저에서 `<img src="/video_feed">`로 실시간 표시  

7. **세션 종료 후 통계 제공**  
   - 각 감정별 누적 시간/횟수를 템플릿에 표시  

---

## 📊 모델 학습
- **데이터셋**
  - Kaggle 공개 데이터셋 (~7,000장) + 자체 수집 데이터(50장)  
  - 표정별 좌표 데이터를 CSV/엑셀로 정리  

- **모델 구조**
  - 다층 퍼셉트론 (MLP, Multi-Layer Perceptron)  
  - 입력: 얼굴 랜드마크 좌표 (정규화된 특징점)  
  - 은닉 계층: Fully Connected Layers  
  - 출력: 4개 감정 클래스 (Happiness, Sadness, Surprise, Anger)  

- **학습 과정**
  - 데이터 전처리 → 학습/검증 데이터 분할  
  - MLP 학습 (TensorFlow → TensorFlow Lite 변환)  
  - 모델 최적화 후 `.tflite`로 배포  

- **추론**
  - Flask 서버에서 `TensorFlow Lite Interpreter`로 모델 로딩  
  - 프레임 단위로 추론 수행 → 결과 반환  

---


## 📸 구현 화면

<img width="2560" height="1600" alt="image" src="https://github.com/user-attachments/assets/372ae2c6-f09f-49d3-bbf3-34a3811002df" />
<img width="2560" height="1600" alt="image" src="https://github.com/user-attachments/assets/850bdf40-201a-4816-8409-36b273692cc8" />
<img width="2560" height="1600" alt="image" src="https://github.com/user-attachments/assets/c1a1e448-b4e2-4dcc-91a8-144476672354" />
<img width="2560" height="1600" alt="image" src="https://github.com/user-attachments/assets/a58d8573-345b-4892-b298-89dd0c5572f4" />
<img width="1959" height="1198" alt="image" src="https://github.com/user-attachments/assets/722c9fba-dd01-4ba9-ab4a-7be515713895" />


------------------------------------------------------------------------

## 🚀 적용 가능 분야

-   비대면 진료 (환자의 감정 상태 파악)
-   온라인 강의 (학습자 상태 모니터링)
-   방송 및 발표 (관객 반응 분석)
-   비대면 인터뷰 (응시자 감정 파악)

------------------------------------------------------------------------

## ⚠️ 한계 및 개선 필요사항

-   일부 코드에서 로컬 절대경로가 하드코딩되어 있음 → 상대경로화 필요
-   `/record_status` 엔드포인트가 미구현된 상태 (녹화 기능 완전 사용
    불가)
