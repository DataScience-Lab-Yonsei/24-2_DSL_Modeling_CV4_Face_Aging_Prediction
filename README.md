# 24-2_DSL_Modeling_CV4_Face_Aging_Prediction
# 실종 아동 현재 모습 예측 모델 구현

# 💭 0. Intro

---

실종 아동 문제는 세계적으로 심각한 사회적 이슈 중 하나입니다. 수많은 아동들이 매년 실종되고 있으며, 우리나라의 경우 18세 미만 실종아동 신고 건수가 매년 줄지 않는 추세입니다. 특히 올해 7월을 기준으로 실종 신고 후 1년 넘게 가족의 품으로 돌아오지 못한 장기실종아동은 1094명에 달합니다. 

![image](https://github.com/user-attachments/assets/e8869133-a0a8-49d5-be24-590a607f54d9)

이 중 93%에 해당하는 1020명은 20년 이상 실종된 상태로 머물러 있습니다. 이러한 장기 실종 아동의 경우, 시간의 경과에 따라 얼굴과 신체 특징이 자연스럽게 변하기 때문에, 초기 사진만으로는 그들을 식별하기가 매우 어려워지기에 시간이 흐르면서 변화한 외모를 예측하는 것이 실종아동을 찾는 과정에서 큰 도전 과제 중 하나입니다.

따라서 저희는 오늘 발전된 딥러닝 기술을 해당 사회 문제에 접목시켜 실종 아동의 현재 모습을 예측하여 3D 영상으로 구현하는 프로젝트를 진행해 보았습니다. 이를 통해 실종아동의 가족들에게 희망을 전달하고, 더 나아가 사회적으로도 실종 아동 문제 해결에 기여할 수 있기를 바랍니다. 

# 🎯 1. Project Pipeline

---

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8b114675-4688-456d-bc96-d12370522413/c764540c-463b-4516-8f71-907dad4b1fd5/image.png)

- **FADING(Face Aging via Diffusion-based Editing)** : diffusion 기반의 age transfer 모델
- **GOAE(Geometry and Occlusion-Aware Encoding) :** GAN 기반의 3D rendering 모델

위의 두 모델을 활용하여 인풋 이미지의 연령을 타겟 연령에 맞게 변환하고, 3D 영상을 생성하는 것이 저희 프로젝트의 목적이었습니다. 

# 🧠 2. Model

---

## 1). FADING

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8b114675-4688-456d-bc96-d12370522413/e1ad6342-d61f-4772-93f6-a1ad147b42d5/image.png)

- Diffusion 기반의 age transfer 모델

해당 모델은 사전 학습된 Latent Diffusion Model을 fine-tuning 시켜 실제 이미지 연령 변환에 활용합니다. 크게 

1. Specialization 
- ‘인간’ 이미지 ‘특정’ 나이로의 변환이라는 태스크에 맞도록 두 개의 프롬프트를 사용해 사전학습된 모델 최적화
1. Age editing 
- 실제 이미지를 텍스트에 맞게 편집하기 위해 null text inversion을 통한 원본 이미지와 textual prompt의 inverting,
- 이후  diffusion process 중에 cross-attention map을 주입하여 이미지를 편집하는 p2p 접근 방식으로 이미지의 연령과 관련 있는 픽셀 변환

의 두 단계로 나누어 볼 수 있습니다. 

### (1) FADING - Fine Tuning

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8b114675-4688-456d-bc96-d12370522413/1b8b5bca-1aec-4a09-98ae-4fdc2f1a101d/image.png)

FADING의 경우 나잇대가 있는 인물의 사진은 연령대에 관계없이 결과물을 잘 출력하였으나, 어린 나이 인물의 이미지를 인풋으로 받는 경우 얼굴의 특징을 제대로 잡지 못해 아웃풋으로 서양인의 얼굴을 뱉는 경우가 있었습니다. 따라서 발표 자료를 통해 확인할 수 있듯이 RetinaFace를 활용한 이미지 데이터 전처리를 거쳐 동양인 얼굴에 맞게 FADING 모델 fine tuning을 시도하였습니다. 그러나 기존의 모델에 비해 성능이 현저히 떨어져 해당 repository상에서는 따로 자세히 언급하지 않겠습니다. 

### (2) FADING - Prompt Changing

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8b114675-4688-456d-bc96-d12370522413/433eecf5-7a0b-45ac-a1a1-09275f00526c/image.png)

앞서 언급했던 기존 FADING의 문제점은 크게 

1) 연령대에 맞지 않는 나이 든 이미지를 출력함. (일반적으로 동일한 연령이더라도 서양인의 경우 나이가 더 들어보인다는 인식이 있음)

2) 여성의 경우 20대 이상부터 서양식의 짙은 화장을 입힌 이미지가 출력됨

으로 나누어 볼 수 있었습니다. 따라서 Age Editing 단계에서의 입력 프롬프트를 기존의 “ Photo of a [$a$] year old person”에서

1. Photo of a [$a$] year old asian person
2. Photo of a [$a$] year old young looking person

으로 바꾸어 태스크를 수행해 보았습니다. 그 결과 출력 이미지에서 유의미한 차이가 나타남을 확인하였습니다. (남성의 경우 큰 차이 없었음)

### (3) FADING - Results

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8b114675-4688-456d-bc96-d12370522413/85630bda-5a8a-48ac-8a50-f6bef6f21453/image.png)

## 2). GOAE

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8b114675-4688-456d-bc96-d12370522413/c95c09fd-a887-4911-b1bb-5192872a0d00/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8b114675-4688-456d-bc96-d12370522413/4078f6a6-6238-4dcb-bb49-afab64ed6e87/image.png)

- GAN 기반의 3D rendering 모델

해당 모델은 기존의 EG3D가 제공하는 3D 형상 정보를 활용하여 이미지의 디테일을 살린 inversion을 구현한 모델입니다. GOAE는 크게 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8b114675-4688-456d-bc96-d12370522413/41023d9b-3df7-430e-b3db-67534677a5aa/image.png)

1. $W$ Space Inversion : Encoder를 통해 인풋 이미지를 잠재 공간 상에 매핑, 사전학습된 EG3D의 Generator를 활용하여 Tri-plane representation 값 반환. 이를 통해 이미지 재구성

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8b114675-4688-456d-bc96-d12370522413/091e1834-00ca-4244-8125-9c28ef46b9b9/image.png)

1. - 1) 인풋 이미지와 stage 1에서 얻은 reconstructed 이미지 간의 차이 계산, 

        -2) 해당 residual image의 feature에서 계산된 value          와 key,  기존 이미지 feature에서 계산한 query를 받아 cross attention을 통한 관계 학습, refining feature F!

        -3) detailed image로 3D rendering

의 단계로 나누어 볼 수 있습니다. 

### 1) GOAE - Data Preprocessing

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8b114675-4688-456d-bc96-d12370522413/e1f63de7-c412-41c7-975b-943b10af7d7f/image.png)

저희는 프로젝트에서 SPI official github의 데이터 전처리 함수를 활용해 GOAE에 필요한 input을 마련하였습니다.

(각 이미지의 16개의 Extrinsic (위치) 변수 + 9개의 Intrinsic (스케일링) 변수) 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8b114675-4688-456d-bc96-d12370522413/dd4c9e47-f08a-40ee-9ad6-649987f6f8ae/image.png)

### 2) GOAE - Results

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8b114675-4688-456d-bc96-d12370522413/3123f287-b5fc-485f-8b7d-5d2c1543dc7c/image.png)

[KakaoTalk_20240924_023245163.mp4](https://prod-files-secure.s3.us-west-2.amazonaws.com/8b114675-4688-456d-bc96-d12370522413/fd591dfd-3fe7-4793-9275-5b9471762a8e/KakaoTalk_20240924_023245163.mp4)

# 🎆 3. Results

---

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8b114675-4688-456d-bc96-d12370522413/1e7b7128-a335-4c00-a388-00a8292c6a69/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/8b114675-4688-456d-bc96-d12370522413/7d074994-0407-43f8-bd14-0b3553e1224b/image.png)

[KakaoTalk_20240924_013439522.mp4](https://prod-files-secure.s3.us-west-2.amazonaws.com/8b114675-4688-456d-bc96-d12370522413/f1416e93-db28-424b-8fd7-89b459d1b360/KakaoTalk_20240924_013439522.mp4)

 

실제 실종아동의 현재 모습을 추정한 영상입니다. 

# 📌 4. Limitations

---

- FADING - 실종 아동의 현재 모습 추정 태스크에 필요한 아동 이미지의 경우 화질이 좋지 않은 이미지가 많음

⇒ 데이터셋 확보 혹은 기존의 이미지 화질 개선해서 fine tuning 시도 가능

- GOAE - 카메라 변수 추정과 재생성 과정에서의 성능 악화

- Input 이미지의 화질, 배경 속 다른 물체의 유무, 배경색의 통일 등에 큰 영향을 받아 아직 일반화에는 무리가 있음
