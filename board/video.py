import io
import os
import time  # time 모듈 import
from flask import Blueprint, request, jsonify, session
from flask_restx import Api
from PIL import Image
import numpy as np
from keras.models import load_model
from diffusers import StableDiffusionImg2ImgPipeline
import torch

bp_video = Blueprint('video', __name__)
api = Api(bp_video)

progress = 0

@bp_video.route('/progress')
def progress_status():
    global progress
    return jsonify({"progress": progress})

def update_progress(value):
    global progress
    progress = value

@bp_video.route('/api/predict', methods=['POST'])
def predict():
    global progress
    progress = 0  # Reset progress at the start

    # 나이와 성별 데이터 가져오기
    age = request.form.get('age', type=int)
    gender = request.form.get('gender', type=str)

    # 성별에 따라 다른 모델 경로 및 레이블 설정
    if gender == 'female':
        model_path = '/workspace/python-flask/board/models/fe_keras_model.h5'
        labels = ['cat', 'chipmunk', 'deer', 'dog', 'fox', 'frog', 'rabbit']
    else:
        model_path = '/workspace/python-flask/board/models/keras_model.h5'
        labels = ['bear', 'dino', 'dog', 'horse', 'monkey', 'tiger', 'wolf']

    model = load_model(model_path)

    image_file = request.files['image']
    image = Image.open(image_file)
    image = image.convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    update_progress(10)

    predictions = model.predict(image_array)
    predicted_label = labels[np.argmax(predictions)]

    update_progress(30)

    # Stable Diffusion 모델을 사용하여 캐릭터 생성
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")  # GPU로 설정

    init_image = image.resize((512, 512)).convert('RGB')  # 모델 입력 크기에 맞게 조정 및 RGB로 변환
    prompt = f"A cute cartoon character with features of a {predicted_label}, reflecting the facial features of the provided image. The character is a {gender} aged {age}."

    update_progress(50)

    # 캐릭터 생성 중 진행 상태 업데이트
    def callback(step, timestep, latents):
        progress_value = 50 + int((step / 37) * 50)  # step을 50에서 100 사이로 스케일링
        update_progress(progress_value)

    generated_images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5, callback=callback, callback_steps=1)

    update_progress(100)

    # 결과 이미지 저장
    output_image_path = "/workspace/python-flask/board/static/generated_image.png"
    generated_images.images[0].save(output_image_path)

    # 결과 반환
    return jsonify({
        'predicted_label': predicted_label,
        'generated_image_url': f'/static/generated_image.png?{int(time.time())}'  # 타임스탬프 추가
    })
