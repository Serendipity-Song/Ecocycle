import urllib.request
import numpy as np
from typing import Optional
from fastapi import FastAPI
import tensorflow as tf
from PIL import Image
import requests

app = FastAPI()
model = tf.saved_model.load("/Users/haein/PycharmProjects/Ecobyte/tensorflow-yolov4-tflite/checkpoints/yolov4-416")

# 이미지 분류 API 엔드포인트
@app.get("/classify")
def classify_image(url: str):
    try:
        # 이미지 다운로드
        response = requests.get(url, verify=False)
        with open("image.jpg", "wb") as f:
            f.write(response.content)

        # 이미지 전처리
        image = Image.open("image.jpg")
        image = image.resize((416, 416))
        image = np.array(image)
        image = image / 255.0  # 이미지 정규화
        image = np.expand_dims(image, axis=0)

        # 예측
        inputs = tf.constant(image, dtype=tf.float32)
        detections = model(inputs)

        class_labels = detections[..., 5:]  # 클래스 레이블 정보 추출

        if class_labels.shape[-1] == 1:
            # 클래스 레이블이 1개인 경우
            class_ids = np.squeeze(class_labels)  # 클래스 인덱스 추출
        else:
            # 클래스 레이블이 3개 이상인 경우
            class_ids = np.argmax(class_labels, axis=-1)  # 가장 높은 확률의 클래스 인덱스 추출

        results = [{"class_id": int(class_id)} for class_id in np.nditer(class_ids)]

        return {"predictions": results}

    except Exception as e:
        return {"error": str(e)}



# 예시 이미지 url
# https://ecocycle-image-upload-bucket.s3.ap-northeast-2.amazonaws.com/upload/image/0547809b-6429-4918-966f-Oab219fc735a

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}



