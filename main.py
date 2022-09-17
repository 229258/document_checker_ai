import os
import logging
import io
import copy

from fastapi import FastAPI, Request  # File, UploadFile, Form, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run

import re
import base64
import cv2

import numpy as np

import tensorflow as tf
# from tensorflow.keras.utils import get_file
# from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow import expand_dims

import country_converter as coco
from babel import Locale

from datetime import datetime, timezone
import pymongo

doc_model: tf.keras.Model

app = FastAPI()
logger = logging.getLogger("api")
model_dir = "dokumenciki5"

# global doc_model
doc_model = tf.keras.models.load_model(model_dir)

class_names = ['AUT-B-', 'AUT-BO', 'BEL-BO', 'BGR-BO', 'CYP-BO', 'CZE-BO', 'DEU-BO', 'DEU-BP', 'ESP-BO', 'EST-BO',
               'FIN-BO', 'FIN-BP', 'FRA-BO', 'GRC-BO', 'GRC-BS', 'HRV-BO', 'HUN-BO', 'HUN-BP', 'ITA-BO', 'LTU-BO',
               'LUX-BO', 'LVA-BO', 'MLT-BO', 'NLD-BO', 'POL-BF', 'POL-BO', 'PRT-BO', 'ROU-BO', 'ROU-BP', 'SVK-BO',
               'SWE-BO']

origins = ["*"]
methods = ["*"]
headers = ["*"]

msg_pack1 = {
    "country_short": "Polska",
    "country": "Polska Rzeczpospolita Ludowa",
    "country_code": "POL",
    "prediction": "Więcej niż 99,99%"
}

msg_pack2 = {
    "country_short": "Włochy",
    "country": "Republika Makaronu",
    "country_code": "ITA",
    "prediction": "Mniej niż 0,01%"
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=methods,
    allow_headers=headers,
)

MONGODB_URI = "mongodb+srv://grzybi:mongoDB@dokumenciki.irdcwgw.mongodb.net/?retryWrites=true&w=majority"
client = pymongo.MongoClient(MONGODB_URI)

print(client.server_info())
print(client.list_database_names())

mydb = client["prediction_requests"]
mycol = mydb["request"]



# @app.http_method("url_path")
# async def functionName():
#     return something


# @app.post("/predict_fake/")
# async def predict_fake():
#     print("fake post")
#
#
#
#
#     return [msg_pack1, msg_pack2]


# class Item(BaseModel):
#     name: str
#     description: Union[str, None] = None
#     price: float
#     tax: Union[float, None] = None


# class Dowod(BaseModel):
#     dataToUpload: List[str]
#


def predict_doc_side(received_image):

    img1 = tf.io.decode_image(received_image, channels=3, expand_animations=False)
    img1 = tf.image.resize(img1, [220, 154])
    img_array_1 = img_to_array(img1)
    img_array_1 = expand_dims(img_array_1, 0)

    global doc_model
    prediction_1 = doc_model.predict(img_array_1)

    # pred_ind = np.argmax(prediction, axis=1)
    # score = tf.nn.softmax(prediction[0])

    list_of_predictions = np.array(prediction_1[0])
    strongest_predictions = np.flip(np.argsort(list_of_predictions)[-2:])
    countries = [class_names[country] for country in strongest_predictions]

    locale = Locale('pl')

    pred11 = 100 * list_of_predictions[strongest_predictions[0]]
    iso2 = coco.convert(names=countries[0][:3], to='ISO2')
    local_msg_pack1 = {
        "country_short": locale.territories[iso2.upper()],
        "country": coco.convert(names=countries[0][:3], to='name_official'),
        # "country_code": class_names[np.argmax(score)][:3],
        "country_code": countries[0][:3],
        "prediction": f"{pred11:.2f}%" if pred11 < 99.99 else 'Więcej niż 99,99%'
    }

    pred12 = 100 * list_of_predictions[strongest_predictions[1]]
    iso2 = coco.convert(names=countries[1][:3], to='ISO2')
    local_msg_pack2 = {
        "country_short": locale.territories[iso2.upper()],
        "country": coco.convert(names=countries[1][:3], to='name_official'),
        "country_code": countries[1][:3],
        "prediction": f"{pred12:.2f}%" if pred12 > 0.01 else 'Mniej niż 0.01%'
    }

    print(local_msg_pack1, local_msg_pack2)

    return [local_msg_pack1, local_msg_pack2]


@app.post("/predict_test/")
async def predict_test():
    print("predict_test")
    return [msg_pack1, msg_pack2]


@app.post("/predict/")
async def predict(
        request: Request
):
    print("predict")
    da = await request.body()
    counter = re.split(b'data:image\/[A-Za-z]+;base64', da)
    # print(counter[1][:500])
    # print(counter[2][:500])

    received_image_1 = base64.decodebytes(counter[1])
    received_image_2 = base64.decodebytes(counter[2])

# return [msg_pack1, msg_pack2]

    # @app.get("/picture/")
    # async def predict_picture(image_link: str = ""):
    #     print("FUNCTION: predict_picture")
    # #     # https://www.wykop.pl/cdn/c3201142/comment_HxeHcT29IWfqTvHoy7LoNapYT7dxQZLc,w400.jpg
    #
    #     if image_link == "":
    #         return {"message": "Can't open the file."}
    #
    #     img_path = get_file("abc.jpg", origin=image_link)

    # img = load_img(img_path, target_size = (220, 154))
    # img = tf.image.resize_bilinear(img, [220, 154], align_corners=False)

    face1 = detect_faces(received_image_1)
    face2 = detect_faces(received_image_2)
    print(f"detected 1: {face1}")
    print(f"detected 2: {face2}")

    global mycol

    if face1 == 0 and face2 == 1:
        print("swap")
        received_image_temp = copy.deepcopy(received_image_1)
        received_image_1 = copy.deepcopy(received_image_2)
        received_image_2 = copy.deepcopy(received_image_temp)
    elif face1 != 1 or face2 != 0:
        print("error: faces_error")
        now = datetime.now(timezone.utc)
        mydict = {"time": now, "first": "faces_error", "second": [face1, face2], "image1": received_image_1, "image2": received_image_2}
        x = mycol.insert_one(mydict)

        return dict({"error": "faces_error"})

    print("dalej")




    #
    # image_bytes = io.BytesIO()
    # img.save(my_image, format="JPEG")

    # pil_img = tf.keras.preprocessing.image.array_to_img(img)
    # image = {}

    p1 = predict_doc_side(received_image_1)
    p2 = predict_doc_side(received_image_2)

    print(f"p1: {p1}")
    print(p1[0]["country_code"])
    print(f"p2: {p2}")
    print(p2[0]["country_code"])

    if p1[0]["country_code"] != p2[0]["country_code"]:
        print("Different docs")

        now = datetime.now(timezone.utc)
        mydict = {"time": now, "first": p1[0], "second": p2[0], "image1": received_image_1, "image2": received_image_2}

        x = mycol.insert_one(mydict)

        return dict({"error": "different_error"})

    now = datetime.now(timezone.utc)
    mydict = {"time": now, "first": p1[0], "second": p1[1], "image1": received_image_1, "image2": received_image_2}

    x = mycol.insert_one(mydict)

    return [p1[0], p1[1]]
    # return [local_msg_pack1, local_msg_pack2]


def detect_faces(received_image):
    jpg_as_np = np.frombuffer(received_image, dtype=np.uint8)
    imgjpg = cv2.imdecode(jpg_as_np, flags=1)
    # print(imgjpg.shape)
    doc_size = imgjpg.shape[0] * imgjpg.shape[1]
    # print(doc_size)

    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(imgjpg, cv2.COLOR_BGR2GRAY)
    min_face_size = int(max(imgjpg.shape[0], imgjpg.shape[1]) * 0.05)
    print(f"min face size {min_face_size}")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        # minSize=(30, 30),
        minSize= (min_face_size, min_face_size),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    print(f"Found {len(faces)} faces.")
    qualified_faces = 0

    for (x, y, w, h) in faces:
        print(x, y, w, h)
        face_size = w * h
        print(face_size)
        ratio = face_size / doc_size
        print(ratio)
        if ratio >= 0.04:
            qualified_faces += 1

    return qualified_faces



if __name__ == '__main__':
    print("main main main")
    port = int(os.environ.get('Port', 5000))

    # run(app, host="127.0.0.1", port=port)
    run(app, host="0.0.0.0", port=port, log_level="info")
