import os
import logging

from fastapi import FastAPI, Request  # File, UploadFile, Form, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run

import re
import base64

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

@app.post("/predict_test/")
async def predict_test():
    return [msg_pack1, msg_pack2]


@app.post("/predict/")
async def predict(
        request: Request
        # dataToUpload: List[str]
):

    da = await request.body()
    counter = re.split(b'data:image\/[A-Za-z]+;base64', da)
    print(counter[1][:500])
    print(counter[2][:500])

    my_image = base64.decodebytes(counter[1])

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

    img = tf.io.decode_image(my_image, channels=3, expand_animations=False)
    img = tf.image.resize(img, [220, 154])
    img_array = img_to_array(img)
    img_array = expand_dims(img_array, 0)

    global doc_model
    prediction = doc_model.predict(img_array)

    # pred_ind = np.argmax(prediction, axis=1)
    # score = tf.nn.softmax(prediction[0])

    list_of_predictions = np.array(prediction[0])
    strongest_predictions = np.flip(np.argsort(list_of_predictions)[-2:])
    countries = [class_names[country] for country in strongest_predictions]

    locale = Locale('pl')

    pred1 = 100 * list_of_predictions[strongest_predictions[0]]
    iso2 = coco.convert(names=countries[0][:3], to='ISO2')
    local_msg_pack1 = {
        "country_short": locale.territories[iso2.upper()],
        "country": coco.convert(names=countries[0][:3], to='name_official'),
        # "country_code": class_names[np.argmax(score)][:3],
        "country_code": countries[0][:3],
        "prediction": f"{pred1:.2f}%" if pred1 < 99.99 else 'Więcej niż 99,99%'
    }

    pred2 = 100 * list_of_predictions[strongest_predictions[1]]
    iso2 = coco.convert(names=countries[1][:3], to='ISO2')
    local_msg_pack2 = {
        "country_short": locale.territories[iso2.upper()],
        "country": coco.convert(names=countries[1][:3], to='name_official'),
        "country_code": countries[1][:3],
        "prediction": f"{pred2:.2f}%" if pred2 > 0.01 else 'Mniej niż 0.01%'
    }

    print(local_msg_pack1, local_msg_pack2)

    now = datetime.now(timezone.utc)
    mydict = {"time": now, "first": local_msg_pack1, "second": local_msg_pack2}
    global mycol
    x = mycol.insert_one(mydict)

    return [local_msg_pack1, local_msg_pack2]


if __name__ == '__main__':
    port = int(os.environ.get('Port', 5000))

    # run(app, host="127.0.0.1", port=port)
    run(app, host="0.0.0.0", port=port, log_level="info")
