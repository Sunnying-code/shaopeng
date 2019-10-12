# coding=utf-8

import os
import tensorflow as tf
import sys

# project_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(project_path)
# from src.predict import Predict, getTestPicArray

from web.modules.mnist.predict import Predict, getTestPicArray
from web.config import test_images_dir
from web.modules.cassandra_handle import MnistDeepImage
from web.modules.mnist.test import predict

predict_handle = Predict()
cassandra_handle = MnistDeepImage()



def deep_predict(image_path):
    # nm = getTestPicArray(image_path)
    # result = predict_handle.test(nm)
    result = predict(image_path)
    print(result)
    return result[0]

def insert_data(req_time, file_name, mnist_result):

    cassandra_handle.insert_data(str(req_time), file_name, mnist_result)
    return True


if __name__ == '__main__':
    image_path = os.path.join(test_images_dir, "6.jpg")
    deep_predict(image_path)
