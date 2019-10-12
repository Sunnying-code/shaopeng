# coding=utf-8

import os
import datetime
from flask import Flask
from flask import request
from flask import render_template
from werkzeug.utils import secure_filename

# from gevent import monkey
# monkey.patch_all()
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# from gevent import wsgi
project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
import sys
sys.path.append(project_path)

from web.module import deep_predict, insert_data
from web.config import upload_dir

app = Flask(__name__)


@app.route('/')
def hello_world():
    # return 'Hello, World!'
    return render_template('index.html')


@app.route('/mnist', methods=["POST"])
def mnist():
    req_time = datetime.datetime.now()
    file_content = request.files["file"]
    file_name = secure_filename(file_content.filename)
    suffix = os.path.splitext(file_name)[-1][1:]
    suffix_list = ["bmp", "jpg", "jpeg", "gif", "png"]
    if suffix not in suffix_list:
        return "noly support type [{0}]".format(";".join(suffix_list))
    file_path = os.path.join(upload_dir, file_name)
    file_content.save(file_path)

    mnist_result = deep_predict(file_path)
    print(mnist_result)
    # return "hello"

    insert_data(req_time, file_name, mnist_result)
    playload = "result: {0}\n ".format(mnist_result)
    return playload


def mnist_deep_handle():

    pass


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)