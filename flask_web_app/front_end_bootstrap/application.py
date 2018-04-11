import json
import logging

from flask_cors import CORS
from flask import Flask, render_template, request, url_for
import os
from logo_brewer_app import *

PEOPLE_FOLDER = os.path.join('static', 'people_photo')

app = Flask(__name__)

cors = CORS(app, resources={r"/*": {"origins": "*"}})
# app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

# @app.route('/')
# @app.route('/index')
# def show_index():
#     # full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'shovon.jpg')
#     return render_template("index.html")

def patch_broken_pipe_error():
    """Monkey Patch BaseServer.handle_error to not write
    a stacktrace to stderr on broken pipe.
    http://stackoverflow.com/a/22618740/362702"""
    import sys
    from SocketServer import BaseServer
    from wsgiref import handlers

    handle_error = BaseServer.handle_error
    log_exception = handlers.BaseHandler.log_exception

    def is_broken_pipe_error():
        type, err, tb = sys.exc_info()
        return repr(err) == "error(32, 'Broken pipe')"

    def my_handle_error(self, request, client_address):
        if not is_broken_pipe_error():
            handle_error(self, request, client_address)

    def my_log_exception(self, exc_info):
        if not is_broken_pipe_error():
            log_exception(self, exc_info)

    BaseServer.handle_error = my_handle_error
    handlers.BaseHandler.log_exception = my_log_exception


patch_broken_pipe_error()

@app.route('/random', methods=['GET','POST'])
def show_index():
    # full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'shovon.jpg')
    generator = gimme_something()
    show_images(generator.predict(make_latent_samples(25, 100)))
    return "Success"


if __name__ == '__main__':
   app.run()