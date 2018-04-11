from flask import Flask, render_template, request, url_for
import os

PEOPLE_FOLDER = os.path.join('static', 'people_photo')

app = Flask(__name__, template_folder='frond_end_bootstrap')
# app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

@app.route('/')
@app.route('/index')
def show_index():
    # full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'shovon.jpg')
    return render_template("frond_end_bootstrap/index.html")


if __name__ == '__main__':
   app.run()