import os

from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for, jsonify)

app = Flask(__name__)


@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/grammar-check', methods=['POST'])
def hello():
   text = request.get_json()

   if text:
       return jsonify({'data': text })
   else:
       return jsonify({'message': 'please send some text'})


if __name__ == '__main__':
   app.run()
