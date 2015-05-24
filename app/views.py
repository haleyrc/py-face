from flask import render_template, make_response, send_file, abort
from app import app
import base64
import os

@app.route('/')
@app.route('/index')
def index():
  return render_template( "index.html" )

@app.route('/cam')
@app.route('/cam/<type>/')
def cam(type="base"):
  filename = "app/static/img/" + type + ".png"
  with open( filename, "rb" ) as f:
    img = f.read()
  encoded_img = base64.b64encode( img )
  response = make_response( encoded_img )
  response.headers['Content-Type'] = 'text'
  # response = make_response( img )
  # response.headers['Content-Type'] = 'image/png'
  return response
