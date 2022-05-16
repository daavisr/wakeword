from flask import Flask, render_template
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)

@app.route('/')
def main():
    return render_template('index.html')
    
@app.route('/test')
def test():
    return "Online"