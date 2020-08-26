from flask import Flask
import os
import sys
from flask_cors import CORS

app = Flask(__name__)
app.secret_key = os.urandom(24)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

import views

if __name__ == "__main__":
    app.run(debug=True)

# os.system("clear-cache.sh")