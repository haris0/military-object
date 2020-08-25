from flask import Flask
import os
import sys

app = Flask(__name__)
app.secret_key = os.urandom(24)

import views

if __name__ == "__main__":
    app.run(debug=True)

# os.system("clear-cache.sh")