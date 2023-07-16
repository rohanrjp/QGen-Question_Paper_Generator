from flask import Flask
from flask_pymongo import PyMongo

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/myDatabase"
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
db = PyMongo(app).db

from questionpapergenerator import routes