from app import app
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy(app)

class Prediction(db.Model):
    id = db.Column(db.Integer(), primary_key=True)
    name = db.Column(db.String(25))
    description = db.Column(db.Text)
    solution = db.Column(db.Text)