import os
import numpy as np
import tensorflow as tf
import keras.utils as image

from flask import jsonify, request
from app import app
from .models import Prediction, db

model = tf.keras.models.load_model('model.h5')

@app.route('/')
def index():
    return 'Welcome to Eye Disease Detection'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        img = request.files["image"]
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
        img.save(img_path)

        img_size = (150, 150)
        img_test = image.load_img(img_path, target_size = img_size)
        img_test = image.img_to_array(img_test)
        img_test = np.expand_dims(img_test, axis=0)

        prediction = model.predict(img_test)

        if prediction == 1:
            name = 'Normal'
            description = 'Mata normal merupakan keadaan dimana mata dalam kondisi baik, semua bagian bagian pada mata berfungsi normal.'
            solution = 'Selalu jaga kesehatan mata dengan mengonsumsi makanan sehat yang kaya vitamin A.'
        else:
            name = 'Katarak'
            description = 'Katarak adalah proses degeneratif berupa kekeruhan di lensa bola mata sehingga menyebabkan menurunnya kemampuan penglihatan sampai kebutaan.'
            solution = 'Hubungi pelayanan medis atau rumah sakit untuk pengecekan lebih lanjut.'

        eye_disease = Prediction(name=name, description=description, solution=solution)
        db.session.add(eye_disease)
        db.session.commit()

        return jsonify({
            'status': 'success',
            'message': 'Prediction success',
            'data': {
                'name': name,
                'description': description,
                'solution': solution
            }
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': 'Prediction failed',
            'error': str(e)
        })


@app.route('/disease/<disease_id>', methods=['GET'])
def disease(disease_id):
    try:
        eye_disease = Prediction.query.get(disease_id)
        
        return jsonify({
            'status': 'success',
            'data': {
                'id': eye_disease.id,
                'name': eye_disease.name,
                'description': eye_disease.description,
                'solution': eye_disease.solution
            }
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        })