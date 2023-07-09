import os
import numpy as np
import tensorflow as tf
import cv2
import gdown

from flask import jsonify, request
from app import app
from .models import Prediction, db

model_path = 'model2.h5'

if not os.path.exists(model_path):
    file_id = '1U7CWNgf9V9WPGvjWdqoTbmHWJTu7uepl'
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output=model_path)
    print('file model berhasil didownload')

model = tf.keras.models.load_model('model2.h5')

@app.route('/')
def index():
    return 'Welcome to Eye Disease Detection 2'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        img = request.files["image"]
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
        img.save(img_path)

        img_size = (224, 224)
        img_test = cv2.imread(img_path)
        img_test = cv2.resize(img_test, img_size)
        img_test = np.expand_dims(img_test, axis=0)

        prediction = model.predict(img_test)
        prediction = (prediction > 0.5).astype(int)

        if prediction == 0:
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