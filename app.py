import numpy as np 

from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app

app = Flask(__name__)

#Load the pickle model

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')

def Home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)


    if prediction ==  'Iris-versicolor':
        flower_img = 'images/irisversicolor.jpeg'
    elif prediction == 'Iris-setosa':
        flower_img = 'images/Irissetosa.jpeg'
    elif prediction == 'Iris-virginica':
        flower_img = 'images/irisvirginica.jpeg'



    return render_template('index.html', prediction_text = 'The flower species is: {}'.format(prediction), flower_img = flower_img)

if __name__ == '__main__':
    app.run(debug=True)
