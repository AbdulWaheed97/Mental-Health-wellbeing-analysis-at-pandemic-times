import numpy as np
from flask import Flask, request, render_template
import pickle

# flask app
app = Flask(__name__)
# loading model
emotional_model = pickle.load(open('emotional_model.pkl', 'rb'))
behavioral_model = pickle.load(open('behavioral_model.pkl', 'rb'))

@app.route('/')
def homepage():
    return render_template('Homepage.html')

@app.route('/Output', methods = ['POST'])
def home():
    x = str(request.form['analysis'])
    if x == 'Behaviour':
        return render_template('Behaviour.html')
    elif x == "Emotion":
        return render_template('Emotion.html')
    else:
        return render_template('Homepage.html')

@app.route('/emotion_prediction' ,methods = ['POST'])
def emotion_prediction():
    final_features = [int(x) for x in request.form.values()]
    final_features = [np.array(final_features)]
    prediction = emotional_model.predict(final_features)

    return render_template('Emotion.html', output='Child Emotional Status is :  {}'.format(prediction[0]))

@app.route('/behaviour_prediction' ,methods = ['POST'])
def behaviour_prediction():
    final_features = [int(x) for x in request.form.values()]
    final_features = [np.array(final_features)]
    prediction = behavioral_model.predict(final_features)

    return render_template('Behaviour.html', output='Child Behavioural Status is :  {}'.format(prediction[0]))



if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
