# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
# loading location_pivot to enable model to take str inputs
location_pivot=pd.read_csv("delhi_location_pivot.csv")
location_pivot=location_pivot.set_index("Location")
locations=location_pivot.index

# Load the Random Forest CLassifier model
filename = 'delhi_house_price-prediction-LR-model.pkl'
model = pickle.load(open(filename, 'rb'))


@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        area= int(request.form['area'])
        BHK = int(request.form['BHK'])
        resale = int(request.form['resale'])
        location= str(request.form['location'])
        
        if location in locations:
            loc_index = np.where(locations==location)[0][0]
            x=np.zeros(4)
            x[0]=area
            x[1]=BHK
            x[2]=resale
            x[3]=np.log(location_pivot.iloc[loc_index,1])
            
        
            my_prediction = np.exp(model.predict([x])[0])
        
            return render_template('result.html', prediction=my_prediction)
    
        else:
            return render_template('result.html', prediction=0)
if __name__ == '__main__':
	app.run(debug=True)
    
    