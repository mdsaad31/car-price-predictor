from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)


with open('enhanced_car_price_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    features = model_data['features']
    relevant_brands = model_data['relevant_brands']
    current_year = model_data['current_year']

@app.route('/')
def home():
    df = pd.read_csv('quikr_car.csv')
    
    all_brands = [c for c in df['company'].unique() if isinstance(c, str)]
    all_brands = sorted(list(set(all_brands)))
  

    fuel_types = [f for f in df['fuel_type'].unique() if isinstance(f, str) and pd.notna(f)]
    
    return render_template('index.html', 
                         companies=all_brands,
                         fuel_types=fuel_types,
                         current_year=current_year,
                         relevant_brands=relevant_brands)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            'company': request.form['company'],
            'name': request.form['name'],
            'year': int(request.form['year']),
            'kms_driven': int(request.form['kms_driven']),
            'fuel_type': request.form['fuel_type']
        }
        
        if input_data['company'] not in relevant_brands:
            available_brands = ", ".join(relevant_brands)
            raise ValueError(f"Sorry, we currently only support these brands: {available_brands}")
            
        if not (1990 <= input_data['year'] <= current_year):
            raise ValueError(f"Year must be between 1990 and {current_year}")
            
        if input_data['kms_driven'] < 0:
            raise ValueError("Kilometers driven cannot be negative")
        
        input_data['age'] = current_year - input_data['year']
        
        input_df = pd.DataFrame([{
            'company': input_data['company'],
            'name': input_data['name'],
            'year': input_data['year'],
            'kms_driven': input_data['kms_driven'],
            'fuel_type': input_data['fuel_type'],
            'age': input_data['age']
        }])
        
        log_price = model.predict(input_df)[0]
        predicted_price = np.exp(log_price)
        
        formatted_price = "₹{:,.2f}".format(predicted_price)
        
        return jsonify({
            'success': True,
            'predicted_price': formatted_price
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/get_models')
def get_models():
    try:
        company = request.args.get('company')
        if not company:
            return jsonify({'error': 'No company specified'})
            
        df = pd.read_csv('quikr_car.csv')
        
        models = df[df['company'] == company]['name'].unique()
        cleaned_models = []
        
        for model_name in models:
            if isinstance(model_name, str):
                parts = model_name.split()[:3]
                cleaned_model = ' '.join(parts)
                if cleaned_model not in cleaned_models:
                    cleaned_models.append(cleaned_model)
        
        cleaned_models.sort()
        return jsonify(cleaned_models)
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)