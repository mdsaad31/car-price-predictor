# 🚗 Car Price Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-green?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-1.0%2B-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.1%2B-purple?logo=bootstrap&logoColor=white)](https://getbootstrap.com)
[![License](https://img.shields.io/badge/License-MIT-brightgreen)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/mdsaad31/car-price-predictor?style=social)](https://github.com/mdsaad31/car-price-predictor)

## 🌟 Features

<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 1rem;">
    <h4>🚀 Accurate Predictions</h4>
    <h4>📊 30+ Brands</h4>
    <h4>💻 Interactive UI</h4>
    <h4>📱 Responsive</h4>
    <h4>⚡ Real-time</h4>
    <h4>🔍 Detailed Analysis</h4>
</div>


## 🛠️ Tech Stack

### Backend
![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/-Flask-000000?logo=flask&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/-Scikit_Learn-F7931E?logo=scikit-learn&logoColor=white)

### Frontend
![HTML5](https://img.shields.io/badge/-HTML5-E34F26?logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/-CSS3-1572B6?logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/-JavaScript-F7DF1E?logo=javascript&logoColor=black)
![Bootstrap](https://img.shields.io/badge/-Bootstrap-7952B3?logo=bootstrap&logoColor=white)

### ML Models
![Random Forest](https://img.shields.io/badge/-Random_Forest-00A86B)
![Gradient Boosting](https://img.shields.io/badge/-Gradient_Boosting-FF6F00)

## 🚀 Quick Start

### Clone repository
```
git clone https://github.com/yourusername/car-price-predictor.git
cd car-price-predictor
```

### Create virtual environment
```
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### Install dependencies
```
pip install -r requirements.txt
```

### Run application
```
python app.py
```
Access at: http://localhost:5000

## 📊 Model Performance
| Metric |	Score |
|-------|-------|
| R² Score | 0.92 |
| Mean Error	| ₹12,500 |
| Training Time |	45 seconds |
|Best Model |	Random Forest (n_estimators=200) |

## 🌐 API Endpoints
Python:
```
import requests

data = {
    "company": "Hyundai",
    "name": "Grand i10",
    "year": 2018,
    "kms_driven": 35000,
    "fuel_type": "Petrol"
}

response = requests.post("http://localhost:5000/predict", data=data)
print(response.json())```
```
Javascript:
```
fetch('/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/x-www-form-urlencoded',
  },
  body: new URLSearchParams({
    company: 'Hyundai',
    name: 'Grand i10',
    year: 2018,
    kms_driven: 35000,
    fuel_type: 'Petrol'
  })
})
.then(response => response.json())
.then(data => console.log(data));
```
### Response
Successful Response:
```
{
  "success": true,
  "predicted_price": "₹425,000",
  "confidence": 0.92,
  "model": "Random Forest"
}
```

Error Response:
```
{
  "success": false,
  "error": "Brand must be one of: Maruti, Hyundai, Honda, ..."
}
```

### Error Handling
Common error responses include:

- 400 Bad Request - Missing required parameters
- 422 Unprocessable Entity - Invalid input values
- 500 Internal Server Error - Server-side issues

Always check the success flag in the response before processing results.

## 🚙 Supported Brands
<div style="display: flex; flex-wrap: wrap; gap: 0.5rem;"> <img src="https://img.shields.io/badge/-Maruti-0072BB?logo=maruti-suzuki&logoColor=white" alt="Maruti"> <img src="https://img.shields.io/badge/-Hyundai-002C5F?logo=hyundai&logoColor=white" alt="Hyundai"> <img src="https://img.shields.io/badge/-Toyota-EB0A1E?logo=toyota&logoColor=white" alt="Toyota"> <img src="https://img.shields.io/badge/-Honda-EB0A1E?logo=honda&logoColor=white" alt="Honda"> <img src="https://img.shields.io/badge/-Tata-3D4D99?logo=tata&logoColor=white" alt="Tata"> <img src="https://img.shields.io/badge/-Mahindra-6A0DAD?logo=mahindra&logoColor=white" alt="Mahindra"> <img src="https://img.shields.io/badge/-BMW-0066B1?logo=bmw&logoColor=white" alt="BMW"> <img src="https://img.shields.io/badge/-Volkswagen-151F5D?logo=volkswagen&logoColor=white" alt="Volkswagen"> </div>
And more ...

## 🤝 How to Contribute
1. Fork the repository
2. Create a feature branch (git checkout -b feature/AmazingFeature)
3. Commit your changes (git commit -m 'Add amazing feature')
4. Push to the branch (git push origin feature/AmazingFeature)
5. Open a Pull Request
