import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle
import os
from datetime import datetime

class EnhancedCarPricePredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.features = ['company', 'name', 'year', 'kms_driven', 'fuel_type']
        self.target = 'Price'
        self.current_year = datetime.now().year
        self.relevant_brands = [
            'Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford', 
            'Volkswagen', 'Tata', 'Mahindra', 'Renault', 'Skoda',
            'BMW', 'Mercedes', 'Audi', 'Nissan', 'Chevrolet',
            'Kia', 'Volvo', 'Jeep', 'MG', 'Fiat'
        ]
        
    def load_data(self, filepath):
        """Load and preprocess the data with enhanced cleaning"""
        car = pd.read_csv(filepath)
        
        car = car[car['year'].str.isnumeric()]
        car['year'] = car['year'].astype(int)
        
        car = car[car['Price'] != 'Ask For Price']
        car['Price'] = car['Price'].str.replace(',','').astype(int)
        
        car['kms_driven'] = car['kms_driven'].str.split().str.get(0).str.replace(',','')
        car = car[car['kms_driven'].str.isnumeric()]
        car['kms_driven'] = car['kms_driven'].astype(int)
        
        car = car[~car['fuel_type'].isna()]
        
        car['name'] = car['name'].str.split().str.slice(start=0,stop=3).str.join(' ')
        
        car = car[car['company'].isin(self.relevant_brands)]
        
        Q1 = car['Price'].quantile(0.25)
        Q3 = car['Price'].quantile(0.75)
        IQR = Q3 - Q1
        car = car[~((car['Price'] < (Q1 - 1.5 * IQR)) | (car['Price'] > (Q3 + 1.5 * IQR)))]
        
        car['age'] = self.current_year - car['year']
        
        return car.reset_index(drop=True)
    
    def preprocess_data(self, df):
        """Prepare data for modeling with feature engineering"""
        X = df[self.features + ['age']]
        y = np.log(df[self.target])
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def build_preprocessor(self):
        """Create enhanced preprocessing pipeline"""
        categorical_features = ['company', 'name', 'fuel_type']
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        numeric_features = ['year', 'kms_driven', 'age']
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        return self.preprocessor
    
    def train_models(self, X_train, y_train):
        """Train and compare multiple models with hyperparameter tuning"""
        models = {
            'Random Forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'regressor__n_estimators': [100, 200],
                    'regressor__max_depth': [None, 10, 20]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'regressor__n_estimators': [100, 200],
                    'regressor__learning_rate': [0.05, 0.1]
                }
            },
            'Lasso': {
                'model': Lasso(),
                'params': {
                    'regressor__alpha': [0.1, 1.0]
                }
            }
        }
        
        best_score = -np.inf
        best_model = None
        
        for name, config in models.items():
            pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('regressor', config['model'])
            ])
            
            grid_search = GridSearchCV(
                pipeline,
                param_grid=config['params'],
                cv=5,
                scoring='r2',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            score = grid_search.best_score_
            
            print(f"{name}: Best R2 = {score:.4f}")
            print(f"Best params: {grid_search.best_params_}")
            
            if score > best_score:
                best_score = score
                best_model = grid_search.best_estimator_
                best_model_name = name
                
        print(f"\nBest model: {best_model_name} with R2 = {best_score:.4f}")
        self.model = best_model
        return best_model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model on test data with detailed metrics"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Test MSE: {mse:.4f}")
        print(f"Test R2: {r2:.4f}")
        
        return mse, r2
    
    def save_model(self, filename):
        """Save the trained model to disk"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'preprocessor': self.preprocessor,
                'features': self.features + ['age'],
                'target': self.target,
                'relevant_brands': self.relevant_brands,
                'current_year': self.current_year
            }, f)
    
    def load_model(self, filename):
        """Load a trained model from disk"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.preprocessor = data['preprocessor']
            self.features = data['features']
            self.target = data['target']
            self.relevant_brands = data['relevant_brands']
            self.current_year = data['current_year']
    
    def predict_price(self, input_data):
        """Predict price for new data with input validation"""
        if self.model is None:
            raise ValueError("Model has not been trained or loaded")
            
        if input_data['company'] not in self.relevant_brands:
            raise ValueError(f"Brand must be one of: {', '.join(self.relevant_brands)}")
            
        if not (1990 <= input_data['year'] <= self.current_year):
            raise ValueError(f"Year must be between 1990 and {self.current_year}")
            
        if input_data['kms_driven'] < 0:
            raise ValueError("Kilometers driven cannot be negative")
            
        input_data['age'] = self.current_year - input_data['year']
        
        input_df = pd.DataFrame([{
            'company': input_data['company'],
            'name': input_data['name'],
            'year': input_data['year'],
            'kms_driven': input_data['kms_driven'],
            'fuel_type': input_data['fuel_type'],
            'age': input_data['age']
        }])
        
        log_price = self.model.predict(input_df)[0]
        return np.exp(log_price)

if __name__ == "__main__":
    predictor = EnhancedCarPricePredictor()
    
    df = predictor.load_data('quikr_car.csv')
    print(f"Data loaded with {len(df)} records after cleaning")
    
    X_train, X_test, y_train, y_test = predictor.preprocess_data(df)
    
    predictor.build_preprocessor()
    
    predictor.train_models(X_train, y_train)
    
    predictor.evaluate_model(X_test, y_test)
    
    predictor.save_model('enhanced_car_price_model.pkl')
    
    sample_input = {
        'company': 'Hyundai',
        'name': 'Hyundai Grand i10',
        'year': 2018,
        'kms_driven': 35000,
        'fuel_type': 'Petrol'
    }
    try:
        predicted_price = predictor.predict_price(sample_input)
        print(f"\nPredicted price: Rs.{predicted_price:,.2f}")
    except ValueError as e:
        print(f"Error: {str(e)}")