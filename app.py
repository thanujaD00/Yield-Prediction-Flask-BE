from flask import Flask, request, jsonify
import pandas as pd
from datetime import datetime
import os
import json
from utils.model_utils import ensemble_predict, load_seasonal_factors
from utils.data_utils import analyze_seasonal_patterns, create_seasonality_plot

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Get prediction year
        prediction_year = data.get('year', datetime.now().year)
        
        # Get monthly data
        monthly_data = data.get('monthly_data', [])
        
        if not monthly_data:
            return jsonify({
                'status': 'error',
                'message': 'No monthly data provided'
            }), 400
        
        # Make predictions for each month
        all_predictions = []
        for month_data in monthly_data:
            try:
                # Extract month
                month = month_data.get('month')
                if not month or not (1 <= month <= 12):
                    return jsonify({
                        'status': 'error',
                        'message': f'Invalid month: {month}'
                    }), 400
                
                # Extract soil and weather data
                soil_data = {
                    'sm_10': month_data.get('sm_10'),
                    'sm_20': month_data.get('sm_20'),
                    'sm_30': month_data.get('sm_30'),
                    'age': month_data.get('age'),
                    'soil_type': month_data.get('soil_type')
                }
                
                weather_data = {
                    'Temperature (°C)': month_data.get('Temperature (°C)'),
                    'Humidity (%)': month_data.get('Humidity (%)'),
                    'Rainfall (mm)': month_data.get('Rainfall (mm)'),
                    'Weather Description': month_data.get('Weather Description', 'normal')
                }

                # Validate input data
                for key, value in {**soil_data, **weather_data}.items():
                    if value is None:
                        return jsonify({
                            'status': 'error',
                            'message': f'Missing parameter for month {month}: {key}'
                        }), 400
                
                # Create prediction date
                prediction_date = pd.Timestamp(year=prediction_year, month=month, day=15)
                
                # Make prediction
                prediction = ensemble_predict(soil_data, weather_data, prediction_date=prediction_date)
                if prediction:
                    all_predictions.append(prediction)
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'message': f'Error processing month {month}: {str(e)}'
                }), 400

        if all_predictions:
            return jsonify({
                'status': 'success',
                'year': prediction_year,
                'monthly_predictions': all_predictions,
                'average_prediction': round(
                    sum(p['ensemble_prediction'] for p in all_predictions) / len(all_predictions), 
                    2
                )
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to generate predictions'
            }), 500
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/analyze_seasonality', methods=['GET'])
def analyze_seasonality():
    try:
        # Path to your historical data
        historical_data_path = 'data/processed_coconut_data.csv'
        
        # Check if the file exists
        if not os.path.exists(historical_data_path):
            return jsonify({
                'status': 'error',
                'message': 'Historical data file not found'
            }), 404
            
        # Perform the analysis
        analysis = analyze_seasonal_patterns(historical_data_path)
        
        if analysis:
            return jsonify({
                'status': 'success',
                'analysis': analysis
            })
            
        return jsonify({
            'status': 'error',
            'message': 'Failed to analyze seasonal patterns'
        }), 500
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/visualize_seasonality', methods=['GET'])
def visualize_seasonality():
    try:
        # Path to your historical data
        historical_data_path = 'data/processed_coconut_data.csv'
        
        # Check if the file exists
        if not os.path.exists(historical_data_path):
            return jsonify({
                'status': 'error',
                'message': 'Historical data file not found'
            }), 404
            
        # Create the visualization
        image = create_seasonality_plot(historical_data_path)
        
        if image:
            return jsonify({
                'status': 'success',
                'image': image
            })
            
        return jsonify({
            'status': 'error',
            'message': 'Failed to create visualization'
        }), 500
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/seasonal_factors', methods=['GET'])
def get_seasonal_factors():
    try:
        seasonal_factors = load_seasonal_factors()
        return jsonify({
            'status': 'success',
            'seasonal_factors': seasonal_factors
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Ensure required directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Create example data file if it doesn't exist
    historical_data_path = 'data/processed_coconut_data.csv'
    if not os.path.exists(historical_data_path):
        # Create a sample file based on your provided data
        sample_data = """Date,Soil Moisture (10 cm) (%),Soil Moisture (20 cm) (%),Soil Moisture (30 cm) (%),Plant Age (years),Temperature (°C),Humidity (%),Rainfall (mm),Rain Status (0/1),Soil Type,Soil Type (Numeric),Coconut Count
1930-05-31,25.233333333333334,31.333333333333332,41.9,5,27.266666666666666,67.43333333333334,5.025,1,Red Yellow Podzolic,4,511.0
1930-06-30,25.233333333333334,31.333333333333332,41.9,4,27.266666666666666,67.43333333333334,5.025,1,Red Yellow Podzolic,4,511.0
1930-07-31,30.433333333333337,31.433333333333334,46.166666666666664,5,28.133333333333336,65.86666666666667,2.6958333333333333,0,Red Yellow Podzolic,4,483.0"""
        
        with open(historical_data_path, 'w') as f:
            f.write(sample_data)
        
        print(f"Created sample historical data file at {historical_data_path}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)