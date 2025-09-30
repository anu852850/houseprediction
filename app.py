from flask import Flask, request, render_template
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Collect input data from form
            data = CustomData(
                area_sqft=request.form.get('area_sqft'),
                bedrooms=request.form.get('bedrooms'),
                bathrooms=request.form.get('bathrooms'),
                location=request.form.get('location'),
                year_built=request.form.get('year_built'),
                has_garage=request.form.get('has_garage'),
            )

            # Convert into DataFrame
            pred_df = data.get_data_as_df()

            # Predict
            pipeline = PredictPipeline()
            results = pipeline.predict(pred_df)

            return render_template('home.html', results=results[0])

        except Exception as e:
            return render_template('home.html', error=str(e))

    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)
