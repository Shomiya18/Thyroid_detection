from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for the home page
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Collect data from form
            data = CustomData(
                age=int(request.form.get('age')),
                sex=request.form.get('sex'),
                sick=request.form.get('sick') == 'true',          # Convert 'true'/'false' strings to boolean
                pregnant=request.form.get('pregnant') == 'true',  # Convert 'true'/'false' strings to boolean
                thyroid=request.form.get('thyroid') == 'true',    # Convert 'true'/'false' strings to boolean
                surgery=request.form.get('surgery') == 'true',    # Convert 'true'/'false' strings to boolean
                I131=request.form.get('I131') == 'true',          # Convert 'true'/'false' strings to boolean
                treatment=request.form.get('treatment') == 'true',# Convert 'true'/'false' strings to boolean
                lithium=request.form.get('lithium') == 'true',    # Convert 'true'/'false' strings to boolean
                goitre=request.form.get('goitre') == 'true',      # Convert 'true'/'false' strings to boolean
                tumor=request.form.get('tumor') == 'true',        # Convert 'true'/'false' strings to boolean
                TSH=int(request.form.get('TSH')),
                T3=int(request.form.get('T3')),
                TT4=int(request.form.get('TT4')),
                T4U=int(request.form.get('T4U')),
                FTI=int(request.form.get('FTI'))
            )

            # Convert data to DataFrame
            pred_df = data.get_data_as_data_frame()

            # Create PredictPipeline instance
            predict_pipeline = PredictPipeline()

            # Make prediction
            result = predict_pipeline.predict(pred_df)[0]

            # Convert result to True/False
            boolean_result = result == 1.0

            # Render result
            return render_template('home.html', results=boolean_result)

        except Exception as e:
            # Handle exceptions and show error message
            return render_template('home.html', error=str(e))

    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8001", debug=True)
