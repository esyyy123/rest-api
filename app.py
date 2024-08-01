from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import preprocessor


app = Flask(__name__)

feature_columns = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "Embarked_Q", "Embarked_S", "Sex_male"]
# Load the trained model
model = joblib.load('best_random_forest_model.pkl')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if isinstance(data, dict):
            df = pd.DataFrame([data])  # Wrap dict in a list to make it a single-row DataFrame
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return jsonify({"Error": "Invalid Input format"}), 400

        # Convert boolean-like strings to integers
        for col in ["Sex_male", "Embarked_Q", "Embarked_S"]:
            if df[col].dtype == 'object':  # Check if column has strings
                df[col] = df[col].map({"True": 1, "False": 0}).fillna(0).astype(int)

        # Reindex the DataFrame to ensure it has all the feature columns
        df = df.reindex(columns=feature_columns, fill_value=0)

        predictions = model.predict(df)
        predictions_list = [int(p) for p in predictions] 

        return jsonify({"Prediction": predictions_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    global model
    data = request.json
    df = pd.DataFrame(data)
    X = df[['Pclass', 'Age', 'Fare']]
    y = df['Survived']
    model = joblib.load('best_random_forest_model.pkl')
    model.fit(X, y)
    joblib.dump(model, 'best_random_forest_model.pkl')
    return jsonify({'message': 'Model trained successfully!'})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'API is running!'})

if __name__ == '__main__':
    app.run(debug=True)
