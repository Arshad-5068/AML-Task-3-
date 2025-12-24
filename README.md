# Car Price Prediction 

**Predict car selling prices using a simple Linear Regression model and a Streamlit web app.**

---

## Project Overview
This repository contains a small end-to-end proof-of-concept that: 
- Loads a car dataset (`car data.csv`),
- Encodes categorical features with one-hot encoding,
- Trains a **Linear Regression** model on the full dataset,
- Exposes an interactive **Streamlit** app (`car.py`) for predicting a car's selling price using user inputs.


## Features
- Interactive UI to select car features (name, year, mileage, fuel, seller type, transmission, etc.)
- Model returns predicted price and shows model performance metrics (R², MAE, RMSE)
- Simple, easy-to-read code meant for learning and quick demos


## Requirements
- Python 3.8+
- streamlit
- pandas
- numpy
- scikit-learn

Suggested install (in a virtual environment):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install streamlit pandas numpy scikit-learn
```

(Optionally create a `requirements.txt` with the exact versions you want.)


## Running the App
1. Ensure `car data.csv` is in the project root (next to `car.py`).
2. Fix the data path in `car.py` if necessary (see Note below).
3. Start the Streamlit app:

```bash
streamlit run car.py
```


## Dataset
File: `car data.csv` (included in this repository)
Columns (sample):
- `Car_Name`, `Year`, `Selling_Price`, `Present_Price`, `Driven_kms`, `Fuel_Type`, `Selling_type`, `Transmission`, `Owner`

The app uses `pd.get_dummies(..., drop_first=True)` to encode categorical variables.


## Model & Metrics
- **Model:** sklearn.linear_model.LinearRegression
- **Training:** model is trained on the entire dataset 
- **Reported metrics:** R², MAE, RMSE 

## Results


