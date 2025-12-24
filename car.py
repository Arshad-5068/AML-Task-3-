import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Car Price Prediction", layout="wide")

st.markdown("""
<style>
body { background-color:#0e1117; color:white; }

.card {
    background:#fff6e5;
    padding:40px;
    border-radius:18px;
    text-align:center;
}

.card h2 {
    color: black;
}

.price {
    font-size:42px;
    font-weight:bold;
    color:#8b0000;
}

.sub {
    color:#666;
}
</style>
""", unsafe_allow_html=True)


data_raw = pd.read_csv(r"E:\Arshad\Almimin labs\Task3\car data.csv")


car_names = sorted(data_raw["Car_Name"].dropna().unique())

data = pd.get_dummies(data_raw, drop_first=True)

X = data.drop("Selling_Price", axis=1)
y = data["Selling_Price"]

model = LinearRegression()
model.fit(X, y)

y_pred_all = model.predict(X)

mae = mean_absolute_error(y, y_pred_all)
rmse = np.sqrt(mean_squared_error(y, y_pred_all))
r2 = r2_score(y, y_pred_all)

st.sidebar.title("Input Car Features")

car_name = st.sidebar.selectbox("Car Name ", car_names)

year = st.sidebar.slider("Year of Purchase", 2005, 2024, 2019)
present_price = st.sidebar.number_input("Present Price (Lakhs)", 1.0, 50.0, 2.0)
kms = st.sidebar.number_input("Kilometers Driven", 500, 300000, 60000)
owner = st.sidebar.selectbox("Previous Owners", [0, 1, 2])

fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel"])
seller_type = st.sidebar.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])

user_input = pd.DataFrame(0, index=[0], columns=X.columns)


if "Year" in user_input.columns:
    user_input["Year"] = year

if "Present_Price" in user_input.columns:
    user_input["Present_Price"] = present_price

for col in user_input.columns:
    if col.lower().replace("_", "") == "kmsdriven":
        user_input[col] = kms

if "Owner" in user_input.columns:
    user_input["Owner"] = owner


if fuel_type == "Diesel":
    if "Fuel_Type_Diesel" in user_input.columns:
        user_input["Fuel_Type_Diesel"] = 1

if seller_type == "Individual":
    if "Seller_Type_Individual" in user_input.columns:
        user_input["Seller_Type_Individual"] = 1

if transmission == "Manual":
    if "Transmission_Manual" in user_input.columns:
        user_input["Transmission_Manual"] = 1


car_col = f"Car_Name_{car_name}"
if car_col in user_input.columns:
    user_input[car_col] = 1

predicted_price = model.predict(user_input)[0]

st.markdown("<h1 style='text-align:center;'>Car Price Prediction</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown(f"""
    <div class="card">
        <h2>Predicted Car Price</h2>
        <div class="price">₹ {predicted_price*10000:,.2f}</div>
        <p class="sub">Car Name: {car_name}</p>
        <p class="sub">Car Age: {2024 - year} years</p>
        <p class="sub">Efficiency Score: {round(kms / (2024 - year + 1) / 1000, 2)} km/year</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("##  Model Performance")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("R² Score", f"{r2:.2f}")

with m2:
    st.metric("MAE (Lakhs)", f"{mae:.2f}")

with m3:
    st.metric("RMSE (Lakhs)", f"{rmse:.2f}")
