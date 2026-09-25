import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="AI Price Guardian",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI PRICE GUARDIAN")
st.subheader("Your Budget. Your Location. Your Best Basket.")

st.write(
    "AI Price Guardian uses historical food-price data and "
    "machine learning to provide price prediction and price-increase risk."
)

st.divider()


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():
    linear_model = joblib.load("linear_price_model.joblib")
    logistic_model = joblib.load("logistic_risk_model.joblib")

    return linear_model, logistic_model


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    prices = pd.read_csv("latest_prices.csv")
    items = pd.read_csv("lookup_item.csv")
    premises = pd.read_csv("lookup_premise.csv")

    return prices, items, premises


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:
    linear_model, logistic_model = load_models()
    prices, items, premises = load_data()

except Exception as e:
    st.error("There was a problem loading the model or data files.")
    st.code(str(e))
    st.stop()


# ============================================================
# PREPARE DATA
# ============================================================

# Remove accidental spaces from column names
prices.columns = prices.columns.str.strip()
items.columns = items.columns.str.strip()
premises.columns = premises.columns.str.strip()


# Convert date
if "date" in prices.columns:
    prices["date"] = pd.to_datetime(
        prices["date"],
        errors="coerce"
    )
else:
    st.error("The price data does not contain a 'date' column.")
    st.stop()


# ------------------------------------------------------------
# Normalize codes
# This helps match values such as:
# 1.0 -> 1
# ------------------------------------------------------------

def normalize_code(series):
    return (
        series
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


if "item_code" in prices.columns:
    prices["item_code"] = normalize_code(prices["item_code"])

if "item_code" in items.columns:
    items["item_code"] = normalize_code(items["item_code"])


if "premise_code" in prices.columns:
    prices["premise_code"] = normalize_code(prices["premise_code"])

if "premise_code" in premises.columns:
    premises["premise_code"] = normalize_code(premises["premise_code"])


# ------------------------------------------------------------
# Remove existing lookup columns if they already exist
# ------------------------------------------------------------

columns_to_remove = [
    "item",
    "state",
    "district",
    "premise_type"
]

for column in columns_to_remove:
    if column in prices.columns:
        prices = prices.drop(columns=[column])


# ------------------------------------------------------------
# Prepare item lookup
# ------------------------------------------------------------

if "item_code" not in items.columns or "item" not in items.columns:
    st.error(
        "lookup_item.csv must contain both 'item_code' and 'item'."
    )

    st.write("Columns found in lookup_item.csv:")
    st.write(list(items.columns))

    st.stop()


item_lookup = items[
    ["item_code", "item"]
].drop_duplicates(
    subset=["item_code"]
)


# ------------------------------------------------------------
# Prepare premise lookup
# ------------------------------------------------------------

required_premise_columns = [
    "premise_code",
    "state",
    "district",
    "premise_type"
]

missing_premise_columns = [
    col
    for col in required_premise_columns
    if col not in premises.columns
]

if missing_premise_columns:
    st.error(
        "lookup_premise.csv is missing required columns."
    )

    st.write("Missing columns:")
    st.write(missing_premise_columns)

    st.write("Columns found in lookup_premise.csv:")
    st.write(list(premises.columns))

    st.stop()


premise_lookup = premises[
    required_premise_columns
].drop_duplicates(
    subset=["premise_code"]
)


# ------------------------------------------------------------
# Merge item information
# ------------------------------------------------------------

prices = prices.merge(
    item_lookup,
    on="item_code",
    how="left"
)


# ------------------------------------------------------------
# Merge premise information
# ------------------------------------------------------------

prices = prices.merge(
    premise_lookup,
    on="premise_code",
    how="left"
)


# ------------------------------------------------------------
# Check that lookup information was successfully added
# ------------------------------------------------------------

if "item" not in prices.columns:
    st.error(
        "The 'item' column could not be created after merging lookup_item.csv."
    )

    st.write("Price data columns:")
    st.write(list(prices.columns))

    st.stop()


if prices["item"].notna().sum() == 0:
    st.error(
        "No item names could be matched between latest_prices.csv "
        "and lookup_item.csv."
    )

    st.write("Example item codes in price data:")
    st.write(
        prices["item_code"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .head(20)
    )

    st.write("Example item codes in lookup:")
    st.write(
        item_lookup["item_code"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .head(20)
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🛒 Shopping Input")


# Product selection
product_list = sorted(
    prices["item"]
    .dropna()
    .astype(str)
    .unique()
)

if len(product_list) == 0:
    st.error("No food items are available.")
    st.stop()


selected_product = st.sidebar.selectbox(
    "Select food item",
    product_list
)


# District selection
district_list = sorted(
    prices["district"]
    .dropna()
    .astype(str)
    .unique()
)

if len(district_list) == 0:
    st.error("No districts are available.")
    st.stop()


selected_district = st.sidebar.selectbox(
    "Select district",
    district_list
)


# Budget
budget = st.sidebar.number_input(
    "Your budget (RM)",
    min_value=1.0,
    value=100.0,
    step=10.0
)


# Prediction button
predict_button = st.sidebar.button(
    "🔮 Predict Price",
    type="primary"
)


# ============================================================
# FILTER SELECTED PRODUCT + DISTRICT
# ============================================================

filtered = prices[
    (prices["item"].astype(str) == selected_product) &
    (prices["district"].astype(str) == selected_district)
].copy()


if len(filtered) == 0:
    st.warning(
        "No price records were found for this food item and district."
    )
    st.stop()


# ============================================================
# GET LATEST RECORD
# ============================================================

filtered = filtered.sort_values("date")

latest = filtered.iloc[-1]


# Current price
try:
    current_price = float(latest["price"])
except Exception:
    st.error("The latest price could not be read.")
    st.stop()


# ============================================================
# PREVIOUS PRICE
# ============================================================

if (
    "previous_price" in latest.index
    and pd.notna(latest["previous_price"])
):
    previous_price = float(latest["previous_price"])

else:
    previous_price = current_price


# ============================================================
# DATE INFORMATION
# ============================================================

date_value = latest["date"]

month = int(date_value.month)
year = int(date_value.year)


# ============================================================
# MODEL INPUT
# ============================================================

model_input = pd.DataFrame({
    "previous_price": [previous_price],
    "price": [current_price],
    "month": [month],
    "year": [year],
    "item": [latest["item"]],
    "state": [latest["state"]],
    "district": [latest["district"]],
    "premise_type": [latest["premise_type"]]
})


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # Linear Regression
        # Predict next observed price
        # ----------------------------------------------------

        predicted_price = linear_model.predict(
            model_input
        )[0]


        # ----------------------------------------------------
        # Logistic Regression
        # Predict probability of price increase
        # ----------------------------------------------------

        increase_probability = logistic_model.predict_proba(
            model_input
        )[0][1]


        increase_prediction = logistic_model.predict(
            model_input
        )[0]


        # Prevent negative predicted prices
        predicted_price = max(
            0,
            predicted_price
        )


        # ====================================================
        # AI PREDICTION
        # ====================================================

        st.header("🤖 AI Prediction")


        col1, col2, col3 = st.columns(3)


        with col1:
            st.metric(
                "Current Price",
                f"RM {current_price:.2f}"
            )


        with col2:
            st.metric(
                "Predicted Next Price",
                f"RM {predicted_price:.2f}"
            )


        with col3:
            st.metric(
                "Price Increase Risk",
                f"{increase_probability * 100:.1f}%"
            )


        st.divider()


        # ====================================================
        # WHAT THE AI SAYS
        # ====================================================

        st.subheader("🔎 What the AI says")


        if increase_prediction == 1:

            st.warning(
                f"The model estimates a "
                f"{increase_probability * 100:.1f}% probability "
                f"that the next observed price will be higher "
                f"than the current price."
            )

        else:

            st.success(
                f"The model estimates a "
                f"{increase_probability * 100:.1f}% probability "
                f"of a price increase in the next observation."
            )


        # ====================================================
        # BUDGET IMPACT
        # ====================================================

        current_quantity = int(
            budget // current_price
        )


        if predicted_price > 0:

            predicted_quantity = int(
                budget // predicted_price
            )

        else:

            predicted_quantity = 0


        st.subheader("💰 Budget Impact")


        b1, b2 = st.columns(2)


        with b1:

            st.metric(
                f"Units you can buy now with RM {budget:.0f}",
                current_quantity
            )


        with b2:

            st.metric(
                f"Units at predicted price",
                predicted_quantity
            )


        # ====================================================
        # PRICE COMPARISON
        # ====================================================

        st.subheader(
            "🏪 Price Comparison in Selected District"
        )


        comparison_columns = [
            "premise_code",
            "premise_type",
            "price",
            "date"
        ]


        comparison = filtered[
            comparison_columns
        ].copy()


        comparison = comparison.sort_values(
            "price"
        )


        comparison = comparison.drop_duplicates(
            "premise_code"
        )


        comparison = comparison.head(10)


        comparison["price"] = comparison[
            "price"
        ].round(2)


        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # AI SHOPPING INSIGHT
        # ====================================================

        st.subheader("💡 AI Shopping Insight")


        difference = (
            predicted_price -
            current_price
        )


        if difference > 0:

            st.write(
                f"📈 The predicted next price is approximately "
                f"RM {difference:.2f} higher than the current price."
            )

        elif difference < 0:

            st.write(
                f"📉 The predicted next price is approximately "
                f"RM {abs(difference):.2f} lower than the current price."
            )

        else:

            st.write(
                "➡️ The predicted next price is approximately "
                "the same as the current price."
            )


        st.write(
            "Use this prediction as decision-support information. "
            "The final purchasing decision remains with the consumer."
        )


    except Exception as e:

        st.error(
            "The model could not make a prediction."
        )

        st.code(str(e))

        st.write("Model input used for prediction:")

        st.dataframe(
            model_input,
            use_container_width=True
        )
