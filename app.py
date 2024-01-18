import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the data
@st.cache
def load_data(file):
    data = pd.read_csv(file)
    return data

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Main content
st.title("CSV File Prediction App")

if uploaded_file is not None:
    data = load_data(uploaded_file)

    # Display the raw data
    st.subheader("Raw Data")
    st.write(data)

    # Sidebar for model training
    st.sidebar.subheader("Train Model")

    # Allow user to select attributes
    selected_attributes = st.sidebar.multiselect("Select attributes for training", data.columns)

    if len(selected_attributes) < 2:
        st.sidebar.warning("Please select at least two attributes.")
    else:
        target_column = st.sidebar.selectbox("Select target column", data.columns)
        test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)

        # Train-test split
        X = data[selected_attributes]
        y = data[target_column]

        # Handle categorical variables with one-hot encoding
        X = pd.get_dummies(X, columns=[col for col in selected_attributes if X[col].dtype == "O"])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Model training
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        # Model evaluation
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        st.sidebar.subheader("Model Evaluation")
        st.sidebar.text(f"Mean Squared Error: {mse}")

        # Prediction
        st.subheader("Make Predictions")

        # User input for predictions
        user_input = {}
        for attribute in selected_attributes:
            user_input[attribute] = st.text_input(f"Enter value for {attribute}")

        # Create a DataFrame from user input
        input_df = pd.DataFrame([user_input])

        # Handle categorical variables with one-hot encoding
        input_df = pd.get_dummies(input_df, columns=[col for col in selected_attributes if input_df[col].dtype == "O"])

        # Ensure feature names match those used during training
        missing_cols = set(X_train.columns) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0

        # Reorder columns to match training data
        input_df = input_df[X_train.columns]

        # Make predictions
        prediction = model.predict(input_df)

        st.subheader("Prediction")
        st.write(f"The predicted value is: {prediction[0]}")
