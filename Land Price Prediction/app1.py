import streamlit as st
import pandas as pd
import pickle
import os

# Optional: pip install plotly
import plotly.graph_objects as go

st.set_page_config(page_title="Bengaluru Land Price Predictor", page_icon="🏙️", layout="wide")
st.title("Bengaluru Land Price Predictor")
st.markdown("This app predicts land prices in Bengaluru based on various features.")

# --- Model and Data Loaders ---
@st.cache_resource
def load_model():
    model_path = r'E:\data\dsatm\6th sem\ML\Land Price Prediction\random_forest_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        st.error("Model file not found. Please run the training script first.")
        return None

@st.cache_data
def load_data():
    data_path = r'E:\data\dsatm\6th sem\ML\Land Price Prediction\bengaluru_land_prices.csv'
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        st.error("Dataset file not found.")
        return pd.DataFrame()

model = load_model()
df = load_data()

# --- Example/Reset Button Logic ---
def fill_example():
    st.session_state['Location'] = df['Location'].iloc[0]
    st.session_state['Property Type'] = df['Property Type'].iloc[0]
    st.session_state['Zoning'] = df['Zoning'].iloc[0]
    st.session_state['Land Area (sq.ft)'] = float(df['Land Area (sq.ft)'].median())
    st.session_state['Road Width (m)'] = float(df['Road Width (m)'].median())
    st.session_state['Price per sq.ft (INR)'] = float(df['Price per sq.ft (INR)'].median())
    st.session_state['BBMP Property Tax (INR)'] = float(df['BBMP Property Tax (INR)'].median())
    st.session_state['Stamp Duty (INR)'] = float(df['Stamp Duty (INR)'].median())
    st.session_state['Proximity to Metro (km)'] = float(df['Proximity to Metro (km)'].median())
    st.session_state['Flood Risk'] = df['Flood Risk'].iloc[0]
    st.session_state['Pollution Index'] = df['Pollution Index'].iloc[0]
    st.session_state['Proximity to IT Hub (km)'] = float(df['Proximity to IT Hub (km)'].median())
    st.session_state['Water Supply'] = df['Water Supply'].iloc[0]
    st.session_state['Crime Rate'] = df['Crime Rate'].iloc[0]
    st.session_state['Historical Price Trend (5Y % Change)'] = float(df['Historical Price Trend (5Y % Change)'].median())

def reset_form():
    for key in st.session_state.keys():
        del st.session_state[key]

if model and not df.empty:
    # --- Sidebar Inputs with Expanders ---
    st.sidebar.header("Property Features")
    st.sidebar.button("Fill Example Data", on_click=fill_example)
    st.sidebar.button("Reset", on_click=reset_form)

    with st.sidebar.expander("Basic Property Details", expanded=True):
        location = st.selectbox("Location", sorted(df['Location'].unique()), key='Location', help="Select the area in Bengaluru.")
        property_type = st.selectbox("Property Type", sorted(df['Property Type'].unique()), key='Property Type')
        zoning = st.selectbox("Zoning", sorted(df['Zoning'].unique()), key='Zoning')

    with st.sidebar.expander("Physical Attributes"):
        min_area = float(df['Land Area (sq.ft)'].min())
        max_area = float(df['Land Area (sq.ft)'].max())
        land_area = st.number_input("Land Area (sq.ft)", min_value=min_area, max_value=max_area, value=float(df['Land Area (sq.ft)'].median()), step=100.0, key='Land Area (sq.ft)', help="Total area of the land in square feet.")
        min_road = float(df['Road Width (m)'].min())
        max_road = float(df['Road Width (m)'].max())
        road_width = st.number_input("Road Width (m)", min_value=min_road, max_value=max_road, value=float(df['Road Width (m)'].median()), step=0.5, key='Road Width (m)')

    with st.sidebar.expander("Financial Details"):
        min_price_per_sqft = float(df['Price per sq.ft (INR)'].min())
        max_price_per_sqft = float(df['Price per sq.ft (INR)'].max())
        price_per_sqft = st.number_input("Price per sq.ft (INR)", min_value=min_price_per_sqft, max_value=max_price_per_sqft, value=float(df['Price per sq.ft (INR)'].median()), step=10.0, key='Price per sq.ft (INR)')
        min_tax = float(df['BBMP Property Tax (INR)'].min())
        max_tax = float(df['BBMP Property Tax (INR)'].max())
        property_tax = st.number_input("BBMP Property Tax (INR)", min_value=min_tax, max_value=max_tax, value=float(df['BBMP Property Tax (INR)'].median()), step=100.0, key='BBMP Property Tax (INR)')
        min_stamp = float(df['Stamp Duty (INR)'].min())
        max_stamp = float(df['Stamp Duty (INR)'].max())
        stamp_duty = st.number_input("Stamp Duty (INR)", min_value=min_stamp, max_value=max_stamp, value=float(df['Stamp Duty (INR)'].median()), step=100.0, key='Stamp Duty (INR)')

    with st.sidebar.expander("Connectivity & Environment"):
        min_metro = float(df['Proximity to Metro (km)'].min())
        max_metro = float(df['Proximity to Metro (km)'].max())
        proximity_metro = st.number_input("Proximity to Metro (km)", min_value=min_metro, max_value=max_metro, value=float(df['Proximity to Metro (km)'].median()), step=0.1, key='Proximity to Metro (km)')
        flood_risk = st.selectbox("Flood Risk", sorted(df['Flood Risk'].unique()), key='Flood Risk')
        pollution_index = st.selectbox("Pollution Index", sorted(df['Pollution Index'].unique()), key='Pollution Index')
        min_it_hub = float(df['Proximity to IT Hub (km)'].min())
        max_it_hub = float(df['Proximity to IT Hub (km)'].max())
        proximity_it_hub = st.number_input("Proximity to IT Hub (km)", min_value=min_it_hub, max_value=max_it_hub, value=float(df['Proximity to IT Hub (km)'].median()), step=0.1, key='Proximity to IT Hub (km)')
        water_supply = st.selectbox("Water Supply", sorted(df['Water Supply'].unique()), key='Water Supply')
        crime_rate = st.selectbox("Crime Rate", sorted(df['Crime Rate'].unique()), key='Crime Rate')
        min_trend = float(df['Historical Price Trend (5Y % Change)'].min())
        max_trend = float(df['Historical Price Trend (5Y % Change)'].max())
        historical_trend = st.number_input("Historical Price Trend (5Y % Change)", min_value=min_trend, max_value=max_trend, value=float(df['Historical Price Trend (5Y % Change)'].median()), step=0.5, key='Historical Price Trend (5Y % Change)')

    predict_button = st.sidebar.button("Predict Land Price")

    # --- Main Content ---
    col1, col2 = st.columns([2, 1])
    property_info = {
        "Location": location,
        "Land Area (sq.ft)": land_area,
        "Property Type": property_type,
        "Price per sq.ft (INR)": price_per_sqft,
        "BBMP Property Tax (INR)": property_tax,
        "Stamp Duty (INR)": stamp_duty,
        "Road Width (m)": road_width,
        "Zoning": zoning,
        "Proximity to Metro (km)": proximity_metro,
        "Flood Risk": flood_risk,
        "Pollution Index": pollution_index,
        "Proximity to IT Hub (km)": proximity_it_hub,
        "Water Supply": water_supply,
        "Crime Rate": crime_rate,
        "Historical Price Trend (5Y % Change)": historical_trend
    }

    with col1:
        st.subheader("Property Information")
        st.table(pd.DataFrame([property_info]))

    if predict_button:
        with st.spinner("Predicting land price..."):
            input_data = pd.DataFrame([property_info])
            try:
                prediction = model.predict(input_data)
                st.success("Prediction complete!")
                with col2:
                    st.subheader("Prediction Result")
                    st.markdown("### Estimated Land Price")
                    st.markdown(f"<h2 style='color:green;'>₹{prediction[0]:,.2f}</h2>", unsafe_allow_html=True)
                    calculated_price_per_sqft = prediction[0] / land_area
                    st.markdown(f"**Price per sq.ft:** ₹{calculated_price_per_sqft:,.2f}")

                    # Visualization
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prediction[0],
                        title={'text': "Predicted Land Price (INR)"},
                        gauge={'axis': {'range': [0, max_price_per_sqft * land_area * 1.2]}}
                    ))
                    st.plotly_chart(fig, use_container_width=True)

                    # Contextual Info
                    if calculated_price_per_sqft > price_per_sqft:
                        st.info("The predicted price is higher than the input price per sq.ft, suggesting this property might be undervalued.")
                    else:
                        st.info("The predicted price is lower than or equal to the input price per sq.ft, suggesting this property might be fairly priced or overvalued.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # --- How it Works Section ---
    with st.expander("How does this app work?"):
        st.markdown("""
        **How it Works:**
        - This app uses a machine learning model (Random Forest) trained on real Bengaluru land price data.
        - Enter property details in the sidebar to get an estimated price.
        - The model considers location, area, property type, taxes, connectivity, and more.
        - **Note:** Results are indicative and may not reflect actual market prices. Always consult a real estate expert before making decisions.
        """)

    st.markdown("---")
    st.markdown("Developed for HCAI Course Project")

else:
    st.warning("Please make sure the model is trained and the dataset is available.")
    if st.button("Train Model"):
        st.info("Running the training script...")
        import subprocess
        try:
            result = subprocess.run(["python", "model/train_random_forest.py"], capture_output=True, text=True, check=True)
            st.success("Model trained successfully! Please refresh the page.")
            st.code(result.stdout)
        except subprocess.CalledProcessError as e:
            st.error("Error training the model")
            st.code(e.stderr)