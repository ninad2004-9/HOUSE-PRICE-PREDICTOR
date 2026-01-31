import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping

import plotly.graph_objects as go
import plotly.express as px
import locale
import time

# ===============================
# Locale safe
# ===============================
try:
    locale.setlocale(locale.LC_ALL, "en_IN.UTF-8")
except:
    locale.setlocale(locale.LC_ALL, "")

def format_inr(x):
    try:
        return locale.format_string("%d", int(x), grouping=True)
    except:
        return f"{x:,.0f}"

# ===============================
# Streamlit config
# ===============================
st.set_page_config(
    page_title="Pan-India House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# ANIMATED CSS (keeping all the beautiful CSS from before)
# ===============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        font-family: 'Poppins', sans-serif;
        color: #e5e7eb;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    h1 {
        color: #ffffff;
        text-align: center;
        font-weight: 800;
        font-size: 3.5rem !important;
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 3s ease infinite, fadeIn 1s ease;
        margin-bottom: 0.5rem !important;
    }
    
    h2 {
        color: #a5b4fc;
        text-align: center;
        font-weight: 600;
        animation: fadeIn 1.2s ease;
    }
    
    h3 {
        color: #c4b5fd;
        font-weight: 600;
        animation: slideInLeft 0.8s ease;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 2px solid rgba(99, 102, 241, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        animation: fadeIn 1s ease, float 6s ease-in-out infinite;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .metric-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 20px 60px rgba(99, 102, 241, 0.4);
        border-color: rgba(99, 102, 241, 0.6);
    }
    
    .price-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 25px;
        padding: 3rem;
        text-align: center;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.6);
        animation: pulse 2s ease-in-out infinite, fadeIn 1s ease;
        margin: 2rem 0;
    }
    
    .price-value {
        font-size: 3.5rem;
        font-weight: 800;
        color: white;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
        margin: 0;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .house-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-left: 5px solid #6366f1;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slideInRight 0.6s ease;
        backdrop-filter: blur(5px);
    }
    
    .house-card:hover {
        transform: translateX(15px) scale(1.02);
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.1) 100%);
        box-shadow: 0 15px 40px rgba(99, 102, 241, 0.3);
        border-left-width: 8px;
    }
    
    .stat-box {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        animation: fadeIn 1s ease;
        transition: all 0.3s ease;
    }
    
    .stat-box:hover {
        background: rgba(99, 102, 241, 0.15);
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.2);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        animation: pulse 2s ease-in-out infinite;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        animation: none;
    }
    
    .filter-section {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(99, 102, 241, 0.2);
        animation: fadeIn 1s ease;
    }
    
    .emoji-icon {
        display: inline-block;
        animation: float 3s ease-in-out infinite;
        font-size: 2rem;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #a5b4fc !important;
        animation: fadeIn 1s ease;
    }
    
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# Animated Header
# ===============================
st.markdown("<h1><span class='emoji-icon'>🏠</span>🏠 HouseScope AI – Intelligent Living Starts Here</h1>", unsafe_allow_html=True)
st.markdown("<h2>🤖 AI-Powered • 🗺️ Location-Aware • 📊 Data-Driven</h2>", unsafe_allow_html=True)

# ===============================
# Helper Functions
# ===============================
def safe_int_convert(value, default=0):
    """Safely convert value to int"""
    try:
        if pd.isna(value):
            return default
        # Handle string 'yes'/'no' values
        if isinstance(value, str):
            value_lower = value.lower()
            if value_lower == 'yes':
                return 1
            elif value_lower == 'no':
                return 0
        return int(float(value))
    except (ValueError, TypeError):
        return default

def get_furnish_label(x):
    """Convert furnishing status to readable label"""
    if pd.isna(x):
        return "Unknown"
    
    if isinstance(x, str):
        x_lower = x.lower()
        if x_lower == "unfurnished":
            return "Unfurnished"
        elif x_lower in ["semi-furnished", "semifurnished"]:
            return "Semi-furnished"
        elif x_lower == "furnished":
            return "Furnished"
        return x.title()
    
    try:
        x_int = int(float(x))
        mapping = {0: "Unfurnished", 1: "Semi-furnished", 2: "Furnished"}
        return mapping.get(x_int, "Unknown")
    except (ValueError, TypeError):
        return "Unknown"

# ===============================
# Load data
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("data/Housing_Pan_India_Extended_Dataset.csv")
    return df

try:
    with st.spinner("🔄 Loading data..."):
        df = load_data()
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# ===============================
# Preprocess (FIXED)
# ===============================
@st.cache_data
def preprocess_data(df):
    # Create a copy
    df = df.copy()
    
    # Convert yes/no columns to 1/0
    yes_no = ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]
    for c in yes_no:
        if c in df.columns:
            # Convert to string first, then map
            df[c] = df[c].astype(str).str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0})
            df[c] = df[c].fillna(0).astype(float)

    # Convert furnishing status
    if "furnishingstatus" in df.columns:
        df["furnishingstatus"] = df["furnishingstatus"].astype(str).str.lower()
        df["furnishingstatus"] = df["furnishingstatus"].replace({
            "unfurnished": 0,
            "semi-furnished": 1,
            "semifurnished": 1,
            "furnished": 2,
            "0": 0,
            "1": 1,
            "2": 2
        })
        df["furnishingstatus"] = pd.to_numeric(df["furnishingstatus"], errors='coerce').fillna(0).astype(float)

    # City average price
    city_price_map = df.groupby("city")["price"].mean()
    df["city_avg_price"] = df["city"].map(city_price_map)

    # Drop non-numeric columns
    df_model = df.drop(["state","city","locality"], axis=1, errors='ignore')
    
    # Ensure all columns are numeric
    for col in df_model.columns:
        df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
    
    # Drop rows with NaN
    df_model = df_model.dropna()

    X = df_model.drop("price", axis=1)
    y = df_model["price"]

    y_scaler = StandardScaler()
    y_norm = y_scaler.fit_transform(y.values.reshape(-1,1)).flatten()

    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)

    X_train, _, y_train, _ = train_test_split(
        X_scaled, y_norm, test_size=0.2, random_state=42
    )

    return X, X_train, y_train, X_scaler, y_scaler, city_price_map

X, X_train, y_train, scaler, y_scaler, city_price_map = preprocess_data(df.copy())

# ===============================
# Models
# ===============================
def build_cnn(shape):
    model = Sequential([
        Input(shape=shape),
        Conv1D(32, 2, activation="relu"),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_lstm(shape):
    model = Sequential([
        Input(shape=shape),
        LSTM(32),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

@st.cache_resource
def train_models(X_train, y_train):
    X_seq = np.expand_dims(X_train, axis=2)

    cnn = build_cnn((X_seq.shape[1],1))
    lstm = build_lstm((X_seq.shape[1],1))

    es = EarlyStopping(patience=5, restore_best_weights=True)

    cnn.fit(X_seq, y_train, epochs=30, batch_size=32, verbose=0, callbacks=[es])
    lstm.fit(X_seq, y_train, epochs=30, batch_size=32, verbose=0, callbacks=[es])

    cnn_p = cnn.predict(X_seq, verbose=0)
    lstm_p = lstm.predict(X_seq, verbose=0)

    meta = RandomForestRegressor(n_estimators=80, random_state=42)
    meta.fit(np.hstack([cnn_p, lstm_p]), y_train)

    return cnn, lstm, meta

if "models" not in st.session_state:
    with st.spinner("🧠 Training AI models..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        st.session_state.models = train_models(X_train, y_train)
        progress_bar.empty()
        st.success("✅ Models trained!")
        time.sleep(0.5)

cnn, lstm, meta = st.session_state.models

# ===============================
# Tabs
# ===============================
tab1, tab2 = st.tabs(["🎯 Price Prediction", "🗺️ Interactive Map Explorer"])

# ===============================
# TAB 1: Prediction
# ===============================
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### 📍 Select Property Location")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        state = st.selectbox("🏛️ State", sorted(df["state"].unique()))
    
    city_df = df[df["state"] == state]
    with col2:
        city = st.selectbox("🌆 City", sorted(city_df["city"].unique()))
    
    local_df = city_df[city_df["city"] == city]
    with col3:
        locality = st.selectbox("📌 Locality", sorted(local_df["locality"].unique()))

    row = local_df[local_df["locality"] == locality].iloc[0]
    city_avg = city_price_map[city]
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🏡 Property Specifications")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        area = st.number_input("📐 Area (sq.ft)", 300, 10000, 1200, step=50)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        bedrooms = st.number_input("🛏️ Bedrooms", 1, 10, 3)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        bathrooms = st.number_input("🚿 Bathrooms", 1, 10, 2)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Summary
    st.markdown("### 📋 Property Summary")
    st.markdown(f"""
    <div class="stat-box">
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
            <div>
                <strong>🏙️ Location:</strong><br>
                {locality}, {city}<br>
                {state}
            </div>
            <div>
                <strong>📏 Specs:</strong><br>
                Area: {area} sq.ft<br>
                Rooms: {bedrooms}BR | {bathrooms}BA
            </div>
            <div>
                <strong>💰 Market:</strong><br>
                City Avg: ₹ {format_inr(city_avg)}<br>
                Furnishing: {get_furnish_label(row['furnishingstatus'])}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_btn = st.button("🎯 Predict Price Now", use_container_width=True)
    
    if predict_btn:
        with st.spinner("🔮 Analyzing..."):
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            steps = ["📊 Preprocessing...", "🧮 CNN model...", 
                    "🔄 LSTM model...", "🎲 Ensemble...", "✨ Finalizing..."]
            
            for i, step in enumerate(steps):
                progress_text.text(step)
                time.sleep(0.3)
                progress_bar.progress((i + 1) * 20)
            
            # Create input with safe conversions
            input_df = pd.DataFrame([{
                "area": float(area),
                "bedrooms": float(bedrooms),
                "bathrooms": float(bathrooms),
                "stories": float(safe_int_convert(row["stories"], 1)),
                "parking": float(safe_int_convert(row["parking"], 0)),
                "mainroad": float(safe_int_convert(row["mainroad"], 1)),
                "guestroom": float(safe_int_convert(row["guestroom"], 0)),
                "basement": float(safe_int_convert(row["basement"], 0)),
                "hotwaterheating": float(safe_int_convert(row["hotwaterheating"], 0)),
                "airconditioning": float(safe_int_convert(row["airconditioning"], 1)),
                "prefarea": float(safe_int_convert(row["prefarea"], 0)),
                "furnishingstatus": float(safe_int_convert(row["furnishingstatus"], 1)),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "city_avg_price": float(city_avg)
            }])[X.columns]

            X_scaled = scaler.transform(input_df)
            X_seq = np.expand_dims(X_scaled, axis=2)

            cnn_pred = cnn.predict(X_seq, verbose=0)[0][0]
            lstm_pred = lstm.predict(X_seq, verbose=0)[0][0]
            meta_pred = meta.predict([[cnn_pred, lstm_pred]])[0]

            p = 0.4*cnn_pred + 0.4*lstm_pred + 0.2*meta_pred
            price = y_scaler.inverse_transform([[p]])[0][0]
            
            progress_text.empty()
            progress_bar.empty()
            
            st.session_state.predicted_price = price
    
    if "predicted_price" in st.session_state:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="price-card">
            <h2 style="color: white; margin: 0; font-size: 1.5rem;">💰 Estimated Value</h2>
            <p class="price-value">₹ {format_inr(st.session_state.predicted_price)}</p>
            <p style="color: rgba(255,255,255,0.8); margin: 0;">AI-Powered Prediction</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Analysis")
        col1, col2, col3, col4 = st.columns(4)
        
        price_diff = st.session_state.predicted_price - city_avg
        price_diff_pct = (price_diff / city_avg) * 100
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🎯 Predicted", f"₹ {format_inr(st.session_state.predicted_price)}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📈 City Avg", f"₹ {format_inr(city_avg)}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("💵 Difference", f"₹ {format_inr(abs(price_diff))}", f"{price_diff_pct:+.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            price_per_sqft = st.session_state.predicted_price / area
            st.metric("📐 Per Sq.Ft", f"₹ {format_inr(price_per_sqft)}")
            st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# TAB 2: MAP
# ===============================
with tab2:
    st.markdown("## 🗺️ Discover Properties")
    st.markdown("**Interactive map with real-time filters**")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    st.markdown("### 🎛️ Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_cities = st.multiselect("🌆 Cities", sorted(df["city"].unique()), default=None)
    
    with col2:
        min_price = int(df["price"].min())
        max_price = int(df["price"].max())
        price_range = st.slider("💰 Price Range", min_price, max_price, (min_price, max_price), format="₹%d")
    
    with col3:
        filter_bedrooms = st.multiselect("🛏️ Bedrooms", sorted(df["bedrooms"].unique()), default=None)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prepare map data
    map_df = df[['city','locality','latitude','longitude','price','area','bedrooms','furnishingstatus']].dropna()
    
    if filter_cities:
        map_df = map_df[map_df['city'].isin(filter_cities)]
    
    map_df = map_df[(map_df['price'] >= price_range[0]) & (map_df['price'] <= price_range[1])]
    
    if filter_bedrooms:
        map_df = map_df[map_df['bedrooms'].isin(filter_bedrooms)]
    
    if len(map_df) > 500:
        map_df = map_df.sample(500)
    
    map_df["furnish_label"] = map_df["furnishingstatus"].apply(get_furnish_label)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if len(map_df) > 0:
        st.markdown("### 📊 Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🏠 Properties", len(map_df))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("💰 Avg", f"₹ {format_inr(map_df['price'].mean())}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📉 Min", f"₹ {format_inr(map_df['price'].min())}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📈 Max", f"₹ {format_inr(map_df['price'].max())}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Map
        fig = go.Figure(go.Scattermapbox(
            lat=map_df['latitude'],
            lon=map_df['longitude'],
            mode='markers',
            marker=dict(
                size=12,
                color=map_df['price'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="Price (₹)",
                        font=dict(color='#e5e7eb')
                    ),
                    tickfont=dict(color='#e5e7eb')
                ),
                opacity=0.85
            ),
            text=map_df.apply(
                lambda r: f"""
                <b>{r['locality']}, {r['city']}</b><br>
                <b>₹{format_inr(r['price'])}</b><br>
                Area: {r['area']} sq.ft<br>
                Bedrooms: {int(r['bedrooms'])}<br>
                Furnishing: {r['furnish_label']}
                """, axis=1
            ),
            hoverinfo="text"
        ))

        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_center={"lat": map_df['latitude'].mean(), "lon": map_df['longitude'].mean()},
            mapbox_zoom=5,
            height=650,
            margin={"r":0,"t":0,"l":0,"b":0}
        )

        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🏘️ Property Listings")
        
        selected_city = st.selectbox("Select city", sorted(map_df["city"].unique()))
        
        city_properties = map_df[map_df["city"] == selected_city].sort_values("price", ascending=False)
        
        st.markdown(f"#### {len(city_properties)} properties in {selected_city}")
        
        for idx, prop in city_properties.head(10).iterrows():
            st.markdown(f"""
            <div class="house-card">
                <h3 style="color: #a5b4fc; margin-top: 0;">📍 {prop['locality']}</h3>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin-top: 1rem;">
                    <div>
                        <strong style="color: #10b981;">💰 Price</strong><br>
                        <span style="font-size: 1.3rem; font-weight: 600;">₹ {format_inr(prop['price'])}</span>
                    </div>
                    <div>
                        <strong style="color: #3b82f6;">📐 Area</strong><br>
                        <span style="font-size: 1.1rem;">{prop['area']} sq.ft</span>
                    </div>
                    <div>
                        <strong style="color: #f59e0b;">🛏️ Bedrooms</strong><br>
                        <span style="font-size: 1.1rem;">{int(prop['bedrooms'])} Rooms</span>
                    </div>
                </div>
                <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(99, 102, 241, 0.3);">
                    <strong>🪑 Furnishing:</strong> {prop['furnish_label']} &nbsp;|&nbsp;
                    <strong>📊 Price/Sq.Ft:</strong> ₹ {format_inr(prop['price']/prop['area'])}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if len(city_properties) > 10:
            st.info(f"ℹ️ Showing top 10. Total: {len(city_properties)}")
    
    else:
        st.warning("⚠️ No properties found. Adjust filters.")

# Footer
st.markdown("<br><br>---", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; padding: 2rem; opacity: 0.8;'>
    <p style='font-size: 1.1rem;'><strong>🏠 Pan-India House Price Predictor</strong></p>
    <p style='font-size: 0.9rem; opacity: 0.7;'>AI • Streamlit • TensorFlow • Plotly</p>
</div>
""", unsafe_allow_html=True)