import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 20px;
        color: #424242;
        margin-top: 0px;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-result {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background-color: #e8f5e9;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<p class="main-header">House Price Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">An AI-powered tool to estimate house prices based on property features</p>', unsafe_allow_html=True)

# Load dataset with error handling
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("Housing.csv")
        return data
    except FileNotFoundError:
        st.error("Housing.csv file not found. Please check the file path.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

df = load_data()

# Sidebar for model information and stats
with st.sidebar:
    st.markdown("### 📊 Model Information")
    st.markdown("This model uses **Gradient Boosting Regression** to predict house prices based on multiple features.")
    
    # Display dataset statistics
    st.markdown("### 📈 Dataset Statistics")
    st.markdown(f"**Total samples:** {df.shape[0]}")
    st.markdown(f"**Price range:** ₹{df['price'].min():,.2f} - ₹{df['price'].max():,.2f}")
    st.markdown(f"**Average price:** ₹{df['price'].mean():,.2f}")
    
    # Add dataset preview option
    if st.checkbox("Show Dataset Preview"):
        st.dataframe(df.head())

# Define tabs for input and analysis
tab1, tab2 = st.tabs(["💰 Price Prediction", "📊 Data Analysis"])

with tab1:
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    # Property details section
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🏠 Property Details")
        
        area = st.number_input("Area (sq ft)", min_value=500, max_value=10000, value=2000, 
                              help="Total area of the property in square feet")
        bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=5, value=2)
        stories = st.number_input("Number of Stories", min_value=1, max_value=4, value=2)
        parking = st.number_input("Parking Spaces", min_value=0, max_value=5, value=1)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Amenities section
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🛋️ Amenities & Location")
        
        mainroad = st.selectbox("Main Road Access", ["yes", "no"])
        guestroom = st.selectbox("Guest Room", ["yes", "no"])
        basement = st.selectbox("Basement", ["yes", "no"])
        hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
        airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
        prefarea = st.selectbox("Preferred Area", ["yes", "no"])
        furnishingstatus = st.selectbox("Furnishing Status", 
                                       ["furnished", "semi-furnished", "unfurnished"])
        st.markdown('</div>', unsafe_allow_html=True)

    # Define categorical and numerical columns
    categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                      'airconditioning', 'prefarea', 'furnishingstatus']
    numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    
    # Preprocessing
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ])
    
    # Splitting data
    X = df.drop(columns=['price'])
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply transformations
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    # Train best model (Gradient Boosting) works on basis of decision tree
    @st.cache_resource
    def train_model():
        best_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, 
                                              max_depth=5, random_state=42)
        best_model.fit(X_train, y_train)
        return best_model, preprocessor
    
    best_model, preprocessor = train_model()
    
    # Model evaluation
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Convert input data
    def get_prediction():
        input_data = pd.DataFrame([[area, bedrooms, bathrooms, stories, parking, mainroad, 
                                   guestroom, basement, hotwaterheating, airconditioning, 
                                   prefarea, furnishingstatus]], 
                                 columns=['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 
                                         'mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                                         'airconditioning', 'prefarea', 'furnishingstatus'])
        input_transformed = preprocessor.transform(input_data)
        predicted_price = best_model.predict(input_transformed)[0]
        return predicted_price
    
    # Prediction section
    st.markdown("### 🧮 Get Prediction")
    predict_col1, predict_col2 = st.columns([3, 1])
    
    with predict_col2:
        predict_button = st.button("Predict Price", type="primary", use_container_width=True)
    
    with predict_col1:
        confidence_level = st.select_slider(
            "Prediction confidence",
            options=["Low", "Medium", "High"],
            value="Medium",
            help="Adjust for wider or narrower price range in prediction"
        )
    
    # Display prediction
    if predict_button:
        predicted_price = get_prediction()
        
        # Add a simple confidence interval based on RMSE
        confidence_multiplier = {"Low": 1.5, "Medium": 1.0, "High": 0.5}
        price_range = rmse * confidence_multiplier[confidence_level]
        
        st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
        st.markdown(f"### Estimated House Price: ₹{predicted_price:,.2f}")
        st.markdown(f"Price Range: ₹{max(0, predicted_price - price_range):,.2f} - ₹{predicted_price + price_range:,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show model metrics
        st.markdown("### Model Performance")
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("R² Score", f"{r2:.2f}", 
                     help="Higher is better. 1.0 is perfect prediction.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with metric_col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("RMSE", f"₹{rmse:,.2f}", 
                     help="Lower is better. Represents average prediction error.")
            st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("### 📊 Feature Importance")
    
    # Calculate feature importance
    @st.cache_data
    def get_feature_importance():
        # Get feature names after one-hot encoding
        ohe = preprocessor.named_transformers_['cat']
        cat_features = ohe.get_feature_names_out(categorical_cols).tolist()
        feature_names = numerical_cols + cat_features
        
        # Get importance and create DataFrame
        importances = best_model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(10)
        
        return importance_df
    
    importance_df = get_feature_importance()
    
    # Create feature importance chart
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top 10 Features Affecting House Price',
        color='Importance',
        color_continuous_scale='blues'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Price distribution chart
    st.markdown("### 💰 Price Distribution")
    price_fig = px.histogram(
        df, 
        x='price', 
        nbins=50,
        title='House Price Distribution',
        color_discrete_sequence=['#1E88E5']
    )
    price_fig.update_layout(height=400)
    st.plotly_chart(price_fig, use_container_width=True)
    
    # Area vs Price scatter plot
    st.markdown("### 📐 Area vs Price Relationship")
    scatter_fig = px.scatter(
        df,
        x='area',
        y='price',
        color='furnishingstatus',
        size='bathrooms',
        hover_data=['bedrooms', 'stories'],
        title='Relationship Between Area and Price',
        color_discrete_sequence=['#1E88E5', '#43A047', '#FFC107']
    )
    scatter_fig.update_layout(height=500)
    st.plotly_chart(scatter_fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #757575; font-size: 14px;">
        <p>© 2025 House Price Prediction System | Built with Streamlit and Scikit-learn | Data updated April 2025</p>
    </div>
    """, 
    unsafe_allow_html=True
)