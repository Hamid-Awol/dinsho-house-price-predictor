import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime
import os
import pickle
from utils import (
    load_data, get_data_summary,
    train_model_with_complexity_control,
    save_model, load_model,
    predict_price, predict_bulk_prices,
    get_sample_csv_template
)

# Page configuration
st.set_page_config(
    page_title="Addis House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Constants
GREEN = "#078930"
YELLOW = "#FCDD09"
RED = "#DA121A"
DISCOUNT_THRESHOLD = 2
DISCOUNT_RATE = 0.10

# CSS styling
st.markdown(f"""
<style>
.header {{
    background: linear-gradient(90deg, {GREEN}, {YELLOW}, {RED});
    padding: 2rem;
    border-radius: 10px;
    text-align: center;
    color: white;
    margin-bottom: 2rem;
}}
.discount-badge {{
    background-color: {GREEN};
    color: white;
    padding: 0.5rem;
    border-radius: 5px;
    text-align: center;
    font-weight: bold;
    margin-bottom: 1rem;
}}
.price-card {{
    background: linear-gradient(135deg, {GREEN}, #0a6b2a);
    padding: 1.5rem;
    border-radius: 10px;
    text-align: center;
    color: white;
    margin: 0.5rem;
}}
.property-card {{
    background: linear-gradient(135deg, #f0f2f6, #ffffff);
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    border-left: 5px solid {GREEN};
}}
.stButton > button {{
    background-color: {GREEN};
    color: white;
    font-weight: bold;
}}
.warning-box {{
    background-color: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}}
</style>
<div class="header">
    <h1>🏠 Addis Ababa House Price Predictor</h1>
    <p>Ethiopian Real Estate Market Intelligence</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "model_data" not in st.session_state:
    st.session_state.model_data = None
if "df" not in st.session_state:
    st.session_state.df = None
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

# Function to train default model if none exists
def get_or_train_default_model(df):
    """Get existing model or train a default one"""
    # Try to load existing model first
    loaded_model = load_model()
    if loaded_model:
        st.session_state.model_data = loaded_model
        st.session_state.model_loaded = True
        return loaded_model
    
    # If no model exists, train a default model
    with st.spinner("🔄 Training default model for immediate predictions..."):
        try:
            default_model = train_model_with_complexity_control(
                df, 
                test_size=0.2,
                n_estimators=100,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                regularization=None,
                alpha=1.0,
                use_cross_validation=True,
                cv_folds=5
            )
            st.session_state.model_data = default_model
            st.session_state.model_loaded = True
            save_model(default_model)
            return default_model
        except Exception as e:
            st.error(f"Error training default model: {str(e)}")
            return None

# Load data
if st.session_state.df is None:
    st.session_state.df = load_data()
    if st.session_state.df is None:
        st.stop()

df = st.session_state.df

# Ensure model is loaded for predictions
if not st.session_state.model_loaded:
    model_data = get_or_train_default_model(df)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Navigation")
    page = st.radio("Select Page", [
        "📁 Data Overview",
        "🎯 Train Model",
        "🔬 Complexity Analysis",
        "🏠 Single House Prediction",
        "📊 Mass Prediction (Bulk)"
    ])
    
    st.markdown("---")
    
    # Show model status
    if st.session_state.model_loaded:
        if st.session_state.model_data:
            metrics = st.session_state.model_data.get('metrics', {})
            st.success(f"✅ Model Ready")
            if metrics:
                st.caption(f"R² Score: {metrics.get('test_r2', 0):.3f}")
    else:
        st.warning("⚠️ Model loading...")
    
    st.markdown("---")
    st.markdown("### 💰 Bulk Purchase Discount")
    st.info(f"""
    **Special Offer!**
    - Buy more than {DISCOUNT_THRESHOLD} houses
    - Get {DISCOUNT_RATE*100:.0f}% discount on total price
    - Perfect for real estate investors
    """)
    
    st.markdown("---")
    st.markdown("### 🏢 Building Types")
    st.info("""
    **Floor Types:**
    - G+0 (Ground floor only)
    - G+1 to G+50 (Ground + floors)
    - Higher floors typically cost more
    """)

# Page: Data Overview
if page == "📁 Data Overview":
    st.header("📊 Dataset Information")
    
    summary = get_data_summary(df)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏘️ Total Properties", summary['total_properties'])
    with col2:
        st.metric("💰 Avg Price", f"ETB {summary['avg_price']:,.0f}")
    with col3:
        st.metric("📏 Avg Area", f"{summary['avg_area']:.0f} sqm")
    with col4:
        st.metric("🛏️ Avg Bedrooms", f"{summary['avg_bedrooms']:.1f}")
    
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(20), use_container_width=True)
    
    st.subheader("💰 Average Price by Sub-City")
    fig = px.bar(
        x=summary['price_by_city'].index,
        y=summary['price_by_city'].values,
        title="Average House Price by Sub-City",
        labels={'x': 'Sub-City', 'y': 'Price (ETB)'},
        color=summary['price_by_city'].values,
        color_continuous_scale=['red', 'yellow', 'green']
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📊 Price Distribution")
    fig2 = px.histogram(
        df, x='price_etb', nbins=30,
        title="Distribution of House Prices",
        labels={'price_etb': 'Price (ETB)'}
    )
    st.plotly_chart(fig2, use_container_width=True)

# Page: Train Model
elif page == "🎯 Train Model":
    st.header("🎯 Train Machine Learning Model with Complexity Control")
    
    st.markdown("""
    ### Control Model Complexity to Prevent Overfitting/Underfitting
    
    Adjust these parameters to find the optimal model complexity:
    - **Tree Depth**: Controls how detailed the model can be
    - **Regularization**: Adds penalty to reduce overfitting
    - **Cross-validation**: Ensures robust performance estimation
    """)
    
    # Show current model info if exists
    if st.session_state.model_data:
        current_metrics = st.session_state.model_data.get('metrics', {})
        st.info(f"""
        **Current Model Performance:**
        - Training R²: {current_metrics.get('train_r2', 0):.4f}
        - Test R²: {current_metrics.get('test_r2', 0):.4f}
        - RMSE: ETB {current_metrics.get('rmse', 0):,.0f}
        """)
    
    tab1, tab2, tab3 = st.tabs(["Model Parameters", "Regularization", "Cross-Validation"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
            n_estimators = st.slider("Number of Trees", 50, 300, 100, 50)
            max_depth = st.selectbox("Max Tree Depth", [3, 5, 10, 15, 20, 30, None], 
                                     format_func=lambda x: "None (unlimited)" if x is None else str(x))
        with col2:
            min_samples_split = st.slider("Min Samples to Split", 2, 20, 2)
            min_samples_leaf = st.slider("Min Samples per Leaf", 1, 10, 1)
            max_features = st.selectbox("Max Features", ["sqrt", "log2", None])
    
    with tab2:
        st.markdown("### Regularization (Prevents Overfitting)")
        use_regularization = st.checkbox("Use Regularization")
        if use_regularization:
            reg_type = st.selectbox("Regularization Type", ["ridge", "lasso"])
            alpha = st.slider("Regularization Strength (alpha)", 0.01, 10.0, 1.0, 0.01)
            st.info(f"""
            **{reg_type.upper()} Regularization:**
            - **Ridge (L2)**: Reduces large coefficients
            - **Lasso (L1)**: Can reduce some coefficients to zero
            - Higher alpha = stronger regularization
            """)
        else:
            reg_type = None
            alpha = 1.0
    
    with tab3:
        st.markdown("### Cross-Validation")
        use_cv = st.checkbox("Use Cross-Validation", value=True)
        if use_cv:
            cv_folds = st.slider("Number of CV Folds", 3, 10, 5)
        else:
            cv_folds = 5
    
    if st.button("🚀 Train New Model", type="primary", width='stretch'):
        with st.spinner("Training model..."):
            model_data = train_model_with_complexity_control(
                df, test_size, n_estimators, max_depth,
                min_samples_split, min_samples_leaf, max_features,
                reg_type if use_regularization else None, alpha,
                use_cv, cv_folds
            )
            st.session_state.model_data = model_data
            st.session_state.model_loaded = True
            save_model(model_data)
            
            st.success("✅ Model trained successfully!")
            
            metrics = model_data['metrics']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Training R²", f"{metrics['train_r2']:.4f}")
            with col2:
                st.metric("Test R²", f"{metrics['test_r2']:.4f}")
            with col3:
                st.metric("RMSE", f"ETB {metrics['rmse']:,.0f}")
            with col4:
                gap = metrics['train_r2'] - metrics['test_r2']
                st.metric("Overfitting Gap", f"{gap:.4f}")

# Page: Complexity Analysis
elif page == "🔬 Complexity Analysis":
    st.header("🔬 Model Complexity Analysis")
    st.markdown("""
    ### Understanding Overfitting and Underfitting
    
    - **Underfitting**: Model too simple, poor performance on both training and test
    - **Overfitting**: Model too complex, excellent on training but poor on test
    - **Optimal**: Balanced complexity that generalizes well
    """)
    
    if st.session_state.model_data is None:
        st.warning("⚠️ Training default model for analysis...")
        model_data = get_or_train_default_model(df)
    else:
        model_data = st.session_state.model_data
    
    metrics = model_data['metrics']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training R²", f"{metrics['train_r2']:.4f}")
    with col2:
        st.metric("Test R²", f"{metrics['test_r2']:.4f}")
    with col3:
        gap = metrics['train_r2'] - metrics['test_r2']
        st.metric("Overfitting Gap", f"{gap:.4f}")
    
    if metrics['train_r2'] < 0.5 and metrics['test_r2'] < 0.5:
        st.warning("⚠️ **Underfitting Detected!** Increase tree depth, reduce min samples")
    elif gap > 0.15:
        st.warning("⚠️ **Overfitting Detected!** Reduce tree depth, use regularization")
    else:
        st.success("✅ **Good Balance!** Model generalizes well")

# Page: Single House Prediction
elif page == "🏠 Single House Prediction":
    st.header("🏠 Single House Price Prediction")
    
    if st.session_state.model_data is None:
        st.info("🔄 Loading model for predictions...")
        model_data = get_or_train_default_model(df)
    else:
        model_data = st.session_state.model_data
    
    with st.form("single_form"):
        st.subheader("🏠 Property Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sub_city = st.selectbox("Sub-City *", df['sub_city'].unique().tolist())
            area = st.number_input("Area (sqm) *", 50, 1000, 150)
            bedrooms = st.selectbox("Bedrooms *", [1, 2, 3, 4, 5, 6])
            bathrooms = st.selectbox("Bathrooms *", [1, 2, 3, 4, 5])
            year_built = st.number_input("Year Built *", 1960, 2024, 2010)
        
        with col2:
            water = st.selectbox("Water Source *", df['water_source'].unique().tolist())
            electricity = st.selectbox("Electricity *", df['electricity'].unique().tolist())
            fence = st.selectbox("Fence *", df['fence'].unique().tolist())
            floor_type = st.selectbox("Building Type *", [f"G+{i}" for i in range(0, 51)])
            condition = st.selectbox("Condition *", df['property_condition'].unique().tolist())
            road = st.selectbox("Road Access *", df['road_access'].unique().tolist())
        
        st.caption("* Required fields")
        submitted = st.form_submit_button("💰 Calculate Price", type="primary", width='stretch')
    
    if submitted:
        input_data = {
            'sub_city': sub_city, 'area_sqm': area, 'bedrooms': bedrooms,
            'bathrooms': bathrooms, 'year_built': year_built, 'water_source': water,
            'electricity': electricity, 'fence': fence, 'floor_type': floor_type,
            'property_condition': condition, 'road_access': road,
            'distance_to_school_km': 2.5, 'distance_to_market_km': 1.5
        }
        
        price = predict_price(
            model_data['model'], model_data['scaler'], input_data,
            model_data['feature_names'], model_data['encoders']
        )
        
        st.markdown(f"""
        <div class="price-card">
            <h2>🏠 Predicted House Price</h2>
            <h1 style="font-size: 3rem;">ETB {price:,.0f}</h1>
            <p>Price per square meter: ETB {price/area:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Property summary
        with st.expander("📋 Property Summary", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **📍 Location:** {sub_city}  
                **📏 Area:** {area} sqm  
                **🛏️ Bedrooms:** {bedrooms}  
                **🛁 Bathrooms:** {bathrooms}  
                **📅 Year Built:** {year_built}
                """)
            with col2:
                st.markdown(f"""
                **💧 Water Source:** {water}  
                **⚡ Electricity:** {electricity}  
                **🔒 Fence:** {fence}  
                **🏢 Building:** {floor_type}  
                **🏗️ Condition:** {condition}  
                **🛣️ Road Access:** {road}
                """)
        
        # Comparison with average
        avg_price_by_city = df.groupby('sub_city')['price_etb'].mean()
        city_avg = avg_price_by_city.get(sub_city, 0)
        
        if price > city_avg:
            diff_pct = ((price - city_avg) / city_avg) * 100
            st.info(f"💡 This property is **{diff_pct:.1f}% above** the average for {sub_city} (ETB {city_avg:,.0f})")
        else:
            diff_pct = ((city_avg - price) / city_avg) * 100
            st.info(f"💡 This property is **{diff_pct:.1f}% below** the average for {sub_city} (ETB {city_avg:,.0f})")

# Page: Mass Prediction
else:
    st.header("📊 Mass House Price Prediction")
    st.markdown(f"""
    <div class="discount-badge">
        🎉 Bulk Purchase Discount: Buy more than {DISCOUNT_THRESHOLD} houses and get {DISCOUNT_RATE*100:.0f}% OFF! 🎉
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.model_data is None:
        st.info("🔄 Loading model for predictions...")
        model_data = get_or_train_default_model(df)
    else:
        model_data = st.session_state.model_data
    
    # Download template
    st.subheader("📥 Step 1: Download Template")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info("Download the sample CSV template to understand the required format")
    with col2:
        sample_df = get_sample_csv_template()
        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Template",
            data=csv,
            file_name="house_price_template.csv",
            mime="text/csv",
            width='stretch'
        )
    
    st.markdown("---")
    
    # Upload CSV
    st.subheader("📤 Step 2: Upload Your Data")
    uploaded_file = st.file_uploader("Upload CSV file with property details", type=['csv'])
    
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(input_df)} properties")
            
            # Show preview
            st.subheader("📋 Uploaded Data Preview")
            st.dataframe(input_df.head(10), use_container_width=True)
            
            # Validate columns
            required_columns = ['sub_city', 'area_sqm', 'bedrooms', 'bathrooms', 'year_built',
                               'water_source', 'electricity', 'fence', 'floor_type',
                               'property_condition', 'road_access']
            
            missing_cols = [col for col in required_columns if col not in input_df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                st.info("Please make sure your CSV includes all required columns. Download the template for reference.")
                st.stop()
            
            # Add optional columns with defaults
            if 'distance_to_school_km' not in input_df.columns:
                input_df['distance_to_school_km'] = 2.5
            if 'distance_to_market_km' not in input_df.columns:
                input_df['distance_to_market_km'] = 1.5
            
            if st.button("🔮 Predict Prices", type="primary", width='stretch'):
                with st.spinner("Calculating prices..."):
                    # Make predictions
                    results_df, summary = predict_bulk_prices(
                        model_data['model'],
                        model_data['scaler'],
                        input_df,
                        model_data['feature_names'],
                        model_data['encoders'],
                        DISCOUNT_THRESHOLD,
                        DISCOUNT_RATE
                    )
                    
                    st.success("✅ Predictions completed!")
                    
                    # Summary Cards
                    st.subheader("📊 Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🏘️ Total Houses", summary['total_houses'])
                    with col2:
                        st.metric("💰 Original Total", f"ETB {summary['original_total']:,.0f}")
                    with col3:
                        if summary['bulk_discount_applied']:
                            st.metric("🎉 Discount Applied", f"{summary['discount_rate']*100:.0f}%")
                            st.metric("💸 Discount Amount", f"ETB {summary['discount_amount']:,.0f}")
                        else:
                            st.metric("🎉 Discount Applied", "None")
                    with col4:
                        st.metric("💰 Final Total", f"ETB {summary['final_total']:,.0f}")
                    
                    # Individual Property Prices Display
                    st.subheader("🏠 Individual Property Prices")
                    
                    # Create a formatted display for each property
                    for idx, row in results_df.iterrows():
                        with st.container():
                            original_price = row['predicted_price_etb']
                            discounted_price = row['discounted_price_etb']
                            
                            st.markdown(f"""
                            <div class="property-card">
                                <h3>🏠 Property #{idx + 1}</h3>
                                <table style="width: 100%;">
                                    <tr>
                                        <td><strong>📍 Location:</strong></td>
                                        <td>{row['sub_city']}</td>
                                        <td><strong>📏 Area:</strong></td>
                                        <td>{row['area_sqm']} sqm</td>
                                    </tr>
                                    <tr>
                                        <td><strong>🛏️ Bedrooms:</strong></td>
                                        <td>{row['bedrooms']}</td>
                                        <td><strong>🛁 Bathrooms:</strong></td>
                                        <td>{row['bathrooms']}</td>
                                    </tr>
                                    <tr>
                                        <td><strong>🏢 Building:</strong></td>
                                        <td>{row['floor_type']}</td>
                                        <td><strong>🏗️ Condition:</strong></td>
                                        <td>{row['property_condition']}</td>
                                    </tr>
                                    <tr>
                                        <td><strong>💰 Original Price:</strong></td>
                                        <td colspan="3"><s>ETB {original_price:,.0f}</s></td>
                                    </tr>
                                    <tr>
                                        <td><strong>💎 Final Price:</strong></td>
                                        <td colspan="3" style="color: {GREEN}; font-weight: bold; font-size: 1.2rem;">ETB {discounted_price:,.0f}</td>
                                    </tr>
                                </table>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Display results table
                    st.subheader("📋 Detailed Results Table")
                    display_df = results_df.copy()
                    display_df['predicted_price_etb'] = display_df['predicted_price_etb'].apply(lambda x: f"ETB {x:,.0f}")
                    display_df['discounted_price_etb'] = display_df['discounted_price_etb'].apply(lambda x: f"ETB {x:,.0f}")
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Visualization
                    st.subheader("📊 Price Comparison Chart")
                    fig = go.Figure()
                    
                    # Add bars for each property
                    fig.add_trace(go.Bar(
                        name='Original Price',
                        x=[f"Property {i+1}" for i in range(len(results_df))],
                        y=results_df['predicted_price_etb'],
                        marker_color='lightgray',
                        text=[f"ETB {p:,.0f}" for p in results_df['predicted_price_etb']],
                        textposition='outside'
                    ))
                    
                    if summary['bulk_discount_applied']:
                        fig.add_trace(go.Bar(
                            name='Discounted Price',
                            x=[f"Property {i+1}" for i in range(len(results_df))],
                            y=results_df['discounted_price_etb'],
                            marker_color=GREEN,
                            text=[f"ETB {p:,.0f}" for p in results_df['discounted_price_etb']],
                            textposition='outside'
                        ))
                    
                    fig.update_layout(
                        title="Price Comparison by Property",
                        xaxis_title="Property",
                        yaxis_title="Price (ETB)",
                        barmode='group',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    results_csv = results_df.to_csv(index=False)
                    
                    st.subheader("💾 Download Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="📥 Download Full Results (CSV)",
                            data=results_csv,
                            file_name=f"house_price_predictions_{timestamp}.csv",
                            mime="text/csv",
                            width='stretch'
                        )
                    
                    # Create summary report
                    summary_report = f"""
                    HOUSE PRICE PREDICTION REPORT
                    Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    {'='*50}
                    
                    SUMMARY
                    {'='*50}
                    Total Houses: {summary['total_houses']}
                    Original Total Price: ETB {summary['original_total']:,.0f}
                    Discount Applied: {'Yes' if summary['bulk_discount_applied'] else 'No'}
                    Discount Rate: {summary['discount_rate']*100:.0f}%
                    Discount Amount: ETB {summary['discount_amount']:,.0f}
                    Final Total Price: ETB {summary['final_total']:,.0f}
                    
                    DETAILED BREAKDOWN
                    {'='*50}
                    """
                    
                    for idx, row in results_df.iterrows():
                        summary_report += f"""
                    Property #{idx + 1}
                    ---------
                    Location: {row['sub_city']}
                    Area: {row['area_sqm']} sqm
                    Bedrooms: {row['bedrooms']}
                    Bathrooms: {row['bathrooms']}
                    Building: {row['floor_type']}
                    Condition: {row['property_condition']}
                    Original Price: ETB {row['predicted_price_etb']:,.0f}
                    Discounted Price: ETB {row['discounted_price_etb']:,.0f}
                    {'-'*40}
                    """
                    
                    with col2:
                        st.download_button(
                            label="📄 Download Summary Report (TXT)",
                            data=summary_report,
                            file_name=f"price_prediction_report_{timestamp}.txt",
                            width='stretch'
                        )
                    
                    # Success message with discount
                    if summary['bulk_discount_applied']:
                        st.balloons()
                        st.success(f"""
                        🎉 **Congratulations!** 🎉
                        
                        You purchased {summary['total_houses']} houses and qualified for a {DISCOUNT_RATE*100:.0f}% bulk discount!
                        - Original total: ETB {summary['original_total']:,.0f}
                        - Discount amount: ETB {summary['discount_amount']:,.0f}
                        - **Final total: ETB {summary['final_total']:,.0f}**
                        
                        You saved ETB {summary['discount_amount']:,.0f}!
                        """)
                        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please make sure your CSV file has the correct format. Download the template for reference.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Made with ❤️ for Addis Ababa Real Estate Market | Ethiopia</p>",
    unsafe_allow_html=True
)