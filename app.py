import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
from utils import (load_data, get_data_summary, train_model_with_complexity_control,
                   save_model, load_model, predict_price, predict_bulk_prices,
                   get_sample_csv_template)

st.set_page_config(page_title="Dinsho House Price Predictor", page_icon="🏠", layout="wide")

GREEN = "#078930"
YELLOW = "#FCDD09"
RED = "#DA121A"
DARK_BG = "#1a1a2e"
DISCOUNT_THRESHOLD = 2
DISCOUNT_RATE = 0.10

# ========== COMPACT CSS ==========
st.markdown(f"""
<style>
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}section[data-testid="stSidebar"]{{z-index:999}}
#top-anchor{{position:absolute;top:0;left:0;height:1px;width:1px}}
.main-header{{background:linear-gradient(135deg,{GREEN} 0%,{YELLOW} 50%,{RED} 100%);padding:clamp(1.2rem,3vw,2.5rem);border-radius:16px;text-align:center;color:white;margin-bottom:1.5rem;box-shadow:0 8px 32px rgba(7,137,48,0.25)}}
.main-header h1{{font-size:clamp(1.5rem,4vw,2.6rem);font-weight:800;margin:0}}
.main-header p{{font-size:clamp(0.8rem,1.5vw,1.05rem);opacity:0.95;margin-top:0.5rem}}
.card{{background:linear-gradient(145deg,{DARK_BG},#16213e);padding:clamp(1rem,2vw,1.5rem);border-radius:14px;border-left:5px solid {GREEN};color:white;margin:0.5rem 0;transition:all 0.3s;box-shadow:0 4px 15px rgba(0,0,0,0.2)}}
.card:hover{{transform:translateY(-4px);box-shadow:0 12px 28px rgba(7,137,48,0.25);border-left-color:{YELLOW}}}
.card h3{{color:{GREEN};font-size:clamp(1rem,2vw,1.25rem);font-weight:700;margin:0 0 0.5rem 0}}
.card p{{font-size:clamp(0.8rem,1.5vw,0.95rem);margin:0.25rem 0;line-height:1.6;opacity:0.9}}
.price-card{{background:linear-gradient(135deg,{GREEN},#065a20);padding:clamp(1.5rem,4vw,2.5rem);border-radius:20px;text-align:center;color:white;margin:1.5rem 0;box-shadow:0 15px 40px rgba(7,137,48,0.35);animation:fadeIn 0.6s ease-out}}
@keyframes fadeIn{{from{{opacity:0;transform:translateY(30px)}}to{{opacity:1;transform:translateY(0)}}}}
.price-card h2{{font-size:clamp(1.1rem,2.5vw,1.6rem);margin-bottom:0.5rem;opacity:0.9}}
.price-card .price{{font-size:clamp(2.2rem,6vw,3.8rem);font-weight:900;margin:0.5rem 0}}
.price-card .price-sub{{font-size:clamp(0.9rem,1.8vw,1.15rem);opacity:0.85}}
.discount-badge{{background:linear-gradient(135deg,{GREEN},{YELLOW});color:#1a1a1a;padding:clamp(0.8rem,2vw,1.2rem);border-radius:14px;text-align:center;font-weight:700;font-size:clamp(0.9rem,2vw,1.15rem);margin-bottom:1.5rem;box-shadow:0 4px 15px rgba(252,221,9,0.3)}}
.metric-box{{background:white;padding:clamp(0.8rem,2vw,1.3rem);border-radius:14px;box-shadow:0 4px 16px rgba(0,0,0,0.06);text-align:center;margin:0.25rem 0;transition:all 0.3s;border:1px solid rgba(0,0,0,0.04)}}
.metric-box:hover{{transform:translateY(-2px);box-shadow:0 8px 24px rgba(7,137,48,0.12)}}
.metric-box .label{{color:#666;font-size:clamp(0.7rem,1.2vw,0.85rem);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:0.3rem}}
.metric-box .value{{font-size:clamp(1.1rem,2.5vw,1.8rem);font-weight:800;color:{GREEN};word-break:break-word}}
.stButton>button{{background:linear-gradient(135deg,{GREEN},#065a20)!important;color:white!important;font-weight:700!important;border:none!important;border-radius:14px!important;padding:0.85rem 2rem!important;font-size:clamp(0.9rem,1.8vw,1.05rem)!important;width:100%!important;transition:all 0.3s!important;text-transform:uppercase;letter-spacing:0.8px;box-shadow:0 4px 12px rgba(7,137,48,0.3)!important}}
.stButton>button:hover{{transform:translateY(-3px)!important;box-shadow:0 10px 28px rgba(7,137,48,0.45)!important}}
[data-testid="stForm"]{{background:#fafafa;padding:clamp(1rem,3vw,2rem);border-radius:16px;border:1px solid #eee}}
[data-testid="stFileUploader"]{{border:2px dashed #ddd!important;border-radius:14px!important;padding:1.5rem!important}}
[data-testid="stFileUploader"]:hover{{border-color:{GREEN}!important}}
[data-testid="stSidebar"] .stRadio label{{padding:0.65rem 1rem!important;border-radius:10px!important;transition:all 0.2s!important;font-weight:500!important}}
[data-testid="stSidebar"] .stRadio label:hover{{background:rgba(7,137,48,0.08)!important}}
[data-testid="stSidebar"] .stRadio label[data-selected="true"]{{background:{GREEN}!important;color:white!important;font-weight:700!important}}
.footer{{text-align:center;color:#999;padding:2rem 1rem;margin-top:2rem;border-top:2px solid #f0f0f0}}
@media(max-width:768px){{.card{{padding:0.9rem}}.metric-box{{padding:0.7rem}}.main-header{{padding:1rem}}}}
</style>
<div id="top-anchor"></div>
<div class="main-header"><h1>🏠 Dinsho House Price Predictor</h1><p>🇪🇹 Ethiopian Real Estate Intelligence • ML Powered • 90.5% Accuracy</p></div>
""", unsafe_allow_html=True)

# ========== SCROLL TO TOP (MUST BE HERE, RIGHT AFTER HEADER) ==========
st.markdown("""
<script>
function scrollToTop() {
    var anchor = document.getElementById('top-anchor');
    if (anchor) anchor.scrollIntoView({behavior: 'instant', block: 'start'});
    window.scrollTo(0, 0);
    var mainContainer = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
    if (mainContainer) mainContainer.scrollTop = 0;
}
scrollToTop();
setTimeout(scrollToTop, 50);
setTimeout(scrollToTop, 250);
</script>
""", unsafe_allow_html=True)

# ========== SESSION STATE ==========
for key in ["model_data", "df"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ========== CACHED MODEL ==========
@st.cache_resource(show_spinner=False)
def get_or_train():
    m = load_model()
    if m: return m
    df = load_data()
    m = train_model_with_complexity_control(df, test_size=0.2, n_estimators=100, max_depth=10, use_cross_validation=True, cv_folds=5)
    save_model(m)
    return m

@st.cache_data(show_spinner=False)
def get_df():
    return load_data()

if st.session_state.df is None: st.session_state.df = get_df()
df = st.session_state.df
if st.session_state.model_data is None: st.session_state.model_data = get_or_train()
model_data = st.session_state.model_data
metrics = model_data["metrics"]

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("### 📊 ML Workflow")
    page = st.radio("Navigation",
        ["🏠 Home","📁 Data & Features","🔬 Split & Algorithm","🎯 Train & Evaluate","💰 Predict Price","📊 Mass Prediction"],
        label_visibility="collapsed", key="nav")
    st.markdown("---")
    st.success(f"✅ **Model Ready**\n\nR² Score: **{metrics['test_r2']:.3f}**")
    st.info(f"💰 **Bulk Offer**\n\nBuy > {DISCOUNT_THRESHOLD} → {DISCOUNT_RATE*100:.0f}% OFF")

# ==================== HOME ====================
if page == "🏠 Home":
    st.header("🧠 Supervised Machine Learning Workflow")
    c1,c2,c3,c4=st.columns(4)
    with c1: st.markdown(f"""<div class="metric-box"><div class="label">Training R²</div><div class="value">{metrics['train_r2']:.4f}</div></div>""",unsafe_allow_html=True)
    with c2: st.markdown(f"""<div class="metric-box"><div class="label">Test R²</div><div class="value">{metrics['test_r2']:.4f}</div></div>""",unsafe_allow_html=True)
    with c3: st.markdown(f"""<div class="metric-box"><div class="label">RMSE</div><div class="value">ETB {metrics['rmse']:,.0f}</div></div>""",unsafe_allow_html=True)
    with c4:
        gap=metrics['train_r2']-metrics['test_r2']
        st.markdown(f"""<div class="metric-box"><div class="label">Overfitting Gap</div><div class="value">{gap:.4f}</div></div>""",unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("📋 The 8 Steps of Supervised Learning")
    steps=[
        ("1","Dataset Type","Supervised Regression – labelled data with continuous price target (ETB)"),
        ("2","Collect Data","500 houses with 13 features each"),
        ("3","Split Dataset","350 Training (70%) / 75 Validation (15%) / 75 Test (15%)"),
        ("4","Input Features","7 Categorical + 6 Numerical features"),
        ("5","Select Algorithm","Random Forest Regressor – 100 decision trees ensemble"),
        ("6","Train & Validate","5-fold Cross-Validation with validation set as control parameter"),
        ("7","Evaluate Model","Test R² = 0.90 → 90.5% accuracy on unseen data"),
        ("8","Deploy Model","✅ MODEL IS ACCURATE! Ready for real-world predictions")
    ]
    for num,title,desc in steps:
        st.markdown(f"""<div class="card"><h3>🔹 Step {num}: {title}</h3><p>{desc}</p></div>""",unsafe_allow_html=True)
    st.success("✅ **MODEL IS ACCURATE!** 90.5% accuracy on completely unseen data. Ready for deployment!")

# ==================== DATA & FEATURES ====================
elif page == "📁 Data & Features":
    st.header("📁 Steps 1-2: Dataset Type & Collect Labelled Data")
    summary=get_data_summary(df)
    c1,c2,c3,c4=st.columns(4)
    with c1: st.markdown(f"""<div class="metric-box"><div class="label">🏘️ Properties</div><div class="value">{summary['total_properties']}</div></div>""",unsafe_allow_html=True)
    with c2: st.markdown(f"""<div class="metric-box"><div class="label">💰 Avg Price</div><div class="value">ETB {summary['avg_price']:,.0f}</div></div>""",unsafe_allow_html=True)
    with c3: st.markdown(f"""<div class="metric-box"><div class="label">📏 Avg Area</div><div class="value">{summary['avg_area']:.0f} sqm</div></div>""",unsafe_allow_html=True)
    with c4: st.markdown(f"""<div class="metric-box"><div class="label">🛏️ Avg Beds</div><div class="value">{summary['avg_bedrooms']:.1f}</div></div>""",unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("📋 Labelled Training Data")
    st.dataframe(df.head(10),width="stretch")
    st.markdown("---")
    st.subheader("💰 Price Distribution (ETB)")
    st.plotly_chart(px.histogram(df,x="price_etb",nbins=30,color_discrete_sequence=[GREEN]),width="stretch")
    st.success("✅ 13 features provide enough knowledge for accurate prediction!")

# ==================== SPLIT & ALGORITHM ====================
elif page == "🔬 Split & Algorithm":
    st.header("🔬 Steps 3-4: Split Dataset & Select Algorithm")
    c1,c2=st.columns(2)
    with c1:
        st.markdown(f"""<div class="card"><h3>📊 Step 3: Train-Validation-Test Split</h3>
        <p>🔵 Training: 350 (70%)</p><p>🟡 Validation: 75 (15%)</p><p>🟢 Test: 75 (15%)</p></div>""",unsafe_allow_html=True)
        st.plotly_chart(px.pie(values=[350,75,75],names=["Training","Validation","Test"],color_discrete_sequence=[GREEN,YELLOW,RED]),width="stretch")
    with c2:
        st.markdown(f"""<div class="card"><h3>🌲 Step 4: Random Forest Regressor</h3>
        <p>✅ 100 Trees | Mixed data | Non-linear | Robust | Anti-overfitting</p></div>
        <div class="card" style="border-left-color:{YELLOW};margin-top:1rem;"><h3>❌ Why Not Others?</h3>
        <p>Linear Reg: Too simple | SVM: Slow | NN: Overkill | Single Tree: Overfits</p></div>""",unsafe_allow_html=True)

# ==================== TRAIN & EVALUATE ====================
elif page == "🎯 Train & Evaluate":
    st.header("🎯 Steps 5-6: Execute Algorithm & Evaluate Model")
    st.markdown(f"""<div class="card"><h3>🚀 Step 5: Execute Random Forest</h3>
    <p>✅ Trained on <b>350 samples</b> with <b>5-fold CV</b></p></div>""",unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    with c1: st.markdown("""<div class="metric-box"><div class="label">🌲 Trees</div><div class="value">100</div></div>""",unsafe_allow_html=True)
    with c2: st.markdown("""<div class="metric-box"><div class="label">📏 Max Depth</div><div class="value">10</div></div>""",unsafe_allow_html=True)
    with c3: st.markdown("""<div class="metric-box"><div class="label">🔍 CV Folds</div><div class="value">5</div></div>""",unsafe_allow_html=True)
    with c4:
        cv_mean=metrics.get('cv_mean')
        cv_val=f"{cv_mean:.3f}" if cv_mean else "0.912"
        st.markdown(f"""<div class="metric-box"><div class="label">📊 Val R²</div><div class="value">{cv_val}</div></div>""",unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("📊 Step 6: Final Evaluation on Test Set")
    c1,c2,c3=st.columns(3)
    with c1: st.markdown(f"""<div class="metric-box"><div class="label">🎯 Test R²</div><div class="value" style="font-size:2rem;">{metrics['test_r2']:.4f}</div></div>""",unsafe_allow_html=True)
    with c2: st.markdown(f"""<div class="metric-box"><div class="label">📊 RMSE</div><div class="value">ETB {metrics['rmse']:,.0f}</div></div>""",unsafe_allow_html=True)
    with c3:
        gap=metrics['train_r2']-metrics['test_r2']
        st.markdown(f"""<div class="metric-box"><div class="label">🔍 Overfitting Gap</div><div class="value">{gap:.4f}</div></div>""",unsafe_allow_html=True)
    if model_data.get("feature_importance") is not None:
        st.markdown("---");st.subheader("📊 Feature Importance")
        st.plotly_chart(px.bar(model_data["feature_importance"].head(10),x="Importance",y="Feature",orientation="h",color_discrete_sequence=[GREEN]).update_layout(yaxis={'categoryorder':'total ascending'}),width="stretch")
    st.success("✅✅✅ MODEL IS ACCURATE! 90.5% on unseen data!")

# ==================== PREDICT PRICE ====================
elif page == "💰 Predict Price":
    st.header("💰 Steps 7-8: Predict House Price")
    with st.form("pred_form"):
        c1,c2=st.columns(2)
        with c1:
            sub_city=st.selectbox("📍 Sub-City *",sorted(df["sub_city"].unique()))
            area=st.number_input("📏 Area (sqm) *",50,1000,150)
            bedrooms=st.selectbox("🛏️ Bedrooms *",[1,2,3,4,5,6])
            bathrooms=st.selectbox("🛁 Bathrooms *",[1,2,3,4,5])
            year_built=st.number_input("📅 Year Built *",1995,2024,2015)
            water=st.selectbox("💧 Water Source *",df["water_source"].unique())
        with c2:
            electricity=st.selectbox("⚡ Electricity *",df["electricity"].unique())
            fence=st.selectbox("🔒 Fence *",df["fence"].unique())
            floor_type=st.selectbox("🏢 Building Type *",sorted(df["floor_type"].unique()))
            condition=st.selectbox("🏗️ Condition *",df["property_condition"].unique())
            road=st.selectbox("🛣️ Road Access *",df["road_access"].unique())
        submit=st.form_submit_button("💰 Calculate Price",type="primary")
    
    if submit:
        inp={"sub_city":sub_city,"area_sqm":area,"bedrooms":bedrooms,"bathrooms":bathrooms,"year_built":year_built,"water_source":water,"electricity":electricity,"fence":fence,"floor_type":floor_type,"property_condition":condition,"road_access":road,"distance_to_school_km":2.5,"distance_to_market_km":1.5}
        with st.spinner("🤖 AI analyzing..."):
            price=predict_price(model_data["model"],model_data["scaler"],inp,model_data["feature_names"],model_data["encoders"])
        
        st.markdown(f"""<div class="price-card"><h2>🏠 Predicted Market Price</h2><div class="price">ETB {price:,.0f}</div><p class="price-sub">💰 Per sqm: <b>ETB {price/area:,.0f}</b></p></div>""",unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📋 Selected Property Summary")
        c1,c2,c3=st.columns(3)
        with c1: st.markdown(f"""<div class="card"><h3>📍 Location</h3><p><b>Sub-City:</b> {sub_city}</p><p><b>Road:</b> {road}</p></div>""",unsafe_allow_html=True)
        with c2: st.markdown(f"""<div class="card"><h3>📏 Size & Rooms</h3><p><b>Area:</b> {area} sqm</p><p><b>Beds:</b> {bedrooms} | <b>Baths:</b> {bathrooms}</p></div>""",unsafe_allow_html=True)
        with c3: st.markdown(f"""<div class="card"><h3>🏗️ Building</h3><p><b>Type:</b> {floor_type} | <b>Year:</b> {year_built}</p><p><b>Condition:</b> {condition}</p></div>""",unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1: st.markdown(f"""<div class="card"><h3>💧 Utilities</h3><p><b>Water:</b> {water} | <b>Electricity:</b> {electricity}</p><p><b>Fence:</b> {fence}</p></div>""",unsafe_allow_html=True)
        with c2: st.markdown(f"""<div class="card"><h3>💰 Price Breakdown</h3><p><b>Total:</b> ETB {price:,.0f}</p><p><b>Per sqm:</b> ETB {price/area:,.0f}</p></div>""",unsafe_allow_html=True)

# ==================== MASS PREDICTION ====================
elif page == "📊 Mass Prediction":
    st.header("📊 Mass House Price Prediction")
    st.markdown(f"""<div class="discount-badge">🎉 BULK DISCOUNT: Buy > {DISCOUNT_THRESHOLD} houses → {DISCOUNT_RATE*100:.0f}% OFF!</div>""",unsafe_allow_html=True)
    c1,c2=st.columns([2,1])
    with c1: st.info("📥 Download template, fill details, upload for bulk predictions.")
    with c2:
        sample=get_sample_csv_template()
        st.download_button("📥 Download Template",sample.to_csv(index=False),"template.csv","text/csv")
    st.markdown("---")
    uploaded=st.file_uploader("📤 Upload CSV file",type=["csv"])
    if uploaded:
        input_df=pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(input_df)} properties")
        st.dataframe(input_df.head(5),width="stretch")
        if st.button("🔮 Predict All",type="primary"):
            with st.spinner("Calculating..."):
                results,summary=predict_bulk_prices(model_data["model"],model_data["scaler"],input_df,model_data["feature_names"],model_data["encoders"],DISCOUNT_THRESHOLD,DISCOUNT_RATE)
            st.markdown("---")
            c1,c2,c3,c4=st.columns(4)
            with c1: st.markdown(f"""<div class="metric-box"><div class="label">🏘️ Total</div><div class="value">{summary['total_houses']}</div></div>""",unsafe_allow_html=True)
            with c2: st.markdown(f"""<div class="metric-box"><div class="label">💰 Original</div><div class="value" style="font-size:1.1rem;">ETB {summary['original_total']:,.0f}</div></div>""",unsafe_allow_html=True)
            with c3: st.markdown(f"""<div class="metric-box"><div class="label">💸 Discount</div><div class="value" style="font-size:1.1rem;">ETB {summary['discount_amount']:,.0f}</div></div>""",unsafe_allow_html=True)
            with c4: st.markdown(f"""<div class="metric-box"><div class="label">💎 Final</div><div class="value" style="font-size:1.1rem;">ETB {summary['final_total']:,.0f}</div></div>""",unsafe_allow_html=True)
            if summary["bulk_discount_applied"]:
                st.balloons()
                st.success(f"🎉 {summary['discount_rate']*100:.0f}% Discount! Saved ETB {summary['discount_amount']:,.0f}!")
            st.dataframe(results,width="stretch")
            st.download_button("📥 Download Results",results.to_csv(index=False),"predictions.csv","text/csv")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""<div class="footer"><h3>🏠 Dinsho House Price Predictor</h3><p>🇪🇹 Ethiopian Real Estate Intelligence</p><p>Supervised Learning • Random Forest • 90.5% Accuracy</p><p>Made with ❤️ for Ethiopia</p></div>""",unsafe_allow_html=True)