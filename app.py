import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Set page config for a premium wide layout
st.set_page_config(
    page_title="Startup Funding Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar layout for Inputs
st.sidebar.image("https://img.icons8.com/clouds/100/000000/rocket.png", width=80)
st.sidebar.header("Control Panel")

# Theme Selection
theme = st.sidebar.selectbox(
    "Theme Style 🎨", 
    ["Space Dark", "Sunset Light", "Nordic Blue"]
)

# Define color schemes based on the selected theme
if theme == "Space Dark":
    bg_color = "#0B0F19"
    sec_bg_color = "#161B26"
    text_color = "#F0F4F8"
    primary_color = "#58A6FF"
    card_border_color = "rgba(88, 166, 255, 0.3)"
    gradient_from = "#58A6FF"
    gradient_to = "#8E2DE2"
elif theme == "Sunset Light":
    bg_color = "#FFFDF9"
    sec_bg_color = "#F5EFEB"
    text_color = "#3E2723"
    primary_color = "#FF7043"
    card_border_color = "rgba(255, 112, 67, 0.2)"
    gradient_from = "#FF7043"
    gradient_to = "#FFB300"
else:  # Nordic Blue
    bg_color = "#ECEFF1"
    sec_bg_color = "#FFFFFF"
    text_color = "#263238"
    primary_color = "#00838F"
    card_border_color = "rgba(0, 131, 143, 0.2)"
    gradient_from = "#00838F"
    gradient_to = "#00ACC1"

# Inject Custom Theme-aware CSS
st.markdown(f"""
<style>
    /* Force main app background and primary text color */
    .stApp {{
        background-color: {bg_color} !important;
        color: {text_color} !important;
    }}
    
    /* Style sidebar */
    [data-testid="stSidebar"] {{
        background-color: {sec_bg_color} !important;
        border-right: 1px solid {card_border_color} !important;
    }}
    
    /* Make sure standard headings and labels match the theme text color */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp p, .stApp span, .stApp label {{
        color: {text_color} !important;
    }}
    
    /* Titles and Subtitles */
    .main-title {{
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, {gradient_from}, {gradient_to});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }}
    .subtitle {{
        font-size: 1.1rem;
        color: {text_color};
        opacity: 0.8;
        margin-bottom: 2rem;
    }}
    
    /* Metric Cards styling */
    .metric-card {{
        background-color: {sec_bg_color};
        border: 1px solid {card_border_color};
        border-left: 6px solid {primary_color};
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        text-align: center;
        transition: transform 0.2s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-2px);
    }}
    .metric-val {{
        font-size: 2rem;
        font-weight: 800;
        color: {primary_color} !important;
    }}
    .metric-label {{
        font-size: 0.85rem;
        color: {text_color};
        opacity: 0.7;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-top: 6px;
    }}
    
    /* Prediction Card styling */
    .prediction-card {{
        background-color: {sec_bg_color};
        border: 2px solid {primary_color};
        border-radius: 14px;
        padding: 2.2rem;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.06);
        margin-top: 1.5rem;
        margin-bottom: 2.5rem;
    }}
    .prediction-val {{
        font-size: 3.5rem;
        font-weight: 850;
        color: {primary_color} !important;
        margin: 1.2rem 0;
        letter-spacing: -0.5px;
    }}
    /* Sidebar button styling for visibility (enabled + disabled) */
    [data-testid="stSidebar"] .stButton>button {{
        color: {text_color} !important;
        background-color: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        box-shadow: none !important;
    }}
    [data-testid="stSidebar"] .stButton>button:hover {{
        background-color: rgba(88,166,255,0.08) !important;
        color: {text_color} !important;
    }}
    [data-testid="stSidebar"] .stButton>button:disabled,
    [data-testid="stSidebar"] .stButton>button[disabled] {{
        color: {text_color} !important;
        opacity: 0.95 !important;
        background-color: rgba(255,255,255,0.02) !important;
    }}
    /* Ensure button text/icon remains readable when Streamlit wraps content */
    [data-testid="stSidebar"] .stButton>button span,
    [data-testid="stSidebar"] .stButton>button svg {{
        fill: {text_color} !important;
        color: {text_color} !important;
    }}
</style>
""", unsafe_allow_html=True)

# Helper function to load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv("startup_funding.csv")
    
    # Keep copy for dashboard/charts before label encoding
    dashboard_df = df.copy()
    dashboard_df['Amount in USD'] = dashboard_df['Amount in USD'].astype(str).str.replace(',', '')
    dashboard_df['Amount in USD'] = pd.to_numeric(dashboard_df['Amount in USD'], errors='coerce')
    dashboard_df.dropna(subset=['Industry Vertical', 'City  Location', 'InvestmentnType', 'Amount in USD'], inplace=True)
    
    return dashboard_df

# Helper function to train the model and cache it
@st.cache_resource
def train_funding_model(df):
    industry_enc = LabelEncoder()
    city_enc = LabelEncoder()
    investment_enc = LabelEncoder()
    
    df_encoded = df.copy()
    df_encoded['Industry Vertical'] = industry_enc.fit_transform(df['Industry Vertical'])
    df_encoded['City  Location'] = city_enc.fit_transform(df['City  Location'])
    df_encoded['InvestmentnType'] = investment_enc.fit_transform(df['InvestmentnType'])
    
    X = df_encoded[['Industry Vertical', 'City  Location', 'InvestmentnType']]
    Y = df_encoded['Amount in USD']
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)
    
    return model, industry_enc, city_enc, investment_enc

# Load and train
try:
    df = load_data()
    model, industry_enc, city_enc, investment_enc = train_funding_model(df)
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data or training model: {e}")
    data_loaded = False

if data_loaded:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Adjust Parameters")
    
    # Collect sorted unique values for selectboxes
    industries = sorted(df['Industry Vertical'].unique())
    cities = sorted(df['City  Location'].unique())
    investments = sorted(df['InvestmentnType'].unique())
    
    # Standard Selectboxes (without st.form to ensure immediate reactivity if desired, but we handle via button)
    selected_industry = st.sidebar.selectbox("Industry Vertical", industries)
    selected_city = st.sidebar.selectbox("City Location", cities)
    selected_investment = st.sidebar.selectbox("Investment Type", investments)
    
    # Submit button below inputs
    submit_btn = st.sidebar.button("Predict Funding Amount 🚀", use_container_width=True)
    
    # Initialize prediction states if not present
    if 'prediction' not in st.session_state:
        # Default prediction for the first item in lists
        def_ind_val = industry_enc.transform([industries[0]])[0]
        def_city_val = city_enc.transform([cities[0]])[0]
        def_inv_val = investment_enc.transform([investments[0]])[0]
        st.session_state.prediction = model.predict([[def_ind_val, def_city_val, def_inv_val]])[0]
        st.session_state.sel_industry = industries[0]
        st.session_state.sel_city = cities[0]
        st.session_state.sel_investment = investments[0]
        st.session_state.last_updated = "Default values loaded"

    # Update prediction session state when Submit Button is clicked
    if submit_btn:
        with st.spinner("Calculating predict friendly amount..."):
            time.sleep(0.5) # Quick delay to make the spinner visual and interactive
            ind_val = industry_enc.transform([selected_industry])[0]
            city_val = city_enc.transform([selected_city])[0]
            inv_val = investment_enc.transform([selected_investment])[0]
            
            st.session_state.prediction = model.predict([[ind_val, city_val, inv_val]])[0]
            st.session_state.sel_industry = selected_industry
            st.session_state.sel_city = selected_city
            st.session_state.sel_investment = selected_investment
            st.session_state.last_updated = f"Updated just now ({time.strftime('%X')})"
        st.toast("Funding prediction recalculated!", icon="💰")

    # Main content header
    st.markdown('<div class="main-title">🚀 Startup Funding Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">An AI-powered dashboard estimating venture funding based on industry, location, and investment structures.</div>', unsafe_allow_html=True)
    
    # Metrics columns (R2 score removed, now 3 columns)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-val">{len(df):,}</div><div class="metric-label">Startups Analysed</div></div>', 
            unsafe_allow_html=True
        )
    with m2:
        avg_fund = df['Amount in USD'].mean()
        st.markdown(
            f'<div class="metric-card"><div class="metric-val">${avg_fund:,.2f}</div><div class="metric-label">Average Funding</div></div>', 
            unsafe_allow_html=True
        )
    with m3:
        max_fund = df['Amount in USD'].max()
        st.markdown(
            f'<div class="metric-card"><div class="metric-val">${max_fund:,.2f}</div><div class="metric-label">Max Funding</div></div>', 
            unsafe_allow_html=True
        )
    
    # Prediction logic display
    st.subheader("💡 Prediction Result")
    
    st.markdown(f"""
    <div class="prediction-card">
        <h3>Estimated Funding Amount for your Startup:</h3>
        <div class="prediction-val">${st.session_state.prediction:,.2f} USD</div>
        <p>Based on a <strong>{st.session_state.sel_industry}</strong> startup in <strong>{st.session_state.sel_city}</strong> raising via <strong>{st.session_state.sel_investment}</strong>.</p>
        <span style="font-size: 0.8rem; opacity: 0.6;">Status: {st.session_state.last_updated}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Visualizations Section
    st.subheader("📊 Data & Market Insights")
    
    tab1, tab2, tab3 = st.tabs(["Top Cities", "Top Industries", "Investment Distribution"])
    
    # Set text colors inside plotly dynamically depending on theme
    plotly_text_color = "#263238" if theme == "Sunset Light" else "#F0F4F8"
    
    with tab1:
        # Top 10 cities by total funding
        city_funding = df.groupby('City  Location')['Amount in USD'].sum().reset_index()
        city_funding = city_funding.sort_values(by='Amount in USD', ascending=False).head(10)
        
        try:
            import plotly.express as px
            fig = px.bar(
                city_funding, 
                x='City  Location', 
                y='Amount in USD', 
                title='Top 10 Cities by Total Funding (USD)',
                labels={'Amount in USD': 'Total Funding (USD)', 'City  Location': 'City'},
                color='Amount in USD',
                color_continuous_scale='Bluered'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=plotly_text_color)
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.bar_chart(city_funding.set_index('City  Location'))
            
    with tab2:
        # Top 10 industries by average funding
        ind_funding = df.groupby('Industry Vertical')['Amount in USD'].mean().reset_index()
        ind_funding = ind_funding.sort_values(by='Amount in USD', ascending=False).head(10)
        
        try:
            import plotly.express as px
            fig = px.bar(
                ind_funding, 
                x='Industry Vertical', 
                y='Amount in USD', 
                title='Top 10 Industries by Average Funding (USD)',
                labels={'Amount in USD': 'Average Funding (USD)', 'Industry Vertical': 'Industry'},
                color='Amount in USD',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=plotly_text_color)
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.bar_chart(ind_funding.set_index('Industry Vertical'))
            
    with tab3:
        # Distribution of investment types
        inv_counts = df['InvestmentnType'].value_counts().reset_index()
        inv_counts.columns = ['Investment Type', 'Count']
        inv_counts = inv_counts.head(8) # Top 8 types
        
        try:
            import plotly.express as px
            fig = px.pie(
                inv_counts, 
                values='Count', 
                names='Investment Type', 
                title='Investment Type Distribution (Top 8)',
                hole=0.4
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=plotly_text_color)
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.write(inv_counts)
else:
    st.warning("Could not load dataset. Make sure 'startup_funding.csv' is present in the workspace.")
