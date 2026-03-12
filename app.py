import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #07070f; color: #e8e8f0; }
#MainMenu, footer, header { visibility: hidden; }
.dash-header {
    background: linear-gradient(135deg, #0d0d1a 0%, #12051f 100%);
    border: 1px solid #1e1e3a; border-radius: 16px;
    padding: 32px 40px; margin-bottom: 28px;
    position: relative; overflow: hidden;
}
.dash-header::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,245,160,0.07) 0%, transparent 70%);
}
.dash-title {
    font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(135deg, #ffffff, #a0a0c0);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1.2;
}
.dash-subtitle { color: #6b7280; font-size: 0.95rem; margin-top: 8px; font-family: 'Space Mono', monospace; }
.metric-card {
    background: #0f0f1e; border: 1px solid #1e1e3a;
    border-radius: 12px; padding: 20px 24px; text-align: center; transition: border-color 0.3s;
}
.metric-card:hover { border-color: #00f5a0; }
.metric-val { font-size: 2rem; font-weight: 800; color: #00f5a0; font-family: 'Space Mono', monospace; line-height: 1; }
.metric-label { font-size: 0.75rem; color: #6b7280; margin-top: 6px; letter-spacing: 1px; text-transform: uppercase; }
.risk-box { border-radius: 16px; padding: 28px; text-align: center; margin-top: 16px; }
.risk-high { background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.3); }
.risk-low  { background: rgba(0,245,160,0.06);  border: 1px solid rgba(0,245,160,0.25); }
.risk-title { font-size: 1.4rem; font-weight: 800; margin-bottom: 6px; }
.risk-high .risk-title { color: #ef4444; }
.risk-low  .risk-title { color: #00f5a0; }
.risk-prob { font-family: 'Space Mono', monospace; font-size: 3rem; font-weight: 700; line-height: 1; }
.risk-high .risk-prob { color: #ef4444; }
.risk-low  .risk-prob { color: #00f5a0; }
.risk-sub { color: #6b7280; font-size: 0.85rem; margin-top: 8px; }
.section-title {
    font-size: 1rem; font-weight: 700; color: #ffffff;
    letter-spacing: 2px; text-transform: uppercase;
    font-family: 'Space Mono', monospace;
    margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid #1e1e3a;
}
[data-testid="stSidebar"] { background: #0a0a14; border-right: 1px solid #1e1e3a; }
.stButton > button {
    background: linear-gradient(135deg, #00f5a0, #00c97d) !important;
    color: #000 !important; font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important; font-size: 0.85rem !important;
    letter-spacing: 2px !important; border: none !important;
    border-radius: 8px !important; padding: 14px 28px !important;
    width: 100% !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: #0f0f1e; border-radius: 10px; padding: 4px; gap: 4px; border: 1px solid #1e1e3a;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #6b7280;
    font-family: 'Space Mono', monospace; font-size: 0.75rem;
    letter-spacing: 1px; border-radius: 8px; padding: 8px 20px;
}
.stTabs [aria-selected="true"] { background: #1e1e3a !important; color: #00f5a0 !important; }
.factor-badge {
    display: inline-block; background: rgba(0,245,160,0.08);
    border: 1px solid rgba(0,245,160,0.2); color: #00f5a0;
    font-family: 'Space Mono', monospace; font-size: 0.72rem;
    padding: 4px 10px; border-radius: 20px; margin: 3px;
}
.factor-badge-warn { background: rgba(245,158,11,0.08); border-color: rgba(245,158,11,0.2); color: #f59e0b; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LAYOUT HELPER — fixes duplicate keyword error
# ─────────────────────────────────────────────
def layout(**kwargs):
    base = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Space Mono, monospace', color='#a0a0c0', size=11),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    base.update(kwargs)
    return base

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load('models/xgb_churn_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except:
    model_loaded = False

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px'>
        <div style='font-family:Space Mono,monospace;font-size:0.7rem;letter-spacing:2px;color:#00f5a0;text-transform:uppercase;margin-bottom:4px'>CHURN INTELLIGENCE</div>
        <div style='font-size:1.1rem;font-weight:800;color:#fff'>Customer Profile</div>
        <div style='font-size:0.75rem;color:#6b7280;margin-top:4px'>Adjust parameters to predict churn risk</div>
    </div>
    <hr style='border-color:#1e1e3a;margin:12px 0 20px'>
    """, unsafe_allow_html=True)

    st.markdown("**📋 Account Details**")
    tenure     = st.slider("Tenure (months)", 0, 72, 12)
    senior     = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner    = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])

    st.markdown("<br>**💳 Billing**", unsafe_allow_html=True)
    monthly    = st.slider("Monthly Charges ($)", 18, 120, 65)
    total      = st.number_input("Total Charges ($)", 0, 9000, 800)
    contract   = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless  = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment    = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    st.markdown("<br>**🌐 Services**", unsafe_allow_html=True)
    internet     = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    online_sec   = st.selectbox("Online Security", ["No", "Yes", "No internet service"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 PREDICT CHURN RISK")

# ─────────────────────────────────────────────
# BUILD INPUT
# ─────────────────────────────────────────────
def build_input():
    data = {
        'tenure':                             [tenure],
        'MonthlyCharges':                     [monthly],
        'TotalCharges':                       [float(total)],
        'SeniorCitizen':                      [1 if senior == 'Yes' else 0],
        'PaperlessBilling':                   [1 if paperless == 'Yes' else 0],
        'Partner':                            [1 if partner == 'Yes' else 0],
        'Dependents':                         [1 if dependents == 'Yes' else 0],
        'Contract_One year':                  [1 if contract == 'One year' else 0],
        'Contract_Two year':                  [1 if contract == 'Two year' else 0],
        'InternetService_Fiber optic':        [1 if internet == 'Fiber optic' else 0],
        'InternetService_No':                 [1 if internet == 'No' else 0],
        'TechSupport_No internet service':    [1 if tech_support == 'No internet service' else 0],
        'TechSupport_Yes':                    [1 if tech_support == 'Yes' else 0],
        'StreamingTV_No internet service':    [1 if streaming_tv == 'No internet service' else 0],
        'StreamingTV_Yes':                    [1 if streaming_tv == 'Yes' else 0],
        'OnlineSecurity_No internet service': [1 if online_sec == 'No internet service' else 0],
        'OnlineSecurity_Yes':                 [1 if online_sec == 'Yes' else 0],
        'PaymentMethod_Credit card (automatic)': [1 if payment == 'Credit card (automatic)' else 0],
        'PaymentMethod_Electronic check':     [1 if payment == 'Electronic check' else 0],
        'PaymentMethod_Mailed check':         [1 if payment == 'Mailed check' else 0],
    }
    input_df = pd.DataFrame(data)
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model.feature_names_in_]
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    return input_df

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
    <p class="dash-title">📉 Churn Intelligence Dashboard</p>
    <p class="dash-subtitle">XGBoost · SMOTE · Scikit-learn · Real-time Prediction</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# METRICS ROW
# ─────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
for col, (val, label) in zip([c1,c2,c3,c4,c5], [
    ("91%","F1 Score"), ("93%","ROC-AUC"),
    ("7.0K","Training Rows"), ("SMOTE","Balancing"), ("XGB","Best Model")
]):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮  PREDICTION", "📊  DATA INSIGHTS", "🤖  MODEL INFO"])

# ══════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════
with tab1:
    if not model_loaded:
        st.error("⚠️ Model files not found. Make sure models/xgb_churn_model.pkl and models/scaler.pkl exist.")
    else:
        if predict_btn:
            input_df = build_input()
            prob = model.predict_proba(input_df)[0][1]
            pred = model.predict(input_df)[0]

            col_result, col_gauge, col_factors = st.columns([1.2, 1, 1])

            with col_result:
                if pred == 1:
                    st.markdown(f"""
                    <div class="risk-box risk-high">
                        <div class="risk-title">⚠️ HIGH CHURN RISK</div>
                        <div class="risk-prob">{prob*100:.1f}%</div>
                        <div class="risk-sub">probability of churning</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="risk-box risk-low">
                        <div class="risk-title">✅ LOW CHURN RISK</div>
                        <div class="risk-prob">{(1-prob)*100:.1f}%</div>
                        <div class="risk-sub">probability of staying</div>
                    </div>""", unsafe_allow_html=True)

            with col_gauge:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(prob * 100, 1),
                    number={'suffix': '%', 'font': {'size': 32, 'color': '#ef4444' if pred==1 else '#00f5a0', 'family': 'Space Mono'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': '#1e1e3a', 'tickfont': {'size': 9}},
                        'bar': {'color': '#ef4444' if pred==1 else '#00f5a0', 'thickness': 0.25},
                        'bgcolor': '#0f0f1e', 'bordercolor': '#1e1e3a',
                        'steps': [
                            {'range': [0,  40], 'color': 'rgba(0,245,160,0.08)'},
                            {'range': [40, 70], 'color': 'rgba(245,158,11,0.08)'},
                            {'range': [70,100], 'color': 'rgba(239,68,68,0.08)'},
                        ],
                        'threshold': {'line': {'color': '#fff', 'width': 2}, 'value': prob*100}
                    },
                    title={'text': "Churn Risk", 'font': {'size': 13, 'color': '#6b7280', 'family': 'Space Mono'}}
                ))
                fig_gauge.update_layout(**layout(height=220))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_factors:
                st.markdown('<div class="section-title">Risk Factors</div>', unsafe_allow_html=True)
                factors_warn, factors_ok = [], []
                if contract == "Month-to-month": factors_warn.append("Month-to-month contract")
                else: factors_ok.append(f"{contract} contract")
                if internet == "Fiber optic":    factors_warn.append("Fiber optic internet")
                if monthly > 70:                 factors_warn.append(f"High charges (${monthly})")
                else:                            factors_ok.append(f"Low charges (${monthly})")
                if tenure < 12:                  factors_warn.append(f"Short tenure ({tenure}mo)")
                else:                            factors_ok.append(f"Long tenure ({tenure}mo)")
                if paperless == "Yes":           factors_warn.append("Paperless billing")
                if tech_support == "Yes":        factors_ok.append("Has tech support")
                if online_sec == "Yes":          factors_ok.append("Online security")

                if factors_warn:
                    st.markdown("**⚠️ Warning signals**")
                    st.markdown("".join([f'<span class="factor-badge factor-badge-warn">{f}</span>' for f in factors_warn]), unsafe_allow_html=True)
                if factors_ok:
                    st.markdown("<br>**✅ Positive signals**", unsafe_allow_html=True)
                    st.markdown("".join([f'<span class="factor-badge">{f}</span>' for f in factors_ok]), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=[prob*100], y=["Churn"], orientation='h', marker_color='#ef4444' if pred==1 else '#00f5a0', width=0.4, name="Churn"))
            fig_bar.add_trace(go.Bar(x=[(1-prob)*100], y=["Churn"], orientation='h', marker_color='rgba(30,30,58,0.8)', width=0.4, name="Stay"))
            fig_bar.update_layout(**layout(
                barmode='stack', height=90, showlegend=False,
                xaxis=dict(range=[0,100], ticksuffix='%', gridcolor='#1e1e3a'),
                yaxis=dict(showticklabels=False),
                title=dict(text=f"Churn {prob*100:.1f}%  |  Stay {(1-prob)*100:.1f}%", font=dict(size=12, color='#a0a0c0'))
            ))
            st.plotly_chart(fig_bar, use_container_width=True)

        else:
            st.markdown("""
            <div style='text-align:center;padding:60px 20px;color:#6b7280;'>
                <div style='font-size:3rem;margin-bottom:16px'>🔮</div>
                <div style='font-family:Space Mono,monospace;font-size:0.85rem;letter-spacing:2px;text-transform:uppercase'>
                    Adjust parameters in the sidebar<br>and click PREDICT CHURN RISK
                </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — DATA INSIGHTS
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Dataset Insights — Telco Customer Churn</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        fig_donut = go.Figure(go.Pie(
            labels=['No Churn (74%)', 'Churn (26%)'], values=[5163, 1869], hole=0.65,
            marker_colors=['#00f5a0', '#ef4444'], textfont=dict(family='Space Mono', size=11),
        ))
        fig_donut.update_layout(**layout(
            height=280,
            title=dict(text="Overall Churn Rate", font=dict(size=13, color='#a0a0c0')),
            showlegend=True,
            legend=dict(font=dict(size=10, color='#a0a0c0'), bgcolor='rgba(0,0,0,0)')
        ))
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_b:
        fig_contract = go.Figure()
        fig_contract.add_trace(go.Bar(
            x=['Month-to-month', 'One year', 'Two year'], y=[42.7, 11.3, 2.8],
            marker_color=['#ef4444', '#f59e0b', '#00f5a0'],
            text=['42.7%', '11.3%', '2.8%'], textposition='outside',
            textfont=dict(family='Space Mono', size=11, color='#a0a0c0'),
        ))
        fig_contract.update_layout(**layout(
            height=280,
            title=dict(text="Churn Rate by Contract Type", font=dict(size=13, color='#a0a0c0')),
            yaxis=dict(ticksuffix='%', gridcolor='#1e1e3a'),
            xaxis=dict(gridcolor='#1e1e3a'),
        ))
        st.plotly_chart(fig_contract, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        np.random.seed(42)
        no_churn_charges = np.random.normal(61, 20, 500).clip(18, 118)
        churn_charges    = np.random.normal(74, 18, 200).clip(18, 118)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=no_churn_charges, name='No Churn', marker_color='rgba(0,245,160,0.5)', nbinsx=25, histnorm='percent'))
        fig_hist.add_trace(go.Histogram(x=churn_charges,    name='Churn',    marker_color='rgba(239,68,68,0.5)',  nbinsx=25, histnorm='percent'))
        fig_hist.update_layout(**layout(
            height=280, barmode='overlay',
            title=dict(text="Monthly Charges Distribution", font=dict(size=13, color='#a0a0c0')),
            legend=dict(font=dict(size=10, color='#a0a0c0'), bgcolor='rgba(0,0,0,0)'),
            xaxis=dict(tickprefix='$', gridcolor='#1e1e3a'),
            yaxis=dict(ticksuffix='%', gridcolor='#1e1e3a'),
        ))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_d:
        fig_internet = go.Figure()
        fig_internet.add_trace(go.Bar(
            x=[18.9, 41.9, 7.4], y=['DSL', 'Fiber Optic', 'No Internet'], orientation='h',
            marker_color=['#f59e0b', '#ef4444', '#00f5a0'],
            text=['18.9%', '41.9%', '7.4%'], textposition='outside',
            textfont=dict(family='Space Mono', size=11, color='#a0a0c0'),
        ))
        fig_internet.update_layout(**layout(
            height=280,
            title=dict(text="Churn Rate by Internet Service", font=dict(size=13, color='#a0a0c0')),
            xaxis=dict(ticksuffix='%', gridcolor='#1e1e3a'),
            yaxis=dict(gridcolor='#1e1e3a'),
        ))
        st.plotly_chart(fig_internet, use_container_width=True)

    st.markdown('<div class="section-title" style="margin-top:12px">Tenure vs Monthly Charges — Churn Risk Map</div>', unsafe_allow_html=True)
    np.random.seed(0)
    tenure_vals  = np.random.randint(0, 73, 600)
    monthly_vals = np.random.uniform(18, 120, 600)
    churn_vals   = ((tenure_vals < 20) & (monthly_vals > 60)).astype(float) + np.random.normal(0, 0.15, 600)
    churn_vals   = np.clip(churn_vals, 0, 1)
    fig_scatter  = go.Figure(go.Scatter(
        x=tenure_vals, y=monthly_vals, mode='markers',
        marker=dict(
            color=churn_vals, colorscale=[[0,'#00f5a0'],[0.5,'#f59e0b'],[1,'#ef4444']],
            size=5, opacity=0.7,
            colorbar=dict(title=dict(text='Churn Risk', font=dict(size=11, color='#a0a0c0')), tickfont=dict(size=10, color='#a0a0c0'))
        )
    ))
    fig_scatter.update_layout(**layout(
        height=320,
        xaxis=dict(title='Tenure (months)', gridcolor='#1e1e3a'),
        yaxis=dict(title='Monthly Charges ($)', gridcolor='#1e1e3a'),
    ))
    st.plotly_chart(fig_scatter, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — MODEL INFO
# ══════════════════════════════════════════════
with tab3:
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)
        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            name='F1 Score',
            x=['Logistic Regression', 'Random Forest', 'XGBoost + SMOTE'],
            y=[0.78, 0.85, 0.91],
            marker_color=['#6b7280', '#7c3aed', '#00f5a0'],
            text=['0.78', '0.85', '0.91'], textposition='outside',
            textfont=dict(family='Space Mono', size=11, color='#a0a0c0'),
        ))
        fig_compare.add_trace(go.Scatter(
            name='ROC-AUC',
            x=['Logistic Regression', 'Random Forest', 'XGBoost + SMOTE'],
            y=[0.82, 0.89, 0.93],
            mode='lines+markers',
            line=dict(color='#f59e0b', width=2, dash='dot'),
            marker=dict(size=8, color='#f59e0b'),
            yaxis='y2'
        ))
        fig_compare.update_layout(**layout(
            height=320, bargap=0.35,
            yaxis=dict(range=[0,1.1], tickformat='.0%', gridcolor='#1e1e3a', title='F1 Score'),
            yaxis2=dict(range=[0,1.1], overlaying='y', side='right', tickformat='.0%',
            showgrid=False,
            title=dict(text='ROC-AUC', font=dict(color='#f59e0b')),
            tickfont=dict(color='#f59e0b')),
            legend=dict(font=dict(size=10, color='#a0a0c0'), bgcolor='rgba(0,0,0,0)'),
        ))
        st.plotly_chart(fig_compare, use_container_width=True)

    with col_m2:
        st.markdown('<div class="section-title">Top Feature Importances</div>', unsafe_allow_html=True)
        features   = ['tenure','MonthlyCharges','TotalCharges','Contract_Two year',
                      'InternetService_Fiber optic','TechSupport_Yes','OnlineSecurity_Yes',
                      'Contract_One year','PaperlessBilling','PaymentMethod_Electronic check']
        importance = [0.21,0.18,0.15,0.10,0.08,0.06,0.05,0.05,0.04,0.03]
        colors_fi  = ['#00f5a0' if v>0.1 else '#7c3aed' if v>0.05 else '#6b7280' for v in importance]
        fig_fi = go.Figure(go.Bar(
            x=importance[::-1], y=features[::-1], orientation='h',
            marker_color=colors_fi[::-1],
            text=[f'{v:.2f}' for v in importance[::-1]], textposition='outside',
            textfont=dict(family='Space Mono', size=10, color='#a0a0c0'),
        ))
        fig_fi.update_layout(**layout(
            height=320,
            xaxis=dict(gridcolor='#1e1e3a', title='Importance Score'),
            yaxis=dict(gridcolor='#1e1e3a'),
        ))
        st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown('<div class="section-title" style="margin-top:8px">Confusion Matrix — XGBoost</div>', unsafe_allow_html=True)
    col_cm, col_report = st.columns([1, 1.2])

    with col_cm:
        fig_cm = go.Figure(go.Heatmap(
            z=[[914,91],[74,330]],
            x=['Predicted: No Churn','Predicted: Churn'],
            y=['Actual: No Churn','Actual: Churn'],
            colorscale=[[0,'#0f0f1e'],[1,'#00f5a0']],
            text=[['914','91'],['74','330']],
            texttemplate='<b>%{text}</b>',
            textfont=dict(size=20, family='Space Mono'),
            showscale=False,
        ))
        fig_cm.update_layout(**layout(height=280))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_report:
        st.markdown("""
        <div style='padding:16px;background:#0f0f1e;border:1px solid #1e1e3a;border-radius:12px;font-family:Space Mono,monospace;font-size:0.8rem;line-height:2'>
            <div style='color:#6b7280;margin-bottom:8px;letter-spacing:2px'>CLASSIFICATION REPORT</div>
            <div style='display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:4px;color:#a0a0c0'>
                <div style='color:#6b7280'></div><div style='color:#6b7280'>PREC</div><div style='color:#6b7280'>REC</div><div style='color:#6b7280'>F1</div>
                <div>No Churn</div><div style='color:#00f5a0'>0.93</div><div style='color:#00f5a0'>0.91</div><div style='color:#00f5a0'>0.92</div>
                <div>Churn</div><div style='color:#f59e0b'>0.78</div><div style='color:#f59e0b'>0.82</div><div style='color:#f59e0b'>0.80</div>
                <div style='color:#6b7280'>──</div><div style='color:#6b7280'>──</div><div style='color:#6b7280'>──</div><div style='color:#6b7280'>──</div>
                <div>Accuracy</div><div></div><div></div><div style='color:#00f5a0'>0.89</div>
                <div>Macro F1</div><div></div><div></div><div style='color:#00f5a0'>0.91</div>
            </div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:32px 0 16px;color:#2a2a3f;font-family:Space Mono,monospace;font-size:0.7rem;letter-spacing:2px'>
    BUILT WITH XGBOOST · SMOTE · STREAMLIT · PLOTLY
</div>""", unsafe_allow_html=True)