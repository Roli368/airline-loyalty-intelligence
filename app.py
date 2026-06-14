import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib

st.set_page_config(
    page_title="Airline Loyalty Intelligence",
    page_icon="bar-chart",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    .main {
        background-color: #0E1117;
    }
    h1, h2, h3 {
        color: #FFFFFF;
    }
    .metric-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #60A5FA;
    }
    .metric-label {
        font-size: 13px;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f2937;
        border-bottom: 2px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    segments_path = os.path.join(base_dir, 'data', 'customer_segments.csv')
    churn_path = os.path.join(base_dir, 'data', 'churn_results.csv')
    
    df_seg = pd.read_csv(segments_path)
    if os.path.exists(churn_path):
        df_churn = pd.read_csv(churn_path)
        return df_seg.merge(df_churn, on="Loyalty Number", how="inner")
    return df_seg

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, 'models', 'churn_pipeline.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data
def load_raw_features():
    base_dir = os.path.dirname(__file__)
    feat_path = os.path.join(base_dir, 'data', 'raw_features.csv')
    if os.path.exists(feat_path):
        return pd.read_csv(feat_path)
    return None

df = load_data()
model = load_model()
raw_features = load_raw_features()

# Sidebar
st.sidebar.title("Altitude Analytics")
st.sidebar.markdown("---")
st.sidebar.markdown("**Executive Summary**")
st.sidebar.markdown("This dashboard provides proactive retention intelligence. It integrates real-time churn prediction, behavioral segmentation, and financial ROI to drive actionable marketing interventions.")
st.sidebar.markdown("---")

has_churn_data = "Churn_Probability" in df.columns
if has_churn_data:
    st.sidebar.metric("Total Evaluated", f"{len(df):,}")
    st.sidebar.metric("Revenue at Risk", f"${df['Revenue_At_Risk'].sum():,.0f}")

st.title("Airline Loyalty Intelligence")
st.markdown("Proactively identify churn risks, understand customer value, and deploy targeted retention strategies.")

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview", 
    "Segmentation", 
    "Smart Retention", 
    "Scenario Simulator"
])

with tab1:
    st.header("Churn Risk Overview")
    if not has_churn_data:
        st.warning("Churn data not found. Please execute the model pipeline.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        total_at_risk = len(df[df['Risk_Level'].isin(['High Risk', 'Critical Risk'])])
        revenue_at_risk = df['Revenue_At_Risk'].sum()
        avg_prob = df['Churn_Probability'].mean() * 100
        
        with col1:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(df):,}</div><div class='metric-label'>Total Members Evaluated</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{total_at_risk:,}</div><div class='metric-label'>High/Critical Risk Customers</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>${revenue_at_risk:,.0f}</div><div class='metric-label'>Total Revenue at Risk</div></div>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{avg_prob:.1f}%</div><div class='metric-label'>Avg Churn Probability</div></div>", unsafe_allow_html=True)

        st.write("---")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Customer Risk Distribution")
            risk_counts = df['Risk_Level'].value_counts().reset_index()
            risk_counts.columns = ['Risk Level', 'Count']
            risk_order = ["Critical Risk", "High Risk", "Medium Risk", "Low Risk"]
            risk_counts['Risk Level'] = pd.Categorical(risk_counts['Risk Level'], categories=risk_order, ordered=True)
            risk_counts = risk_counts.sort_values('Risk Level')
            
            fig_risk = px.pie(
                risk_counts, 
                names='Risk Level', 
                values='Count',
                hole=0.4,
                color='Risk Level',
                color_discrete_map={
                    "Critical Risk": "#ef4444",
                    "High Risk": "#f97316",
                    "Medium Risk": "#eab308",
                    "Low Risk": "#22c55e"
                },
                template='plotly_dark'
            )
            st.plotly_chart(fig_risk, use_container_width=True)
            
        with c2:
            st.subheader("Value vs Risk Profile")
            table = pd.crosstab(df['Risk_Level'], df['Value_Segment'])
            try:
                table = table.reindex(["Critical Risk", "High Risk", "Medium Risk", "Low Risk"])
                table = table[["Premium", "Gold", "Silver", "Bronze"]]
                
                fig_heatmap = px.imshow(
                    table,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="Blues",
                    template='plotly_dark'
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
            except Exception:
                st.write("Heatmap not available.")

        st.write("---")
        st.subheader("Top Priority Customers")
        top_risk = df.sort_values("Priority_Score", ascending=False).head(100)
        st.dataframe(top_risk[['Loyalty Number', 'Persona', 'Risk_Level', 'Value_Segment', 'Churn_Probability', 'Revenue_At_Risk', 'Recommended_Action']], use_container_width=True)

with tab2:
    st.header("Persona Breakdown")
    persona_counts = df['Persona'].value_counts().reset_index()
    persona_counts.columns = ['Persona', 'Count']

    fig = px.bar(
        persona_counts, 
        y='Persona', 
        x='Count', 
        orientation='h',
        color='Persona',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        template='plotly_dark'
    )
    fig.update_layout(showlegend=False, xaxis_title="Number of Members", yaxis_title="", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.write("---")
    st.header("Persona Profiles")
    
    cluster_means = df.groupby('Persona')[['Recency (Months)', 'Total Flights', 'Points Redeemed', 'Salary']].mean().reset_index()
    from sklearn.preprocessing import MinMaxScaler
    scaler_radar = MinMaxScaler()
    metrics = ['Recency (Months)', 'Total Flights', 'Points Redeemed', 'Salary']
    cluster_means[metrics] = scaler_radar.fit_transform(cluster_means[metrics])

    fig_radar = go.Figure()
    for i, row in cluster_means.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row['Recency (Months)'], row['Total Flights'], row['Points Redeemed'], row['Salary'], row['Recency (Months)']],
            theta=['Recency', 'Flights', 'Points Redeemed', 'Salary', 'Recency'],
            fill='toself',
            name=row['Persona'],
            opacity=0.6
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
        showlegend=True,
        template='plotly_dark',
        height=500,
        margin=dict(l=80, r=80, t=20, b=20)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with tab3:
    st.header("Smart Retention and ROI")
    
    if not has_churn_data:
        st.warning("Churn data not found. Please execute the model pipeline.")
    else:
        # ROI Logic
        intervention_costs = {
            "Personal Relationship Manager Call": 50,
            "Free Upgrade Voucher": 150,
            "Priority Lounge Access": 25,
            "Double Loyalty Points Campaign": 100,
            "20% Bonus Miles Offer": 50,
            "Targeted Email Campaign": 5,
            "No Action Required": 0
        }
        df_roi = df.copy()
        df_roi['Intervention_Cost'] = df_roi['Recommended_Action'].map(intervention_costs)
        # Assuming a conservative 15% success rate for interventions
        df_roi['Expected_Value_Saved'] = df_roi['Revenue_At_Risk'] * 0.15
        df_roi['Net_ROI'] = df_roi['Expected_Value_Saved'] - df_roi['Intervention_Cost']
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            selected_risk = st.selectbox("Filter by Risk Level:", ["All", "Critical Risk", "High Risk", "Medium Risk", "Low Risk"])
        with col_f2:
            selected_segment = st.selectbox("Filter by Persona:", ["All"] + list(df['Persona'].unique()))
            
        filtered_df = df_roi.copy()
        if selected_risk != "All":
            filtered_df = filtered_df[filtered_df['Risk_Level'] == selected_risk]
        if selected_segment != "All":
            filtered_df = filtered_df[filtered_df['Persona'] == selected_segment]
            
        st.write(f"Showing **{len(filtered_df):,}** customers.")
        
        if len(filtered_df) > 0:
            total_cost = filtered_df['Intervention_Cost'].sum()
            total_roi = filtered_df['Net_ROI'].sum()
            
            c_roi1, c_roi2 = st.columns(2)
            c_roi1.metric("Estimated Campaign Cost", f"${total_cost:,.0f}")
            c_roi2.metric("Projected Net ROI", f"${total_roi:,.0f}")
            
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Intervention List",
                data=csv_data,
                file_name="retention_interventions.csv",
                mime='text/csv'
            )
            
            st.write("---")
            st.subheader("Intervention Summary")
            action_counts = filtered_df['Recommended_Action'].value_counts().reset_index()
            action_counts.columns = ['Recommended Action', 'Count']
            
            fig_actions = px.bar(
                action_counts,
                x='Count',
                y='Recommended Action',
                orientation='h',
                color='Recommended Action',
                template='plotly_dark'
            )
            fig_actions.update_layout(showlegend=False, xaxis_title="Number of Interventions", yaxis_title="")
            st.plotly_chart(fig_actions, use_container_width=True)
            
            st.write("---")
            st.subheader("Customer Action Details")
            st.dataframe(filtered_df[[
                'Loyalty Number', 'Persona', 'Risk_Level', 
                'Recommended_Action', 'Intervention_Cost', 'Net_ROI'
            ]].head(100).style.format({"Intervention_Cost": "${:.2f}", "Net_ROI": "${:.2f}"}), use_container_width=True)

with tab4:
    st.header("Scenario Simulator")
    if model is None or raw_features is None:
        st.warning("Model or raw features not found. Ensure the pipeline export is successful.")
    else:
        st.markdown("Modify behavioral metrics for a specific customer to observe the real-time impact on their Churn Probability.")
        
        target_id = st.selectbox("Select Customer by Loyalty Number:", raw_features['Loyalty Number'].head(1000).values)
        customer_feat = raw_features[raw_features['Loyalty Number'] == target_id].copy()
        
        if not customer_feat.empty:
            st.write("---")
            c_sim1, c_sim2 = st.columns(2)
            
            base_prob = model.predict_proba(customer_feat.drop(columns=["Loyalty Number"]))[0][1] * 100
            
            with c_sim1:
                st.subheader("Adjust Behavioral Metrics")
                flights_3m = st.number_input("Flights (Last 3 Months)", min_value=0, max_value=100, value=int(customer_feat['Flights_Last_3M'].iloc[0]))
                points_3m = st.number_input("Points Accumulated (Last 3 Months)", min_value=0, max_value=1000000, value=int(customer_feat['Points_Last_3M'].iloc[0]))
                months_since = st.number_input("Months Since Last Flight", min_value=0.0, max_value=120.0, value=float(customer_feat['Months_Since_Last_Flight'].iloc[0]))
                
                # Apply changes
                customer_feat['Flights_Last_3M'] = flights_3m
                customer_feat['Points_Last_3M'] = points_3m
                customer_feat['Months_Since_Last_Flight'] = months_since
                
                new_prob = model.predict_proba(customer_feat.drop(columns=["Loyalty Number"]))[0][1] * 100
                prob_diff = new_prob - base_prob
                
            with c_sim2:
                st.subheader("Simulation Results")
                st.metric("Base Churn Probability", f"{base_prob:.1f}%")
                st.metric("Simulated Churn Probability", f"{new_prob:.1f}%", delta=f"{prob_diff:.1f}%", delta_color="inverse")
                
                if prob_diff < 0:
                    st.success("This intervention successfully reduces churn risk.")
                elif prob_diff > 0:
                    st.error("This behavioral change increases churn risk.")
                else:
                    st.info("No significant change in risk profile.")


