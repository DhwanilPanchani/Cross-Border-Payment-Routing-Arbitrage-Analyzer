import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
import numpy as np
from pathlib import Path

# Page config
st.set_page_config(
    page_title="FX Routing Optimizer",
    page_icon="💰",
    layout="wide"
)

# Load data from database
@st.cache_data
def load_data():
    """Load data with error handling for missing files/columns."""
    
    # Try SQLite database first
    db_path = Path("data/fintech_analytics.db")
    
    if db_path.exists():
        engine = create_engine("sqlite:///data/fintech_analytics.db")
        
        try:
            merchants = pd.read_sql("SELECT * FROM merchant_summary", engine)
        except:
            merchants = pd.DataFrame()
        
        try:
            transactions = pd.read_sql("SELECT * FROM transactions_enriched LIMIT 50000", engine)
        except:
            transactions = pd.DataFrame()
    else:
        merchants = pd.DataFrame()
        transactions = pd.DataFrame()
    
    # Load recommendations from CSV (more reliable)
    recs_path = Path("outputs/reports/routing_recommendations.csv")
    if recs_path.exists():
        recs = pd.read_csv(recs_path)
    else:
        recs = pd.DataFrame()
    
    # Load anomalies from CSV
    anomaly_path = Path("outputs/reports/merchant_anomalies.csv")
    if anomaly_path.exists():
        anomalies = pd.read_csv(anomaly_path)
    else:
        anomalies = pd.DataFrame()
    
    return merchants, recs, transactions, anomalies

merchants_df, recs_df, txns_df, anomalies_df = load_data()

# Sidebar
st.sidebar.title("💰 FX Routing Optimizer")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Merchant Analysis", "Routing Recommendations", "Anomalies"]
)

# OVERVIEW PAGE
if page == "Overview":
    st.title("Cross-Border Payment Routing Arbitrage Dashboard")
    
    # Check if we have data
    if len(merchants_df) == 0 and len(txns_df) == 0:
        st.error("⚠️ No data available. Please run the Jupyter notebook first to generate data.")
        st.info("""
        **Steps to generate data:**
        1. Open `fx_routing_arbitrage_analysis.ipynb`
        2. Run all cells sequentially (Cell 1 → Cell 30)
        3. Refresh this dashboard
        """)
    else:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            merchant_count = len(merchants_df) if len(merchants_df) > 0 else txns_df['MerchantID'].nunique() if 'MerchantID' in txns_df.columns else 0
            st.metric("Total Merchants", f"{merchant_count:,}")
        
        with col2:
            total_volume = merchants_df["TotalVolume"].sum() if "TotalVolume" in merchants_df.columns else txns_df["TransactionAmt"].sum() if "TransactionAmt" in txns_df.columns else 0
            st.metric("Total Volume", f"${total_volume/1e6:.1f}M")
        
        with col3:
            total_leakage = merchants_df["TotalFeeLeakage"].sum() if "TotalFeeLeakage" in merchants_df.columns else 0
            if total_leakage > 0:
                st.metric("Total Fee Leakage", f"${total_leakage/1e3:.0f}K", delta=f"-{total_leakage/total_volume*100:.2f}%")
            else:
                st.metric("Total Fee Leakage", "N/A")
        
        with col4:
            total_savings = recs_df["ExpectedSavings"].sum() if "ExpectedSavings" in recs_df.columns else recs_df["PotentialSavings"].sum() if "PotentialSavings" in recs_df.columns else 0
            if total_savings > 0:
                st.metric("Potential Savings", f"${total_savings/1e3:.0f}K")
            else:
                st.metric("Potential Savings", "N/A")
        
        st.markdown("---")
        
        # Visualizations
        if len(merchants_df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Fee Leakage Distribution")
                if "TotalFeeLeakage" in merchants_df.columns:
                    fig = px.histogram(
                        merchants_df,
                        x="TotalFeeLeakage",
                        nbins=50,
                        title="Distribution of Merchant Fee Leakage",
                        labels={"TotalFeeLeakage": "Fee Leakage ($)"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Fee leakage data not available")
            
            with col2:
                st.subheader("Cross-Border Transaction Rate")
                if "CrossBorderPct" in merchants_df.columns:
                    fig = px.histogram(
                        merchants_df,
                        x="CrossBorderPct",
                        nbins=30,
                        title="Cross-Border Transaction %",
                        labels={"CrossBorderPct": "Cross-Border %"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Cross-border data not available")
            
            # Top risky merchants
            st.subheader("🚨 Top 20 Merchants by Risk Score")
            if "RiskScore" in merchants_df.columns:
                display_cols = [col for col in ["MerchantID", "TotalVolume", "TotalFeeLeakage", "CrossBorderPct", "RiskScore"] if col in merchants_df.columns]
                top_risk = merchants_df.nlargest(20, "RiskScore")[display_cols]
                st.dataframe(top_risk, use_container_width=True)
            else:
                st.info("Risk score data not available")
        else:
            st.info("Merchant summary data not available. Run the notebook to generate insights.")

# MERCHANT ANALYSIS PAGE
elif page == "Merchant Analysis":
    st.title("Merchant Deep Dive")
    
    if len(txns_df) > 0 and 'MerchantID' in txns_df.columns:
        # Get top merchants by volume
        top_merchants = txns_df.groupby('MerchantID')['TransactionAmt'].sum().nlargest(50).index.tolist()
        
        selected_merchant = st.selectbox("Select Merchant", top_merchants)
        
        # Merchant metrics
        merchant_txns = txns_df[txns_df["MerchantID"] == selected_merchant]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Volume", f"${merchant_txns['TransactionAmt'].sum():,.0f}")
        with col2:
            st.metric("Transaction Count", f"{len(merchant_txns):,}")
        with col3:
            cb_pct = merchant_txns['IsCrossBorder'].mean() * 100 if 'IsCrossBorder' in merchant_txns.columns else 0
            st.metric("Cross-Border %", f"{cb_pct:.1f}%")
        
        st.markdown("---")
        
        # Transaction time series
        if 'TransactionDate' in merchant_txns.columns and 'FeeLeakageScore' in merchant_txns.columns:
            merchant_txns['TransactionDate'] = pd.to_datetime(merchant_txns['TransactionDate'])
            
            daily_metrics = merchant_txns.groupby(merchant_txns['TransactionDate'].dt.date).agg({
                'FeeLeakageScore': 'mean',
                'TransactionAmt': 'sum'
            }).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_metrics['TransactionDate'],
                y=daily_metrics['FeeLeakageScore'],
                mode='lines',
                name='Fee Leakage Score'
            ))
            fig.update_layout(title=f"Fee Leakage Over Time: {selected_merchant}")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Transaction data not available. Please run the notebook first.")

# ROUTING RECOMMENDATIONS PAGE
elif page == "Routing Recommendations":
    st.title("💡 Routing Optimization Recommendations")
    
    st.markdown("""
    **Optimization Strategy:** Route transactions through optimal card networks 
    based on currency corridors to minimize FX markup and interchange fees.
    """)
    
    if len(recs_df) > 0:
        # Determine which columns are available
        savings_col = 'ExpectedSavings' if 'ExpectedSavings' in recs_df.columns else 'PotentialSavings' if 'PotentialSavings' in recs_df.columns else None
        
        if savings_col:
            # Filter recommendations
            min_savings = st.slider("Minimum Expected Savings ($)", 0, int(recs_df[savings_col].max()), 10, 10)
            filtered_recs = recs_df[recs_df[savings_col] >= min_savings].copy()
            
            st.subheader(f"📊 {len(filtered_recs)} Recommendations (${filtered_recs[savings_col].sum():,.0f} total potential)")
            
            # Display recommendations table with available columns
            display_cols = []
            for col in ["MerchantID", "CurrentRoutingCost", "OptimalRoutingCost", "ExpectedSavings", 
                       "ExpectedSavingsPct", "ConfidenceScore", "CrossBorderVolume", "AvgCostPct", "PotentialSavings"]:
                if col in filtered_recs.columns:
                    display_cols.append(col)
            
            st.dataframe(
                filtered_recs[display_cols].sort_values(savings_col, ascending=False),
                use_container_width=True
            )
            
            # Visualization
            if 'MonthlyVolume' in filtered_recs.columns or 'CrossBorderVolume' in filtered_recs.columns:
                volume_col = 'MonthlyVolume' if 'MonthlyVolume' in filtered_recs.columns else 'CrossBorderVolume'
                
                fig = px.scatter(
                    filtered_recs,
                    x=volume_col,
                    y=savings_col,
                    size="ConfidenceScore" if "ConfidenceScore" in filtered_recs.columns else None,
                    color=savings_col,
                    hover_data=["MerchantID"],
                    title="Savings Opportunity vs. Volume",
                    labels={
                        volume_col: "Volume ($)",
                        savings_col: "Savings ($)"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No savings data available in recommendations.")
    else:
        st.warning("No recommendations available. Please run the notebook to generate recommendations.")

# ANOMALIES PAGE
elif page == "Anomalies":
    st.title("🚨 Anomaly Detection")
    
    st.markdown("""
    Merchants with detected change points or high variance in fee patterns.
    This could indicate fraud, routing changes, or pricing updates.
    """)
    
    if len(anomalies_df) > 0:
        # Determine which type of anomaly data we have
        if 'MaxChange' in anomalies_df.columns:
            # Change point detection results
            st.subheader(f"Detected {len(anomalies_df)} Merchants with Change Points")
            st.dataframe(anomalies_df.sort_values("MaxChange", ascending=False), use_container_width=True)
            
            # Visualize
            fig = px.bar(
                anomalies_df.nlargest(20, "MaxChange"),
                x="MerchantID",
                y="MaxChange",
                title="Top 20 Merchants by Maximum Fee Change",
                labels={"MaxChange": "Max Change (pp)"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif 'CV' in anomalies_df.columns:
            # Statistical outlier results
            st.subheader(f"Detected {len(anomalies_df)} Merchants with High Fee Variance")
            
            display_cols = [col for col in ['MerchantID', 'AvgFee', 'StdFee', 'CV', 'TxnCount'] if col in anomalies_df.columns]
            st.dataframe(anomalies_df[display_cols].sort_values("CV", ascending=False), use_container_width=True)
            
            # Visualize
            fig = px.scatter(
                anomalies_df,
                x='AvgFee',
                y='CV',
                size='TxnCount',
                hover_data=['MerchantID'],
                title="Fee Variance Analysis",
                labels={'AvgFee': 'Average Fee', 'CV': 'Coefficient of Variation'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(anomalies_df, use_container_width=True)
    else:
        st.warning("No anomaly data found. Please run the notebook anomaly detection cell.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**About:** This dashboard demonstrates cross-border payment routing optimization.
Built for senior data analyst portfolio.

**Data Source:** IEEE-CIS Fraud Detection + Synthetic FX Rates

**Status:** 
- ✅ Dashboard operational
- 📊 Data from notebook analysis
""")

# Add refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()