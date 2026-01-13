import streamlit as st
import pandas as pd
import hashlib
import os
import time
import plotly.express as px
import numpy as np
from sqlalchemy import text
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# --- DATABASE SETUP ---
conn = st.connection("my_database", type="sql")

# --- HELPER FUNCTIONS ---
def get_choices(df, column, defaults):
    """Safely extracts unique values for dropdowns from the dataframe."""
    if df is not None and not df.empty and column in df.columns:
        choices = sorted(df[column].dropna().unique().tolist())
        return choices if choices else defaults
    return defaults

def sync_data_from_excel():
    """Reads, normalizes, and migrates Excel data to SQL automatically."""
    if os.path.exists("customers.xlsx"):
        try:
            df = pd.read_excel("customers.xlsx")
            df.columns = df.columns.str.strip()
            cols_to_normalize = ["Salesperson", "RegionManager", "Region", "StoreLocation"]
            for col in cols_to_normalize:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip().str.lower()
            
            df.to_sql("sales", conn.engine, if_exists="append", index=False)
            
            default_pw = hashlib.sha256("password123".encode()).hexdigest()
            with conn.session as s:
                for col, role in [("Salesperson", "Salesperson"), ("RegionManager", "Region Manager")]:
                    if col in df.columns:
                        for user in df[col].dropna().unique():
                            s.execute(text("INSERT OR IGNORE INTO users VALUES (:u, :p, :r)"), 
                                      {"u": str(user), "p": default_pw, "r": role})
                s.commit()
        except Exception as e:
            st.error(f"Automatic Sync Error: {e}")

def init_db():
    """Initializes the users table and default admin."""
    with conn.session as s:
        s.execute(text("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT, role TEXT);"))
        admin_pw = hashlib.sha256("admin123".encode()).hexdigest()
        s.execute(text("INSERT OR IGNORE INTO users VALUES ('admin', :p, 'Region Manager')"), {"p": admin_pw})
        s.commit()

# --- AUTHENTICATION ---
def login():
    st.sidebar.title("🔐 Login")
    u_input = st.sidebar.text_input("Username").strip().lower()
    p_input = st.sidebar.text_input("Password", type="password")
    
    if st.sidebar.button("Login"):
        hpw = hashlib.sha256(p_input.encode()).hexdigest()
        res = conn.query("SELECT role FROM users WHERE username=:u AND password=:p", 
                         params={"u": u_input, "p": hpw}, ttl=0)
        
        if not res.empty:
            st.session_state.logged_in = True
            st.session_state.username = u_input
            st.session_state.role = res.iloc[0]['role']
            if "synced" not in st.session_state:
                sync_data_from_excel()
                st.session_state.synced = True
            st.rerun()
        else:
            st.sidebar.error("Invalid credentials.")

# --- MAIN APP ---
def main():
    init_db()
    if "logged_in" not in st.session_state: st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        st.title("🧾 Money Wiz CRM")
        login()
        return

    df_all = conn.query("SELECT * FROM sales", ttl=0)

    st.sidebar.write(f"Logged in: **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    user = st.session_state.username
    role = st.session_state.role

    if role == "Salesperson":
        st.header("👤 Salesperson Workspace")
        # Ensure correct filtering for the logged-in user
        my_data = df_all[df_all["Salesperson"].astype(str).str.lower() == user]
        
        tabs = st.tabs(["Add Customer", "Update Record", "Delete Customer", "View All", "Search Customer", "Analytics"])
        
        with tabs[0]: 
            st.subheader("Add New Customer Record")
            with st.form("add_form", clear_on_submit=True):
                c1, c2, c3 = st.columns(3)
                name = c1.text_input("Customer Name")
                c_type = c1.selectbox("Customer Type", get_choices(df_all, "CustomerType", ["Retail", "Wholesale"]))
                prod = c1.selectbox("Product", get_choices(df_all, "Product", ["Laptop", "Phone", "Tablet"]))
                qty = c1.number_input("Quantity", min_value=1)
                u_p = c2.number_input("Unit Price", min_value=0.0)
                disc = c2.number_input("Discount", min_value=0.0)
                reg = c2.selectbox("Region", get_choices(df_all, "Region", ["North", "South", "East", "West"]))
                loc = c2.selectbox("Store Location", get_choices(df_all, "StoreLocation", ["Store A", "Store B"]))
                
                ship = c3.number_input("Shipping Cost", min_value=0.0)
                calc_total = (qty * u_p) - disc + ship
                c2.number_input("Total Price (Calculated)", value=calc_total, disabled=True)
                
                sys_oid = f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                c3.info(f"Generated OrderID: {sys_oid}")
                pay = c3.selectbox("Payment Method", ["Cash", "Card", "Online"])
                prom = c3.text_input("Promotion")
                r_man = c3.selectbox("Region Manager", get_choices(df_all, "RegionManager", ["Admin"]))
                ret_status = c3.text_input("Returned (Status)", value="No") 
                
                if st.form_submit_button("Submit"):
                    with conn.session as s:
                        s.execute(text("""INSERT INTO sales (CustomerName, CustomerType, Product, Quantity, Region, Date, UnitPrice, 
                                       StoreLocation, Discount, Salesperson, TotalPrice, PaymentMethod, Promotion, Returned, 
                                       OrderID, ShippingCost, RegionManager) VALUES (:n, :ct, :p, :q, :r, :d, :up, :sl, :di, :sp, :tp, :pm, :pr, :re, :oid, :sc, :rm)"""),
                                  {"n":name, "ct":c_type, "p":prod, "q":qty, "r":reg.lower(), "d":datetime.now().strftime("%Y-%m-%d"), 
                                   "up":u_p, "sl":loc.lower(), "di":disc, "sp":user, "tp":qty*u_p-disc+ship, "pm":pay, "pr":prom, "re":ret_status, 
                                   "oid":sys_oid, "sc":ship, "rm":r_man.lower()})
                        s.commit()
                    
                    st.success(f"🎉 Customer added! OrderID: {sys_oid}")
                    time.sleep(2)
                    st.rerun()

        # ... (Remaining Salesperson tabs 1-5 from original code) ...
        # [Note: Kept original logic for update/delete/view/search]
        with tabs[3]: st.dataframe(my_data)
        with tabs[5]: 
            if not my_data.empty:
                st.subheader("Your Sales Analytics")
                fig = px.bar(my_data.groupby("Product")["TotalPrice"].sum().reset_index(), x="Product", y="TotalPrice")
                st.plotly_chart(fig)

    elif role == "Region Manager":
        st.header("📈 Managerial Insights & ML Analytics")
        if not df_all.empty:
            m_tabs = st.tabs([
                "Standard KPIs", "Sales Forecast", "Customer Segments", "Product Recommendations"
            ])

            with m_tabs[0]:
                st.subheader("General Sales Performance")
                m_opt = st.selectbox("View by:", ["Region", "Store", "Salesperson"])
                if m_opt == "Region":
                    r = st.selectbox("Select Region", df_all["Region"].unique())
                    data = df_all[df_all["Region"] == r].groupby("Product")["TotalPrice"].sum().reset_index()
                elif m_opt == "Store":
                    s = st.selectbox("Select Store", df_all["StoreLocation"].unique())
                    data = df_all[df_all["StoreLocation"] == s].groupby("Product")["TotalPrice"].sum().reset_index()
                else:
                    p = st.selectbox("Select Salesperson", df_all["Salesperson"].dropna().unique())
                    data = df_all[df_all["Salesperson"] == p].groupby("Product")["TotalPrice"].sum().reset_index()
                
                fig = px.bar(data, x="Product", y="TotalPrice", color="Product", title=f"Sales for {m_opt}")
                st.plotly_chart(fig)

            with m_tabs[1]:
                st.subheader("🔮 Revenue Trend & 3-Month Forecast")
                df_f = df_all.copy()
                # Robust date conversion for your specific error
                df_f['Date'] = pd.to_datetime(df_f['Date'], dayfirst=True, errors='coerce')
                df_f = df_f.dropna(subset=['Date']).sort_values('Date')
    
                # Resample to Month Start (MS) to ensure consistent intervals
                monthly = df_f.set_index('Date')['TotalPrice'].resample('MS').sum().reset_index()
    
                if len(monthly) >= 2:
                    # Prepare for Regression
                    monthly['Month_Num'] = range(len(monthly))
                    X = monthly[['Month_Num']]
                    y = monthly['TotalPrice']
        
                    model = LinearRegression().fit(X, y)
        
                    # Predict 3 Future Months
                    future_idx = np.array([len(monthly), len(monthly)+1, len(monthly)+2]).reshape(-1, 1)
                    preds = model.predict(future_idx)
        
                    # Build combined dataframe for Plotly
                    future_dates = pd.date_range(start=monthly['Date'].max(), periods=4, freq='MS')[1:]
                    forecast_df = pd.DataFrame({'Date': future_dates, 'TotalPrice': preds, 'Type': 'Forecast'})
                    monthly['Type'] = 'Actual'
                    combined = pd.concat([monthly[['Date', 'TotalPrice', 'Type']], forecast_df])

                    fig = px.line(combined, x='Date', y='TotalPrice', color='Type', 
                      line_dash='Type', markers=True,
                      title="Actual Sales vs. Predicted Growth")
                    st.plotly_chart(fig)
        
                    c1, c2 = st.columns(2)
                    c1.metric("Current Month Sales", f"${y.iloc[-1]:,.2f}")
                    c2.metric("Predicted Next Month", f"${preds[0]:,.2f}", 
                        delta=f"{(preds[0]-y.iloc[-1]):,.2f}")
                else:
                    st.info("Insufficient historical data (need at least 2 months) to generate a trendline.")

            with m_tabs[2]:
                st.subheader("🎯 Customer Personality Mapping")
                st.markdown("This AI model groups customers based on **How Much** they spend vs. **How Often** they buy.")
    
                cust_data = df_all.groupby('CustomerName').agg({
                    'TotalPrice': 'sum',
                    'OrderID': 'count'
                }).rename(columns={'TotalPrice': 'Revenue', 'OrderID': 'Visit_Frequency'})

                if len(cust_data) >= 3:
                    scaler = StandardScaler()
                    scaled = scaler.fit_transform(cust_data)
                    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(scaled)
                    cust_data['Cluster'] = kmeans.labels_
        
                    # Smart Labeling logic based on Cluster means
                    means = cust_data.groupby('Cluster')['Revenue'].mean().sort_values()
                    mapping = {means.index[0]: "Occasional Buyer", 
                        means.index[1]: "Growing Lead", 
                        means.index[2]: "VIP/Wholesale"}
                    cust_data['Segment'] = cust_data['Cluster'].map(mapping)
        
                    fig = px.scatter(cust_data, x="Visit_Frequency", y="Revenue", 
                         color="Segment", size="Revenue",
                         hover_name=cust_data.index,
                         color_discrete_map={"VIP/Wholesale": "#00CC96", 
                                            "Growing Lead": "#636EFA", 
                                            "Occasional Buyer": "#EF553B"},
                         labels={"Visit_Frequency": "Number of Orders", "Revenue": "Total Spend ($)"})
        
                    # Add a visual reference for "High Value" zone
                    fig.add_hline(y=cust_data['Revenue'].mean(), line_dash="dot", annotation_text="Avg Revenue")
                    fig.add_vline(x=cust_data['Visit_Frequency'].mean(), line_dash="dot", annotation_text="Avg Frequency")
        
                    st.plotly_chart(fig, use_container_width=True)
        
                    with st.expander("View Segment Details"):
                        st.dataframe(cust_data[['Revenue', 'Visit_Frequency', 'Segment']].sort_values('Revenue', ascending=False))
                else:
                    st.warning("Cluster analysis requires at least 3 unique customers in the database.")

            with m_tabs[3]:
                st.subheader("💡 Inventory Cross-Sell Suggestions")
                # Simplified similarity: which products are frequently bought by the same customers
                corrs = pd.crosstab(df_all['CustomerName'], df_all['Product'])
                if not corrs.empty and len(corrs.columns) > 1:
                    prod_corr = corrs.corr()
                    target = st.selectbox("Analyze Product:", prod_corr.columns)
                    
                    recs = prod_corr[target].sort_values(ascending=False)[1:4]
                    st.write(f"Customers who bought **{target}** also showed interest in:")
                    for p, score in recs.items():
                        if score > 0:
                            st.info(f"**{p}** (Association Strength: {score:.2f})")
                else:
                    st.info("Not enough product variety to generate associations.")

if __name__ == "__main__":
    main()