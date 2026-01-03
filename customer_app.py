# customer_segmentation_dashboard.py
# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import datetime
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide", initial_sidebar_state="expanded")
st.title("🧠 Customer Segmentation Dashboard — Insights & Robust Upload")
st.markdown("Enter inputs in the sidebar. Upload a dataset to compare cluster averages (app will assign clusters if missing).")

# -----------------------------
# Load model & scaler
# -----------------------------
@st.cache_resource
def load_model_and_scaler(kmodel_path="kmeans_model.pkl", scaler_path="scaler.pkl"):
    try:
        with open(kmodel_path, "rb") as f:
            kmeans_model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e.filename}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model/scaler: {e}")
        st.stop()
    return kmeans_model, scaler

kmeans_model, scaler = load_model_and_scaler()

# derive feature name lists
model_feature_names = list(getattr(kmeans_model, "feature_names_in_", [])) or None
scaler_feature_names = list(getattr(scaler, "feature_names_in_", [])) or None

# fallback feature list (edit if your training used different columns)
fallback_features = [
    'Year_Birth','Income','Kidhome','Teenhome','Recency','MntWines','MntFruits','MntMeatProducts',
    'MntFishProducts','MntSweetProducts','MntGoldProds','NumDealsPurchases','NumWebPurchases',
    'NumCatalogPurchases','NumStorePurchases','NumWebVisitsMonth','AcceptedCmp3','AcceptedCmp4',
    'AcceptedCmp5','AcceptedCmp1','AcceptedCmp2','Complain','Total Spending','Response',
    'Education_Basic','Education_Graduation','Education_Master','Education_PhD','Education_2n Cycle',
    'Marital_Status_Alone','Marital_Status_Divorced','Marital_Status_Married','Marital_Status_Single',
    'Marital_Status_Together','Marital_Status_Widow','Marital_Status_YOLO','Age','Total Children','Customer_Tenure'
]

if model_feature_names is None:
    st.warning("Model missing feature_names_in_. Falling back to default feature list. Please verify matches training columns.")
    model_feature_names = fallback_features.copy()

if scaler_feature_names is None:
    st.info("Scaler missing feature_names_in_. Assuming scaler used the same feature set as the model.")
    scaler_feature_names = model_feature_names.copy()

# Debug expander to inspect feature names if needed
with st.expander("Debug: model & scaler feature names"):
    st.write("Model features (count):", len(model_feature_names))
    st.write(model_feature_names)
    st.write("Scaler features (count):", len(scaler_feature_names))
    st.write(scaler_feature_names)
    st.write("KMeans cluster count:", getattr(kmeans_model, "n_clusters", "unknown"))

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("🧾 Customer Input Details")

# Personal & Demographics
st.sidebar.subheader("Personal & Demographics")
year_birth = st.sidebar.number_input('Year of Birth', min_value=1900, max_value=datetime.date.today().year, value=1980)
income = st.sidebar.number_input('Income ($)', min_value=0.0, max_value=1_000_000.0, value=50000.0, step=100.0)
kidhome = st.sidebar.number_input('Number of Kids', 0, 10, 0)
teenhome = st.sidebar.number_input('Number of Teens', 0, 10, 0)
recency = st.sidebar.slider('Recency (Days since last purchase)', min_value=0, max_value=100, value=50)
education = st.sidebar.selectbox('Education', ['Graduation', 'Master', 'PhD', 'Basic', '2n Cycle'])
marital_status = st.sidebar.selectbox('Marital Status', ['Married', 'Together', 'Single', 'Divorced', 'Alone', 'Widow', 'YOLO', 'Absurd'])

# Spending
st.sidebar.subheader("Spending")
mnt_wines = st.sidebar.number_input('MntWines ($)', 0.0, 100000.0, 100.0)
mnt_meat_products = st.sidebar.number_input('MntMeatProducts ($)', 0.0, 100000.0, 50.0)
mnt_fruits = st.sidebar.number_input('MntFruits ($)', 0.0, 100000.0, 10.0)
mnt_fish_products = st.sidebar.number_input('MntFishProducts ($)', 0.0, 100000.0, 20.0)
mnt_sweet_products = st.sidebar.number_input('MntSweetProducts ($)', 0.0, 100000.0, 10.0)
mnt_gold_prods = st.sidebar.number_input('MntGoldProds ($)', 0.0, 100000.0, 10.0)

# Purchases & Campaigns
st.sidebar.subheader("Purchase Channels & Campaigns")
num_deals_purchases = st.sidebar.number_input('NumDealsPurchases', 0, 200, 1)
num_web_purchases = st.sidebar.number_input('NumWebPurchases', 0, 200, 1)
num_catalog_purchases = st.sidebar.number_input('NumCatalogPurchases', 0, 200, 1)
num_store_purchases = st.sidebar.number_input('NumStorePurchases', 0, 200, 3)
num_web_visits_month = st.sidebar.number_input('NumWebVisitsMonth', 0, 200, 5)

accepted_cmp1 = st.sidebar.selectbox('AcceptedCmp1', [0, 1], index=0)
accepted_cmp2 = st.sidebar.selectbox('AcceptedCmp2', [0, 1], index=0)
accepted_cmp3 = st.sidebar.selectbox('AcceptedCmp3', [0, 1], index=0)
accepted_cmp4 = st.sidebar.selectbox('AcceptedCmp4', [0, 1], index=0)
accepted_cmp5 = st.sidebar.selectbox('AcceptedCmp5', [0, 1], index=0)
complain = st.sidebar.selectbox('Complain', [0, 1], index=0)

response_input = 0
if 'Response' in model_feature_names:
    response_input = st.sidebar.selectbox('Response (Last Campaign)', [0, 1], index=0)

dt_customer_input = st.sidebar.date_input("Enrollment date (Dt_Customer)", value=datetime.date(2021, 1, 1))

# -----------------------------
# Helper: build aligned feature dict for a single customer
# -----------------------------
def build_single_customer_dict():
    today = datetime.date.today()
    age = today.year - year_birth
    total_children = kidhome + teenhome
    customer_tenure = (today - dt_customer_input).days
    total_spending = float(mnt_wines + mnt_fruits + mnt_meat_products + mnt_fish_products + mnt_sweet_products + mnt_gold_prods)

    final = {c: 0.0 for c in model_feature_names}

    numeric_map = {
        'Year_Birth': year_birth, 'Income': income, 'Kidhome': kidhome, 'Teenhome': teenhome,
        'Recency': recency, 'MntWines': mnt_wines, 'MntFruits': mnt_fruits, 'MntMeatProducts': mnt_meat_products,
        'MntFishProducts': mnt_fish_products, 'MntSweetProducts': mnt_sweet_products, 'MntGoldProds': mnt_gold_prods,
        'NumDealsPurchases': num_deals_purchases, 'NumWebPurchases': num_web_purchases, 'NumCatalogPurchases': num_catalog_purchases,
        'NumStorePurchases': num_store_purchases, 'NumWebVisitsMonth': num_web_visits_month,
        'AcceptedCmp3': accepted_cmp3, 'AcceptedCmp4': accepted_cmp4, 'AcceptedCmp5': accepted_cmp5,
        'AcceptedCmp1': accepted_cmp1, 'AcceptedCmp2': accepted_cmp2, 'Complain': complain,
        'Response': response_input, 'Total Spending': total_spending, 'Age': age, 'Total Children': total_children,
        'Customer_Tenure': customer_tenure
    }
    for k, v in numeric_map.items():
        if k in final:
            final[k] = float(v)

    edu_map = {'Graduation': 'Education_Graduation', 'Master': 'Education_Master',
               'PhD': 'Education_PhD', 'Basic': 'Education_Basic', '2n Cycle': 'Education_2n Cycle'}
    if education in edu_map and edu_map[education] in final:
        final[edu_map[education]] = 1.0

    marital_map = {
        'Alone': 'Marital_Status_Alone', 'Divorced': 'Marital_Status_Divorced', 'Married': 'Marital_Status_Married',
        'Together': 'Marital_Status_Together', 'Widow': 'Marital_Status_Widow', 'YOLO': 'Marital_Status_YOLO',
        'Single': 'Marital_Status_Single'
    }
    if marital_status in marital_map and marital_map[marital_status] in final:
        final[marital_map[marital_status]] = 1.0

    return final

# -----------------------------
# Prediction function for single customer
# -----------------------------
@st.cache_data
def predict_single_customer(final_dict):
    # DataFrame in model order
    input_df = pd.DataFrame([final_dict], columns=model_feature_names).astype(float)

    # scale in scaler order (only those present)
    scaler_cols = [c for c in scaler_feature_names if c in input_df.columns]
    if scaler is not None and scaler_cols:
        input_df.loc[:, scaler_cols] = scaler.transform(input_df[scaler_cols])

    # predict using model (ensure columns order)
    pred = kmeans_model.predict(input_df[model_feature_names])
    cluster = int(pred[0])

    # compute distance to centroid and top diffs if cluster centers exist
    centroids = getattr(kmeans_model, "cluster_centers_", None)
    distance = None
    feature_diff_df = None
    if centroids is not None:
        # pick centroid ordering: try scaler_cols match first, else model_feature_names
        if centroids.shape[1] == len(scaler_cols):
            centroid_cols = scaler_cols
        else:
            centroid_cols = [c for c in model_feature_names if c in input_df.columns]
        vec = np.array([input_df[c].iloc[0] if c in input_df.columns else 0.0 for c in centroid_cols])
        centroid_vec = centroids[cluster][:len(centroid_cols)]
        distance = float(np.linalg.norm(vec - centroid_vec))
        abs_diff = np.abs(vec - centroid_vec)
        feature_diff_df = pd.DataFrame({"feature": centroid_cols, "abs_diff": abs_diff}).sort_values("abs_diff", ascending=False).reset_index(drop=True)

    return cluster, input_df, distance, feature_diff_df

# -----------------------------
# Main: predict single customer when button pressed
# -----------------------------
if st.sidebar.button("🔮 Predict Segment"):
    final_dict = build_single_customer_dict()
    try:
        cluster_num, input_df_scaled, dist_to_centroid, feature_diff_df = predict_single_customer(final_dict)

        st.success(f"The predicted customer behaviour is: **Cluster {cluster_num}** 🎉")
        st.balloons()

        # Overview cards
        total_spending = float(final_dict.get('Total Spending', 0.0))
        age_display = int(final_dict.get('Age', 0))
        total_children_display = int(final_dict.get('Total Children', 0))
        col1, col2, col3 = st.columns(3)
        col1.metric("Cluster", cluster_num)
        col2.metric("Age", age_display)
        col3.metric("Total Spending ($)", f"{total_spending:,.0f}")

        # Core features display
        core = {
            "Income ($)": f"{final_dict.get('Income',0):,.0f}",
            "Total Spending ($)": f"{total_spending:,.0f}",
            "Recency (Days)": int(final_dict.get('Recency', 0)),
            "Education": education,
            "Marital Status": marital_status,
            "Total Children": total_children_display
        }
        st.table(pd.DataFrame(core, index=[0]).T.rename_axis("Feature"))

        # show scaled input
        with st.expander("Show model input (scaled)"):
            st.dataframe(input_df_scaled.T)

        # distance and top diffs
        if dist_to_centroid is not None:
            st.metric("Distance to centroid (scaled)", f"{dist_to_centroid:.4f}")
            if feature_diff_df is not None and not feature_diff_df.empty:
                st.markdown("Top feature differences vs. centroid (scaled)")
                st.dataframe(feature_diff_df.head(8).style.format({"abs_diff":"{:.4f}"}), use_container_width=True)
                fig = px.bar(feature_diff_df.head(8), x='feature', y='abs_diff', title="Top feature differences (scaled)")
                st.plotly_chart(fig, use_container_width=True)

        # spending pie
        st.subheader("Spending breakdown")
        spend_df = pd.DataFrame({
            "Category": ["Wines","Fruits","Meat","Fish","Sweets","Gold"],
            "Amount": [final_dict.get('MntWines',0), final_dict.get('MntFruits',0), final_dict.get('MntMeatProducts',0),
                       final_dict.get('MntFishProducts',0), final_dict.get('MntSweetProducts',0), final_dict.get('MntGoldProds',0)]
        })
        spend_df = spend_df[spend_df['Amount']>0]
        if not spend_df.empty:
            fig_pie = px.pie(spend_df, names='Category', values='Amount', title=f"Spending Breakdown (Total ${spend_df['Amount'].sum():,.0f})")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No spending entered to show breakdown.")

        # radar normalized
        st.subheader("Normalized customer profile")
        radar_vals = [
            min(final_dict.get('Income',0)/100000, 1.0),
            min(total_spending/2000, 1.0),
            1 - min(final_dict.get('Recency', 0)/365, 1.0),
            min(final_dict.get('Age',0)/100, 1.0),
            min(final_dict.get('Total Children',0)/10, 1.0)
        ]
        fig_rad = go.Figure(data=[go.Scatterpolar(r=np.array(radar_vals)*100, theta=['Income','Spending','Recency','Age','Children'], fill='toself')])
        fig_rad.update_layout(polar=dict(radialaxis=dict(range=[0,100])), showlegend=False)
        st.plotly_chart(fig_rad, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -----------------------------
# Upload dataset: robust parsing, auto-cluster if needed, show summaries & download
# -----------------------------
st.markdown("---")
st.header("📁 Upload Dataset (CSV) — cluster assignment & comparison")
st.info("Upload a CSV. If it lacks a 'Cluster' column, the app will assign clusters using the loaded scaler + model.")

uploaded = st.file_uploader("Upload CSV (e.g., marketing_campaign.csv)", type=["csv"])

def read_csv_with_bom_handling(uploaded_file):
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig', low_memory=False)
        return df
    except Exception:
        uploaded_file.seek(0)
        content = uploaded_file.read().decode(errors='replace')
        from csv import Sniffer
        try:
            dialect = Sniffer().sniff('\n'.join(content.splitlines()[:2]))
            sep = dialect.delimiter
        except Exception:
            sep = ','
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=sep, encoding='utf-8-sig', low_memory=False)
        return df

if uploaded is not None:
    try:
        df_upload = read_csv_with_bom_handling(uploaded)
        st.success(f"Loaded {uploaded.name} — {df_upload.shape[0]} rows × {df_upload.shape[1]} cols")
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        st.stop()

    # normalize header names
    df_upload.columns = [c.strip() for c in df_upload.columns]

    # detect Mnt columns
    mnt_cols = [c for c in df_upload.columns if c.lower().startswith('mnt')]
    if not mnt_cols:
        mnt_cols = [c for c in df_upload.columns if 'mnt' in c.lower()]

    # compute total spending
    if mnt_cols:
        for c in mnt_cols:
            df_upload[c] = pd.to_numeric(df_upload[c], errors='coerce').fillna(0)
        df_upload['Total_Spending_Uploaded'] = df_upload[mnt_cols].sum(axis=1)
    else:
        df_upload['Total_Spending_Uploaded'] = 0.0

    # detect cluster column
    cluster_candidates = [c for c in df_upload.columns if 'cluster' in c.lower()]
    if cluster_candidates:
        cluster_col = cluster_candidates[0]
        st.info(f"Found cluster column in uploaded file: {cluster_col} — using it for summaries.")
        df_upload['Cluster'] = df_upload[cluster_col]
    else:
        # assign clusters
        st.info("No cluster column found. Assigning clusters using current model & scaler...")
        # build aligned rows for entire df_upload (vectorized approach)
        # Start with zeros for all features
        n = df_upload.shape[0]
        final_df = pd.DataFrame(0.0, index=range(n), columns=model_feature_names)

        # fill known numeric columns if present
        for col in ['Year_Birth','Income','Kidhome','Teenhome','Recency','NumDealsPurchases','NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumWebVisitsMonth','AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','Complain','Response']:
            if col in df_upload.columns and col in final_df.columns:
                final_df[col] = pd.to_numeric(df_upload[col], errors='coerce').fillna(0.0).astype(float)

        # Mnt columns mapping
        for m in mnt_cols:
            if m in final_df.columns:
                final_df[m] = df_upload[m]

        # engineered features
        if 'Total Spending' in final_df.columns:
            final_df['Total Spending'] = df_upload.get('Total_Spending_Uploaded', 0.0)
        if 'Age' in final_df.columns and 'Year_Birth' in df_upload.columns:
            final_df['Age'] = (datetime.date.today().year - pd.to_numeric(df_upload['Year_Birth'], errors='coerce')).fillna(0.0)
        if 'Total Children' in final_df.columns and 'Kidhome' in df_upload.columns:
            final_df['Total Children'] = pd.to_numeric(df_upload.get('Kidhome', 0), errors='coerce').fillna(0.0) + pd.to_numeric(df_upload.get('Teenhome', 0), errors='coerce').fillna(0.0)
        if 'Customer_Tenure' in final_df.columns and 'Dt_Customer' in df_upload.columns:
            try:
                dt = pd.to_datetime(df_upload['Dt_Customer'], errors='coerce')
                final_df['Customer_Tenure'] = (pd.Timestamp.now().normalize() - dt.dt.normalize()).dt.days.fillna(0.0)
            except Exception:
                final_df['Customer_Tenure'] = 0.0

        # one-hot education & marital
        edu_map = {'Graduation': 'Education_Graduation', 'Master': 'Education_Master', 'PhD': 'Education_PhD', 'Basic': 'Education_Basic', '2n Cycle': 'Education_2n Cycle'}
        for k,v in edu_map.items():
            if v in final_df.columns and 'Education' in df_upload.columns:
                final_df.loc[df_upload['Education'].astype(str).str.strip()==k, v] = 1.0

        marital_map = {
            'Alone':'Marital_Status_Alone','Divorced':'Marital_Status_Divorced','Married':'Marital_Status_Married',
            'Together':'Marital_Status_Together','Widow':'Marital_Status_Widow','YOLO':'Marital_Status_YOLO','Single':'Marital_Status_Single'
        }
        for k,v in marital_map.items():
            if v in final_df.columns and 'Marital_Status' in df_upload.columns:
                final_df.loc[df_upload['Marital_Status'].astype(str).str.strip()==k, v] = 1.0

        # scale using scaler_feature_names order
        scaler_cols = [c for c in scaler_feature_names if c in final_df.columns]
        if scaler is not None and scaler_cols:
            final_df.loc[:, scaler_cols] = scaler.transform(final_df[scaler_cols])
        else:
            st.warning("Scaler columns couldn't be matched; predictions may be wrong. Inspect debug expander.")

        # predict clusters
        predicted_clusters = kmeans_model.predict(final_df[model_feature_names])
        df_upload['Cluster'] = predicted_clusters

    # compute cluster summary
    agg_cols = [c for c in ['Income', 'Total_Spending_Uploaded', 'Recency'] if c in df_upload.columns]
    if not agg_cols and mnt_cols:
        agg_cols = [mnt_cols[0]]
    if agg_cols:
        cluster_summary = df_upload.groupby('Cluster')[agg_cols].mean().reset_index()
        st.markdown("##### Cluster summary (averages)")
        st.dataframe(cluster_summary)
        # plot
        plot_cols = [c for c in ['Income','Total_Spending_Uploaded','Recency'] if c in cluster_summary.columns]
        if plot_cols:
            fig = px.bar(cluster_summary.rename(columns={'Cluster':'Cluster ID'}), x='Cluster ID', y=plot_cols, barmode='group', title='Cluster averages (uploaded file)')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns found to aggregate for cluster summary.")

    # show sample rows
    st.markdown("##### Sample rows (with Cluster)")
    st.dataframe(df_upload.head(20))

    # download augmented CSV
    to_download = df_upload.copy()
    # ensure Cluster is first column for clarity
    cols_order = ['Cluster'] + [c for c in to_download.columns if c!='Cluster']
    to_download = to_download[cols_order]
    csv_bytes = to_download.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download uploaded file with Cluster", data=csv_bytes, file_name=f"{uploaded.name.replace('.csv','')}_with_cluster.csv", mime="text/csv")
