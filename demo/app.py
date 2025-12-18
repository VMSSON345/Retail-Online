# ===============================
# IMPORT
# ===============================
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from mlxtend.frequent_patterns import fpgrowth, association_rules
import pyarrow.parquet as pq
import glob, os
import networkx as nx
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================
# PARQUET_DIR = "/mnt/d/data/chunk_2"
PARQUET_DIR = "demo/chunk_2"
MAX_ROWS = 150_000

st.set_page_config(
    page_title="eCommerce Analytics Dashboard",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ eCommerce Analytics Dashboard")
st.markdown("---")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data(show_spinner=True)
def load_parquet(files, max_rows):
    dfs, total = [], 0
    for f in files:
        df = pq.read_table(f).to_pandas()
        dfs.append(df)
        total += len(df)
        if total >= max_rows:
            break
    return pd.concat(dfs, ignore_index=True).head(max_rows)

st.sidebar.header("📂 Parquet Loader")

files = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))
n_files = st.sidebar.slider("Số file parquet", 1, len(files), min(10, len(files)))

if st.sidebar.button("🚀 Load data"):
    df = load_parquet(files[:n_files], MAX_ROWS)
    df["event_time"] = pd.to_datetime(df["event_time"])
    df["event_type"] = df["event_type"].astype(str)
    df["user_session"] = df["user_session"].astype(str)
    df["product_id"] = df["product_id"].astype(str)
    st.session_state.df = df

if "df" not in st.session_state:
    st.info("👈 Load dữ liệu ở sidebar")
    st.stop()

df = st.session_state.df

# ===============================
# KPI
# ===============================
st.subheader("📊 Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Users", df.user_id.nunique())
c2.metric("Sessions", df.user_session.nunique())
c3.metric("Products", df.product_id.nunique())
c4.metric("Revenue", f"{df[df.event_type=='purchase'].price.sum():,.0f}")

# ===============================
# FUNNEL
# ===============================
funnel = df.groupby("event_type").size().reindex(["view","cart","purchase"]).fillna(0)
st.plotly_chart(go.Figure(go.Funnel(y=funnel.index, x=funnel.values)), width="stretch")

# ===============================
# MARKET BASKET (NOTEBOOK STYLE)
# ===============================
st.markdown("---")
st.header("🔗 Market Basket Analysis (Product-level)")

min_sup = st.slider("Min Support", 0.0005, 0.01, 0.001, 0.0005)
min_lift = st.slider("Min Lift", 1.0, 3.0, 1.2, 0.05)
top_k_products = st.slider("Top-K Products", 100, 1000, 500, 50)
top_k_rules = st.slider("Top-K Rules", 5, 40, 20)

purchase_df = df[df.event_type=="purchase"].copy()

# session ≥2
valid_sessions = purchase_df.groupby("user_session").product_id.nunique()
purchase_df = purchase_df[purchase_df.user_session.isin(valid_sessions[valid_sessions>1].index)]

# top-K product
top_products = purchase_df.product_id.value_counts().head(top_k_products).index
purchase_df = purchase_df[purchase_df.product_id.isin(top_products)]

basket = (
    purchase_df
    .groupby(["user_session","product_id"])
    .size()
    .unstack(fill_value=0)
)
basket = (basket>0).astype(int)

frequent = fpgrowth(basket, min_support=min_sup, use_colnames=True)
rules = association_rules(frequent, metric="lift", min_threshold=min_lift)

rules = rules.sort_values("lift", ascending=False).head(top_k_rules)
rules["A"] = rules["antecedents"].apply(lambda x: list(x)[0])
rules["B"] = rules["consequents"].apply(lambda x: list(x)[0])

st.subheader("📋 Top Association Rules")
st.dataframe(
    rules[["A","B","support","confidence","lift"]],
    width="stretch"
)

from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile

st.subheader("Network Graph (Product-level)")

# ===== FILTER RULE MẠNH HƠN =====
rules_vis = rules[
    (rules["confidence"] >= 0.3) &
    (rules["lift"] >= min_lift)
].copy()

if rules_vis.empty:
    st.warning("Không có rule đủ mạnh để vẽ graph")
else:
    net = Network(
        height="600px",
        width="100%",
        bgcolor="#0f172a",
        font_color="white",
        directed=True
    )

    # ===== PHYSICS (CỰC QUAN TRỌNG) =====
    net.barnes_hut(
        gravity=-25000,
        central_gravity=0.3,
        spring_length=140,
        spring_strength=0.01,
        damping=0.09
    )

    # ===== ĐẾM DEGREE ĐỂ SCALE NODE =====
    from collections import Counter
    cnt = Counter(
        rules_vis["A"].tolist() + rules_vis["B"].tolist()
    )

    for _, r in rules_vis.iterrows():
        a, b = r["A"], r["B"]
        lift, conf = r["lift"], r["confidence"]

        net.add_node(
            a,
            label=a,
            size=10 + cnt[a]*3,
            title=f"Product {a}<br>Degree: {cnt[a]}",
            color="#60a5fa"
        )

        net.add_node(
            b,
            label=b,
            size=10 + cnt[b]*3,
            title=f"Product {b}<br>Degree: {cnt[b]}",
            color="#38bdf8"
        )

        net.add_edge(
            a, b,
            value=lift * 2,
            title=f"Lift: {lift:.2f}<br>Confidence: {conf:.2f}",
            color="rgba(255,255,255,0.6)",
            arrows="to"
        )

    # ===== SAVE & EMBED =====
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        with open(tmp.name, "r", encoding="utf-8") as f:
            components.html(f.read(), height=620)

    os.remove(tmp.name)

# ===============================
# BUSINESS INTERPRETATION
# ===============================
st.subheader("🧠 Business Insights")

for _, r in rules_vis.head(5).iterrows():
    st.markdown(
        f"""
        **🛒 Product {r['A']} → Product {r['B']}**  
        - Lift: **{r['lift']:.2f}**  
        - Confidence: **{r['confidence']:.2f}**  

        👉 Customers who buy **{r['A']}** are **{r['lift']:.1f}× more likely**
        to also buy **{r['B']}** than random.
        """
    )


# ===============================
# RETAIL INSIGHTS & VISUALIZATION
# ===============================
st.markdown("---")
st.header("📈 Retail Insights & Customer Behavior")


# ==================================================
# 1. PURCHASE BY TIME (HOURLY DEMAND)
# ==================================================
st.subheader("🕒 Purchase Distribution by Hour")

df["hour"] = df["event_time"].dt.hour

hourly_purchase = (
    df[df.event_type == "purchase"]
    .groupby("hour")
    .size()
    .reset_index(name="count")
)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hourly_purchase["hour"],
    y=hourly_purchase["count"],
    mode="lines+markers",
    line=dict(width=3),
    marker=dict(size=8)
))
fig.update_layout(
    title="Number of Purchases by Hour of Day",
    xaxis_title="Hour",
    yaxis_title="Purchases",
)
st.plotly_chart(fig, width="stretch")

st.caption("💡 Giúp xác định giờ cao điểm để tối ưu quảng cáo & flash sale.")


# ==================================================
# 3. TOP PRODUCTS BY REVENUE
# ==================================================
st.subheader("💰 Top Products by Revenue")

top_products_rev = (
    df[df.event_type == "purchase"]
    .groupby("product_id")["price"]
    .sum()
    .sort_values(ascending=False)
    .head(15)
    .reset_index()
)

fig = go.Figure(go.Bar(
    x=top_products_rev["price"],
    y=top_products_rev["product_id"],
    orientation="h"
))
fig.update_layout(
    title="Top 15 Products by Revenue",
    xaxis_title="Revenue",
    yaxis_title="Product ID"
)
st.plotly_chart(fig, width="stretch")

st.caption("💡 Có sản phẩm bán ít nhưng tạo doanh thu cao → premium items.")

# ==================================================
# 4. PRODUCTS PER SESSION (BASKET SIZE)
# ==================================================
st.subheader("🧺 Basket Size Distribution")

basket_size = (
    df[df.event_type == "purchase"]
    .groupby("user_session")["product_id"]
    .nunique()
)

fig = go.Figure(go.Histogram(
    x=basket_size,
    nbinsx=20
))
fig.update_layout(
    title="Number of Unique Products per Session",
    xaxis_title="Products per Session",
    yaxis_title="Number of Sessions"
)
st.plotly_chart(fig, width="stretch")


# ==================================================
# 5. PURCHASE FREQUENCY PER USER
# ==================================================
st.subheader("👤 Purchase Frequency per User")

user_purchase_cnt = (
    df[df.event_type == "purchase"]
    .groupby("user_id")
    .size()
)

fig = go.Figure(go.Histogram(
    x=user_purchase_cnt,
    nbinsx=30
))
fig.update_layout(
    title="Purchase Frequency per User",
    xaxis_title="Number of Purchases",
    yaxis_title="Number of Users"
)
st.plotly_chart(fig, width="stretch")

st.caption("💡 Phần lớn user mua ít, một nhóm nhỏ mua nhiều → khách hàng trung thành.")

# ==================================================
# 6. TIME TO PURCHASE (VIEW → PURCHASE)
# ==================================================
st.subheader("⏱️ Time from First View to Purchase")

view_time = (
    df[df.event_type == "view"]
    .groupby("user_session")["event_time"]
    .min()
)
purchase_time = (
    df[df.event_type == "purchase"]
    .groupby("user_session")["event_time"]
    .min()
)

time_df = pd.concat([view_time, purchase_time], axis=1)
time_df.columns = ["view_time", "purchase_time"]
time_df = time_df.dropna()

time_df["minutes"] = (
    time_df["purchase_time"] - time_df["view_time"]
).dt.total_seconds() / 60

fig = go.Figure(go.Box(
    y=time_df["minutes"]
))
fig.update_layout(
    title="Time from First View to Purchase (minutes)",
    yaxis_title="Minutes"
)
st.plotly_chart(fig, width="stretch")

st.caption("💡 Phát hiện hành vi impulse buying vs. cân nhắc lâu.")

# ==================================================
# 7. PRODUCTS MOST INFLUENTIAL IN RULES
# ==================================================
st.subheader("🔗 Most Influential Products in Association Rules")

from collections import Counter
rule_products = Counter(rules["A"].tolist() + rules["B"].tolist())

rule_df = (
    pd.DataFrame(rule_products.items(), columns=["product_id", "count"])
    .sort_values("count", ascending=False)
    .head(10)
)

fig = go.Figure(go.Bar(
    x=rule_df["count"],
    y=rule_df["product_id"],
    orientation="h"
))
fig.update_layout(
    title="Products Appearing Most Frequently in Association Rules",
    xaxis_title="Rule Count",
    yaxis_title="Product ID"
)
st.plotly_chart(fig, width="stretch")

st.caption("💡 Đây là các sản phẩm đóng vai trò \"anchor\" trong cross-selling.")




# ==================================================
# CUSTOMER SEGMENTATION: RFM + K-MEANS (OPTIMIZED)
# ==================================================
st.markdown("---")
st.header("👥 Customer Segmentation (RFM + K-Means)")
st.caption("Phân nhóm khách hàng – chỉ chạy lại khi bấm Apply")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import numpy as np

# --------------------------------------------------
# 1. BUILD RFM TABLE (CACHE)
# --------------------------------------------------
@st.cache_data
def build_rfm(df):
    rfm_df = df[df.event_type == "purchase"].copy()
    snapshot_date = rfm_df["event_time"].max() + pd.Timedelta(days=1)

    rfm = (
        rfm_df.groupby("user_id")
        .agg({
            "event_time": lambda x: (snapshot_date - x.max()).days,
            "user_session": "nunique",
            "price": "sum"
        })
        .reset_index()
    )

    rfm.columns = ["user_id", "Recency", "Frequency", "Monetary"]
    return rfm


rfm = build_rfm(df)

if rfm.empty:
    st.warning("❌ Không có dữ liệu purchase để phân khúc khách hàng")
    st.stop()

st.subheader("📋 Bảng RFM (mẫu)")
st.dataframe(rfm.head(), use_container_width=True)

# --------------------------------------------------
# 2. STANDARDIZE (CACHE)
# --------------------------------------------------
@st.cache_data
def scale_rfm(rfm):
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    return X

X = scale_rfm(rfm)

# --------------------------------------------------
# 3. COMPUTE ELBOW & SILHOUETTE (CACHE)
# --------------------------------------------------
@st.cache_data
def compute_k_metrics(X):
    K_RANGE = range(2, 11)
    inertias = []
    sil_scores = []

    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)

        if len(set(labels)) > 1:
            sil_scores.append(silhouette_score(X, labels))
        else:
            sil_scores.append(np.nan)

    return list(K_RANGE), inertias, sil_scores


K_RANGE, inertias, sil_scores = compute_k_metrics(X)

# --------------------------------------------------
# 4. CONFIG FORM (KHÔNG AUTO RERUN)
# --------------------------------------------------
with st.form("kmeans_config"):
    st.subheader("⚙️ Cấu hình phân cụm")

    k_method = st.radio(
        "Phương pháp chọn số cụm",
        ["Elbow Method", "Silhouette Score"],
        horizontal=True
    )

    submit_config = st.form_submit_button("📐 Xác định K tối ưu")

# --------------------------------------------------
# 5. VISUALIZE METHOD
# --------------------------------------------------
if submit_config:
    if k_method == "Elbow Method":
        best_k = K_RANGE[np.argmin(np.diff(inertias, 2)) + 1]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=K_RANGE,
            y=inertias,
            mode="lines+markers",
            name="Inertia"
        ))

        fig.add_vline(
            x=best_k,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Elbow K = {best_k}"
        )

        fig.update_layout(
            title="Elbow Method",
            xaxis_title="K",
            yaxis_title="Inertia",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)
        st.success(f"✅ K tối ưu theo Elbow: **{best_k}**")

    else:
        best_k = K_RANGE[np.nanargmax(sil_scores)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=K_RANGE,
            y=sil_scores,
            mode="lines+markers",
            name="Silhouette Score"
        ))

        fig.add_vline(
            x=best_k,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Best K = {best_k}"
        )

        fig.update_layout(
            title="Silhouette Score",
            xaxis_title="K",
            yaxis_title="Score",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)
        st.success(f"✅ K tối ưu theo Silhouette: **{best_k}**")

    st.session_state["best_k"] = best_k

# --------------------------------------------------
# 6. RUN KMEANS (FORM RIÊNG)
# --------------------------------------------------
if "best_k" in st.session_state:
    with st.form("run_kmeans"):
        k_selected = st.slider(
            "Chọn số cụm K",
            min(K_RANGE),
            max(K_RANGE),
            st.session_state["best_k"]
        )

        run_kmeans = st.form_submit_button("🚀 Chạy K-Means")

    if run_kmeans:
        kmeans = KMeans(
            n_clusters=k_selected,
            random_state=42,
            n_init=10
        )

        rfm["Cluster"] = kmeans.fit_predict(X).astype(str)

        st.success(f"🎯 Đã phân cụm với K = {k_selected}")

        # ------------------------------
        # 3D VISUALIZATION
        # ------------------------------
        st.subheader("🧬 Trực quan hóa cụm khách hàng (3D)")

        fig_3d = px.scatter_3d(
            rfm,
            x="Recency",
            y="Frequency",
            z="Monetary",
            color="Cluster",
            hover_data=["user_id"],
            opacity=0.75
        )

        fig_3d.update_layout(height=650)
        st.plotly_chart(fig_3d, use_container_width=True)

        # ------------------------------
        # CLUSTER SUMMARY
        # ------------------------------
        st.subheader("📊 Đặc trưng trung bình của từng cụm")

        cluster_summary = (
            rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
            .mean()
            .reset_index()
        )

        st.dataframe(
            cluster_summary.style.background_gradient(cmap="Blues"),
            use_container_width=True
        )



# ==================================================
# CLICKSTREAM: PRODUCT → PRODUCT (DISPLAY AS BRAND)
# ==================================================
st.markdown("---")
st.header("🖱️ Clickstream Analysis – Brand to Brand")
st.caption("Luồng click được tính theo sản phẩm nhưng hiển thị theo thương hiệu.")

# --------------------------------------------------
# 1. VIEW EVENTS
# --------------------------------------------------
view_df = (
    df[df.event_type == "view"]
    .sort_values(["user_session", "event_time"])
    .copy()
)

# BẮT BUỘC: phải có brand
if "brand" not in view_df.columns:
    st.error("❌ Dataset không có cột 'brand' → không thể hiển thị theo brand")
    st.stop()

view_df["brand"] = view_df["brand"].astype(str)

# --------------------------------------------------
# 2. NEXT PRODUCT + NEXT BRAND
# --------------------------------------------------
view_df["next_product"] = (
    view_df.groupby("user_session")["product_id"].shift(-1)
)

view_df["next_brand"] = (
    view_df.groupby("user_session")["brand"].shift(-1)
)

# --------------------------------------------------
# 3. CLEAN DATA
# --------------------------------------------------
flow_df = view_df.dropna(subset=["next_product", "next_brand"])

# bỏ self-loop (Samsung → Samsung)
flow_df = flow_df[flow_df["brand"] != flow_df["next_brand"]]

# bỏ brand unknown / null
flow_df = flow_df[
    (flow_df["brand"].notna()) &
    (flow_df["next_brand"].notna()) &
    (flow_df["brand"] != "unknown") &
    (flow_df["next_brand"] != "unknown")
]

# --------------------------------------------------
# 4. ĐẾM LUỒNG BRAND → BRAND
# --------------------------------------------------
flow_cnt = (
    flow_df
    .groupby(["brand", "next_brand"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
    .head(30)
)

if flow_cnt.empty:
    st.warning("Không đủ dữ liệu clickstream theo brand.")
else:
    # --------------------------------------------------
    # 5. SANKEY NODES
    # --------------------------------------------------
    nodes = pd.unique(
        flow_cnt[["brand", "next_brand"]].values.ravel()
    )

    node_map = {v: i for i, v in enumerate(nodes)}

    # --------------------------------------------------
    # 6. SANKEY PLOT
    # --------------------------------------------------
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=20,
            thickness=22,
            label=nodes,          
            color="rgba(37,99,235,0.85)",
            line=dict(color="black", width=0.4)
        ),
        link=dict(
            source=flow_cnt["brand"].map(node_map),
            target=flow_cnt["next_brand"].map(node_map),
            value=flow_cnt["count"],
            color="rgba(180,180,180,0.45)"
        )
    ))

    fig.update_layout(
        title="Top 30 Brand-to-Brand Clickstream (View → Next View)",
        font_size=13,
        height=800
    )

    st.plotly_chart(fig, width="stretch")

    st.caption(
        "💡 Mỗi node là một thương hiệu. "
        "Độ dày luồng thể hiện số lần người dùng chuyển từ brand này sang brand khác."
    )



# ===============================
# MARKET BASKET ANALYSIS (PRODUCT-LEVEL – FIXED)
# ===============================
st.markdown("---")
st.header("🔗 Market Basket Analysis")

# --------- CONTROLS ----------
min_sup = st.slider(
    "Min Support (product-level)",
    0.00005, 0.005, 0.0001, 0.00005
)

min_lift = st.slider(
    "Min Lift",
    1.0, 5.0, 1.2, 0.1
)

top_k_rules = st.slider(
    "Top-K Rules",
    20, 300, 100, 20
)

# --------- PURCHASE ONLY ----------
purchase_df = df[df.event_type == "purchase"].copy()

if purchase_df.empty:
    st.warning("❌ Không có dữ liệu purchase")
    st.stop()

# --------- SESSION >= 2 PRODUCT ----------
session_counts = purchase_df.groupby("user_session")["product_id"].nunique()
valid_sessions = session_counts[session_counts >= 2].index
purchase_df = purchase_df[purchase_df["user_session"].isin(valid_sessions)]

st.caption(f"✔ Sessions ≥ 2 products: {purchase_df.user_session.nunique()}")

# --------- BASKET ----------
basket = (
    purchase_df
    .groupby(["user_session", "product_id"])
    .size()
    .unstack(fill_value=0)
)

basket = (basket > 0).astype(bool)  # ⚠️ BẮT BUỘC bool

st.caption(f"Basket shape: {basket.shape}")

# --------- FP-GROWTH ----------
frequent = fpgrowth(
    basket,
    min_support=min_sup,
    use_colnames=True
)

if frequent.empty:
    st.warning("❌ Không có frequent itemsets – giảm Min Support")
    st.stop()

rules = association_rules(
    frequent,
    metric="lift",
    min_threshold=min_lift
)

if rules.empty:
    st.warning("❌ Không có association rules – giảm Lift")
    st.stop()

rules = rules.sort_values("lift", ascending=False).head(top_k_rules)

# ⚠️ SINGLE ITEM RULES ONLY (CHO RECOMMEND)
rules = rules[
    (rules["antecedents"].apply(len) == 1) &
    (rules["consequents"].apply(len) == 1)
].copy()

rules["A"] = rules["antecedents"].apply(lambda x: list(x)[0])
rules["B"] = rules["consequents"].apply(lambda x: list(x)[0])

st.subheader("📋 Top Association Rules")
st.dataframe(
    rules[["A", "B", "support", "confidence", "lift"]],
    use_container_width=True
)

# ===============================
# 🧪 DEBUG: BRAND CÓ NẰM TRONG RULES KHÔNG
# ===============================
st.subheader("Brand coverage in rules")

product_brand_map = (
    df[["product_id", "brand"]]
    .dropna()
    .astype(str)
    .drop_duplicates()
    .set_index("product_id")["brand"]
    .str.lower()
    .to_dict()
)

rules["brand_A"] = rules["A"].map(product_brand_map)
rules["brand_B"] = rules["B"].map(product_brand_map)

brand_rule_counts = rules["brand_A"].value_counts().head(10)
st.write("Top brands appearing in rules:")
st.dataframe(brand_rule_counts)




# ===============================
# 🎯 BRAND-BASED RECOMMENDATION (WORKING)
# ===============================
st.markdown("---")
st.header("🎯 Brand-based Recommendation")
st.caption("Nhập brand → gợi ý sản phẩm dựa trên association rules")

# --------- INPUT ----------
all_brands = (
    df["brand"]
    .dropna()
    .astype(str)
    .str.lower()
    .unique()
)

brand_input = st.selectbox(
    "🔎 Chọn brand",
    sorted(all_brands)
)

top_k_products = st.slider(
    "Top K sản phẩm gợi ý",
    3, 30, 10
)

min_conf = st.slider(
    "Min Confidence",
    0.05, 1.0, 0.2, 0.05
)

# --------- FILTER RULES BY BRAND ----------
brand_rules = rules[
    (rules["brand_A"] == brand_input) &
    (rules["confidence"] >= min_conf)
].sort_values(
    by=["lift", "confidence"],
    ascending=False
)

if brand_rules.empty:
    st.error(
        f"❌ Không tìm thấy luật cho brand '{brand_input}'. "
        "👉 Giảm Min Support hoặc kiểm tra Debug phía trên."
    )
    st.stop()

st.subheader("🔗 Brand thường được mua cùng")
st.dataframe(
    brand_rules[["A", "B", "brand_A", "brand_B", "confidence", "lift"]],
    use_container_width=True
)

# --------- RECOMMENDED BRANDS ----------
recommended_brands = (
    brand_rules["brand_B"]
    .dropna()
    .unique()
    .tolist()
)

# --------- TOP PRODUCTS FROM THOSE BRANDS ----------
product_candidates = (
    df[
        (df.event_type == "purchase") &
        (df["brand"].str.lower().isin(recommended_brands))
    ]
    .groupby(["product_id", "brand"])
    .agg(
        purchase_count=("user_session", "nunique"),
        revenue=("price", "sum")
    )
    .reset_index()
    .sort_values(
        by=["purchase_count", "revenue"],
        ascending=False
    )
    .head(top_k_products)
)

st.subheader("🛒 Sản phẩm được gợi ý")

st.dataframe(
    product_candidates.style.background_gradient(
        subset=["purchase_count", "revenue"],
        cmap="Greens"
    ),
    use_container_width=True
)

# --------- BUSINESS EXPLANATION ----------
st.subheader("🧠 Giải thích")

for _, r in brand_rules.head(3).iterrows():
    st.markdown(
        f"""
        - Người mua **{r['brand_A']}** có khả năng mua **{r['brand_B']}**
        **cao hơn {r['lift']:.1f} lần**  
        (Confidence = {r['confidence']:.2f})
        """
    )

