#  Retail-Online Analytics Dashboard

This repository contains the implementation of an **interactive e-commerce analytics dashboard** developed as part of the **Data Mining course** at  
**University of Engineering and Technology (UET) – Vietnam National University (VNU)**.

The project demonstrates the application of data mining and analytics techniques on large-scale online retail data, focusing on **customer behavior analysis, market basket analysis, and recommendation insights**.

---
# TEAM
Vũ Minh Sơn
Nguyễn Bá Quang
Tạ Nguyên Thành
---
##  Live Demo
**Streamlit App:**  
https://retail-online-4cwu5c9sbh3rcnvocpvkzg.streamlit.app/
---

##  Key Features

### 1. Business Overview & KPI
- Total users, sessions, products, and revenue
- Conversion funnel: **View → Cart → Purchase**

### 2. Market Basket Analysis
- FP-Growth algorithm for frequent itemset mining
- Association rule generation (support, confidence, lift)
- Product-level network graph visualization
- Cross-selling insight extraction

### 3. Retail Insights & Customer Behavior
- Hourly purchase distribution
- Top products by revenue
- Basket size distribution
- Purchase frequency per user
- Time from first view to purchase

### 4. Customer Segmentation (RFM + K-Means)
- RFM feature construction (Recency, Frequency, Monetary)
- Elbow Method & Silhouette Score for cluster selection
- 3D visualization of customer segments
- Cluster-level behavioral summary

### 5. Clickstream Analysis (Brand-to-Brand)
- Sequential clickstream analysis based on user sessions
- Sankey diagram to visualize brand transition flows
- Identification of brand switching patterns

### 6. Brand-based Recommendation
- Brand-aware recommendation using association rules
- Product suggestions based on cross-brand purchasing behavior
- Business interpretation of recommendation rules

---

##  Techniques & Algorithms Used
- FP-Growth (Frequent Pattern Mining)
- Association Rule Mining
- RFM Analysis
- K-Means Clustering
- Clickstream Sequence Analysis
- Data Visualization (Plotly, Network Graphs, Sankey)

---

##  Technologies
- **Python**
- **Streamlit**
- **Pandas / NumPy**
- **Scikit-learn**
- **MLxtend**
- **Plotly**
- **PyArrow (Parquet)**
- **NetworkX / PyVis**

---
