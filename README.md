# ⚡ Predictive Analysis for Urban Power Consumption
### Data Mining & Warehousing (DMW) Project — Pune Metropolitan Region

---

## 📁 Project Structure

```
dmw_project/
├── app.py              # Main Streamlit dashboard (all-in-one)
├── requirements.txt    # Python dependencies
└── README.md           # This file
---

## 📊 DMW Concepts Covered

| Concept | Implementation | Marks |
|---------|---------------|-------|
| Data Collection | Hourly simulation: kWh, Temp, Humidity, Location | ✅ |
| Preprocessing | Missing value handling, Integration, Min-Max scaling | 5 marks |
| Data Warehouse | Star Schema: Fact_EnergyUsage + 3 Dim tables | 4 marks |
| Clustering | K-Means with interactive k slider | 6 marks |
| Association Rules | Apriori with tunable support/confidence | 6 marks |
| Evaluation | Elbow Method + Silhouette Score | 2 marks |
| Visualization | 8+ charts: heatmap, scatter, time-series, pie | ✅ |

---

## 🎛️ Interactive Features

- **k slider** → Change number of clusters in real time
- **Location filter** → Analyze specific Pune suburbs
- **Month range** → Drill into seasons
- **Support/Confidence sliders** → Tune Apriori rules live

---

## 📚 Technologies Used

- **Streamlit** — Dashboard framework
- **Pandas + NumPy** — Data manipulation
- **scikit-learn** — K-Means, MinMaxScaler, Silhouette Score
- **mlxtend** — Apriori, Association Rules
- **Matplotlib + Seaborn** — Visualizations
