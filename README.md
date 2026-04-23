# ⚡ Predictive Analysis for Urban Power Consumption
### Data Mining & Warehousing (DMW) Project — Pune Metropolitan Region

---

## 📁 Project Structure

```
dmw_project/
├── app.py              # Main Streamlit dashboard (all-in-one)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🚀 How to Run Locally (Step-by-Step)

### Prerequisites
- Python 3.9 or higher installed
- pip (Python package manager)

### Step 1: Create a Virtual Environment
```bash
# Windows
python -m venv dmw_env
dmw_env\Scripts\activate

# macOS / Linux
python3 -m venv dmw_env
source dmw_env/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the App
```bash
streamlit run app.py
```

The browser opens automatically at `http://localhost:8501`

---

## ☁️ How to Deploy on Streamlit Cloud (Free, Public URL)

### Step 1: Push to GitHub
1. Create a free account at https://github.com
2. Create a new **public** repository named `dmw-power-analytics`
3. Upload both `app.py` and `requirements.txt` to the repo

```bash
git init
git add app.py requirements.txt README.md
git commit -m "Initial DMW project commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/dmw-power-analytics.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select your repository: `dmw-power-analytics`
5. Set Main file path: `app.py`
6. Click **"Deploy!"**

✅ Your app gets a public URL like:  
`https://YOUR_USERNAME-dmw-power-analytics-app-xxxx.streamlit.app`

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
