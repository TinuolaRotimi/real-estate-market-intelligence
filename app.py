from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

print("Loading Melbourne housing data and training model...")

url = "https://raw.githubusercontent.com/plotly/datasets/master/melbourne_housing_snapshot.csv"
df_raw = pd.read_csv(url)

if 'Bedroom2' in df_raw.columns:
    df_raw.rename(columns={'Bedroom2': 'Bedroom'}, inplace=True)

df = df_raw.dropna(subset=['Price']).copy()
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df['Year_Sold'] = df['Date'].dt.year
df['Price_Per_Sqm'] = np.where(df['Landsize'] > 0, df['Price'] / df['Landsize'], np.nan)
df['Price_Per_Sqm'] = df['Price_Per_Sqm'].fillna(df['Price_Per_Sqm'].median())

avm_features = ['Rooms', 'Type', 'Distance', 'Postcode', 'Bedroom', 'Bathroom', 
                'Car', 'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea']
X = df[avm_features].copy()
y = df['Price']
X = pd.get_dummies(X, columns=['Type', 'CouncilArea'], drop_first=True)

region_counts = df['Regionname'].value_counts()
valid_regions = region_counts[region_counts > 5].index
df['Stratify_Region'] = df['Regionname'].where(df['Regionname'].isin(valid_regions), 'Other')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=df['Stratify_Region'])

model = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

suburbs = sorted(df['Suburb'].unique().tolist())
types = sorted(df['Type'].unique().tolist())

city_median_price_sqm = df['Price_Per_Sqm'].median()
suburb_stats = df.groupby('Suburb').agg(
    Avg_Price=('Price', 'mean'),
    Median_Price=('Price', 'median'),
    Avg_Price_Per_Sqm=('Price_Per_Sqm', 'mean'),
    Transaction_Count=('Price', 'count')
).round(0)

total_properties = len(df)
avg_price = df['Price'].mean()
median_price = df['Price'].median()
unique_suburbs = df['Suburb'].nunique()

print(f"Model ready. {len(df):,} properties, {unique_suburbs} suburbs.")

PAGE_CSS = """
:root{--bg:#0a0a0f;--card:#12121a;--border:#252540;--text:#e8e8f0;--muted:#9898b0;--accent:#6366f1;--green:#22c55e;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,sans-serif;line-height:1.6;}
.wrap{max-width:1000px;margin:0 auto;padding:40px 24px;}
header{border-bottom:1px solid var(--border);padding-bottom:24px;margin-bottom:32px;}
h1{font-size:2rem;font-weight:700;margin-bottom:6px;}
.sub{color:var(--muted);font-size:14px;}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:28px;}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:20px;}
.card .k{font-family:monospace;font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:var(--muted);margin-bottom:8px;}
.card .v{font-family:monospace;font-size:28px;font-weight:700;}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:28px 0;}
label{font-family:monospace;font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--muted);}
select,input{background:var(--card);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:10px 14px;font-size:15px;}
select{min-width:240px;}
input{width:120px;}
button{background:var(--accent);color:white;border:none;padding:12px 24px;border-radius:6px;font-size:14px;font-weight:600;cursor:pointer;}
button:hover{opacity:0.9;}
.result-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-top:20px;}
table{width:100%;border-collapse:collapse;margin-top:16px;font-size:13px;}
th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--border);}
th{font-family:monospace;font-size:10px;text-transform:uppercase;letter-spacing:1px;color:var(--muted);}
td.num{font-family:monospace;text-align:right;}
footer{margin-top:48px;padding-top:20px;border-top:1px solid var(--border);color:var(--muted);font-size:12px;text-align:center;}
"""

@app.route("/")
def index():
    selected_suburb = request.args.get("suburb", suburbs[0])
    if selected_suburb not in suburbs:
        selected_suburb = suburbs[0]
    
    stats = suburb_stats.loc[selected_suburb] if selected_suburb in suburb_stats.index else None
    
    options = "".join(f'<option value="{s}"{" selected" if s == selected_suburb else ""}>{s}</option>' for s in suburbs[:100])
    
    stats_html = ""
    if stats is not None:
        stats_html = f"""
<div class="result-grid">
  <div class="card"><div class="k">Average Price</div><div class="v" style="font-size:22px;">${stats['Avg_Price']:,.0f}</div></div>
  <div class="card"><div class="k">Median Price</div><div class="v" style="font-size:22px;">${stats['Median_Price']:,.0f}</div></div>
  <div class="card"><div class="k">Price per Sqm</div><div class="v" style="font-size:22px;">${stats['Avg_Price_Per_Sqm']:,.0f}</div></div>
  <div class="card"><div class="k">Transactions</div><div class="v">{int(stats['Transaction_Count']):,}</div></div>
</div>
"""
    
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Real Estate Market Intelligence</title>
<style>{PAGE_CSS}</style>
</head><body><div class="wrap">
<header>
  <h1>Real Estate Market Intelligence</h1>
  <p class="sub">Melbourne Housing Market. {total_properties:,} properties across {unique_suburbs} suburbs.</p>
</header>
<div class="grid">
  <div class="card"><div class="k">Total Properties</div><div class="v">{total_properties:,}</div></div>
  <div class="card"><div class="k">Average Price</div><div class="v" style="font-size:22px;">${avg_price:,.0f}</div></div>
  <div class="card"><div class="k">Median Price</div><div class="v" style="font-size:22px;">${median_price:,.0f}</div></div>
  <div class="card"><div class="k">Suburbs</div><div class="v">{unique_suburbs}</div></div>
</div>
<form method="get" class="controls">
  <label for="suburb">Suburb</label>
  <select id="suburb" name="suburb" onchange="this.form.submit()">{options}</select>
</form>
{stats_html}
<footer>Data: Melbourne Housing Snapshot | {total_properties:,} real property transactions analyzed</footer>
</div></body></html>"""

@app.route("/api/suburb/<suburb_name>")
def api_suburb(suburb_name):
    if suburb_name not in suburb_stats.index:
        return {"error": "Suburb not found"}, 404
    stats = suburb_stats.loc[suburb_name]
    return {
        "suburb": suburb_name,
        "avg_price": round(float(stats['Avg_Price']), 0),
        "median_price": round(float(stats['Median_Price']), 0),
        "avg_price_per_sqm": round(float(stats['Avg_Price_Per_Sqm']), 0),
        "transactions": int(stats['Transaction_Count'])
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
