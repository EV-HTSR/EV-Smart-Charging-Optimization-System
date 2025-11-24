# ⚡ EV Smart Charging Optimization System

A smart and modular system designed to optimize Electric Vehicle (EV) charging using **machine learning**, **routing algorithms**, and **SQLite databases**. 
The system predicts charging demand, identifies optimal charging stations, and generates efficient travel routes using real-world datasets.

---

## 🚀 Key Features

### 🔋 Intelligent Charging Demand Prediction
Uses machine learning models to forecast when and where EV charging demand will rise.

### 🗺️ Route & Charging Optimization
Computes the most efficient travel routes while selecting the best available charging stations.

### 🗂️ SQLite Database Integration
Stores processed station data, charging sessions, and predictive outputs.

### 🤖 Automated Data Pipeline
Handles dataset cleaning, feature engineering, and geospatial preprocessing.

### 🌐 Backend API
Provides endpoints for predictions, routing, and station insights.

### 💻 Lightweight Frontend
Displays predicted routes, demand insights, and data results.

---

## 🧠 Tech Stack

- Python 3.x  
- ML & Data: scikit-learn, numpy, pandas  
- Routing & Maps: geopy, shapefiles  
- Backend: FastAPI or custom Python API  
- Database: SQLite  
- Compatible With: Windows & Linux environments  

---

## 📁 Project Structure

```
ev-smart-charging/
|
├── app_backend.py            # Backend API endpoints
├── app_frontend.py           # Simple UI / frontend
├── data_pipeline.py          # Data preprocessing pipeline
├── routing_provider.py       # Route + station optimization logic
├── train_models.py           # Model training script
├── database.py               # SQLite helper module
├── check_database.py         # Database consistency checker
├── charging_stations.db      # Local station database
├── ev_charging.db            # Local EV session database
├── requirements.txt          # Dependencies
├── How to run.txt            # Step-by-step instructions
└── California_*              # Geospatial dataset files (SHP / DBF / PRJ / XML / etc.)
```

---

## 🛠️ How to Run

### 1️⃣ Create virtual environment (Windows)
```
python -m venv venv
```

### 2️⃣ Activate the environment (PowerShell)
```
.env\Scripts\Activate.ps1
```

### 3️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 4️⃣ One-time setup (only run these once)
```
python database.py
python data_pipeline.py
python train_models.py
```

### 5️⃣ Start backend (keep this running)
```
python app_backend.py
```

### 6️⃣ New terminal → Activate venv → Start frontend
```
.env\Scripts\Activate.ps1
streamlit run app_frontend.py
```

---

## 👥 Contributors
Team **EV-HTSR**
Harshit Pathak, Saiyam Jain, Rajdeep, Tarun Attri

---

## 📜 License
This project is licensed under the **MIT License**.  
See the `LICENSE` file for details.
