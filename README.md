# 🛡️ Smart Tourist Safety Monitoring and Incident Response System using AI

An intelligent, real-time tourist safety monitoring dashboard built with **Streamlit**, powered by **Ensemble Machine Learning**, **Explainable AI (SHAP)**, and **Geo-Fencing** — designed to protect tourists in remote and low-connectivity areas.

---

## 📌 Problem Statement

Tourist safety in remote destinations faces critical challenges:

| Challenge | Impact |
|-----------|--------|
| **Low network signal** | Delayed incident reporting and response |
| **Low prediction accuracy** | Single-model AI systems miss complex risk patterns |
| **Lack of transparency** | AI decisions are opaque — authorities can't trust or act on them |
| **Privacy concerns** | Tourist personal data shared without consent |

This system directly addresses all four problems with a unified, offline-capable platform.

---

## ✨ Key Features

### 🤖 AI-Powered Risk Assessment
- **Ensemble Model** combining Random Forest + Gradient Boosting for higher accuracy
- Real-time risk classification: **Safe** / **Warning** / **Danger**
- Trained on 10 contextual features (terrain, weather, signal, altitude, etc.)

### 🧠 Explainable AI with SHAP
- Every AI prediction comes with a **SHAP explanation**
- Authorities can see *why* a tourist was flagged as at-risk
- Feature importance visualizations for full transparency

### 🗺️ Geo-Fencing & Live Monitoring
- Interactive **Folium** maps with configurable safe zones
- Real-time breach detection when a tourist exits a safe zone
- Incidents automatically logged to the local database

### 🔒 Consent-Based Privacy System
- Tourist data is shared **only** with explicit consent
- In emergencies without consent — only location (no personal info) is shared
- Full compliance with privacy-first design principles

### 📡 Offline-First Architecture
- **SQLite** local database works without internet
- Designed for low-signal, remote tourist destinations
- Syncs when connectivity is restored

### 📊 Analytics Dashboard
- Incident type distribution & severity analysis
- Risk level breakdown charts
- Real-time metric cards with live status indicators

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────┐
│              Streamlit Web Dashboard              │
│  ┌─────────────┐ ┌─────────────┐ ┌────────────┐  │
│  │  Live       │ │  AI Risk    │ │  Geo-Fence  │  │
│  │  Dashboard  │ │  Assessment │ │  Monitor    │  │
│  └──────┬──────┘ └──────┬──────┘ └──────┬─────┘  │
│         │               │               │         │
│  ┌──────┴───────────────┴───────────────┴──────┐  │
│  │           Core Processing Engine            │  │
│  │  ┌──────────┐  ┌───────┐  ┌──────────────┐  │  │
│  │  │ Ensemble │  │ SHAP  │  │ Privacy &    │  │  │
│  │  │ ML Model │  │ XAI   │  │ Consent Mgr  │  │  │
│  │  └──────────┘  └───────┘  └──────────────┘  │  │
│  └──────────────────┬──────────────────────────┘  │
│                     │                             │
│  ┌──────────────────┴──────────────────────────┐  │
│  │          SQLite Local Database               │  │
│  │  tourists │ incidents │ zone_alerts          │  │
│  └─────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit, Plotly, Folium |
| **ML Engine** | scikit-learn (Random Forest, Gradient Boosting) |
| **Explainability** | SHAP (SHapley Additive exPlanations) |
| **Database** | SQLite (offline-capable) |
| **Mapping** | Folium + streamlit-folium |
| **Language** | Python 3.10+ |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Naga-srivalli/smart-tourist-safety-monitoring-and-incidence-response-using-AI.git
   cd smart-tourist-safety-monitoring-and-incidence-response-using-AI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run tourist_safety_system.py
   ```

4. **Open in browser**
   Navigate to `http://localhost:8501` in your web browser.

---

## 📂 Project Structure

```
├── tourist_safety_system.py   # Main application (UI + ML + Database)
├── requirements.txt           # Python dependencies
├── .gitignore                 # Excludes DB files & sensitive data
└── README.md                  # Project documentation
```

---

## 📱 Dashboard Modules

### 1. 📊 Live Dashboard
Real-time overview showing safe/warning/danger tourist counts, zone alert statuses (color-coded), signal coverage per zone, and offline mode indicators.

### 2. 🤖 AI Risk Assessment
Input tourist parameters (age, terrain, weather, signal, altitude, etc.) and receive an instant risk classification with SHAP-based explanations showing exactly which factors contributed to the prediction.

### 3. 🗺️ Geo-Fence Monitor
Interactive map powered by Folium with pre-configured safe zones. Simulate tourist positions and detect geofence breaches in real time. Breaches are logged as incidents in the local database.

### 4. 🔒 Privacy & Consent
Register tourists with optional emergency data-sharing consent. Simulates emergency vs. normal access modes — personal data is only revealed when both emergency mode is active and the tourist has consented.

### 5. 📈 Analytics
Visual breakdowns of incidents by type, severity levels, and risk category distributions using interactive Plotly charts.

---

## 🔬 AI Model Details

| Model | Type | Ensemble Role |
|-------|------|---------------|
| **Random Forest** | Bagging | Reduces variance, handles noisy features |
| **Gradient Boosting** | Boosting | Reduces bias, captures complex patterns |

### Input Features (10 total)

| Feature | Description |
|---------|-------------|
| `age` | Tourist age |
| `signal_strength` | Network signal (0–100%) |
| `distance_from_safe_zone` | Distance from nearest safe zone (km) |
| `time_since_last_checkin` | Hours since last check-in |
| `weather_severity` | Weather condition severity (1–5) |
| `terrain_difficulty` | Terrain difficulty rating (1–5) |
| `group_size` | Size of tourist group |
| `experience_level` | Tourist experience rating (1–5) |
| `temperature` | Ambient temperature (°C) |
| `altitude` | Current altitude (meters) |

### Output Classes

| Class | Meaning |
|-------|---------|
| 🟢 **Safe** | Low risk — no action needed |
| 🟡 **Warning** | Moderate risk — monitor closely |
| 🔴 **Danger** | High risk — immediate response required |

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

## 👥 Authors

- **Naga Srivalli** — [GitHub](https://github.com/Naga-srivalli)

---
