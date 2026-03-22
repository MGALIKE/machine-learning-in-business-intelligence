# Belek Analytics — Churn Intelligence Platform 🔮

[![Next.js](https://img.shields.io/badge/Next.js-14-black?style=for-the-badge&logo=next.js)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-18-blue?style=for-the-badge&logo=react)](https://reactjs.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge&logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)

An enterprise-grade, end-to-end **Machine Learning Pipeline** and **Business Intelligence Dashboard** designed to predict customer churn, isolate behavioral risk factors, and provide actionable, data-driven retention strategies.

Built by **Team Belek**.

---

## 🌟 Key Features

### 🧠 Advanced Machine Learning Engine
- **Stacking Ensemble Architecture**: Utilizes a meta-model combining **XGBoost**, **LightGBM**, and **Random Forest** classifiers to achieve incredibly high accuracy.
- **Imbalance Handling**: Implements **SMOTE** (Synthetic Minority Over-sampling Technique) to ensure the model correctly identifies rare churn events without heavy bias.
- **Temperature Softening**: Raw probability logits are mathematically calibrated ($T=2.5$) to provide smooth, continuous, and highly organic risk scores rather than polarized 0% or 100% predictions.
- **Hyperparameter Tuning**: Optuna-driven automated search for optimal tree depths, learning rates, and subset features.

### 💻 Next.js Business Intelligence Dashboard
- **Server-Side Rendered (SSR)**: Built on the Next.js App Router for aggressive JavaScript bundle code-splitting and instantaneous initial page loads.
- **Stunning Glassmorphism UI**: A deeply immersive, modern dark-mode aesthetic utilizing raw CSS radial gradients, `backdrop-filter` blurs, and `framer-motion` staggered micro-animations.
- **Infinite Scroll Customer Grid**: A highly scalable `IntersectionObserver` React data table capable of lazy-loading thousands of ML-scored holdout users without sacrificing frame rates.
- **Real-Time Filtering**: Instantaneous client-side search indexing and status-filtering (Healthy, Monitoring, At Risk) directly tied to the algorithm's predictions.

---

## 🏗️ Architecture & Stack

### Frontend (User Interface)
- **Framework**: Next.js 14 (App Router)
- **Styling**: Tailwind CSS / Vanilla CSS (Hardware-accelerated animations)
- **Data Visualization**: Recharts (Responsive SVG Canvas)
- **Animation**: Framer Motion
- **Icons**: Lucide React

### Backend (Data Science Pipeline)
- **Language**: Python 3.10+
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, XGBoost, LightGBM, Imbalanced-Learn
- **Extraction**: Custom scripts bridging `.csv` outputs seamlessly to frontend `JSON` data endpoints.

---

## 🚀 Getting Started

### 1. Data Science Pipeline (Python)
To regenerate the models, tune hyperparameters, or extract new dashboard data:

```bash
# Clean the raw dataset
python clean.py

# Train the Ensemble Models (XGBoost, LightGBM, RF)
python train.py

# Export ML metrics & predictions to the Next.js Frontend
python extract_dashboard_data.py
```

### 2. Frontend Dashboard (Next.js)
To run the Business Intelligence dashboard locally:

```bash
cd frontend

# Install dependencies
npm install

# Run the development server
npm run dev
```

Navigate to `http://localhost:3000` to view the application.

---

## 📊 Deployment (Vercel)

The frontend is specifically optimized for Edge deployment on **Vercel**. 
Because it utilizes static Server Components for the heavy routing and isolated Client Components for the interactive charts, the production build footprint is virtually zero.

```bash
# Execute a production build locally to verify code-splitting
npm run build

# Start the optimized production server
npm run start
```

---

*Designed and engineered with ❤️ for complex data solving.*
