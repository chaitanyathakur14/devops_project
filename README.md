# Online Retail MLOps & Customer Value Platform

An end-to-end MLOps project implementing **Git**, **DVC (Data Version Control)**, **Scikit-Learn**, and **Streamlit** to predict Customer Lifetime Value (CLV) from historical retail transactions.

## Project Architecture & Flow
1. **Data Tracking:** Large raw datasets (`online_retail_II.xlsx`) are versioned using **DVC**.
2. **Data Processing:** Cleans transaction logs, removes cancellations, and computes total spend (`src/process_data.py`).
3. **Model Training:** Aggregates data into RFM features and trains a **Random Forest Regressor** (`src/train.py`).
4. **Interactive Dashboard:** Serves real-time predictions and EDA visualizations via **Streamlit** (`app.py`).

## Tech Stack
* **Languages:** Python
* **ML/Data:** Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib
* **MLOps / DevOps:** Git, DVC, Joblib
* **UI:** Streamlit

https://devopsproject-7xjvd56wzabyfrxcn7w2qx.streamlit.app/

E009
## How to Run Locally
1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt

