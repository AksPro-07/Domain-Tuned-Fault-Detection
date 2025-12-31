# Golden Period Selection Utility

**Project Type:** Personal Independent Project
**Domain:** Industrial Data Science / MLOps

## Project Overview

The **Golden Period Selection Utility** is a Python-based preprocessing tool designed to improve the stability and prediction accuracy of predictive maintenance models (e.g., Suresense-style architectures). It ingests high-frequency time-series sensor data and employs a "Domain-Tuned Cascade" of algorithms to identify operating ranges, detect faults, and clean data for downstream model training.

This utility addresses the challenge of noisy industrial data by automating the cleaning process, significantly reducing manual effort while ensuring explainable results through comprehensive reporting.

## Key Features

* **Univariate & Multivariate Fault Detection:**
    * Implements a suite of algorithms including **Isolation Forest**, **Local Outlier Factor (LOF)**, **Rate of Change (ROC)**, and **IQR**.
    * Includes **Robust Covariance** analysis to detect multivariate anomalies (correlations between tags).
    * Scalable across 100+ tags with >90% precision for operating-range identification.

* **Domain-Tuned Cascade Logic:**
    * Features a voting mechanism allowing users to combine multiple detection methods.
    * Improves result consistency by requiring consensus between tree-based, density-based, and statistical models before flagging an anomaly.

* **Interactive Streamlit Frontend:**
    * **Data Preview:** Visualize raw sensor data.
    * **Manual Cleaning:** Interactively select and delete specific data points using box/lasso tools.
    * **Configuration:** Dynamic tuning of contamination factors and voting thresholds.

* **Reporting & Golden Period Extraction:**
    * Generates a "Coverage & Anomaly Report" detailing outlier percentages per tag.
    * Exports a clean "Golden Period" dataset (CSV) ready for model training.

## Tech Stack

* **Frontend:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (Isolation Forest, LOF, Elliptic Envelope)
* **Visualization:** Plotly Express / Graph Objects

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd golden-period-utility
    ```

2.  **Create a virtual environment (Optional but recommended):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

2.  **Workflow:**
    * **Upload:** Load your time-series CSV (first column must be timestamps).
    * **Preview:** Inspect individual tags.
    * **Manual Clean (Optional):** Visually remove obvious errors if needed.
    * **Cascade Detection:** Select algorithms (e.g., Isolation Forest + ROC) and set the voting threshold. Click "Run Cascade Analysis".
    * **Report:** View the anomaly summary and download the cleaned "Golden Period" dataset.

## Project Structure

```text
├── app.py                # Main Streamlit application (Frontend)
├── algorithms.py         # Outlier detection logic (Backend)
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── dummy_sensor_data.csv # Sample dataset for testing