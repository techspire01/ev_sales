
# EV Sales Analysis by State — India

A data analytics / reporting project to analyze electric vehicle (EV) sales across different Indian states, report trends, and visualize insights.

---

## Project Overview

This repository aims to explore and analyze state-wise electric vehicle sales in India. Using a dataset of EV sales by state, the project provides a detailed analysis, generates a report, and enables visual understanding of how EV adoption is evolving across states and over time.

Key components:

* Dataset (`Electric Vehicle Sales by State in India.csv`) containing state-level data of EV sales.
* Python analysis script (`ev.py`) to process, analyze, and visualize the data.
* Analytical report files (`EV_Sales_Analysis_Report.html` and `.pdf`) summarizing insights, trends, and state comparisons.
* Supporting files (e.g. `.gitignore`, etc.) for clean workflow.

---

## Objectives

* Understand which Indian states have the highest EV sales, and how they compare.
* Identify growth trends: which states are accelerating, which are lagging.
* Provide visualizations (charts, graphs) to make comparisons easier.
* Support decision making: e.g. where policies or incentives may be needed, or where market potential is high.

---

## How It Works

1. **Data Loading** — The dataset is loaded, cleaned, missing values handled (if any).
2. **Exploratory Data Analysis (EDA)** — Basic summary statistics, state-wise totals, growth calculations, etc.
3. **Visualization** — Graphs for comparing states, time trends, maybe maps, etc.
4. **Reporting** — The outputs (HTML / PDF) include charts, key metrics, and summary of findings.

---

## Technologies Used

* Python (for data analysis / scripting)
* Data files (CSV)
* Libraries for data manipulation and visualization (e.g. pandas, matplotlib / seaborn / plotly etc.)
* Report generation (converting analysis into a shareable HTML and PDF)

---

## Key Findings (example / possible)

Here are types of findings the report may present (you can fill actual numbers):

* Top-states by EV sales (cumulative)
* States with fastest growth rate
* Differences in EV adoption trend over time across states
* Insights about how state policies / infrastructure might correlate with EV sales

---

## How to Run / Use

1. Clone the repository.
2. Ensure required libraries are installed (list them in `requir.txt` if not already).
3. Run `ev.py` to process data and generate outputs.
4. View the generated report (`EV_Sales_Analysis_Report.html`) or PDF version.

---

## File Structure

| File / Folder                                  | Purpose                                                            |
| ---------------------------------------------- | ------------------------------------------------------------------ |
| `Electric Vehicle Sales by State in India.csv` | The raw data source for analysis.                                  |
| `ev.py`                                        | The Python script that does the data processing and visualization. |
| `EV_Sales_Analysis_Report.html` / `.pdf`       | The generated report summarizing the analysis.                     |
| `.gitignore`                                   | Ignored files / cleanup stuff.                                     |
| `re.txt`                                       | *\[If relevant: Describe what this is]*                            |
| `d_data`                                       | *\[If present: Contains data etc.]*                                |

---

## Potential Enhancements

* Add more time-series data (over more years) if available.
* Introduce interactive dashboards (e.g. using Plotly Dash or Streamlit).
* Geographical visualizations / maps to show spatial distribution.
* Correlate with infrastructure data (charging stations, government policy, incentives) to understand drivers.
* Forecasting EV sales for states.
