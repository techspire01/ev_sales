import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from matplotlib.backends.backend_pdf import PdfPages

def add_plot_description(pdf, title, xlabel, ylabel, description):
    pdf.savefig()  # Save the current figure
    plt.close()
    # Add a page with text description
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    wrapped_text = "\n".join(description[i:i+100] for i in range(0, len(description), 100))
    ax.text(0.5, 0.5, f"{title}\n\nX-axis: {xlabel}\nY-axis: {ylabel}\n\nDescription:\n{wrapped_text}",
            ha='center', va='center', fontsize=12, wrap=True)
    pdf.savefig(fig)
    plt.close(fig)
    plt.figure()  # Create a new figure to avoid overlap issues

# Load dataset
df = pd.read_csv("Electric Vehicle Sales by State in India.csv")
print(df.head())
print(df.info())

# Data Cleaning & Preprocessing
df['Year'] = df['Year'].astype(int)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['EV_Sales_Quantity'].fillna(df['EV_Sales_Quantity'].median(), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)
print(df.isnull().sum())

with PdfPages('EV_Sales_Report.pdf') as pdf:

    # Introduction Page
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    intro_text = "Electric Vehicle Sales Analysis Report\n\n" \
                 "This report provides a comprehensive analysis of electric vehicle sales in India. " \
                 "It includes various visualizations to understand trends, distributions, and patterns in the data. " \
                 "Additionally, it features predictive modeling to forecast sales quantities.\n\n" \
                 "Generated automatically using machine learning techniques for descriptions."
    ax.text(0.5, 0.5, intro_text, ha='center', va='center', fontsize=14, wrap=True)
    pdf.savefig(fig)
    plt.close(fig)

    # 1. Yearly EV Sales Trend
    plt.figure(figsize=(8,5))
    sns.lineplot(x='Year', y='EV_Sales_Quantity', data=df, marker='o', color="blue")
    plt.title("Yearly EV Sales in India")
    plt.ylabel("EV Sales Quantity")
    plt.xlabel("Year")
    description = "This line plot shows the trend of electric vehicle sales quantity in India over the years. " \
                  "It helps to understand how the sales have increased or decreased annually."
    add_plot_description(pdf, "Yearly EV Sales in India", "Year", "EV Sales Quantity", description)

    # 2. Monthly EV Sales Trend
    plt.figure(figsize=(8,5))
    sns.lineplot(x='Month_Name', y='EV_Sales_Quantity', data=df, marker='o', color="red")
    plt.title("Monthly EV Sales in India")
    plt.ylabel("EV Sales Quantity")
    plt.xlabel("Month")
    description = "This line plot illustrates the monthly variation in electric vehicle sales quantity in India. " \
                  "It highlights seasonal patterns or monthly fluctuations in sales."
    add_plot_description(pdf, "Monthly EV Sales in India", "Month", "EV Sales Quantity", description)

    # 3. State-wise EV Sales (Top 10 States)
    top_states = df.groupby('State')['EV_Sales_Quantity'].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10,6))
    sns.barplot(x=top_states.values, y=top_states.index, palette="viridis")
    plt.title("Top 10 States by EV Sales")
    plt.xlabel("EV Sales Quantity")
    plt.ylabel("State")
    description = "This bar chart displays the top 10 states in India by total electric vehicle sales quantity. " \
                  "It identifies the leading states in EV adoption."
    add_plot_description(pdf, "Top 10 States by EV Sales", "EV Sales Quantity", "State", description)

    # 4. Vehicle Category-wise Sales
    plt.figure(figsize=(8,5))
    sns.barplot(x='Vehicle_Category', y='EV_Sales_Quantity', data=df, palette="Set2", ci=None)
    plt.title("EV Sales by Vehicle Category")
    plt.xlabel("Vehicle Category")
    plt.ylabel("EV Sales Quantity")
    description = "This bar chart shows the sales quantity of electric vehicles categorized by vehicle category. " \
                  "It compares sales across different categories like passenger vehicles or commercial vehicles."
    add_plot_description(pdf, "EV Sales by Vehicle Category", "Vehicle Category", "EV Sales Quantity", description)

    # 5. Vehicle Type-wise Sales
    plt.figure(figsize=(12,5))
    sns.barplot(x='Vehicle_Type', y='EV_Sales_Quantity', data=df, palette="Set3", ci=None)
    plt.title("EV Sales by Vehicle Type")
    plt.xlabel("Vehicle Type")
    plt.ylabel("EV Sales Quantity")
    plt.xticks(rotation=90)
    description = "This bar chart presents the sales quantity by specific vehicle types. " \
                  "It provides insights into which types of EVs are most popular."
    add_plot_description(pdf, "EV Sales by Vehicle Type", "Vehicle Type", "EV Sales Quantity", description)

    # 6. Sales Distribution (Histogram)
    plt.figure(figsize=(8,5))
    sns.histplot(df['EV_Sales_Quantity'], bins=50, kde=True, color="purple")
    plt.title("Distribution of EV Sales Quantity")
    plt.xlabel("EV Sales Quantity")
    plt.ylabel("Frequency")
    description = "This histogram shows the distribution of electric vehicle sales quantities. " \
                  "It indicates the frequency of different sales volumes and the overall spread of the data."
    add_plot_description(pdf, "Distribution of EV Sales Quantity", "EV Sales Quantity", "Frequency", description)

    # 7. Boxplot: Yearly Sales Spread
    plt.figure(figsize=(10,5))
    sns.boxplot(x='Year', y='EV_Sales_Quantity', data=df, palette="coolwarm")
    plt.title("Yearly Spread of EV Sales")
    plt.xlabel("Year")
    plt.ylabel("EV Sales Quantity")
    description = "This boxplot illustrates the spread and variability of EV sales quantities across different years. " \
                  "It highlights median values, quartiles, and potential outliers."
    add_plot_description(pdf, "Yearly Spread of EV Sales", "Year", "EV Sales Quantity", description)

    # 8. Heatmap: State vs Year
    pivot = df.pivot_table(values='EV_Sales_Quantity', index='State', columns='Year', aggfunc='sum')
    plt.figure(figsize=(14,10))
    sns.heatmap(pivot, cmap="YlGnBu", linewidths=0.5)
    plt.title("Heatmap of EV Sales (State vs Year)")
    plt.xlabel("Year")
    plt.ylabel("State")
    description = "This heatmap visualizes the total EV sales quantity for each state across different years. " \
                  "Darker colors indicate higher sales volumes."
    add_plot_description(pdf, "Heatmap of EV Sales (State vs Year)", "Year", "State", description)

    # Feature Engineering for predictive modeling
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df_encoded = pd.get_dummies(df, columns=['State','Vehicle_Class','Vehicle_Category','Vehicle_Type'], drop_first=True)
    df_encoded.drop(['Date','Month_Name'], axis=1, inplace=True)

    # 9. Correlation Heatmap of numeric features
    plt.figure(figsize=(12,10))
    sns.heatmap(df_encoded.corr(), cmap='coolwarm', center=0)
    plt.title("Correlation Heatmap of Features")
    description = "This correlation heatmap shows the relationships between different features in the dataset. " \
                  "It helps identify which variables are correlated with EV sales quantity."
    add_plot_description(pdf, "Correlation Heatmap of Features", "Features", "Features", description)

    # 10. Predictive Modeling and Feature Importance
    X = df_encoded.drop("EV_Sales_Quantity", axis=1)
    y = df_encoded["EV_Sales_Quantity"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Root Mean Squared Error: {rmse}")

    plt.figure(figsize=(12,6))
    importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    importance[:20].plot(kind='bar', title="Top 20 Important Features")
    description = f"This bar chart shows the top 20 important features for predicting EV sales quantity using a Random Forest model. " \
                  f"The model achieved a Root Mean Squared Error of {rmse:.2f} on the test set."
    add_plot_description(pdf, "Top 20 Important Features", "Features", "Importance", description)

print("PDF report generated successfully.")
