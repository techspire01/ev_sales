import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from PIL import Image as PILImage
from transformers import pipeline

# Load the text generation model
generator = pipeline('text-generation', model='gpt2')

def generate_description(df, x, y, kind):
    """
    Auto-generate plot title, axis labels, and detailed description using NLP/ML
    """
    title = f"Analysis of {y} by {x}"
    xlabel = x.replace("_", " ").title()
    ylabel = y.replace("_", " ").title()

    # Stats for prompt
    mean_val = df[y].mean()
    max_val = df[y].max()
    min_val = df[y].min()
    total = df[y].sum()
    count = len(df[y])

    prompt = f"Describe a {kind} chart showing {ylabel} by {xlabel} for electric vehicle sales in India. The chart has {count} data points, average {mean_val:.2f}, total {total:.2f}, max {max_val:.2f}, min {min_val:.2f}. Explain what the chart shows and insights."

    generated = generator(prompt, max_length=150, num_return_sequences=1, truncation=True)
    desc = generated[0]['generated_text'].replace(prompt, "").strip()

    return title, xlabel, ylabel, desc

def auto_plot(df, x, y, kind="bar"):
    title, xlabel, ylabel, desc = generate_description(df, x, y, kind)

    fig, ax = plt.subplots(figsize=(8,5))
    if kind == "line":
        sns.lineplot(x=x, y=y, data=df, marker="o", ax=ax)
    elif kind == "bar":
        sns.barplot(x=x, y=y, data=df, errorbar=None, ax=ax)
    elif kind == "hist":
        sns.histplot(df[y], bins=30, kde=True, ax=ax)
    elif kind == "box":
        sns.boxplot(x=x, y=y, data=df, ax=ax)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()

    # Save to BytesIO
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf, desc

def generate_pdf_report(plots_with_desc, filename="EV_Sales_Analysis_Report.pdf"):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    elements = []

    # Generate introduction using NLP/ML
    intro_prompt = "Write an introduction for a report on electric vehicle sales analysis in India, including visualizations and descriptions."
    intro_generated = generator(intro_prompt, max_length=200, num_return_sequences=1, truncation=True)
    intro_text = intro_generated[0]['generated_text'].replace(intro_prompt, "").strip()

    elements.append(Paragraph("<b>Electric Vehicle Sales Analysis in India</b>", styles['Title']))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(intro_text, styles['Normal']))
    elements.append(Spacer(1, 40))

    for plot, desc in plots_with_desc:
        elements.append(Image(plot, width=500, height=300))
        elements.append(Spacer(1, 10))
        elements.append(Paragraph(desc, styles['Normal']))
        elements.append(Spacer(1, 30))

    doc.build(elements)

# Load dataset
df = pd.read_csv("Electric Vehicle Sales by State in India.csv")

# Generate plots and descriptions
plots = []
plots.append(auto_plot(df, "Year", "EV_Sales_Quantity", kind="line"))
plots.append(auto_plot(df, "Month_Name", "EV_Sales_Quantity", kind="line"))
plots.append(auto_plot(df, "State", "EV_Sales_Quantity", kind="bar"))
plots.append(auto_plot(df, "Vehicle_Category", "EV_Sales_Quantity", kind="bar"))
plots.append(auto_plot(df, "Vehicle_Type", "EV_Sales_Quantity", kind="bar"))
plots.append(auto_plot(df, "Year", "EV_Sales_Quantity", kind="box"))
plots.append(auto_plot(df, "EV_Sales_Quantity", "EV_Sales_Quantity", kind="hist"))

# Generate PDF report
generate_pdf_report(plots, "EV_Sales_Analysis_Report.pdf")

print(" generated successfully.")
