import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from PIL import Image as PILImage
from transformers import pipeline
import base64
import os

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

def aggregate_data_for_plot(df, x, y, kind):
    """
    Aggregate data appropriately for plotting based on chart kind.
    For bar, box, scatter plots with categorical x, aggregate y by sum or mean.
    """
    if kind in ['bar', 'box', 'scatter']:
        if df[x].dtype == 'object' or df[x].dtype.name == 'category':
            # Aggregate y by sum for bar, mean for box and scatter
            if kind == 'bar':
                agg_df = df.groupby(x)[y].sum().reset_index()
            else:
                agg_df = df.groupby(x)[y].mean().reset_index()
            return agg_df
    # For other kinds or numeric x, return original df
    return df

def auto_plotly_fig(df, x, y, kind="bar"):
    # Aggregate data if needed
    plot_df = aggregate_data_for_plot(df, x, y, kind)

    title, xlabel, ylabel, desc = generate_description(plot_df, x, y, kind)

    if kind == "line":
        fig = px.line(plot_df, x=x, y=y, title=title, labels={x: xlabel, y: ylabel}, markers=True)
    elif kind == "bar":
        fig = px.bar(plot_df, x=x, y=y, title=title, labels={x: xlabel, y: ylabel})
    elif kind == "hist":
        fig = px.histogram(df, x=y, nbins=30, title=title, labels={y: ylabel})
    elif kind == "box":
        fig = px.box(plot_df, x=x, y=y, title=title, labels={x: xlabel, y: ylabel})
    elif kind == "scatter":
        fig = px.scatter(plot_df, x=x, y=y, title=title, labels={x: xlabel, y: ylabel})
    else:
        fig = px.bar(plot_df, x=x, y=y, title=title, labels={x: xlabel, y: ylabel})

    # Improve x-axis label readability for categorical data
    if plot_df[x].dtype == 'object' or plot_df[x].dtype.name == 'category':
        fig.update_xaxes(tickangle=45, tickmode='array')

    # Enable hover and click data display (Plotly default)
    fig.update_traces(marker=dict(size=10), selector=dict(mode='markers'))

    return fig, desc

def generate_combined_html_report(plots_with_desc, filename="EV_Sales_Analysis_Report.html"):
    html_content = """
    <html>
    <head>
    <title>EV Sales Analysis Report</title>
    <style>
    body { font-family: Arial, sans-serif; margin: 20px; background: #f9f9f9; }
    h1 { color: #333; }
    .chart-container { margin-bottom: 50px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);}
    .description { font-size: 14px; color: #555; margin-top: 10px; }
    /* Animation for chart container */
    .chart-container {
        animation: fadeInUp 1s ease forwards;
        opacity: 0;
        transform: translateY(20px);
    }
    @keyframes fadeInUp {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    </style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
    <h1>Electric Vehicle Sales Analysis in India</h1>
    """

    # Generate introduction using NLP/ML
    intro_prompt = "Write an introduction for a report on electric vehicle sales analysis in India, including visualizations and descriptions."
    intro_generated = generator(intro_prompt, max_length=200, num_return_sequences=1, truncation=True)
    intro_text = intro_generated[0]['generated_text'].replace(intro_prompt, "").strip()
    html_content += f"<p>{intro_text}</p>"

    for i, (fig, desc) in enumerate(plots_with_desc):
        div_id = f"plot_{i}"
        fig_html = pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id=div_id)
        html_content += f'<div class="chart-container">{fig_html}<p class="description">{desc}</p></div>'

    html_content += """
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </body>
    </html>
    """

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)

def generate_pdf_report(plots_with_desc, filename="EV_Sales_Analysis_Report.pdf"):
    try:
        doc = SimpleDocTemplate(filename)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("<b>Electric Vehicle Sales Analysis in India</b>", styles['Title']))
        elements.append(Spacer(1, 20))

        # Generate introduction using NLP/ML
        intro_prompt = "Write an introduction for a report on electric vehicle sales analysis in India, including visualizations and descriptions."
        intro_generated = generator(intro_prompt, max_length=200, num_return_sequences=1, truncation=True)
        intro_text = intro_generated[0]['generated_text'].replace(intro_prompt, "").strip()
        elements.append(Paragraph(intro_text, styles['Normal']))
        elements.append(Spacer(1, 40))

        for fig, desc in plots_with_desc:
            # Convert Plotly fig to image
            img_data = pio.to_image(fig, format='png')
            img_buf = BytesIO(img_data)
            pil_img = PILImage.open(img_buf)
            pil_img.save("temp_img.png")
            elements.append(Image("temp_img.png", width=500, height=300))
            elements.append(Spacer(1, 10))
            elements.append(Paragraph(desc, styles['Normal']))
            elements.append(Spacer(1, 30))

        doc.build(elements)
        if os.path.exists("temp_img.png"):
            os.remove("temp_img.png")
    except Exception as e:
        print(f"Error generating PDF report: {e}")

# Load dataset
df = pd.read_csv("Electric Vehicle Sales by State in India.csv")

# Generate plots and descriptions with Plotly
plots = []
plots.append(auto_plotly_fig(df, "Year", "EV_Sales_Quantity", kind="line"))
plots.append(auto_plotly_fig(df, "Month_Name", "EV_Sales_Quantity", kind="line"))
plots.append(auto_plotly_fig(df, "State", "EV_Sales_Quantity", kind="bar"))
plots.append(auto_plotly_fig(df, "Vehicle_Category", "EV_Sales_Quantity", kind="bar"))
plots.append(auto_plotly_fig(df, "Vehicle_Type", "EV_Sales_Quantity", kind="bar"))
plots.append(auto_plotly_fig(df, "Year", "EV_Sales_Quantity", kind="box"))
plots.append(auto_plotly_fig(df, "EV_Sales_Quantity", "EV_Sales_Quantity", kind="hist"))

# Additional 10 charts
plots.append(auto_plotly_fig(df, "State", "EV_Sales_Quantity", kind="box"))
plots.append(auto_plotly_fig(df, "Year", "EV_Sales_Quantity", kind="scatter"))
plots.append(auto_plotly_fig(df, "Month_Name", "EV_Sales_Quantity", kind="box"))
plots.append(auto_plotly_fig(df, "Vehicle_Category", "EV_Sales_Quantity", kind="hist"))
plots.append(auto_plotly_fig(df, "Vehicle_Type", "EV_Sales_Quantity", kind="scatter"))
plots.append(auto_plotly_fig(df, "Year", "EV_Sales_Quantity", kind="hist"))
plots.append(auto_plotly_fig(df, "Month_Name", "EV_Sales_Quantity", kind="bar"))
plots.append(auto_plotly_fig(df, "State", "EV_Sales_Quantity", kind="hist"))
plots.append(auto_plotly_fig(df, "Vehicle_Category", "EV_Sales_Quantity", kind="box"))
plots.append(auto_plotly_fig(df, "Vehicle_Type", "EV_Sales_Quantity", kind="line"))

# Generate combined styled interactive HTML report
generate_combined_html_report(plots, "EV_Sales_Analysis_Report.html")

# Generate PDF report combining all charts and descriptions
generate_pdf_report(plots, "EV_Sales_Analysis_Report.pdf")

print("Combined interactive HTML and PDF EV Sales Analysis Reports generated successfully with animations and styling.")
