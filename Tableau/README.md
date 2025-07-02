# Data Analysis and Visualizations

## Leveraging Python, Pandas, NumPy, Dask, PySpark and Plotly with Tableau or Power BI for Data Visualization

You can integrate Python and the Plotly library with both Tableau and Power BI, allowing you to create custom visualizations and enhance data analysis capabilities.

### Tableau

TabPy 

Tableau allows integration with Python through TabPy (Tableau Python Server) or the Tableau Python Data Connector (TDC). 

TabPy allows you to run Python scripts from calculated fields in Tableau workbooks.

You can embed Plotly visualizations in Tableau dashboards by the next steps:

1. Creating the visualization with Plotly.

2. Generating an embed URL using Plotly's share function.

3. Adding the URL to a data source (e.g., Google Sheets).

4. Connecting Tableau to the data source.

5. Building sheets around the Plotly visualization.

6. Adding the sheets to a dashboard and dragging a web page object to embed the URL.

### Power BI

1. Python scripting

Enable Python scripting within Power BI Desktop to use Python visuals.

2. Add Python visuals on Power BI

Select the Python visual icon in Power BI's Visualizations pane and enable script visuals.

3. Write Python script code

In the Python script editor, write code to create your Plotly visualization. Power BI passes data to your script as a pandas DataFrame.

4. Export as an image

Since Power BI's Python visuals don't natively support interactive Plotly charts, you'll need to save the Plotly visual as an image (e.g., using kaleido) and display the image within Power BI.





