# callbacks.py
"""
Robust callbacks for Global Cost of Living Dashboard
- Loads real CSVs if present (merged into app_df)
- Falls back to demo data otherwise (demo_df)
- Safe plotting helpers to avoid Invalid value / tiny-matrix issues
"""

import os
from dash import Input, Output, State, callback, no_update, dash_table, dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# -------------------------
# Config / Paths
# -------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, '..', 'data')  # adjust if needed
WB_PATH = os.path.join(DATA_DIR, 'world_bank_sample_data.csv')
TRENDS_PATH = os.path.join(DATA_DIR, 'google_trends_sample.csv')

# -------------------------
# Helpers
# -------------------------
def create_demo_figure(message="Visuals will appear here"):
    """Safe placeholder Plotly figure."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="gray")
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=420,
        plot_bgcolor='white',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def safe_corr_heatmap(df, cols, title="Correlation"):
    """Return a heatmap figure or a placeholder if insufficient data."""
    try:
        sub = df[cols].dropna()
    except Exception:
        return create_demo_figure("Correlation not available")
    if sub.shape[0] < 2 or len(cols) < 2:
        return create_demo_figure("Not enough data for correlation")
    corr = sub.corr()
    if corr.isna().all().all() or corr.size == 1:
        return create_demo_figure("Correlation not available")
    return px.imshow(corr, text_auto=True, title=title)

def safe_parse_date(series):
    """Parse a series into month-end timestamps (use 'ME' to avoid FutureWarning)."""
    s = pd.to_datetime(series, errors='coerce')
    # Convert to month period then to month-end timestamp using 'ME'
    try:
        s = s.dt.to_period('M').dt.to_timestamp('ME')
    except Exception:
        # Fallback: coerce to month-end by taking last day of month
        s = s.dt.to_period('M').dt.to_timestamp('ME', how='end')
    return s

def normalize_columns(df, mapping):
    """Rename columns using mapping if they exist in df."""
    # mapping keys: possible column names in CSV; values: desired canonical names
    present = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=present)

def upsample_monthly(df, value_col='value'):
    """
    Accept df with columns ['Date','Country', value_col] where Date may be sparse.
    Returns monthly-resampled dataframe per country with interpolation.
    """
    # drop rows without Date/Country
    df = df.dropna(subset=['Date', 'Country']).copy()
    if df.empty:
        return pd.DataFrame(columns=['Date', 'Country', value_col])

    df = df.set_index('Date')
    out = []
    for c, g in df.groupby('Country'):
        # ensure index is DatetimeIndex
        g = g.sort_index()
        start = g.index.min()
        end = g.index.max()
        if pd.isna(start) or pd.isna(end):
            continue
        idx = pd.date_range(start.to_period('M').to_timestamp('ME'), end.to_period('M').to_timestamp('ME'), freq='M')
        # reindex and interpolate
        g2 = g.reindex(idx)
        # Interpolate numeric columns only
        numeric_cols = g2.select_dtypes(include='number').columns
        if len(numeric_cols) > 0:
            g2[numeric_cols] = g2[numeric_cols].interpolate(method='time').ffill().bfill()
        g2['Country'] = c
        g2 = g2.reset_index().rename(columns={'index': 'Date'})
        out.append(g2[['Date','Country', value_col]] if value_col in g2.columns else g2.reset_index()[['Date','Country']])
    if out:
        return pd.concat(out, ignore_index=True)
    return pd.DataFrame(columns=['Date', 'Country', value_col])

# -------------------------
# Data loader
# -------------------------
def load_data():
    """
    Load and merge WB CPI and Google Trends CSVs.
    Returns (merged_df, error_message_or_None)
    """
    if not os.path.exists(WB_PATH) or not os.path.exists(TRENDS_PATH):
        return None, f"Data files missing. Expected: {WB_PATH} and {TRENDS_PATH}"

    try:
        wb = pd.read_csv(WB_PATH)
        trends = pd.read_csv(TRENDS_PATH)
    except Exception as e:
        return None, f"Error reading CSVs: {e}"

    # Normalize common names
    wb = normalize_columns(wb, {
        'cpi': 'Cost_Index', 'CPI': 'Cost_Index', 'value': 'Cost_Index',
        'date': 'Date', 'month': 'Date', 'year_month': 'Date',
        'country': 'Country', 'country_name': 'Country', 'Country Name': 'Country'
    })
    trends = normalize_columns(trends, {
        'search_inflation': 'search_inflation', 'trend': 'search_inflation', 'search_index': 'search_inflation',
        'date': 'Date', 'month': 'Date', 'country': 'Country', 'country_name': 'Country'
    })

    # Heuristics: ensure Date and Country exist
    if 'Date' not in wb.columns and len(wb.columns) >= 1:
        wb = wb.rename(columns={wb.columns[0]: 'Date'})
    if 'Country' not in wb.columns:
        # try to guess a country-like column
        possible = [c for c in wb.columns if 'country' in c.lower()]
        if possible:
            wb = wb.rename(columns={possible[0]: 'Country'})
    if 'Cost_Index' not in wb.columns:
        numeric_cols = wb.select_dtypes(include='number').columns.tolist()
        if numeric_cols:
            wb = wb.rename(columns={numeric_cols[0]: 'Cost_Index'})
    if 'Date' not in trends.columns and len(trends.columns) >= 1:
        trends = trends.rename(columns={trends.columns[0]: 'Date'})
    if 'Country' not in trends.columns:
        possible = [c for c in trends.columns if 'country' in c.lower()]
        if possible:
            trends = trends.rename(columns={possible[0]: 'Country'})

    # Parse dates robustly
    wb['Date'] = safe_parse_date(wb['Date']) if 'Date' in wb.columns else pd.NaT
    trends['Date'] = safe_parse_date(trends['Date']) if 'Date' in trends.columns else pd.NaT

    # Ensure Country is string
    if 'Country' in wb.columns:
        wb['Country'] = wb['Country'].astype(str)
    else:
        return None, "World Bank CSV missing 'Country' column (or equivalent)."

    if 'Country' in trends.columns:
        trends['Country'] = trends['Country'].astype(str)
    else:
        # If trends missing country, set a default to allow merge; but warn
        trends['Country'] = 'Unknown'

    # Ensure Cost_Index present
    if 'Cost_Index' not in wb.columns:
        return None, "World Bank CSV missing cost/inflation column (expected Cost_Index/CPI/value)."

    # Normalize trends column
    if 'search_inflation' not in trends.columns:
        numeric_cols = trends.select_dtypes(include='number').columns.tolist()
        if numeric_cols:
            trends = trends.rename(columns={numeric_cols[0]: 'search_inflation'})
        else:
            trends['search_inflation'] = 0.0

    # Keep only necessary columns
    wb_small = wb[['Date','Country','Cost_Index']].dropna(subset=['Date','Country'])
    trends_small = trends[['Date','Country','search_inflation']].dropna(subset=['Date','Country'])

    # Upsample to monthly per country
    try:
        wb_monthly = upsample_monthly(wb_small, value_col='Cost_Index')
        trends_monthly = upsample_monthly(trends_small.rename(columns={'search_inflation':'search_inflation'}), value_col='search_inflation')
    except Exception as e:
        return None, f"Error during monthly upsampling: {e}"

    # Merge
    merged = pd.merge(wb_monthly, trends_monthly, on=['Date','Country'], how='left')
    if 'search_inflation' in merged.columns:
        merged['search_inflation'] = merged['search_inflation'].fillna(0.0)
    merged = merged.sort_values(['Country','Date']).reset_index(drop=True)

    # Coerce Cost_Index numeric
    merged['Cost_Index'] = pd.to_numeric(merged['Cost_Index'], errors='coerce')
    # if Cost_Index is still all NaN, fail
    if merged['Cost_Index'].dropna().empty:
        return None, "After parsing, Cost_Index contains no numeric values."

    # Forward fill any remaining NaNs within groups
    merged['Cost_Index'] = merged.groupby('Country')['Cost_Index'].apply(lambda g: g.fillna(method='ffill').fillna(method='bfill'))

    return merged, None

# -------------------------
# Load data at import time (safe)
# -------------------------
app_df, load_error = load_data()

# If loading failed, create demo data
def create_demo_data():
    dates = pd.date_range('2020-01-01', periods=36, freq='M')
    countries = ['United States', 'Germany', 'Japan', 'UK', 'Canada']
    rows = []
    for c in countries:
        vals = np.random.normal(100, 5, len(dates)) + np.linspace(0, 8, len(dates))
        trend = np.random.rand(len(dates)) * 2
        for d, v, t in zip(dates, vals, trend):
            rows.append({'Date': d, 'Country': c, 'Cost_Index': max(0, v + t), 'search_inflation': np.random.rand() * 10})
    return pd.DataFrame(rows)

if app_df is None:
    demo_df = create_demo_data()
else:
    demo_df = app_df.copy()

# -------------------------
# Callbacks Registration
# -------------------------
def register_callbacks(app):
    @app.callback(
        Output('main-graph', 'figure'),
        Output('secondary-graph', 'figure'),
        Output('correlation-heatmap', 'figure'),
        Output('analysis-results', 'data'),
        Input('update-button', 'n_clicks'),
        State('country-dropdown', 'value'),
        State('analysis-type', 'value'),
        State('forecast-slider', 'value'),
        State('comparison-countries', 'value'),
        prevent_initial_call=True
    )
    def update_dashboard(n_clicks, country, analysis_type, forecast_months, comparison_countries):
        if not n_clicks:
            return create_demo_figure("Click 'Update Analysis'"), create_demo_figure(), create_demo_figure(), {}

        try:
            df = app_df if app_df is not None else demo_df

            # Available countries
            available_countries = df['Country'].dropna().unique().tolist()
            if not available_countries:
                return create_demo_figure("No country data available"), create_demo_figure(), create_demo_figure(), {}

            # If selection invalid, fallback but show message
            if not country or country not in available_countries:
                note = f"Selected country '{country}' not found. Showing '{available_countries[0]}' instead."
                country = available_countries[0]
                main_fig = create_demo_figure(note)
                # continue with visuals for the fallback country below

            country_data = df[df['Country'] == country].sort_values('Date')

            if country_data['Cost_Index'].dropna().empty:
                return create_demo_figure(f"No Cost_Index data for {country}"), create_demo_figure(), create_demo_figure(), {}

            # --------- Forecast ----------
            if analysis_type == 'forecast':
                last_date = country_data['Date'].max()
                # compute slope robustly (use last 12 months if available)
                if len(country_data) >= 12:
                    slope = (country_data['Cost_Index'].iloc[-1] - country_data['Cost_Index'].iloc[-12]) / 12.0
                else:
                    slope = float(country_data['Cost_Index'].diff().fillna(0).mean()) or 0.5
                forecast_dates = pd.date_range(last_date + pd.Timedelta(days=28), periods=int(forecast_months), freq='M')
                last_value = float(country_data['Cost_Index'].iloc[-1])
                forecast_values = last_value + slope * np.arange(1, len(forecast_dates) + 1)
                forecast_df = pd.DataFrame({'Date': forecast_dates, 'Cost_Index': forecast_values, 'Type': 'Forecast'})
                actual_df = country_data[['Date','Cost_Index']].copy()
                actual_df['Type'] = 'Actual'
                combined = pd.concat([actual_df, forecast_df], ignore_index=True)
                main_fig = px.line(combined, x='Date', y='Cost_Index', color='Type', title=f"{country} Forecast ({forecast_months} mo)")
                try:
                    main_fig.add_vline(x=last_date, line_dash='dash', line_color='gray')
                except Exception:
                    pass
                secondary_fig = px.line(country_data, x='Date', y='search_inflation', title=f"{country}: Search Interest") if 'search_inflation' in country_data.columns else create_demo_figure("No search data")
                heatmap_fig = safe_corr_heatmap(country_data, ['Cost_Index','search_inflation'], title=f"{country}: Correlation")

            # --------- Correlation ----------
            elif analysis_type == 'correlation':
                main_fig = px.line(country_data, x='Date', y='Cost_Index', title=f"{country}: Cost Index")
                if 'search_inflation' in country_data.columns and not country_data['search_inflation'].dropna().empty:
                    scatter_df = country_data.dropna(subset=['search_inflation','Cost_Index'])
                    if scatter_df.shape[0] >= 2:
                        secondary_fig = px.scatter(scatter_df, x='search_inflation', y='Cost_Index', trendline='ols', title=f"{country}: Search vs CPI")
                    else:
                        secondary_fig = create_demo_figure("Not enough points for scatter")
                    heatmap_fig = safe_corr_heatmap(country_data, ['Cost_Index','search_inflation'], title=f"{country}: Correlation")
                else:
                    secondary_fig = create_demo_figure("No search data available")
                    heatmap_fig = create_demo_figure("Correlation not available")

            # --------- Comparison ----------
            elif analysis_type == 'comparison':
                comps = comparison_countries if comparison_countries else available_countries[:3]
                comp_data = df[df['Country'].isin(comps)]
                if comp_data.empty:
                    main_fig = create_demo_figure("No comparison data available")
                else:
                    main_fig = px.line(comp_data, x='Date', y='Cost_Index', color='Country', title=f"Comparison: {', '.join(comps)}")
                secondary_fig = create_demo_figure("Secondary placeholder")
                heatmap_fig = create_demo_figure("Correlation placeholder")

            else:
                main_fig = create_demo_figure()
                secondary_fig = create_demo_figure()
                heatmap_fig = create_demo_figure()

            results_data = {'country': country, 'analysis_type': analysis_type, 'status': 'real' if app_df is not None else 'demo'}
            return main_fig, secondary_fig, heatmap_fig, results_data

        except Exception as e:
            print("update_dashboard error:", e)
            return create_demo_figure(f"Error: {e}"), create_demo_figure(), create_demo_figure(), {}

    # Tab content callback
    @app.callback(
        Output('tab-content', 'children'),
        Input('main-tabs', 'active_tab')
    )
    def update_tab_content(active_tab):
        if active_tab == "tab-data":
            df = app_df if app_df is not None else demo_df
            # Present only a preview in the table to avoid huge payloads
            return dash_table.DataTable(
                data=df.head(50).to_dict('records'),
                columns=[{'name': c, 'id': c} for c in df.columns],
                page_size=10,
                style_table={'overflowX': 'auto'}
            )
        elif active_tab == "tab-insights":
            return html.Div([
                html.H4("Insights"),
                html.P(load_error if load_error else "Data loaded successfully."),
                html.Ul([
                    html.Li("Forecast: simple linear demo/proxy forecast (replace with Prophet for production)"),
                    html.Li("Correlation: Pearson correlation between Cost_Index and search interest"),
                    html.Li("Comparison: multi-country time series view")
                ])
            ])
        else:
            return html.Div([
                html.H5("Visual Analysis"),
                html.P("Click 'Update Analysis' to generate graphs.")
            ])

# Print basic load status for debugging when module imported
print("callbacks.py loaded. Data status:", "real data" if app_df is not None else f"demo mode ({load_error})")
