# callbacks.py - COMPLETE UPDATED VERSION
"""
Robust callbacks for Global Cost of Living Dashboard
Updated to match your specific CSV structures
"""

import os
from dash import Input, Output, State, callback, no_update, dash_table, dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime

# -------------------------
# Config / Paths
# -------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, '..', 'data')
WB_PATH = os.path.join(DATA_DIR, 'world_bank_sample_data.csv')
TRENDS_PATH = os.path.join(DATA_DIR, 'google_trends_sample.csv')

# -------------------------
# Helper Functions
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

def add_safe_vline(fig, x_value, **kwargs):
    """Safely add vertical line to plotly figure, handling timestamp issues."""
    try:
        # Convert timestamp to string if it's a pandas timestamp
        if hasattr(x_value, 'strftime'):
            x_value = x_value.strftime('%Y-%m-%d')
        fig.add_vline(x=x_value, **kwargs)
    except Exception as e:
        # Fallback to add_shape
        fig.add_shape(
            type="line",
            x0=x_value,
            x1=x_value,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color=kwargs.get('line_color', 'gray'), 
                     width=2, 
                     dash=kwargs.get('line_dash', 'solid'))
        )
        # Add annotation if provided
        if 'annotation_text' in kwargs:
            fig.add_annotation(
                x=x_value,
                y=1.02,
                yref="paper",
                text=kwargs['annotation_text'],
                showarrow=False,
                font=dict(size=10, color=kwargs.get('line_color', 'gray'))
            )
    return fig

# -------------------------
# Data loader for your specific CSV structure
# -------------------------
def load_data():
    """
    Load and merge CSVs for your specific data structure
    """
    print(f"Loading data from:")
    print(f"  World Bank: {WB_PATH}")
    print(f"  Trends: {TRENDS_PATH}")
    
    if not os.path.exists(WB_PATH):
        return None, f"World Bank file not found at: {WB_PATH}"
    if not os.path.exists(TRENDS_PATH):
        return None, f"Trends file not found at: {TRENDS_PATH}"

    try:
        # Load data with proper date parsing
        wb = pd.read_csv(WB_PATH, parse_dates=['ds'], dayfirst=True)
        trends = pd.read_csv(TRENDS_PATH, parse_dates=['ds'], dayfirst=True)
        
        print(f"\n✅ Data loaded successfully:")
        print(f"  World Bank shape: {wb.shape}")
        print(f"  World Bank columns: {list(wb.columns)}")
        print(f"  World Bank countries: {sorted(wb['country'].unique())}")
        
        print(f"\n  Trends shape: {trends.shape}")
        print(f"  Trends columns: {list(trends.columns)}")
        print(f"  Trends countries: {sorted(trends['country'].unique())}")
        
    except Exception as e:
        print(f"❌ Error reading CSVs: {e}")
        return None, f"Error reading CSVs: {e}"

    # Find common countries
    wb_countries = set(wb['country'].unique())
    trends_countries = set(trends['country'].unique())
    common_countries = sorted(wb_countries.intersection(trends_countries))
    
    print(f"\n🌍 Common countries in both datasets: {common_countries}")
    
    # Convert to standard column names
    wb_clean = wb.copy()
    wb_clean = wb_clean.rename(columns={
        'country': 'Country',
        'ds': 'Date',
        'CPI': 'Cost_Index'
    })
    
    trends_clean = trends.copy()
    trends_clean = trends_clean.rename(columns={
        'country': 'Country',
        'ds': 'Date',
        'search_inflation': 'search_inflation'
    })
    
    # World Bank data is annual, Trends data is monthly
    # We need to create monthly CPI data by forward-filling
    
    # Create monthly date range for each country
    all_countries = list(common_countries)
    monthly_data = []
    
    for country in all_countries:
        # Get country data
        wb_country = wb_clean[wb_clean['Country'] == country].copy()
        trends_country = trends_clean[trends_clean['Country'] == country].copy()
        
        if wb_country.empty or trends_country.empty:
            continue
            
        # Get date ranges
        min_date = min(wb_country['Date'].min(), trends_country['Date'].min())
        max_date = max(wb_country['Date'].max(), trends_country['Date'].max())
        
        # Create monthly index
        monthly_dates = pd.date_range(
            start=min_date,
            end=max_date,
            freq='M'
        )
        
        # Create monthly dataframe
        monthly_df = pd.DataFrame({'Date': monthly_dates})
        monthly_df['Country'] = country
        
        # Merge World Bank annual data
        wb_country['Year'] = wb_country['Date'].dt.year
        monthly_df['Year'] = monthly_df['Date'].dt.year
        
        # Merge CPI data
        merged_df = pd.merge(
            monthly_df,
            wb_country[['Year', 'Cost_Index']],
            on='Year',
            how='left'
        )
        
        # Merge Trends monthly data
        merged_df = pd.merge(
            merged_df,
            trends_country[['Date', 'search_inflation']],
            left_on='Date',
            right_on='Date',
            how='left'
        )
        
        # Forward fill CPI data within each year
        merged_df['Cost_Index'] = merged_df.groupby('Year')['Cost_Index'].ffill()
        
        # Select final columns
        final_df = merged_df[['Date', 'Country', 'Cost_Index', 'search_inflation']].copy()
        monthly_data.append(final_df)
    
    if not monthly_data:
        return None, "No data could be merged"
    
    # Combine all countries
    merged = pd.concat(monthly_data, ignore_index=True)
    
    # Drop rows with missing Cost_Index
    merged = merged.dropna(subset=['Cost_Index'])
    
    # Sort data
    merged = merged.sort_values(['Country', 'Date']).reset_index(drop=True)
    
    print(f"\n📊 Final merged data:")
    print(f"  Shape: {merged.shape}")
    print(f"  Countries: {sorted(merged['Country'].unique())}")
    print(f"  Date range: {merged['Date'].min().date()} to {merged['Date'].max().date()}")
    print(f"  Sample data:")
    print(merged.head(10))
    
    return merged, None

# Load data
app_df, load_error = load_data()

# Create demo data if loading failed
def create_demo_data():
    """Create sample data matching your CSV structure"""
    dates = pd.date_range('2020-01-01', periods=36, freq='M')
    countries = ['United States', 'United Kingdom', 'China', 'Japan', 'Germany']
    rows = []
    
    # Realistic CPI values for each country
    cpi_values = {
        'United States': np.random.normal(3.0, 0.5, 36) + np.linspace(0, 1, 36),
        'United Kingdom': np.random.normal(3.5, 0.6, 36) + np.linspace(0, 1.2, 36),
        'China': np.random.normal(2.5, 0.4, 36) + np.linspace(0, 0.8, 36),
        'Japan': np.random.normal(2.0, 0.3, 36) + np.linspace(0, 0.5, 36),
        'Germany': np.random.normal(2.8, 0.5, 36) + np.linspace(0, 1.0, 36)
    }
    
    for country in countries:
        country_cpi = cpi_values[country]
        for i, date in enumerate(dates):
            rows.append({
                'Date': date,
                'Country': country,
                'Cost_Index': max(0.5, country_cpi[i]),  # Ensure positive
                'search_inflation': np.random.rand() * 100  # Scale to 0-100
            })
    
    demo_df = pd.DataFrame(rows)
    print(f"\n⚠️ Created demo data with {len(countries)} countries")
    print(f"  Countries: {sorted(demo_df['Country'].unique())}")
    return demo_df

# Use real data if available, otherwise demo
if app_df is None or load_error:
    print(f"\n⚠️ Using demo data due to: {load_error}")
    demo_df = create_demo_data()
else:
    demo_df = app_df.copy()
    print(f"\n✅ Real data loaded successfully!")

# Make sure we have consistent data for the dashboard
working_df = demo_df.copy()
print(f"\n📈 Dashboard will show {len(working_df['Country'].unique())} countries:")
for country in sorted(working_df['Country'].unique()):
    country_data = working_df[working_df['Country'] == country]
    print(f"  - {country}: {len(country_data)} records, "
          f"CPI range: {country_data['Cost_Index'].min():.1f}-{country_data['Cost_Index'].max():.1f}")

# -------------------------
# Callbacks Registration
# -------------------------
def register_callbacks(app):
    global working_df
    
    # Control visibility callback for showing/hiding comparison controls
    @app.callback(
        [Output('forecast-controls', 'style'),
         Output('comparison-controls', 'style')],
        Input('analysis-type', 'value')
    )
    def toggle_controls(analysis_type):
        """Show/hide controls based on analysis type."""
        if analysis_type == 'comparison':
            return {'display': 'none'}, {'display': 'block'}
        else:
            return {'display': 'block'}, {'display': 'none'}
    
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
            df = working_df
            
            # Available countries
            available_countries = df['Country'].dropna().unique().tolist()
            if not available_countries:
                return create_demo_figure("No country data available"), create_demo_figure(), create_demo_figure(), {}
            
            # --------- Forecast ----------
            if analysis_type == 'forecast':
                # Validate country selection for forecast
                if not country or country not in available_countries:
                    country = available_countries[0]
                
                country_data = df[df['Country'] == country].sort_values('Date')
                
                if country_data['Cost_Index'].dropna().empty:
                    return create_demo_figure(f"No Cost_Index data for {country}"), create_demo_figure(), create_demo_figure(), {}
                
                last_date = country_data['Date'].max()
                
                # Simple linear forecast
                if len(country_data) >= 12:
                    slope = (country_data['Cost_Index'].iloc[-1] - country_data['Cost_Index'].iloc[-12]) / 12.0
                else:
                    slope = float(country_data['Cost_Index'].diff().fillna(0).mean()) or 0.01
                
                # Use proper date arithmetic with pandas offsets
                forecast_dates = pd.date_range(
                    last_date + pd.DateOffset(months=1), 
                    periods=int(forecast_months), 
                    freq='M'
                )
                
                last_value = float(country_data['Cost_Index'].iloc[-1])
                forecast_values = last_value + slope * np.arange(1, len(forecast_dates) + 1)
                
                forecast_df = pd.DataFrame({
                    'Date': forecast_dates, 
                    'Cost_Index': forecast_values, 
                    'Type': 'Forecast'
                })
                actual_df = country_data[['Date','Cost_Index']].copy()
                actual_df['Type'] = 'Actual'
                combined = pd.concat([actual_df, forecast_df], ignore_index=True)
                
                # Main forecast graph
                main_fig = px.line(
                    combined, 
                    x='Date', 
                    y='Cost_Index', 
                    color='Type', 
                    title=f"{country}: CPI Forecast ({forecast_months} months)",
                    labels={'Cost_Index': 'CPI (%)', 'Date': 'Date'},
                    color_discrete_map={'Actual': 'blue', 'Forecast': 'red'}
                )
                
                # Add forecast start line using safe method
                main_fig = add_safe_vline(
                    main_fig, 
                    last_date, 
                    line_dash='dash', 
                    line_color='gray',
                    annotation_text="Forecast Start"
                )
                
                # Add forecast details annotation
                main_fig.add_annotation(
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    text=f"<b>Forecast Details:</b><br>Slope: {slope:.3f}/month<br>Last CPI: {last_value:.2f}%",
                    showarrow=False,
                    align="left",
                    bgcolor="white",
                    bordercolor="gray",
                    borderwidth=1,
                    font=dict(size=10)
                )
                
                # Secondary graph - search trends
                if 'search_inflation' in country_data.columns and not country_data['search_inflation'].dropna().empty:
                    secondary_fig = px.line(
                        country_data, 
                        x='Date', 
                        y='search_inflation', 
                        title=f"{country}: Search Interest for Inflation",
                        labels={'search_inflation': 'Search Index', 'Date': 'Date'},
                        line_shape='spline'
                    )
                else:
                    secondary_fig = create_demo_figure("No search data available")
                
                # Heatmap - correlation
                heatmap_fig = safe_corr_heatmap(
                    country_data, 
                    ['Cost_Index','search_inflation'], 
                    title=f"{country}: CPI vs Search Correlation"
                )
                
                results_data = {
                    'country': country, 
                    'analysis_type': analysis_type, 
                    'status': 'real' if app_df is not None else 'demo',
                    'data_points': len(country_data),
                    'forecast_months': forecast_months,
                    'date_range': f"{country_data['Date'].min().date()} to {country_data['Date'].max().date()}"
                }
            
            # --------- Correlation ----------
            elif analysis_type == 'correlation':
                # Validate country selection for correlation
                if not country or country not in available_countries:
                    country = available_countries[0]
                
                country_data = df[df['Country'] == country].sort_values('Date')
                
                # Main graph - CPI over time
                main_fig = px.line(
                    country_data, 
                    x='Date', 
                    y='Cost_Index', 
                    title=f"{country}: Consumer Price Index (CPI)",
                    labels={'Cost_Index': 'CPI (%)', 'Date': 'Date'},
                    line_shape='spline'
                )
                
                # Add statistics annotation
                if len(country_data) > 0:
                    avg_cpi = country_data['Cost_Index'].mean()
                    max_cpi = country_data['Cost_Index'].max()
                    min_cpi = country_data['Cost_Index'].min()
                    
                    main_fig.add_annotation(
                        xref="paper", yref="paper",
                        x=0.02, y=0.98,
                        text=f"<b>CPI Statistics:</b><br>Avg: {avg_cpi:.2f}%<br>Max: {max_cpi:.2f}%<br>Min: {min_cpi:.2f}%",
                        showarrow=False,
                        align="left",
                        bgcolor="white",
                        bordercolor="gray",
                        borderwidth=1,
                        font=dict(size=10)
                    )
                
                # Scatter plot - Search vs CPI
                if 'search_inflation' in country_data.columns:
                    scatter_df = country_data.dropna(subset=['search_inflation','Cost_Index'])
                    if scatter_df.shape[0] >= 2:
                        # Calculate correlation
                        correlation = scatter_df['Cost_Index'].corr(scatter_df['search_inflation'])
                        
                        secondary_fig = px.scatter(
                            scatter_df, 
                            x='search_inflation', 
                            y='Cost_Index',
                            title=f"{country}: Search vs CPI (r = {correlation:.3f})",
                            labels={'search_inflation': 'Search Index', 'Cost_Index': 'CPI (%)'},
                            trendline='ols',
                            hover_data=['Date'],
                            trendline_color_override='red'
                        )
                        
                        # Add correlation strength text
                        if abs(correlation) > 0.7:
                            strength = "Strong"
                        elif abs(correlation) > 0.3:
                            strength = "Moderate"
                        else:
                            strength = "Weak"
                            
                        secondary_fig.add_annotation(
                            xref="paper", yref="paper",
                            x=0.02, y=0.98,
                            text=f"<b>Correlation:</b> {strength}",
                            showarrow=False,
                            align="left",
                            bgcolor="white",
                            bordercolor="gray",
                            borderwidth=1,
                            font=dict(size=10)
                        )
                    else:
                        secondary_fig = create_demo_figure("Not enough data for scatter plot")
                else:
                    secondary_fig = create_demo_figure("No search data available")
                
                # Heatmap - correlation matrix
                heatmap_fig = safe_corr_heatmap(
                    country_data, 
                    ['Cost_Index','search_inflation'], 
                    title=f"{country}: Correlation Heatmap"
                )
                
                results_data = {
                    'country': country, 
                    'analysis_type': analysis_type, 
                    'status': 'real' if app_df is not None else 'demo',
                    'data_points': len(country_data),
                    'date_range': f"{country_data['Date'].min().date()} to {country_data['Date'].max().date()}"
                }
            
            # --------- Comparison ----------
            elif analysis_type == 'comparison':
                # Handle comparison countries selection
                if not comparison_countries:
                    comparison_countries = available_countries[:3]
                    print(f"No comparison countries selected, defaulting to: {comparison_countries}")
                
                # Ensure we have at least 2 countries to compare
                if len(comparison_countries) < 2:
                    comparison_countries = available_countries[:min(3, len(available_countries))]
                    print(f"Need at least 2 countries for comparison, using: {comparison_countries}")
                
                comp_data = df[df['Country'].isin(comparison_countries)].copy()
                
                if comp_data.empty:
                    main_fig = create_demo_figure("No comparison data available")
                else:
                    # Sort by date for better visualization
                    comp_data = comp_data.sort_values(['Country', 'Date'])
                    
                    # Main comparison graph
                    main_fig = px.line(
                        comp_data, 
                        x='Date', 
                        y='Cost_Index', 
                        color='Country',
                        title=f"CPI Comparison: {', '.join(comparison_countries)}",
                        labels={'Cost_Index': 'CPI (%)', 'Date': 'Date'},
                        line_shape='spline',
                        hover_data=['search_inflation']
                    )
                    
                    # Add interactive features
                    main_fig.update_layout(
                        hovermode='x unified',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                
                # For comparison view, show summary statistics as secondary graph
                summary_stats = []
                for comp_country in comparison_countries:
                    country_stats = df[df['Country'] == comp_country]
                    if not country_stats.empty:
                        stats = {
                            'Country': comp_country,
                            'Avg CPI': f"{country_stats['Cost_Index'].mean():.2f}%",
                            'Max CPI': f"{country_stats['Cost_Index'].max():.2f}%",
                            'Min CPI': f"{country_stats['Cost_Index'].min():.2f}%",
                            'Latest CPI': f"{country_stats['Cost_Index'].iloc[-1]:.2f}%" if len(country_stats) > 0 else "N/A",
                            'Data Points': len(country_stats)
                        }
                        summary_stats.append(stats)
                
                if summary_stats:
                    summary_df = pd.DataFrame(summary_stats)
                    
                    secondary_fig = go.Figure(data=[
                        go.Table(
                            header=dict(
                                values=list(summary_df.columns),
                                fill_color='lightblue',
                                align='center',
                                font=dict(size=12, color='black')
                            ),
                            cells=dict(
                                values=[summary_df[col] for col in summary_df.columns],
                                fill_color='lavender',
                                align='center',
                                font=dict(size=11)
                            )
                        )
                    ])
                    
                    secondary_fig.update_layout(
                        title=f"CPI Summary Statistics",
                        height=400,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                else:
                    secondary_fig = create_demo_figure("No summary statistics available")
                
                # For comparison view, show a correlation heatmap between selected countries
                # Create a pivot table for correlation between countries
                pivot_data = df[df['Country'].isin(comparison_countries)].pivot_table(
                    index='Date', 
                    columns='Country', 
                    values='Cost_Index'
                ).dropna()
                
                if len(pivot_data) >= 2 and len(comparison_countries) >= 2:
                    correlation_matrix = pivot_data.corr()
                    
                    heatmap_fig = px.imshow(
                        correlation_matrix,
                        text_auto=True,
                        color_continuous_scale='RdBu',
                        title=f"Correlation between Countries",
                        labels=dict(color="Correlation")
                    )
                    
                    heatmap_fig.update_layout(
                        xaxis_title="Country",
                        yaxis_title="Country"
                    )
                else:
                    heatmap_fig = create_demo_figure("Select at least 2 countries with overlapping dates for correlation")
                
                results_data = {
                    'countries': comparison_countries,
                    'analysis_type': analysis_type, 
                    'status': 'real' if app_df is not None else 'demo',
                    'num_countries': len(comparison_countries),
                    'date_range': f"{comp_data['Date'].min().date()} to {comp_data['Date'].max().date()}" if not comp_data.empty else "N/A"
                }
            
            else:
                # Default fallback
                main_fig = create_demo_figure("Select an analysis type")
                secondary_fig = create_demo_figure()
                heatmap_fig = create_demo_figure()
                results_data = {}
            
            return main_fig, secondary_fig, heatmap_fig, results_data
        
        except Exception as e:
            print(f"❌ Dashboard update error: {e}")
            import traceback
            traceback.print_exc()
            error_msg = str(e)
            return (
                create_demo_figure(f"Error: {error_msg[:100]}"),
                create_demo_figure("Error occurred"),
                create_demo_figure("Error occurred"),
                {'error': error_msg}
            )
    
    # Tab content callback
    @app.callback(
        Output('tab-content', 'children'),
        Input('main-tabs', 'active_tab')
    )
    def update_tab_content(active_tab):
        if active_tab == "tab-data":
            # Show data preview
            preview_df = working_df.head(100).copy()
            preview_df['Date'] = preview_df['Date'].dt.strftime('%Y-%m-%d')
            
            return dash_table.DataTable(
                data=preview_df.to_dict('records'),
                columns=[{'name': col, 'id': col} for col in preview_df.columns],
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={
                    'backgroundColor': 'lightblue',
                    'fontWeight': 'bold'
                },
                filter_action="native",
                sort_action="native",
                sort_mode="multi",
                export_format="csv",
                export_headers="display"
            )
        elif active_tab == "tab-insights":
            return html.Div([
                html.H4("📊 Data Insights", className="mb-3"),
                html.Div([
                    html.Div([
                        html.H5("📈 Data Status"),
                        html.P([
                            html.Strong("Source: "),
                            "Real CSV data" if app_df is not None else "Demo data"
                        ], className="mb-2"),
                        html.P([
                            html.Strong("Countries: "),
                            f"{len(working_df['Country'].unique())} countries available"
                        ], className="mb-2"),
                        html.P([
                            html.Strong("Time period: "),
                            f"{working_df['Date'].min().date()} to {working_df['Date'].max().date()}"
                        ], className="mb-2"),
                        html.P([
                            html.Strong("Total records: "),
                            f"{len(working_df):,}"
                        ]),
                    ], className="card p-3 mb-3"),
                    
                    html.Div([
                        html.H5("🔍 Analysis Types"),
                        html.Ul([
                            html.Li([
                                html.Strong("📈 Forecasting: "),
                                "Simple linear forecast based on recent CPI trends"
                            ], className="mb-2"),
                            html.Li([
                                html.Strong("🔗 Correlation: "),
                                "Analyze relationship between CPI and Google search interest"
                            ], className="mb-2"),
                            html.Li([
                                html.Strong("🌍 Comparison: "),
                                "Compare CPI trends across multiple countries (select 2+ countries)"
                            ])
                        ])
                    ], className="card p-3"),
                    
                    html.Div([
                        html.H5("💡 Tips"),
                        html.Ul([
                            html.Li("For comparison analysis, select 2 or more countries"),
                            html.Li("Hover over graphs to see detailed values"),
                            html.Li("Use the slider to adjust forecast period"),
                            html.Li("Export data using the Export buttons")
                        ], className="small")
                    ], className="card p-3 mt-3")
                ])
            ])
        else:  # tab-visual
            return html.Div([
                html.H5("📊 Visual Analysis Dashboard"),
                html.P([
                    "Select analysis type and parameters, then click '",
                    html.Strong("Update Analysis"),
                    "' to generate visualizations."
                ], className="mb-3"),
                html.Div([
                    html.P([
                        html.I(className="bi bi-info-circle me-2"),
                        f"Currently loaded: {len(working_df['Country'].unique())} countries, {len(working_df):,} data points"
                    ], className="text-info mb-2"),
                    html.P([
                        html.I(className="bi bi-lightbulb me-2"),
                        "Tip: Select multiple countries for comparison analysis"
                    ], className="text-warning small")
                ], className="alert alert-light")
            ])
    
    # Download callbacks (placeholder - you can implement these)
    @app.callback(
        Output("download-data", "data"),
        Input("export-data", "n_clicks"),
        prevent_initial_call=True
    )
    def export_data(n_clicks):
        if n_clicks:
            return dcc.send_data_frame(working_df.to_csv, "global_cpi_data.csv")
    
    @app.callback(
        Output("download-report", "data"),
        Input("export-report", "n_clicks"),
        prevent_initial_call=True
    )
    def export_report(n_clicks):
        if n_clicks:
            # Create a simple text report
            report = f"""
            Global Cost of Living Analysis Report
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Data Summary:
            - Countries: {len(working_df['Country'].unique())}
            - Total Records: {len(working_df):,}
            - Date Range: {working_df['Date'].min().date()} to {working_df['Date'].max().date()}
            - Data Source: {'Real CSV' if app_df is not None else 'Demo'}
            
            Available Countries:
            {', '.join(sorted(working_df['Country'].unique()))}
            
            Note: This is a basic report. Implement more detailed reporting as needed.
            """
            return dict(content=report, filename="cpi_analysis_report.txt")

# Print load status
print("\n" + "="*60)
print("📊 Global Cost of Living Dashboard - Callbacks Loaded")
print("="*60)
print(f"Data source: {'Real CSV data' if app_df is not None else 'Demo data'}")
print(f"Countries available: {sorted(working_df['Country'].unique())}")
print(f"Total records: {len(working_df)}")
print(f"Date range: {working_df['Date'].min().date()} to {working_df['Date'].max().date()}")
print("="*60)
print("✅ Dashboard ready!")
print("   Select 'Comparison' analysis type and choose multiple countries")
print("="*60)
