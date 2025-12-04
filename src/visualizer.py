"""
visualizer.py: Visualization functions for dashboard - SIMPLIFIED CORRECTED VERSION
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

class DashboardVisualizer:
    """Creates interactive visualizations for the dashboard."""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.loader = analyzer.loader
    
    def create_forecast_plot(self, country: str, forecast_result: dict) -> go.Figure:
        """Create interactive forecast plot for a country."""
        if forecast_result is None:
            return self._create_empty_plot("No forecast available")
        
        forecast_df = forecast_result['forecast'].copy()
        training_df = forecast_result['training_data'].copy()
        
        # Ensure datetime format
        forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
        training_df['ds'] = pd.to_datetime(training_df['ds'])
        
        # Create figure
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=training_df['ds'],
            y=training_df['y'],
            mode='markers',
            name='Historical CPI',
            marker=dict(color='blue', size=6),
            hovertemplate='Date: %{x|%Y-%m}<br>CPI: %{y:.2f}%<extra></extra>'
        ))
        
        # Add forecast line
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2),
            hovertemplate='Date: %{x|%Y-%m}<br>Forecast: %{y:.2f}%<extra></extra>'
        ))
        
        # Add uncertainty band (simplified approach)
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_lower'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.1)',
            name='95% Confidence',
            hoverinfo='skip'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Inflation Forecast for {country}",
            xaxis_title="Date",
            yaxis_title="CPI (%)",
            hovermode='x unified',
            showlegend=True,
            height=500
        )
        
        # Add metrics annotation
        if 'metrics' in forecast_result:
            metrics = forecast_result['metrics']
            metrics_text = f"Model Performance:<br>MAE: {metrics.get('MAE', 'N/A')}<br>RMSE: {metrics.get('RMSE', 'N/A')}<br>MAPE: {metrics.get('MAPE', 'N/A')}%"
            
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                text=metrics_text,
                showarrow=False,
                align="left",
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
        
        # Add forecast start indicator (alternative approach)
        last_historical = training_df['ds'].max()
        if not pd.isna(last_historical):
            # Add a shape instead of vline
            fig.add_vrect(
                x0=last_historical,
                x1=last_historical,
                fillcolor="gray",
                opacity=0.5,
                layer="below",
                line_width=1,
                line_dash="dash"
            )
            
            # Add annotation
            fig.add_annotation(
                x=last_historical,
                y=fig.data[0].y.max() if len(fig.data[0].y) > 0 else 0,
                text="Forecast Start",
                showarrow=True,
                arrowhead=1
            )
        
        return fig
    
    def create_correlation_heatmap(self, country: str, corr_result: dict) -> go.Figure:
        """Create correlation heatmap for a country."""
        if corr_result is None:
            return self._create_empty_plot("No correlation data available")
        
        corr_matrix = corr_result['pearson']
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont=dict(size=12)
        ))
        
        fig.update_layout(
            title=f"Correlation Analysis for {country}",
            xaxis_title="Variable",
            yaxis_title="Variable",
            height=500,
            width=600
        )
        
        return fig
    
    def create_country_comparison_chart(self, metric: str = 'CPI') -> go.Figure:
        """Create comparison chart across countries."""
        if self.loader.merged_df is None:
            return self._create_empty_plot("No data available")
        
        # Get latest value for each country
        latest_data = []
        countries = self.loader.get_available_countries()
        
        for country in countries:
            try:
                country_data = self.loader.get_country_data(country)
                if metric in country_data.columns and not country_data[metric].isna().all():
                    # Get most recent non-null value
                    valid_data = country_data[country_data[metric].notna()]
                    if not valid_data.empty:
                        latest = valid_data.iloc[-1]
                        latest_data.append({
                            'country': country,
                            'value': latest[metric],
                            'date': latest['ds']
                        })
            except:
                continue
        
        if not latest_data:
            return self._create_empty_plot(f"No {metric} data available")
        
        df = pd.DataFrame(latest_data).sort_values('value', ascending=False)
        
        # Create bar chart
        fig = px.bar(
            df,
            x='country',
            y='value',
            title=f"Latest {metric} by Country",
            labels={'value': metric, 'country': 'Country'},
            color='value',
            color_continuous_scale='Viridis'
        )
        
        fig.update_traces(
            texttemplate='%{y:.1f}',
            textposition='outside'
        )
        
        fig.update_layout(
            height=500,
            xaxis_tickangle=-45,
            coloraxis_showscale=False
        )
        
        return fig
    
    def create_search_vs_cpi_scatter(self, country: str) -> go.Figure:
        """Create scatter plot of search index vs CPI."""
        try:
            country_data = self.loader.get_country_data(country)
            
            # Clean data
            plot_data = country_data[['ds', 'CPI', 'search_inflation']].dropna()
            
            if len(plot_data) < 10:
                return self._create_empty_plot(f"Insufficient data for {country}")
            
            # Calculate correlation
            correlation = plot_data['CPI'].corr(plot_data['search_inflation'])
            
            # Create scatter plot
            fig = px.scatter(
                plot_data,
                x='search_inflation',
                y='CPI',
                title=f"{country}: Search Interest vs CPI (r = {correlation:.3f})",
                labels={'search_inflation': 'Inflation Search Index', 'CPI': 'CPI (%)'},
                trendline='ols',
                hover_data=['ds']
            )
            
            fig.update_layout(height=500)
            fig.data[1].line.color = 'red'
            
            return fig
            
        except:
            return self._create_empty_plot(f"Could not create plot for {country}")
    
    def _create_empty_plot(self, message: str) -> go.Figure:
        """Create an empty plot with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=400
        )
        return fig