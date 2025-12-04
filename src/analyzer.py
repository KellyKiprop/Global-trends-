"""
analyzer.py: Phase 2 - Analytical core (Prophet + Correlation)
"""
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class CostOfLivingAnalyzer:
    """Performs forecasting and correlation analysis."""
    
    def __init__(self, data_loader):
        """
        Initialize analyzer with a DataLoader instance.
        
        Parameters:
        -----------
        data_loader : DataLoader
            Instance of DataLoader class
        """
        self.loader = data_loader
        self.merged_df = data_loader.merged_df
        self.forecast_models = {}
        self.correlation_results = {}
    
    def forecast_country(self, country: str, periods: int = 12) -> dict:
        """
        Run Prophet forecast for a specific country.
        
        Parameters:
        -----------
        country : str
            Country name
        periods : int
            Number of months to forecast
            
        Returns:
        --------
        dict: Forecast results and metrics
        """
        # Get country data
        country_data = self.loader.get_country_data(country)
        
        if len(country_data) < 24:
            print(f"⚠️  Insufficient data for {country}. Need 24+ months, have {len(country_data)}")
            return None
        
        # Prepare Prophet data
        prophet_df = country_data[['ds', 'CPI']].rename(columns={'CPI': 'y'})
        prophet_df = prophet_df.dropna()
        
        if len(prophet_df) < 24:
            print(f"⚠️  Insufficient non-NaN CPI data for {country}")
            return None
        
        # Configure and fit Prophet
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            interval_width=0.95
        )
        
        try:
            model.fit(prophet_df)
            
            # Create future dates and forecast
            future = model.make_future_dataframe(periods=periods, freq='M', include_history=True)
            forecast = model.predict(future)
            
            # Calculate performance metrics
            historical = forecast[forecast['ds'].isin(prophet_df['ds'])].copy()
            historical = historical.merge(prophet_df, on='ds')
            
            metrics = self._calculate_metrics(historical['y'], historical['yhat'])
            
            # Store results
            self.forecast_models[country] = {
                'model': model,
                'forecast': forecast,
                'metrics': metrics,
                'training_data': prophet_df
            }
            
            return self.forecast_models[country]
            
        except Exception as e:
            print(f"❌ Forecast failed for {country}: {e}")
            return None
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate forecast accuracy metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MAE': round(mae, 3),
            'RMSE': round(rmse, 3),
            'MAPE': round(mape, 2),
            'n_samples': len(y_true)
        }
    
    def analyze_correlations(self, country: str) -> dict:
        """
        Analyze correlations between CPI and search trends.
        
        Parameters:
        -----------
        country : str
            Country name
            
        Returns:
        --------
        dict: Correlation results
        """
        country_data = self.loader.get_country_data(country)
        
        if len(country_data) < 10:
            print(f"⚠️  Insufficient data for correlation in {country}")
            return None
        
        # Select relevant columns
        corr_cols = ['CPI', 'search_inflation', 'search_gasprice', 'search_rent']
        analysis_df = country_data[corr_cols].dropna()
        
        if len(analysis_df) < 10:
            print(f"⚠️  Insufficient non-NaN data for correlation in {country}")
            return None
        
        # Calculate correlations
        pearson_corr = analysis_df.corr(method='pearson')
        spearman_corr = analysis_df.corr(method='spearman')
        
        # Calculate p-values
        p_values = self._calculate_p_values(analysis_df)
        
        # Store results
        self.correlation_results[country] = {
            'pearson': pearson_corr,
            'spearman': spearman_corr,
            'p_values': p_values,
            'data': analysis_df,
            'n_observations': len(analysis_df)
        }
        
        return self.correlation_results[country]
    
    def _calculate_p_values(self, df):
        """Calculate p-values for Pearson correlations."""
        p_values = pd.DataFrame(index=df.columns, columns=df.columns)
        
        for i in df.columns:
            for j in df.columns:
                if i == j:
                    p_values.loc[i, j] = 0.0
                else:
                    try:
                        corr, p_val = stats.pearsonr(df[i].values, df[j].values)
                        p_values.loc[i, j] = p_val
                    except:
                        p_values.loc[i, j] = np.nan
        
        return p_values
    
    def run_comprehensive_analysis(self, countries=None):
        """
        Run full analysis for multiple countries.
        
        Parameters:
        -----------
        countries : list, optional
            List of countries to analyze
        """
        if countries is None:
            countries = self.loader.get_available_countries()
        
        results = {}
        
        for country in countries:
            print(f"\n{'='*60}")
            print(f"ANALYZING: {country}")
            print('='*60)
            
            # Forecasting
            print(f"\n1. Prophet Forecasting...")
            forecast_result = self.forecast_country(country)
            
            if forecast_result:
                print(f"   Model Performance:")
                for metric, value in forecast_result['metrics'].items():
                    print(f"   - {metric}: {value}")
            
            # Correlation
            print(f"\n2. Correlation Analysis...")
            corr_result = self.analyze_correlations(country)
            
            if corr_result:
                print(f"   Key correlations with CPI:")
                cpi_corrs = corr_result['pearson']['CPI'].sort_values(ascending=False)
                
                for var, corr in cpi_corrs.items():
                    if var != 'CPI':
                        p_val = corr_result['p_values'].loc['CPI', var]
                        sig = "✓" if p_val < 0.05 else "✗"
                        print(f"   - {var}: {corr:.3f} (p={p_val:.4f}) {sig}")
            
            results[country] = {
                'forecast': forecast_result,
                'correlation': corr_result
            }
        
        return results