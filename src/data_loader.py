"""
data_loader.py: Phase 1 - Data loading and preprocessing
Handles loading World Bank and Google Trends CSVs,
merging them into a monthly dataset, and providing
country-level data for visualization.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List


class DataLoader:
    """Handles loading, validating, and merging WB & Google Trends data."""

    def __init__(self, wb_path: str, trends_path: str):
        """
        Initialize DataLoader with CSV paths.
        """
        self.wb_path = wb_path
        self.trends_path = trends_path
        self.wb_df: pd.DataFrame | None = None
        self.trends_df: pd.DataFrame | None = None
        self.merged_df: pd.DataFrame | None = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and validate both datasets."""
        self.wb_df = pd.read_csv(self.wb_path)
        self.trends_df = pd.read_csv(self.trends_path)

        # Convert date columns to datetime
        self.wb_df['ds'] = pd.to_datetime(self.wb_df['ds'])
        self.trends_df['ds'] = pd.to_datetime(self.trends_df['ds'])

        self._validate_data()

        print(f"✅ Data loaded successfully:")
        print(f"   World Bank: {len(self.wb_df)} records, {self.wb_df['country'].nunique()} countries")
        print(f"   Google Trends: {len(self.trends_df)} records, {self.trends_df['country'].nunique()} countries")

        return self.wb_df, self.trends_df

    def _validate_data(self):
        """Validate essential columns exist."""
        required_wb_cols = ['country', 'ds', 'CPI']
        required_trends_cols = ['country', 'ds', 'search_inflation']

        for col in required_wb_cols:
            if col not in self.wb_df.columns:
                raise ValueError(f"Missing required column in WB data: {col}")

        for col in required_trends_cols:
            if col not in self.trends_df.columns:
                raise ValueError(f"Missing required column in Trends data: {col}")

    def merge_data(self) -> pd.DataFrame:
        """Merge WB annual data with monthly Google Trends data."""
        if self.wb_df is None or self.trends_df is None:
            self.load_data()

        monthly_data = []

        wb_countries = sorted(self.wb_df['country'].dropna().unique())
        trends_countries = set(self.trends_df['country'].dropna().unique())

        print(f"\n🔗 Merging data for {len(wb_countries)} countries (WB dataset base)...")

        for country in wb_countries:
            wb_country = self.wb_df[self.wb_df['country'] == country].copy()
            trends_country = self.trends_df[self.trends_df['country'] == country].copy()

            # If trends data missing, create monthly index from WB dates
            if trends_country.empty:
                min_ds = wb_country['ds'].min()
                max_ds = wb_country['ds'].max()
                if pd.isna(min_ds) or pd.isna(max_ds):
                    continue
                monthly_idx = pd.date_range(
                    start=min_ds.to_period('M').to_timestamp('ME'),
                    end=max_ds.to_period('M').to_timestamp('ME'),
                    freq='M'
                )
                monthly_df = pd.DataFrame({'ds': monthly_idx})
                monthly_df['search_inflation'] = np.nan
                # Optional trend columns
                for col in ['search_gasprice', 'search_rent']:
                    if col in self.trends_df.columns:
                        monthly_df[col] = np.nan
            else:
                # Use actual trends data
                trend_cols = ['search_inflation']
                for col in ['search_gasprice', 'search_rent']:
                    if col in trends_country.columns:
                        trend_cols.append(col)
                monthly_df = trends_country[['ds'] + trend_cols].copy()

            # Merge WB annual values
            monthly_df['year'] = monthly_df['ds'].dt.year
            wb_country['year'] = wb_country['ds'].dt.year
            wb_for_merge = wb_country[['year', 'CPI', 'FOOD_INFL', 'GDP_PC']].copy()

            monthly_df = monthly_df.merge(wb_for_merge, on='year', how='left')

            # Forward-fill WB columns
            for col in ['CPI', 'FOOD_INFL', 'GDP_PC']:
                if col in monthly_df.columns:
                    monthly_df[col] = monthly_df[col].ffill()

            monthly_df['country'] = country
            monthly_data.append(monthly_df)

        if not monthly_data:
            raise ValueError("No countries could be processed for merging!")

        self.merged_df = pd.concat(monthly_data, ignore_index=True)

        # Interpolate missing CPI values
        self.merged_df['CPI'] = self.merged_df.groupby('country')['CPI'].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )

        # Reorder columns
        cols = [
            'country', 'ds', 'year', 'CPI', 'FOOD_INFL', 'GDP_PC',
            'search_inflation'
        ]
        for col in ['search_gasprice', 'search_rent']:
            if col in self.merged_df.columns:
                cols.append(col)
        self.merged_df = self.merged_df[cols]

        print(f"✅ Merged data created: {len(self.merged_df)} monthly records")
        print(f"   Date range: {self.merged_df['ds'].min().date()} to {self.merged_df['ds'].max().date()}")

        return self.merged_df

    def get_country_data(self, country: str) -> pd.DataFrame:
        """Return merged data for a specific country."""
        if self.merged_df is None:
            self.merge_data()

        country_data = self.merged_df[self.merged_df['country'] == country].copy()
        if country_data.empty:
            available = self.merged_df['country'].unique()
            raise ValueError(f"Country '{country}' not found. Available: {available}")
        return country_data

    def get_available_countries(self) -> List[str]:
        """Return sorted list of countries in merged data."""
        if self.merged_df is None:
            self.merge_data()
        return sorted(self.merged_df['country'].unique())
