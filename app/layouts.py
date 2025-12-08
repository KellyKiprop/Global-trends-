"""
layouts.py: UI layout components
Updated to use actual data from callbacks
"""
from dash import dcc, html
import dash_bootstrap_components as dbc

# Import the working dataframe from callbacks
try:
    from callbacks import working_df
    
    # Get available countries from actual data
    available_countries = sorted(working_df['Country'].dropna().unique().tolist())
    
    # If no countries found, use defaults
    if not available_countries:
        print("⚠️ No countries found in data, using defaults")
        available_countries = [
            'United States', 'United Kingdom', 'China', 'Japan', 'Germany',
            'United Arab Emirates', 'Kenya', 'Brazil', 'South Africa'
        ]
    else:
        print(f"✅ Layout: Found {len(available_countries)} countries in data")
        
except ImportError:
    print("⚠️ Could not import working_df, using default countries")
    available_countries = [
        'United States', 'United Kingdom', 'China', 'Japan', 'Germany',
        'United Arab Emirates', 'Kenya', 'Brazil', 'South Africa'
    ]

def create_header():
    """Create dashboard header."""
    return dbc.Navbar(
        dbc.Container([
            html.A(
                dbc.Row([
                    dbc.Col(html.I(className="bi bi-globe-americas me-2")),
                    dbc.Col(dbc.NavbarBrand("Global Cost of Living Dashboard", className="ms-2")),
                ], align="center", className="g-0"),
                href="/",
                style={"textDecoration": "none"},
            ),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("Forecasting", href="#")),
                    dbc.NavItem(dbc.NavLink("Correlation", href="#")),
                    dbc.NavItem(dbc.NavLink("Comparison", href="#")),
                    dbc.NavItem(dbc.NavLink("Methodology", href="#")),
                ], className="ms-auto", navbar=True),
                id="navbar-collapse",
                navbar=True,
            ),
        ]),
        color="primary",
        dark=True,
        sticky="top",
    )

def create_controls(available_countries):
    """Create control panel with actual countries from data."""
    return dbc.Card([
        dbc.CardHeader("Dashboard Controls", className="bg-primary text-white"),
        dbc.CardBody([
            html.H6("Country Selection", className="mt-2"),
            dcc.Dropdown(
                id='country-dropdown',
                options=[{'label': c, 'value': c} for c in available_countries],
                value=available_countries[0] if available_countries else 'United States',
                clearable=False,
                className="mb-3"
            ),
            
            html.H6("Analysis Type", className="mt-3"),
            dcc.RadioItems(
                id='analysis-type',
                options=[
                    {'label': '📈 Forecasting', 'value': 'forecast'},
                    {'label': '🔗 Correlation', 'value': 'correlation'},
                    {'label': '🌍 Comparison', 'value': 'comparison'}
                ],
                value='forecast',
                className="mb-3"
            ),
            
            html.Div([
                html.H6("Forecast Period", className="mt-3", id='forecast-label'),
                dcc.Slider(
                    id='forecast-slider',
                    min=6,
                    max=36,
                    step=6,
                    value=12,
                    marks={i: f'{i} mo' for i in range(6, 37, 6)},
                    className="mb-4"
                ),
            ], id='forecast-controls', style={'display': 'block'}),
            
            html.Div([
                html.H6("Comparison Countries", className="mt-3", id='comparison-label'),
                dcc.Dropdown(
                    id='comparison-countries',
                    options=[{'label': c, 'value': c} for c in available_countries],
                    value=available_countries[:3] if len(available_countries) >= 3 else available_countries,
                    multi=True,
                    className="mb-3"
                ),
            ], id='comparison-controls', style={'display': 'none'}),
            
            dbc.Button(
                "Update Analysis",
                id='update-button',
                color="primary",
                className="w-100 mt-3",
                n_clicks=0
            ),
            
            html.Hr(),
            
            html.H6("Export Options", className="mt-3"),
            dbc.ButtonGroup([
                dbc.Button("📥 Download Report", id='export-report', color="success", size="sm"),
                dbc.Button("📊 Export Data", id='export-data', color="info", size="sm"),
            ], className="w-100"),
            
            html.Hr(),
            
            html.Div([
                html.H6("Data Status", className="mt-3"),
                html.P([
                    html.I(className="bi bi-check-circle-fill text-success me-2"),
                    f"Loaded {len(available_countries)} countries"
                ], className="small text-success mb-1"),
                html.P([
                    html.I(className="bi bi-info-circle-fill text-info me-2"),
                    "Click Update to refresh visualizations"
                ], className="small text-info mb-0"),
            ])
        ])
    ], className="h-100")

def create_metrics_cards():
    """Create summary metrics cards."""
    try:
        from callbacks import working_df
        num_countries = len(working_df['Country'].unique())
        date_range = working_df['Date']
        start_year = date_range.min().year
        end_year = date_range.max().year
        years_covered = end_year - start_year + 1
        
        # Count total records
        total_records = len(working_df)
        
    except:
        num_countries = 10
        years_covered = 10
        total_records = 360
    
    return dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H5(f"{num_countries}", className="card-title text-primary"),
                    html.P("Countries Analyzed", className="card-text")
                ])
            ]),
            width=3
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H5(f"{years_covered} Years", className="card-title text-primary"),
                    html.P("Data Coverage", className="card-text")
                ])
            ]),
            width=3
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H5("Monthly", className="card-title text-primary"),
                    html.P("Update Frequency", className="card-text")
                ])
            ]),
            width=3
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H5(f"{total_records:,}", className="card-title text-primary"),
                    html.P("Data Points", className="card-text")
                ])
            ]),
            width=3
        ),
    ], className="mb-4")

def create_main_content():
    """Create main content area."""
    return dbc.Col([
        # Tabs for different views
        dbc.Tabs([
            dbc.Tab(label="📈 Visual Analysis", tab_id="tab-visual"),
            dbc.Tab(label="📊 Data Table", tab_id="tab-data"),
            dbc.Tab(label="💡 Insights", tab_id="tab-insights"),
        ], id="main-tabs", active_tab="tab-visual", className="mb-3"),
        
        # Tab content
        html.Div(id="tab-content", className="mb-4"),
        
        # Graphs container
        html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Primary Analysis", className="bg-light"),
                        dbc.CardBody([
                            dcc.Graph(
                                id='main-graph',
                                config={'displayModeBar': True, 'displaylogo': False}
                            )
                        ])
                    ])
                ], width=12, className="mb-4"),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Secondary Analysis", className="bg-light"),
                        dbc.CardBody([
                            dcc.Graph(
                                id='secondary-graph',
                                config={'displayModeBar': True, 'displaylogo': False}
                            )
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Correlation Analysis", className="bg-light"),
                        dbc.CardBody([
                            dcc.Graph(
                                id='correlation-heatmap',
                                config={'displayModeBar': True, 'displaylogo': False}
                            )
                        ])
                    ])
                ], width=6),
            ]),
        ], id="graphs-container")
    ], width=9)

def create_footer():
    """Create dashboard footer."""
    try:
        from callbacks import app_df, load_error
        data_source = "Real CSV Data" if app_df is not None else "Demo Data"
        if load_error:
            data_source += f" ({load_error[:50]}...)"
    except:
        data_source = "Demo Data"
    
    return dbc.Container([
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.P("Final Project: Global Cost of Living Analysis", 
                      className="text-muted"),
                html.P([
                    "Analytical Methods: ",
                    html.Strong("Time Series Forecasting"),
                    " & ",
                    html.Strong("Search-CPI Correlation")
                ], className="text-muted small"),
                html.P([
                    "Data Source: ",
                    html.Strong(data_source)
                ], className="text-muted small"),
            ], width=8),
            dbc.Col([
                html.P("University Final Project", className="text-muted text-end"),
                html.P([
                    html.I(className="bi bi-exclamation-triangle me-1"),
                    "For Academic & Demonstration Purposes"
                ], className="text-warning small text-end"),
                html.P([
                    html.I(className="bi bi-github me-1"),
                    "Global Cost of Living Dashboard"
                ], className="text-muted small text-end"),
            ], width=4),
        ])
    ], className="mt-5")

def create_layout():
    """Create complete dashboard layout."""
    print(f"📋 Creating layout with {len(available_countries)} countries: {available_countries}")
    
    return dbc.Container([
        create_header(),
        
        html.Br(),
        
        create_metrics_cards(),
        
        dbc.Row([
            dbc.Col(create_controls(available_countries), width=3),
            create_main_content(),
        ], className="g-3"),
        
        create_footer(),
        
        # Hidden storage and downloads
        dcc.Store(id='analysis-results'),
        dcc.Download(id="download-report"),
        dcc.Download(id="download-data"),
        
        # JavaScript for showing/hiding controls based on analysis type
        dcc.Store(id='analysis-type-store'),
    ], fluid=True, className="px-4", style={'minHeight': '100vh'})

# Additional callback to show/hide controls based on analysis type
# This should be registered in app.py or callbacks.py
def register_control_callbacks(app):
    @app.callback(
        [Output('forecast-controls', 'style'),
         Output('comparison-controls', 'style')],
        Input('analysis-type', 'value')
    )
    def toggle_controls(analysis_type):
        if analysis_type == 'comparison':
            return {'display': 'none'}, {'display': 'block'}
        else:
            return {'display': 'block'}, {'display': 'none'}
