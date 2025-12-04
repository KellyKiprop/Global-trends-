"""
layouts.py: UI layout components
"""
from dash import dcc, html
import dash_bootstrap_components as dbc

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
    """Create control panel."""
    return dbc.Card([
        dbc.CardHeader("Dashboard Controls", className="bg-primary text-white"),
        dbc.CardBody([
            html.H6("Country Selection", className="mt-2"),
            dcc.Dropdown(
                id='country-dropdown',
                options=[{'label': c, 'value': c} for c in available_countries],
                value='United States',
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
            
            html.H6("Comparison Countries", className="mt-3", id='comparison-label'),
            dcc.Dropdown(
                id='comparison-countries',
                options=[{'label': c, 'value': c} for c in available_countries],
                value=['United States', 'Germany', 'Japan'],
                multi=True,
                className="mb-3"
            ),
            
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
        ])
    ], className="h-100")

def create_metrics_cards():
    """Create summary metrics cards."""
    return dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H5("10", className="card-title text-primary"),
                    html.P("Countries Analyzed", className="card-text")
                ])
            ]),
            width=3
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H5("10 Years", className="card-title text-primary"),
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
                    html.H5("Prophet", className="card-title text-primary"),
                    html.P("Forecast Model", className="card-text")
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
            dbc.Tab(label="Visual Analysis", tab_id="tab-visual"),
            dbc.Tab(label="Data Table", tab_id="tab-data"),
            dbc.Tab(label="Insights", tab_id="tab-insights"),
        ], id="main-tabs", active_tab="tab-visual", className="mb-3"),
        
        # Tab content
        html.Div(id="tab-content"),
        
        # Graphs container
        html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(id='main-graph'), width=12, className="mb-4"),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='secondary-graph'), width=6),
                dbc.Col(dcc.Graph(id='correlation-heatmap'), width=6),
            ]),
        ], id="graphs-container")
    ], width=9)

def create_footer():
    """Create dashboard footer."""
    return dbc.Container([
        html.Hr(),
        dbc.Row([
            dbc.Col([
                html.P("Final Project: Global Cost of Living Analysis", 
                      className="text-muted"),
                html.P([
                    "Analytical Methods: ",
                    html.Strong("Prophet Forecasting"),
                    " & ",
                    html.Strong("Search-Inflation Correlation")
                ], className="text-muted small"),
            ], width=8),
            dbc.Col([
                html.P("University Final Project", className="text-muted text-end"),
                html.P([
                    html.I(className="bi bi-exclamation-triangle me-1"),
                    "Demonstration Data - For Academic Purposes"
                ], className="text-warning small text-end"),
            ], width=4),
        ])
    ], className="mt-5")

def create_layout():
    """Create complete dashboard layout."""
    # Simulate available countries (will be replaced with real data)
    available_countries = [
        'United States', 'United Kingdom', 'China', 'India', 'Japan',
        'United Arab Emirates', 'Kenya', 'Germany', 'Brazil', 'South Africa'
    ]
    
    return dbc.Container([
        create_header(),
        
        html.Br(),
        
        create_metrics_cards(),
        
        dbc.Row([
            dbc.Col(create_controls(available_countries), width=3),
            create_main_content(),
        ]),
        
        create_footer(),
        
        # Hidden storage
        dcc.Store(id='analysis-results'),
        dcc.Download(id="download-report"),
        dcc.Download(id="download-data"),
    ], fluid=True, className="px-4")