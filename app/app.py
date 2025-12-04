"""
app.py: Main Dash application for Global Cost of Living Dashboard
Deployable to Heroku, Render, PythonAnywhere
"""
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from flask import Flask
import os
import sys

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize app with Bootstrap
server = Flask(__name__)
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
    suppress_callback_exceptions=True
)

# Set app title
app.title = "Global Cost of Living Dashboard"

# Import and create layout
from layouts import create_layout
from callbacks import register_callbacks

# Create layout
app.layout = create_layout()

# Register callbacks
register_callbacks(app)

# For deployment
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    print(f"\n🚀 Server starting on port {port}...")
    print(f"🌐 Open your browser to: http://localhost:{port}")
    print("🛑 Press CTRL+C to stop")
    print("="*60)
    
    # Use 'run' instead of 'run_server' for newer Dash versions
    app.run(
        host='0.0.0.0',
        port=port,
        debug=True
    )