from flask import Flask
from .routes import init_routes
from config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Initialize routes
    init_routes(app)
    
    return app