from flask import Flask
from app.models import init_db
from app.routes import bp
from config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    init_db(app)
    app.register_blueprint(bp)

    return app
