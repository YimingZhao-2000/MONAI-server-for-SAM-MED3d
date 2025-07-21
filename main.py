from config import SAMMed3DApp
import os

def app():
    studies = os.environ.get("MONAI_LABEL_DATASTORE", "./data")
    app_dir = os.path.dirname(__file__)
    return SAMMed3DApp(app_dir=app_dir, studies=studies) 