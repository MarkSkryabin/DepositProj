from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv(Path(__file__).resolve().with_name(".env"))

name = os.environ.get("NAME")
inference_model_name = os.environ.get("INFERENCE_MODEL", "iteration_14")
