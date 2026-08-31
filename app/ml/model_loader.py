import json
import joblib
import pandas as pd
from pathlib import Path

from app.utils.config import PIPELINE_FILE, COLUMN_METADATA_FILE
from app.utils.logger import get_logger

logger = get_logger("model_loader")

_pipeline = None
_metadata = None
_model_version = "1.0"


def load_pipeline():
    global _pipeline
    if _pipeline is None:
        logger.info("Loading model from %s", PIPELINE_FILE)
        _pipeline = joblib.load(PIPELINE_FILE)
        logger.info("Model v%s loaded successfully", _model_version)
    return _pipeline


def load_metadata():
    global _metadata
    if _metadata is None:
        logger.info("Loading column metadata from %s", COLUMN_METADATA_FILE)
        with open(COLUMN_METADATA_FILE) as f:
            _metadata = json.load(f)
        logger.info(
            "Metadata loaded: %d numeric, %d categorical features",
            len(_metadata["numeric_cols"]),
            len(_metadata["categorical_cols"]),
        )
    return _metadata


def get_feature_columns():
    meta = load_metadata()
    return meta["numeric_cols"] + meta["categorical_cols"]


def get_model_version():
    return _model_version


def reload():
    global _pipeline, _metadata
    _pipeline = None
    _metadata = None
    logger.info("Reloading model and metadata...")
    load_pipeline()
    load_metadata()
    logger.info("Reload complete")
