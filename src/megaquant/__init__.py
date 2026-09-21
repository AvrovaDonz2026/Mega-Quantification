"""Mega-Quantification: generic Hugging Face PTQ pipeline."""

from megaquant.calibration import DummyCalibIter, build_calibration_iter
from megaquant.config import (
    CLI_OVERRIDE_MAP,
    CalibrationSpec,
    ExportSpec,
    LayerGroup,
    ModelSpec,
    PrecisionSpec,
    Recipe,
    load_recipe,
    recipe_from_mapping,
)
from megaquant.exceptions import (
    BackendError,
    CalibrationError,
    DryRun,
    FamilyError,
    MegaQuantError,
    RecipeError,
)
from megaquant.pipeline import QuantPipeline, ResolvedPlan, format_plan
from megaquant.registry import (
    detect_family,
    get_backend,
    get_family,
    get_scheme,
    list_backends,
    list_families,
    list_schemes,
    register_backend,
    register_family,
    register_scheme,
)

__version__ = "0.1.0"

__all__ = [
    "CLI_OVERRIDE_MAP",
    "BackendError",
    "CalibrationError",
    "CalibrationSpec",
    "DryRun",
    "DummyCalibIter",
    "ExportSpec",
    "FamilyError",
    "LayerGroup",
    "MegaQuantError",
    "ModelSpec",
    "PrecisionSpec",
    "QuantPipeline",
    "Recipe",
    "RecipeError",
    "ResolvedPlan",
    "__version__",
    "build_calibration_iter",
    "detect_family",
    "format_plan",
    "get_backend",
    "get_family",
    "get_scheme",
    "list_backends",
    "list_families",
    "list_schemes",
    "load_recipe",
    "recipe_from_mapping",
    "register_backend",
    "register_family",
    "register_scheme",
]
