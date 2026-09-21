"""Mega-Quantification error types."""


class MegaQuantError(Exception):
    """Base error for Mega-Quantification."""


class RecipeError(MegaQuantError):
    """Invalid recipe YAML, mapping, or overrides."""


class BackendError(MegaQuantError):
    """Quantization backend missing, unavailable, or failed."""


class FamilyError(MegaQuantError):
    """Model-family adapter missing or failed to detect."""


class CalibrationError(MegaQuantError):
    """Calibration data loading or tokenization failed."""


class EvalError(MegaQuantError):
    """Benchmark eval failed."""


class DryRun(MegaQuantError):
    """Signals a dry-run path (plan only; no weights loaded)."""
