"""SphereQuant-specific exception types."""

from __future__ import annotations


class SphereQuantPreflightWarning(Exception):
    """Raised when the SphereQuant fan-in audit predicts the method will
    underperform QuaRot on the supplied model.

    The decision belongs to the caller: pass ``preflight=False`` to
    :func:`spherequant.quantize_model` to override and quantize anyway.

    The triggering :class:`spherequant.audit.ModelReport` is attached as
    ``self.report`` for downstream inspection.
    """

    def __init__(self, message: str, report=None):
        super().__init__(message)
        self.report = report
