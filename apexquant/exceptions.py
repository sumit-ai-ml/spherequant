"""ApexQuant-specific exception types."""

from __future__ import annotations


class ApexQuantPreflightWarning(Exception):
    """Raised when the ApexQuant fan-in audit predicts the method will
    underperform QuaRot on the supplied model.

    The decision belongs to the caller: pass ``preflight=False`` to
    :func:`apexquant.quantize_model` to override and quantize anyway.

    The triggering :class:`apexquant.audit.ModelReport` is attached as
    ``self.report`` for downstream inspection.
    """

    def __init__(self, message: str, report=None):
        super().__init__(message)
        self.report = report
