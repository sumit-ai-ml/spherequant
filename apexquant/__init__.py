"""ApexQuant: training-free post-training weight quantization via random
orthogonal rotation + Beta-matched scalar codebook.
"""

from apexquant.audit import (
    BAD,
    EMPTY,
    GOOD,
    MARGINAL,
    LayerReport,
    ModelReport,
    audit,
)
from apexquant.exceptions import ApexQuantPreflightWarning
from apexquant.ptq import (
    LayerStats,
    beta_codebook,
    quantize_model,
    quantize_model_baseline,
    quantize_model_h3,
    quantize_model_quarot,
    quantize_model_rtn_absmax,
    quantize_model_apexquant,
    uniform_codebook,
)
from apexquant.rotation_utils import (
    SRHTRotation,
    apply_rotation,
    beta_ks_test,
    build_torch_rotation,
    make_rotation,
    materialize_rotation_matrix,
    random_orthogonal,
    torch_apply_rotation,
    torch_inverse_rotation,
)

__all__ = [
    "BAD",
    "EMPTY",
    "GOOD",
    "LayerReport",
    "LayerStats",
    "MARGINAL",
    "ModelReport",
    "SRHTRotation",
    "ApexQuantPreflightWarning",
    "apply_rotation",
    "audit",
    "beta_codebook",
    "beta_ks_test",
    "build_torch_rotation",
    "make_rotation",
    "materialize_rotation_matrix",
    "quantize_model",
    "quantize_model_baseline",
    "quantize_model_h3",
    "quantize_model_quarot",
    "quantize_model_rtn_absmax",
    "quantize_model_apexquant",
    "random_orthogonal",
    "torch_apply_rotation",
    "torch_inverse_rotation",
    "uniform_codebook",
]
