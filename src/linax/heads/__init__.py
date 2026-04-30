"""This module contains the heads implemented in Linax."""

from linax.heads.base import Head, HeadConfig
from linax.heads.classification import (
    ClassificationHead,
    ClassificationHeadConfig,
)
from linax.heads.fc import FCHead, FCHeadConfig
from linax.heads.freq_conv_glu import (
    FreqConvGLUHead,
    FreqConvGLUHeadConfig,
)
from linax.heads.regression import (
    RegressionHead,
    RegressionHeadConfig,
)
from linax.heads.spectral_decoder import (
    MagDecoderHead,
    MagDecoderHeadConfig,
    PhaseDecoderHead,
    PhaseDecoderHeadConfig,
)

__all__ = [
    "HeadConfig",
    "Head",
    "ClassificationHead",
    "ClassificationHeadConfig",
    "FCHead",
    "FCHeadConfig",
    "FreqConvGLUHead",
    "FreqConvGLUHeadConfig",
    "MagDecoderHead",
    "MagDecoderHeadConfig",
    "PhaseDecoderHead",
    "PhaseDecoderHeadConfig",
    "RegressionHead",
    "RegressionHeadConfig",
]
