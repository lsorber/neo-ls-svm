"""Neo LS-SVM types."""

from typing import TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt

C = TypeVar("C", np.complex64, np.complex128)
F = TypeVar("F", np.float32, np.float64)
K = TypeVar("K", np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.intp)
N = TypeVar("N", np.float32, np.float64, np.int32, np.int64, np.intp)

# TODO: https://github.com/numpy/numpy/issues/16544
ComplexVector: TypeAlias = npt.NDArray[C]
ComplexMatrix: TypeAlias = npt.NDArray[C]
ComplexTensor: TypeAlias = npt.NDArray[C]

FloatVector: TypeAlias = npt.NDArray[F]
FloatMatrix: TypeAlias = npt.NDArray[F]
FloatTensor: TypeAlias = npt.NDArray[F]

GenericVector: TypeAlias = npt.NDArray[np.generic]
GenericMatrix: TypeAlias = npt.NDArray[np.generic]
GenericTensor: TypeAlias = npt.NDArray[np.generic]

IntegerVector: TypeAlias = npt.NDArray[K]
IntegerMatrix: TypeAlias = npt.NDArray[K]
IntegerTensor: TypeAlias = npt.NDArray[K]

NumberVector: TypeAlias = npt.NDArray[N]
NumberMatrix: TypeAlias = npt.NDArray[N]
NumberTensor: TypeAlias = npt.NDArray[N]
