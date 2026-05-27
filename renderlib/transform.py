from typing import Any
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from typing import SupportsFloat


class Vec3:
    """
    Class for simplified work with numpy arrays of size 3.
    """

    def __init__(
        self,
        x: SupportsFloat | Any | None = None,
        y: SupportsFloat | None = None,
        z: SupportsFloat | None = None,
    ):
        """
        Creates vector:
        When x is float or None vector will be (x, y, z) None will default to 0.
        When x is Vec3 it will be copied.
        When x is anything else 'np.array' will try to do the conversion.
        """

        if x is None and y is None and z is None:
            self.vec = np.zeros(3, dtype=float)

        elif y is None and z is None:
            try:
                if isinstance(x, SupportsFloat):
                    self.vec = np.array([float(x), 0.0, 0.0], dtype=float)

            except TypeError:
                if isinstance(x, Vec3):
                    self.vec = x.vec.copy()

                else:
                    arr = np.array(x, dtype=float)
                    if arr.shape != (3,):
                        raise ValueError("Vec3 must be a 1D array of length 3")
                    self.vec = arr

        elif z is None:
            self.vec = np.array([x, y, 0.0], dtype=float)

        elif y is None:
            self.vec = np.array([x, 0.0, z], dtype=float)

        else:
            self.vec = np.array([x, y, z], dtype=float)

    def __str__(self) -> str:
        return f"({self.x}, {self.y}, {self.z})"

    def __repr__(self) -> str:
        return f"Vec3{self.__str__()}"

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.vec + other.vec)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.vec - other.vec)

    def __mul__(self, other: float) -> "Vec3":
        return Vec3(self.vec * other)

    def __truediv__(self, other: float) -> "Vec3":
        return Vec3(self.vec / other)

    def norm(self) -> float:
        """
        Returns norm of the vector.
        """
        return float(np.linalg.norm(self.vec))

    def normalize(self) -> "Vec3":
        """
        Normalizes vector.
        """

        norm = self.norm()
        EPSILON = 1e-8
        if norm == EPSILON:
            return Vec3.zero()
        return Vec3(self.vec / norm)

    @staticmethod
    def one() -> "Vec3":
        """
        Creates vector(1, 1, 1).
        """
        return Vec3(1, 1, 1)

    @staticmethod
    def zero() -> "Vec3":
        """
        Creates vector(0, 0, 0).
        """
        return Vec3(0, 0, 0)

    @property
    def x(self) -> float:
        return self.vec[0]

    @x.setter
    def x(self, val: float) -> None:
        self.vec[0] = val

    @property
    def y(self) -> float:
        return self.vec[1]

    @y.setter
    def y(self, val: float) -> None:
        self.vec[1] = val

    @property
    def z(self) -> float:
        return self.vec[2]

    @z.setter
    def z(self, val: float) -> None:
        self.vec[2] = val


class Transform:
    """
    Class for storing and working with transform matrix.
    """

    def __init__(
        self,
        position: Vec3 | None = None,
        rotation: Rotation | None = None,
        scale: Vec3 | None = None,
    ):
        """
        Creates transform where position, rotation and scale are kept as separate values.
        When position is unspecified it defaults to (0, 0, 0).
        When rotation is unspecified it defaults to identity.
        When scale is unspecified it defaults to (1, 1, 1).
        """

        self.position = Vec3.zero() if position is None else position
        self.rotation = Rotation.identity() if rotation is None else rotation
        self.scale = Vec3.one() if scale is None else scale

    def get_matrix(self) -> NDArray:
        """
        Returns 4x4 transform matrix that performs scaling then rotation and lastly translation.
        """

        T = np.eye(4)
        T[:-1, 3] = self.position.vec

        S = np.diag([*self.scale.vec, 1])

        R = np.eye(4)
        R[:3, :3] = self.rotation.as_matrix()

        return T @ R @ S

    def get_inverse_matrix(self) -> NDArray:
        """
        Returns inverse of 4x4 transform matrix that performs scaling then rotation and lastly translation.
        """

        t = self.get_matrix()
        t = np.linalg.inv(t)
        return t
