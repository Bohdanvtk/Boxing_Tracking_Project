# src/boxing_project/kalman_filter/kalman.py

import numpy as np
from filterpy.kalman import KalmanFilter
from typing import Union, Tuple


def _q_block(dt: float, var: float) -> np.ndarray:
    """1D constant-acceleration Q-блок для [pos, vel]."""
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    return var * np.array(
        [
            [dt4 / 4.0, dt3 / 2.0],
            [dt3 / 2.0, dt2],
        ],
        dtype=float,
    )


def _ensure_state(x0: Union[np.ndarray, list, tuple]) -> np.ndarray:
    """
    Приводимо початковий стан до (4,1): [x, y, vx, vy]^T.

    Дозволено:
      - [x, y]
      - [x, y, vx, vy]
    """
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    if x0.size == 2:
        x0 = np.array([x0[0], x0[1], 0.0, 0.0], dtype=float)
    elif x0.size != 4:
        raise ValueError("x0 must have length 2 ([x,y]) or 4 ([x,y,vx,vy])")
    return x0.reshape(4, 1)


def _ensure_measurement(z: Union[np.ndarray, list, tuple]) -> np.ndarray:
    """Приводимо вимір до (2,1): [x, y]^T."""
    z = np.asarray(z, dtype=float).reshape(-1)
    if z.size != 2:
        raise ValueError("Measurement z must have length 2: [x, y]")
    return z.reshape(2, 1)


class KalmanTracker:
    """
    Проста 2D Kalman-модель з постійною швидкістю.

    Стан:        [x, y, vx, vy]^T
    Вимір:       [x, y]^T
    """

    def __init__(
        self,
        x0: Union[np.ndarray, list, tuple],
        dt: float,
        process_var: float,
        measure_var: float,
        p0: float,
    ):
        # нормалізуємо початковий стан
        x0 = _ensure_state(x0)
        self.dt = float(dt)

        # внутрішній KalmanFilter з filterpy
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # Матриця переходу стану (constant velocity)
        self.kf.F = np.array(
            [
                [1.0, 0.0, self.dt, 0.0],
                [0.0, 1.0, 0.0, self.dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        # Вимірюємо тільки позицію (x, y)
        self.kf.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        # Коваріація вимірювального шуму
        self.kf.R = float(measure_var) * np.eye(2, dtype=float)

        # Процесний шум (constant acceleration у x,y)
        Q_cv = np.block(
            [
                [_q_block(self.dt, process_var), np.zeros((2, 2))],
                [np.zeros((2, 2)), _q_block(self.dt, process_var)],
            ]
        )

        # Перестановка під порядок стану [x, y, vx, vy]
        Pperm = np.array(
            [
                [1, 0, 0, 0],  # x
                [0, 0, 1, 0],  # y
                [0, 1, 0, 0],  # vx
                [0, 0, 0, 1],  # vy
            ],
            dtype=float,
        )
        self.kf.Q = Pperm @ Q_cv @ Pperm.T

        # Початковий стан і коваріація
        self.kf.x = x0
        self.kf.P = np.eye(4, dtype=float) * float(p0)

    # -------- основний API --------

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Прогноз на один крок.

        Повертає:
          state: (4,) [x, y, vx, vy]
          cov:   (4,4) P
        """
        self.kf.predict()
        return self.get_state(), self.get_cov()

    def update(self, z: Union[np.ndarray, list, tuple]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Оновлення за новим виміром z = [x, y].

        Повертає:
          state: (4,) оновлений стан
          cov:   (4,4) P
        """
        z_norm = _ensure_measurement(z)
        self.kf.update(z_norm)
        return self.get_state(), self.get_cov()

    # -------- допоміжні методи --------

    def project(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Проєкція поточного стану в простір вимірювань.

        Повертає:
          z_hat: (2,1) прогнозований вимір
          S:     (2,2) коваріація інновації
        """
        H = self.kf.H
        x = self.kf.x
        P = self.kf.P
        z_hat = H @ x
        S = H @ P @ H.T + self.kf.R
        return z_hat, S

    def gating_distance(self, z: Union[np.ndarray, list, tuple]) -> float:
        """
        Махаланобісова відстань^2 між виміром z та прогнозом.
        Використовується для χ²-gating.
        """
        z = _ensure_measurement(z)
        z_hat, S = self.project()
        r = z - z_hat  # інновація

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S + 1e-6 * np.eye(2))

        d2 = float(r.T @ S_inv @ r)
        return d2

    def get_state(self) -> np.ndarray:
        """Поточний стан як (4,) [x, y, vx, vy]."""
        return self.kf.x.reshape(-1).copy()

    def get_cov(self) -> np.ndarray:
        """Поточна коваріація (4x4)."""
        return self.kf.P.copy()

    @property
    def F(self) -> np.ndarray:
        return self.kf.F

    @property
    def Q(self) -> np.ndarray:
        return self.kf.Q

    @property
    def R(self) -> np.ndarray:
        return self.kf.R

    @property
    def H(self) -> np.ndarray:
        return self.kf.H
