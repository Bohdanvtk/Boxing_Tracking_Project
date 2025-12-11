import numpy as np
from filterpy.kalman import KalmanFilter


def _q_block(dt: float, var: float) -> np.ndarray:

    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt

    return var * np.array([[dt4 / 4.0, dt3 / 2.0],
                           [dt3 / 2.0, dt2]], dtype=float)


class KalmanTracker:

    def __init__(self,
                 x0: np.ndarray | list,
                 dt: float,
                 process_var: float,
                 measure_var: float,
                 p0: float,
                 ):

        x0 = np.asarray(x0, dtype=float).reshape(-1)
        if x0.size == 2:
            x0 = np.array([x0[0], x0[1], 0, 0], dtype=float)

        elif x0.size != 4:
            raise ValueError("x0 must has length 2 [x0, y0] or 4 [x0, y0, vx, vy]")

        self.dt = float(dt)

        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        F = np.array([[1, 0, self.dt, 0],
                      [0, 1, 0, self.dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=float)

        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=float)

        R = measure_var * np.eye(2, dtype=float)

        Q_cv = np.block([
            [_q_block(self.dt, process_var), np.zeros((2, 2))],
            [np.zeros((2, 2)), _q_block(self.dt, process_var)]
        ])  # для порядку [x, vx, y, vy]

        Pperm = np.array([
            [1, 0, 0, 0],  # x = x
            [0, 0, 1, 0],  # y = y
            [0, 1, 0, 0],  # vx = vx
            [0, 0, 0, 1],  # vy = vy
        ], dtype=float)
        Q = Pperm @ Q_cv @ Pperm.T

        self.kf.F = F
        self.kf.H = H
        self.kf.R = R
        self.kf.Q = Q
        self.kf.x = x0.reshape(4, 1)
        self.kf.P = np.eye(4, dtype=float) * float(p0)


    def predict(self) -> tuple[np.ndarray, np.ndarray]:

        self.kf.predict()

        return self.get_state(), self.get_cov()

    def update(self, z: np.ndarray | list) -> tuple[np.ndarray, np.ndarray]:

        z = np.asarray(z, dtype=float).reshape(2, 1)
        self.kf.update(z)
        return self.get_state(), self.get_cov()


    def project(self) -> tuple[np.ndarray, np.ndarray]:

        H = self.kf.H
        x = self.kf.x
        P = self.kf.P
        z_hat = H @ x
        S = H @ P @ H.T + self.kf.R

        return z_hat, S


    def gating_distance(self, z: np.ndarray| list) -> float:

        z = np.asarray(z, dtype=float).reshape(2, 1)
        z_hat, S = self.project()
        r = z - z_hat
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S + 1e-6 * np.eye(2))

        d2 = float(r.T @ S_inv @ r)

        return d2

    def get_state(self) -> np.ndarray:
        """curent state as a (4, ) vector."""
        return self.kf.x.reshape(-1).copy()

    def get_cov(self) -> np.ndarray:
        """current covariation P (4x4)."""
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





