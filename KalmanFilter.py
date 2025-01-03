import numpy as np

class KalmanFilter():
    def __init__(self, d_t, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        self.dt = d_t
        self.u_x = u_x
        self.u_y = u_y
        self.std_acc = std_acc
        self.x_std_meas = x_std_meas
        self.y_std_meas = y_std_meas

        self.id = np.identity(4)

        self.mat_P = np.zeros((4,4))

        self.u = np.array([self.u_x, self.u_y])
        self.xk = np.expand_dims(np.array([0, 0, 0, 0]), 1)
        self.mat_a = np.array([
            [1, 0, d_t, 0],
            [0, 1, 0, d_t],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        )

        self.mat_b = np.array([
            [0.5 * d_t**2, 0],
            [0, 0.5 * d_t**2],
            [d_t, 0],
            [0, d_t]
        ])

        self.mat_h = np.array([
            [1,0,0,0],
            [0,1,0,0]
        ])

        self.mat_q = np.array([
            [0.25 * d_t**4, 0, 0.5 * d_t**3, 0],
            [0, 0.25 * d_t**4, 0, 0.5 * d_t**3],
            [0.5 * d_t**3, 0, d_t**2, 0],
            [0, 0.5 * d_t**3, 0, d_t**2]
        ]) * self.std_acc ** 2

        self.mat_r = np.array([
            [self.x_std_meas, 0],
            [0, self.y_std_meas]
        ])

    def predict(self):
        #print('xk',self.xk.shape)
        #print('A', self.mat_a.shape)
        #print('B', self.mat_b.shape)
        u = np.expand_dims(self.u, 0).T
        #print('u', u.shape)
        #x_k = self.mat_a @ self.xk + self.mat_b @ self.u
        tmp = self.mat_a @ self.xk
        #print('tmp',tmp.shape)
        tmp2 = self.mat_b @ u
        #print('tmp2', tmp2.shape)

        x_k = tmp + tmp2
        #print(x_k.shape)

        P_k = self.mat_a @ self.mat_P @ self.mat_a.T + self.mat_q

        return x_k, P_k
    
    def update(self, z_k, x_k, P_k):
        Sk = self.mat_h @ P_k @ self.mat_h.T + self.mat_r
        Kk = P_k @ self.mat_h.T @ np.linalg.inv(Sk)

        self.xk = x_k + Kk @ (z_k - (self.mat_h @ x_k))
        self.mat_P = (self.id - (Kk @ self.mat_h)) @ P_k