import numpy as np


class VehicleModel:

    def __init__(self):

        self.m  = 2979.51   # the vehicle mass (kg)
        self.Iz = 4549.29   # the mass moment of inertia (Nm s^2)
        self.lf = 2.49      # the longitudinal distance from the CoG to the front wheels (m)
        self.lr = 2.75      # the longitudinal distance from the CoG to the rear wheels (m)
        self.Cf = 33595.26  # the cornering stiffness of the front tires (N/rad)
        self.Cr = 57855.95  # the cornering stiffness of the rear tires (N/rad)

        self.nx = 4
        self.nu = 1
        self.nw = 1

        self.lbu = -0.3
        self.ubu =  0.3


    def get_lti(self, vx, psiref):

        mvx = self.m * vx
        Izvx = self.Iz * vx
        lfCf = self.lf * self.Cf
        lrCr = self.lr * self.Cr

        A = np.array(
            [
                [0.0,                              1.0,                                0.0,                                             0.0],
                [0.0, -2.0 * (self.Cf + self.Cr) / mvx, 2.0 * (self.Cf + self.Cr) / self.m,                      -2.0 * (lfCf - lrCr) / mvx],
                [0.0,                              0.0,                                0.0,                                             1.0],
                [0.0,      -2.0 * (lfCf - lrCr) / Izvx,      2.0 * (lfCf - lrCr) / self.Iz, -2.0 * (self.lf * lfCf - self.lr * lrCr) / Izvx]
            ]
        )
    
        B = np.array(
            [
                [                   0.0],
                [2.0 * self.Cf / self.m],
                [                   0.0],
                [  2.0 * lfCf / self.Iz]
            ]
        )
        d = np.array(
            [
                [                                                     0.0],
                [              (-2.0 * (lfCf - lrCr) / mvx - vx) * psiref],
                [                                                     0.0],
                [-2.0 * (self.lf * lfCf - self.lr * lrCr) / Izvx * psiref]
            ]
        )

        return A, B, d
