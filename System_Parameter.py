import numpy as np
import cv2

dx = 6.5/150
nx = 512
ny = nx
lx = nx * dx
ly = ny * dx
lfx = 1 / dx
lfy = 1 / dx
n = np.linspace(0, nx - 1, nx)
m = np.linspace(0, ny - 1, ny)
x = dx * np.linspace(-nx / 2, nx / 2 - 1, nx)
y = dx * np.linspace(-ny / 2, ny / 2 - 1, ny)  # spatial coordinate
x, y = np.meshgrid(x, y)
rho, theta = cv2.cartToPolar(x, y)
fx = -lfx / 2 + lfx / nx * (n - 1)  # spatial frequency coordinate
fy = -lfy / 2 + lfy / ny * (m - 1)
fx, fy = np.meshgrid(fx, fy)
f_rho, f_theta = cv2.cartToPolar(fx, fy)
