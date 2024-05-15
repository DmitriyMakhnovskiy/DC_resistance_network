#
# Two methods for solving a system of linear equations A * a = b, where a and b are vectors:
# (1) Matrix A is square (N, N)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve.html
#
# (2) Matrix A is not square (N, M), where the number of rows (N conditions) exceeds the number of columns (M unknowns)
# https://scipy.github.io/devdocs/reference/generated/scipy.linalg.lstsq.html#scipy.linalg.lstsq
#
# This algorithm computes the vector such that the norm ||b - A * a|| is minimized.
#
# Dr. Dmitriy Makhnovskiy, City College Plymouth, England, 07.04.2024
#

from scipy.linalg import lstsq

# Parameters used in the A matrix
R1 = 4.0
R2 = 7.0
R3 = 9.0
R4 = 5.0
R5 = 20.0

# A matrix
A = [
     [1.0, 1.0, -1.0, -1.0, 0.0],
     [1.0, 0.0, -1.0, 0.0, -1.0],
     [0.0, 1.0, 0.0, -1.0, 1.0],
     [R1, 0.0, R3, 0.0, 0.0],
     [0.0, R2, 0.0, R4, 0.0],
     [-R1, R2, 0.0, 0.0, -R5],
     [0.0, 0.0, -R3, R4, R5],
     [R1, -R2, R3, -R4, 0.0],
     ]

# b vector (right part of the equation)
b = [
     [0.0],
     [0.0],
     [0.0],
     [32],
     [32],
     [0.0],
     [0.0],
     [0.0],
      ]

# Minimization of the norm ||b - A * a|| --> 0
a, res, rnk, s = lstsq(A, b)
a = a.flatten()  # 2D a-array was converted to a 1D array
a = [round(x, 3) for x in a]  # rounding to three significant figures after the dot

print('I1 = ', a[0], 'A')
print('I2 = ', a[1], 'A')
print('I3 = ', a[2], 'A')
print('I4 = ', a[3], 'A')
print('I5 = ', a[4], 'A')
print('Total current = ', a[0] + a[1], 'A')
print('')
print('V1 = ', a[0] * R1, 'V')
print('V2 = ', a[1] * R2, 'V')
print('V3 = ', a[2] * R3, 'V')
print('V4 = ', a[3] * R4, 'V')
print('V5 = ', a[4] * R5, 'V')
