#!/usr/bin/env python3
"""Kelvin wake simulation model, a component of the MSc project:
Analysis of Thames wave data & investigation of energy potential.

Copyright (c) 2019, 2022 Gordon Mills

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import signal

# Instance an command line parser and get arguments.
parser = argparse.ArgumentParser()
parser.add_argument('velocity', type=float, help='vessel velocity (m/s)')
parser.add_argument('distance', type=float, help='distance to observation point (m)')
parser.add_argument('-f', '--filename', type=str, default='kelvinwake.png',
                    help='output filename (default kelvinwake.png)')
args = parser.parse_args()

u = args.velocity
z = args.distance
g = 9.81

# Identify contributing points of stationary phase, theta_1 - transverse, theta_2 - divergent.
# Derived from analysis in MIT 1.138J Wave Propagation - Chapter 4. Gravity Waves In Water.
theta_1 = lambda x, z: np.arctan((-x + np.emath.sqrt(x ** 2 - 8 * z ** 2)) / (4 * z))
theta_2 = lambda x, z: np.arctan((-x - np.emath.sqrt(x ** 2 - 8 * z ** 2)) / (4 * z))

# Calculate phase and amplitude of componenet waves, and total surface elevation.
# Based on analysis presented in Newman (1977) - Marine Hydrodynamics.
phi = lambda theta, x, z: (g / u ** 2) * (x / np.cos(theta) + z / np.cos(theta) ** 2 * np.sin(theta))
wav_1 = lambda x, z: np.exp(1j * (phi(theta_1(x, z), x, z) + np.pi / 4))
wav_2 = lambda x, z: np.exp(1j * (phi(theta_2(x, z), x, z) - np.pi / 4))
A = lambda x, z: (x ** 2 + z ** 2) ** (-1 / 4)
eta = lambda x, z: A(x, z) * np.real(wav_1(x, z) + wav_2(x, z))

# Restrict to wake region R, derived from definition of Kelvin wake angle.
in_R = lambda x, z: abs(z / x) < 2 ** (-3 / 2)

# Create figure.
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6.4, 6.4))

# Plot surface elevation field.
x_max, z_max = 8 * u,  3 * u
[X, Z] = np.meshgrid(np.linspace(0, x_max, 200), np.linspace(-z_max, z_max, 200))
Y = in_R(X, Z) * eta(X, Z)
axes[0].pcolormesh(X, Z, Y, cmap=cm.jet)
axes[0].set_xlabel('x (m)')
axes[0].set_ylabel('z (m)')
axes[0].set_title(f'Spatial surface elevation at velocity u = {u} m/s')

# Plot time-frequency spectrogram.
N, M, T = 2 ** 10, 2 ** 8, 30
t_start = 2 ** (3 / 2) * z / u
fs = N / T
x = u * np.linspace(t_start, t_start + T, N)
f, t, S = signal.spectrogram(eta(x, z), fs = fs, nperseg=M, noverlap=31 * M / 32)
axes[1].pcolormesh(t, f, S, cmap=cm.jet)
axes[1].set_xlabel('t (s)')
axes[1].set_ylabel('f (Hz)')
axes[1].set_ylim([0, fs / 8])
axes[1].set_title(f'Time-frequency at observation point on z = {z} m')

# Save figure.
fig.tight_layout()
fig.savefig(args.filename)
