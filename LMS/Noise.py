from Adaptive import LMS
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import normal
from scipy.fft import fft
from scipy.signal import tf2zpk as ZTransform
from scipy.signal import group_delay as GroupDelay
from scipy.signal import freqz as FrequencyResponse

def RightShift(x, D):
    shifted_x = np.zeros(len(x) + D)
    shifted_x[D:] = x
    return shifted_x

def CancelNoise(x, N, D = 1):
    shifted_x = RightShift(x, D)
    return LMS(shifted_x, x, N, delta = 0.000001)

M = 1000
t = np.linspace(0, 4 * np.pi, num = M)
s = np.sin(t)
x = s + normal(loc = 0, scale = 0.2, size = M)
h, filtered_signal = CancelNoise(x, N = 11)
plt.subplot(131)
plt.title("Original With Noisy")
plt.plot(x, color = "darkblue")
plt.plot(s, color = "red")
plt.subplot(132)
plt.title("Noisy With Filtered")
plt.plot(x, color = "darkblue")
plt.plot(filtered_signal, color = "deepskyblue")
plt.subplot(133)
plt.title("Original With Filtered")
plt.plot(filtered_signal, color = "deepskyblue")
plt.plot(s, color = "red")
plt.show()
denominator = np.zeros_like(h)
denominator[0] = 1
_, group_delay = GroupDelay((h, denominator))
w, phase_response = FrequencyResponse(h, denominator)
plt.subplot(121)
plt.xlabel("Frequency (rad)")
plt.title("Group Delay")
plt.plot(w, group_delay)
plt.subplot(122)
plt.xlabel("Frequency (rad)")
plt.title("Phase Response")
plt.plot(w, np.angle(phase_response))
plt.show()
u = np.linspace(0, 1, num = 200)
frequency_spectrum = np.abs(fft(h, 200))
plt.title("Magnitude Specturm")
plt.plot(u, frequency_spectrum)
plt.show()
zeros, _, _ = ZTransform(h, denominator)
print("The zeros of the Z transform are:", zeros)