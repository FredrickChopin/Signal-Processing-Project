import wave
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from scipy.signal import correlate

#Our code does the folliwing:
#1) Read the WAV file named Our Voice.wav
#2) With d = 0.5 seconds and alpha = 0.5, apply the echo filter
#3) Plot the original recording together with the echoed recording
#4) Plot the autocorrelation array of the echoed recoding
#5) Reconstruct the original recording from the echoed recording
#6) Save the echoed recording and the reconstructed recording

def SaveSignalLikeAnotherObj(signal, original_obj, filename):
    with wave.open(filename, "wb") as obj_new:
        obj_new.setnchannels(original_obj.getnchannels())
        obj_new.setsampwidth(original_obj.getsampwidth())
        obj_new.setframerate(obj.getframerate())
        obj_new.writeframes(signal.astype(np.int16).tobytes())

def ExtractSignalFromObj(obj):
    frames = obj.readframes(-1) #Read all frames
    return np.frombuffer(frames, dtype = np.int16).astype(np.double)

def PlotSignal(signal, framerate):
    #A general function to plot a signal
    #The framerate is needed for us to get a true time scale in seconds 
    time = np.linspace(0, len(signal) / framerate, num = len(signal))
    plt.plot(time, signal)

def ApplyEchoFilter(x, d, alpha):
    #Using np.convolve ran a lot slower
    #Better just use the difference equation
    y = np.empty(len(x) + d)
    for n in range(0, d):
        y[n] = x[n]
    for n in range(d, len(x)):
         y[n] = x[n] + alpha * x[n-d]
    for n in range(len(x) + 1, len(y)):
        y[n] = alpha * x[n-d]
    return y

def RestoreD(y):
    #method = "fft" allows us to use the efficient computation algorithm
    #that we described in the PDF
    auto_correlation = correlate(y, y, mode = "full", method = "fft")
    #The definition in the documentation is a bit different from ours
    #We truncate the auto_correlation array
    auto_correlation = auto_correlation[len(y) - 1:] 
    plt.title("Auto Correlation")
    plt.plot(auto_correlation)
    plt.show()
    #The algorithm is as described in the PDF
    prev = auto_correlation[0]
    k = 1
    for k in range(1, len(auto_correlation)):
        curr = auto_correlation[k]
        if curr > prev:
            break
        prev = curr
    return k + np.argmax(auto_correlation[k:])

def GetN(y, d):
    return len(y) - d

def GetM(N, d):
    return int(np.ceil(N / d))

def TargetFunction(y, d, m, alpha):
    #The root of this function is the alpha we need to restore
    sum = 0
    coeff = 1
    for j in range(0, m):
        coeff *= -alpha
        sum += y[(m - j) * d] * coeff
    return sum

def RestoreAlpha(y, d, N):
    m = GetM(N, d)
    initial_guess = 0.6
    #0.6 is an arbitrary choice of a number in (0,1)
    #It is the initial point of search for fsolve
    return fsolve(lambda alpha : TargetFunction(y, d, m, alpha), initial_guess)[0] 
    

def RestoreX(y):
    d = RestoreD(y)
    print("Restored d =", d)
    N = GetN(y,d)
    alpha = RestoreAlpha(y, d, N)
    print("Restored alpha =", alpha)
    #After restoring d and alpha,
    #we can restore x by using the difference equation
    x = np.empty(N)
    for n in range(0, d):
        x[n] = y[n]
    for n in range(d + 1, N):
        x[n] = y[n] - alpha * x[n - d]
    return x

#Import the recording file
with wave.open("Our Voice.wav", "rb") as obj:
    framerate = obj.getframerate()
    x = ExtractSignalFromObj(obj)
    print("Length of x = ", len(x))
    d = 3 * obj.getframerate() #Echo with d = 3 seconds
    alpha = 0.5
    print("d = ", d)
    print("alpha =", alpha)
    y = ApplyEchoFilter(x, d, alpha)
    print("Length of y =", len(y))
    plt.subplot(121)
    plt.title("Original Recording")
    PlotSignal(x, framerate)
    plt.subplot(122)
    plt.title("Recording With Echo")
    PlotSignal(y, framerate)
    plt.show()
    x = RestoreX(y)
    SaveSignalLikeAnotherObj(y, obj, "Echoed Recording.wav")
    SaveSignalLikeAnotherObj(x, obj, "Restored Recording.wav")