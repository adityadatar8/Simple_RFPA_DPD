import scipy
import numpy as np
import matplotlib.pyplot as plt

threshold = 1.1
highValue = 3
degrees = 70

x = np.linspace(0, 3, 1000)

def saturationCurve(val):
    return 2 - np.exp(-1*(3*val - 0.693147))
def inverseSaturationCurve(val):
    return (1.0/3.0)*(-1*(np.log(2-val)) + 0.693147)
def filter(val):
    return (val*np.heaviside(-1*(x-threshold), 1) + highValue*np.heaviside(x-threshold, 1))


f = saturationCurve(x) # original RFPA curve
h = 5*x*np.heaviside(-1*(x-0.4), 1) + 2*np.heaviside(x-0.401, 1) # desired curve
g = inverseSaturationCurve(x) # inverse curve

# rotate inverse curve
theta = np.deg2rad(degrees)
rotation = [[np.cos(theta), -1*np.sin(theta)], 
              [np.sin(theta), np.cos(theta)]]
[a, b] = np.dot(rotation, [x, g])

# cascade the rotated inverse curve with the orignal RFPA curve
linearizedOut = saturationCurve(b)
linearizedIn = filter(a)

plt.plot(x, f, label='PA Output')
plt.plot(x, h, label='Desired Output')
# plt.plot(x, g, label='Inverse PA Output')
# plt.plot(a, b, label='Rotated Inverse Output 67.5 deg')
plt.plot(linearizedIn, linearizedOut, label='Linearized Output')
plt.legend()
plt.xlabel('Input Power')
plt.ylabel('Output Power')
plt.title('Comparison of PA Output, Desired Output, and Test Curves')
plt.grid(True)
plt.show()


x_analyze = []
f_analyze = []
h_analyze = []
a_analyze = []
b_analyze = []
i=0
while (x[i] < 0.8):
    x_analyze.append(x[i])
    f_analyze.append(f[i])
    h_analyze.append(h[i])
    a_analyze.append(linearizedIn[i])
    b_analyze.append(linearizedOut[i])
    i+=1
x_analyze = np.array(x_analyze)
f_analyze = np.array(f_analyze)
h_analyze = np.array(h_analyze)
a_analyze = np.array(a_analyze)
b_analyze = np.array(b_analyze)

originalCoefficients = np.polyfit(x_analyze, f_analyze, 1)
desiredCoefficients = np.polyfit(x_analyze, h_analyze, 1)
testCoefficients = np.polyfit(a_analyze, b_analyze, 1)

y_original = (originalCoefficients[0]*x_analyze) + originalCoefficients[1]
y_desired = (desiredCoefficients[0]*x_analyze) + desiredCoefficients[1]
y_test = (testCoefficients[0]*a_analyze) + testCoefficients[1]
residuals_original = f_analyze - y_original
residuals_desired = h_analyze - y_desired
residuals_test = b_analyze - y_test

mse_original = np.mean(residuals_original**2)
mse_desired = np.mean(residuals_desired**2)
mse_test = np.mean(residuals_test**2)

print("MSE Original: " + str(mse_original))
print("MSE Desired: " + str(mse_desired))
print("MSE Test: " + str(mse_test))

# Filter out end parts
# y_test = y_test * np.heaviside(-1*(x_analyze-2), 1)
y_test_filtered = []
a_analyze_filtered = []
for i in range(len(y_test)):
    if (y_test[i] != 0):
        y_test_filtered.append(y_test[i])
        a_analyze_filtered.append(a_analyze[i])
x_test_filtered = np.array(a_analyze_filtered)
y_test_filtered = np.array(y_test_filtered)

plt.plot(x, f, label='PA Output')
# plt.plot(x, h, label='Desired Output')
# plt.plot(x, g, label='Inverse PA Output')
# plt.plot(a, b, label='Rotated Inverse Output 67.5 deg')
plt.plot(linearizedIn, linearizedOut, label='Linearized Output')

plt.plot(x_analyze, y_original, label='LinReg Original')
# plt.plot(x_analyze, y_desired, label='LinReg Desired')
plt.plot(a_analyze, y_test, label='LinReg Test')

plt.legend()
plt.xlabel('Input Power')
plt.ylabel('Output Power')
plt.title('Comparison of PA Output, Desired Output, and Test Curves')
plt.grid(True)
plt.show()