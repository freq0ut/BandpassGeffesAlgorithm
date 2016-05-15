# -*- coding: utf-8 -*-
"""
Created on Wed May 11 04:52:32 2016

@author: Zack
"""

from matplotlib.widgets import Cursor
import numpy as np
from scipy import signal
import matplotlib.pyplot as plot

def iteration(start, end, step):
    while start <= end:
        yield start
        start += step
        
# set testMode to one to debug script        
testMode = 0

if(testMode == 0):
    print(" ")
    print("Let's design a butterworth bandpass filter using Geffe's algorithm and Friend Circuits...")
    print(" ")
    a_min = float(input("Enter the minimum attenuation (in dB) in the stop band (a_min): "))
    a_max = float(input("Enter the maximum attenuation (in dB) in the pass band (a_max): "))
    print(" ")
    omega_1 = float(input("Enter f1 (the beginning of the pass band) in Hz: "))
    omega_2 = float(input("Enter f2 (the end of the pass band) in Hz: "))
    print(" ")
    errorCheck = 0
    choice = 0
    while (errorCheck == 0):   
        print("Do you wish to specify f3 (the end of the HPF stop band)? Enter (3).")
        print("Or f4 (the beginning of the LPF's stop band)? Enter (4).")
        print(" ")
        choice = int(input("Choose option (3) or (4): "))
        print(" ")
        if(choice == 3 or choice == 4):
            errorCheck = 1
        else:
            print("Please enter a valid input.")
            print(" ")
    
    if(choice == 3): # take in ω3, solve for ω4
        omega_3 = float(input("Enter f3 (end of HPF's stop band) in Hz: "))
        omega_4 = float(omega_1*omega_2/omega_3)
    else: # take in ω4, solve for ω3
        omega_4 = float(input("Enter f4 (beginning of LPF's stop band) in Hz: "))
        omega_3 = float(omega_1*omega_2/omega_4)
else:
    a_min = 30
    a_max = 0.5
    omega_1 = 2
    omega_2 = 4
    omega_3 = 1.8
    omega_4 = 4.5

# convert natural frequency to radian frequency
omega_1 = omega_1*np.pi*2
omega_2 = omega_2*np.pi*2
omega_3 = omega_3*np.pi*2
omega_4 = omega_4*np.pi*2

# solve for Ωs and define Ωp as unity
OMEGA_stop = (omega_4 - omega_3)/(omega_2 - omega_1)
OMEGA_pass = 1

# solve for minimum n required for given specs
n_numerator = np.log10( (np.power(10,a_min/10)-1) / (np.power(10,a_max/10)-1) )
n_denominator = 2*np.log10(OMEGA_stop/OMEGA_pass)
n_full = (n_numerator / n_denominator)
n = np.ceil(n_numerator / n_denominator)

# solve for Ωo, bandwidth, ω0, and q_c
OMEGA_0 = OMEGA_pass/(np.power((np.power(10,a_max/10)-1),1/(2*n)))
band_width = omega_2-omega_1
omega_0 = np.sqrt(omega_2*omega_1)
q_c = omega_0/band_width

# determine pole spacing in degrees for even or odd n
if(n%2 == 0): # even case
    first_poles = 90/(n)
    pole_spacing = 180/(n)
    nEven = True
    numIter = n
    numIterPoles = n
else: # odd case
    first_poles = 0
    pole_spacing = 180/(n)
    nEven = False
    numIter = n
    numIterPoles = (n+1)/2

# print(" ")
# print("OMEGA_s = " + str(round(OMEGA_stop,2)) + ", OMEGA_o = " + str(round(OMEGA_0,2)) + ", and OMEGA_p has been normalized to " + str(OMEGA_pass) + ".")

print(" ")
print(" ")
print(" ")

# create continer arrays for holding calculated values
maxOrder = 50

poleLocations = np.zeros(maxOrder)
poleLocations = np.array(poleLocations,dtype = complex)
sigma_array = np.zeros(maxOrder)
omega_array = np.zeros(maxOrder)
C_array = np.zeros(maxOrder)
D_array = np.zeros(maxOrder)
E_array = np.zeros(maxOrder)
G_array = np.zeros(maxOrder)
Q_array = np.zeros(maxOrder)
K_array = np.zeros(maxOrder)
W_array = np.zeros(maxOrder)
w0_array = np.zeros(maxOrder)
kf_array = np.zeros(maxOrder)
km_array = np.zeros(maxOrder)
individGainStage = np.zeros(maxOrder)
r1single_array = np.zeros(maxOrder)
r1_array = np.zeros(maxOrder)
r2_array = np.zeros(maxOrder)
r3_array = np.zeros(maxOrder)

print("Butterworth pole locations: ")
print("----------------------------------------------")
# determine pole locations
if(nEven == True): # even number of poles calculations
    print("Poles begin at +/-: " + str(first_poles) + "deg from the negative real axis.")
    print("Poles are spaced by: " + str(pole_spacing) + " deg")
    poleCounter = 1
    for i in iteration(1,numIterPoles,1):
        degrees = first_poles + (i-1)*pole_spacing
        degrees = degrees*np.pi/180
        positivePole = OMEGA_0*(-np.cos(degrees)+1j*np.sin(degrees))
        negativePole = OMEGA_0*(-np.cos(degrees)-1j*np.sin(degrees))
        print(" ")
        print("Pole #" + str(poleCounter) + ": " + str(round(np.absolute(positivePole),2)) + " arg " + str(round(np.angle(positivePole, deg = True),2)) + "deg ... " + str(round(positivePole,2)))
        poleLocations[poleCounter-1] = positivePole
        poleCounter += 1
        print("Pole #" + str(poleCounter) + ": " + str(round(np.absolute(negativePole),2)) + " arg " + str(round(np.angle(negativePole, deg = True),2)) + "deg ... " + str(round(negativePole,2)))
        poleLocations[poleCounter-1] = positivePole
        poleCounter += 1
        
else: # odd number of poles calcualtions
    print("A single pole lies at: " + str(first_poles) + " deg")
    print("Poles are spaced by: " + str(pole_spacing) + " deg")
    print(" ")
    poleCounter = 2
    realAxisPole = OMEGA_0*(-np.cos(0)+1j*np.sin(0))
    print("Pole #1: " + str(round(np.absolute(realAxisPole),2)) + " arg " + str(round(np.angle(realAxisPole, deg = True),2)) + " deg ... " + str(round(realAxisPole,2)))
    poleLocations[0] = realAxisPole
    for i in iteration(2,numIterPoles,1):
        degrees = (i-1)*pole_spacing
        degrees = degrees*np.pi/180
        positivePole = OMEGA_0*(-np.cos(degrees)+1j*np.sin(degrees))
        negativePole = OMEGA_0*(-np.cos(degrees)-1j*np.sin(degrees))
        print(" ")
        print("Pole #" + str(poleCounter) + ": " + str(round(np.absolute(positivePole),2)) + " arg " + str(round(np.angle(positivePole, deg = True),2)) + " deg ... " + str(round(positivePole,2)))
        poleLocations[poleCounter-1] = positivePole
        poleCounter += 1
        print("Pole #" + str(poleCounter) + ": " + str(round(np.absolute(negativePole),2)) + " arg " + str(round(np.angle(negativePole, deg = True),2)) + " deg ... " + str(round(negativePole,2)))
        poleLocations[poleCounter-1] = positivePole
        poleCounter += 1
print("----------------------------------------------")

roundToDigits = 4

# begin Geffe's algorithm calcualtions:

# calculate ∑_i and Ω_i
if(testMode == 1):
    print(" ")
    print("SIGMA_i and OMEGA_i values: ")
for i in iteration(0,numIter-1,1):
    sigma_array[i] = np.abs(np.real(poleLocations[i]))
    if(testMode == 1):
    	print("SIGMA_" + str(i+1) + ": " + str(round(sigma_array[i],roundToDigits)))
    omega_array[i] = np.abs(np.imag(poleLocations[i])) 
    if(testMode == 1):
    	print("OMEGA_= " + str(i+1) + ": " + str(round(omega_array[i],roundToDigits)))

# calculate C_i
if(testMode == 1):
    print(" ")
    print("C_i values: ")
for i in iteration(0,numIter-1,1):
    C_array[i] = np.power(sigma_array[i],2) + np.power(omega_array[i],2)
    if(testMode == 1):
    	print("C_" + str(i+1) + ": " + str(round(C_array[i],roundToDigits)))

# calculate D_i
if(testMode == 1):
    print(" ")
    print("D_i values: ")
for i in iteration(0,numIter-1,1):
    D_array[i] = 2*sigma_array[i]/q_c
    if(testMode == 1):
    	print("D_" + str(i+1) + ": " + str(round(D_array[i],roundToDigits)))

# calculate E_i
if(testMode == 1):
    print(" ")
    print("E_i values: ")
for i in iteration(0,numIter-1,1):
    E_array[i] = 4 + C_array[i]/np.power(q_c,2)
    if(testMode == 1):
    	print("E_" + str(i+1) + ": " + str(round(E_array[i],roundToDigits)))

# calculate G_i
if(testMode == 1):    
    print(" ")
    print("G_i values: ")
for i in iteration(0,numIter-1,1):
    G_array[i] = np.sqrt(np.power(E_array[i],2) - 4*np.power(D_array[i],2))
    if(testMode == 1):
    	print("G_" + str(i+1) + ": " + str(round(G_array[i],roundToDigits)))
   
# calculate Q_i
Q_total = 1
if(testMode == 1):  
    print(" ")
    print("Q_i values: ")
for i in iteration(0,numIter-1,1):
    Q_array[i] = (1/D_array[i])*np.sqrt(0.5*(E_array[i] + G_array[i]))
    Q_total = Q_total*Q_array[i]
    if(testMode == 1):
    	print("Q_" + str(i+1) + ": " + str(round(Q_array[i],roundToDigits)))

# calculate K_i
if(testMode == 1):
    print(" ")
    print("K_i values: ")
for i in iteration(0,numIter-1,1):
    K_array[i] = (sigma_array[i]*Q_array[i])/q_c
    if(testMode == 1):
    	print("K_" + str(i+1) + ": " + str(round(K_array[i],roundToDigits)))

# calculate W_i
if(testMode == 1):
    print(" ")
    print("W_i values: ")
for i in iteration(0,numIter-1,1):  
    W_array[i] = round(K_array[i],8) + np.sqrt(round(np.power(K_array[i],2),8) - 1)
    if(testMode == 1):
    	print("W_" + str(i+1) + ": " + str(round(W_array[i],roundToDigits)))

# calculate w0_i
if(testMode == 1):  
    print(" ")
    print("w0_i values: ")
for i in iteration(0,numIter-1,1): 
    if(i%2 == 0): # these are the odd w0's... (array index starts at 0)
        w0_array[i] = omega_0/W_array[i]
        if(testMode == 1):
        	print("w0_" + str(i+1) + ": " + str(round(w0_array[i],roundToDigits)))
    else: # these are the even w0's
        w0_array[i] = omega_0*W_array[i]
        if(testMode == 1):
        	print("w0_" + str(i+1) + ": " + str(round(w0_array[i],roundToDigits)))

# calculate kf
if(testMode == 1):     
    print(" ")
    print("kf_i values: ")
for i in iteration(0,numIter-1,1): 
    kf_array[i] = w0_array[i]
    if(testMode == 1):
    	print("kf_" + str(i+1) + ": " + str(round(kf_array[i],roundToDigits)))

# print back the specs
print(" ")
print(" ")
print(" ")
print("The bandpass filter will be designed using the following parameters: ")
print("----------------------------------------------")
print("n (minimum filter order):      " + str(n) + " (" + str(round(n_full,2)) + ")")
print("a_min (atten. in stop band):   " + str(a_min) + " dB")
print("a_max (atten. in pass band):   " + str(a_max) + " dB")
print("f1 (start of pass band):       " + str(round(omega_1/(2*np.pi),1)) + " Hz")
print("f2 (end of pass band):         " + str(round(omega_2/(2*np.pi),1)) + " Hz")
print("f3 (end of HPF attenuation):   " + str(round(omega_3/(2*np.pi),1)) + " Hz")
print("f4 (start of LPF attenuation): " + str(round(omega_4/(2*np.pi),1)) + " Hz")
print("f0 (center frequency):         " + str(round(omega_0/(2*np.pi),1)) + " Hz")
print("Bandwidth:                     " + str(round(band_width/(2*np.pi),1)) + " Hz")
print("Q:                             " + str(round(Q_total,1)))
print("----------------------------------------------")

# calculate the gain of each stage and total gain
print(" ")
print(" ")
print(" ")
print("Gain of each stage / Total gain: ")
print("----------------------------------------------")
gainStageCounter = 1
totalGain = 1
for i in iteration(0,numIter-1,1):
    gainStageNum = np.power((2*Q_array[i]*w0_array[i]*omega_0),2)
    gainStageDen = np.power(np.power(w0_array[i],2) - np.power(omega_0,2),2) + np.power(w0_array[i]*omega_0/Q_array[i],2)
    gainStage = np.sqrt(gainStageNum/gainStageDen)
    individGainStage[i] = gainStage
    print("The gain of stage " + str(gainStageCounter) + " would be " + str(round(gainStage,2)) + ", but has been corrected to unity.")
    gainStageCounter += 1
    totalGain = totalGain * gainStage
print(" ")
print("The overall gain would be: " + str(round(totalGain,roundToDigits)) + " or " + str(round(20*np.log10(totalGain),roundToDigits)) + " dB (but has been corrected to 0 dB)")
print("----------------------------------------------")

# find capacitor value for realistic resistor values (need to determine this before calculating k_m)
capVal = 1e-12 # start at one picofarad and then increase until reasonable level (R2 of last stage is reference)
maxIndex = numIter-1
checkVal = (1/(2*kf_array[maxIndex]*Q_array[maxIndex]*capVal))*(1/(1-(1/individGainStage[maxIndex])))
fail = 0
while(checkVal > 1000):
    if(fail == 1):
        capVal *= 1e1
    checkVal = (1/(2*kf_array[maxIndex]*Q_array[maxIndex]*capVal))*(1/(1-(1/individGainStage[maxIndex])))
    fail = 1

# caps stay unchanged... use capVal value
c1 = capVal
c2 = capVal

# calculate km_i
if(testMode == 1):
    print(" ")
    print("km_i values: ")
for i in iteration(0,numIter-1,1): 
    km_array[i] = 1/(2*kf_array[i]*Q_array[i]*capVal)
    if(testMode == 1):
    	print("km_" + str(i+1) + ": " + str(round(km_array[i],roundToDigits)))
    

# calculate all R1s (top resistor in input divider)
for i in iteration(0,numIter-1,1):
    r1_array[i] = km_array[i]*individGainStage[i]

# set all R2s (bottom resistor in input divider)
for i in iteration(0,numIter-1,1):
	r2_array[i] = km_array[i]*(1/(1-(1/individGainStage[i])))

# calculate all R3s
# this value is determined by km*4*Q^2 (feedback resistor)
for i in iteration(0,numIter-1,1):
    r3_array[i] = km_array[i]*4*np.power(Q_array[i],2)

# construct transfer functions using signal library
num = np.zeros(2)
den = np.zeros(3)

# first convolve two on their own, then loop through the rest
i = 0
firstNum = [-1/(r1_array[i]*c1), 0]
firstDen = [1, (c1+c2)/(r3_array[i]*c1*c2), (r1_array[i]+r2_array[i])/(r1_array[i]*r2_array[i]*r3_array[i]*c1*c2)]

i = 1
secondNum = [-1/(r1_array[i]*c1), 0]
secondDen = [1, (c1+c2)/(r3_array[i]*c1*c2), (r1_array[i]+r2_array[i])/(r1_array[i]*r2_array[i]*r3_array[i]*c1*c2)]

tf_num = np.convolve(firstNum, secondNum)
tf_den = np.convolve(firstDen, secondDen)

for i in iteration(2,numIter-1,1):
    num = [-1/(r1_array[i]*c1), 0]
    den = [1, (c1+c2)/(r3_array[i]*c1*c2), (r1_array[i]+r2_array[i])/(r1_array[i]*r2_array[i]*r3_array[i]*c1*c2)]
    tf_num = np.convolve(tf_num, num)
    tf_den = np.convolve(tf_den, den)

print(" ")
print(" ")
print(" ")

print("Component values:")
print("----------------------------------------------")
resistorCounter = 1
for i in iteration(0,numIter-1,1): 
    print("Rin_" + str(resistorCounter) + ": " + str(round(r1_array[i],2)))
    resistorCounter += 1
    print("Rin_" + str(resistorCounter) + ": " + str(round(r2_array[i],2)))
    resistorCounter += 1
    print("Rf_" + str(resistorCounter) + ":  " + str(round(r3_array[i],2)))
    resistorCounter += 1
    print(" ")
if(capVal*1e9 < 999):
    print("The capacitor value for all caps is: " + str(round(capVal*1e9,4)) + " nF.")
else:
    print("The capacitor value for all caps is: " + str(round(capVal*1e6,4)) + " uF.")
print("----------------------------------------------")
print(" ")

# classes for plot snap-to cursors
class Cursor(object):
    def __init__(self, ax):
        self.ax = ax
        self.lx = ax.axhline(color='k')  # the horiz line
        self.ly = ax.axvline(color='k')  # the vert line

        # text location in axes coords
        self.txt = ax.text(0.7, 0.9, '', transform=ax.transAxes)

    def mouse_move(self, event):
        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata
        # update the line positions
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)

        self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))
        plot.draw()

class SnaptoCursor(object):
    """
    Like Cursor but the crosshair snaps to the nearest x,y point
    For simplicity, I'm assuming x is sorted
    """
    def __init__(self, ax, x, y):
        self.ax = ax
        self.lx = ax.axhline(color='k')  # the horiz line
        self.ly = ax.axvline(color='k')  # the vert line
        self.x = x
        self.y = y
        # text location in axes coords
        self.txt = ax.text(0.7, 0.9, '', transform=ax.transAxes)

    def mouse_move(self, event):

        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata

        indx = np.searchsorted(self.x, [x])[0]
        x = self.x[indx]
        y = self.y[indx]
        # update the line positions
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)

        self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))
        # print('x=%1.2f, y=%1.2f' % (x, y))
        plot.draw()

# define start/end freq and freq step for bode plots
startFreq = omega_3/10
endFreq = omega_4*10
freqStep = np.ceil((endFreq-startFreq)/1000)

# bode plot for magnitude and phase
tf = signal.lti(tf_num, tf_den)
w, mag, phase = signal.bode(tf, np.arange(startFreq, endFreq, freqStep).tolist())
fig, ax = plot.subplots()

plot.semilogx(w/(2*np.pi), mag, color="blue", linewidth="2")
#plot.autoscale(enable=False, axis='both', tight=False)
plot.xlabel("Frequency (Hz)")
plot.ylabel("Magnitude (dB)")
plot.axis([omega_3/(2*np.pi)/10, omega_4/(2*np.pi)*10, -100, 10])
cursor = SnaptoCursor(ax, w/(2*np.pi), mag)
plot.connect('motion_notify_event', cursor.mouse_move)

plot.show()