# Bandpass-Geffes-Algorithm
Python program for bandpass filter design


# Using this Python script

It is required that you have Python 3 along with numpy, matplotlib, and scipy installed on your machine. The script can be execuited through the Powershell if you are a windows user by navigating to the directory where the Geffe_Friend_Bandpass.py file is located and then typing: python .\Geffe_Friend_Bandpass.py



# Entering Input Parameters
[![Input](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_image1.png)]

Upon execuition, the python script will prompt you for some input parameters.

You are first asked for the maximum attenuation in the stop bands, and the minimum attenuation in the pass bands.

Then, you're asked for the pass band frequencies and one of the attenuation frequencies, the other one is solved for automatically, you can do this by choosing option (3) or (4) when prompt to do so. See the following figure.



# Parameter Details
[![Parameter Details](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_image2.png)](#features)

The script will then return to you the information you had just entered, along with the minimum order filter required to meet your specifications.



# Butterworth Pole Locations
[![Butterworth Poles](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_image3.png)](#features)

The locations of the poles are returned in both polar and rectangular format.



# Gain Information
[![Gain Picture](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_image4.png)](#features)

The gain of each stage is normalized to unity, resulting in unity gain at the center frequency. 

The script can be tweaked to create a filter/amplifier, or an additional amplifier can be added after the filter.



# Resistor and Capacitor Values
[![Resistor and Cap Output](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_image5.png)](#features)

This script is used for bandpass filter design using Friend Circuits. The script will automatically calcualte reasonable component values for each Friend Circuit.

n operational amplifiers (Friend Circuits) are required for an nth order filter.

In this particualr example, the output values correspond to the following topology. This example can be used as a general case, and the same arrangement will arise for any nth order output.

[![Schematic](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_schematic.png)](#features)



# Auto Generated Bode Magnidute Plot
[![Plot Output](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_plot.png)](#features)

The script will then call on matplotlib to generate the bode magnitude plot of the filter. Interactive cursors have been added to allow for inspection of the pass and stop band frequencies.

Here is the frequency resposne of the circuit once it had been built in LTSpice:
[![LTSpice Plot Output](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_lt_plot.png)](#features)
