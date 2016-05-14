# Bandpass-Geffes-Algorithm
Python program for bandpass filter design


## Using this Python script


# Entering Input Parameters
[![solarized dualmode](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_image1.png)](#features)

Upon execuition, the python script will prompt you for some input parameters.

You are first asked for the maximum attenuation in the stop bands, and the minimum attenuation in the pass bands.

Then, you're asked for the pass band frequencies and one of the attenuation frequencies, the other one is solved for automatically, you can do this by choosing option (3) or (4) when prompt to do so. See the following figure.



# Parameter Details
[![solarized dualmode](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_image2.png)](#features)

The script will then return to you the information you had just entered, along with the minimum order filter required to meet your specifications.



# Butterworth Pole Locations
[![solarized dualmode](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_image3.png)](#features)

The locations of the poles are returned in both polar and rectangular format.



# Gain Information
[![solarized dualmode](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_image4.png)](#features)

The gain of each stage is normalized to unity, resulting in unity gain at the center frequency. 

The script can be tweaked to create a filter/amplifier, or an additional amplifier can be added after the filter.



# Resistor and Capacitor Values
[![solarized dualmode](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_image5.png)](#features)

This script is used for bandpass filter design using Friend Circuits. The script will automatically calcualte reasonable component values for each Friend Circuit.

n operational amplifiers (Friend Circuits) are required for an nth order filter.

