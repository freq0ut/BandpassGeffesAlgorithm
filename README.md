# Bandpass-Geffes-Algorithm
[Click here for the Python Script](/Geffe_Friend_Bandpass.py) 


# Using this Python script

It is required that you have Python 3 along with numpy, matplotlib, and scipy installed on your machine. The script can be execuited through the Powershell if you are a windows user by navigating to the directory where the Geffe_Friend_Bandpass.py file is located and then typing: python .\Geffe_Friend_Bandpass.py



# Entering Input Parameters

Upon execuition, the python script will prompt you for some input parameters (Figure 1).

You are first asked for the maximum attenuation in the stop bands and the minimum attenuation in the pass band.

Then, you are asked to specify the pass band frequencies (f1 and f2) and *one* of the stop band frequencies (f3 or f4). The other stop band frequency is solved for automatically.

[![Input](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_image1.png)](#features)
Figure 1


# Parameter Details

The script will then return to you the information you had just entered, along with the minimum order filter required to meet your specifications (Figure 2).

[![Parameter Details](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_image2.png)](#features)
Figure 2



# Butterworth Pole Locations

The locations of the poles are returned in both polar and rectangular format (Figure 3).

[![Butterworth Poles](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_image3.png)](#features)
Figure 3



# Gain Information

The gain of each stage is normalized to unity, resulting in unity gain at the center frequency. 

The script can be tweaked to create a filter/amplifier, or an additional amplifier can be added after the filter.

[![Gain Picture](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_image4.png)](#features)
Figure 4



# Resistor and Capacitor Values

This script is used for bandpass filter design using Friend Circuits. The script will automatically calcualte reasonable component values for each Friend Circuit (Figure 5). 

n operational amplifiers (Friend Circuits) are required for an nth order filter.

In this particualr example, the output values correspond to the following topology. This example can be used as a general case, and the same arrangement will arise for any nth order output (Figure 6).

[![Resistor and Cap Output](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_image5.png)](#features)
Figure 5

[![Schematic](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_schematic.png)](#features)
Figure 6



# Auto Generated Bode Magnidute Plot

The script will then call on matplotlib to generate the bode magnitude plot of the filter. Interactive cursors have been added to allow for inspection of the pass and stop band frequencies.

[![Plot Output](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_plot.png)](#features)
Figure 7


# LT Spice AC Sweep

Here is the frequency resposne of the circuit once it had been built in LTSpice:

[![LTSpice Plot Output](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_lt_plot.png)](#features)
Figure 8

[Click here for the Python Script](/Geffe_Friend_Bandpass.py) 

-Zack
