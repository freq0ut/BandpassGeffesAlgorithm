# Bandpass-Geffes-Algorithm

[Click here for my Blog post on Geffe's Algorithm Python Script](http://www.zackslab.com/2016/05/bandpass-filter-design-using-geffes-algorithm/) 


# Introduction

Analog filter design can be a tedious task, requiring frequency transformations, pages and pages of algebra (which is prone to calculation errors), and many simulations to check that you are getting the response you expected to. Through the use of Geffe’s algorithm and Delyiannis-Friend circuits, bandpass filter design can be partially automated. The purpose of this post isn’t to detail the specifics behind Geffe’s algorithm, but rather to share with you a python script I wrote that will apply Geffe’s algorithm to a given set of input parameters.

It is required that you have Python 3 along with numpy, matplotlib, and scipy installed on your machine. The script can be executed through the Powershell if you are a windows user by navigating to the directory where the Geffe_Friend_Bandpass.py file is located and then typing: python .\Geffe_Friend_Bandpass.py



# Entering Input Parameters

Upon execution, the python script will prompt you for some input parameters.

You are first asked for the maximum attenuation in the stop bands and the minimum attenuation in the pass band.

Then, you are asked to specify the pass band frequencies (f1 and f2) and one of the stop band frequencies (f3 or f4). The other stop band frequency is solved for automatically. See the following image [taken from M.E. Van Valkenburg's Analog Filter Design]:

[![Input](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/BPattenuation.png)](#features)

[![Input](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_image1.png)](#features)



# Parameter Details

The script will then return to you the information you had just entered, along with the minimum order filter required to meet your specifications.

[![Parameter Details](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_image2.png)](#features)



# Butterworth Pole Locations

The locations of the poles are returned in both polar and rectangular format.

[![Butterworth Poles](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_image3.png)](#features)



# Gain Information

The gain of each stage is normalized to unity, resulting in unity gain at the center frequency.

The script can be tweaked to create a filter/amplifier, or an additional amplifier can be added after the filter.

[![Gain Picture](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_image4.png)](#features)



# Resistor and Capacitor Values

The script will automatically calculate reasonable component values for each Friend circuit.

Note that n operational amplifiers (Friend circuits) are required for an nth order filter.

[![Resistor and Cap Output](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_image5.png)](#features)

Click on the image to enlarge it, notice how the resistors and capacitors correspond to the output generated by the python script.

[![Schematic](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_schematic.png)](#features)

This example can be used as a general case, and the same arrangement will arise for any nth order output.



# Auto Generated Bode Magnidute Plot

The script will then call on matplotlib to generate the bode magnitude plot of the filter. Interactive cursors have been added to allow for inspection of the pass and stop band frequencies.

[![Plot Output](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_plot.png)](#features)



# LT Spice AC Sweep

Here is the frequency response of the circuit once it had been built in LTSpice:

[![LTSpice Plot Output](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/geffe_lt_plot.png)](#features)



# Conclusion

So what is this script good for? Let’s say you wanted to build a circuit that required multiple analog bandpass filters, but don’t want to crank out all the math to calculate the component values.

For example, to create an analog audio spectrum analyzer, you would need 8-10 bandpass filters to select for various bands of the audio spectrum. You could use this script to design 8 bandpass filters very quickly!

[![LTSpice Plot Output](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/audioBandpassFiltersSch.png)](#features)

[![LTSpice Plot Output](https://github.com/freq0ut/Bandpass-Geffes-Algorithm/blob/master/Pics/audioBandpassFiltersPlot.png)](#features)

[Click here for the Python Script](/Geffe_Friend_Bandpass.py) 

-Zack
