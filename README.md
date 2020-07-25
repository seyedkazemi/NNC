# NNC

To make a final decision about the anomalies, we introduced the NNC which is composed of the five sub-NNCs each of these sub-NNCs increase the NNC by one unit when they detect any anomalies.

Based on the 15 days we create a tool to compute the NNC from day 16 we start to compute the NNCs, and we compute the average NNC for last 15 days, and in the day that we want to check the anomaly, we compute the NNC if the variation of NNC of check day from average NNC of last 15 days is higher than 30% there is an anomaly in the behavior of the person.   


Datasets:
ARAS (Real-world data sets for Activity Recognition with Ambient Sensing)

http://aras.cmpe.boun.edu.tr/

https://ieeexplore.ieee.org/document/6563930

from ARAS datasets Selected House A


Pay Attention:

           To use this code, the reading part of the datasets must be changed

DATASETS :
           Change to CSV format 
           Add a row (first row) for the sensor name and resident number (In short)
           Add a column to show the seconds.

Anomaly state is checked for resident 1 in House A

We divided the datasets into two equal parts

We used all the second 15 days of the datasets in  to check for anomalies

You can change this division

The first 15 days (two weeks) are used for the initial learning of the algorithm.

from the sixteenth day, NNC begins to be calculated.

Feel free, You can ask questions via email:  
e.seyedkazemi@yahoo.com
