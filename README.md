# hot-dust 
Notebooks and scripts that analyze infrared data from the VIIRS satellite instrument and train a neural network to quantify the dust in the atmosphere.  

## VIIRS Satelite and Infrared Bands 
[Visible Infrared Imaging Radiometer Suite (VIIRS)](https://www.earthdata.nasa.gov/sensors/viirs) has 22 spectral bands, ranging from visible to longwave IR. There are 16 moderate resolution bands, 5 imagining bands, and 1 day-night band. The thermal infrared (IR) bands on radiometers such as the VIIRS instruments are sensitive to coarse elevated aerosols like mineral dust, but not other smaller-sized aerosols. 

## Estimating Dust Aerosol Optical Thickness with IR Bands


## Aim 
We trained a neural network model to quantify dust in the atmosphere utilizing simulated data. After analyzing the model's performance on simulated data applied the network on actual VIIRS data from [AERONET](https://aeronet.gsfc.nasa.gov/). 

In order to do this we had to preform several step in order to prepare the data for the network. 
  1. Step 1
  2. Step 2

## Results  
On the simulated data the model prefored well. When applied to actual VIIRS data the network did not predict realistic dust optical thickness. See the notebook in the repsitory **evaluate**.  

## Data avalibility 
Don't know if I need this section yet...
