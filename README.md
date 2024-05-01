# hot-dust 
Notebooks and scripts that analyze infrared data from the VIIRS satellite instrument and train a neural network to quantify the dust in the atmosphere.  

## Aerosols and the VIIRS Satelite 
Monitoring aerosols (which includes dust, smoke, volcanic ash, sea spray, sulfate, and more) is important for climate, air quality, and hazard avoidance purposes. Being able to identify the amounts of specific types of aerosol particle in the atmosphere makes these tasks easier. However, to date, most approaches applied to satellite data are able to determine only the total columnar aerosol optical thickness (AOT, an optical measure of aerosol amount) and not robustly quantify individual components.

The [Visible Infrared Imaging Radiometer Suite (VIIRS)](https://www.earthdata.nasa.gov/sensors/viirs) has 22 spectral bands, ranging from visible to thermal infrated (IR). There are 16 moderate resolution bands, 5 imaging bands, and 1 day-night band. VIIRS flies on multiple Earth-orbiting satellites, and other satellites carry instrumentation with similar IR capabilities. The IR bands on radiometers such as the VIIRS instruments are sensitive to coarse elevated aerosols like mineral dust, but not other smaller-sized aerosols.

## Estimating Dust Aerosol Optical Thickness with IR Bands

## Aim 
We trained a neural network model to quantify dust in the atmosphere utilizing simulated data. The simulations were performed using the freely-available [libRadtran](https://libradtran.org/doku.php) radiative transfer code. They provide a labeled truth data set for the prediction target (i.e. AOT). After analyzing the model's performance on simulated data applied the network on actual VIIRS data from NASA's data archive [LAADS](https://ladsweb.modaps.eosdis.nasa.gov/). 

In order to do this we had to perform several steps in order to prepare the data for the network. 
  1. Step 1
  2. Step 2

## Results  
On the simulated data the model performed well. When applied to actual VIIRS data the network did not predict realistic dust optical thickness, suggesting that something about the simulated data was not sufficiently realistic. See the `evaluate` evaluate notebook.  

## Data avalibility 
Don't know if I need this section yet...
