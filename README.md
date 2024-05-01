# hot-dust 
Notebooks and scripts that analyze infrared data from the VIIRS satellite instrument and train a neural network to quantify the dust in the atmosphere.  

## Aerosols and the VIIRS Satelite 
Monitoring aerosols (which includes dust, smoke, volcanic ash, sea spray, sulfate, and more) is important for climate, air quality, and hazard avoidance purposes. Being able to identify the amounts of specific types of aerosol particle in the atmosphere makes these tasks easier. However, to date, most approaches applied to satellite data are able to determine only the total columnar aerosol optical thickness (AOT, an optical measure of aerosol amount) and not robustly quantify individual components.

The [Visible Infrared Imaging Radiometer Suite (VIIRS)](https://www.earthdata.nasa.gov/sensors/viirs) has 22 spectral bands, ranging from visible to thermal infrated (IR). There are 16 moderate resolution bands, 5 imaging bands, and 1 day-night band. VIIRS flies on multiple Earth-orbiting satellites, and other satellites carry instrumentation with similar IR capabilities. The IR bands on radiometers such as the VIIRS instruments are sensitive to coarse elevated aerosols like mineral dust, but not other smaller-sized aerosols.

## Estimating Dust Aerosol Optical Thickness with IR Bands

## Aim 
We trained a neural network model to quantify dust in the atmosphere utilizing simulated data. The simulations were performed using the freely-available [libRadtran](https://libradtran.org/doku.php) radiative transfer code. They provide a labeled truth data set for the prediction target (i.e. AOT). After analyzing the model's performance on simulated data applied the network on actual VIIRS data from NASA's data archive [LAADS](https://ladsweb.modaps.eosdis.nasa.gov/). 

## The Model
In order to do this we had the network take the _sensor zenith angle_, _surface pressure_, _water vapor_, _ozone_, _surface wind speed_, _surface temperature_, and VIIRS bands _M14_, _M15_, and _M16_ as inputs. The network then predicts the base 10 logarithm of dust aerosol optical thickness (AOT) at 10 μm. Hyperparameters for model development include use of the Adam optimizer with learning rate of 3e-4, a single hidden layer with 32 nodes, a batch size of 64 samples, mean squared error (MSE) loss, and early termination after 20 epochs without improvement in the loss calculated on the validation data.

There are two addtional models that can be found in the branches labelled `2-output` and `subtracted-bt`. The `2-output` model removes _surface temperature_ as an input variable and makes it an output variable. The model then predicts _dust optical thickness_ and _surface temperature_. The `subtracted-bt` model takes the brightness temperature differences as inputs instead of the brightness temperature bands (i.e. _"M14-M15"_ and _"M15-M16"_ as opposed to _M14_, _M15_, and _M16_). Then the model predicts the _dust optical thickness_. 

The changes to the models can be found in thier respective branches and `preprocess` notebooks. 

## Results  
On the simulated data the model performed well. When applied to actual VIIRS data the network did not predict realistic dust optical thickness, suggesting that something about the simulated data was not sufficiently realistic. See the `evaluate` evaluate notebook. 



## Data avalibility 
The data that was utilized to the train and test the model can be found on [Zendo](https://doi.org/10.5281/zenodo.11098890). 
