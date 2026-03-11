In this directory (arctic_reconstruction/) make a script named build_model_input.py. The goal of this script is to assemble all the required data to later run the model trained by lstm_pytorch_pd_mcdo.py to predict profiles all over the arctic. We will also add some additional information, such as ocean masks and bathymetry. For that, the data have been already interpolated to an ease horizontal grid and, when needed, the same depth levels as the profiles used for training the neural network.

See the other scripts in the directory to understand the processed data that was created. This new script will merge the information from the following files or directories:

1. arctic_reconstruction/data/gebco_ease_grid_25km.nc
This file contains bathymetry information.

2. arctic_reconstruction/data/ocean_mask_ease_grid_25km.nc
This file contains a mask indicating ocean locations

3. SST (sea surface temperature, coming as analysed_sst) is in directories organized by year and month as: /home/nico/Desktop/AUX_DIR_FRESH_CARE/satellite/SST/data_ease/YYYY/MM/
The whole dataset could opened using a glob pattern with xr.open_mfdataset() but it's not a good idea, it takes time and it's just a lazy load.

4. SSS (sea surface salinity, coming as sss) will be in /home/nico/Desktop/AUX_DIR_FRESH_CARE/satellite/SSS_cci_v55/regridded_filled_wg_ease/
Organized in yearly files, such as sss_merge_cci_2010_regridded_025_filled_wg.nc

These can also be opened with xr.open_mfdataset(), but there should be no need, i am just informing that the files are aligned with the time coordinate.

5. ADT (absolut dynamic topography, coming as adt) will be in 
/home/nico/Desktop/AUX_DIR_FRESH_CARE/satellite/ADT/aviso_regridded_0.25_north_pole_interp_ease/YYYY/ where YYYY indicates the year. Every yearly directory contains many files, one per timestep. These can also be opened with open_mfdataset() thanks to the time coordinate alignment, but don't do it all at once either (RAM is little)

6. T_glorys (coming as thetao), S_glorys (coming as so) and SH_glorys (coming as SH) will be extracted from this directory: /home/nico/Desktop/AUX_DIR_FRESH_CARE/glorys_2012_ease_woaDepths/
This corresponds to 2012 data, it is also aligned with time coordinate accross nc files. Avoid loading full 3d variables if opened with open_mfdataset (thetao, so, SH)

### How to do it:
Inspect the scripts, files and directories so you understand the context.

Set times steps as the day 15 or every month (monthly steps), you can take as example (not as dinamical reference) the timesteps from the 2012 glorys dataset. Then build a dataset with all the data from points 3,4,5,6 linearly interpolating the data to these timesteps. Make sure to always take a time slice that covers before and after the target time (sometimes you may need to access a previous/next file from a parallel directory for this). You may load a 16 day time slice at each step or something like that, just to make sure you cover several timesteps.

To this dataset add the information from points 1 and 2.

Finally add a variable name DOY that is the day of the year (integer), which will only depend on the time coordinate.

### Considerations
All data except the DOY is spatially gridded on a 25 km EASE grid, both in  input and output. Some data may have depth and/or time dependency as well.

Temperature

We will run the script using only 2012 data, but this script should be adaptable to run over any period. don't really know how to make this user friendly...

use decode times true so that you can read the data in datetime64 without mixing different time units. But the final output must have units set to days since 1950-01-01 (with the good spelling format).