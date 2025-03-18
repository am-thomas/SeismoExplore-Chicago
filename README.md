# SeismoExplore-Chicago
[![DOI](https://zenodo.org/badge/934328936.svg)](https://doi.org/10.5281/zenodo.15047292)

Detect and cluster anomalous seismic events in a noisy urban environment (Greater Chicago area). Applies a PSD-misfit detector and K-means clustering for event detection and clustering, respectively.

## Installation 
* Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for your OS 
* Create a new conda environment as follows: 
```
conda create -n seismo python=3.10
```
* Activate the new conda environment as follows: 
```
conda activate seismo
```
* Install Obspy 
```
conda install -c conda-forge obspy
```
* Download the SeismoExplore-Chicago repository
```
git clone https://github.com/am-thomas/SeismoExplore-Chicago.git
```
* Navigate to the SeismoExplore-Chicago folder and install the remaining requirements: 
```
cd SeismoExplore-Chicago
pip install -r requirements.txt
```
## Sample Code
This section provides instructions to apply our detection and clustering workflow for 7 days in 2017 (2017-10-01 to 2017-10-07). The default parameters are for station NW.HQIL in suburban Chicago but you run the code for a different station by adding the appropriate station parameters (i.e., net, sta, loc, chan/chan_list, samp_rate) as command-line arguments. 

1. Activate conda environment and navigate to the src folder
```
conda activate seismo
cd src
```
2. Get 3-component waveforms from EarthScope/IRIS Web servicees and save miniSEED files locally
```
python download.py --start 2017-09-30 --end 2017-10-07
```
3. Compute and save PSD misfits for all three components in the 7-day period
```
python get_PSDmisfit.py --start_time 2017-10-01T00:00:00 --days 7 --year 2017 --exp_name 2017-10-01_7days --chan HH1
```
```
python get_PSDmisfit.py --start_time 2017-10-01T00:00:00 --days 7 --year 2017 --exp_name 2017-10-01_7days --chan HH2
```
```
python get_PSDmisfit.py --start_time 2017-10-01T00:00:00 --days 7 --year 2017 --exp_name 2017-10-01_7days --chan HHZ
```
4. Compute statistical features for the 7-day period***
```
python get_statfeatures.py --start_time 2017-10-01T00:00:00 --days 2 --exp_name 2017-10-01_7days
```
5. Apply PSD misfit detector and create a csv of all feature values for detected events
```
python create_featcsv.py --exp_name 2017-10-01_7days --days 7 --start_day 2017-10-01
```
6. Apply a K-means clustering model on the 7-day period and save labels and cluster centers
```
python cluster.py --load_expdirs 2017-10-01_7days --load_savedmodel --clust_dir clusters_2017-10-01_7days
```

***get_statfeatures.py and its dependies are adapted from our Chicago earthquake detection study. The program computes statistical features for all overlapping (50%) 10-s windows in continous data. If adapting our workflow to a different environment, we recommend modifying the file to only compute statistical features on detected 10-s windows and avoid unncessarily computation.

The above sample code utilizes model features, background PSD windows, and a pre-trained model used in our final study. This repository contains code and capacities to select your own parameters and train your own model. See the header text and command-line arguments for each Python file to learn more. This repository also contains optional files to aid in feature selection (feature_analysis.py), k-selection (select_k.py), and event visualization (viz_eventtype.py). Thank you for visiting this page and learning more about our work. 
