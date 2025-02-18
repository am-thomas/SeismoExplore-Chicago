# SeismoExplore-Chicago
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
* Download the EQ-Chicago repository
```
git clone https://github.com/am-thomas/SeismoExplore-Chicago.git
```
* Navigate to the SeismoExplore-Chicago folder and install the remaining requirements: 
```
cd SeismoExplore-Chicago
pip install -r requirements.txt
```
