# ML Zoomcamp midterm project - Sleep Detection
This is a midterm project of ML Engineering Zoomcamp by DataTalks - Cohort 2023

The goal of this project is to detect sleep from the data recorded by wrist-worn accelerometer for sleep monitoring.<br>
This is a simplified problem of the one from the Kaggle competition - ["Child Mind Institute - Detect Sleep States - overview"](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states).

## Contents:
1. [Exploratory Data Analysis - Notebook](https://github.com/Olks/mlzoomcamp-midterm-project/blob/main/sleep_detection_eda.ipynb)
2. [Model Training - Notebook](https://github.com/Olks/mlzoomcamp-midterm-project/blob/main/model_training.ipynb)


## Project Dataset
Original Kaggle data is too large to be trained on a single computer so here only the subset of data is included.<br>
- 120 out of 277 series_id,
- 3 first days of of each 120 series (originally they have about 14 or more days).<br>

Additionally, to decrease the size of the dataset we calculate one minute values (mean, max, min, std).<br>

Below, there is a plot for one series_id. 
<img src="series_plot.png"/>

The final dataset that will be used in the project have the following fields:

| variable | describtion |
|:---|:---|
| series_id | Unique identifier for each accelerometer series. |
| step | An integer timestep for each observation within a series. |
|dt_minute| Observation timestamp truncated to full minutes. |
|anglez_mean | One minute mean of anglez values. While the original data contains every 5 seconds data then here we have the mean of 12 signals. |
|anglez_std | One minute standard deviation of anglez values. While the original data contains every 5 seconds data then here we have the standard deviation of 12 signals.	
| anglez_max  | One minute maximum of anglez values. While the original data contains every 5 seconds data then here we have the maximum value of 12 signals. |	
| anglez_min | One minute maximum of anglez values. While the original data contains every 5 seconds data then here we have the maximum value of 12 signals. |	
| enmo_mean	| One minute mean of enmo values. While the original data contains every 5 seconds data then here we have the mean of 12 signals. |
| enmo_max | One minute maximum of enmo values. While the original data contains every 5 seconds data then here we have the maximum value of 12 signals. |	
| enmo_min | One minute minumum of enmo values. While the original data contains every 5 seconds data then here we have the minimum value of 12 signals. |	
| anglez_1st_diffs_sum | One minute sum of first differences of anglez values. |
| enmo_1st_diffs_sum | One minute sum of first differences of enmo values. |	
| <b>target</b> | The label of one of three values: <b>0 - awake, 1 - asleep, 2 - device not worn.</b> |

## Original Dataset


The original dataset comes from Kaggle competition "Child Mind Institute - Detect Sleep States - data".<br>
It comprises of every 5 seconds accelerometer signals transformed into two variables:
- <b>ENMO</b>: The Euclidean Norm Minus One (ENMO) with negative values rounded to zero.
  <br>It has been shown to correlate with the magnitude of acceleration and human energy expenditure.
  <br>ENMO is computed as follows:<br>
  <!-- width="350" height="350" -->
  <img src="enmo.jpg" width="310" height="70"/>
- <b>anglez</b>: Z-angle, computed using the equation below; corresponds to the angle between the accelerometer axis perpendicular to the skin surface and the horizontal plane.
  <br>Any change (or lack of change) in the z-angle over successive time intervals may be an indicator of posture change.<br>
  <img src="anglez.jpg" width="300" height="70"/>





