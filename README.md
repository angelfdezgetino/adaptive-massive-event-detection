# Adaptive Massive Event Detection Algorithm

This repository contains a Python algorithm for detecting massive event-related demand peaks in public transport station time series.

The method combines seasonal decomposition, residual behavior analysis, a Hampel-based preprocessing stage, and adaptive event detection using either IQR or MAD to compute thresholds depending on the detected temporal behaviour.

## Overview

The pipeline is designed to process daily station demand data and identify dates associated with unusual demand increases that may correspond to massive events.

The workflow includes:

1. **Time series preprocessing**
2. **Seasonal decomposition**
3. **Computation of CEI/PEI indicators**
4. **Hampel-based filtering and STL decomposition**
5. **Adaptive event detection**
   - **IQR-based detection** for collective event behaviours
   - **MAD-based detection** for punctual event behaviours
6. **Optional evaluation** against known event calendars
7. **Optional visualization** for single-station analysis
8. **CSV export** of detected events

## Repository contents
```text
metro-event-detection/
├── .gitignore
├── README.md
├── requirements.txt
├── event_detection_pipeline.py
└── data/
    └── README.md
```
## Main script

The main script in this repository is:

* `event_detection_pipeline.py`

The code is provided as a single script in order to preserve the original research workflow and execution logic.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

The required Python packages are:

* `numpy`
* `pandas`
* `statsmodels`
* `matplotlib`
* `scikit-learn`

## Data availability

The datasets used in this project are **not publicly available** and are therefore **not included in this repository**.

This repository only provides the source code for the event detection methodology. Users who wish to run the pipeline must provide their own compatible datasets and reproduce the folder structure expected by the script.

## Expected input data

This script expects external CSV files stored in relative paths.

### Main ridership dataset

The main input file is expected at:

```text
../data/metro_madrid/daily_records_since_2017.csv
```

This file must contain at least the following columns:

* `FECHA` — date
* `VALOR` — daily ridership value
* `NOMBRE` — station name

The script also optionally uses:

* `ORDEN` — station order within the metro line

### Optional ground-truth event files

For evaluation against known events, the script expects the following files:

```text
../data/eventos_ventas_completo.csv
../data/eventos_santiago_bernabeu.csv
../data/eventos_estadio_metropolitano.csv
```

These files must contain at least:

* `FECHA` — event date

## Execution

Run the script from the command line:

```bash
python event_detection_pipeline.py
```

When executed, the script prompts the user for two inputs:

* **Station**: exact station name or `All`
* **Year**: a specific year such as `2023` or `All`

Example:

```text
Station (exact name or All): Ventas
Year (e.g., 2023 or All): 2023
```

### Supported execution modes

The script supports four main execution modes.

#### 1. Single station, single year

```text
Station (exact name or All): Ventas
Year (e.g., 2023 or All): 2023
```

This mode analyzes one station for one specific year.

#### 2. Single station, all available years

```text
Station (exact name or All): Ventas
Year (e.g., 2023 or All): All
```

This mode analyzes one station across all available years in the selected date range.

#### 3. All stations, single year

```text
Station (exact name or All): All
Year (e.g., 2023 or All): 2023
```

This mode analyzes all stations for one specific year.

#### 4. All stations, all available years

```text
Station (exact name or All): All
Year (e.g., 2023 or All): All
```

This mode runs the full pipeline over all stations and all available years.

### Input validation

Station names must match the names available in the input CSV. The year must be an integer available in the filtered dataset, or `All`. If an invalid station or year is entered, the script raises an error.

### Internal execution flow

For each selected station and year, the script performs the following steps:

1. Load and preprocess the station time series.
2. Compute the baseline seasonal decomposition.
3. Estimate the CEI and PEI indicators.
4. Apply the Hampel-based preprocessing and STL decomposition.
5. Select the detection mode:

   * **IQR-based detection** for collective behavior
   * **MAD-based detection** for punctual behavior
6. Filter low-level detections.
7. Compute evaluation metrics when ground-truth data is available.
8. Save detected events to a CSV file.

### Evaluation

If a station has an associated ground-truth event file, the script computes:

* confusion matrix
* precision
* recall
* accuracy
* ROC/AUC score when possible

## Plots

Plots are only generated when a single specific station is selected.

In that case, the script may display figures such as:

* observed series and minimum rolling baseline
* observed vs expected series
* residuals vs detection threshold
* STL trend and seasonal components
* ROC curve, when ground-truth labels are available

## Output

Detected events are appended to the configured CSV output file.

By default, the script writes results to:

```text
../data/detection_events/events_detected_all.csv
```

The exported file includes the following columns:

* `station_name`
* `year`
* `date`
* `resid`
* `threshold`
* `mode`

## Notes on execution
The current configuration is set for the **post-COVID** period:

* `START_DATE = "2021-01-01"`
* `END_DATE = "2024-09-30"`


