# Models
This section contains example .yaml files for models.
Note: ${} means that it should be replaced with a relevant value

## Fault Isolation
    Classification models
    
### Decision Tree

```
load_object :
  module: framework.FaultIsolation
  name: FaultIsolationExperiment
setup:
    datetime_column : ${ds} #-> datetime changes with dataset
    target : ${target} # -> target column name changes with dataset
eda:
create_model :
    model : "dt" 
    params:

```

### RandomForest
```
load_object :
  module: framework.FaultIsolation
  name: FaultIsolationExperiment
setup:
    datetime_column : ${ds} #-> datetime changes with dataset
    target : ${target} # -> target column name changes with dataset
eda:
create_model :
    model : "rf" 
    params:
```

### NaiveBayes

```
load_object :
  module: framework.FaultIsolation
  name: FaultIsolationExperiment
setup:
    datetime_column : ${ds} #-> datetime changes with dataset
    target : ${target} # -> target column name changes with dataset
eda:
create_model :
    model : "nb"
    params:
```

### Hidden Markov Models
```
load_object :
  module: framework.FaultIsolation
  name: FaultIsolationExperiment
setup:
    datetime_column : ${ds} #-> datetime changes with dataset
    target : ${target} # -> target column name changes with dataset
eda:
create_model :
    model : "hmm" 
    params:
      n_iter : 1000 # how many iterations
      covariance_type : "diag"
      n_mix: 10 #
```

### Bayes Network
```
load_object :
  module: framework.FaultIsolation
  name: FaultIsolationExperiment
setup:
    datetime_column : ${ds} #-> datetime changes with dataset
    target : ${target} # -> target column name changes with dataset
eda:
create_model :
    model : "bn" # Random Forest
    params: 
      learningMethod : 'MIIC'
      prior : 'Smoothing'
      priorWeight : 1
      discretizationNbBins : 30 #'elbowMethod',
      discretizationStrategy : "quantile" #"quantile", "uniform", "kmeans", "NML", "CAIM" and "MDLP"
      discretizationThreshold : 0.01
      usePR : False
```
For furter information on the parameters, please see the [link](https://pyagrum.readthedocs.io/en/latest/skbnClassifier.html)
NOTE: This model's predict cannot be used without target variable. 
Originally this model would have been used with autoML, but the package continously referenced a null-pointer, and so were omitted.
### Markov Chain
```
load_object :
  module: framework.FaultIsolation
  name: FaultIsolationExperiment
setup:
    datetime_column : ${ds} #-> datetime changes with dataset
    target : ${target} # -> target column name changes with dataset
eda:
create_model :
    model : "mc" 
    params: 
```

### Fault Isolation Bugs

- BayesNet fails if not started locally (cannot find pyAgrum in Docker)

## Time Series Analysis

### ARIMA + SARIMA
 
```
load_object :
  module: framework.TimeSeriesAnalysis
  name:  TimeSeriesAnomalyExperiment
setup: 
    ds : ${ds} #-> datetime changes with dataset
    target : ${y} # -> time series column name, changes with dataset
eda:    ## generating EDA, can be removed
create_model :
    model : "arima"
    params:  
      start_p : 10 #  start value for p
      d : #differencing order
      start_q : 10 #start value for q
      max_p : 100 # max value for p
      max_q : 100 # max value for p
      seasonal : False
      threshold_for_anomaly : 3 #threshold for anomaly detection
```

### Autoencoder
```
load_object :
  module: framework.TimeSeriesAnalysis
  name:  TimeSeriesAnomalyExperiment
setup: 
    ds : ${ds} #-> datetime changes with dataset
    target : ${y} # -> time series column name, changes with dataset
eda:
create_model :
    model : "ae"
    params:
      layer_no : 6 # number of layers
      window : 250 # moving window size
      epoch_no : 10 ## how many times?
      batch_size : 64 ## 
      shuffle : False ## Shuffle data
      threshold_for_anomaly : 3
      neuron_no_enc : [30, 25, 20, 15, 10,5] # number of cells in encoder part
      # - must be as much integers as layers
      neuron_no_dec : [5,10, 15, 20, 25, 30] # number of cells in decoder part
      # - must be as much integers as layers
      act_enc : 'relu' # layer type for cells in encoder (Rectified Linear Unit activation)
      act_dec : 'relu' # layer type for cells in decoder 
```

### Exponential Smoothing
```
load_object :
  module: framework.TimeSeriesAnalysis
  name:  TimeSeriesAnomalyExperiment
setup: 
    ds : ${ds} #-> datetime changes with dataset
    target : ${y} # -> time series column name, changes with dataset
eda:
create_model :
    model : "es"
    params:
      trend : "mul" # determining the trend -multiplication/addition type
      seasonal :  # use seasonal analysis
      seasonal_periods :  # ...
      threshold_for_anomaly : 2.5 
      freq :  # day, hour, year, by general notation ("D", )
```
Parameters (add to params): 

- trend : str (default = None) Type of trend component.
- damped_trend : bool (default = False) Should the trend component be damped.
- seasonal : str (default = None) Type of seasonal component.
- seasonal_periods : int (default = None)
The number of periods in a complete seasonal cycle, e.g., 4 for quarterly data or 7 for daily data with a weekly cycle.
- freq : str (default = None)
Frequency of the time series. A Pandas offset or 'B', 'D', 'W', 'M', 'A', or 'Q'.
- threshold_for_anomaly : float (default = 3) Threshold for anomaly detection.

### LSTM

```
load_object :
  module: framework.TimeSeriesAnalysis
  name:  TimeSeriesAnomalyExperiment
setup: 
    ds : ${ds} #-> datetime changes with dataset
    target : ${y} # -> time series column name, changes with dataset
eda:
create_model :
    model : "lstm" # Random Forest
    params:
```
Parameters (add to params): 

- seq_length : int
                Length of the sequence.
- target_var : str
                Name of the target variable.
- layer_no : int (default = 1)
                Number of LSTM layers.
- cell_no : int (default = 1)
                Number of cells in LSTM layer.
- epoch_no : int (default = 100)
                Number of epochs.
- batch_size : int (default = 32)
                Batch size.
- shuffle : bool (default = True)
                Shuffle data.
- patience : int (default = 10)
                Number of epochs with no improvement after which training will be stopped.
- threshold_for_anomaly : float (default = 3)
                Threshold for anomaly detection.

### Prophet

```
##Add pycaret function params here
load_object :
  module: framework.TimeSeriesAnalysis
  name:  TimeSeriesAnomalyExperiment
setup: 
    ds : ${ds} #-> datetime changes with dataset
    target : ${y} # -> time series column name, changes with dataset
eda:
create_model :
    model : "prophet"
    params:
      periods : 0 # Int number of periods to forecast forward. Has not been tested.
      factor : 1.0 # anomaly detection factor

```

### SSA

```
load_object :
  module: framework.TimeSeriesAnalysis
  name:  TimeSeriesAnomalyExperiment
setup: 
    ds : ${ds} #-> datetime changes with dataset
    target : ${y} # -> time series column name, changes with dataset
eda:
create_model :
    model : "ssa"
    params:
      window_size : 10
      lower_frequency_bound : 0.05
      lower_frequency_contribution : 0.975
      threshold : 3
```

- window_size : int or float (default = 10)
    Size of the sliding window (i.e. the size of each word). 
    If float, it represents the percentage of the size of each time series and must be between 0 and 1.
     The window size will be computed as max(2, ceil(window_size * n_timestamps)).

- lower_frequency_bound : float (default = 0.05)
    The boundary of the periodogram to characterize trend, 
    seasonal and residual components. It must be between 0 and 0.5. 
    Ignored if groups is not set to 'auto'.

- lower_frequency_contribution : float (default = 0.975)
    The relative threshold to characterize trend,
     seasonal and residual components by considering the periodogram. 
     It must be between 0 and 1. Ignored if groups is not set to 'auto'.

- threshold : float (default = 3)


## Fault Detection

  Clustering

### PCA

```
load_object :
  module: framework.FaultDetection  
  name: FaultDetectionExperiment
setup:
    # MUST NOT HAVE A data KEYWORD!
    datetime_column : ${ds}
create_model :
    model : "pca" 
    params:
```

Parameters (add to params):
  alpha : float (default = 0.05)   Alpha is the threshold for the hotellings T2 test to determine outliers in the data.
  detect_outliers : list[str] (default = ['ht2', 'spe']) Type of outlier detections. Types: hotellings T2 and SPE (from center distance)
  n_components : float  (default = 0.95) Amount of variance the relevant principal components explain 
  normalize : bool (default = True) Normalize data (currently not relevant)

### DBSCAN

```
load_object : 
  module: framework.FaultDetection  
  name: FaultDetectionExperiment
setup:
    # MUST NOT HAVE A data KEYWORD!
    datetime_column : ${ds}
create_model :
    model : "dbscan"
    params:
      eps : 2 # The maximum distance between two samples 
      #for one to be considered as in the neighborhood of the other.
      #Will define the strictness of anomaly detection
      min_samples :  # Automatically defined as twice the variables.
      # The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
```

### Elliptic Envelope
```
load_object : 
  module: framework.FaultDetection  
  name: FaultDetectionExperiment
setup: 
    # MUST NOT HAVE A data KEYWORD!
    datetime_column :  ${ds}
create_model :
    model : "ee" # EllipticEnvelope
    params:
      contamination : 0.1 # between 0 and 0.5
      # How much anomalies?
```


### Isolation Forest

```
load_object :
  module: framework.FaultDetection  
  name: FaultDetectionExperiment
setup:
    # MUST NOT HAVE A data KEYWORD!
    datetime_column : ${ds}
create_model :
    model : "iforest" 
    params:
      n_estimators : 100 # amount of tress used
      contamination : "auto" # auto
      random_state : 0
```

## Process Mining

### HeuristicsMiner
```
load_object :
  module: framework.ProcessMining
  name: ProcessMiningExperiment
setup:
create_model :
    model : "heuristics"
    params : 
```


### TopKRules

```
load_object :
  module: framework.ProcessMining
  name: ProcessMiningExperiment
setup:
create_model :
    model : "topk"
    params : 
```
### Apriori association rules

```
load_object :
  module: framework.ProcessMining
  name: ProcessMiningExperiment
setup:
create_model :
    model : "apriori"
    params : 
```

### CMSPAM

```
load_object :
  module: framework.ProcessMining
  name: ProcessMiningExperiment
setup:
create_model :
    model : "cmspam" 
    params : 
```