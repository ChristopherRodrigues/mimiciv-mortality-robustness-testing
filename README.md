MIMIC-IV Mortality Robustness Testing
=========================

Python suite to construct in-hospital mortality robustness testing artifacts using the MIMIC-IV clinical database.


## Motivation

The increase in the adoption of EHR systems has opened a variety of opportunities for Big Data and Machine Learning in the
medical domain. Early prediction of patient mortality in the intensive care unit (ICU) can support timely clinical intervention and improve 
patient outcomes. Machine learning models have shown considerable promise for this task; however, their adoption in clinical 
practice depends not only on predictive performance but also on their robustness, calibration, and interpretability under 
realistic clinical conditions.

This work investigates the robustness of both classical machine learning and deep learning models for 
in-hospital mortality prediction using the publicly available MIMIC-IV dataset. Four classical models 
(Logistic Regression, Random Forest, XGBoost, and a Multi-Layer Perceptron) and four deep learning architectures 
(LSTM, LSTM with Target Replication, Channel-wise LSTM, and Channel-wise LSTM with Target Replication) are trained and evaluated.

Models are evaluated using standard discrimination, ranking and calibration metrics in the medical domain, and model explainability is 
achieved using SHapley Additive Explanations(SHAP). To evaluate robustness, clinically plausible distribution shifts are simulated 
to represent changes in patient condition. The resulting changes in predictive performance and calibration are analysed across 
multiple perturbation scenarios and severity levels.

## Citation

MIMIC-IV v3.1:
Johnson, A., Bulgarelli, L., Pollard, T., Gow, B., Moody, B., Horng, S., Celi, L. A., & Mark, R. (2024). MIMIC-IV (version 3.1). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/kpb9-mt58

Original publication:
Johnson, A.E.W., Bulgarelli, L., Shen, L. et al. MIMIC-IV, a freely accessible electronic health record dataset. Sci Data 10, 1 (2023). https://doi.org/10.1038/s41597-022-01899-x

Standard citation for PhysioNet:
Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.

Original benchmark repository based on MIMIC-III:
https://github.com/YerevaNN/mimic3-benchmarks/tree/v1.0.0-alpha?tab=License-1-ov-file

Publication linked to repository:
Hrayr Harutyunyan, Hrant Khachatrian, David C. Kale, Greg Ver Steeg, Aram Galstyan. Multitask learning and benchmarking with clinical time series data https://doi.org/10.48550/arXiv.1703.07771

Ported Version to MIMIC-IV:
https://github.com/vincenzorusso3/mimic-iv-benchmarks/tree/master


## Structure
The content of this repository can be divided into three parts:
* Tools for creating the benchmark dataset.  
* Tools for building the baseline models.
* Robustness testing scripts.

The `mimic4benchmark/scripts` directory contains scripts for creating the benchmark dataset.
The reading tools are in `mimic4benchmark/readers.py`.
All evaluation scripts are stored in the `mimic4benchmark/evaluation` directory.
The `mimic4models` directory contains the baseline models along with some helper tools.
Those tools include discretizers, normalizers and functions for computing metrics.
The `mimic4models/in_hospital_mortality/Testing` directory contains the ICU specific robustness testing scripts.


## Requirements

We do not provide the MIMIC-IV data itself. You must acquire the data yourself from https://physionet.org/content/mimiciv/3.1/. Specifically, download the CSVs.

For classical model baselines [sklearn](http://scikit-learn.org/) is required. LSTM models use [Keras](https://keras.io/).


## Building the benchmark

Here are the required steps as executed in the notebook BenchmarkCreation.ipynb to build the benchmark and carry out the robustness testing. It assumes that you already have the MIMIC-IV dataset (lots of CSV files) on the disk. 

1. Clone the repo.

       git clone https://github.com/ChristopherRodrigues/mimiciv-mortality-robustness-testing/
    
2. The following command takes MIMIC-IV CSVs, generates one directory per `SUBJECT_ID` and writes ICU stay information to `data/{SUBJECT_ID}/stays.csv`, diagnoses to `data/{SUBJECT_ID}/diagnoses.csv`, and events to `data/{SUBJECT_ID}/events.csv`. This step might take around an hour.

       python -m mimic4benchmark.scripts.extract_subjects ./mimic-iv data/root/

3. The following command attempts to fix some issues (ICU stay ID is missing) and removes the events that have missing information. About 80% of events remain after removing all suspicious rows (more information can be found in [`mimic4benchmark/scripts/more_on_validating_events.md`](mimic4benchmark/scripts/more_on_validating_events.md)).

       python -m mimic4benchmark.scripts.validate_events data/root/

4. The next command breaks up per-subject data into separate episodes (pertaining to ICU stays). Time series of events are stored in ```{SUBJECT_ID}/episode{#}_timeseries.csv``` (where # counts distinct episodes) while episode-level information (patient age, gender, ethnicity, height, weight) and outcomes (mortality, length of stay, diagnoses) are stored in ```{SUBJECT_ID}/episode{#}.csv```. This script requires two files, one that maps event ITEMIDs to clinical variables and another that defines valid ranges for clinical variables (for detecting outliers, etc.). **Outlier detection is disabled in the current version**.

       python -m mimic4benchmark.scripts.extract_episodes_from_subjects data/root/

5. The next command splits the whole dataset into training and testing sets.

       python -m mimic4benchmark.scripts.split_train_and_test data/root/
	
6. The following command will generate the in-hospital mortality task-specific dataset.

       python -m mimic4benchmark.scripts.create_in_hospital_mortality data/root/ data/in-hospital-mortality/

7. Commands to generate datasets for other tasks from the original repository:

       python -m mimic4benchmark.scripts.create_decompensation data/root/ data/decompensation/
       python -m mimic4benchmark.scripts.create_length_of_stay data/root/ data/length-of-stay/
       python -m mimic4benchmark.scripts.create_phenotyping data/root/ data/phenotyping/
       python -m mimic4benchmark.scripts.create_multitask data/root/ data/multitask/

After the above commands are done, there will be a directory `data/{task}` for each created benchmark task.
These directories have two sub-directories: `train` and `test`.
Each of them contains bunch of ICU stays and one file with name `listfile.csv`, which lists all samples in that particular set.
Each row of `listfile.csv` has the following form: `icu_stay, period_length, label(s)`.
A row specifies a sample for which the input is the collection of ICU event of `icu_stay` that occurred in the first `period_length` hours of the stay and the target is/are `label(s)`.
In in-hospital mortality prediction task `period_length` is always 48 hours, so it is not listed in corresponding listfiles.

8. Command to create the validation set split:

       python -m mimic4models.split_train_val data/in-hospital-mortality



## Readers
To simplify the reading of benchmark data the original repository provided special classes.
The `mimic4benchmark/readers.py` contains class `Reader` and five other task-specific classes derived from it.
These are designed to simplify reading of benchmark data. The classes require a directory containing ICU stays and a listfile specifying the samples.
Again, we encourage to use these readers to avoid mistakes in the reading step (for example using events that happened after the first `period_length` hours).  
For more information about using readers view the [`mimic4benchmark/more_on_readers.md`](mimic4benchmark/more_on_readers.md) file.


## Evaluation
The original repository provided scripts for evaluating models.
These scripts receive a `csv` file containing the predictions and produce a `json` file containing the scores and confidence intervals for different metrics.
We highly encourage to use these scripts to prevent any mistake in the evaluation step.
For details about the usage of the evaluation scripts view the [`mimic4benchmark/evaluation/README.md`](mimic4benchmark/evaluation/README.md) file.


## Baselines
For in-hospital mortality we provide 8 baselines:  
* Logistic regression
* Random Forest
* XGBoost
* Multi-layer Perceptron
* Standard LSTM
* Standard LSTM + target replication
* Channel-wise LSTM
* Channel-wise LSTM + target replication


Linear models can be found in:

       Logistic Regression: mimic4models/in_hospital_mortality/logistic/main.py
       Random Forest: mimic4models/in_hospital_mortality/RandomForest/main_tuned.py
       XGBoost: mimic4models/in_hospital_mortality/XGBoost/main_tuned.py
       MLP: mimic4models/in_hospital_mortality/MLP/main.py

Deep learning models can be found in:
       LSTM: mimic4models/in_hospital_mortality/main.py
       LSTM + TR: mimic4models/in_hospital_mortality/main_DeepSupervision.py
       Channel-wise LSTM: mimic4models/in_hospital_mortality/main_ChannelwiseLSTM.py
       Channel-wise LSTM + TR: mimic4models/in_hospital_mortality/main_ChannelwiseLSTM_DS.py

Please note that running these models can take hours because of extensive search and feature extraction.


###  Baseline Model Contruction for In-hospital mortality prediction

The following command parameters are used to train the LSTM which gives the best result.
       
       python -um mimic4models.in_hospital_mortality.main --network mimic4models/keras_models/lstm.py --dim 16 --timestep 1.0 --depth 2 --dropout 0.3 --mode train --batch_size 8 --output_dir mimic4models/in_hospital_mortality

When using target-replication:
       
       python -m mimic4models.in_hospital_mortality.main_DeepSupervision --network mimic4models/keras_models/lstm.py --mode train --dim 16 --depth 2 --dropout 0.3 --batch_size 8 --epochs 100 --timestep 1.0 --target_repl_coef 0.5 --output_dir mimic4models/in_hospital_mortality/LSTM_DS

Use the following command to train logistic regression. The best model used L2 regularization with `C=0.001`:
       
       python -um mimic4models.in_hospital_mortality.logistic.main --l2 --C 0.001 --output_dir mimic4models/in_hospital_mortality/logistic








