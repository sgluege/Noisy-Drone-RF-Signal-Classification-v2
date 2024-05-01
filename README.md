# Noisy Drone RF Signal Classification v2 Dataset
Scripts to load and inspect the Noisy Drone RF Signal Classification v2 Dataset

## Dataset
The dataset is available at [kaggle](https://www.kaggle.com/datasets/sgluege/noisy-drone-rf-signal-classification-v2). Download the dataset and place it in a subfolder `dataset/`. 

It comes in the form of 3 filetypes:
- `class_stats.csv`: a single file containing the number of samples per class
- `SNR_stats.csv`:  a single file containing the number of samples per SNR
- `IQdata_sampleX_targetY_snrZ.pt`: sample files that contain the IQ signal of sample X, with target Y at a SNR level of Z

## Load and inspect the dataset
Use the script `load_dataset.py` to load the dataset using a custom torch Dataloader. It also plots a sample of the dataset which should look like this: 
![sample_input_data.jpg](doc/img/sample_input_data.png)

## Cite the data

If you find the data useful for your work please cite our related paper 

Bibtex:
```

```
