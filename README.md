# Noisy Drone RF Signal Classification v2 Dataset
Scripts to load and inspect the Noisy Drone RF Signal Classification v2 Dataset

## Dataset
The dataset is available at [kaggle](https://www.kaggle.com/datasets/sgluege/noisy-drone-rf-signal-classification-v2). Download the dataset and place it in a subfolder `data/`. 

It comes in the form of 3 filetypes:
- `class_stats.csv`: a single file containing the number of samples per class
- `SNR_stats.csv`:  a single file containing the number of samples per SNR
- `IQdata_sampleX_targetY_snrZ.pt`: sample files that contain the IQ signal of sample X, with target Y at SNR level of Z

## Load and inspect the dataset
Use the script `load_dataset.py` to load the dataset using a custom torch Dataloader. It also plots a sample of the dataset which should look like this: 
![sample_input_data.png](doc/img/Noise_snr22.png)
![sample_input_data.png](doc/img/DJI_snr30.png)
![sample_input_data.png](doc/img/DJI_snr6.png)
![sample_input_data.png](doc/img/FutabaT14_snr26.png)
![sample_input_data.png](doc/img/Taranis_snr-4.png)


Note that we plot the power spectrum of the IQ signal, which is the log10 of the absolute value of the FFT of the IQ signal 
```python
power_spec = np.log10(np.sqrt(spectrogram_2d[0,:,:]**2 + spectrogram_2d[1,:,:]**2))
```
## Related Literature

Further information about the data, and how to build a classifier, can be found in our related manuscript. Please cite it if you find it useful.

S. Glüge, M. Nyfeler, A. Aghaebrahimian, N. Ramagnano and C. Schüpbach, "Robust Low-Cost Drone Detection and Classification Using Convolutional Neural Networks in Low SNR Environments," in IEEE Journal of Radio Frequency Identification, vol. 8, pp. 821-830, 2024, doi: [10.1109/JRFID.2024.3487303](https://doi.org/10.1109/JRFID.2024.3487303)

```
@ARTICLE{10737118,
  author={Glüge, Stefan and Nyfeler, Matthias and Aghaebrahimian, Ahmad and Ramagnano, Nicola and Schüpbach, Christof},
  journal={IEEE Journal of Radio Frequency Identification}, 
  title={Robust Low-Cost Drone Detection and Classification Using Convolutional Neural Networks in Low SNR Environments}, 
  year={2024},
  volume={8},
  number={},
  pages={821-830},
  doi={10.1109/JRFID.2024.3487303}
}
```