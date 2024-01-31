## [Revealing the Underlying Patterns: Investigating Dataset Similarity, Performance, and Generalization](https://www.sciencedirect.com/science/article/abs/pii/S0925231223013280)

A model trained on a particular (primary) dataset may or may not perform well on an unseen (secondary) dataset. The proposed distance metrics can give an idea on how 'far' the secondary data is from the primary dataset and whether retraining/finetuning is required. The image-to-dataset distances are also computed as they are useful in selection of the images to be included in the training dataset. Images that are farther from the primary dataset can help in improving the training dataset.

This repository contains the code to compute the dataset-dataset and image-dataset distances.

### Execution
- Put your primary and secondary dataset folders in `./data`.
- Edit the `data_config.yaml` with the names of primary and secondary dataset folders.
- Run the `compute_distance.sh` script to obtain the distances computed using a pretrained EfficientNet model.
#### Custom Model
The distance computation can be performed using other models using the following steps:
- Add the model class in the `models.py`.
- Provide the model name and associated checjpoint in the  `model_config.yaml`.
- Update the `save_outputs` method in `compute_distance.py` to process input images and parse the model outputs.
- Run `compute_distance.sh` with the custom model name as argument.

It should be noted that the final image features obtained from the model should be flat vectors with equal length.

### Reference
If you find the code useful for your research, please cite our paper:
```
@article{ACHARA2024127205,
title = {Revealing the underlying patterns: Investigating dataset similarity, performance, and generalization},
journal = {Neurocomputing},
volume = {573},
pages = {127205},
year = {2024},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2023.127205},
author = {Akshit Achara and Ram Krishna Pandey}
}
```