# [Feedback Prize - Predicting Effective Arguements](https://www.kaggle.com/competitions/feedback-prize-effectiveness/overview)

### Top 3% solution (46/1557) based on an ensemble of transformer models.

## Requirements

- Python 3.9.7
- [Pytorch](https://pytorch.org/) 1.10.1
- [Transformers](https://huggingface.co/docs/transformers/index) 4.15.0


## Architecture

- The solution is an ensemble of 2 transformer models - `Deberta V3` and `Deberta V1`.
- Two different types of architectures were used - `Token Classification` and `Text Classification`.
- `Token Classification` models had `max length = 1700` and `Text Classification` has `max length = 768`.
- Training utilised `Half-Precision` and `TPU` was used to speed up training.

## Data Preparation

- For `Token Classification` approach a discourse type e.g `Lead` was prefixed by a start token `[LEAD BEGIN]` and suffixed by a end token `[LEAD END]`. Prediction of scores was made only for the start token. This was repeated for all discourse types.
- For `Text Classification` no special preprocessing was applied.
- `Dynamic Padding` was used to speed up training as well as inference.

## Training
- For training `Token Classification` models use the `token-classification-approach.ipynb` notebook. The data required for training is available on the competition homepage.
- For training `Text Classification` models use the `text-classification-approach.ipynb` notebook. The notebooks requires `TPU` for training and data has to be preprocessed prior to training.

## Inference

- [This](https://www.kaggle.com/code/shreyasadhari123/final-submission) kaggle kernel was used for final submission.
