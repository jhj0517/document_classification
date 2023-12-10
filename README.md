# document clasification
This repository is dedicated to fine-tune the text classification models.
It primarily focuses on fine-tuning the pre-trained BERT model, utilizing the [ratsnlp](https://github.com/ratsgo/ratsnlp) package.

# Notebook
If you want to try it in the colab, please refer to [notebook](https://colab.research.google.com/github/jhj0517/document_classification/blob/master/notebook/document_classification.ipynb?authuser=2#scrollTo=8WU3ufLF6kpw) here.

# Dataset
You will need to prepare a dataset comprising two columns: one for the `document` and the other for `label`. An example of the dataset format is as follows:

| label    | document       |
|----------|----------------|
| sadness   | I'm so sad  |
| happiness  | I'm happy!!  |

For a more detailed understanding, please refer to the [example dataset](https://github.com/jhj0517/document_classification/tree/master/example_data).

This repository includes a very small sample example dataset sourced from Kaggle, available here: [Kaggle Dataset](https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text)

Note: The sample in the repo is very small size, it is recommended to prepare a much larger dataset.

