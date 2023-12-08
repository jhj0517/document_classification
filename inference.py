from ratsnlp.nlpbook.classification import ClassificationDeployArguments
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
import torch
from torch.nn.functional import softmax
import pandas as pd
import argparse


class DataSet:
    def __init__(self, cmd_args):
        self.data_path = cmd_args.data_path
        self.df = pd.read_excel(self.data_path)
        pass

    def get_labels(self):
        excel_data_df = self.df
        intents = excel_data_df['label']
        return list(intents.unique())

    @property
    def num_labels(self):
        return len(self.get_labels())


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="example_data\\tweet_emotions.xlsx", help='place where the dataset is in.')
parser.add_argument('--model_path', type=str, default="models", help='place where the trained model is saved')
cmd_args = parser.parse_args()

data_set = DataSet(cmd_args=cmd_args)

args = ClassificationDeployArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_model_dir=cmd_args.model_path,
    max_seq_length=128,
)
tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_name,
    do_lower_case=False,
)
fine_tuned_model_ckpt = torch.load(
    args.downstream_model_checkpoint_fpath,
    map_location=torch.device("cpu")
)

pretrained_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
    num_labels=fine_tuned_model_ckpt['state_dict']['model.classifier.bias'].shape.numel(),
)
model = BertForSequenceClassification(pretrained_model_config)
model.load_state_dict({k.replace("model.", ""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()})
model.eval()


def inference_fn(sentence):
    label_names = data_set.get_labels()
    # Preprocess the input
    inputs = tokenizer(
        [sentence],
        max_length=args.max_seq_length,
        padding="max_length",
        truncation=True,
    )

    # Inference
    with torch.no_grad():
        outputs = model(**{k: torch.tensor(v) for k, v in inputs.items()})

    probabilities = softmax(outputs.logits, dim=1)
    probabilities = probabilities.squeeze().tolist()  # If there's only one input sentence

    # Pair each label with its corresponding probability
    label_probabilities = zip(label_names, probabilities)

    # Print the probabilities per label
    for label, prob in label_probabilities:
        print(f"{label}: {prob:.4f}")

    return probabilities