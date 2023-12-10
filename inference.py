from ratsnlp.nlpbook.classification import ClassificationDeployArguments, ClassificationExample
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

    def get_examples(self):
        excel_data_df = self.df
        examples = []
        for i in range(len(excel_data_df)):  # 70587
            text_a, label = excel_data_df.loc[i]['content'], excel_data_df.loc[i]['label']
            examples.append(ClassificationExample(text_a=text_a, text_b=None, label=label))
        return examples

    def get_labels(self):
        excel_data_df = self.df
        intents = excel_data_df['label']
        return list(intents.unique())

    @property
    def num_labels(self):
        return len(self.get_labels())


class ClassificationModel(DataSet):
    def __init__(self, cmd_args):
        super().__init__(cmd_args)
        self.args = ClassificationDeployArguments(
            pretrained_model_name="beomi/kcbert-base",
            downstream_model_dir=cmd_args.model_path,
            max_seq_length=128,
        )
        self.tokenizer = BertTokenizer.from_pretrained(
            self.args.pretrained_model_name,
            do_lower_case=False,
        )
        self.fine_tuned_model_ckpt = torch.load(
            self.args.downstream_model_checkpoint_fpath,
            map_location=torch.device("cpu")
        )
        self.pretrained_model_config = BertConfig.from_pretrained(
            self.args.pretrained_model_name,
            num_labels=self.fine_tuned_model_ckpt['state_dict']['model.classifier.bias'].shape.numel(),
        )
        self.model = BertForSequenceClassification(self.pretrained_model_config)
        self.activate = False

    def activate_model(self):
        if not self.activate:
            self.activate = True
            self.model.load_state_dict(
                {k.replace("model.", ""): v for k, v in self.fine_tuned_model_ckpt['state_dict'].items()})
            self.model.eval()

    def inference(self, sentence):
        self.activate_model()
        label_names = data_set.get_labels()
        # Preprocess the input
        inputs = self.tokenizer(
            [sentence],
            max_length=self.args.max_seq_length,
            padding="max_length",
            truncation=True,
        )

        # Inference
        with torch.no_grad():
            outputs = self.model(**{k: torch.tensor(v) for k, v in inputs.items()})

        probabilities = softmax(outputs.logits, dim=1)
        probabilities = probabilities.squeeze().tolist()  # If there's only one input sentence

        # Pair each label with its corresponding probability
        label_probabilities = zip(label_names, probabilities)

        # Print the probabilities per label
        for label, prob in label_probabilities:
            print(f"{label}: {prob:.4f}")

        return probabilities


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="example_data/한국어_단발성_대화_데이터셋.xlsx",
                    help='place where the dataset is in.')
parser.add_argument('--model_path', type=str, default="models", help='place where the trained model is saved')
parser.add_argument('--input', type=str, default="", help='input any text')
cmd_args = parser.parse_args()

data_set = DataSet(cmd_args=cmd_args)
my_model = ClassificationModel(data_set)

if __name__ == '__main__':
    my_model.inference(cmd_args.input)
