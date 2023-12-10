import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import SequentialSampler
from transformers import BertConfig, BertForSequenceClassification
from ratsnlp import nlpbook
from ratsnlp.nlpbook.classification import ClassificationExample, ClassificationTrainArguments, ClassificationDataset, ClassificationTask, NsmcCorpus
from transformers import BertTokenizer
import argparse


class DataSet:
    def __init__(self, cmd_args):
        self.data_path = cmd_args.data_path
        self.df = pd.read_excel(self.data_path)
        pass

    def get_examples(self, data_root_path, mode):
        excel_data_df = self.df
        examples = []
        for i in range(len(excel_data_df)):
            text_a, label = excel_data_df.loc[i]['document'], excel_data_df.loc[i]['label']
            examples.append(ClassificationExample(text_a=text_a, text_b=None, label=label))
        return examples

    def get_labels(self):
        excel_data_df = self.df
        intents = excel_data_df['label']
        return list(intents.unique())

    @property
    def num_labels(self):
        return len(self.get_labels())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="example_data/한국어_단발성_대화_데이터셋.xlsx", help='Location where the dataset is located.')
    parser.add_argument('--model_path', type=str, default="models", help='Location where the output model is located.')
    parser.add_argument('--batch_size', type=int, default=32, help='training config, batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='training config, learning rate')
    parser.add_argument('--max_seq_length', type=int, default=128, help='training config, max_seq_length')
    parser.add_argument('--epochs', type=int, default=6, help='training config, epochs')
    parser.add_argument('--seed', type=int, default=7, help='training config, seed')
    cmd_args = parser.parse_args()
    # set args for training
    args = ClassificationTrainArguments(
        pretrained_model_name="beomi/kcbert-base",
        downstream_corpus_name="emote",
        downstream_corpus_root_dir="example_data",
        downstream_model_dir=cmd_args.model_path,
        batch_size=cmd_args.batch_size if torch.cuda.is_available() else 4,
        learning_rate=cmd_args.learning_rate,
        max_seq_length=cmd_args.max_seq_length,
        epochs=cmd_args.epochs,
        tpu_cores=0 if torch.cuda.is_available() else 8,
        seed=cmd_args.seed,
    )
    # dataset preparation
    nlpbook.set_seed(args)
    nlpbook.set_logger(args)
    tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_model_name,
        do_lower_case=False,
    )
    corpus = DataSet(cmd_args=cmd_args)
    train_dataset = ClassificationDataset(
        args=args,
        corpus=corpus,
        mode="train",
        tokenizer=tokenizer,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset, replacement=False),
        collate_fn=nlpbook.data_collator,
        drop_last=False,
        num_workers=args.cpu_workers,
    )
    # set dataset for validation
    val_dataset = ClassificationDataset(
        args=args,
        corpus=corpus,
        tokenizer=tokenizer,
        mode="test",
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(val_dataset),
        collate_fn=nlpbook.data_collator,
        drop_last=False,
        num_workers=args.cpu_workers,
    )
    # load pretrained model
    pretrained_model_config = BertConfig.from_pretrained(
        args.pretrained_model_name,
        num_labels=corpus.num_labels,
    )
    model = BertForSequenceClassification.from_pretrained(
            args.pretrained_model_name,
            config=pretrained_model_config,
    )
    # training
    task = ClassificationTask(model, args)
    trainer = nlpbook.get_trainer(args)
    trainer.fit(
        task,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )