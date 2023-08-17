import json
import os.path

import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.utils import download_url
from transformers import DistilBertTokenizerFast


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


class SquadDataHandler:
    def __init__(self, data_params, batch_size):
        """

        Parameters
        ----------
        data_params: from config file
        batch_size: from config file
        """
        self.data_dir = data_params['data_dir']
        self.batch_size = batch_size

    def add_end_idx(self, answers, contexts):
        """
        Getting the character position at which the answer ends in the passage (we are given the starting position).
        Sometimes SQuAD answers are off by one or two characters, so we will also adjust for that
        Parameters
        ----------
        answers: strings from squad
        contexts: strings from squad

        Returns
        -------
        answers
        contexts

        """
        for answer, context in zip(answers, contexts):
            gold_text = answer['text']
            start_idx = answer['answer_start']
            end_idx = start_idx + len(gold_text)

            # Fixing squad answers that are off by 1 or 2 chars
            if context[start_idx:end_idx] == gold_text:
                answer['answer_end'] = end_idx
            elif context[start_idx - 1:end_idx - 1] == gold_text:
                answer['answer_start'] = start_idx - 1
                answer['answer_end'] = end_idx - 1
            elif context[start_idx - 2:end_idx - 2] == gold_text:
                answer['answer_start'] = start_idx - 2
                answer['answer_end'] = end_idx - 2

        return answers, contexts

    def add_token_positions(self, tokenizer, encodings, answers):
        """
        Converting our character start/end positions to token start/end positions
        Parameters
        ----------
        tokenizer: bert fast tokenizer
        encodings: contexts and questions after tokenization
        answers: string answers

        Returns
        -------
        encodings: contexts and questions after tokenization with token start/end positions

        """
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
            end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
            # if None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = tokenizer.model_max_length
        encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
        return encodings

    def get_data(self):
        path = Path(self.data_dir)
        if not os.path.exists(path):
            data_name = os.path.basename(path)
            dataset_url =f'https://rajpurkar.github.io/SQuAD-explorer/dataset/{data_name}'
            os.makedirs('../dataset')
            download_url(dataset_url, '../dataset/')

        with open(path, 'rb') as f:
            squad_dict = json.load(f)

        contexts = []
        questions = []
        answers = []
        for group in squad_dict['data']:
            for passage in group['paragraphs']:
                context = passage['context']
                for qa in passage['qas']:
                    question = qa['question']
                    for answer in qa['answers']:
                        contexts.append(context)
                        questions.append(question)
                        answers.append(answer)

        split = int(0.75*len(contexts))
        train_contexts, train_questions, train_answers = contexts[:split], questions[:split], answers[:split]
        val_contexts, val_questions, val_answers = contexts[split:], questions[split:], answers[split:]
        print(f'Train samples: {len(train_contexts)}')
        print(f'Validation samples: {len(val_contexts)}')

        train_answers, train_contexts = self.add_end_idx(train_answers, train_contexts)
        val_answers, val_contexts = self.add_end_idx(val_answers, val_contexts)

        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

        train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
        val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

        train_encodings = self.add_token_positions(tokenizer, train_encodings, train_answers)
        val_encodings = self.add_token_positions(tokenizer, val_encodings, val_answers)

        # Squad dataset
        train_dataset = SquadDataset(train_encodings)
        val_dataset = SquadDataset(val_encodings)

        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

        return train_loader, val_loader, tokenizer
