from collections import OrderedDict
import torch

from transformers import DistilBertForQuestionAnswering
from transformers import DistilBertTokenizerFast


class SquadModel(object):
    def __init__(self,
                 config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DistilBertForQuestionAnswering(config)

    def normalize_text(self, s):
        """Removing articles and punctuation, and standardizing whitespace are all typical
        text processing steps."""
        import string, re

        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex, " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @torch.no_grad()
    def predict(self, context, question):
        """
        Predicts a single example
        Parameters
        ----------
        context: string context
        question: string question

        Returns
        -------
        dict with question and predicted answer
        """
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        encoding = tokenizer(context, question, truncation=True, padding=True)
        input_id = torch.unsqueeze(torch.tensor(encoding.data['input_ids']), 0)
        attention_mask = torch.unsqueeze(torch.tensor(encoding.data['attention_mask']), 0)
        # predict
        output = self.model(input_id, attention_mask=attention_mask)

        start_logits = torch.argmax(output[0])
        end_logits = torch.argmax(output[1])
        # get predicted answer as string
        ans_tokens = input_id[start_logits: end_logits + 1]
        answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)
        predicted = tokenizer.convert_tokens_to_string(answer_tokens)
        answer = self.normalize_text(predicted)
        return {'question': question, 'answer': answer}

    def remove_module(self, layers_dict):
        """
        Helper function to remove "module" prefix from layers because a model might has trained
        on GPU which creates this prefix
        Parameters
        ----------
        layers_dict

        Returns
        -------

        """
        new_state_dict = OrderedDict()
        for name, v in layers_dict.items():
            if 'module' in name:
                name = name[7:]  # remove `module.`
            new_state_dict[name] = v
        return new_state_dict

    def load_layers(self, checkpoint, model_dict):
        """
        Checks which layers to load
        Parameters
        ----------
        checkpoint: dict loaded from ckpt
        model_dict: model state dict

        Returns
        -------

        """
        new_state_dict = self.remove_module(checkpoint)
        # Keep matching layers
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(pretrained_dict, strict=False)

    def load_model(self, ckpt_path):
        """
        Loads model from ckpt file
        Parameters
        ----------
        ckpt_path: dir to ckpt file

        Returns
        -------

        """
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model_dict = self.model.state_dict()
        self.load_layers(checkpoint['model_weights'], model_dict)

