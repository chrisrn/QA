import os
from collections import OrderedDict
import numpy as np
import torch

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, OneCycleLR
import torch.nn as nn

from transformers import DistilBertForQuestionAnswering, DistilBertConfig


class SquadModelHandler(object):
    def __init__(self,
                 config,
                 train_loader,
                 test_loader,
                 tokenizer,
                 summary_writer):
        """

        Parameters
        ----------
        config: config from json file
        train_loader: data loader with train samples
        test_loader: data loader with test samples
        tokenizer: bert fast tokenizer
        summary_writer: obj to write in tensorboard
        """
        self.config = config
        train_params = config['hyper_parameters']
        self.epochs = train_params['epochs']
        self.batch_size = train_params['batch_size']
        self.learning_rate = train_params['learning_rate']
        self.optimizer_name = train_params['optimizer']
        self.grad_clip = train_params['grad_clip']
        self.weight_decay = train_params['weight_decay']
        self.momentum = train_params['momentum']

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.summary_writer = summary_writer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Callbacks dict
        self.callbacks = config['callbacks']

        # Fine-tuning file to continue training
        model_params = config['model']
        self.use_pretrained_bert = model_params['use_pretrained_bert']
        self.finetuning_file = model_params['fine-tuning-file']
        self.exclude_layers = model_params['exclude_layers']
        self.load_weights_only = model_params['load_weights_only']
        self.train_loaded_weights = model_params['train_loaded_weights']
        self.epochs_per_save = model_params['epochs_per_save']
        self.results_dir = model_params['results_dir']

        # Steps to print results
        self.steps_per_log = model_params['steps_per_log']

        # Values of each module
        if self.use_pretrained_bert:
            self.model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
        else:
            config = DistilBertConfig(max_position_embeddings=self.config['data']['max_seq_length'],
                                      n_layers=train_params['num_hidden_layers'],
                                      n_heads=train_params['num_heads'],
                                      dim=train_params['dim_encoder'],
                                      hidden_dim=train_params['hidden_trans_dim'],
                                      dropout=train_params['dropout'],
                                      activation=train_params['activation'])
            self.model = DistilBertForQuestionAnswering(config)

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.start_epoch = 0
        # Evaluation metrics
        self.mean_em = []
        self.mean_f1 = []
        self.all_metrics = {'train_loss': [],
                            'val_loss': [],
                            'val_em': [],
                            'val_f1': []}

    def get_optimizer(self):
        """
        Gets optimizer obj
        Returns
        -------

        """
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                    weight_decay=self.weight_decay)
        elif self.optimizer_name == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        elif self.optimizer_name == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                   momentum=self.momentum,
                                   weight_decay=self.weight_decay)
        elif self.optimizer_name == 'adadelta':
            return torch.optim.Adadelta(self.model.parameters(), lr=self.learning_rate,
                                        weight_decay=self.weight_decay)
        elif self.optimizer_name == 'adagrad':
            return torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate,
                                       weight_decay=self.weight_decay)
        elif self.optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate,
                                       momentum=self.momentum,
                                       weight_decay=self.weight_decay)
        else:
            raise ValueError('Supported optimizers: adam, adamw, sgd, adadelta, adagrad, rmsprop')

    def normalize_text(self, s):
        """Removing articles and punctuation, and standardizing whitespace are all typical text
        processing steps."""
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

    def compute_f1(self, prediction, truth):
        """
        Computes the F-1 score of a prediction, based on the tokens
        Parameters
        ----------
        prediction: predicted answer
        truth: ground truth

        Returns
        -------
        the f-1 score of the prediction
        """
        pred_tokens = self.normalize_text(prediction).split()
        truth_tokens = self.normalize_text(truth).split()

        # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)

        # get tokens that are in the prediction and gt
        common_tokens = set(pred_tokens) & set(truth_tokens)

        # if there are no common tokens then f1 = 0
        if len(common_tokens) == 0:
            return 0

        # calculate precision and recall
        prec = len(common_tokens) / len(pred_tokens)
        rec = len(common_tokens) / len(truth_tokens)

        return 2 * (prec * rec) / (prec + rec)

    def compute_exact_match(self, prediction, truth):
        """
        Returns true if the predicted is an exact match, else False
        Parameters
        ----------
        prediction: predicted answer
        truth: ground truth

        Returns
        -------
        1 if exact match, else 0
        """
        return int(self.normalize_text(prediction) == self.normalize_text(truth))

    def cross_entropy_loss(self, outputs, start_positions, end_positions):
        """
        Computes cross entropy loss between predicted and ground-truth positions
        Parameters
        ----------
        outputs: model outputs
        start_positions: start truth index of answer
        end_positions: end truth index of answer

        Returns
        -------
        cross entropy loss
        """
        start_logits = outputs[1]
        end_logits = outputs[2]
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        return total_loss

    def train_step(self, batch):
        """
        Runs a train step on a single batch
        Parameters
        ----------
        batch: data loader batch

        Returns
        -------
        batch cross entropy loss
        """
        # move-tensors-to-device (CPU or GPU)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        start_positions = batch['start_positions'].to(self.device)
        end_positions = batch['end_positions'].to(self.device)

        # clear-the-gradients-of-all-optimized-variables
        self.optimizer.zero_grad()

        # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
        outputs = self.model(input_ids, attention_mask=attention_mask,
                             start_positions=start_positions,
                             end_positions=end_positions)
        total_loss = self.cross_entropy_loss(outputs, start_positions, end_positions)
        total_loss.backward()

        # Gradient clipping
        if self.grad_clip:
            nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip)

        # perform-a-single-optimization-step (parameter-update)
        self.optimizer.step()

        return total_loss.item()


    @torch.no_grad()
    def evaluation_step(self, batch):
        """
        Evaluates a single batch with em and f1 score
        Parameters
        ----------
        batch: data loader batch

        Returns
        -------
        batch cross entropy loss
        """
        # get test data and transfer to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        start_positions = batch['start_positions'].to(self.device)
        end_positions = batch['end_positions'].to(self.device)

        # predict
        outputs = self.model(input_ids, attention_mask=attention_mask,
                             start_positions=start_positions, end_positions=end_positions)

        total_loss = self.cross_entropy_loss(outputs, start_positions, end_positions)

        # iterate over samples, calculate EM and F-1 for all
        for input_i, s, e, trues, truee in zip(input_ids, outputs[1], outputs[2], start_positions, end_positions):
            # get predicted start and end logits (maximum score)
            start_logits = torch.argmax(s)
            end_logits = torch.argmax(e)

            # get predicted answer as string
            ans_tokens = input_i[start_logits: end_logits + 1]
            answer_tokens = self.tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)
            predicted = self.tokenizer.convert_tokens_to_string(answer_tokens)

            # get ground truth as string
            ans_tokens = input_i[trues: truee + 1]
            answer_tokens = self.tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)
            true = self.tokenizer.convert_tokens_to_string(answer_tokens)

            # compute score
            em_score = self.compute_exact_match(predicted, true)
            f1_score = self.compute_f1(predicted, true)
            self.mean_em.append(em_score)
            self.mean_f1.append(f1_score)
        return total_loss.item()

    def run_epoch(self, scheduler_oc, epoch, mode='train'):
        """
        Runs a single epoch
        Parameters
        ----------
        scheduler_oc: Lr plan
        epoch: epoch num
        mode: train or validation mode

        Returns
        -------

        """

        if mode == 'train':
            self.model.train()
            data_loader = self.train_loader
        else:
            self.model.eval()
            data_loader = self.test_loader

        step = 0
        print('{} mode'.format(mode))
        epoch_train_loss = []
        epoch_val_loss = []
        for batch in data_loader:
            if mode == 'train':
                train_loss = self.train_step(batch)
                epoch_train_loss.append(train_loss)
            else:
                val_loss = self.evaluation_step(batch)
                epoch_val_loss.append(val_loss)

            if self.callbacks['one_cycle_lr'] and epoch >= self.callbacks['epoch_begin']:
                scheduler_oc.step()

            step += 1
            if step % self.steps_per_log == 0:
                print(f'Step {step}/{len(data_loader)} \t')

        if mode == 'train':
            avg_train_loss = np.mean(epoch_train_loss)
            print(f"Epoch mean train loss: {avg_train_loss}")
            self.all_metrics['train_loss'].append(avg_train_loss)
        if mode == 'test':
            avg_em = np.mean(self.mean_em)
            avg_f1 = np.mean(self.mean_f1)
            avg_val_loss = np.mean(epoch_val_loss)
            print(f"Epoch mean validation loss: {avg_val_loss}")
            print(f"Epoch mean EM: {avg_em}", )
            print(f"Epoch mean F-1: {avg_f1}", )
            self.all_metrics['val_loss'].append(avg_val_loss)
            self.all_metrics['val_em'].append(avg_em)
            self.all_metrics['val_f1'].append(avg_f1)

    def get_lr(self, optimizer):
        """
        Gets lr from optimizer state
        Parameters
        ----------
        optimizer: optimizer obj

        Returns
        -------

        """
        for param_group in optimizer.param_groups:
            return param_group['lr']

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

    def variables_to_train(self):
        """
        Decides which variables are trainable
        Returns
        -------

        """
        if not self.train_loaded_weights:
            for name, param in self.model.named_parameters():
                if name in self.exclude_layers:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

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
        # Exclude layers
        for layer in self.exclude_layers:
            if layer in pretrained_dict.keys():
                print(f'Layer {layer} excluded')
                del pretrained_dict[layer]
            else:
                raise ValueError(f'Layer: {layer} to exclude is not in model')

        model_dict.update(pretrained_dict)
        self.model.load_state_dict(pretrained_dict, strict=False)
        self.variables_to_train()

    def check_ckpt_params(self, checkpoint):
        """
        Checks whether to load only the weights or load the epoch num, the optimizer state
        and the metrics from last epoch
        Parameters
        ----------
        checkpoint: checkpoint dict loaded from ckpt file

        Returns
        -------

        """
        if self.load_weights_only:
            self.optimizer = self.get_optimizer()
            self.start_epoch = 0
        else:
            self.start_epoch = checkpoint['epoch']
            self.optimizer = checkpoint['optimizer']
            if 'train_metrics' in checkpoint:
                self.train_metrics = checkpoint['train_metrics']
            if 'test_metrics' in checkpoint:
                self.test_metrics = checkpoint['test_metrics']

    def check_pretrained(self):
        """
        Check whether to load ckpt file, use pretrained bert model from Hugging Face or
        train from scratch
        Returns
        -------

        """
        if self.finetuning_file:
            print(f'Pre-trained model loaded from: {self.finetuning_file}')
            checkpoint = torch.load(self.finetuning_file, map_location='cpu')
            model_dict = self.model.state_dict()
            self.load_layers(checkpoint['model_weights'], model_dict)
            self.check_ckpt_params(checkpoint)
        elif self.use_pretrained_bert:
            self.variables_to_train()
            self.optimizer = self.get_optimizer()
            self.start_epoch = 0
        else:
            self.optimizer = self.get_optimizer()
            self.start_epoch = 0

    def save_model(self, epoch, results_dir, best=False):
        """
        Save model to ckpt file
        Parameters
        ----------
        epoch: epoch number
        results_dir: directory with hp results
        best: flag to save best model

        Returns
        -------

        """
        model_dir = f'{results_dir}/model'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if best:
            ckpt_file = f'{model_dir}/best_model.pth.tar'
            print('Best model saved\n')
        else:
            ckpt_file = f'{model_dir}/model_{epoch}.pth.tar'
            print('Model saved\n')
        torch.save({'epoch': epoch,
                    'model_weights': self.model.state_dict(),
                    'optimizer': self.optimizer,
                    'all_metrics': self.all_metrics}, ckpt_file)

    def parameter_names(self):
        """
        Logs parameter names to help us exclude layers by name if we want
        Returns
        -------

        """
        for name, _ in self.model.named_parameters():
            print(name)

    def run(self):
        """
        Runs hyper-parameter single experiment
        Returns
        -------

        """
        self.parameter_names()
        self.check_pretrained()
        print(f'Model parameters: {sum(p.numel() for p in self.model.parameters())}')
        print(f'Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model)
        scheduler = self.get_callbacks()
        max_f1 = 0.0
        epoch_max_f1 = self.start_epoch
        for epoch in range(self.start_epoch + 1, self.epochs + self.start_epoch + 1):
            lr = self.get_lr(self.optimizer)
            print('Epoch: {} \t Learning rate {}'.format(epoch, lr))
            self.run_epoch(scheduler, epoch)
            self.run_epoch(scheduler, epoch, mode='test')

            if epoch >= self.callbacks['epoch_begin']:
                self.activate_callbacks(scheduler, self.all_metrics['val_loss'][epoch - 1])

            if self.epochs_per_save and (epoch % self.epochs_per_save) == 0:
                self.save_model(epoch, self.results_dir)

            if self.all_metrics['val_f1'][epoch - 1] > max_f1:
                self.save_model(epoch, self.results_dir, best=True)
                max_f1 = self.all_metrics['val_f1'][epoch - 1]
                epoch_max_f1 = epoch

        print(f'Best f1 score = {max_f1}, epoch = {epoch_max_f1}')
        # Add metrics to tensorboard
        for metric in self.all_metrics.keys():
            for i, (metric_value) in enumerate(self.all_metrics[metric]):
                self.summary_writer.add_scalar(f'Per_epoch/{metric}', metric_value, i)

        return self.all_metrics

    def get_callbacks(self):
        """
        Getting lr plans
        Returns
        -------

        """
        if self.callbacks['exponential_lr']:
            return StepLR(optimizer=self.optimizer,
                          step_size=self.callbacks['num_epochs_per_decay'],
                          gamma=self.callbacks['lr_decay_factor'])
        elif self.callbacks['plateau_learning_rate']:
            return ReduceLROnPlateau(self.optimizer,
                                     factor=self.callbacks['plateau_decay'],
                                     patience=self.callbacks['plateau_patience_epochs'],
                                     min_lr=self.callbacks['plateau_min_lr'])
        elif self.callbacks['one_cycle_lr']:
            return OneCycleLR(self.optimizer, self.callbacks['max_one_cycle_lr'],
                              epochs=self.epochs, steps_per_epoch=len(self.train_loader))
        else:
            return None

    def activate_callbacks(self, scheduler, epoch_test_loss):
        """

        Parameters
        ----------
        scheduler: lr scheduler obj
        epoch_test_loss: validation loss

        Returns
        -------

        """
        if self.callbacks['exponential_lr']:
            scheduler.step()
        elif self.callbacks['plateau_learning_rate']:
            scheduler.step(epoch_test_loss)
