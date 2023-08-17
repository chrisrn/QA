# Question-Answering project
## General description
In this repository there are 2 main sections. The training-evaluation pipeline and the Rest-API application. The training was done on CPU so there are no reliable results based on the hours the model needs to be trained on GPU. The code is capable of running into GPU but it was not tested because of no access to GPU. However, the pipeline runs from training a model to deployment as a rest-api.
## Installation
Create and activate python virtual environment:
```commandline
virtualenv -p python3 venv
source venv/bin/activate
```
If your native python-version is not 3.9, you can install pyenv with the python version
you need and then:
```commandline
pyenv virtualenv 3.9.4 venv
pyenv local venv
```

Then install all the packages:
```commandline
pip install -e .
```

## Training-evaluation pipeline
Squad 2.0 dev dataset was used to train and test [DistilBERT](https://arxiv.org/pdf/1910.01108.pdf) model. It's a model commonly used for NLP tasks and more specifically for QA purposes and it is a lighter version of BERT architecture which was a breakthrough in NLP. The pretrained DistilBERT tokenizer also was used from Hugging Face. To run the pipeline:
```
python main.py
```
The input of this script is `config.json` which is a file with 4 main fields: `hyper_parameters` which contains the parameters of training in lists. For example if we feed `"learning_rate": [5e-5, 2e-5]` and `"weight_decay": [0.01, 0.1]` the pipeline will run 4 experiments, the combination of them. We can play with the architecture of the model by feeding different number of layers. To understand what these values are better:
```
"num_hidden_layers": Number of hidden layers in the Transformer encoder
"num_heads": Number of attention heads for each attention layer in the Transformer encoder
"dim_encoder": Dimensionality of the encoder layers and the pooler layer
"hidden_trans_dim": The size of the "intermediate" (often named feed-forward) layer in the Transformer encoder
```
Then we can view the results of all experiments using `tensorboard --logdir=results`. The 2nd field is `data` and if we feed  `"data_dir": "../dataset/dev-v2.0.json"` the pipeline will automatically download the squad2 dev set. The 3rd field is `callbacks` to activate learning rate schedules. The 4th field is `model` which has to do with model loading-saving. With the `use_pretrained_bert` flag we can use the pretrained bert from Hugging Face and we can also train only some layers by feeding `"train_loaded_weights": false` and `"exclude_layers" : [layer1.weights, layer1.bias etc.],`. We can also train from scratch and continue training using a checkpoint file. 
The evaluation metrics used are EM(Exact Match) which is true if the predicted answer is an exact match with the ground-truth answer and F1 score. The number of shared words between the prediction and the truth is the basis of the F1 score: precision is the ratio of the number of shared words to the total number of words in the prediction, and recall is the ratio of the number of shared words to the total number of words in the ground truth.
## Rest-API
The `app.py` and `inference.py` are responsible to deploy a trained model using Flask. All we have to do is opening `app.py` and pointing the `experiment_path` into the path of our best experiment which is in the `results` dir. For example `experiment_path = 'results/2023_08_16_14_35_46/exp_0'` to choose the checkpoint file and the configuration this model has. Then we can start the server:
```
python app.py
```
and hit a request:
```
curl -X POST http://0.0.0.0:8000/predict -H 'Content-Type: application/json' -d '{ "context": "Immigrants arrived from all over the world to search for gold, especially from Ireland and China. Many Chinese miners worked in Victoria, and their legacy is particularly strong in Bendigo and its environs. Although there was some racism directed at them, there was not the level of anti-Chinese violence that was seen at the Lambing Flat riots in New South Wales. However, there was a riot at Buckland Valley near Bright in 1857. Conditions on the gold fields were cramped and unsanitary; an outbreak of typhoid at Buckland Valley in 1854 killed over 1,000 miners.","question":"Where is the Asian influence strongest in Victoria?" }'
```
