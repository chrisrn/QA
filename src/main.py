import argparse
import json
import numpy as np
import os
import pandas as pd
from itertools import product
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from data_utils import SquadDataHandler
from train_utils import SquadModelHandler


def main(config_file):
    """

    Parameters
    ----------
    config_file: config.json file

    Returns
    -------

    """
    with open(config_file) as json_file:
        config = json.load(json_file)

    os.environ["CUDA_VISIBLE_DEVICES"] = config['model']['cuda_visible_devices']
    hyper_parameters = config['hyper_parameters']
    param_values = [v for v in hyper_parameters.values()]
    results = []
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    results_dir = f'{config["model"]["results_dir"]}/{timestamp}'
    for i, params in enumerate(product(*param_values)):
        config['hyper_parameters'] = {param: value for param, value in zip(hyper_parameters.keys(), params)}
        config['model']['results_dir'] = f'{results_dir}/exp_{i}'
        os.makedirs(config['model']['results_dir'])
        with open(f'{config["model"]["results_dir"]}/config.json', 'w') as fp:
            json.dump(config, fp)
        # Tensorboard object
        summary_writer = SummaryWriter(f'{results_dir}/exp_{i}/{str(config["hyper_parameters"].values())}')
        # Get data loaders

        data_handler = SquadDataHandler(config['data'],
                                        config['hyper_parameters']['batch_size'])
        train_loader, test_loader, tokenizer = data_handler.get_data()
        # Run training-testing
        model_handler = SquadModelHandler(config,
                                          train_loader,
                                          test_loader,
                                          tokenizer,
                                          summary_writer)

        all_metrics = model_handler.run()
        index_max_f1 = np.argmax(np.array(all_metrics['val_f1']))
        # Results
        all_metrics = {f'hparam/{metric}': value[index_max_f1] for metric, value in all_metrics.items()}
        summary_writer.add_hparams(config["hyper_parameters"], all_metrics)
        config['hyper_parameters'].update(all_metrics)
        df = pd.DataFrame(config['hyper_parameters'], index=[i])
        results.append(df)
        summary_writer.close()

    results_df = pd.concat(results)
    best_exp = results_df[results_df['hparam/val_f1'] == results_df['hparam/val_f1'].max()]
    print('Best model:')
    print(best_exp.T)
    results_csv = os.path.dirname(config['model']['results_dir']) + '/results.csv'
    results_df.to_csv(results_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json',
                        help='Json file with parameters for experiments')

    args = parser.parse_args()
    main(args.config)
