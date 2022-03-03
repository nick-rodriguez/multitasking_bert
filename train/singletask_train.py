# from sys import argv
import atexit
import gin
import logging
import mlflow
import os
import socket
import time
import torch
from torch.utils.data import DataLoader, RandomSampler


from multi_tasking_transformers.data import NERDataset
from multi_tasking_transformers.heads import SubwordClassificationHead
from multi_tasking_transformers.multitaskers import MultiTaskingBert

''' Changes from bml_20200521_nr_0:
- added data_directory as a parameter to train()
'''

log = logging.getLogger('root')


def setup_logger():
    # import os
    # Set run specific envirorment configurations
    timestamp = time.strftime("run_%Y_%m_%d_%H_%M_%S") + "_{machine}".format(machine=socket.gethostname())

    gin.bind_parameter('multi_tasking_train.model_storage_directory',
                       os.path.join(gin.query_parameter('multi_tasking_train.model_storage_directory'), timestamp))

    os.makedirs(gin.query_parameter('multi_tasking_train.model_storage_directory'), exist_ok=True)

    log.handlers.clear()
    formatter = logging.Formatter('%(message)s')
    fh = logging.FileHandler(
        os.path.join(gin.query_parameter('multi_tasking_train.model_storage_directory'), "log.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.setLevel(logging.INFO)
    log.addHandler(ch)

    # Set global GPU state
    if torch.cuda.is_available() and gin.query_parameter('multi_tasking_train.device') == 'cuda':
        log.info("Using CUDA device:{0}".format(torch.cuda.current_device()))
    else:
        if gin.query_parameter('multi_tasking_train.device') == 'cpu':
            log.info("Utilizing CPU")
        else:
            raise Exception(f"Unrecognized device: {gin.query_parameter('multi_tasking_train.device')}")

    # ML-Flow
    mlflow.set_tracking_uri(f"{gin.query_parameter('multi_tasking_train.ml_flow_directory')}")
    mlflow.set_experiment(f"/{gin.query_parameter('multi_tasking_train.experiment_name')}")

    mlflow.start_run()
    gin_parameters = gin.config._CONFIG.get(list(gin.config._CONFIG.keys())[0])
    mlflow.log_params(gin_parameters)

    # all_params = {x[1].split('.')[-1]: gin.config._CONFIG.get(x) for x in list(gin.config._CONFIG.keys())}
    all_params = gin.config_str()
    with open('config_log.txt', 'w') as f:
        f.write(all_params)
    mlflow.log_artifact("config_log.txt")
    mlflow.log_artifact(__file__)


HEADS = {
    'subword_classification': SubwordClassificationHead
}

DATASETS = {
    'ner': NERDataset,
}


@gin.configurable
def get_huner_tasks(preprocessed_directory, huner_datasets, batch_size=4):
    """
    :param huner_datasets: is a dictionary with tasks/datasets.
    For example:

    huner_datasets = {
    'cellline': ['gellus_cellline'],
    'chemical': ['chebi_chemical'],
    'gene': ['variome_gene'],
    'species': ['variome_species']
    }
    """
    TASKS = {
        'ner': {},
    }
    for task in huner_datasets:
        for dataset in huner_datasets[task]:
            dataset = f'{dataset}_{task}'  # datasets with more than one entity type needs to have a unique dataset name, otherwise it'll be skipped during training.
            if dataset not in TASKS['ner']:
                TASKS['ner'][dataset] = {}
            TASKS['ner'][dataset].update(
                {
                    'head': 'subword_classification',
                    'batch_size': batch_size,
                    'train': f"{preprocessed_directory}/biomedical/huner/{dataset}/train",
                    'test': f"{preprocessed_directory}/biomedical/huner/{dataset}/test",
                    'dev': f"{preprocessed_directory}/biomedical/huner/{dataset}/dev"
                }
            )
    return TASKS


@gin.configurable('multi_tasking_train')
def train(experiment_name,
          ml_flow_directory,
          data_directory,
          ablation_amount,
          transformer_weights,
          use_pretrained_heads,
          model_storage_directory,
          device,
          learning_rate,
          seed,
          repeat_in_epoch_sampling,
          evaluation_interval=1,
          checkpoint_interval=1,
          shuffle=True,
          num_workers=0,
          num_epochs=1,
          transformer_hidden_size=768,
          transformer_dropout_prob=.1,
          eval_on_dev=True,
          write_predictions=False):
    """

    Args:
        eval_on_dev: If True, evaluation is performed on using the development
            dataset during training/fine-tuning.
        write_predictions: If True, model output predictions and probabilities
            will be written to file.
        ablation_amount (float): the proportion of examples to ablate from
            training dataset, leaving 1-ablation_amount remaining.
    """
    log.info(gin.config_str())
    mlflow.set_tag('ablation', ablation_amount)
    torch.random.manual_seed(seed)
    ablation_amount = float(ablation_amount)  # type check this

    heads_and_datasets = prepare_datasets(ablation_amount, data_directory, num_workers, shuffle,
                                          transformer_dropout_prob, transformer_hidden_size,
                                          eval_on_dev=True)  # uses dev set for training eval
    validation_heads_and_dataloaders = prepare_datasets(ablation_amount, data_directory, num_workers, shuffle,
                                                        transformer_dropout_prob, transformer_hidden_size,
                                                        eval_on_dev=False)  # uses test set for training eval

    print(f'Data loaded. Begin Training {len(heads_and_datasets)} Tasks.')
    print(f'ABLATION AMOUNT {ablation_amount} ({(1 - ablation_amount) * 100} percent of data retained')

    # TRAIN MODEL
    heads = [head for head, _, _ in heads_and_datasets]
    mlflow.set_tag('number_tasks', str(len(heads)))

    time_stamp = time.strftime("%Y%m%d%H%M%S")
    for (head, train_loader, test_loader), (_, _, test_loader_val) in zip(heads_and_datasets, validation_heads_and_dataloaders):
        heads = [head]
        mtb = MultiTaskingBert(heads,
                               model_storage_directory=model_storage_directory,
                               transformer_weights=transformer_weights,
                               device=device,
                               learning_rate=learning_rate,
                               use_pretrained_heads=use_pretrained_heads,
                               write_predictions=write_predictions,
                               time_stamp=time_stamp)

        mtb.fit([(head, train_loader, test_loader)],
                num_epochs=num_epochs,
                evaluation_interval=evaluation_interval,
                checkpoint_interval=checkpoint_interval,
                repeat_in_epoch_sampling=repeat_in_epoch_sampling,
                validation_heads_and_dataloaders=[(head, train_loader, test_loader_val)])



def prepare_datasets(ablation_amount, data_directory, num_workers, shuffle, transformer_dropout_prob,
                     transformer_hidden_size, eval_on_dev=None):
    # LOAD DATASETS
    print('Loading Datasets...')
    heads_and_datasets = []
    # from multi_tasking_transformers.data.configured_tasks import get_huner_tasks
    TASKS = get_huner_tasks(data_directory)
    print(TASKS)
    for task in TASKS:
        for dataset in TASKS[task]:
            train_dataset = DATASETS[task](TASKS[task][dataset]['train'])
            if eval_on_dev:
                test_dataset = DATASETS[task](TASKS[task][dataset]['dev'])
            else:
                test_dataset = DATASETS[task](TASKS[task][dataset]['test'])

            # ABLATION SAMPLING
            num_samples = int(len(train_dataset) * (1 - ablation_amount))

            labels = train_dataset.entity_labels if hasattr(train_dataset, 'entity_labels') else None
            if hasattr(train_dataset, 'class_labels'):
                labels = train_dataset.class_labels

            head = HEADS[TASKS[task][dataset]['head']](dataset,
                                                       labels=labels,
                                                       hidden_size=transformer_hidden_size,
                                                       hidden_dropout_prob=transformer_dropout_prob
                                                       )

            if TASKS[task][dataset]['head'] == 'subword_classification':
                if 'evaluate_biluo' in TASKS[task][dataset]:
                    head.config.evaluate_biluo = TASKS[task][dataset]['evaluate_biluo']
                else:
                    head.config.evaluate_biluo = False

            heads_and_datasets.append((head,
                                       DataLoader(train_dataset,
                                                  shuffle=shuffle,
                                                  batch_size=TASKS[task][dataset]['batch_size'],
                                                  num_workers=num_workers
                                                  ),
                                       DataLoader(test_dataset,
                                                  batch_size=TASKS[task][dataset]['batch_size'],
                                                  shuffle=shuffle,
                                                  num_workers=num_workers
                                                  )
                                       )
                                      )
    return heads_and_datasets


if __name__ == "__main__":
    gin.parse_config_file('./config_STL.gin', skip_unknown=True)

    setup_logger()
    train()
    mlflow.end_run()


@atexit.register
def cleanup_on_kill():
    log.info("Training was abruptly killed.")
    mlflow.end_run()
    # shutil.rmtree(gin.query_parameter('multi_tasking_train.model_storage_directory'))
