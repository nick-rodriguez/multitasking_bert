# multitasking_bert

## Installation

In the top level directory `multitasking_bert/`, run:

```bash
pip install .
```

## Usage

### Prepare datasets

For NER tasks, prepare dataset using `NERDataset.create_ner_dataset()`
in `multi_tasking_transformers/data/ner_dataset.py`.

For the training scripts to work also for evaluation, partition the dataset into train, test, dev. The training script
will create dataloaders and the multitasker will evaluation on the dev set during training and the test set at an
interval specified in configuration (see Configuration).

### Configuration

`config.gin` shows an example of configuration of multitasking runs (see also `config_STL.gin` for single task runs).
Parameters include the path to preprocessed data, pre-trained model weights, and other model parameters.

### Training

`multitask_train.py` and `singletask_train.py` show examples of a training run using a biomedical dataset collection
containing separate tasks for entities and datasets.

To run training scripts:

`python multitask_train.py`



