# Datasets

This folder is mainly for storing datasets used for training/validation/testing.

## Practice

1. Separate your codes and datasets. So it is better to soft link your dataset (such as DIV2K, FFHQ, *etc*) here.
    ```bash
    ln -s DATASET_PATH ./
    ```

## Example Datasets

We provide two example datasets for demo.

1. [BSDS100](https://github.com/xinntao/BasicSR-examples/releases/download/0.0.0/BSDS100.zip) for training
1. [Set5](https://github.com/xinntao/BasicSR-examples/releases/download/0.0.0/Set5.zip) for validation

You can easily download them by running the following command in the BasicSR-examples root path:

```bash
python scripts/prepare_example_data.py
```

The example datasets are now in the `datasets/example` folder.
