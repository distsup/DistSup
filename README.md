# DistSup

A framework for unsupervised and distant-supervised
representation learning with variational autoencoders (VQ-VAE, SOM-VAE, etc),
brought to life during the [2019 Sixth Frederick Jelinek Memorial Summer
Workshop](https://www.clsp.jhu.edu/workshops/19-workshop/).

Papers:
* [Unsupervised Neural Segmentation and Clustering for Unit Discovery in
Sequential Data](https://pgr-workshop.github.io/img/PGR009.pdf) (see [egs/segmental](egs/segmental))
* Robust Training of Vector Quantized Bottleneck Models (under review; see [egs/robustvq](egs/robustvq))
* Neural Variational representation learning for spoken language (under review; TBA)

## Docker
The easiest way to begin training is to build a Docker container
```
docker build --tag distsup:latest .
docker run distsup:latest
```

## Installation
We supply all dependencies in a conda environment. Read [how to set up the
environment](docs/environment.md).

## Training
To get started, train a simple model on MNIST dataset.
See [egs/](egs) for more examples.

Make sure to [load the environment](docs/environment.md). You can train models using the `train.py`/`train.sh` script:
```
./train.sh egs/mnist/yamls/mlp.yaml runs/mnist/mlp
```
`train.sh` is a tiny wrapper around `train.py` which saves the source code and captures
all output to a file, helping to recover the settings of finished experiments.

Some useful command line options are:
-  `-c LAST` resumes training from the last checkpoint.
   (It is safe to always use it, even during the first training run).
- `-m param val` overrides parameters from an experiment `.yaml` file; for instance, to disable
  weight noise:
```
./train.sh -c LAST egs/mnist/yamls/mlp.yaml runs/mnist -m Trainer.weight_noise 0.0
```

For training ScribbleLens models, download the data with
```
bash egs/scribblelens/download_data.sh
```

## Evaluating models
A saved checkpoint can be loaded and its evaluation metrics run with:
```
python evaluate.py runs/mnist/mlp/
```

## Visualizing training
Progress of training is logged to Tensorboard. To view training stats run `tensorboard --logdir PATH_TO_RUN_FOLDER`.

## Contributing
* All contributions are welcomed!
* Neural modules lay out the data as `NWHC` (that is `batch_size x width x heigth
x channel` or equivalently `batch_size x time x frequency x channel`) with
setting width/time being the variable dimension and setting the `H` dimension
to 1 for 1D modules.
* Please use `distsup.utils.safe_squeeze` to remove it for an adidtional
protection (`torch.squeeze` silently doesn't squeeze in such case).
