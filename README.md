# CaBiR

This repository holds the code for the paper

**A causality-oriented foundational framework for multimodal fusion analysis**

All the materials released in this library can **ONLY** be used for **RESEARCH** purposes and not for commercial use.

The authors' institution (Biomedical Image and Health Informatics Lab, School of Biomedical Engineering, Shanghai Jiao Tong University) preserves the copyright and all legal rights of these codes.

# Author List

Xinlu Tang, Rui Guo, and Xiaohua Qian

# Abstract

The surging multimodal fusion analysis is increasingly valued with its widespread application across numerous fields, such as biomedicine. It essentially relies on the effective fusion of complementary information for comprehensive understanding and decision-making. However, this process is persistently impeded by bias, a fundamental challenge that remains unresolved. To eliminate biases and integrate complementary causal-inclined features, we propose a foundational multimodal-fusion framework, named Causality-oriented Bias-Revision (CaBiR). This CaBiR systematically collaborates an intervention constraint module based on adaptive learning of latent biases and a counterfactual revision strategy based on environmental factors. Extensive experiments certify the sustained stability and generalizability of CaBiR across various modalities, scopes, and conditions. Its effectiveness in reducing bias and prioritizing clinically significant features were intuitively demonstrated in bias impact analyses and interpretability analyses. Overall, we provide a causality-oriented fusion foundation to improve reliability, trustworthiness, and universality for comprehensive insights, promising to inspire broader ranges of research and applications.

# System requirements and installation

## Hardware requirements

Our `CaBiR` method requires only a standard computer with enough RAM to support the in-memory operations.

The reported results were obtained using a standard NVIDIA GeForce GTX 2080Ti graphics card with an 11-GB memory.

## OS Requirements

Our code has been tested on the following systems:

- Windows 10
- Linux: Ubuntu 18.04

## Dependencies

Our code is mainly based on **Python 3.9.12** and **PyTorch 1.10.1**.

Other useful Python libraries:

- NumPy

- pandas

- scikit-learn

- SciPy

## Environment

The environment can be created via conda as follows:

```shell
conda env create -f ./environment.yaml
conda activate environment
```

# Raw Data

- NACC: https://naccdata.org

- OASIS: https://sites.wustl.edu/oasisbrains/home/oasis-3/

- ADNI & AIBL data: https://ida.loni.usc.edu

- Accesses for these four datasets require registration and requestion, which includes institutional support and justification of data use.

- Cell: https://github.com/daifengwanglab/scMNC

- ABIDE: Preprocessed ABIDE data is fetched using Nilearn toolbox (https://nilearn.github.io/stable/index.html), and the description of preprocessing pipelines are provided by the preprocess connectome projects (PCP) on http://preprocessed-connectomes-project.org/abide/ . The details about the original data are available on http://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html . 

# Data processing

## T1-weighted MRI --> Image

See `preprocessing/NACC/scripts`

## Resting-state functional MRI --> Graph

See `preprocessing/ABIDE/scripts`

## Tabular

See `data/scripts/handle_csv`

# Running

[Taking the NACC-training-AIBL-testing experiment as an example]

```shell
cd ./CaBiR
python main.py
```

Hyperparameters can be found and adjusted within the configuration file located at `./CaBiR/conf/config.json`. This allows users to tailor the framework to their specific datasets.

Upon execution, several output files will be generated: `trend_metrics_test.csv` for recording test metrics, and `log.txt`  for logging textual information, as well as directories named `tb_log` for TensorBorad visualization and `epoch` for storing model checkpoints and intermediate results.

# 

# Contact

For any questions, feel free to contact

> Xinlu Tang : [tangxl20@sjtu.edu.cn](mailto:tangxl20@sjtu.edu.cn)
