# PLMABFW: A Deep Learning Framework for Predicting Antibody-Antigen Interactions Using Protein Language Model

> The PLMABFW is a novel framework for antibody prediction designed to address the challenge of predicting antigen-antibody interactions when antigen sequences are highly similar. It leverages protein language pre-trained models to encode antigens and antibodies, capturing richer representations, while its architectural design enables accurate characterization of the subtle differences in antigen sequences recognized by antibodies. Testing has demonstrated that PLMABFW can accurately identify the neutralization relationships between different SARS-CoV-2 variants and antibodies.

## Contents

- [PLMABFW: A Deep Learning Framework for Predicting Antibody-Antigen Interactions Using Protein Language Model](#plmabfw-a-deep-learning-framework-for-predicting-antibody-antigen-interactions-using-protein-language-model)
  - [Contents](#contents)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
    - [Modify the Config File](#modify-the-config-file)
    - [Train your own model](#train-your-own-model)
    - [Folder Structure Details](#folder-structure-details)
  - [Contributing](#contributing)
  - [License](#license)

## Installation

To get started with PLMABFW, follow these steps:

1. Clone this repository to your local machine:
```bash
git clone https://github.com/Chenyb939/PLMTHP
cd PLMABFW
```

2. Creat irtual environment and :
```bash
   conda create -n PLMABFW python=3.9
   conda activate PLMABFW
```

3. install the required dependencies:  
PLMABFM utilizes [ESM](https://github.com/facebookresearch/esm) and [AntiBERTy](https://github.com/jeffreyruffolo/AntiBERTy) for encoding purposes. Before you begin, ensure that both ESM and AntiBERTy are properly installed on your system. If not, you can use the following commands to install them.

3.1 Install [ESM](https://github.com/facebookresearch/esm)
```bash
pip install fair-esm  # latest release, OR:
pip install git+https://github.com/facebookresearch/esm.git
```

3.2 Install [AntiBERTy](https://github.com/jeffreyruffolo/AntiBERTy)
```bash
pip install antiberty  # latest release, OR:
git clone git@github.com:jeffreyruffolo/AntiBERTy.git 
pip install AntiBERTy
```

3.3 Install Other dependencies
```bash
pip install -r requirements.txt  # OR:
pip install Biopython pandas tensorboard matplotlib scikit-learn
```

## Quick Start
After pip install, you can use PLMABFW as follows:

### Modify the Config File
Before run the PLMABFW, please adjust the configuration file according to your specific requirements.

```config
[path]
input_data = ./data  # Path to the input data directory

[parameter]
batch = 512  # Number of samples per batch
epoch = 500  # Total number of training epochs
patience = 30  # How long to wait after last time model improved

[output]
tag = example  # label your output
model_dir = ./model  # Directory for store the trained model
result_dir = ./result  # Directory for store results
log_dir = ./log  # Directory for store training logs

[other]
gpus = 1  # The GPU number used for running PLMABFW
```

### Train your own model
You can use the following command to train you own model:

```bash
python main.py train --config ./config
```

Once the training process is complete, you can utilize the following command to test the model:

```bash
python main.py test --config ./config
```

The procedure to utilize PLMABFW is outlined as follows:

```bash
  usage: python main.py [mode] [--config] [--target_chain]
  positional arguments:
  {train,test}     Mode of operation, either "train" or "test".

optional arguments:
  -h, --help       Show help message and exit
  --config CONFIG  Path to the config file.
  --seed SEED      Random seed.
  ```

### Folder Structure Details
When you run PLMABFW using the default parameters from the config file, the resulting folder structure will be as follows:

   ``` fold
   PLMABFW
   │
   ├── data                 Directory for store the dataset
   ├── model                Directory for store the trained model
   ├── result               Directory for store results
   ├── log                  Directory for store training logs
   │
   ├── main.py              PLMABFW main program
   ├── Model.py             PLMABFW Model
   ├── MyData.py            PLMABFW dataset
   ├── utils.py             PLMABFW dependencies
   ├── config               Configuration file
   ├── LICENSE              LICENSE
   └── requirements.txt     Python environment dependencies

   ```

<!-- - ## Documentation / - [Documentation](#documentation)-->

## Contributing

Contributions of code, issue reports, or improvement suggestions are welcome.

## License

PLMABFW is licensed under the MIT License.