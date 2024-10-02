# Cheese Classification challenge
Welcome to the Cheese Challenge made as a final challenge for the INF473V Deep Learning in Computer Vision course at École polytechnique.


## Instalation

Cloning the repo:
```
git clone git@github.com:YassineAII/CheeseChallenge.git
cd CheeseCallenge
```
Install dependencies:
```
conda create -n CheeseChallenge python=3.10
conda activate CheeseChallenge
```
### Install requirements:
##### Install Pytorch:
If CUDA>=12.0:
```
pip install torch torchvision
```
If CUDA == 11.8
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
```
Then install the rest of the requirements
```
pip install -r requirements.txt
```

Download the data from kaggle and copy them in the dataset folder
The data should be organized as follow: ```dataset/val```, ```dataset/test```. then the generated train sets will go to ```dataset/train/your_new_train_set```

## Using this codebase
This codebase is centered around 2 components: generating your training data and training your model.
Both rely on a config management library called hydra. It allow you to have modular code where you can easily swap methods, hparams, etc

### Training

To train your model you can run 

```
python train.py
```

This will save a checkpoint in checkpoints with the name of the experiment you have. Careful, if you use the same exp name it will get overwritten

to change experiment name, you can do

```
python train.py experiment_name=new_experiment_name
```

### Generating datasets
You can generate datasets with the following command

```
python generate.py
```

If you want to create a new dataset generator method, write a method that inherits from `data.dataset_generators.base.DatasetGenerator` and create a new config file in `configs/generate/dataset_generator`.
You can then run

```
python generate.py dataset_generator=your_new_generator
```

### VRAM issues
If you have vram issues either use smaller diffusion models (SD 1.5) or try CPU offloading (much slower). For example for sdxl lightning you can do

```
python generate.py image_generator.use_cpu_offload=true
```

