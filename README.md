# Text2HOI
Official code of Text2HOI: Text-guided 3D Motion Generation for Hand-Object Interaction in CVPR 2024<br>
[Arxiv paper](https://arxiv.org/pdf/2404.00562v2.pdf). / [Project page](https://junukcha.github.io/project/text2hoi/).

## Data
### H2O
[Donwload](https://h2odataset.ethz.ch/). 

### GRAB
[Donwload](https://grab.is.tue.mpg.de/index.html). 

### ARCTIC
[Donwload](https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md#download-full-arctic). 

[Text description for Arctic](https://drive.google.com/file/d/18AtaBpQa9Z9pnQTkjObgOHjSSijT59gz/view?usp=sharing).

### MANO
[Donwload](https://mano.is.tue.mpg.de/).

### Object pickle file
[Download](https://drive.google.com/drive/folders/1-bnfGdKPb-iqkjrO7kIJe72BmqUqDzyI?usp=sharing).

### Preprocessing GRAB object
GRAB objects have so many vertiecs. So we reduce the number of vertices to 4,000.
```
python preprocessing_grab_object.py
```

### Folder Tree
```
data
├─ h2o
│  ├─ obj.pkl
│  └─ object
│     ├─ book
│     ├─ cappuccino
│     ├─ chips
│     ├─ cocoa
│     ├─ espresso
│     ├─ lotion
│     ├─ milk
│     └─ spray
|
├─ grab
│  ├─ obj.pkl
│  ├─ processed_object_meshes
│  └─ contact_meshes
│     ├─ airplane.ply
│     ├─ alarmclock.ply
│     ├─ apple.ply
│     ├─ banana.ply
│     ...
|
├─ arctic
│  ├─ obj.pkl
│  └─ downloads/data/meta/object_vtemplates
│     ├─ box
│     ├─ capsulemachine
│     ├─ espressomachine
│     ├─ ketchup
│     ...
│
└─ mano
   └─ mano_v1_2
      └─ models
         ├─ info.txt
         ├─ LICENSE.txt
         ├─ MANO_LEFT.pkl
         └─ MANO_RIGHT.pkl
```

## Checkpoints
[Download](https://drive.google.com/drive/folders/1bfYF94-dVy-mA0n4cIRb_wI4ohPC6KK5?usp=sharing)
```
checkpoints
├─ h2o
├─ grab
└─ arctic
```

## Installation
```
source scripts/install.sh
```

## Demo
```
source scripts/demo.sh
```

## Train
### Data preparation (preprocessing)
```
python preprocessing.py
```

### Train Text2HOI
```
source scripts/train/train_texthom.sh
```

### Train Refiner
```
source scripts/train/train_refiner.sh
```

## For Future Work
[Text description for Arctic](https://drive.google.com/file/d/18AtaBpQa9Z9pnQTkjObgOHjSSijT59gz/view?usp=sharing).
