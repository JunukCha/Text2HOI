# Text2HOI
Official code of Text2HOI: Text-guided 3D Motion Generation for Hand-Object Interaction in CVPR 2024<br>
[Arxiv paper](https://arxiv.org/pdf/2404.00562v2.pdf). / [Project page](https://junukcha.github.io/project/text2hoi/).

## Data
| **Dataset**                 | **Resource**                                                                                                                                 |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| **H2O**                     | [Download](https://h2odataset.ethz.ch/)                                                                                                      |
| **GRAB**                    | [Download](https://grab.is.tue.mpg.de/index.html)                                                                                            |
| **ARCTIC**                  | [Download](https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md#download-full-arctic)                                        |
| **ARCTIC Text Description** | [Download](https://drive.google.com/file/d/18AtaBpQa9Z9pnQTkjObgOHjSSijT59gz/view?usp=sharing)                                               |
| **MANO**                    | [Download](https://mano.is.tue.mpg.de/)                                                                                                      |
| **Object Pickle File**      | [Download](https://drive.google.com/drive/folders/1-bnfGdKPb-iqkjrO7kIJe72BmqUqDzyI?usp=sharing)                                             |

### Preprocessing GRAB object
GRAB objects have so many vertiecs. So we reduce the number of vertices to 4,000.
```
python preprocessing_grab_object.py
```

### Folder Tree for Demo
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
### Folder Tree for Preprocessing
```
data
├─ h2o
│  ├─ object
│  ├─ subject1
│  ├─ subject2
│  └─ subject3
|
├─ grab
│  ├─ contact_meshes
│  ├─ processed_object_meshes
│  ├─ s1
│  ├─ s2
│  ├─ ...
│  └─ s10
│
├─ arctic
│  └─ downloads
│     └─ data
│        ├─ raw_seqs
|        ├─ description   # (Download available in the 'Data - Arctic' section above or in the 'For Future Work' section below.)
│        └─ meta
│           └─ object_vtemplates
│
└─ mano
   └─ mano_v1_2
      └─ models
         ├─ info.txt
         ├─ LICENSE.txt
         ├─ MANO_LEFT.pkl
         └─ MANO_RIGHT.pkl
```

### Data preparation (preprocessing)
```
python preprocessing.py
```
Alternatively, you can download the preprocessed data files, [Download](https://drive.google.com/drive/folders/1vQXrplvS9fukMqHBH7JOne5DoaqTCL5w?usp=sharing).

```
data
├─ h2o
│  ├─ balance_weights.pkl
│  ├─ data.npz
│  ├─ obj.pkl
│  ├─ text.json
│  ├─ text_count.json
│  └─ text_length.json
|
├─ grab
│  ├─ balance_weights.pkl
│  ├─ data.npz
│  ├─ obj.pkl
│  ├─ text.json
│  ├─ text_count.json
│  └─ text_length.json
│
└─ arctic
   ├─ balance_weights.pkl
   ├─ data.npz
   ├─ obj.pkl
   ├─ text.json
   ├─ text_count.json
   └─ text_length.json
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
