import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from constants.h2o_constants import h2o_obj_name
from constants.grab_constants import grab_obj_name
from lib.utils.file import read_pkl
from lib.datasets.h2o import (
    SequenceH2O, ContactH2O, MotionH2O
)
from lib.datasets.grab import (
    SequenceGRAB, ContactGRAB, MotionGRAB
)
from lib.datasets.arctic import (
    SequenceARCTIC, ContactARCTIC, MotionARCTIC
)

def get_dataset(dataset_name, dataset_config, test=False):
    assert dataset_name in [
        "Sequenceh2o", "Contacth2o", "Motionh2o", 
        "Sequencegrab", "Contactgrab", "Motiongrab", 
        "Sequencearctic", "Contactarctic", "Motionarctic", 
    ]
    if dataset_name == "Sequenceh2o":
        dataset = SequenceH2O(**dataset_config)
    elif dataset_name == "Contacth2o":
        dataset = ContactH2O(obj_name=h2o_obj_name, **dataset_config)
    elif dataset_name == "Motionh2o":
        dataset = MotionH2O(obj_name=h2o_obj_name, **dataset_config)
        
    elif dataset_name == "Sequencegrab":
        dataset = SequenceGRAB(**dataset_config)
    elif dataset_name == "Contactgrab":
        dataset = ContactGRAB(obj_name=grab_obj_name, **dataset_config)
    elif dataset_name == "Motiongrab":
        dataset = MotionGRAB(obj_name=grab_obj_name, **dataset_config)
    
    elif dataset_name == "Sequencearctic":
        dataset = SequenceARCTIC(**dataset_config)
    elif dataset_name == "Contactarctic":
        dataset = ContactARCTIC(**dataset_config)
    elif dataset_name == "Motionarctic":
        dataset = MotionARCTIC(**dataset_config)
    return dataset

def get_dataloader(dataset_name, config, data_config, test=False):
    dataset = get_dataset(
        dataset_name, 
        data_config, 
        test, 
    )
    if config.balance_weights and not test:
        if "Action" in dataset_name:
            balance_weights_path = data_config.action_balance_weights_path
        else:
            balance_weights_path = data_config.balance_weights_path
        print(f"Balance weights: {balance_weights_path}")
        balance_weights = read_pkl(balance_weights_path)
        sampler = WeightedRandomSampler(torch.FloatTensor(balance_weights), len(dataset))
        dataloader = DataLoader(
            dataset, 
            batch_size=config.batch_size, 
            sampler=sampler,
            num_workers=config.num_workers,
            drop_last=config.drop_last
        )
    else:
        dataloader = DataLoader(
            dataset, 
            batch_size=config.batch_size, 
            shuffle=False,
            num_workers=config.num_workers,
        )
    return dataloader