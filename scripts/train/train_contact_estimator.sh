python train/train_contact_estimator.py \
       dataset=h2o \
       dataset.augm=True \
       hydra.output_subdir=null \
       hydra/job_logging=disabled \
       hydra/hydra_logging=disabled

# python train/train_contact_estimator.py \
#        dataset=grab \
#        dataset.augm=True \
#        hydra.output_subdir=null \
#        hydra/job_logging=disabled \
#        hydra/hydra_logging=disabled
       
# python train/train_contact_estimator.py \
#        dataset=arctic \
#        dataset.augm=True \
#        hydra.output_subdir=null \
#        hydra/job_logging=disabled \
#        hydra/hydra_logging=disabled