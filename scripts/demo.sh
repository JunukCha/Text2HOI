python demo/demo.py \
    +test_text="[Place a cappuccino with the right hand.]" \
    +nsamples=4 \
    hydra.output_subdir=null \
    hydra/job_logging=disabled \
    hydra/hydra_logging=disabled

python demo/demo.py dataset=grab \
    +test_text="[Play a flute with both hands.]" \
    +nsamples=4 \
    hydra.output_subdir=null \
    hydra/job_logging=disabled \
    hydra/hydra_logging=disabled

python demo/demo.py dataset=arctic \
    +test_text="[Type a labtop with both hands.]" \
    +nsamples=4 \
    texthom.obj_nfeats=10 \
    hydra.output_subdir=null \
    hydra/job_logging=disabled \
    hydra/hydra_logging=disabled