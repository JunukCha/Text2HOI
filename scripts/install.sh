conda create -n text2hoi python=3.8 -y
conda activate text2hoi

pip install pyyaml==6.0.1

# Install pytorch3d 0.7.2
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y

pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu116_pyt1130/download.html

pip install -r requirements.txt

python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git