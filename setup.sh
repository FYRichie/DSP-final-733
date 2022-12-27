#! /bin/bash
pip3 install torchvision
pip3 install torchaudio
pip3 install transformers
pip3 install speechbrain --ignore-installed ruamel.yaml
pip3 install gdown

mkdir -p checkpoint
gdown 1nfjQ7R2P4pc2WD3_EAOMsAyajo7gHrWg
mv GPT2_12L_10G.zip checkpoint/
unzip checkpoint/GPT2_12L_10G.zip -d checkpoint
rm -f checkpoint/GPT2_12L_10G.zip