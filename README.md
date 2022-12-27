# DSP-final-733

## Discription
We wish to create a English learning platform by implementing the `733右腦學習法`. To achieve such work, we split the whole process into three stages:
1. English to Chinese Translation
2. English to Chinese Homophonics
3. Sequence Generation

## Usage

### Setup
Install python packages and download pretrained model weights to checkpoint/
```bash
bash setup.sh
```

### Run
The interface will prompt you to enter a single English word, it will later return three things:
1. Chinese translation
2. At most 3 Chinese homophonics
3. At most 3 sequences containing the translation and the homophonic
```bash
bash run.sh
```
If you wish to stop the interface, please hold `ctrl+c`
#### Notice: If it is your first time running this script, it will take a few minutes to download the network pretrained weights.

## References

### English to Chinese Translation

### English to Chinese Homophonics

### Sequence Generation
1. [Guyu](https://github.com/lipiji/Guyu)
2. [Ckip Transformers](https://github.com/ckiplab/ckip-transformers)