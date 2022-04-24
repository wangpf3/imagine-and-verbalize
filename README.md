# Imagine :thought_balloon: and Verbalize :speaking_head: 

This is a Pytorch implementation for our ICLR 2022 paper: 
Contextualized Scene Imagination for Generative Commonsense Reasoning [[arxiv](https://arxiv.org/abs/2112.06318)].

Code folders: 

(1) `imagination_learning`: Train the imagination moudule.

(2) `verbalization_learning`: Train the verbalization module.


## Dependencies

- Python >= 3.6
- PyTorch == 1.8.0
- transformers == 4.9.1
- Java == 1.8.0
- pycocoevalcap == 1.2

## Learning to Imagine

### 1. Download the data

The data for training the imagination module can be obtained from [link](https://drive.google.com/file/d/1cRzwxyVAGvmS46Q852tvoiSxbCV2MQV1/view?usp=sharing). After downloading, untar the file `data_for_imagination.tar.gz`, and do
```bash
cd imagination_learning 
tar zxvf data_for_imagination.tar.gz
mv data_for_imagination data
```

### 2. Train a imagination module

```bash
./scripts/run.sh
```
After training, the imagination module is saved to `$IMAGINATION_CHECKPOINT_1='./checkpoint'`. Then copy the file `relation_vocab.json` in the folder `./data/skg_multisource` to `$IMAGINATION_CHECKPOINT_1` for later use. 

Alternatively, you can download our well-trained imagination module [checkpoint](https://drive.google.com/file/d/1GQFbirHjASKobcKwxfJGDJcLHXtNDcK4/view?usp=sharing).

### 3. Customize the imagination module for downstream datasets (optional)

We can further fine-tune the imagination module on the silver-standard SKGs of the downstream dataset whose distribution of context and concepts is different from that of the continual-pretraining dataset used in Step 2.
```bash
./scripts/finetuning.sh $IMAGINATION_CHECKPOINT_1
```

The imagination_module is saved to `$IMAGINATION_CHECKPOINT_2='./checkpoint'` (different from `$IMAGINATION_CHECKPOINT_1`). Then again copy the file `relation_vocab.json` in the folder `./data/skg_multisource` to `$IMAGINATION_CHECKPOINT_2` for later use.

### 4. Apply the imagination module to obtain the SKGs for downstream datasets

We have provided the generated SKGs for the downstream datasets in the `verbalization_learning` folder. If you want to use the trained imagination module to annotate your own dataset, do
```bash
./evalulate.sh $IMAGINATION_CHECKPOINT_1 (or 2 if you have done Step 3) $DATASET $SPLIT
```

Note that the generated SKGs from this step is just for learning the verbalization module but not for inference since the imagination module makes use of the ground-truth prefix sentences.

## Learning to Verbalize

### Train & evaluate
```bash
cd verbalization_learning
tar zxvf data.tar.gz
bash ./scripts/run.sh $IMAGINATION_CHECKPOINT_1 (or 2 if you have done Step 3)
```

Key command line arguments to specify the task / imagination module checkpoint / dataset-specific hyper-parameters

```plain
dataset={commongen_inhouse/vist_concept2story/roc_concept2story}
num_epoch={5 (for Concept2Sentence) / 10 (for Concept2Story)}
graph_generator_dir=IMAGINATION_CHECKPOINT
```

## Citation
```
@inproceedings{
wang2022contextualized,
title={Contextualized Scene Imagination for Generative Commonsense Reasoning},
author={PeiFeng Wang and Jonathan Zamora and Junfeng Liu and Filip Ilievski and Muhao Chen and Xiang Ren},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=Oh1r2wApbPv}
}
```
