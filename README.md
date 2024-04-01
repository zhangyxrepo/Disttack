# Disttack: Graph Adversarial Attacks Toward Distributed GNN Training 
An open-sourced repository for **Disttack**

This repo contains comprehensive [Introduction](#Introduction) of **Disttack**, the first adversarial attack framework tailored for distributed GNN training together with the official [implementation](#implementation) of Disttack </b>.

![image](overview.pdf)

## Introduction
Adversarial attack in graph domains involves the graph's structure and feature perturbations. Communication complexity among multiple computing nodes amplifies security challenges in distributed scenarios. This is the first work thar bridges adversarial attack and distributed GNN training, filling the blank in the related realm.

## Requirements
* `Python 3.9`
* `cuda 11.7 with driver accordingly`
* `DGL`
* `numpy 1.20.3 or higher`
* `matplotlib 3.7.2` 
* `pytorch 1.1.4`

## Platforms
| Platform | Configuration |
| ---------- |---------- |
| CPU | 32-core Intel Xeon Platinum 8350C CPU (2.60GHz) |
| GPU | NVIDIA A100 SXM 80GB |

## Datasets
Except datasets mentioned in our work, any other datasets that can be converted to pytorch graph strcure can be easily adopted. They are available on  https://drive.google.com/drive/folders/1ycwDpOUHTeS1BRxCF9JYAV_KT73t9iHW. For more detile please check README in ```dataset``` folder.
* `Flickr`
* `Arxiv`
* `RedditSV`
* `Reddit`
------------------------------
* `other torch.geometric datasets are supported`

## Implementation
Please use the following steps to execute Disttack.
```
python Disttack/code/main_disttack.py
```
For baselines we used in our work, you can execute them use the following steps:
```
python Disttack/baselines/python test_dice.py #for example
```
