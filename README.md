# Disttack
An open-sourced repository for **Disttack**

This repo contains comprehensive [Introduction](#Introduction) of **Disttack**, the first adversarial attack framework tailored for distributed GNN training together with the official [implementation](#implementation) of Disttack </b>.

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
Except datasets mentioned in our work, any other datasets that can be converted to pytorch graph strcure can be easily adopted. For more detile please check README in ```dataset``` folder.

## Implementation
Please use the following steps to execute Disttack.
```
python Disttack/code/main_disttack.py
```
Step 1: Train surrogate model to provide gradient information
Step 2：
