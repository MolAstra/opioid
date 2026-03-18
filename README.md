# Opioid

## Setups

```bash
mamba create python=3.12 -nopioid_tmp
mamba activate opioid_tmp

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126
git clone https://github.com/zhaisilong/dpnet
cd dpnet

# run ml baselines
pip install -e ".[ml]"
```

## Datasets

The training/valid data was derived from [Machine Learned Classification of Ligand Intrinsic Activities at Human mu-Opioid Receptor](https://pubs.acs.org/doi/10.1021/acschemneuro.4c00212) and [opioids GitHub](https://github.com/JanaShenLab/opioids/blob/main/training_gnn.csv).

### Data Split

```bash
mamba
```

## Experiments

```bash

```

### 数据划分

1. 数据清洗：/data/home/silong/paper/opioid/dataset/0.data_clearn.ipynb
2. 按骨架划分：/data/home/silong/paper/opioid/dataset/1.split_data.ipynb

3. Opioid 数据集构建
4. 训练三个模型 | 集成指标 (x1_logits + x2_logits + x3_logits)/3
   1. XXX
   2. ChemProp
   3. RF
5. 半监督数据集构建
   1. ZINC 抽取相似的数据
   2. 生成模型生成类似的数据
   3. 三个模型打标签并过滤
6. 重新在扩充数据集上训练 | 集成指标 (x1_logits + x2_logits + x3_logits)/3

introduction

methods

data split 

ML SVM RF XGB Graph-based Chemprop Unimol-opioid (ours) -> benchmark

acschemneuro split methods -> 拿原论文结果 VS Unimol-opioid (ours) (extented benchmark)

WebServer 如何构建

Largescaled screened BD (potential mols): query ZINC 按照相似度生成 结合分子() + 给定分子集合先验训练并生成 (DL/REINVENT) -> Predor -> pos/neg 

WebServer Upload DB

## unimol train

- 

## 模型
