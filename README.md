# ABKD: Pursuing a Proper Allocation of the Probability Mass in Knowledge Distillation via α-β-Divergence

<!-- <a href="https://arxiv.org/abs/2402.03898"><img src="https://img.shields.io/badge/Paper-arXiv:2402.03898-Green"></a>
<a href=#bibtex><img src="https://img.shields.io/badge/Paper-BibTex-yellow"></a> -->

This repository is the official PyTorch implementation of ABKD (💫ICML 2025 Spotlight💫 **2.6%**). The paper is available [here](https://arxiv.org/abs/2505.04560).

**Paper Title: ABKD: Pursuing a Proper Allocation of the Probability Mass in Knowledge Distillation via α-β-Divergence**

**Authors: Guanghui Wang, [Zhiyong Yang*](https://joshuaas.github.io/), [Zitai Wang](https://wang22ti.com/), [Shi Wang](https://ictkc.github.io/), [Qianqian Xu](https://qianqianxu010.github.io/), [Qingming Huang*](https://people.ucas.ac.cn/~qmhuang)**  

![fig1_compressed](https://github.com/user-attachments/assets/75795455-1d12-45b0-87ee-8363d5cd51be)

## 🚀 Updates
- [x] (24.05.02) 🥳🥳🥳 Our paper has been accepted in **ICML 2025**. We are open to receiving any discussions and will reflect them in the camera-ready version.

## Table of Contents

- [Natural Language Tasks](#Natural-Language-Tasks)
- [Standard Classification Task](#Standard-Classification-Task)
- [Base to New Classification Task](#Base-to-New-Classification-Task)

## Natural Language Tasks

Please make sure you are in the `distillation_llm` directory:

```
cd distillation_llm
```

### Installing Requirements

```
conda create -n distillm python=3.8.10
conda activate distillm
bash install.sh
```

Our code is based on [this commit](https://github.com/huggingface/transformers/commit/85fde09c97213bf7e8625f83096bb2a9e183f987) of HuggingFace Transformers **by following DISTILLM**.


### Data
#### Resources
+ The training/evaluation intruction-response data before processing can be downloaded from MiniLLM [link](https://github.com/microsoft/LMOps/tree/main/minillm).
+ The plain-text corpus $\mathcal{D}_\text{PT}$ can be download from the HugginFace datasets [repository](https://huggingface.co/datasets/openwebtext).


#### Data Processing

SFT data

```
bash scripts/gpt2/tools/process_data_dolly.sh /PATH_TO/LMOps/minillm # Process Dolly Train / Validation Data
bash scripts/openllama2/tools/process_data_dolly.sh /PATH_TO/LMOps/minillm # Process Dolly Train / Validation Data
```

Get plain-text corpus $\mathcal{D}_\text{PT}$:
```bash
python3 tools/get_openwebtext.py
```
This script will replace the continuous `\n` in each document with a special token "<@x(x!>" and write each document in OpenWebText in a line, which is convenient for parallel processing. In `data/openwebtext/data.txt`, we give an example of the resulting format. You can follow this format to prepare other corpus beyond OpenWebText.

Tokenize the data and store them in binary files:
```bash
bash scripts/gpt2/tools/process_data_dolly.sh ${/PATH/TO/ABKD} ${MASTER_PORT} ${GPU_NUM} # Process Dolly Train / Validation Data
bash scripts/gpt2/tools/process_data_pretrain.sh ${/PATH/TO/ABKD} ${MASTER_PORT} ${GPU_NUM} # Process OpenWebText Train / Validation Data

bash scripts/llama/tools/process_data_dolly.sh ${/PATH/TO/ABKD} ${MASTER_PORT} ${GPU_NUM} # Process Dolly Train / Validation Data
bash scripts/llama/tools/process_data_pretrain.sh ${/PATH/TO/ABKD} ${MASTER_PORT} ${GPU_NUM} # Process OpenWebText Corpus Train / Validation Data
```

### Base Pre-trained Models
To run fine-tuning or standard KD baselines, you need to download the model checkpoints from [Huggingface Model Hub] and put them in `checkpoints/`. For example, for gpt2-large, you can download the model from this [link](https://huggingface.co/gpt2-large/tree/main) and put them in `checkpoints/gpt2-large`.

Alternatively, you can also change the `CKPT` variable in each script to the corresponding model name to enable Transformers to download the base models automatically. For example, set `CKPT="gpt2-large"` in `scripts/gpt2/sft/sft_large.sh` causes download of the gpt2-large base model from the HugginFace model hub.

### Train
We provide example commands for GPT-2 models. Similar scripts for model families can be found in `scripts/openllama2`. All our experiments are conducted on 4~8 RTX 3090 24GB GPUs.

#### Baselines
The final checkpoints are selected by the **ROUGE-L** scores.

##### Fine-tune the teacher models
```bash
bash scripts/gpt2/sft/sft_xlarge.sh ${/PATH/TO/ABKD} ${MASTER_PORT} ${GPU_NUM}
```
##### SFT Baselines
```bash
bash scripts/gpt2/sft/sft_base.sh ${/PATH/TO/ABKD} ${MASTER_PORT} ${GPU_NUM}
```

##### KD Baselines
```bash
bash scripts/gpt2/kd/kd_base.sh ${/PATH/TO/ABKD} ${MASTER_PORT} ${GPU_NUM}
```

##### SeqKD Baselines
Generate and process responses with the teacher:
```bash
bash scripts/gpt2/tools/generate_data_seqkd.sh ${/PATH/TO/ABKD} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/tools/process_pseudo_data_seqkd.sh ${/PATH/TO/ABKD} ${MASTER_PORT} ${GPU_NUM}
```
Fine-tune the model with SeqKD:
```bash
bash scripts/gpt2/seqkd/seqkd_base.sh ${/PATH/TO/ABKD} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/seqkd/seqkd_medium.sh ${/PATH/TO/ABKD} ${MASTER_PORT} ${GPU_NUM}
bash scripts/gpt2/seqkd/seqkd_large.sh ${/PATH/TO/ABKD} ${MASTER_PORT} ${GPU_NUM}
```

##### Student Initialization
The final checkpoints are selected by the **validation loss**.
```bash
bash scripts/gpt2/init/init_base.sh ${/PATH/TO/ABKD} ${MASTER_PORT} ${GPU_NUM}
```

##### MiniLLM Baselines

Please refer to the original [MiniLLM repository](https://github.com/microsoft/LMOps/tree/main/minillm) for detailed implementation.

##### GKD Baselines
```bash
bash scripts/gpt2/gkd/gkd_base_xl.sh ${/PATH/TO/ABKD} ${MASTER_PORT} ${GPU_NUM}
```

##### DistiLLM Baselines
```bash
bash scripts/gpt2/distillm/train_0.1B_1.5B.sh ${/PATH/TO/ABKD} ${MASTER_PORT} ${GPU_NUM}
```

#### ABKD (Ours)
```bash
bash scripts/gpt2/ab/train_0.1B_1.5B.sh ${/PATH/TO/ABKD} ${MASTER_PORT} ${GPU_NUM}
```

### Run Evaluation
```bash
bash scripts/gpt2/eval/run_eval.sh ${GPU_IDX} ${/PATH/TO/ABKD}
```


## Standard Classification Task

Please make sure you are in the `standard_classification` directory:

```
cd standard_classification
```

Fetch the pretrained teacher models by:
```
bash scripts/fetch_pretrained_teachers.sh
```
which will download and save the models to `save/models`

### Running

- To run vanilla KD:
```
bash train_kd.sh 
```
- To calibrate the loss function used in vanilla KD and obtain our proposed **ABKD**:
```
bash train_ab.sh 
  1.1 \  # start_alpha_beta: Starting value of (alpha + beta)
  1.1 \  # end_alpha_beta: Ending value of (alpha + beta)
  0.8 \   # start_alpha
  0.8 \   # end_alpha
  resnet56 \   # teacher_model
  resnet20 \   # student_model
  0 \   # gpu_id
  32    # b (weight for distillation loss)
```
  
- To run other baselines (e.g., LSD):
```
bash train_ls.sh 1.0 1.0 0 0 resnet56 resnet20 \
  0 # gpu id
```

- To calibrate the loss function of LSD and obtain ABLSD:
```
bash train_ls.sh 1.2 1.2 0.9 0.9 resnet56 resnet20 0
```

The resulting log file of an experiment recording test accuracy after each epoch is saved in './save'.

## Base to New Classification Task

Please make sure you are in the  `base_to_new_classification` directory:
```
cd base_to_new_classification
```

### Installing Requirements and Training/Downloading Teacher Models

We recommend the reader refer to [PromptKD](https://github.com/zhengli97/PromptKD) for instructions on setting up the environment and teacher models.

### Datasets

Please follow the instructions detailed in [DATASETS.md.](https://github.com/zhengli97/PromptKD/blob/main/docs/DATASETS.md)

### Download Pretrained Models

Download the original ViT-B/16 and ViT-L/14 CLIP model weights from the official OpenAI website. Then place these models in the `./clip` folder.  
[[ViT-B/16 CLIP](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)] [[ViT-L/14 CLIP](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt)]

### Running

#### Vanilla KD
```
# Run training on caltech101 with multiple seeds
for seed in 10 20 30 40 50; do
  bash scripts/promptkd/base2new_train_kd.sh caltech101 $seed  
done
```

#### DKD
```
# Run training on caltech101 with multiple seeds
for seed in 10 20 30 40 50; do
  bash scripts/promptkd/base2new_train_dkd.sh caltech101 $seed  
done
```

#### ABKD
```
for seed in 10 20 30 40 50; do
  bash scripts/promptkd/base2new_train_ab.sh fgvc_aircraft 1.0 1.3 0.5 1.2 2 $seed  100.0
done
```

The output results will be automatically saved at  `output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed_${SEED}`.

## BibTeX
If you find this repo useful for your research, please consider citing our paper:

```
@article{wang2025abkd,
  title={ABKD: Pursuing a Proper Allocation of the Probability Mass in Knowledge Distillation via $$\backslash$alpha $-$$\backslash$beta $-Divergence},
  author={Wang, Guanghui and Yang, Zhiyong and Wang, Zitai and Wang, Shi and Xu, Qianqian and Huang, Qingming},
  journal={arXiv preprint arXiv:2505.04560},
  year={2025}
}
```

## Contact
- Guanghui Wang: guanghui6691@gmail.com

## Acknowledgements

Our code is based on [DISTILLM](https://github.com/jongwooko/distillm), [PromptKD](https://github.com/zhengli97/PromptKD/blob/main/README.md), [TTM](https://github.com/zkxufo/TTM) and [MINILLM](https://github.com/microsoft/LMOps/tree/main/minillm). We thank the authors for releasing their code.
