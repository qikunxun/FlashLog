# Code for the paper "Towards Scalable and Fast End-to-end Logical Rule Learning"

FlashLog supports scalable rule learning, extraction, and GPU inference for [NeuralLP](https://github.com/fanyangxyz/Neural-LP), [DRUM](https://github.com/alisadeghian/DRUM), and [smDRUM/mmDRUM](https://github.com/xiaxia-wang/FaithfulRE).

## Installation

Tested with Python 3.12, PyTorch 2.3, and CUDA 12.1. Requires Linux, a CUDA toolkit with `nvcc`, and a C++17 compiler with OpenMP. Extensions compile automatically on first use.

```bash
cd FlashLog
pip install -r requirements.txt
```

Run the commands below from this directory.

## Datasets

| Dataset | Download |
| --- | --- |
| Family | [Neural-LP](https://github.com/fanyangxyz/Neural-LP) |
| UMLS, Kinship, WN18RR, FB15k-237 | [RNNLogic](https://github.com/DeepGraphLearning/RNNLogic) |
| YAGO3-10 | [Hugging Face](https://huggingface.co/datasets/VLyb/YAGO3-10) |
| Inductive FB15k-237 and NELL-995 | [GraIL](https://github.com/kkteru/grail) |
| Wikidata5M | [Download](https://web.informatik.uni-mannheim.de/pi1/kge-datasets/wikidata5m.tar.gz) |
| Freebase | [Download](http://web.informatik.uni-mannheim.de/pi1/kge-datasets/freebase.tar.gz) |

Prepare each dataset under `data/<dataset>/` with:

- `entities.dict`, `relations.dict`: tab-separated `id`, `name` columns.
- `train_id.txt`, `valid_id.txt`, `test_id.txt`: tab-separated `head_id`, `relation_id`, `tail_id` columns, without headers.

Use contiguous zero-based IDs and entity dictionary rows in ID order. Include only base relations; inverse relations are added automatically.

## Training and evaluation

For Family, UMLS, and Kinship:

```bash
python src/main.py \
  --data_dir data/family --exps_dir runs --exp_name family_drum \
  --model_name DRUM --use_gpu --gpu_id 0 \
  --L 3 --N 3 --emb_size 128 --batch_size 32 \
  --learning_rate 0.001 --dropout 0 --seed 1234 \
  --max_epoch 10 --early_stop --raw --do_train --do_test
```

Change `--model_name` to `NeuralLP`, `smDRUM`, or `mmDRUM` for other models. `--L` sets the rule length; `--N` sets the number of DRUM rule channels. `--c1` sets the entity budget; `--c2` sets the total fact budget. For WN18RR, FB15k-237, and YAGO3-10, change the dataset path and add `--use_topk --c1 1000 --c2 400`.

With `--use_topk`, training uses weighted entity sampling and capacity-aware relation-group allocation; evaluation uses deterministic top-k.

For Freebase, use sparse execution and a 20,000-second training budget:

```bash
python src/main.py \
  --data_dir data/freebase --exps_dir runs --exp_name freebase_drum \
  --model_name DRUM --use_gpu --gpu_id 0 --sparse \
  --use_topk --c1 1000 --c2 400 \
  --L 3 --N 3 --emb_size 1024 --batch_size 256 \
  --learning_rate 0.001 --dropout 0 --seed 1234 \
  --max_epoch 100000 --max_time 20000 \
  --time_checkpoints 60 300 20000 --test_checkpoints 20000 \
  --do_train --do_test
```

For Wikidata5M, change the dataset path; keep `--batch_size 256`.

Checkpoints are saved under `runs/<exp_name>/`. To evaluate an existing model, reuse its settings with `--do_test` only; use `--ckpt_step 20000` for the timed checkpoint. Test evaluation reports filtered MRR and Hit@1/3/10. `--raw` applies only to validation ranking for early stopping.

For **inductive evaluation**, train on the base split (e.g., `fb237_v1`), then evaluate the same checkpoint on `fb237_v1_ind` with `--do_test` only. Its `train_id.txt` supplies the inference graph. Both splits must share relation IDs.

## Rule extraction and evaluation

Match the checkpoint's `--method`, `--L`, `--N`, `--emb-size`, and `--tau-1`/`--tau-2`. Rule length and channel count default to 3.

Prepare the training graph:

```bash
python rules/prepare.py runs/family_index \
  --data-dir data/family --derive-heads-from-train
```

Extract rules:

```bash
CUDA_VISIBLE_DEVICES=0 python rules/run.py extract --method DRUM \
  runs/family_rules \
  --data-dir data/family --checkpoint runs/family_drum/model.pt --emb-size 128 \
  --csr-cache runs/family_index/train_compact_csr.pt \
  --source-cache runs/family_index/train_sources_8192.pt \
  --query-head-file runs/family_index/train_relation_heads.txt \
  --rules-output runs/family_rules/rules.tsv
```

Evaluate Test MRR and Hit@1/3/10 on GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python rules/run.py evaluate --method DRUM \
  --data-dir data/family --rule-file runs/family_rules/rules.tsv \
  --csr-cache runs/family_index/train_compact_csr.pt \
  --output-dir runs/family_rule_test
```

Set `--eval-split valid` to evaluate the validation set.

Use `--help` on `src/main.py`, `rules/prepare.py`, `rules/extract.py`, or `rules/evaluate.py` for all options.
