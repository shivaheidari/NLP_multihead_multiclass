# NLP_multihead_multiclass
project_root/
├── configs/
│   ├── base.yaml
│   ├── biobert_multitask_small.yaml
│   └── biobert_multitask_full.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py        # DicomText
│   │   ├── collate.py        # DicomCollator
│   │   └── label_maps.py     # helpers to build/load mappings
│   ├── models/
│   │   ├── __init__.py
│   │   └── biobert_multitask.py  # BioBertMultiHead
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py        # train_one_epoch, eval_loop, fit()
│   │   └── optim.py          # build_optimizer, build_scheduler
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py        # per-head accuracy/F1
│   │   └── inference.py      # predict_one / predict_batch
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── seed.py
│   │   └── logging.py
│   └── env/
│       ├── __init__.py
│       └── paths.py          # data paths, output dirs
├── scripts/
│   ├── train.py              # calls into src.training
│   └── evaluate.py
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── runs/
│   └── models/
├── requirements.txt / pyproject.toml
└── README.md
