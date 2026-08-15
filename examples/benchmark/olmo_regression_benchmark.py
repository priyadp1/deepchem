import argparse
import gc
import os
import ssl
import time

try:
    import certifi
    ssl._create_default_https_context = lambda: ssl.create_default_context(
        cafile=certifi.where())
except ImportError:
    pass

import deepchem as dc
from deepchem.models.torch_models.olmo import Olmo
from deepchem.models.lightning import LightningTorchModel
import numpy as np
import pandas as pd
import torch

MAX_SAMPLES = 300  # subset for quick testing
PRETRAINED_DIR = "./olmo_pretrained_backbone"
FINETUNE_DIR = "./olmo_checkpoints_regression"


def load_lipo():
    _, (train_dataset, _, test_dataset), _ = dc.molnet.load_lipo(
        featurizer=dc.feat.RawFeaturizer(smiles=True),
        splitter='random',
        transformers=[])
    return train_dataset, test_dataset


def load_freesolv():
    _, (train_dataset, _, test_dataset), _ = dc.molnet.load_freesolv(
        featurizer=dc.feat.RawFeaturizer(smiles=True),
        splitter='random',
        transformers=[])
    return train_dataset, test_dataset


def load_clearance():
    _, (train_dataset, _, test_dataset), _ = dc.molnet.load_clearance(
        featurizer=dc.feat.RawFeaturizer(smiles=True),
        splitter='random',
        transformers=[])
    return train_dataset, test_dataset


def load_bace_pic50():
    _, (train_dataset, _, test_dataset), _ = dc.molnet.load_bace_regression(
        featurizer=dc.feat.RawFeaturizer(smiles=True),
        splitter='random',
        transformers=[])
    return train_dataset, test_dataset


def build_delaney_regression_dataset():
    df = pd.read_csv("datasets/delaney-processed.csv")
    # Skip the first MAX_SAMPLES rows: those are the continued-pretraining
    # corpus (build_pretraining_delaney_dataset), so excluding them here
    # keeps this dataset disjoint from it.
    smiles = df["smiles"].values[MAX_SAMPLES:]
    solubility = df["measured log solubility in mols per litre"].values[
        MAX_SAMPLES:].astype(np.float32).reshape(-1, 1)
    dataset = dc.data.DiskDataset.from_numpy(X=smiles, y=solubility)
    train_dataset, test_dataset = dc.splits.RandomSplitter().train_test_split(
        dataset, frac_train=0.8, seed=42)
    return train_dataset, test_dataset, 1


def build_bace_regression_dataset():
    train_dataset, test_dataset = load_bace_pic50()
    return train_dataset, test_dataset, 1


def build_lipo_regression_dataset():
    train_dataset, test_dataset = load_lipo()
    return train_dataset, test_dataset, 1


def build_freesolv_regression_dataset():
    train_dataset, test_dataset = load_freesolv()
    return train_dataset, test_dataset, 1


def build_clearance_regression_dataset():
    train_dataset, test_dataset = load_clearance()
    return train_dataset, test_dataset, 1


REGRESSION_DATASETS = {
    "delaney": build_delaney_regression_dataset,
    "bace": build_bace_regression_dataset,
    "lipo": build_lipo_regression_dataset,
    "freesolv": build_freesolv_regression_dataset,
    "clearance": build_clearance_regression_dataset,
}


def finetune_regression(dataset_name="delaney", nb_epoch=10, batch_size=10):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    train_dataset, test_dataset, n_tasks = REGRESSION_DATASETS[dataset_name]()
    print(f"[{dataset_name}] Train size: {len(train_dataset)}, "
          f"Test size: {len(test_dataset)}, n_tasks: {n_tasks}")

    model_dir = f"{FINETUNE_DIR}_{dataset_name}"
    finetune_model = Olmo(task_type="regression",
                          tokenizer_path="allenai/OLMo-1B-hf",
                          n_tasks=n_tasks,
                          torch_dtype=dtype,
                          finetune_strategy="qlora",
                          gradient_checkpointing=True,
                          model_dir=model_dir,
                          batch_size=batch_size,
                          learning_rate=3e-5,
                          skip_weight_init=True)

    finetune_model.load_from_pretrained(PRETRAINED_DIR, from_hf_checkpoint=True)

    metric = dc.metrics.Metric(dc.metrics.rms_score)

    baseline_rms = finetune_model.evaluate(test_dataset,
                                           metrics=[metric])["rms_score"]
    print(f"Baseline test RMS (random regression head, pretrained backbone): "
          f"{baseline_rms:.3f}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    num_gpus = torch.cuda.device_count()
    trainer = LightningTorchModel(
        model=finetune_model,
        batch_size=batch_size,
        model_dir=model_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1 if torch.cuda.is_available() else 1,
        strategy="ddp" if num_gpus > 1 else "auto",
        enable_progress_bar=True,
        log_every_n_steps=1)

    t0 = time.time()
    trainer.fit(train_dataset, nb_epoch=nb_epoch, num_workers=0)
    elapsed = time.time() - t0

    finetune_model.model.to(finetune_model.device)

    train_rms = finetune_model.evaluate(train_dataset,
                                        metrics=[metric])["rms_score"]
    test_rms = finetune_model.evaluate(test_dataset,
                                       metrics=[metric])["rms_score"]
    peak_mem = (torch.cuda.max_memory_allocated() /
                1e9 if torch.cuda.is_available() else 0.0)

    print(
        f"[{dataset_name}] After {nb_epoch} epochs on {num_gpus} GPU(s) "
        f"({elapsed:.1f}s): train_rms={train_rms:.3f} test_rms={test_rms:.3f} "
        f"peak_mem={peak_mem:.2f}GB")
    print(f"Checkpoints saved under {os.path.join(model_dir, 'checkpoints')}")

    del finetune_model, trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        choices=list(REGRESSION_DATASETS),
                        default=None,
                        help="Run a single dataset instead of all of them.")
    args = parser.parse_args()

    if args.dataset:
        finetune_regression(args.dataset)
    else:
        for dataset_name in REGRESSION_DATASETS:
            finetune_regression(dataset_name)
