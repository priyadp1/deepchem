import gc
import os
import shutil
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


def build_pretraining_delaney_dataset():
    df = pd.read_csv("datasets/delaney-processed.csv")
    smiles = df["smiles"].values[:MAX_SAMPLES]
    solubility = df[
        "measured log solubility in mols per litre"].values[:MAX_SAMPLES]
    text_list = [
        f"SMILES: {i}. Solubility: {j}." for i, j in zip(smiles, solubility)
    ]
    return dc.data.DiskDataset.from_numpy(X=np.array(text_list),
                                          y=np.array(text_list))


def build_pretraining_bbbp_dataset():
    train_dataset, test_dataset = load_bbbp()
    smiles = np.concatenate([train_dataset.X, test_dataset.X])[:MAX_SAMPLES]
    labels = np.concatenate([train_dataset.y,
                             test_dataset.y]).flatten()[:MAX_SAMPLES]
    text_list = [
        f"SMILES: {i}. BBB Permeant: {int(j)}." for i, j in zip(smiles, labels)
    ]
    return dc.data.DiskDataset.from_numpy(X=np.array(text_list),
                                          y=np.array(text_list))


def run_generation(hf_model, quantized):
    prompts = [
        "OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O.",
        "Cc1occc1C(=O)Nc2ccccc2."
    ]

    def _restore_from_lightning_checkpoint():
        LightningTorchModel(model=hf_model,
                            batch_size=hf_model.batch_size,
                            model_dir=hf_model.model_dir).restore(strict=False)

    if quantized:
        _restore_from_lightning_checkpoint()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        hf_model.model.to("cpu")
        torch.cuda.empty_cache()
        _restore_from_lightning_checkpoint()
        hf_model.model.to(device)

    # Sanity check for NaN or Inf in the logits before generation

    hf_model.model.eval()
    with torch.no_grad():
        probe = hf_model.tokenizer(prompts, padding=True, return_tensors="pt")
        probe = {k: v.to(hf_model.device) for k, v in probe.items()}
        logits = hf_model.model(**probe).logits
        print(f"[diagnostic] forward logits: "
              f"any_nan={torch.isnan(logits).any().item()} "
              f"any_inf={torch.isinf(logits).any().item()} "
              f"min={logits.min().item():.3f} max={logits.max().item():.3f}")

    prompt_dataset = dc.data.NumpyDataset(X=np.array(prompts))
    outputs = hf_model.generate(prompt_dataset,
                                max_length=100,
                                do_sample=True,
                                temperature=0.7,
                                top_k=50,
                                top_p=0.9,
                                repetition_penalty=1.1)

    print("Generated Outputs:")
    for out in outputs:
        print(out)


PRETRAINED_DIR = "./olmo_pretrained_backbone"


def continued_pretraining(batch_size=8):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print("\n Task: causal_lm (continued pretraining)")
    delaney_dataset = build_pretraining_delaney_dataset()
    bbbp_dataset = build_pretraining_bbbp_dataset()
    train_text_dataset = dc.data.DiskDataset.from_numpy(
        X=np.concatenate([delaney_dataset.X, bbbp_dataset.X]),
        y=np.concatenate([delaney_dataset.y, bbbp_dataset.y]))

    pretrain_model = Olmo(task_type="causal_lm",
                          tokenizer_path="allenai/OLMo-1B-hf",
                          torch_dtype=dtype,
                          finetune_strategy="qlora",
                          model_dir="./olmo_checkpoints_causal_lm",
                          learning_rate=1e-5,
                          gradient_checkpointing=True,
                          skip_weight_init=True,
                          batch_size=batch_size)

    pretrain_model.load_from_pretrained("allenai/OLMo-1B-hf",
                                        from_hf_checkpoint=True)

    num_gpus = torch.cuda.device_count()
    trainer = LightningTorchModel(
        model=pretrain_model,
        batch_size=batch_size,
        model_dir="./olmo_checkpoints_causal_lm",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1 if torch.cuda.is_available() else 1,
        strategy="ddp" if num_gpus > 1 else "auto",
        enable_progress_bar=True,
        log_every_n_steps=1)

    print(f"Starting continued pretraining on causal_lm dataset "
          f"({num_gpus} GPU(s))...")
    trainer.fit(train_text_dataset,
                nb_epoch=5,
                max_checkpoints_to_keep=1,
                num_workers=0)

    is_main_process = trainer.trainer.is_global_zero

    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    if is_main_process:
        run_generation(pretrain_model, quantized=True)

        if pretrain_model.finetune_strategy in ("lora", "qlora"):
            pretrain_model.model = pretrain_model.model.merge_and_unload()

        shutil.rmtree(PRETRAINED_DIR, ignore_errors=True)
        pretrain_model.model.save_pretrained(PRETRAINED_DIR)
        pretrain_model.tokenizer.save_pretrained(PRETRAINED_DIR)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    del pretrain_model
    gc.collect()
    torch.cuda.empty_cache()


FINETUNE_DIR = "./olmo_checkpoints_regression"


def build_regression_dataset():
    df = pd.read_csv("datasets/delaney-processed.csv")
    # Skip the first MAX_SAMPLES rows: those are the continued-pretraining
    # corpus (build_pretraining_delaney_dataset), so excluding them here
    # keeps this dataset disjoint from it.
    smiles = df["smiles"].values[MAX_SAMPLES:]
    solubility = df["measured log solubility in mols per litre"].values[
        MAX_SAMPLES:].astype(np.float32).reshape(-1, 1)
    dataset = dc.data.DiskDataset.from_numpy(X=smiles, y=solubility)
    return dc.splits.RandomSplitter().train_test_split(dataset,
                                                       frac_train=0.8,
                                                       seed=42)


def finetune_regression(nb_epoch=20, batch_size=8):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    train_dataset, test_dataset = build_regression_dataset()
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    finetune_model = Olmo(task_type="regression",
                          tokenizer_path="allenai/OLMo-1B-hf",
                          n_tasks=1,
                          torch_dtype=dtype,
                          finetune_strategy="qlora",
                          gradient_checkpointing=True,
                          model_dir=FINETUNE_DIR,
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
        model_dir=FINETUNE_DIR,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1 if torch.cuda.is_available() else 1,
        strategy="ddp" if num_gpus > 1 else "auto",
        enable_progress_bar=True,
        log_every_n_steps=1)

    t0 = time.time()
    trainer.fit(train_dataset, nb_epoch=nb_epoch, num_workers=0)
    elapsed = time.time() - t0

    train_rms = finetune_model.evaluate(train_dataset,
                                        metrics=[metric])["rms_score"]
    test_rms = finetune_model.evaluate(test_dataset,
                                       metrics=[metric])["rms_score"]
    peak_mem = (torch.cuda.max_memory_allocated() /
                1e9 if torch.cuda.is_available() else 0.0)

    print(f"After {nb_epoch} epochs on {num_gpus} GPU(s) ({elapsed:.1f}s): "
          f"train_rms={train_rms:.3f} test_rms={test_rms:.3f} "
          f"peak_mem={peak_mem:.2f}GB")
    print(f"Checkpoints saved under "
          f"{os.path.join(FINETUNE_DIR, 'checkpoints')}")

    del finetune_model, trainer
    gc.collect()
    torch.cuda.empty_cache()


CLASSIFICATION_FINETUNE_DIR = "./olmo_checkpoints_classification"


def load_bbbp():
    _, (train_dataset, _, test_dataset), _ = dc.molnet.load_bbbp(
        featurizer=dc.feat.RawFeaturizer(smiles=True),
        splitter='random',
        transformers=[])
    return train_dataset, test_dataset


def build_classification_dataset():
    return load_bbbp()


def finetune_classification(nb_epoch=20, batch_size=8):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    train_dataset, test_dataset = build_classification_dataset()
    print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

    finetune_model = Olmo(task_type="classification",
                          tokenizer_path="allenai/OLMo-1B-hf",
                          n_tasks=1,
                          torch_dtype=dtype,
                          finetune_strategy="qlora",
                          gradient_checkpointing=True,
                          model_dir=CLASSIFICATION_FINETUNE_DIR,
                          batch_size=batch_size,
                          learning_rate=3e-5,
                          skip_weight_init=True)

    finetune_model.load_from_pretrained(PRETRAINED_DIR, from_hf_checkpoint=True)

    metric = dc.metrics.Metric(dc.metrics.roc_auc_score)

    baseline_auc = finetune_model.evaluate(test_dataset,
                                           metrics=[metric])["roc_auc_score"]
    print(
        f"Baseline test ROC-AUC (random classification head, pretrained backbone): "
        f"{baseline_auc:.3f}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    num_gpus = torch.cuda.device_count()
    trainer = LightningTorchModel(
        model=finetune_model,
        batch_size=batch_size,
        model_dir=CLASSIFICATION_FINETUNE_DIR,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1 if torch.cuda.is_available() else 1,
        strategy="ddp" if num_gpus > 1 else "auto",
        enable_progress_bar=True,
        log_every_n_steps=1)

    t0 = time.time()
    trainer.fit(train_dataset, nb_epoch=nb_epoch, num_workers=0)
    elapsed = time.time() - t0

    train_auc = finetune_model.evaluate(train_dataset,
                                        metrics=[metric])["roc_auc_score"]
    test_auc = finetune_model.evaluate(test_dataset,
                                       metrics=[metric])["roc_auc_score"]
    peak_mem = (torch.cuda.max_memory_allocated() /
                1e9 if torch.cuda.is_available() else 0.0)

    print(f"After {nb_epoch} epochs on {num_gpus} GPU(s) ({elapsed:.1f}s): "
          f"train_auc={train_auc:.3f} test_auc={test_auc:.3f} "
          f"peak_mem={peak_mem:.2f}GB")
    print(f"Checkpoints saved under "
          f"{os.path.join(CLASSIFICATION_FINETUNE_DIR, 'checkpoints')}")

    del finetune_model, trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    continued_pretraining()
    finetune_regression()
    finetune_classification()
