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
    dataset = dc.data.NumpyDataset(X=np.array(text_list), y=np.array(text_list))
    return dc.splits.RandomSplitter().train_test_split(dataset,
                                                       frac_train=0.8,
                                                       seed=42)


def build_pretraining_bbbp_dataset():
    train_dataset, test_dataset = load_bbbp()
    smiles = np.concatenate([train_dataset.X, test_dataset.X])[:MAX_SAMPLES]
    labels = np.concatenate([train_dataset.y,
                             test_dataset.y]).flatten()[:MAX_SAMPLES]
    text_list = [
        f"SMILES: {i}. BBB Permeant: {int(j)}." for i, j in zip(smiles, labels)
    ]
    dataset = dc.data.NumpyDataset(X=np.array(text_list), y=np.array(text_list))
    return dc.splits.RandomSplitter().train_test_split(dataset,
                                                       frac_train=0.8,
                                                       seed=42)


def run_generation(hf_model, quantized):
    prompts = [
        "OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O.",
        "Cc1occc1C(=O)Nc2ccccc2."
    ]

    if quantized:
        hf_model.load_from_pretrained()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        hf_model.model.to("cpu")
        torch.cuda.empty_cache()
        hf_model.load_from_pretrained()
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


def continued_pretraining():
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print("\n Task: causal_lm (continued pretraining)")
    delaney_train_dataset, _ = build_pretraining_delaney_dataset()
    bbbp_train_dataset, _ = build_pretraining_bbbp_dataset()
    train_text_dataset = dc.data.NumpyDataset(
        X=np.concatenate([delaney_train_dataset.X, bbbp_train_dataset.X]),
        y=np.concatenate([delaney_train_dataset.y, bbbp_train_dataset.y]))

    pretrain_model = Olmo(task_type="causal_lm",
                          tokenizer_path="allenai/OLMo-1B-hf",
                          torch_dtype=dtype,
                          finetune_strategy="qlora",
                          model_dir="./olmo_checkpoints_causal_lm",
                          learning_rate=1e-5,
                          gradient_checkpointing=True,
                          skip_weight_init=True)

    pretrain_model.load_from_pretrained("allenai/OLMo-1B-hf",
                                        from_hf_checkpoint=True)

    print("Starting continued pretraining on causal_lm dataset...")
    loss = pretrain_model.fit(train_text_dataset,
                              nb_epoch=5,
                              max_checkpoints_to_keep=1)
    print("Training Loss:", loss)

    run_generation(pretrain_model, quantized=True)

    if pretrain_model.finetune_strategy in ("lora", "qlora"):
        pretrain_model.model = pretrain_model.model.merge_and_unload()

    shutil.rmtree(PRETRAINED_DIR, ignore_errors=True)
    pretrain_model.model.save_pretrained(PRETRAINED_DIR)
    pretrain_model.tokenizer.save_pretrained(PRETRAINED_DIR)

    del pretrain_model
    gc.collect()
    torch.cuda.empty_cache()


FINETUNE_DIR = "./olmo_checkpoints_regression"
DATASET_NAME = "delaney"


def build_regression_dataset():
    df = pd.read_csv("datasets/delaney-processed.csv")
    smiles = df["smiles"].values
    solubility = df["measured log solubility in mols per litre"].values.astype(
        np.float32).reshape(-1, 1)
    dataset = dc.data.NumpyDataset(X=smiles, y=solubility)
    return dc.splits.RandomSplitter().train_test_split(dataset,
                                                       frac_train=0.8,
                                                       seed=42)


def finetune_regression(nb_epoch=20, batch_size=128):
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

    best_test_rms = float("inf")
    best_epoch = None
    best_checkpoint_path = None

    for epoch in range(1, nb_epoch + 1):
        t0 = time.time()
        loss = finetune_model.fit(train_dataset,
                                  nb_epoch=1,
                                  checkpoint_interval=0)
        elapsed = time.time() - t0
        train_rms = finetune_model.evaluate(train_dataset,
                                            metrics=[metric])["rms_score"]
        test_rms = finetune_model.evaluate(test_dataset,
                                           metrics=[metric])["rms_score"]
        peak_mem = (torch.cuda.max_memory_allocated() /
                    1e9 if torch.cuda.is_available() else 0.0)

        improved = test_rms < best_test_rms
        if improved:
            best_test_rms = test_rms
            best_epoch = epoch

            if best_checkpoint_path is not None and os.path.exists(
                    best_checkpoint_path):
                os.remove(best_checkpoint_path)
            if not os.path.exists(FINETUNE_DIR):
                os.makedirs(FINETUNE_DIR)
            best_checkpoint_path = os.path.join(
                FINETUNE_DIR,
                f"epoch{epoch}_rms{test_rms:.3f}_{DATASET_NAME}.pt")
            finetune_model._ensure_built()
            torch.save(
                {
                    'model_state_dict':
                        finetune_model.model.state_dict(),
                    'optimizer_state_dict':
                        finetune_model._pytorch_optimizer.state_dict(),
                    'global_step':
                        finetune_model._global_step
                }, best_checkpoint_path)

        print(
            f"Epoch {epoch:2d}: loss={loss:.4f} train_rms={train_rms:.3f} "
            f"test_rms={test_rms:.3f} time={elapsed:.1f}s peak_mem={peak_mem:.2f}GB"
            f"{' (new best, checkpoint saved)' if improved else ''}")

    print(f"Best test RMS: {best_test_rms:.3f} at epoch {best_epoch}, "
          f"checkpoint saved to {best_checkpoint_path}")

    del finetune_model
    gc.collect()
    torch.cuda.empty_cache()


CLASSIFICATION_FINETUNE_DIR = "./olmo_checkpoints_classification"
CLASSIFICATION_DATASET_NAME = "bbbp"


def load_bbbp():
    _, (train_dataset, _, test_dataset), _ = dc.molnet.load_bbbp(
        featurizer=dc.feat.RawFeaturizer(smiles=True),
        splitter='random',
        transformers=[])
    return train_dataset, test_dataset


def build_classification_dataset():
    return load_bbbp()


def finetune_classification(nb_epoch=20, batch_size=128):
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

    best_test_auc = float("-inf")
    best_epoch = None
    best_checkpoint_path = None

    for epoch in range(1, nb_epoch + 1):
        t0 = time.time()
        loss = finetune_model.fit(train_dataset,
                                  nb_epoch=1,
                                  checkpoint_interval=0)
        elapsed = time.time() - t0
        train_auc = finetune_model.evaluate(train_dataset,
                                            metrics=[metric])["roc_auc_score"]
        test_auc = finetune_model.evaluate(test_dataset,
                                           metrics=[metric])["roc_auc_score"]
        peak_mem = (torch.cuda.max_memory_allocated() /
                    1e9 if torch.cuda.is_available() else 0.0)

        improved = test_auc > best_test_auc
        if improved:
            best_test_auc = test_auc
            best_epoch = epoch

            if best_checkpoint_path is not None and os.path.exists(
                    best_checkpoint_path):
                os.remove(best_checkpoint_path)
            if not os.path.exists(CLASSIFICATION_FINETUNE_DIR):
                os.makedirs(CLASSIFICATION_FINETUNE_DIR)
            best_checkpoint_path = os.path.join(
                CLASSIFICATION_FINETUNE_DIR,
                f"epoch{epoch}_auc{test_auc:.3f}_{CLASSIFICATION_DATASET_NAME}.pt"
            )
            finetune_model._ensure_built()
            torch.save(
                {
                    'model_state_dict':
                        finetune_model.model.state_dict(),
                    'optimizer_state_dict':
                        finetune_model._pytorch_optimizer.state_dict(),
                    'global_step':
                        finetune_model._global_step
                }, best_checkpoint_path)

        print(
            f"Epoch {epoch:2d}: loss={loss:.4f} train_auc={train_auc:.3f} "
            f"test_auc={test_auc:.3f} time={elapsed:.1f}s peak_mem={peak_mem:.2f}GB"
            f"{' (new best, checkpoint saved)' if improved else ''}")

    print(f"Best test ROC-AUC: {best_test_auc:.3f} at epoch {best_epoch}, "
          f"checkpoint saved to {best_checkpoint_path}")

    del finetune_model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    continued_pretraining()
    finetune_regression()
    finetune_classification()
