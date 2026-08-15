import gc
import shutil
import ssl

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


def load_bbbp():
    _, (train_dataset, _, test_dataset), _ = dc.molnet.load_bbbp(
        featurizer=dc.feat.RawFeaturizer(smiles=True),
        splitter='random',
        transformers=[])
    return train_dataset, test_dataset


def load_bace():
    _, (train_dataset, _, test_dataset), _ = dc.molnet.load_bace_classification(
        featurizer=dc.feat.RawFeaturizer(smiles=True),
        splitter='random',
        transformers=[])
    return train_dataset, test_dataset


def load_hiv():
    _, (train_dataset, _, test_dataset), _ = dc.molnet.load_hiv(
        featurizer=dc.feat.RawFeaturizer(smiles=True),
        splitter='random',
        transformers=[])
    return train_dataset, test_dataset


def load_sider():
    _, (train_dataset, _, test_dataset), _ = dc.molnet.load_sider(
        featurizer=dc.feat.RawFeaturizer(smiles=True),
        splitter='random',
        transformers=[])
    return train_dataset, test_dataset


def load_clintox():
    _, (train_dataset, _, test_dataset), _ = dc.molnet.load_clintox(
        featurizer=dc.feat.RawFeaturizer(smiles=True),
        splitter='random',
        transformers=[])
    return train_dataset, test_dataset


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


def build_pretraining_tox21_dataset():
    _, (train_dataset, _, _), _ = dc.molnet.load_tox21(
        featurizer=dc.feat.RawFeaturizer(smiles=True),
        splitter='random',
        transformers=[])
    smiles = train_dataset.X[:MAX_SAMPLES]
    labels = train_dataset.y[:MAX_SAMPLES, 0]
    text_list = [
        f"SMILES: {i}. Toxicity: {int(j)}." for i, j in zip(smiles, labels)
    ]
    return dc.data.DiskDataset.from_numpy(X=np.array(text_list),
                                          y=np.array(text_list))


def build_pretraining_bbbp_dataset():
    train_dataset, _ = load_bbbp()
    smiles = train_dataset.X[:MAX_SAMPLES]
    labels = train_dataset.y.flatten()[:MAX_SAMPLES]
    text_list = [
        f"SMILES: {i}. BBB Permeant: {int(j)}." for i, j in zip(smiles, labels)
    ]
    return dc.data.DiskDataset.from_numpy(X=np.array(text_list),
                                          y=np.array(text_list))


def build_pretraining_bace_dataset():
    train_dataset, _ = load_bace()
    smiles = train_dataset.X[:MAX_SAMPLES]
    labels = train_dataset.y.flatten()[:MAX_SAMPLES]
    text_list = [
        f"SMILES: {i}. BACE Inhibitor: {int(j)}."
        for i, j in zip(smiles, labels)
    ]
    return dc.data.DiskDataset.from_numpy(X=np.array(text_list),
                                          y=np.array(text_list))


def build_pretraining_hiv_dataset():
    train_dataset, _ = load_hiv()
    smiles = train_dataset.X[:MAX_SAMPLES]
    labels = train_dataset.y.flatten()[:MAX_SAMPLES]
    text_list = [
        f"SMILES: {i}. HIV Active: {int(j)}." for i, j in zip(smiles, labels)
    ]
    return dc.data.DiskDataset.from_numpy(X=np.array(text_list),
                                          y=np.array(text_list))


def build_pretraining_sider_dataset():
    train_dataset, _ = load_sider()
    smiles = train_dataset.X[:MAX_SAMPLES]
    labels = train_dataset.y[:MAX_SAMPLES, 0]
    text_list = [
        f"SMILES: {i}. Side Effect: {int(j)}." for i, j in zip(smiles, labels)
    ]
    return dc.data.DiskDataset.from_numpy(X=np.array(text_list),
                                          y=np.array(text_list))


def build_pretraining_clintox_dataset():
    train_dataset, _ = load_clintox()
    smiles = train_dataset.X[:MAX_SAMPLES]
    labels = train_dataset.y[:MAX_SAMPLES, 0]
    text_list = [
        f"SMILES: {i}. FDA Approved: {int(j)}." for i, j in zip(smiles, labels)
    ]
    return dc.data.DiskDataset.from_numpy(X=np.array(text_list),
                                          y=np.array(text_list))


def build_pretraining_lipo_dataset():
    train_dataset, _ = load_lipo()
    smiles = train_dataset.X[:MAX_SAMPLES]
    values = train_dataset.y.flatten()[:MAX_SAMPLES]
    text_list = [
        f"SMILES: {i}. Lipophilicity: {j}." for i, j in zip(smiles, values)
    ]
    return dc.data.DiskDataset.from_numpy(X=np.array(text_list),
                                          y=np.array(text_list))


def build_pretraining_freesolv_dataset():
    train_dataset, _ = load_freesolv()
    smiles = train_dataset.X[:MAX_SAMPLES]
    values = train_dataset.y.flatten()[:MAX_SAMPLES]
    text_list = [
        f"SMILES: {i}. Hydration Free Energy: {j}."
        for i, j in zip(smiles, values)
    ]
    return dc.data.DiskDataset.from_numpy(X=np.array(text_list),
                                          y=np.array(text_list))


def build_pretraining_clearance_dataset():
    train_dataset, _ = load_clearance()
    smiles = train_dataset.X[:MAX_SAMPLES]
    values = train_dataset.y.flatten()[:MAX_SAMPLES]
    text_list = [
        f"SMILES: {i}. Clearance: {j}." for i, j in zip(smiles, values)
    ]
    return dc.data.DiskDataset.from_numpy(X=np.array(text_list),
                                          y=np.array(text_list))


# Builders concatenated together into the continued-pretraining corpus.
PRETRAINING_DATASET_BUILDERS = [
    build_pretraining_delaney_dataset,
    build_pretraining_bbbp_dataset,
    build_pretraining_tox21_dataset,
    build_pretraining_bace_dataset,
    build_pretraining_hiv_dataset,
    build_pretraining_sider_dataset,
    build_pretraining_clintox_dataset,
    build_pretraining_lipo_dataset,
    build_pretraining_freesolv_dataset,
    build_pretraining_clearance_dataset,
]


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


def continued_pretraining(batch_size=10):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print("\n Task: causal_lm (continued pretraining)")
    pretraining_datasets = [
        builder() for builder in PRETRAINING_DATASET_BUILDERS
    ]
    train_text_dataset = dc.data.DiskDataset.from_numpy(
        X=np.concatenate([d.X for d in pretraining_datasets]),
        y=np.concatenate([d.y for d in pretraining_datasets]))

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


if __name__ == "__main__":
    continued_pretraining()
