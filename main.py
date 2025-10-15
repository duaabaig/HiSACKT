import os
import argparse
import json
import pickle
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from data_loaders.assist2009 import ASSIST2009
from data_loaders.statics2011 import Statics2011
from data_loaders.algebra2005 import Algebra2005
from data_loaders.assist2015 import ASSIST2015
from models.dkt import DKT
from models.dkt_plus import DKTPlus
from models.dkvmn import DKVMN
from models.sakt import SAKT
from models.hisackt import HiSACKT
from models.hisackt_no_skill_clustering import HiSACKT_NoSkillClustering
from models.hisackt_no_local import HiSACKT_NoLocal
from models.hisackt_no_cross_level import HiSACKT_NoCrossLevel
from models.hisackt_no_skill_group import HiSACKT_NoSkillGroup
from models.gkt import PAM, MHA
from models.gktLLM import LLM_AM
from models.utils import collate_fn
import optuna
import numpy as np
import csv


def objective(trial, dataset, train_loader, val_loader, test_loader, ckpt_path, num_epochs, base_config, model_name):
    torch.cuda.empty_cache()
    trial_config = base_config.copy()
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.002, log=True)

    if model_name in ["dkt", "dkt+"]:
        trial_config["hidden_size"] = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
        trial_config["emb_size"] = trial.suggest_categorical("emb_size", [50, 100, 200])
        if model_name == "dkt":
            model = DKT(dataset.num_q, **trial_config).to('cuda')
        elif model_name == "dkt+":
            trial_config["lambda_r"] = trial.suggest_float("lambda_r", 0.001, 0.1, log=True)
            trial_config["lambda_w1"] = trial.suggest_float("lambda_w1", 0.0001, 0.01, log=True)
            trial_config["lambda_w2"] = trial.suggest_float("lambda_w2", 1.0, 10.0, log=True)
            model = DKTPlus(dataset.num_q, **trial_config).to('cuda')
    elif model_name == "dkvmn":
        trial_config["dim_s"] = trial.suggest_categorical("dim_s", [16, 32, 64])
        trial_config["size_m"] = trial.suggest_categorical("size_m", [10, 20, 50])
        model = DKVMN(dataset.num_q, **trial_config).to('cuda')
    elif model_name in ["sakt", "gkt", "gktLLM"]:
        if model_name == "gkt":
            trial_config["num_attn_heads"] = trial.suggest_categorical("num_attn_heads", [2, 4])
        else:
            trial_config["num_attn_heads"] = trial.suggest_categorical("num_attn_heads", [4, 8, 16])
        if model_name == "sakt":
            trial_config["d"] = trial.suggest_categorical("d", [32, 64, 128])
            trial_config["dropout"] = trial.suggest_float("dropout", 0.2, 0.5)
            model = SAKT(dataset.num_q, **trial_config).to('cuda')
        else:
            trial_config["hidden_size"] = trial.suggest_categorical("hidden_size", [8, 16, 32])
            if model_name == "gkt":
                trial_config["method"] = base_config["method"]
                if trial_config["method"] == "PAM":
                    model = PAM(dataset.num_q, **trial_config).to('cuda')
                elif trial_config["method"] == "MHA":
                    model = MHA(dataset.num_q, **trial_config).to('cuda')
            elif model_name == "gktLLM":
                trial_config["method"] = "LLM_AM"
                model = LLM_AM(dataset.num_q, **trial_config).to('cuda')
    elif model_name in ["hisackt", "hisackt_no_skill_clustering", "hisackt_no_local", "hisackt_no_cross_level",
                        "hisackt_no_skill_group"]:
        num_attn_heads = trial.suggest_categorical("num_attn_heads", [4, 8, 16])
        d = trial.suggest_categorical("d", [32, 64, 128])
        while d % num_attn_heads != 0:
            d = trial.suggest_categorical("d", [32, 64, 128])
        dropout = trial.suggest_float("dropout", 0.2, 0.5)

        trial_config = {
            "n": 100,
            "d": d,
            "num_attn_heads": num_attn_heads,
            "ha_num_heads_per_level": [num_attn_heads] * 4,
            "local_window": base_config["local_window"],
            "dropout": dropout
        }

        if model_name == "hisackt":
            model = HiSACKT(dataset.num_q, **trial_config).to('cuda')
        elif model_name == "hisackt_no_skill_clustering":
            model = HiSACKT_NoSkillClustering(dataset.num_q, **trial_config).to('cuda')
        elif model_name == "hisackt_no_local":
            model = HiSACKT_NoLocal(dataset.num_q, **trial_config).to('cuda')
        elif model_name == "hisackt_no_cross_level":
            model = HiSACKT_NoCrossLevel(dataset.num_q, **trial_config).to('cuda')
        elif model_name == "hisackt_no_skill_group":
            model = HiSACKT_NoSkillGroup(dataset.num_q, **trial_config).to('cuda')

    opt = Adam(model.parameters(), learning_rate)
    aucs, loss_means, max_val_auc = model.train_model(
        train_loader, val_loader, test_loader, num_epochs, opt, ckpt_path, patience=10
    )

    torch.cuda.empty_cache()
    return max_val_auc


def main(model_name, dataset_name):
    if not os.path.isdir("ckpts"):
        os.mkdir("ckpts")
    ckpt_path = os.path.join("ckpts", model_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)
    ckpt_path = os.path.join(ckpt_path, dataset_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    with open("config.json") as f:
        config = json.load(f)

        
        config_key = "hisackt" if model_name.startswith("hisackt") else model_name
        model_config = config[config_key]
        train_config = config["train_config"]

    batch_size = train_config["batch_size"]
    num_epochs = train_config["num_epochs"]
    train_ratio = train_config["train_ratio"]
    val_ratio = train_config["val_ratio"]
    seq_len = train_config["seq_len"]

    if dataset_name == "ASSIST2009":
        dataset = ASSIST2009(seq_len)
    elif dataset_name == "Statics2011":
        dataset = Statics2011(seq_len)
    elif dataset_name == "algebra_2005_2006":
        dataset = Algebra2005(seq_len)
    elif dataset_name == "ASSIST2015":
        dataset = ASSIST2015(seq_len)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)
    with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=4)

    dataset_size = len(dataset)
    train_size = int(len(dataset) * train_ratio)
    val_size = int(len(dataset) * val_ratio)
    test_size = len(dataset) - train_size - val_size
    torch.manual_seed(42)
    generator = torch.Generator(device=device)
    train_indices_path = os.path.join(dataset.dataset_dir, "train_indices.pkl")
    val_indices_path = os.path.join(dataset.dataset_dir, "val_indices.pkl")
    test_indices_path = os.path.join(dataset.dataset_dir, "test_indices.pkl")

    if os.path.exists(train_indices_path) and os.path.exists(val_indices_path) and os.path.exists(test_indices_path):
        with open(train_indices_path, "rb") as f:
            train_indices = pickle.load(f)
        with open(val_indices_path, "rb") as f:
            val_indices = pickle.load(f)
        with open(test_indices_path, "rb") as f:
            test_indices = pickle.load(f)
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

        for split_name, indices in [("train", train_indices), ("val", val_indices), ("test", test_indices)]:
            df = pd.DataFrame({
                "user_id": [dataset.u2idx.get(dataset.u_list[i], "") if i < len(dataset.u_list) else "" for i in
                            indices],
                "q_seqs": [dataset.q_seqs[i] for i in indices],
                "r_seqs": [dataset.r_seqs[i] for i in indices]
            })
            if isinstance(dataset, Algebra2005):
                df["q_seqs"] = df["q_seqs"].apply(lambda x: "[" + ",".join(map(str, map(int, x))) + "]")
                df["r_seqs"] = df["r_seqs"].apply(lambda x: "[" + " ".join(map(str, map(int, x))) + "]")
            else:
                df["q_seqs"] = df["q_seqs"].apply(lambda x: "[" + " ".join(map(str, map(int, x))) + "]")
                df["r_seqs"] = df["r_seqs"].apply(lambda x: "[" + " ".join(map(str, map(int, x))) + "]")
            padded_df = df[
                df["q_seqs"].apply(lambda x: all(float(v) == -1.0 for v in x.strip("[]").replace(",", " ").split()))]
            if not padded_df.empty:
                print(f"Warning: {len(padded_df)} fully padded sequences in {split_name}_data.csv")
                print(padded_df.head())
            df.to_csv(os.path.join(dataset.dataset_dir, f"{split_name}_data.csv"), index=False)
            print(f"{split_name}_data.csv saved with {len(df)} rows")

    else:
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )
        with open(train_indices_path, "wb") as f:
            pickle.dump(train_dataset.indices, f)
        with open(val_indices_path, "wb") as f:
            pickle.dump(val_dataset.indices, f)
        with open(test_indices_path, "wb") as f:
            pickle.dump(test_dataset.indices, f)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, generator=generator
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, generator=generator
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_size, shuffle=False,
        collate_fn=collate_fn, generator=generator
    )
    print(f"Hyperparameter searching for {model_name}:")

    trial_results = []

    def callback(study, trial):
        trial_data = {
            "Trial": trial.number,
            "AUC": trial.value,
            **trial.params,
            "Best_Trial": 1 if trial.number == study.best_trial.number else 0
        }
        trial_results.append(trial_data)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, dataset, train_loader, val_loader, test_loader, ckpt_path, num_epochs,
                                model_config, model_name),
        n_trials=10,
        callbacks=[callback]
    )

    best_params = study.best_params
    best_val_auc = study.best_value
    print(f"Best Val AUC: {best_val_auc}, Best Params: {best_params}")

    csv_path = os.path.join(ckpt_path, "trial_results.csv")
    headers = ["Trial", "AUC"] + list(best_params.keys()) + ["Best_Trial"]
    with open(csv_path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for trial_data in trial_results:
            writer.writerow(trial_data)
    print(f"Trial results saved to {csv_path}")

    final_config = model_config.copy()
    if model_name in ["dkt", "dkt+"]:
        final_config["hidden_size"] = best_params["hidden_size"]
        final_config["emb_size"] = best_params["emb_size"]
        if model_name == "dkt":
            model = DKT(dataset.num_q, **final_config).to(device)
        elif model_name == "dkt+":
            final_config["lambda_r"] = best_params["lambda_r"]
            final_config["lambda_w1"] = best_params["lambda_w1"]
            final_config["lambda_w2"] = best_params["lambda_w2"]
            model = DKTPlus(dataset.num_q, **final_config).to(device)
    elif model_name == "dkvmn":
        final_config["dim_s"] = best_params["dim_s"]
        final_config["size_m"] = best_params["size_m"]
        model = DKVMN(dataset.num_q, **final_config).to(device)
    elif model_name in ["sakt", "gkt", "gktLLM"]:
        final_config["num_attn_heads"] = best_params["num_attn_heads"]
        if model_name == "sakt":
            final_config["d"] = best_params["d"]
            final_config["dropout"] = best_params["dropout"]
            model = SAKT(dataset.num_q, **final_config).to(device)
        else:
            final_config["hidden_size"] = best_params["hidden_size"]
            if model_name == "gkt":
                final_config["method"] = model_config["method"]
                if final_config["method"] == "PAM":
                    model = PAM(dataset.num_q, **final_config).to(device)
                elif final_config["method"] == "MHA":
                    model = MHA(dataset.num_q, **final_config).to(device)
            elif model_name == "gktLLM":
                final_config["method"] = "LLM_AM"
                model = LLM_AM(dataset.num_q, **final_config).to(device)
    elif model_name in ["hisackt", "hisackt_no_skill_clustering", "hisackt_no_local", "hisackt_no_cross_level",
                        "hisackt_no_skill_group"]:
        final_config["num_attn_heads"] = best_params["num_attn_heads"]
        final_config["d"] = best_params["d"]
        final_config["dropout"] = best_params["dropout"]

        if model_name == "hisackt":
            model = HiSACKT(dataset.num_q, **final_config).to(device)
        elif model_name == "hisackt_no_skill_clustering":
            model = HiSACKT_NoSkillClustering(dataset.num_q, **final_config).to(device)
        elif model_name == "hisackt_no_local":
            model = HiSACKT_NoLocal(dataset.num_q, **final_config).to(device)
        elif model_name == "hisackt_no_cross_level":
            model = HiSACKT_NoCrossLevel(dataset.num_q, **final_config).to(device)
        elif model_name == "hisackt_no_skill_group":
            model = HiSACKT_NoSkillGroup(dataset.num_q, **final_config).to(device)

    opt = Adam(model.parameters(), best_params["learning_rate"])
    print(f"Hyperparameter optimization completed..."
          f"Best Parameters are from trial number: {study.best_trial.number} with value: {study.best_value}."
          f"Calling model with optimized parameters")

    aucs, loss_means, max_val_auc = model.train_model(
        train_loader, val_loader, test_loader, num_epochs, opt, ckpt_path, patience=10
    )

    with open(os.path.join(ckpt_path, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)
    with open(os.path.join(ckpt_path, "aucs.pkl"), "wb") as f:
        pickle.dump(aucs, f)
    with open(os.path.join(ckpt_path, "loss_means.pkl"), "wb") as f:
        pickle.dump(loss_means, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="hisackt")
    parser.add_argument("--dataset_name", type=str,
                        default="ASSIST2015")  # algebra_2005_2006 , Statics2011 , ASSIST2009
    args = parser.parse_args()
    main(args.model_name, args.dataset_name)
    print("Done!")
