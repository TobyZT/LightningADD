import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers

import yaml
import argparse
from pathlib import Path
import shutil

from datasets.datasets import ASVspoof2019LA, ASVspoofEval, IntheWild
from utils import *

from train import LightningADDTrainer

if __name__ == "__main__":
    # args parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    # read config
    config = yaml.safe_load(open(args.config))
    exp_name = config["train"]["exp_name"]
    seed = config["train"]["seed"]
    num_epochs = config["train"]["num_epochs"]
    num_workers = config["train"]["num_workers"]
    max_len = config["train"]["max_len"]
    batch_size = config["train"]["batch_size"]
    check_val_every_n_epoch = config["train"]["check_val_every_n_epoch"]
    save_top_k = config["train"]["save_top_k"]
    trainset_path = config["path"]["ASVspoof2019LA"]
    # set seed
    L.seed_everything(seed=seed)

    # set logger
    version = "eval" if args.eval else None
    logger = pl_loggers.TensorBoardLogger(
        save_dir="exp", name=exp_name, version=version
    )
    logger.log_hyperparams(config)

    checkpoint_callback = ModelCheckpoint(
        monitor="eer",
        dirpath=logger.log_dir + "/checkpoints",
        filename="{epoch:02d}",
        save_top_k=save_top_k,
        save_last=True,
        mode="min",
        every_n_epochs=1,
    )

    # set devices
    trainer = L.Trainer(
        devices=1,
        accelerator="gpu",
        max_epochs=num_epochs,
        log_every_n_steps=40,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=check_val_every_n_epoch,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        callbacks=[checkpoint_callback],
        logger=logger,
    )

    #####################################
    # train on ASVspoof 2019 LA dataset #
    #####################################
    model = LightningADDTrainer(config=config)

    if not args.eval:
        dataloader = ASVspoof2019LA(
            base_dir=trainset_path,
            batch_size=batch_size,
            num_workers=num_workers,
            max_len=max_len,
        )
        trainer.fit(model, dataloader)
        exit(0)

    ##############
    # Evaluation #
    ##############

    # load best model
    last_run_version = sorted(
        list(Path(logger.root_dir).glob("version_*")), reverse=True
    )[0]
    last_model_path = last_run_version / "checkpoints" / "last.ckpt"
    model = LightningADDTrainer.load_from_checkpoint(last_model_path, config=config)

    eval_datasets: list[str] = config["eval"]["datasets"]

    scores_dir = Path(logger.log_dir) / "scores"
    scores_dir.mkdir(exist_ok=True)
    output_dir = Path(logger.log_dir) / "outputs"
    output_dir.mkdir(exist_ok=True)

    ##################################
    # Evaluation on ASVspoof 2019 LA #
    ##################################

    if "ASVspoof2019LA" in eval_datasets:
        dataloader = ASVspoof2019LA(
            base_dir=trainset_path,
            batch_size=batch_size,
            num_workers=num_workers,
            max_len=max_len,
        )
        trainer.test(model, dataloader)
        cm_scores_file = scores_dir / "2019LA.txt"
        shutil.copy2(Path(logger.log_dir) / "test_scores.txt", cm_scores_file)
        asv_scores_file = (
            Path(trainset_path)
            / "ASVspoof2019_LA_asv_scores"
            / "ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
        )
        calculate_tDCF_EER_19LA(
            cm_scores_file,
            asv_scores_file,
            output_dir / "2019LA.txt",
        )

    ##################################
    # Evaluation on ASVspoof 2021 LA #
    ##################################

    if "ASVspoof2021LA" in eval_datasets:
        base_dir = config["path"]["ASVspoof2021LA"]
        protocol_dir = f"datasets/labels/ASVspoof2021LA.txt"
        dataloader = ASVspoofEval(
            base_dir=base_dir,
            protocol_dir=protocol_dir,
            max_len=max_len,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        trainer.test(model, dataloader)
        cm_scores_file = scores_dir / "2021LA.txt"
        shutil.copy2(Path(logger.log_dir) / "test_scores.txt", cm_scores_file)
        truth_dir = "datasets/keys/LA_2021"
        calculate_tDCF_EER_21LA(
            cm_scores_file,
            truth_dir,
            output_dir / "2021LA.txt",
        )

    ##################################
    # Evaluation on ASVspoof 2021 DF #
    ##################################

    if "ASVspoof2021DF" in eval_datasets:
        base_dir = config["path"]["ASVspoof2021DF"]
        protocol_dir = f"datasets/labels/ASVspoof2021DF.txt"
        dataloader = ASVspoofEval(
            base_dir=base_dir,
            protocol_dir=protocol_dir,
            max_len=max_len,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        trainer.test(model, dataloader)
        cm_scores_file = scores_dir / "2021DF.txt"
        shutil.copy2(Path(logger.log_dir) / "test_scores.txt", cm_scores_file)
        cm_key_file = "datasets/keys/DF_2021/CM/trial_metadata.txt"
        calculate_EER_21DF(
            cm_scores_file,
            cm_key_file,
            output_dir / "2021DF.txt",
        )

    #############################
    # Evaluation on In-the-Wild #
    #############################

    if "In-the-Wild" in eval_datasets:
        base_dir = config["path"]["In-the-Wild"]
        protocol_dir = f"datasets/labels/In-the-Wild.txt"
        dataloader = IntheWild(
            base_dir=base_dir,
            protocol_dir=protocol_dir,
            max_len=max_len,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        trainer.test(model, dataloader)
        cm_scores_file = scores_dir / "In-the-Wild.txt"
        shutil.copy2(Path(logger.log_dir) / "test_scores.txt", cm_scores_file)
        calculate_EER_IntheWild(
            cm_scores_file, protocol_dir, output_dir / "In-the-Wild.txt"
        )
