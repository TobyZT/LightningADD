import torch
import lightning as L

from models.RawNet2 import RawNet
from utils import *


class LightningADDTrainer(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = RawNet(
            {
                "nb_samp": 64600,
                "first_conv": 1024,
                "in_channels": 1,
                "filts": [20, [20, 20], [20, 128], [128, 128]],
                "blocks": [2, 4],
                "nb_fc_node": 1024,
                "gru_node": 1024,
                "nb_gru_layer": 3,
                "nb_classes": 2,
            }
        )

        weight = torch.FloatTensor([0.1, 0.9])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight)

    def configure_optimizers(self):
        lr = float(self.config["optimizer"]["lr"])
        weight_decay = float(self.config["optimizer"]["weight_decay"])
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        x, y, *_ = train_batch

        out = self.model(x)
        loss = self.criterion(out, y)
        self.log("train_loss", loss, sync_dist=True)

        return loss

    def on_validation_epoch_start(self):
        self.utt_ids = []
        self.eval_scores = []

    def validation_step(self, val_batch, batch_idx):
        x, y, utt_ids = val_batch
        out = self.model(x)
        loss = self.criterion(out, y)
        self.utt_ids.extend(utt_ids)
        self.eval_scores.extend(out[:, 1].cpu().numpy())
        return loss

    def on_validation_epoch_end(self):
        eval_scores_path = self.logger.log_dir + "/eval_scores.txt"
        cm_path = self.trainer.datamodule.dev_cm
        produce_evaluation_file(
            self.utt_ids, self.eval_scores, cm_path, eval_scores_path
        )
        asv_scores_path = self.trainer.datamodule.dev_asv_scores
        output_path = self.logger.log_dir + "/eval_output.txt"
        eer, min_tDCF = calculate_tDCF_EER_19LA(
            eval_scores_path, asv_scores_path, output_path
        )
        self.log("eer", eer, sync_dist=True)
        self.eval_scores.clear()

    def on_test_epoch_start(self):
        self.utt_ids = []
        self.eval_scores = []

    def test_step(self, test_batch, batch_idx):
        if len(test_batch) == 3:
            x, _, utt_ids = test_batch
        else:
            x, utt_ids = test_batch
        out = self.model(x)
        self.utt_ids.extend(utt_ids)
        self.eval_scores.extend(out[:, 1].cpu().numpy())

    def on_test_epoch_end(self):
        test_scores_path = self.logger.log_dir + "/test_scores.txt"
        cm_path = getattr(self.trainer.datamodule, "eval_cm", None)
        produce_evaluation_file(
            self.utt_ids, self.eval_scores, cm_path, test_scores_path
        )
        self.eval_scores.clear()
