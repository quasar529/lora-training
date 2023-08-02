import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import wandb
import random
import math
from utils.util import epoch_time, count_parameters, initialize_weights
import time


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        config,
        device,
        data_loader,
        name,
        valid_data_loader=None,
        lr_scheduler=None,
        len_epoch=None,
    ):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker("loss", *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker("loss", *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        self.name = name
        wandb.init(
            # set the wandb project where this run will be logged
            project="lora-training",
            # track hyperparameters and run metadata
            config={
                "learning_rate": 0.0005,
                "epochs": 100,
            },
        )
        wandb.run.name = f"{self.name}"

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        epoch_loss = 0
        # 전체 학습 데이터를 확인하며

        # output: [배치 크기, trg_len - 1, output_dim]
        # trg: [배치 크기, trg_len]

        # output: [배치 크기 * trg_len - 1, output_dim]
        # trg: [배치 크기 * trg len - 1]

        for i, batch in enumerate(self.data_loader):
            src = batch.src
            trg = batch.trg
            self.optimizer.zero_grad()
            # 출력 단어의 마지막 인덱스(<eos>)는 제외
            # 입력을 할 때는 <sos>부터 시작하도록 처리
            output, _ = self.model(src, trg[:, :-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            # 출력 단어의 인덱스 0(<sos>)은 제외
            trg = trg[:, 1:].contiguous().view(-1)

            # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
            loss = self.criterion(output, trg)
            loss.backward()  # 기울기(gradient) 계산

            # 기울기(gradient) clipping 진행
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

            # 파라미터 업데이트
            self.optimizer.step()
            epoch_loss += loss.item()

            ###
            # data, target = data.to(self.device), target.to(self.device)

            # self.optimizer.zero_grad()
            # output = self.model(data)
            # loss = self.criterion(output, target)
            # loss.backward()
            # self.optimizer.step()
            # epoch_loss += loss.item()

            self.writer.set_step((epoch - 1) * self.len_epoch + i)
            self.train_metrics.update("loss", loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, trg))

            if i % self.log_step == 0:
                self.logger.debug("Train Epoch: {} {} Loss: {:.6f}".format(epoch, self._progress(i), loss.item()))
                # self.writer.add_image("input", make_grid(src.cpu(), nrow=8, normalize=True))

            if i == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation:
            val_log, _ = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log, epoch_loss / len(self.data_loader)

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        epoch_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(self.valid_data_loader):
                src = batch.src
                trg = batch.trg
                # 출력 단어의 마지막 인덱스(<eos>)는 제외
                # 입력을 할 때는 <sos>부터 시작하도록 처리
                output, _ = self.model(src, trg[:, :-1])
                # output: [배치 크기, trg_len - 1, output_dim]
                # trg: [배치 크기, trg_len]

                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                # 출력 단어의 인덱스 0(<sos>)은 제외
                trg = trg[:, 1:].contiguous().view(-1)

                # output: [배치 크기 * trg_len - 1, output_dim]
                # trg: [배치 크기 * trg len - 1]

                # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
                loss = self.criterion(output, trg)

                # 전체 손실 값 계산
                epoch_loss += loss.item()
                # for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                #     data, target = data.to(self.device), target.to(self.device)

                #     output = self.model(data)
                #     loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + i, "valid")
                self.valid_metrics.update("loss", loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, trg))
            #     self.writer.add_image("input", make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins="auto")
        return self.valid_metrics.result(), epoch_loss / len(self.valid_data_loader)

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        best_valid_loss = float("inf")
        for epoch in range(self.start_epoch, self.epochs + 1):
            start_time = time.time()
            # result = self._train_epoch(epoch).values()
            # valid_result = self._valid_epoch(epoch).values()
            result, train_loss = self._train_epoch(epoch)
            _, valid_loss = self._valid_epoch(epoch)
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            wandb.log(
                {
                    "Time": f"{epoch_mins}m {epoch_secs}s",
                    "Train loss": train_loss,
                    "Train PPL": math.exp(train_loss),
                    "Valid Loss": valid_loss,
                    "Valid PPL": math.exp(valid_loss),
                }
            )
            print(f"Epoch: {epoch:02} | Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}")
            print(f"\tValidation Loss: {valid_loss:.3f} | Validation PPL: {math.exp(valid_loss):.3f}")
            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info("    {:15s}: {}".format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != "off":
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == "min" and log[self.mnt_metric] <= self.mnt_best) or (
                        self.mnt_mode == "max" and log[self.mnt_metric] >= self.mnt_best
                    )
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. "
                        "Model performance monitoring is disabled.".format(self.mnt_metric)
                    )
                    self.mnt_mode = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.early_stop)
                    )
                    break

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                # if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
