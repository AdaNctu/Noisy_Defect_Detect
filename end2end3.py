import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from models import SegDecNet
import numpy as np
import os
from torch import nn as nn
import torch
import utils
import pandas as pd
from data.dataset_catalog import get_dataset
import random
import cv2
from config import Config
from torch.utils.tensorboard import SummaryWriter
from add_noise import *
from ADL import ADL

LVL_ERROR = 10
LVL_INFO = 5
LVL_DEBUG = 1

LOG = 1  # Will log all mesages with lvl greater than this
SAVE_LOG = True

WRITE_TENSORBOARD = False


class End2End3:
    def __init__(self, cfg: Config):
        self.cfg: Config = cfg
        self.storage_path: str = os.path.join(self.cfg.RESULTS_PATH, self.cfg.DATASET)
        self.ADL = ADL()

    def _log(self, message, lvl=LVL_INFO):
        n_msg = f"{self.run_name} {message}"
        if lvl >= LOG:
            print(n_msg)

    def train(self):
        self._set_results_path()
        self._create_results_dirs()
        self.print_run_params()
        if self.cfg.REPRODUCIBLE_RUN:
            self._log("Reproducible run, fixing all seeds to:1337", LVL_DEBUG)
            np.random.seed(1337)
            torch.manual_seed(1337)
            random.seed(1337)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        train_loader = get_dataset("TRAIN", self.cfg)
        validation_loader = get_dataset("VAL", self.cfg)
        
        device = self._get_device()
        model = self._get_model().to(device)
        optimizer = self._get_optimizer(model)
        loss_seg = self._get_loss(True)

        tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_path) if WRITE_TENSORBOARD else None

        train_results = self._train_model(device, model, train_loader, loss_seg, optimizer, validation_loader, tensorboard_writer)
        self._save_train_results(train_results)
        self._save_model(model)

        self.eval(model, device, self.cfg.SAVE_IMAGES, False, False, validation_loader)

        self._save_params()

    def eval(self, model, device, save_images, plot_seg, reload_final, test_loader):
        self.reload_model(model, reload_final)
        self.eval_model(device, model, test_loader, save_folder=self.outputs_path, save_images=save_images, is_validation=False, plot_seg=plot_seg)
    
    def training_ADL(self, data, device, model):
        images, seg_masks, seg_loss_masks, is_segmented, _, is_pos, train_masks, index = data
        batch_size = self.cfg.BATCH_SIZE
        memory_fit = self.cfg.MEMORY_FIT  # Not supported yet for >1

        num_subiters = int(batch_size / memory_fit)
        
        with torch.no_grad():
            _, output_seg_mask = model(images.to(device), seg_masks.to(device))
            self.ADL.add_output(output_seg_mask, seg_masks, index)
        
    def training_iteration(self, data, device, model, criterion_seg, optimizer, tensorboard_writer, iter_index):
        images, seg_masks, seg_loss_masks, is_segmented, _, is_pos, train_masks, index = data

        batch_size = self.cfg.BATCH_SIZE
        memory_fit = self.cfg.MEMORY_FIT  # Not supported yet for >1

        num_subiters = int(batch_size / memory_fit)

        optimizer.zero_grad()

        total_loss = 0

        for sub_iter in range(num_subiters):
            images_ = images[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(device)
            seg_masks_ = seg_masks[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(device)
            train_masks_ = seg_masks[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(device)
            seg_loss_masks_ = seg_loss_masks[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(device)
            is_pos_ = is_pos[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :].to(device)
            index_ = index[sub_iter * memory_fit:(sub_iter + 1) * memory_fit]
            ADL_mask_ = self.ADL.get_mask(index_).to(device)

            if tensorboard_writer is not None and iter_index % 100 == 0:
                tensorboard_writer.add_image(f"{iter_index}/image", images_[0, :, :, :])
                tensorboard_writer.add_image(f"{iter_index}/seg_mask", seg_masks[0, :, :, :])
                tensorboard_writer.add_image(f"{iter_index}/seg_loss_mask", seg_loss_masks_[0, :, :, :])
            
            decision, output_seg_mask = model(images_, seg_masks_)
            
            train_masks_smooth = 0.99*train_masks_ + 0.01*(1.0-train_masks_)
            if self.cfg.WEIGHTED_SEG_LOSS:
                loss_seg = torch.mean(criterion_seg(output_seg_mask, train_masks_smooth) * seg_loss_masks_ * ADL_mask_)
            else:
                loss_seg = criterion_seg(output_seg_mask, train_masks_smooth)
            
            loss = loss_seg
            
            loss /= num_subiters
            total_loss += loss.item()

            loss.backward()

        # Backward and optimize
        optimizer.step()
        optimizer.zero_grad()

        return total_loss

    def _train_model(self, device, model, train_loader, criterion_seg, optimizer, validation_set, tensorboard_writer):
        losses = []
        validation_data = []
        max_validation = -1
        validation_step = self.cfg.VALIDATION_N_EPOCHS

        num_epochs = self.cfg.EPOCHS
        batch_per_epoch = len(train_loader)
        samples_per_epoch = len(train_loader) * self.cfg.BATCH_SIZE
        num_sample, num_noise, label_shape = train_loader.dataset.get_ADL()

        for epoch in range(num_epochs):
            if epoch % 5 == 0:
                self._save_model(model, f"ep_{epoch:02}.pth")
            
            if epoch == num_epochs - num_epochs//4:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr']*0.1
            epoch_loss = 0
            from timeit import default_timer as timer

            time_acc = 0
            start = timer()
            
            self.ADL.new_epoch(num_sample, label_shape)
            
            drop_num = (epoch+1)/num_epochs
            drop_num = max(drop_num, 0.1)
            drop_num = min(drop_num, 0.5)
            drop_num = int((0.6-drop_num)*num_noise)
            
            model.eval()
            for iter_index, (data) in enumerate(train_loader):
                self.training_ADL(data, device, model)
            self.ADL.sample_selection(drop_num)
                
            model.train()
            for iter_index, (data) in enumerate(train_loader):
                start_1 = timer()
                result = self.training_iteration(
                                                data, device, model,
                                                criterion_seg,
                                                optimizer,
                                                tensorboard_writer, 
                                                (epoch * samples_per_epoch + iter_index))
                                                
                curr_loss = result
                end_1 = timer()
                time_acc = time_acc + (end_1 - start_1)
                
                epoch_loss += curr_loss

            end = timer()

            epoch_loss = epoch_loss / batch_per_epoch
            losses.append((epoch_loss, epoch))

            self._log(
                f"Epoch {epoch + 1}/{num_epochs} ==> avg_loss={epoch_loss:.5f} in {end - start:.2f}s/epoch (fwd/bck in {time_acc:.2f}s/epoch)")

            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("Loss/Train/segmentation", epoch_loss_seg, epoch)
                tensorboard_writer.add_scalar("Loss/Train/classification", epoch_loss_dec, epoch)
                tensorboard_writer.add_scalar("Loss/Train/joined", epoch_loss, epoch)
                tensorboard_writer.add_scalar("Accuracy/Train/", epoch_correct / samples_per_epoch, epoch)

            if self.cfg.VALIDATE and (epoch % validation_step == 0 or epoch == num_epochs - 1):
                validation_aauc, validation_ap, validation_accuracy = self.eval_model(device, model, validation_set, None, False, True, False)
                validation_data.append((validation_aauc, validation_ap, epoch))

                if validation_ap > max_validation:
                    max_validation = validation_ap
                    self._save_model(model, "best_state_dict.pth")

                model.train()
                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar("Accuracy/Validation/", validation_accuracy, epoch)

        return losses, validation_data

    def eval_model(self, device, model, eval_loader, save_folder, save_images, is_validation, plot_seg):
        model.eval()

        dsize = self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT

        res = []
        predictions, ground_truths = [], []
        diff = []
        
        for data_point in eval_loader:
            image, human_mask, seg_loss_mask, _, sample_name, is_pos, seg_mask, original_image = data_point
            image, human_mask = image.to(device), human_mask.to(device)
            is_pos = is_pos.item()
            prediction, pred_seg = model(image, human_mask)
            pred_seg = nn.Sigmoid()(pred_seg)

            prediction = prediction.item()
            image = image.detach().cpu().numpy()
            pred_seg = pred_seg.detach().cpu().numpy()
            seg_mask = seg_mask.detach().cpu().numpy()
            human_mask = human_mask.detach().cpu().numpy()
            original_image = original_image.numpy()
            diff.append((seg_mask!=human_mask).sum().item())

            predictions.append(prediction)
            ground_truths.append(is_pos)
            res.append((prediction, None, None, is_pos, sample_name[0]))
            
            
            if not is_validation:
                if save_images:
                    image = cv2.resize(np.transpose(original_image[0, :, :, :], (1, 2, 0)), dsize)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    #pred_seg = cv2.resize(pred_seg[0, 0, :, :], dsize) if len(pred_seg.shape) == 4 else cv2.resize(pred_seg[0, :, :], dsize)
                    #seg_mask = cv2.resize(seg_mask[0, 0, :, :], dsize)
                    pred_seg = pred_seg[0, 0, :, :]
                    seg_mask = seg_mask[0, 0, :, :]
                    human_mask = human_mask[0, 0, :, :]
                    utils.plot_sample(sample_name[0], image, pred_seg, seg_mask, human_mask, save_folder, decision=prediction, plot_seg=plot_seg)
        
        diff = torch.tensor(diff).view(-1)
        disagree = torch.tensor(predictions).view(-1)
        _, dff_order = diff.sort()
        _, dis_order = disagree.sort()
        cheat = (diff[dff_order]*torch.arange(1,len(diff)+1)).sum()
        random = diff.sum()*(len(diff)+1)*0.5
        query = (diff[dis_order]*torch.arange(1,len(diff)+1)).sum()
        AAUC = (query-random)/(cheat-random)
        
        if is_validation:
            metrics = utils.get_metrics(np.array(ground_truths), np.array(predictions))
            FP, FN, TP, TN = list(map(sum, [metrics["FP"], metrics["FN"], metrics["TP"], metrics["TN"]]))
            self._log(f"VALIDATION || AAUC={AAUC:f}, AUC={metrics['AUC']:f}, and AP={metrics['AP']:f}, with best thr={metrics['best_thr']:f} "
                      f"at f-measure={metrics['best_f_measure']:.3f} and FP={FP:d}, FN={FN:d}, TOTAL SAMPLES={FP + FN + TP + TN:d}")

            return AAUC, metrics["AP"], metrics["accuracy"]
        else:
            utils.evaluate_metrics(AAUC, res, self.run_path, self.run_name)

    def reload_model(self, model, load_final=False):
        if self.cfg.USE_BEST_MODEL:
            path = os.path.join(self.model_path, "best_state_dict.pth")
            model.load_state_dict(torch.load(path))
            self._log(f"Loading model state from {path}")
        elif load_final:
            path = os.path.join(self.model_path, "final_state_dict.pth")
            model.load_state_dict(torch.load(path))
            self._log(f"Loading model state from {path}")
        else:
            self._log("Keeping same model state")

    def _save_params(self):
        params = self.cfg.get_as_dict()
        params_lines = sorted(map(lambda e: e[0] + ":" + str(e[1]) + "\n", params.items()))
        fname = os.path.join(self.run_path, "run_params.txt")
        with open(fname, "w+") as f:
            f.writelines(params_lines)

    def _save_train_results(self, results):
        losses, validation_data = results
        l, le = map(list, zip(*losses))
        plt.plot(le, l, label="Loss", color="red")
        plt.ylim(bottom=0)
        plt.grid()
        plt.xlabel("Epochs")
        plt.legend()
        if self.cfg.VALIDATE:
            aauc, v, ve = map(list, zip(*validation_data))
            plt.twinx()
            plt.plot(ve, v, label="Validation AP", color="Green")
            plt.plot(ve,aauc, label="Validation AAUC", color="Blue")
            plt.ylim((0, 1))
        plt.legend()
        plt.savefig(os.path.join(self.run_path, "loss_val"), dpi=200)

        df_loss = pd.DataFrame(data={"loss": l, "epoch": le})
        df_loss.to_csv(os.path.join(self.run_path, "losses.csv"), index=False)

        if self.cfg.VALIDATE:
            df_loss = pd.DataFrame(data={"loss": l, "epoch": le})
            df_loss.to_csv(os.path.join(self.run_path, "losses.csv"), index=False)

    def _save_model(self, model, name="final_state_dict.pth"):
        output_name = os.path.join(self.model_path, name)
        self._log(f"Saving current model state to {output_name}")
        if os.path.exists(output_name):
            os.remove(output_name)

        torch.save(model.state_dict(), output_name)

    def _get_optimizer(self, model):
        return torch.optim.SGD(model.parameters(), self.cfg.LEARNING_RATE, weight_decay=1e-4, momentum=0.9)

    def _get_loss(self, is_seg):
        reduction = "none" if self.cfg.WEIGHTED_SEG_LOSS and is_seg else "mean"
        return nn.BCEWithLogitsLoss(reduction=reduction).to(self._get_device())

    def _get_device(self):
        return f"cuda:{self.cfg.GPU}"

    def _set_results_path(self):
        self.run_name = f"{self.cfg.RUN_NAME}_FOLD_{self.cfg.FOLD}" if self.cfg.DATASET in ["KSDD", "DAGM"] else self.cfg.RUN_NAME

        results_path = os.path.join(self.cfg.RESULTS_PATH, self.cfg.DATASET)
        self.tensorboard_path = os.path.join(results_path, "tensorboard", self.run_name)

        run_path = os.path.join(results_path, self.cfg.RUN_NAME)
        if self.cfg.DATASET in ["KSDD", "DAGM"]:
            run_path = os.path.join(run_path, f"FOLD_{self.cfg.FOLD}")

        self._log(f"Executing run with path {run_path}")

        self.run_path = run_path
        self.model_path = os.path.join(run_path, "models")
        self.outputs_path = os.path.join(run_path, "test_outputs")

    def _create_results_dirs(self):
        list(map(utils.create_folder, [self.run_path, self.model_path, self.outputs_path, ]))

    def _get_model(self):
        seg_net = SegDecNet(self._get_device(), self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT, self.cfg.INPUT_CHANNELS)
        return seg_net

    def print_run_params(self):
        for l in sorted(map(lambda e: e[0] + ":" + str(e[1]) + "\n", self.cfg.get_as_dict().items())):
            k, v = l.split(":")
            self._log(f"{k:25s} : {str(v.strip())}")
