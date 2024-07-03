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
from gmm import Gmm

LVL_ERROR = 10
LVL_INFO = 5
LVL_DEBUG = 1

LOG = 1  # Will log all mesages with lvl greater than this
SAVE_LOG = True

WRITE_TENSORBOARD = False


class End2End2:
    def __init__(self, cfg: Config):
        self.cfg: Config = cfg
        self.storage_path: str = os.path.join(self.cfg.RESULTS_PATH, self.cfg.DATASET)
        
        pool = torch.nn.AvgPool2d(32, stride=16, padding=0)
        fake_image = torch.zeros(1, 1, self.cfg.INPUT_HEIGHT, self.cfg.INPUT_WIDTH)
        shape = pool(fake_image).shape
        self.output_shape = shape[-1]*shape[-2]
        self.counter = 0
        self.decision_mean = torch.zeros(256)
        self.gmm1 = Gmm()
        self.gmm2 = Gmm()
        
    def update_th(self,decision, is_pos):
        for d in decision[(is_pos==0)]:
            self.decision_mean[self.counter] = d.cpu()
            self.counter = (self.counter+1)%256
        return self.decision_mean.mean()+self.decision_mean.std()*2.0
        
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
        model_1 = self._get_model().to(device)
        model_2 = self._get_model().to(device)
        optimizer1 = self._get_optimizer(model_1)
        optimizer2 = self._get_optimizer(model_2)
        loss_seg = self._get_loss(True)

        tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_path) if WRITE_TENSORBOARD else None

        train_results = self._train_model(device, model_1, model_2, train_loader, loss_seg, optimizer1, optimizer2, validation_loader, tensorboard_writer)
        self._save_train_results(train_results)
        self._save_model(model_1, model_2)

        self.eval(model_1, model_2, device, self.cfg.SAVE_IMAGES, False, False)

        self._save_params()

    def eval(self, model_1, model_2, device, save_images, plot_seg, reload_final):
        self.reload_model(model_1, model_2, reload_final)
        test_loader = get_dataset("TEST", self.cfg)
        self.eval_model(device, model_1, model_2, test_loader, save_folder=self.outputs_path, save_images=save_images, is_validation=False, plot_seg=plot_seg)

    def training_iteration(self, data, device, model_1, model_2, criterion_seg, optimizer, tensorboard_writer, iter_index, epoch, gmm, warmup):
        images, seg_masks, seg_loss_masks, is_segmented, _, is_pos, train_masks = data

        batch_size = self.cfg.BATCH_SIZE
        memory_fit = self.cfg.MEMORY_FIT  # Not supported yet for >1

        num_subiters = int(batch_size / memory_fit)
        #
        total_correct = 0
        seg_correct = 0
        cheat = 0
        drop_pos = 0

        optimizer.zero_grad()

        total_loss = 0

        for sub_iter in range(num_subiters):
            images_ = images[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(device)
            seg_masks_ = seg_masks[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(device)
            train_masks_ = train_masks[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(device)
            seg_loss_masks_ = seg_loss_masks[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :, :, :].to(device)
            is_pos_ = is_pos[sub_iter * memory_fit:(sub_iter + 1) * memory_fit, :].to(device)

            if tensorboard_writer is not None and iter_index % 100 == 0:
                tensorboard_writer.add_image(f"{iter_index}/image", images_[0, :, :, :])
                tensorboard_writer.add_image(f"{iter_index}/seg_mask", seg_masks[0, :, :, :])
                tensorboard_writer.add_image(f"{iter_index}/seg_loss_mask", seg_loss_masks_[0, :, :, :])
            
            with torch.no_grad():
                decision, output_seg_mask = model_2(images_, seg_masks_)
                gmm.add_output(output_seg_mask, train_masks_, epoch)
                
            if not warmup:
                drop = gmm.eval(output_seg_mask, train_masks_, self.cfg.DROP_RATE)
                drop_pos += ((is_pos_>0.0).cpu().view(-1)*drop).sum().item()
            
                if drop.all():
                    continue
                else:
                    images_ = images_[drop.logical_not()]
                    seg_masks_ = seg_masks_[drop.logical_not()]
                    train_masks_ = train_masks_[drop.logical_not()]
                    seg_loss_masks_ = seg_loss_masks_[drop.logical_not()]
                    is_pos_ = is_pos_[drop.logical_not()]
                    
            decision, output_seg_mask = model_1(images_, seg_masks_)
            
            cheat += (seg_masks_ != train_masks_).long().max()
            
            if self.cfg.WEIGHTED_SEG_LOSS:
                #seg_loss_masks_ = (train_masks_>0.0).float()*10.0+1.0
                ad = (train_masks_>0.0).float()*self.cfg.AD
                loss_seg = torch.mean(criterion_seg(output_seg_mask-ad, train_masks_) * seg_loss_masks_)
            else:
                loss_seg = criterion_seg(output_seg_mask, train_masks_)
            
            pos_th = self.update_th(decision, is_pos_)
            
            total_correct += ((decision > pos_th) == is_pos_).sum()
            seg_correct += ((output_seg_mask > 0.0) == train_masks_).float().mean(dim=(1,2,3)).sum()
            loss = loss_seg
            
            loss /= num_subiters
            total_loss += loss.item()

            loss.backward()

        # Backward and optimize
        optimizer.step()
        optimizer.zero_grad()

        return total_loss, total_correct, seg_correct, cheat, drop_pos

    def _train_model(self, device, model_1, model_2, train_loader, criterion_seg, optimizer1, optimizer2, validation_set, tensorboard_writer):
        losses = []
        validation_data = []
        max_validation = -1
        validation_step = self.cfg.VALIDATION_N_EPOCHS

        num_epochs = self.cfg.EPOCHS
        batch_per_epoch = len(train_loader)
        samples_per_epoch = len(train_loader) * self.cfg.BATCH_SIZE
        model_counter = 0
        
        for epoch in range(num_epochs):
            self.gmm1.new_epoch()
            self.gmm2.new_epoch()
            if epoch % 5 == 0:
                self._save_model(model_1, model_2, f"ep_{epoch:02}")
            
            if epoch == num_epochs - num_epochs//4:
                optimizer1 = self._get_optimizer(model_1, 0.1)
                optimizer2 = self._get_optimizer(model_2, 0.1)
            
            epoch_loss = 0
            epoch_correct = 0
            epoch_seg_correct = 0
            epoch_cheat = 0
            epoch_drop_pos = 0
            warmup = epoch < num_epochs//8
            
            from timeit import default_timer as timer

            time_acc = 0
            start = timer()
            for iter_index, (data) in enumerate(train_loader):
                start_1 = timer()
                if model_counter:
                    model_2.train()
                    model_1.eval()
                    result = self.training_iteration(
                                                    data, device, model_2, model_1,
                                                    criterion_seg,
                                                    optimizer2,
                                                    tensorboard_writer, 
                                                    (epoch * samples_per_epoch + iter_index), epoch, self.gmm1, warmup)
                else:
                    model_1.train()
                    model_2.eval()
                    result = self.training_iteration(
                                                    data, device, model_1, model_2,
                                                    criterion_seg,
                                                    optimizer1,
                                                    tensorboard_writer, 
                                                    (epoch * samples_per_epoch + iter_index), epoch, self.gmm2, warmup)
                
                                                
                curr_loss, correct, seg_correct, cheat, drop_pos = result
                end_1 = timer()
                time_acc = time_acc + (end_1 - start_1)
                
                epoch_loss += curr_loss

                epoch_correct += correct
                epoch_seg_correct += seg_correct
                epoch_cheat += cheat
                epoch_drop_pos += drop_pos
                model_counter = 1 - model_counter
            
            self.gmm1.train()
            self.gmm2.train()
            
            drop_th1 = self.gmm1.get_drop_th(self.cfg.DROP_RATE).item()
            drop_th2 = self.gmm2.get_drop_th(self.cfg.DROP_RATE).item()
            end = timer()

            epoch_loss = epoch_loss / batch_per_epoch
            losses.append((epoch_loss, epoch))

            self._log(
                f"Epoch {epoch + 1}/{num_epochs} ==> avg_loss={epoch_loss:.5f}, correct={epoch_correct}/{samples_per_epoch}, seg_correct={epoch_seg_correct:.3f}, cheat={epoch_cheat:.0f}, drop_pos={epoch_drop_pos:.0f}, drop_th={drop_th1:.3f}/{drop_th2:.3f} in {end - start:.2f}s/epoch (fwd/bck in {time_acc:.2f}s/epoch)")

            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("Loss/Train/segmentation", epoch_loss_seg, epoch)
                tensorboard_writer.add_scalar("Loss/Train/classification", epoch_loss_dec, epoch)
                tensorboard_writer.add_scalar("Loss/Train/joined", epoch_loss, epoch)
                tensorboard_writer.add_scalar("Accuracy/Train/", epoch_correct / samples_per_epoch, epoch)

            if self.cfg.VALIDATE and (epoch % validation_step == 0 or epoch == num_epochs - 1):
                validation_aauc, validation_ap, validation_accuracy = self.eval_model(device, model_1, model_2, validation_set, None, False, True, False)
                validation_data.append((validation_aauc, validation_ap, epoch))

                if validation_ap > max_validation:
                    max_validation = validation_ap
                    self._save_model(model_1, model_2, "best_state_dict")

                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar("Accuracy/Validation/", validation_accuracy, epoch)

        return losses, validation_data

    def eval_model(self, device, model_1, model_2, eval_loader, save_folder, save_images, is_validation, plot_seg):
        model_1.eval()
        model_2.eval()

        dsize = self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT

        res = []
        predictions, ground_truths = [], []
        diff = []
        
        for data_point in eval_loader:
            image, human_mask, seg_loss_mask, _, sample_name, is_pos, seg_mask = data_point
            image, human_mask = image.to(device), human_mask.to(device)
            is_pos = is_pos.item()
            prediction1, pred_seg1 = model_1(image, human_mask)
            pred_seg1 = nn.Sigmoid()(pred_seg1)
            prediction2, pred_seg2 = model_2(image, human_mask)
            pred_seg2 = nn.Sigmoid()(pred_seg2)
            
            pred_seg = 0.5*pred_seg1 + 0.5*pred_seg2
            prediction = 0.5*prediction1 + 0.5*prediction2
            
            prediction = prediction.item()
            image = image.detach().cpu().numpy()
            pred_seg = pred_seg.detach().cpu().numpy()
            seg_mask = seg_mask.detach().cpu().numpy()
            human_mask = human_mask.detach().cpu().numpy()
            diff.append((seg_mask!=human_mask).sum())

            predictions.append(prediction)
            ground_truths.append(is_pos)
            res.append((prediction, None, None, is_pos, sample_name[0]))
            if not is_validation:
                if save_images:
                    image = cv2.resize(np.transpose(image[0, :, :, :], (1, 2, 0)), dsize)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    #pred_seg = cv2.resize(pred_seg[0, 0, :, :], dsize) if len(pred_seg.shape) == 4 else cv2.resize(pred_seg[0, :, :], dsize)
                    #seg_mask = cv2.resize(seg_mask[0, 0, :, :], dsize)
                    pred_seg = pred_seg[0, 0, :, :]
                    seg_mask = seg_mask[0, 0, :, :]
                    human_mask = human_mask[0, 0, :, :]
                    utils.plot_sample(sample_name[0], image, pred_seg, seg_mask, human_mask, save_folder, decision=prediction, plot_seg=plot_seg)
        
        diff = torch.tensor(np.array(diff)).view(-1)
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
            utils.evaluate_metrics(res, self.run_path, self.run_name)

    def reload_model(self, model_1, model_2, load_final=False):
        if self.cfg.USE_BEST_MODEL:
            path = os.path.join(self.model_path, "best_state_dict_1.pth")
            model_1.load_state_dict(torch.load(path))
            self._log(f"Loading model state from {path}")
            path = os.path.join(self.model_path, "best_state_dict_2.pth")
            model_2.load_state_dict(torch.load(path))
            self._log(f"Loading model state from {path}")
        elif load_final:
            path = os.path.join(self.model_path, "final_state_dict_1.pth")
            model_1.load_state_dict(torch.load(path))
            self._log(f"Loading model state from {path}")
            path = os.path.join(self.model_path, "final_state_dict_2.pth")
            model_2.load_state_dict(torch.load(path))
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

    def _save_model(self, model_1, model_2, name="final_state_dict"):
        name1 = f"{name}_1.pth"
        name2 = f"{name}_2.pth"
        
        output_name = os.path.join(self.model_path, name1)
        self._log(f"Saving current model state to {output_name}")
        if os.path.exists(output_name):
            os.remove(output_name)

        torch.save(model_1.state_dict(), output_name)
        
        output_name = os.path.join(self.model_path, name2)
        self._log(f"Saving current model state to {output_name}")
        if os.path.exists(output_name):
            os.remove(output_name)

        torch.save(model_2.state_dict(), output_name)

    def _get_optimizer(self, model, decay=1.0):
        return torch.optim.SGD(model.parameters(), self.cfg.LEARNING_RATE*decay, weight_decay=1e-4)

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
        seg_net = SegDecNet(self._get_device(), self.cfg.INPUT_WIDTH, self.cfg.INPUT_HEIGHT, self.cfg.INPUT_CHANNELS, self.output_shape, self.cfg.TT)
        return seg_net

    def print_run_params(self):
        for l in sorted(map(lambda e: e[0] + ":" + str(e[1]) + "\n", self.cfg.get_as_dict().items())):
            k, v = l.split(":")
            self._log(f"{k:25s} : {str(v.strip())}")
