import torch, torchmetrics, tqdm, copy, time
from utils import (
    LinearLR,
    unlearn_func,
    ssd_tuning,
    distill_kl_loss,
    compute_accuracy,
    assd_tuning,
    alfssd_tuning,
)
from torch.cuda.amp import autocast
import numpy as np
from torch.cuda.amp import GradScaler
from os import makedirs
from os.path import exists
from torch.nn import functional as F
import itertools
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os

device = torch.device("cuda" if torch.cuda.is_available() else "mps")  # MAC
PYTORCH_MPS_HIGH_WATERMARK_RATIO = 0.0


ITERATIVE_SEARCH = True
STEP_MULT = 1.1
MAX_TRY = 500  # easily enough
MIN_ACC = "OVERRIDEN"

# Specify the file names for the importances
file_name_1 = "original_importances.pkl"
file_name_2 = "sample_importances.pkl"


# FYI: We left some additional experiment code chunks in the code for people to potentially build upon/experiment with


class Naive:
    def __init__(self, opt, model, prenet=None):
        self.opt = opt
        self.curr_step, self.best_top1 = 0, 0
        self.best_model = None
        self.set_model(model, prenet)
        self.save_files = {"train_top1": [], "val_top1": [], "train_time_taken": 0}
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.opt.max_lr,
            momentum=0.9,
            weight_decay=self.opt.wd,
        )
        self.scheduler = LinearLR(
            self.optimizer,
            T=self.opt.train_iters * 1.25,
            warmup_epochs=self.opt.train_iters // 100,
        )  # Spend 1% time in warmup, and stop 66% of the way through training
        # self.top1 = torchmetrics.Accuracy(task="multiclass", num_classes=self.opt.num_classes).cuda()
        self.top1 = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.opt.num_classes
        ).to(
            device
        )  # MAC
        self.scaler = GradScaler()

    def set_model(self, model, prenet=None):
        self.prenet = None
        self.model = model
        self.model.to(device)  # MAC
        # self.model.cuda()

    def forward_pass(self, images, target, infgt):
        if self.prenet is not None:
            with torch.no_grad():
                feats = self.prenet(images)
            output = self.model(feats)
        else:
            output = self.model(images)
        loss = F.cross_entropy(output, target)
        self.top1(output, target)
        return loss

    def train_one_epoch(self, loader):
        self.model.train()
        self.top1.reset()

        for images, target, infgt in tqdm.tqdm(loader):
            # images, target, infgt = images.cuda(), target.cuda(), infgt.cuda()
            images, target, infgt = (
                images.to(device),
                target.to(device),
                infgt.to(device),
            )
            with autocast():
                self.optimizer.zero_grad()
                loss = self.forward_pass(images, target, infgt)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.curr_step += 1
                if self.curr_step > self.opt.train_iters:
                    break

        top1 = self.top1.compute().item()
        self.top1.reset()
        self.save_files["train_top1"].append(top1)
        print(f"Step: {self.curr_step} Train Top1: {top1:.3f}")
        return

    def eval(self, loader, save_model=True, save_preds=False):
        self.model.eval()
        self.top1.reset()

        if save_preds:
            preds, targets = [], []

        with torch.no_grad():
            for images, target in tqdm.tqdm(loader):
                with autocast():
                    # images, target = images.cuda(), target.cuda()
                    images, target = images.to(device), target.to(device)  # MAC
                    output = (
                        self.model(images)
                        if self.prenet is None
                        else self.model(self.prenet(images))
                    )
                self.top1(output, target)
                if save_preds:
                    preds.append(output.cpu().numpy())
                    targets.append(target.cpu().numpy())

        top1 = self.top1.compute().item()
        self.top1.reset()
        if not save_preds:
            print(f"Step: {self.curr_step} Val Top1: {top1*100:.2f}")

        if save_model:
            self.save_files["val_top1"].append(top1)
            if top1 > self.best_top1:
                self.best_top1 = top1
                self.best_model = copy.deepcopy(self.model).cpu()

        self.model.train()
        if save_preds:
            preds = np.concatenate(preds, axis=0)
            targets = np.concatenate(targets, axis=0)
            return preds, targets
        return

    def unlearn(self, train_loader, test_loader, eval_loaders=None):
        while self.curr_step < self.opt.train_iters:
            time_start = time.process_time()
            self.train_one_epoch(loader=train_loader)
            self.eval(test_loader)
            self.save_files["train_time_taken"] += time.process_time() - time_start
        return

    def get_save_prefix(self):
        self.unlearn_file_prefix = (
            self.opt.pretrain_file_prefix
            + "/"
            + self.opt.unlearn_method
            + "_"
            + self.opt.exp_name
        )
        if self.opt.unlearn_method != "Naive":
            self.unlearn_file_prefix = (
                self.opt.pretrain_file_prefix
                + "/"
                + str(self.opt.deletion_size)
                + "_"
                + self.opt.unlearn_method
                + "_"
                + self.opt.exp_name
            )
        return

    def compute_and_save_results(
        self,
        train_test_loader,
        test_loader,
        adversarial_train_loader,
        adversarial_test_loader,
    ):
        print("==> Compute And Save Results In Progress")

        self.get_save_prefix()
        print(self.unlearn_file_prefix)
        if not exists(self.unlearn_file_prefix):
            makedirs(self.unlearn_file_prefix)

        torch.save(
            self.best_model.state_dict(), self.unlearn_file_prefix + "/model.pth"
        )
        np.save(
            self.unlearn_file_prefix + "/train_top1.npy", self.save_files["train_top1"]
        )
        np.save(self.unlearn_file_prefix + "/val_top1.npy", self.save_files["val_top1"])
        np.save(
            self.unlearn_file_prefix + "/unlearn_time.npy",
            self.save_files["train_time_taken"],
        )
        # self.model = self.best_model.cuda()
        self.model = self.best_model.to(device)  # MAC

        print(
            "==> Completed! Unlearning Time: [{0:.3f}]\t".format(
                self.save_files["train_time_taken"]
            )
        )

        for loader, name in [
            (train_test_loader, "train"),
            (test_loader, "test"),
            (adversarial_train_loader, "adv_train"),
            (adversarial_test_loader, "adv_test"),
        ]:
            if loader is not None:
                preds, targets = self.eval(loader=loader, save_preds=True)
                np.save(self.unlearn_file_prefix + "/preds_" + name + ".npy", preds)
                np.save(self.unlearn_file_prefix + "/targets" + name + ".npy", targets)
        return


class ApplyK(Naive):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)

    def set_model(self, model, prenet):
        prenet, model = self.divide_model(
            model, k=self.opt.k, model_name=self.opt.model
        )
        model = unlearn_func(
            model=model,
            method=self.opt.unlearn_method,
            factor=self.opt.factor,
            device=self.opt.device,
        )
        self.model = model
        self.prenet = prenet
        # self.model.cuda()
        self.model.to(device)
        if self.prenet is not None:
            # self.prenet.cuda().eval()
            self.prenet.to(device).eval()  # MAC

    def divide_model(self, model, k, model_name):
        if k == -1:  # -1 means retrain all layers
            net = model
            prenet = None
            return prenet, net

        if model_name == "resnet9":
            assert k in [1, 2, 4, 5, 7, 8]
            mapping = {1: 6, 2: 5, 4: 4, 5: 3, 7: 2, 8: 1}
            dividing_part = mapping[k]
            all_mods = [
                model.conv1,
                model.conv2,
                model.res1,
                model.conv3,
                model.res2,
                model.conv4,
                model.fc,
            ]
            prenet = torch.nn.Sequential(*all_mods[:dividing_part])
            net = torch.nn.Sequential(*all_mods[dividing_part:])

        elif model_name == "resnetwide28x10":
            assert k in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
            all_mods = [
                model.conv1,
                model.layer1,
                model.layer2,
                model.layer3,
                model.norm,
                model.fc,
            ]
            mapping = {1: 5, 9: 3, 17: 2, 25: 1}

            if k in mapping:
                intervention_point = mapping[k]
                prenet = torch.nn.Sequential(*all_mods[:intervention_point])
                net = torch.nn.Sequential(*all_mods[intervention_point:])
            else:
                vals = list(mapping.keys())
                for val in vals:
                    if val > k:
                        sel_idx = val
                        break
                layer = mapping[sel_idx]
                prenet_list = all_mods[:layer]
                prenet_additions = list(
                    all_mods[layer][: int(4 - (((k - 1) // 2) % 4))]
                )
                prenet = torch.nn.Sequential(*(prenet_list + prenet_additions))
                net_list = list(all_mods[layer][int(4 - (((k - 1) // 2) % 4)) :])
                net_additions = all_mods[layer + 1 :]
                net = torch.nn.Sequential(*(net_list + net_additions))

        elif model_name == "vitb16":
            assert k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            all_mods = [model.patch_embed, model.blocks, model.norm, model.head]
            mapping = {1: 3, 13: 1}

            if k in mapping:
                intervention_point = mapping[k]
                prenet = torch.nn.Sequential(*all_mods[:intervention_point])
                net = torch.nn.Sequential(*all_mods[intervention_point:])
            else:
                prenet = [model.patch_embed]
                k = 13 - k
                prenet += [model.blocks[:k]]
                prenet = torch.nn.Sequential(*prenet)
                net = [model.blocks[k:], model.norm, model.head]
                net = torch.nn.Sequential(*net)

        prenet.to(self.opt.device)
        net.to(self.opt.device)
        return prenet, net

    def get_save_prefix(self):
        self.unlearn_file_prefix = (
            self.opt.pretrain_file_prefix
            + "/"
            + str(self.opt.deletion_size)
            + "_"
            + self.opt.unlearn_method
            + "_"
            + self.opt.exp_name
        )
        self.unlearn_file_prefix += (
            "_" + str(self.opt.train_iters) + "_" + str(self.opt.k)
        )

        return


class Scrub(ApplyK):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)
        self.og_model = copy.deepcopy(model)
        # self.og_model.cuda().eval()
        self.og_model.to(device).eval()  # MAC

    def forward_pass(self, images, target, infgt):
        if self.prenet is not None:
            with torch.no_grad():
                feats = self.prenet(images)
            output = self.model(feats)
        else:
            output = self.model(images)

        with torch.no_grad():
            logit_t = self.og_model(images)

        loss = F.cross_entropy(output, target)
        loss += self.opt.alpha * distill_kl_loss(output, logit_t, self.opt.kd_T)

        if self.maximize:
            loss = -loss

        self.top1(output, target)
        return loss

    def unlearn(self, train_loader, test_loader, forget_loader, eval_loaders=None):
        self.maximize = False
        while self.curr_step < self.opt.train_iters:
            if self.curr_step < self.opt.msteps:
                self.maximize = True
                time_start = time.process_time()
                self.train_one_epoch(loader=forget_loader)
                self.save_files["train_time_taken"] += time.process_time() - time_start
                self.eval(loader=test_loader)

            self.maximize = False
            time_start = time.process_time()
            self.train_one_epoch(loader=train_loader)
            self.save_files["train_time_taken"] += time.process_time() - time_start
            self.eval(loader=test_loader)
        return

    def get_save_prefix(self):
        self.unlearn_file_prefix = (
            self.opt.pretrain_file_prefix
            + "/"
            + str(self.opt.deletion_size)
            + "_"
            + self.opt.unlearn_method
            + "_"
            + self.opt.exp_name
        )
        self.unlearn_file_prefix += (
            "_" + str(self.opt.train_iters) + "_" + str(self.opt.k)
        )
        self.unlearn_file_prefix += (
            "_"
            + str(self.opt.kd_T)
            + "_"
            + str(self.opt.alpha)
            + "_"
            + str(self.opt.msteps)
        )
        return


class BadT(ApplyK):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)
        self.og_model = copy.deepcopy(model)
        self.og_model.to(self.opt.device)
        self.og_model.eval()
        self.random_model = unlearn_func(model, "EU")
        self.random_model.eval()
        self.kltemp = 1

    def forward_pass(self, images, target, infgt):
        if self.prenet is not None:
            with torch.no_grad():
                feats = self.prenet(images)
            output = self.model(feats)
        else:
            output = self.model(images)

        full_teacher_logits = self.og_model(images)
        unlearn_teacher_logits = self.random_model(images)
        f_teacher_out = torch.nn.functional.softmax(
            full_teacher_logits / self.kltemp, dim=1
        )
        u_teacher_out = torch.nn.functional.softmax(
            unlearn_teacher_logits / self.kltemp, dim=1
        )
        labels = torch.unsqueeze(infgt, 1)
        overall_teacher_out = labels * u_teacher_out + (1 - labels) * f_teacher_out
        student_out = F.log_softmax(output / self.kltemp, dim=1)
        loss = F.kl_div(student_out, overall_teacher_out)

        self.top1(output, target)
        return loss

    def get_save_prefix(self):
        self.unlearn_file_prefix = (
            self.opt.pretrain_file_prefix
            + "/"
            + str(self.opt.deletion_size)
            + "_"
            + self.opt.unlearn_method
            + "_"
            + self.opt.exp_name
        )
        self.unlearn_file_prefix += (
            "_" + str(self.opt.train_iters) + "_" + str(self.opt.k)
        )
        return


class No_unlearning(ApplyK):
    # does nothing
    pass


class SSD(ApplyK):
    def __init__(self, opt, model, prenet=None):
        print("===> opt", opt)
        super().__init__(opt, model, prenet)

    def unlearn(self, train_loader, test_loader, forget_loader, eval_loaders=None):
        self.opt.train_iters = len(train_loader) + len(forget_loader)
        time_start = time.process_time()
        self.best_model = ssd_tuning(
            self.model,
            forget_loader,
            self.opt.SSDdampening,
            self.opt.SSDselectwt,
            train_loader,
            self.opt.device,
        )
        self.save_files["train_time_taken"] += time.process_time() - time_start
        return

    def get_save_prefix(self):
        self.unlearn_file_prefix = (
            self.opt.pretrain_file_prefix
            + "/"
            + str(self.opt.deletion_size)
            + "_"
            + self.opt.unlearn_method
            + "_"
            + self.opt.exp_name
        )
        self.unlearn_file_prefix += (
            "_" + str(self.opt.train_iters) + "_" + str(self.opt.k)
        )
        self.unlearn_file_prefix += (
            "_" + str(self.opt.SSDdampening) + "_" + str(self.opt.SSDselectwt)
        )
        return


class ASSD(ApplyK):
    def __init__(self, opt, model, prenet=None):
        print("===> opt", opt)
        super().__init__(opt, model, prenet)

    def get_acc(self, input_model, input_loader):
        # calculate the accuracy of model on the loader
        input_model.eval()
        input_model.to(device)
        inputl_model_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.opt.num_classes
        ).to(device)
        inputl_model_acc.reset()
        with torch.no_grad():
            for images, target, _ in input_loader:
                images, target = images.to(device), target.to(device)
                output = input_model(images)
                inputl_model_acc(output, target)
        inputl_model_acc = inputl_model_acc.compute().item()
        print("input_model_accuracy: ", inputl_model_acc)

        return inputl_model_acc

    def unlearn(
        self,
        train_loader,
        test_loader,
        forget_loader,
        eval_loaders=None,
        frac_dl=None,
        min_acc_val=None,
    ):
        time_start = time.process_time()
        if ITERATIVE_SEARCH:
            # ------------------------------------------------
            # start the frac iterations
            original_model = copy.deepcopy(
                self.model
            )  # do not forget to pass to device after reassigning

            max_tries = MAX_TRY
            min_acc = min_acc_val  # MIN_ACC
            # calculate the accuracy of the model on the train_loader
            org_model_acc = self.get_acc(original_model, forget_loader)

            # Remove the old files
            try:
                os.remove(file_name_1)
                os.remove(file_name_2)
            except:
                print("No previous importance files")

            for loop_i in range(max_tries):
                loop_start_t = time.process_time()
                self.best_model = assd_tuning(
                    self.model,
                    forget_loader,
                    None,
                    None,
                    train_loader,
                    self.opt.device,
                    frac_dl,
                )

                # calculate the accuracy of the model on the train_loader
                new_model_acc = self.get_acc(self.model, forget_loader)

                # check if minimum acc reached
                if new_model_acc <= min_acc * org_model_acc:
                    print("minimum accuracy reached")
                    break
                else:
                    frac_dl = STEP_MULT * frac_dl
                    self.model = copy.deepcopy(original_model)
                    print("retry with new frac_dl: ", frac_dl)
                loop_end_t = time.process_time() - loop_start_t
                print(loop_i, " loop time: ", loop_end_t)
            self.save_files["train_time_taken"] += time.process_time() - time_start
        else:
            self.opt.train_iters = len(train_loader) + len(forget_loader)
            time_start = time.process_time()
            self.best_model = assd_tuning(
                self.model,
                forget_loader,
                None,
                None,
                train_loader,
                self.opt.device,
                frac_dl,
            )
            self.save_files["train_time_taken"] += time.process_time() - time_start
        return

    def get_save_prefix(self):
        self.unlearn_file_prefix = (
            self.opt.pretrain_file_prefix
            + "/"
            + str(self.opt.deletion_size)
            + "_"
            + self.opt.unlearn_method
            + "_"
            + self.opt.exp_name
        )
        self.unlearn_file_prefix += (
            "_" + str(self.opt.train_iters) + "_" + str(self.opt.k)
        )
        self.unlearn_file_prefix += "_adaptive"
        return


# includes repair
class ASSDR(ApplyK):
    def __init__(self, opt, model, prenet=None):
        print("===> opt", opt)
        super().__init__(opt, model, prenet)

    def unlearn(
        self,
        train_loader,
        test_loader,
        forget_loader,
        eval_loaders=None,
        frac_dl=None,
        min_acc_val=None,
    ):
        self.opt.train_iters = len(train_loader) + len(forget_loader)
        time_start = time.process_time()
        self.best_model = assd_tuning(
            self.model,
            forget_loader,
            None,
            None,
            train_loader,
            self.opt.device,
            frac_dl,
        )

        # repair for one epoch
        for i in range(25):
            self.train_one_epoch(loader=train_loader)

        return

    def get_save_prefix(self):
        self.unlearn_file_prefix = (
            self.opt.pretrain_file_prefix
            + "/"
            + str(self.opt.deletion_size)
            + "_"
            + self.opt.unlearn_method
            + "_"
            + self.opt.exp_name
        )
        self.unlearn_file_prefix += (
            "_" + str(self.opt.train_iters) + "_" + str(self.opt.k)
        )
        self.unlearn_file_prefix += "_adaptiver"
        return


class ALFSSD(ApplyK):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)

    def get_acc(self, input_model, input_loader):
        # calculate the accuracy of model on the loader
        input_model.eval()
        input_model.to(device)
        inputl_model_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.opt.num_classes
        ).to(device)
        inputl_model_acc.reset()
        with torch.no_grad():
            for images, target, _ in input_loader:
                images, target = images.to(device), target.to(device)
                output = input_model(images)
                inputl_model_acc(output, target)
        inputl_model_acc = inputl_model_acc.compute().item()
        print("input_model_accuracy: ", inputl_model_acc)

        return inputl_model_acc

    def unlearn(
        self,
        train_loader,
        test_loader,
        forget_loader,
        eval_loaders=None,
        frac_dl=None,
        min_acc_val=None,
    ):
        print("--- Unlearning with ALFSSD ---")
        self.opt.train_iters = len(train_loader) + len(forget_loader)
        time_start = time.process_time()

        # ------------------------------------------------
        # start the frac iterations
        original_model = copy.deepcopy(
            self.model
        )  # do not forget to pass to device after reassigning

        max_tries = MAX_TRY
        min_acc = min_acc_val  # MIN_ACC
        # calculate the accuracy of the model on the train_loader
        org_model_acc = self.get_acc(original_model, forget_loader)

        # Remove the old files
        try:
            os.remove(file_name_1)
            os.remove(file_name_2)
        except:
            print("No previous importance files")

        for _ in range(max_tries):
            self.best_model = alfssd_tuning(
                self.model,
                forget_loader,
                None,
                None,
                train_loader,  # train_loader,
                self.opt.device,
                frac_dl,
                train_loader,
            )

            # calculate the accuracy of the model on the train_loader
            new_model_acc = self.get_acc(self.model, forget_loader)

            # check if minimum acc reached
            if new_model_acc <= min_acc * org_model_acc:
                print("minimum accuracy reached")
                break
            else:
                frac_dl = STEP_MULT * frac_dl
                self.model = copy.deepcopy(original_model)
                print("retry with new frac_dl: ", frac_dl)

        self.save_files["train_time_taken"] += time.process_time() - time_start

        return

    def get_save_prefix(self):
        self.unlearn_file_prefix = (
            self.opt.pretrain_file_prefix
            + "/"
            + str(self.opt.deletion_size)
            + "_"
            + self.opt.unlearn_method
            + "_"
            + self.opt.exp_name
        )
        self.unlearn_file_prefix += (
            "_" + str(self.opt.train_iters) + "_" + str(self.opt.k)
        )
        self.unlearn_file_prefix += "_adaptivelf"
        return


class XALFSSD(ApplyK):
    def __init__(self, opt, model, prenet=None):
        super().__init__(opt, model, prenet)

    def get_acc(self, input_model, input_loader):
        # calculate the accuracy of model on the loader
        input_model.eval()
        input_model.to(device)
        inputl_model_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.opt.num_classes
        ).to(device)
        inputl_model_acc.reset()
        with torch.no_grad():
            for images, target, _ in input_loader:
                images, target = images.to(device), target.to(device)
                output = input_model(images)
                inputl_model_acc(output, target)
        inputl_model_acc = inputl_model_acc.compute().item()
        print("input_model_accuracy: ", inputl_model_acc)

        return inputl_model_acc

    def calculate_lipschitz_smoothness(self, model, sample, delta=0.01):
        # Ensure the model is in evaluation mode
        model.eval()

        # Convert the sample to a PyTorch tensor, if it's not already
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample)

        # Add an extra dimension to the sample if necessary
        if True:  # len(sample.shape) == 1:
            sample = sample.unsqueeze(0)

        # Get the model's output for the sample
        original_output = model(sample)

        # Create a perturbation
        # ensure consistency for replicability
        # torch.manual_seed(42)
        # torch.cuda.manual_seed_all(42)
        perturbation = torch.randn_like(sample) * delta

        # Add the perturbation to the sample
        perturbed_sample = sample + perturbation

        # Get the model's output for the perturbed sample
        perturbed_output = model(perturbed_sample)

        # Calculate the change in the model's output
        change_in_output = torch.norm(perturbed_output - original_output)

        # Calculate the Lipschitz constant
        lipschitz_constant = change_in_output / torch.norm(perturbation)

        return lipschitz_constant.item()

    def calculate_lipschitz_for_dataloader(self, dataloader, delta=0.01):
        # Initialize a list to store the Lipschitz constants
        lipschitz_constants = []

        # TODO: We could cluster for similar data first and only run over that to speed things up

        # Iterate over the DataLoader
        i = 0
        for inputs, _, _ in tqdm.tqdm(dataloader):
            inputs = inputs.to(device)
            # Calculate the Lipschitz constant for each sample in the batch
            for sample in inputs:

                lipschitz_constant = self.calculate_lipschitz_smoothness(
                    self.model, sample, delta
                )
                lipschitz_constants.append(lipschitz_constant)

        return lipschitz_constants

    def unlearn(
        self,
        train_loader,
        test_loader,
        forget_loader,
        eval_loaders=None,
        frac_dl=None,
        min_acc_val=None,
    ):
        print("--- Unlearning with XALFSSD ---")
        self.opt.train_iters = len(train_loader) + len(forget_loader)
        time_start = time.process_time()

        # LF pre-filter yes/no
        if False:
            # Lipschitz smoothness pre-filter
            # ------------------------------------------------

            # calculate the lipschitz smoothness of the model output for the train_loader and forget_loader
            print(
                "Starting to calculate the lipschitz smoothness of the model output for the train_loader and forget_loader"
            )
            ls_values_forget = self.calculate_lipschitz_for_dataloader(
                forget_loader, delta=0.01
            )
            # print mean of the lipschitz smoothness of the model output for the forget_loader
            print("ls_values forget mean", np.mean(ls_values_forget))
            print("ls_values forget median", np.median(ls_values_forget))
            print("-----")
            ls_values_retain = self.calculate_lipschitz_for_dataloader(
                train_loader, delta=0.01
            )
            # print mean of the lipschitz smoothness of the model output for the train_loader
            print("ls_values retain mean", np.mean(ls_values_retain))
            print("ls_values retain median", np.median(ls_values_retain))

            # remove all samples from the train_loader where ls_values_retain >= ls_values_forget and create a new dataloader called clean_retain_loader

            # Alternative approach
            # cutoff = np.mean(ls_values_retain) + 2 * np.std(ls_values_retain)

            # cutoff

            cutoff = np.mean(ls_values_retain) + 2 * np.std(ls_values_retain)

            print("ls_values retain mean", np.mean(ls_values_retain))
            print("forget mean", np.mean(ls_values_forget))
            print("cutoff", cutoff)
            clean_retain_idx = np.where(ls_values_retain < cutoff)[0]
            clean_retain_sampler = torch.utils.data.SequentialSampler(clean_retain_idx)

            # create a deepcopy of train_loader
            copy_retain_loader = copy.deepcopy(train_loader)

            clean_retain_loader = torch.utils.data.DataLoader(
                train_loader.dataset,
                batch_size=train_loader.batch_size,
                sampler=clean_retain_sampler,
            )

            # print how many samples were removed from the train_loader
            print(
                "removed",
                len(train_loader) - len(clean_retain_loader),
                "samples from the train_loader",
            )
            print("idxs: ", len(clean_retain_idx))
        else:
            clean_retain_loader = train_loader

        # ------------------------------------------------

        if ITERATIVE_SEARCH:  # iterative
            # ------------------------------------------------
            # start the frac iterations
            original_model = copy.deepcopy(
                self.model
            )  # do not forget to pass to device after reassigning

            max_tries = MAX_TRY
            min_acc = min_acc_val  # MIN_ACC
            # calculate the accuracy of the model on the train_loader
            org_model_acc = self.get_acc(original_model, forget_loader)

            # Remove the old files
            try:
                os.remove(file_name_1)
                os.remove(file_name_2)
            except:
                print("No previous importance files")

            for try_i in range(max_tries):
                print("----------------> Attempt #", try_i)
                self.best_model = alfssd_tuning(
                    self.model,
                    forget_loader,
                    None,
                    None,
                    train_loader,  # train_loader,
                    self.opt.device,
                    frac_dl,
                    clean_retain_loader,
                    x_D=True,  # turn on alternative D calc
                )

                # calculate the accuracy of the model on the train_loader
                new_model_acc = self.get_acc(self.model, forget_loader)

                # check if minimum acc reached
                if new_model_acc <= min_acc * org_model_acc:
                    print("minimum accuracy reached")
                    break
                else:
                    frac_dl = STEP_MULT * frac_dl
                    self.model = copy.deepcopy(original_model)
                    print("retry with new frac_dl: ", frac_dl)

        else:
            # ---------------------------

            self.best_model = alfssd_tuning(
                self.model,
                forget_loader,
                None,
                None,
                train_loader,  # train_loader,
                self.opt.device,
                frac_dl,
                clean_retain_loader,
                x_D=True,  # turn on alternative D calc
            )

        self.save_files["train_time_taken"] += time.process_time() - time_start

        train_loader = clean_retain_loader

        return

    def get_save_prefix(self):
        self.unlearn_file_prefix = (
            self.opt.pretrain_file_prefix
            + "/"
            + str(self.opt.deletion_size)
            + "_"
            + self.opt.unlearn_method
            + "_"
            + self.opt.exp_name
        )
        self.unlearn_file_prefix += (
            "_" + str(self.opt.train_iters) + "_" + str(self.opt.k)
        )
        self.unlearn_file_prefix += "_adaptivexlf"
        return
