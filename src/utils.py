import random, torch, copy, tqdm
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from functools import partial
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from typing import Dict, List
import pickle
import os

PYTORCH_MPS_HIGH_WATERMARK_RATIO = 0.0
device = ("cuda" if torch.cuda.is_available() else "mps",)


# Reference: https://github.com/if-loops/selective-synaptic-dampening/blob/main/src/forget_random_strategies.py
# Hessian based method that is more efficient than Fisher etc. and outperforms.
class ParameterPerturber_old:
    def __init__(
        self,
        model,
        opt,
        device="cuda" if torch.cuda.is_available() else "cpu",
        parameters=None,
    ):
        self.model = model
        self.opt = opt
        self.device = device
        self.alpha = None
        self.xmin = None

        print(parameters)
        self.lower_bound = parameters["lower_bound"]
        self.exponent = parameters["exponent"]
        self.magnitude_diff = parameters["magnitude_diff"]  # unused
        self.min_layer = parameters["min_layer"]
        self.max_layer = parameters["max_layer"]
        self.forget_threshold = parameters["forget_threshold"]  # unused
        self.dampening_constant = parameters["dampening_constant"]  # lambda
        self.selection_weighting = parameters["selection_weighting"]  # alpha

    def zerolike_params_dict(self, model: torch.nn) -> Dict[str, torch.Tensor]:
        """
        Taken from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Returns a dict like named_parameters(), with zeroed-out parameter valuse
        Parameters:
        model (torch.nn): model to get param dict from
        Returns:
        dict(str,torch.Tensor): dict of zero-like params
        """
        return dict(
            [
                (k, torch.zeros_like(p, device=p.device))
                for k, p in model.named_parameters()
            ]
        )

    def calc_importance(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Adapated from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Calculate per-parameter, importance
            returns a dictionary [param_name: list(importance per parameter)]
        Parameters:
        DataLoader (DataLoader): DataLoader to be iterated over
        Returns:
        importances (dict(str, torch.Tensor([]))): named_parameters-like dictionary containing list of importances for each parameter
        """
        criterion = nn.CrossEntropyLoss()
        importances = self.zerolike_params_dict(self.model)
        for x, y, idx in tqdm.tqdm(dataloader):
            x, y = x.to(self.device), y.to(self.device)

            self.opt.zero_grad()

            out = self.model(x)
            loss = criterion(out, y)

            # override with ALFSSD loss
            # loss = torch.norm(out, p="fro", dim=1).abs().mean()

            loss.backward()

            for (k1, p), (k2, imp) in zip(
                self.model.named_parameters(), importances.items()
            ):
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)  # original
                    # imp.data += p.grad.data.clone().abs()

        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))
        return importances

    def modify_weight(
        self,
        original_importance: List[Dict[str, torch.Tensor]],
        forget_importance: List[Dict[str, torch.Tensor]],
    ) -> None:
        """
        Perturb weights based on the SSD equations given in the paper
        Parameters:
        original_importance (List[Dict[str, torch.Tensor]]): list of importances for original dataset
        forget_importance (List[Dict[str, torch.Tensor]]): list of importances for forget sample
        threshold (float): value to multiply original imp by to determine memorization.

        Returns:
        None

        """

        with torch.no_grad():
            for (n, p), (oimp_n, oimp), (fimp_n, fimp) in zip(
                self.model.named_parameters(),
                original_importance.items(),
                forget_importance.items(),
            ):
                # Synapse Selection with parameter alpha
                oimp_norm = oimp.mul(self.selection_weighting)
                locations = torch.where(fimp > oimp_norm)

                # Synapse Dampening with parameter lambda
                weight = ((oimp.mul(self.dampening_constant)).div(fimp)).pow(
                    self.exponent
                )
                update = weight[locations]
                # Bound by 1 to prevent parameter values to increase.
                min_locs = torch.where(update > self.lower_bound)
                update[min_locs] = self.lower_bound
                p[locations] = p[locations].mul(update)


# set random seeds for random, numpy and torch
SEED = 42
BATCHSIZE = 1024
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
# sampler = TPESampler(seed=SEED)  # Make the sampler behave in a deterministic way.


def convert_boolean_to_int(df):
    for column in df.columns:
        if df[column].dtype == bool:
            df[column] = df[column].astype(int)
    return df


def define_model(trial):
    # We optimize the number of layers, hidden units, and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 3, 3, step=1)

    layers = []

    in_features = X_train.shape[1]  # Input feature size based on X_train

    out_features = trial.suggest_int("n_units", 100, 100, step=50)

    for i in range(int(n_layers)):

        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.BatchNorm1d(out_features))  # Add BatchNorm1d layer
        layers.append(nn.ReLU())

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


def get_custom_dataset(X, y):
    # Convert the dataframes to PyTorch tensors
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)

    custom_dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    data_loader = torch.utils.data.DataLoader(
        custom_dataset, batch_size=BATCHSIZE, shuffle=False
    )

    return data_loader


def objective(trial):
    # Generate the model.
    global model
    model = define_model(trial).to(DEVICE)

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    l2_pen = trial.suggest_float("l2_pen", 1e-6, 1e-1, log=True)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=l2_pen)

    EPOCHS = trial.suggest_int("epochs", 25, 25, step=5)

    # Get the custom dataset.
    # split X_train and y_train_class into train and validation sets
    X_Train, X_Val, y_Train_class, y_Val_class = train_test_split(
        X_train, y_train_class, test_size=0.2, random_state=42
    )  # CHANGED

    train_loader = get_custom_dataset(X_Train, y_Train_class)
    valid_loader = get_custom_dataset(X_Val, y_Val_class)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma_)

    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


def callback(study, trial):
    global best_model
    if study.best_trial == trial:
        best_model = model


class ParameterPerturber:
    def __init__(
        self,
        model,
        opt,
        device="cuda" if torch.cuda.is_available() else "mps",
        parameters=None,
        adaptive=False,
        label_free=False,
        x_D=False,
    ):
        self.model = model.to(device)
        self.opt = opt
        self.device = "mps"
        self.alpha = parameters["selection_weighting"]
        self.xmin = None
        self.device = device
        self.adaptive = adaptive
        self.label_free = label_free
        self.x_D = x_D

        self.lower_bound = 1
        self.exponent = 1
        self.magnitude_diff = None  # unused
        self.min_layer = -1
        self.max_layer = -1
        self.forget_threshold = 1  # unused
        self.dampening_constant = "Adaptive"  # parameters["dampening_constant"]
        self.selection_weighting = "Adaptive"  # parameters["selection_weighting"]

    def get_layer_num(self, layer_name: str) -> int:
        layer_id = layer_name.split(".")[1]
        if layer_id.isnumeric():
            return int(layer_id)
        else:
            return -1

    def zerolike_params_dict(self, model: torch.nn) -> Dict[str, torch.Tensor]:
        """
        Taken from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Returns a dict like named_parameters(), with zeroed-out parameter valuse
        Parameters:
        model (torch.nn): model to get param dict from
        Returns:
        dict(str,torch.Tensor): dict of zero-like params
        """
        return dict(
            [
                (k, torch.zeros_like(p, device=p.device))
                for k, p in model.named_parameters()
            ]
        )

    def calc_importance(
        self, dataloader: DataLoader, extra_noise=False
    ) -> Dict[str, torch.Tensor]:
        """
        Adapated from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Calculate per-parameter, importance
            returns a dictionary [param_name: list(importance per parameter)]
        Parameters:
        DataLoader (DataLoader): DataLoader to be iterated over
        Returns:
        importances (dict(str, torch.Tensor([]))): named_parameters-like dictionary containing list of importances for each parameter
        """
        criterion = nn.CrossEntropyLoss()
        importances = self.zerolike_params_dict(self.model)
        for x, y, idx in dataloader:
            x, y = x.to(self.device), y.to(self.device)

            if extra_noise:
                print("NOISE IS ON")
                # add torch rand x% noise
                x += torch.randn_like(x) * 0.01

            self.opt.zero_grad()

            out = self.model(x)

            if self.label_free:
                if self.x_D:
                    # loss = torch.abs(out).sum(dim=1).mean() # L1
                    # loss = torch.norm(out, p="fro", dim=1).abs().mean() # prev
                    loss = torch.norm(out, p="fro", dim=1).abs().mean()
                else:  # original
                    # loss = torch.norm(out, p="fro", dim=1).abs().mean()
                    loss = torch.norm(out, p="fro", dim=1).pow(2).mean()  # actual one
                    # loss = (torch.norm(out, p="fro", dim=1).pow(2).mean())  # original is pow(2) not abs -> l2 vs l1 norm
            else:
                loss = criterion(out, y)

            loss.backward()

            for (k1, p), (k2, imp) in zip(
                self.model.named_parameters(), importances.items()
            ):
                if p.grad is not None:
                    if self.label_free:
                        if self.x_D:
                            # print("Using X_d")
                            imp.data += p.grad.data.clone().abs()
                            # imp.data += (p.grad.data.clone().abs()) # prev
                        else:  # original is abs
                            # print("Using original LF")
                            imp.data += p.grad.data.clone().abs()
                    else:
                        # print("Using original SSD")
                        imp.data += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))
        return importances

    def modify_weight(
        self,
        original_importance: List[Dict[str, torch.Tensor]],
        forget_importance: List[Dict[str, torch.Tensor]],
        PERCENTILE="PERCENTILE NOT AUTOMATICALLY SET BUT ADAPTIVE SELECTED - ERROR",
        x_D=False,
    ) -> None:
        """
        Perturb weights based on the SSD equations given in the paper
        Parameters:
        original_importance (List[Dict[str, torch.Tensor]]): list of importances for original dataset
        forget_importance (List[Dict[str, torch.Tensor]]): list of importances for forget sample
        threshold (float): value to multiply original imp by to determine memorization.

        Returns:
        None

        """

        if self.adaptive:  # adaptive SSD
            rel_list = list()

            # Get the indices of the fully connected layers
            fully_connected_layer_indices = list()
            for idx, layer in enumerate(self.model.children()):
                if isinstance(layer, nn.Linear):
                    fully_connected_layer_indices.append(idx + 1)

            # print("LAYERS: ",fully_connected_layer_indices)

            num_layers = sum(1 for name, layer in self.model.named_children())

            all_relative_values = []
            with torch.no_grad():
                for (n, p), (oimp_n, oimp), (fimp_n, fimp) in zip(
                    self.model.named_parameters(),
                    original_importance.items(),
                    forget_importance.items(),
                ):
                    LAYER_SIZE_CUTOFF = 0  # overrride
                    if p.shape[0] >= LAYER_SIZE_CUTOFF:  # only look at large layers

                        divs_ = fimp.div(oimp)
                        # select only the non nan values of divs_ to avoid errors
                        divs_ = divs_[~torch.isnan(divs_)]

                        # remove inf
                        divs_ = divs_[~torch.isinf(divs_)]

                        all_relative_values.append(divs_.reshape(-1).cpu().numpy())

            all_relative_values = np.concatenate(
                all_relative_values
            )  # flatten the array

            # PERCENTILE = NONE
            # print("USED Percentile: ", PERCENTILE)
            print("percentile:", PERCENTILE)
            print(all_relative_values)
            percentile = np.nanpercentile(all_relative_values, PERCENTILE)

            # print("USED cutoff value: ", percentile)
            percentile = percentile.item()
            print("percentile value:", percentile)
            # Main part after finding cutoff value

            curr_layer = 0
            with torch.no_grad():
                for (n, p), (oimp_n, oimp), (fimp_n, fimp) in zip(
                    self.model.named_parameters(),
                    original_importance.items(),
                    forget_importance.items(),
                ):

                    LAYER_SIZE_CUTOFF = 100  # can be added if you want to avoid modifying small layers to increase robustness of the method
                    # if x_D:
                    #    run_unlearn = p.shape[0]>=LAYER_SIZE_CUTOFF
                    # else:
                    #    run_unlearn = True
                    if True:  # p.shape[0]>=LAYER_SIZE_CUTOFF:
                        divs_ = fimp.div(oimp)

                        # select only the non nan values of divs_
                        divs_ = divs_[~torch.isnan(divs_)]

                        # print(fimp, "XXX", oimp)
                        relative = torch.mean(divs_)
                        rel_std = torch.std(divs_)
                        rel_median = torch.median(divs_)
                        # calculate absolute difference between median and mean
                        abs_diff = torch.abs(rel_median - relative)

                        # Always adaptive
                        self.selection_weighting = percentile
                        self.dampening_constant = 1  # constant from ASSD paper

                        # if self.alpha == "Adaptive":
                        #     # print(relative, 2 * rel_std)
                        #     self.selection_weighting = percentile
                        # else:
                        #     # print("Backup alpha 10")
                        #     self.selection_weighting = self.alpha

                        rel_list.append(self.selection_weighting)

                        # Synapse Selection with parameter alpha
                        oimp_norm = oimp.mul(self.selection_weighting)
                        locations = torch.where(fimp > oimp_norm)

                        # Synapse Dampening with parameter lambda
                        weight = ((oimp.mul(self.dampening_constant)).div(fimp)).pow(
                            self.exponent
                        )

                        update = weight[locations]
                        # Bound by 1 to prevent parameter values to increase.
                        min_locs = torch.where(update > self.lower_bound)

                        # for update take the update value where update > 0.1, otherwise set update 0.1
                        # We do not use this in the paper but this can be used to avoid dead nerons for extra robustness
                        # if x_D:
                        #    dampen_limit = 0.01
                        # else:
                        #    dampen_limit = 0
                        dampen_limit = 0
                        update[update < dampen_limit] = dampen_limit

                        update[min_locs] = self.lower_bound
                        p[locations] = p[locations].mul(update)

        else:  # vanilla SSD
            with torch.no_grad():
                for (n, p), (oimp_n, oimp), (fimp_n, fimp) in zip(
                    self.model.named_parameters(),
                    original_importance.items(),
                    forget_importance.items(),
                ):
                    # Synapse Selection with parameter alpha
                    oimp_norm = oimp.mul(self.selection_weighting)
                    locations = torch.where(fimp > oimp_norm)

                    # Synapse Dampening with parameter lambda
                    weight = ((oimp.mul(self.dampening_constant)).div(fimp)).pow(
                        self.exponent
                    )
                    update = weight[locations]
                    # Bound by 1 to prevent parameter values to increase.
                    min_locs = torch.where(update > self.lower_bound)
                    update[min_locs] = self.lower_bound
                    p[locations] = p[locations].mul(update)


# default values:
# "dampening_constant" lambda: 1,
# "selection_weighting" alpha: 10 * model_size_scaler,
# model_size_scaler = 1
# if args.net == "ViT":
#     model_size_scaler = 0.5

# We found hyper-parameters using 50
# runs of the TPE search from Optuna (Akiba et al. 2019), for
# values α ∈ [0.1, 100]) and λ ∈ [0.1, 5]. We only conducted
# this search for the Rocket and Veh2 classes. We use λ=1
# and α=10 for all ResNet18 CIFAR tasks. For PinsFaceRecognition, we use α=50 and λ=0.1 due to the much greater
# similarity between classes. ViT also uses λ=1 on all CIFAR
# tasks. We change α=10 to α=5 for slightly improved performance on class and α=25 on sub-class unlearning.


def ssd_tuning(
    model,
    forget_train_dl,
    dampening_constant,
    selection_weighting,
    full_train_dl,
    device,
):
    parameters = {
        "lower_bound": 1,
        "exponent": 1,
        "magnitude_diff": None,
        "min_layer": -1,
        "max_layer": -1,
        "forget_threshold": 1,
        "dampening_constant": dampening_constant,
        "selection_weighting": selection_weighting,
    }

    # load the trained model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    pdr = ParameterPerturber_old(model, optimizer, device, parameters)
    model = model.eval()

    sample_importances = pdr.calc_importance(forget_train_dl)

    original_importances = pdr.calc_importance(full_train_dl)
    pdr.modify_weight(original_importances, sample_importances)
    return model


def assd_tuning(
    model,
    forget_train_dl,
    dampening_constant,
    selection_weighting,
    full_train_dl,
    device,
    frac_dl,
):
    parameters = {
        "lower_bound": 1,
        "exponent": 1,
        "magnitude_diff": None,
        "min_layer": -1,
        "max_layer": -1,
        "forget_threshold": 1,
        "dampening_constant": None,  # adaptive overwrites this
        "selection_weighting": None,  # adaptive overwrites this
    }

    print("----- Using ASSD -----")

    # Sweep loop (ASSD extension)
    sweeps_n = 1
    frac_dl = frac_dl * (1 / sweeps_n)
    for sweep_i in range(sweeps_n):

        # load the trained model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        pdr = ParameterPerturber(model, optimizer, device, parameters, adaptive=True)
        model = model.eval()

        ########################## SPEEDUP
        file_name = "original_importances.pkl"
        if os.path.exists(file_name):
            print("##### LOADING IMPORTANCES")
            with open("original_importances.pkl", "rb") as f:
                original_importances = pickle.load(f)
            with open("sample_importances.pkl", "rb") as f:
                sample_importances = pickle.load(f)
        else:
            # ----
            print("##### CALCULATING IMPORTANCES")
            # Calculate the importances of D (see paper)
            original_importances = pdr.calc_importance(full_train_dl)
            # safe the importances locally for reuse

            with open("original_importances.pkl", "wb") as f:
                pickle.dump(original_importances, f)

            # Calculation of the forget set importances
            sample_importances = pdr.calc_importance(forget_train_dl)

            with open("sample_importances.pkl", "wb") as f:
                pickle.dump(sample_importances, f)

        ########################## SPEEDUP

        # auto select percentile
        len_all = len(full_train_dl.dataset) + len(forget_train_dl.dataset)

        len_forget = len(forget_train_dl.dataset)

        # due to some issue with .dataset we use batches as a proxy
        # len_forget = len(forget_train_dl)
        # len_all = len(full_train_dl) + len(forget_train_dl)

        len_forget = len(forget_train_dl.dataset)
        len_all = len(full_train_dl.dataset) + len(forget_train_dl.dataset)

        # share_off = np.sqrt((len_forget/len_all)*100)/MODIFIER_PERCENTILE
        # share_off = np.log(1 + (len_forget / len_all) * 100)
        share_off = np.log(1 + frac_dl * 100)

        percentile = 100 - share_off
        print("###### ----- Length based percentile: ", percentile)

        # Dampen selected parameters
        alpha_list = pdr.modify_weight(
            original_importances, sample_importances, PERCENTILE=percentile
        )
        # ---

    return model


# placeholder for now reusing assd
def alfssd_tuning(
    model,
    forget_train_dl,
    dampening_constant,
    selection_weighting,
    full_train_dl,
    device,
    frac_dl,
    filtered_loader,
    x_D=False,
):
    parameters = {
        "lower_bound": 1,
        "exponent": 1,
        "magnitude_diff": None,
        "min_layer": -1,
        "max_layer": -1,
        "forget_threshold": 1,
        "dampening_constant": None,  # adaptive
        "selection_weighting": None,
    }

    # Sweep loop (ASSD extension)
    if x_D:
        sweeps_n = 1
        frac_dl = frac_dl * (1 / sweeps_n)
    else:
        sweeps_n = 1  # i.e. original
        frac_dl = frac_dl * (1 / sweeps_n)

    for sweep_i in range(sweeps_n):
        sweep_i += 1
        print("Sweep #", sweep_i)
        # load the trained model
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        pdr = ParameterPerturber(
            model,
            optimizer,
            device,
            parameters,
            adaptive=True,
            label_free=True,
            x_D=x_D,
        )
        model = model.eval()

        ########################## SPEEDUP
        file_name = "original_importances.pkl"
        if os.path.exists(file_name):
            with open("original_importances.pkl", "rb") as f:
                original_importances = pickle.load(f)
            with open("sample_importances.pkl", "rb") as f:
                sample_importances = pickle.load(f)
        else:
            # ----
            # Calculate the importances of D (see paper)
            original_importances = pdr.calc_importance(
                filtered_loader, extra_noise=False
            )
            # safe the importances locally for reuse

            with open("original_importances.pkl", "wb") as f:
                pickle.dump(original_importances, f)

            # Calculation of the forget set importances
            sample_importances = pdr.calc_importance(forget_train_dl, extra_noise=False)

            with open("sample_importances.pkl", "wb") as f:
                pickle.dump(sample_importances, f)

        ########################## SPEEDUP

        # sample_importances = pdr.calc_importance(forget_train_dl, extra_noise=False)
        # original_importances = pdr.calc_importance(filtered_loader, extra_noise=False)

        len_forget = len(forget_train_dl.dataset)
        len_all = len(filtered_loader.dataset) + len(forget_train_dl.dataset)

        # share_off = np.sqrt((len_forget/len_all)*100)/MODIFIER_PERCENTILE
        # share_off = np.log(1 + (len_forget / len_all) * 100)

        if x_D:
            share_off = np.log(1 + frac_dl * 100)  # orig 100
            percentile = 100 - share_off
        else:  # original
            share_off = np.log(1 + frac_dl * 100)
            percentile = 100 - share_off

        print("###### ----- Length based percentile: ", percentile)

        # Dampen selected parameters
        alpha_list = pdr.modify_weight(
            original_importances, sample_importances, PERCENTILE=percentile, x_D=x_D
        )

    return model


class LinearLR(_LRScheduler):
    r"""Set the learning rate of each parameter group with a linear
    schedule: :math:`\eta_{t} = \eta_0*(1 - t/T)`, where :math:`\eta_0` is the
    initial lr, :math:`t` is the current epoch or iteration (zero-based) and
    :math:`T` is the total training epochs or iterations. It is recommended to
    use the iteration based calculation if the total number of epochs is small.
    When last_epoch=-1, sets initial lr as lr.
    It is studied in
    `Budgeted Training: Rethinking Deep Neural Network Training Under Resource
     Constraints`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T (int): Total number of training epochs or iterations.
        last_epoch (int): The index of last epoch or iteration. Default: -1.

    .. _Budgeted Training\: Rethinking Deep Neural Network Training Under
    Resource Constraints:
        https://arxiv.org/abs/1905.04753
    """

    def __init__(self, optimizer, T, warmup_epochs=100, last_epoch=-1):
        self.T = float(T)
        self.warm_ep = warmup_epochs
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch - self.warm_ep >= 0:
            rate = 1 - ((self.last_epoch - self.warm_ep) / self.T)
        else:
            rate = (self.last_epoch + 1) / (self.warm_ep + 1)
        return [rate * base_lr for base_lr in self.base_lrs]

    def _get_closed_form_lr(self):
        return self.get_lr()


def cutmix(x, y, alpha=1.0):
    assert alpha > 0
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)  # .cuda()

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def seed_everything(seed):
    """
    Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random

    Args:
        seed: the integer value seed for global random state
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


class Model_with_TTA(torch.nn.Module):
    def __init__(self, model, mult_fact=1, tta_type="flip"):
        super(Model_with_TTA, self).__init__()
        self.model = model
        self.mult_fact = mult_fact
        self.tta_type = tta_type

    def forward(self, x):
        out = self.model(x) * self.mult_fact
        if self.tta_type == "flip":
            out += self.model(torch.flip(x, dims=[3]))
            out /= 2
        return self.net(out)


def get_targeted_classes(dataset):
    if dataset == "CIFAR10":
        classes = (3, 5)
    elif dataset == "CIFAR100":
        classes = (47, 53)
    elif dataset in ["PCAM", "DermNet", "Pneumonia"]:
        classes = (0, 1)
    elif dataset in ["LFWPeople", "CelebA"]:
        # Raise NotImplemented Error
        assert False, "Not Implemented Yet"
    return classes


def unlearn_func(
    model, method, factor=0.1, device="cuda" if torch.cuda.is_available() else "mps"
):
    model = copy.deepcopy(model)
    model = model.cpu()
    if method == "EU":
        model.apply(initialize_weights)
    elif method == "Mixed":
        partialfunc = partial(modify_weights, factor=factor)
        model.apply(partialfunc)
    else:
        pass
    model.to(device)
    return model


def initialize_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        m.reset_parameters()
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        m.reset_parameters()
    elif isinstance(m, torch.nn.Linear):
        m.reset_parameters()
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)


def modify_weights(m, factor=0.1):
    if isinstance(m, torch.nn.Conv2d):
        m.weight.data = m.weight.data * factor
        if m.bias is not None:
            m.bias.data = m.bias.data * factor
    elif isinstance(m, torch.nn.BatchNorm2d):
        if m.affine:
            m.weight.data = m.weight.data * factor
            m.bias.data = m.bias.data * factor
    elif isinstance(m, torch.nn.Linear):
        m.weight.data = m.weight.data * factor
        if m.bias is not None:
            m.bias.data = m.bias.data * factor


def distill_kl_loss(y_s, y_t, T, reduction="sum"):
    p_s = torch.nn.functional.log_softmax(y_s / T, dim=1)
    p_t = torch.nn.functional.softmax(y_t / T, dim=1)
    loss = torch.nn.functional.kl_div(p_s, p_t, reduction=reduction)
    if reduction == "none":
        loss = torch.sum(loss, dim=1)
    loss = loss * (T**2) / y_s.shape[0]
    return loss


def compute_accuracy(preds, y):
    return np.equal(np.argmax(preds, axis=1), y).mean()


class SubsetSequentialSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)
