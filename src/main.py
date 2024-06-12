import torch, methods, resnet, timm, time
import numpy as np
from os import makedirs
from os.path import exists
from torch.utils.data.sampler import SubsetRandomSampler
from opts import parse_args
from utils import seed_everything, SubsetSequentialSampler, get_targeted_classes
from datasets import load_dataset, DatasetWrapper, manip_dataset, get_deletion_set
import os
import wandb

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

device = torch.device("cuda" if torch.cuda.is_available() else "mps")  # MAC

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    seed_everything(seed=0)
    # assert(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")  # MAC
    opt = parse_args()
    opt.device = device  # overwrite the device
    print("==> Opts: ", opt)

    # Get model
    if opt.model == "vitb16":
        # model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=opt.num_classes).cuda()
        model = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=opt.num_classes
        ).to(device)
    else:
        # model = getattr(resnet, opt.model)(opt.num_classes).cuda()
        model = getattr(resnet, opt.model)(opt.num_classes).to(device)  # MAC

    # Get dataloaders done
    train_set, train_noaug_set, test_set, train_labels, max_val = load_dataset(
        dataset=opt.dataset, root=opt.data_dir
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    manip_dict, manip_idx, untouched_idx = manip_dataset(
        dataset=opt.dataset,
        train_labels=train_labels,
        method=opt.dataset_method,
        manip_set_size=opt.forget_set_size,
        save_dir=opt.save_dir,
    )
    print("==> Loaded the dataset!")

    wtrain_noaug_cleanL_set = DatasetWrapper(train_noaug_set, manip_dict, mode="test")
    train_test_loader = torch.utils.data.DataLoader(
        wtrain_noaug_cleanL_set,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    untouched_noaug_cleanL_loader = torch.utils.data.DataLoader(
        wtrain_noaug_cleanL_set,
        batch_size=opt.batch_size,
        shuffle=False,
        sampler=SubsetSequentialSampler(untouched_idx),
        num_workers=4,
        pin_memory=True,
    )
    manip_noaug_cleanL_loader = torch.utils.data.DataLoader(
        wtrain_noaug_cleanL_set,
        batch_size=opt.batch_size,
        shuffle=False,
        sampler=SubsetSequentialSampler(manip_idx),
        num_workers=4,
        pin_memory=True,
    )
    eval_loaders = {}
    if opt.dataset_method == "poisoning":
        corrupt_val = np.array(max_val)
        corrupt_size = opt.patch_size
        wtrain_noaug_adv_cleanL_set = DatasetWrapper(
            train_noaug_set,
            manip_dict,
            mode="test_adversarial",
            corrupt_val=corrupt_val,
            corrupt_size=corrupt_size,
        )
        adversarial_train_loader = torch.utils.data.DataLoader(
            wtrain_noaug_adv_cleanL_set,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        untouched_noaug_cleanL_loader = torch.utils.data.DataLoader(
            wtrain_noaug_adv_cleanL_set,
            batch_size=opt.batch_size,
            shuffle=False,
            sampler=SubsetSequentialSampler(untouched_idx),
            num_workers=4,
            pin_memory=True,
        )
        manip_noaug_cleanL_loader = torch.utils.data.DataLoader(
            wtrain_noaug_adv_cleanL_set,
            batch_size=opt.batch_size,
            shuffle=False,
            sampler=SubsetSequentialSampler(manip_idx),
            num_workers=4,
            pin_memory=True,
        )
        wtest_adv_cleanL_set = DatasetWrapper(
            test_set,
            manip_dict,
            mode="test_adversarial",
            corrupt_val=corrupt_val,
            corrupt_size=corrupt_size,
        )
        adversarial_test_loader = torch.utils.data.DataLoader(
            wtest_adv_cleanL_set,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        eval_loaders["adv_test"] = adversarial_test_loader
    else:
        adversarial_train_loader, adversarial_test_loader, corrupt_val, corrupt_size = (
            None,
            None,
            None,
            None,
        )

    eval_loaders["manip"] = manip_noaug_cleanL_loader
    if opt.dataset_method == "labeltargeted":
        classes = get_targeted_classes(opt.dataset)
        indices = []
        for batch_idx, (data, target) in enumerate(test_loader):
            matching_indices = (target == classes[0]) | (target == classes[1])
            absolute_indices = (
                batch_idx * test_loader.batch_size + torch.where(matching_indices)[0]
            )
            indices.extend(absolute_indices.tolist())
        eval_loaders["unseen_forget"] = torch.utils.data.DataLoader(
            test_set,
            batch_size=opt.batch_size,
            shuffle=False,
            sampler=SubsetSequentialSampler(indices),
            num_workers=4,
            pin_memory=True,
        )

    wtrain_manip_set = DatasetWrapper(
        train_set,
        manip_dict,
        mode="pretrain",
        corrupt_val=corrupt_val,
        corrupt_size=corrupt_size,
    )
    pretrain_loader = torch.utils.data.DataLoader(
        wtrain_manip_set,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Stage 1: Pretraining
    opt.pretrain_file_prefix = (
        opt.save_dir
        + "/"
        + opt.dataset
        + "_"
        + opt.model
        + "_"
        + opt.dataset_method
        + "_"
        + str(opt.forget_set_size)
        + "_"
        + str(opt.patch_size)
        + "_"
        + str(opt.pretrain_iters)
        + "_"
        + str(opt.pretrain_lr)
        + "_"
        + str(opt.p_cut)
    )
    if not exists(opt.pretrain_file_prefix):
        print("==> Creating the pretrain directory as it did not exist yet")
        makedirs(opt.pretrain_file_prefix)

    if not exists(opt.pretrain_file_prefix + "/Naive_pretrainmodel/model.pth"):
        print("==> Creating the pretrain directory as it did not exist yet")
        opt.max_lr, opt.train_iters, expname, unlearn_method = (
            opt.pretrain_lr,
            opt.pretrain_iters,
            opt.exp_name,
            opt.unlearn_method,
        )

        # We now actually pretrain by calling unlearn(), misnomer
        opt.unlearn_method, opt.exp_name = "Naive", "pretrainmodel"
        method = getattr(methods, opt.unlearn_method)(opt=opt, model=model)
        method.unlearn(train_loader=pretrain_loader, test_loader=test_loader)
        method.compute_and_save_results(
            train_test_loader,
            test_loader,
            adversarial_train_loader,
            adversarial_test_loader,
        )
        opt.exp_name, opt.unlearn_method = expname, unlearn_method
    else:
        print("==> Loading the pretrained model!")
        model.load_state_dict(
            torch.load(opt.pretrain_file_prefix + "/Naive_pretrainmodel/model.pth")
        )
        # model.to(opt.device)
        model.to(device)  # MAC
        print("==> Loaded the pretrained model!")

    # deletion set
    if opt.deletion_size is None:
        opt.deletion_size = opt.forget_set_size
    forget_idx, retain_idx = get_deletion_set(
        opt.deletion_size,
        manip_dict,
        train_size=len(train_labels),
        dataset=opt.dataset,
        method=opt.dataset_method,
        save_dir=opt.save_dir,
    )

    # original adaptive version
    if True:  # frac_dl == 0.0:
        print("overriding with adaptive")
        # Adaptive calc original
        frac_dl = len(forget_idx) / (len(forget_idx) + len(retain_idx))

        frac_dl = frac_dl / 25

    print(" ")
    print("----------------------------")
    print("deletion size ", opt.deletion_size)
    print("forget_idx: ", len(forget_idx))
    print("retain_idx: ", len(retain_idx))
    print("fraction: ", frac_dl)
    print("----------------------------")
    print(" ")

    opt.max_lr, opt.train_iters = opt.unlearn_lr, opt.unlearn_iters
    if opt.deletion_size != len(manip_dict):
        delete_noaug_cleanL_loader = torch.utils.data.DataLoader(
            wtrain_noaug_cleanL_set,
            batch_size=opt.batch_size,
            shuffle=False,
            sampler=SubsetSequentialSampler(forget_idx),
            num_workers=4,
            pin_memory=True,
        )
        if opt.dataset_method == "poisoning":
            delete_noaug_cleanL_loader = torch.utils.data.DataLoader(
                wtrain_noaug_adv_cleanL_set,
                batch_size=opt.batch_size,
                shuffle=False,
                sampler=SubsetSequentialSampler(forget_idx),
                num_workers=4,
                pin_memory=True,
            )
        eval_loaders["delete"] = delete_noaug_cleanL_loader

    # Stage 2: Unlearning
    method = (
        getattr(methods, "ApplyK")(opt=opt, model=model)
        if opt.unlearn_method in ["EU", "CF"]
        else getattr(methods, opt.unlearn_method)(opt=opt, model=model)
    )

    wtrain_delete_set = DatasetWrapper(
        train_set,
        manip_dict,
        mode="pretrain",
        corrupt_val=corrupt_val,
        corrupt_size=corrupt_size,
        delete_idx=forget_idx,
    )
    # Get the dataloaders
    retain_loader = torch.utils.data.DataLoader(
        wtrain_delete_set,
        batch_size=opt.batch_size,
        shuffle=False,
        sampler=SubsetRandomSampler(retain_idx),
        num_workers=4,
        pin_memory=True,
    )
    train_loader = torch.utils.data.DataLoader(
        wtrain_delete_set,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    forget_loader = torch.utils.data.DataLoader(
        wtrain_delete_set,
        batch_size=opt.batch_size,
        shuffle=False,
        sampler=SubsetRandomSampler(forget_idx),
        num_workers=4,
        pin_memory=True,
    )

    # print dimensions of retain and forget loader
    print(" ")
    print("----------------------------")
    print("retain_loader .dataset: ", len(retain_loader.dataset))
    print("forget_loader .dataset: ", len(forget_loader.dataset))
    print("retain_loader batches: ", len(retain_loader))
    print("forget_loader batches: ", len(forget_loader))
    print("----------------------------")
    print(" ")

    # start timing here
    project_name_used = "poison_timing_step2"
    wandb.init(project=f"{project_name_used}", name=f"{opt.unlearn_method}")

    time_start = time.process_time()

    if opt.unlearn_method in ["Naive", "EU", "CF", "No_unlearning"]:
        method.unlearn(
            train_loader=retain_loader,
            test_loader=test_loader,
            eval_loaders=eval_loaders,
        )
    elif opt.unlearn_method in ["BadT"]:
        method.unlearn(
            train_loader=train_loader,
            test_loader=test_loader,
            eval_loaders=eval_loaders,
        )
    elif opt.unlearn_method in ["Scrub", "SSD"]:
        method.unlearn(
            train_loader=retain_loader,
            test_loader=test_loader,
            forget_loader=forget_loader,
            eval_loaders=eval_loaders,
        )

    elif opt.unlearn_method in ["ASSD", "ASSDR", "ALFSSD", "XALFSSD"]:
        method.unlearn(
            train_loader=retain_loader,
            test_loader=test_loader,
            forget_loader=forget_loader,
            eval_loaders=eval_loaders,
            frac_dl=frac_dl,
            min_acc_val=opt.p_cut,
        )

    # end and log timing here
    time_total = time.process_time() - time_start
    wandb.log(
        {
            "method time": time_total,
            "dataset": opt.dataset,
            "forget_size": opt.forget_set_size,
            "deletion_size": opt.deletion_size,
            "p_cut": opt.p_cut,
        }
    )
    wandb.finish()

    method.compute_and_save_results(
        train_test_loader,
        test_loader,
        adversarial_train_loader,
        adversarial_test_loader,
    )
    print("==> Experiment completed! Exiting..")
