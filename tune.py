import argparse
import datetime

import optuna
import torch
import torch.optim as optim
import numpy as np

import segmentation_models_pytorch as smp
from torchmetrics.segmentation import DiceScore

from loss.earlystop import EarlyStopping
from metrics import StreamSegMetrics
from models.unetdropout import UNETDropout
from models.modelling import ENCODER_CHOICES, MODEL_CHOICES
from train import create_ckpt, train_epoch, train_loop, validate_epoch
from dataset.bean import COLOR_TO_CLASS
from loss.cedice import CEDiceLoss
from dataset.utils import get_dataloader
from utils import save_ckpt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


OPTIMISERS = [
    "Adadelta",
    "Adagrad",
    "Adam",
    "Adamax",
    "AdamW",
    "ASGD",
    "NAdam",
    "RAdam",
    "RMSprop",
    "Rprop",
    "SGD",
]

POLICIES = [
    "MultiplicativeLR",
    "StepLR",
    "MultiStepLR",
    "ConstantLR",
    "LinearLR",
    "ExponentialLR",
    "SequentialLR",
    "CosineAnnealingLR",
    "ChainedScheduler",
    "ReduceLROnPlateau",
    "CosineAnnealingWarmRestarts",
    "PolynomialLR",
    "WarmupCosine",
]



def get_objective_optimiser(optimiser, trial, model):
    """Return an optimizer with hyperparameters chosen by Optuna."""
    prefix = optimiser.lower() + "_"

    # Common hyperparams
    lr = trial.suggest_float(prefix + "lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float(prefix + "weight_decay", 1e-8, 1e-2, log=True)

    params = model.parameters()

    if optimiser.lower() == "adadelta":
        rho = trial.suggest_float(prefix + "rho", 0.8, 0.999)
        eps = trial.suggest_float(prefix + "eps", 1e-8, 1e-5, log=True)
        return optim.Adadelta(
            params, lr=lr, rho=rho, eps=eps, weight_decay=weight_decay
        )

    elif optimiser.lower() == "adagrad":
        initial_accumulator_value = trial.suggest_float(
            prefix + "init_acc_val", 0.0, 0.1
        )
        eps = trial.suggest_float(prefix + "eps", 1e-10, 1e-5, log=True)
        return optim.Adagrad(
            params,
            lr=lr,
            initial_accumulator_value=initial_accumulator_value,
            eps=eps,
            weight_decay=weight_decay,
        )

    elif optimiser.lower() in ["adam", "adamax", "adamw", "nadam", "radam"]:
        betas = (
            trial.suggest_float(prefix + "beta1", 0.8, 0.999),
            trial.suggest_float(prefix + "beta2", 0.9, 0.999),
        )
        eps = trial.suggest_float(prefix + "eps", 1e-10, 1e-6, log=True)
        kwargs = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        if optimiser.lower() == "adam":
            return optim.Adam(params, **kwargs)
        elif optimiser.lower() == "adamax":
            return optim.Adamax(params, **kwargs)
        elif optimiser.lower() == "adamw":
            return optim.AdamW(params, **kwargs)
        elif optimiser.lower() == "nadam":
            momentum_decay = trial.suggest_float(
                prefix + "momentum_decay", 0.004, 0.1
            )
            return optim.NAdam(params, **kwargs, momentum_decay=momentum_decay)
        elif optimiser.lower() == "radam":
            return optim.RAdam(params, **kwargs)

    elif optimiser.lower() == "asgd":
        lambd = trial.suggest_float(prefix + "lambd", 1e-6, 1e-3)
        alpha = trial.suggest_float(prefix + "alpha", 0.5, 0.99)
        t0 = trial.suggest_int(prefix + "t0", 1, 1000)
        return optim.ASGD(
            params, lr=lr, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay
        )

    elif optimiser.lower() == "rmsprop":
        alpha = trial.suggest_float(prefix + "alpha", 0.8, 0.999)
        momentum = trial.suggest_float(prefix + "momentum", 0.0, 0.99)
        return optim.RMSprop(
            params,
            lr=lr,
            alpha=alpha,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    elif optimiser.lower() == "rprop":
        etas = (
            trial.suggest_float(prefix + "eta_min", 0.1, 0.99),
            trial.suggest_float(prefix + "eta_max", 1.01, 2.0),
        )
        step_sizes = (
            trial.suggest_float(prefix + "step_size_min", 1e-6, 1e-2, log=True),
            trial.suggest_float(prefix + "step_size_max", 0.1, 1.0, log=True),
        )
        return optim.Rprop(params, lr=lr, etas=etas, step_sizes=step_sizes)

    elif optimiser.lower() == "sgd":
        momentum = trial.suggest_float(prefix + "momentum", 0.0, 0.99)
        dampening = trial.suggest_float(prefix + "dampening", 0.0, 0.5)
        return optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
        )

    else:
        raise ValueError(f"Unknown optimiser: {optimiser}")


def get_objective_policy(policy, trial, optimiser, epochs):
    """Return an LR scheduler with hyperparameters chosen by Optuna."""
    prefix = policy.lower() + "_"

    # MultiplicativeLR
    if policy.lower() == "multiplicativelr":
        multiplier = trial.suggest_float(prefix + "mult_factor", 0.9, 1.0)
        return torch.optim.lr_scheduler.MultiplicativeLR(
            optimiser, lr_lambda=lambda _: multiplier
        )

    # StepLR
    elif policy.lower() == "steplr":
        step_size = trial.suggest_int(prefix + "step_size", 5, 50)
        gamma = trial.suggest_float(prefix + "gamma", 0.1, 0.9)
        return torch.optim.lr_scheduler.StepLR(
            optimiser, step_size=step_size, gamma=gamma
        )

    # MultiStepLR
    elif policy.lower() == "multisteplr":
        num_milestones = trial.suggest_int(prefix + "num_milestones", 1, 5)
        milestones = sorted(
            [
                trial.suggest_int(f"{prefix}milestone_{i}", 2, epochs - 1)
                for i in range(num_milestones)
            ]
        )
        gamma = trial.suggest_float(prefix + "gamma", 0.1, 0.9)
        return torch.optim.lr_scheduler.MultiStepLR(
            optimiser, milestones=milestones, gamma=gamma
        )

    # ConstantLR
    elif policy.lower() == "constantlr":
        factor = trial.suggest_float(prefix + "constant_factor", 0.5, 1.0)
        total_iters = trial.suggest_int(prefix + "total_iters", 1, epochs)
        return torch.optim.lr_scheduler.ConstantLR(
            optimiser, factor=factor, total_iters=total_iters
        )

    # LinearLR
    elif policy.lower() == "linearlr":
        start_factor = trial.suggest_float(prefix + "start_factor", 0.1, 1.0)
        total_iters = trial.suggest_int(prefix + "total_iters", 1, epochs)
        return torch.optim.lr_scheduler.LinearLR(
            optimiser, start_factor=start_factor, total_iters=total_iters
        )

    # ExponentialLR
    elif policy.lower() == "exponentiallr":
        gamma = trial.suggest_float(prefix + "gamma", 0.8, 0.999)
        return torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=gamma)

    # SequentialLR
    elif policy.lower() == "sequentiallr":
        # Build two schedulers and chain them
        step_size1 = trial.suggest_int(prefix + "seq_step1", 5, 20)
        gamma1 = trial.suggest_float(prefix + "seq_gamma1", 0.1, 0.9)
        step1 = torch.optim.lr_scheduler.StepLR(
            optimiser, step_size=step_size1, gamma=gamma1
        )

        step_size2 = trial.suggest_int(prefix + "seq_step2", 5, 20)
        gamma2 = trial.suggest_float(prefix + "seq_gamma2", 0.1, 0.9)
        step2 = torch.optim.lr_scheduler.StepLR(
            optimiser, step_size=step_size2, gamma=gamma2
        )

        milestone = trial.suggest_int(prefix + "seq_milestone", 1, epochs - 1)
        return torch.optim.lr_scheduler.SequentialLR(
            optimiser, schedulers=[step1, step2], milestones=[milestone]
        )

    # CosineAnnealingLR
    elif policy.lower() == "cosineannealinglr":
        T_max = trial.suggest_int(prefix + "T_max", 5, epochs)
        eta_min = trial.suggest_float(prefix + "eta_min", 1e-6, 1e-3, log=True)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=T_max, eta_min=eta_min
        )

    # ChainedScheduler
    elif policy.lower() == "chainedscheduler":
        exp_gamma = trial.suggest_float(prefix + "chain_exp_gamma", 0.8, 0.999)
        cosine_T_max = trial.suggest_int(prefix + "chain_T_max", 5, epochs)
        sched1 = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=exp_gamma)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=cosine_T_max
        )
        return torch.optim.lr_scheduler.ChainedScheduler([sched1, sched2])

    # ReduceLROnPlateau
    elif policy.lower() == "reducelronplateau":
        patience = trial.suggest_int(prefix + "plateau_patience", 2, 10)
        factor = trial.suggest_float(prefix + "plateau_factor", 0.1, 0.9)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", patience=patience, factor=factor
        )

    # CosineAnnealingWarmRestarts
    elif policy.lower() == "cosineannealingwarmrestarts":
        T_0 = trial.suggest_int(prefix + "T_0", 5, epochs)
        T_mult = trial.suggest_int(prefix + "T_mult", 1, 3)
        eta_min = trial.suggest_float(prefix + "eta_min", 1e-6, 1e-3, log=True)
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimiser, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )
    
    # PolynomialLR
    elif policy.lower() == "polynomiallr":
        power = trial.suggest_float(prefix + "poly_power", 0.5, 2.0)
        total_iters = trial.suggest_int(prefix + "poly_total_iters", 10, 100)
        return torch.optim.lr_scheduler.PolynomialLR(
            optimiser, total_iters=total_iters, power=power
        )

    # Warmup w/ LinearLr & Cosine Annealing
    elif policy.lower() == "warmupcosine":
        warmup_frac = trial.suggest_float(prefix + "warmup_frac", 0.01, 0.1)
        warmup_epochs = max(1, int(warmup_frac * epochs))
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimiser, start_factor=0.1, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=epochs - warmup_epochs
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimiser, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )

    else:
        raise ValueError(f"Unknown scheduler policy: {policy}")

def get_model(name: str, encoder, weights):
    if name == "unetplusplus":
        return smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=weights,
            encoder_depth=5,
            in_channels=3,
            # decoder_channels=[128, 64, 32, 16, 8],  # [256, 128, 64, 32, 16]
            decoder_attention_type="scse",
            classes=len(COLOR_TO_CLASS),
        )
    elif name == "unet":
        return smp.Unet(
            encoder_name=encoder,
            encoder_weights=weights,
            encoder_depth=5,
            in_channels=3,
            # decoder_channels=[128, 64, 32, 16, 8],  # [256, 128, 64, 32, 16]
            classes=len(COLOR_TO_CLASS),
        )
    elif name == "unetdropout":
        return UNETDropout(
            encoder_name=encoder,
            encoder_weights=weights,
            encoder_depth=5,
            in_channels=3,
            num_classes=len(COLOR_TO_CLASS),
        )
    elif name == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=3,
            classes=len(COLOR_TO_CLASS),
        )
    elif name == "deeplabv3":
        return smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=3,
            classes=len(COLOR_TO_CLASS),
        )
    elif name == "deeplabv3plus":
        return smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=3,
            classes=len(COLOR_TO_CLASS),
        )
    elif name == "segformer":
        return smp.Segformer(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=3,
            classes=len(COLOR_TO_CLASS),
        )

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True, help="model to use", choices=MODEL_CHOICES)
    parser.add_argument("--encoder", type=str, default="resnet50", help="", choices=ENCODER_CHOICES,)
    parser.add_argument("--weights", type=str, default=None, help="", choices=["imagenet"])
    parser.add_argument("--optimiser", type=str, default=None)
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument("--n_trials", type=int, required=True)
    parser.add_argument("--epochs_per_trial", type=int, default=10)

    return parser

def objective(trial):

    opts = get_args().parse_args()

    print("TUNING")
    epochs = opts.epochs_per_trial

    # Model
    model = get_model(opts.model, opts.encoder, opts.weights)
    model.to(DEVICE)

    if opts.optimiser is None:
        optimiser_name = trial.suggest_categorical("optimizer_name", OPTIMISERS)
    else:
        optimiser_name = opts.optimiser
    
    if opts.policy is None:
        policy = trial.suggest_categorical("scheduler_policy", POLICIES)
    else:
        policy = opts.policy

    # Optimizer & Scheduler from Optuna
    optimiser = get_objective_optimiser(optimiser_name, trial, model)
    scheduler = get_objective_policy(policy, trial, optimiser, epochs)
    criterion = smp.losses.DiceLoss(smooth=1.0, mode='multiclass')

    # Data
    train_loader, val_loader = get_dataloader(
        dataset="all",
        batch_size=8,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    )

    best_vloss = 999
    cur_itrs = 0

    grad_scaler = torch.amp.GradScaler(DEVICE, enabled=True)
    metrics = StreamSegMetrics(len(COLOR_TO_CLASS))

    for epoch in range(epochs):

        elapsed_ttime, avg_tloss = train_epoch(
            model=model,
            loss_fn=criterion,
            optimiser=optimiser,
            scaler=grad_scaler,
            loader=train_loader,
            device=DEVICE,
            epochs=(epoch, epochs),
            gradient_clipping=0.1,
            use_amp=True,
            verbose=False,
        )

        cur_itrs += len(train_loader)

        elapsed_vtime, val_score, avg_vloss = validate_epoch(
            model=model,
            loader=val_loader,
            metrics=metrics,
            epochs=(epoch, epochs),
            loss_fn=criterion,
            device=DEVICE,
            verbose=False,
        )

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_vloss)
        else:
            scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} - Avg Train Loss: {avg_tloss:.4f}, Avg Val Loss: {avg_vloss:.4f}, Mean IoU: {val_score['Mean IoU']:.4f}",
            end=" "
        )
        print(
            f"Training time: {str(datetime.timedelta(seconds=int(elapsed_ttime)))},",
            end=" ",
        )
        print(f"Validation time: {str(datetime.timedelta(seconds=int(elapsed_vtime)))}")

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss

        trial.report(avg_vloss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        torch.cuda.empty_cache()  # clear cache
    return best_vloss


if __name__ == "__main__":
    opts = get_args().parse_args()
    print("Training model:", opts.model)
    print(f"Encoder: {opts.encoder}")
    print(f"Weights: {opts.weights}")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=opts.n_trials)



    print("Best trial:")
    trial = study.best_trial
    print(f"  Dice score: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
