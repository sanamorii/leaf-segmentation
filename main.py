import argparse
import datetime

import optuna
import torch
import torch.optim as optim
import numpy as np

import segmentation_models_pytorch as smp

from loss.earlystop import EarlyStopping
from metrics import StreamSegMetrics
from models.unetdropout import UNETDropout
from train import create_ckpt, train_epoch, train_fn, validate_epoch
from dataset.bean import COLOR_TO_CLASS
from loss.cedice import CEDiceLoss
from dataset.utils import get_dataloader
from utils import save_ckpt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CHOICES = [
    "segformer",
    "unet",
    "unetplusplus",
    "unetdropout",
    "fpn",
    "deeplabv3plus",
    "deeplabv3",
]
ENCODER_CHOICES = [
    "mit_b0",
    "mit_b1",
    "mit_b2",
    "mit_b3",
    "mit_b4",
    "mit_b5",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="model to use", choices=MODEL_CHOICES
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="training dataset",
        choices=["all", "bean", "kale"],
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="resnet50",
        help="",
        choices=ENCODER_CHOICES,
    )
    parser.add_argument(
        "--weights", type=str, default=None, help="", choices=["imagenet"]
    )

    parser.add_argument("--optimiser", type=str, default="rmsprop")
    parser.add_argument("--policy", type=str, default="plateau")

    parser.add_argument("--epochs", type=int, default=50)
    # parser.add_argument("--total_itrs", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-8)
    parser.add_argument("--gradient_clip", type=float, default=1.0)

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--shuffle", type=bool, default=True)

    return parser


# def get_objective_optimiser(trial, model):
#     """Return an optimizer with hyperparameters chosen by Optuna."""
#     optimizer_name = trial.suggest_categorical(
#         'optimizer_name', ['adam', 'adamw', 'sgd', 'rmsprop']
#     )

#     learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
#     weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)

#     if optimizer_name in ['adam', 'adamw']:
#         betas = (trial.suggest_uniform('beta1', 0.8, 0.99),
#                  trial.suggest_uniform('beta2', 0.9, 0.999))
#         if optimizer_name == 'adam':
#             return torch.optim.Adam(model.parameters(),
#                                     lr=learning_rate,
#                                     weight_decay=weight_decay,
#                                     betas=betas)
#         else:  # adamw
#             return torch.optim.AdamW(model.parameters(),
#                                      lr=learning_rate,
#                                      weight_decay=weight_decay,
#                                      betas=betas)

#     elif optimizer_name == 'sgd':
#         momentum = trial.suggest_uniform('momentum', 0.8, 0.99)
#         return torch.optim.SGD(model.parameters(),
#                                lr=learning_rate,
#                                momentum=momentum,
#                                weight_decay=weight_decay)

#     elif optimizer_name == 'rmsprop':
#         momentum = trial.suggest_uniform('momentum', 0.8, 0.99)
#         alpha = trial.suggest_uniform('alpha', 0.8, 0.99)
#         return torch.optim.RMSprop(model.parameters(),
#                                    lr=learning_rate,
#                                    momentum=momentum,
#                                    alpha=alpha,
#                                    weight_decay=weight_decay)


# def get_objective_policy(trial, optimizer, epochs):
#     """Return an LR scheduler with hyperparameters chosen by Optuna."""
#     policy = trial.suggest_categorical(
#         'scheduler_policy', ['plateau', 'step', 'warmupcosine', 'cosine', 'poly']
#     )

#     if policy == "plateau":
#         patience = trial.suggest_int('plateau_patience', 2, 10)
#         factor = trial.suggest_uniform('plateau_factor', 0.1, 0.9)
#         return torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, mode="min", patience=patience, factor=factor
#         )

#     elif policy == "step":
#         step_size = trial.suggest_int('step_size', 5, 50)
#         gamma = trial.suggest_uniform('gamma', 0.1, 0.9)
#         return torch.optim.lr_scheduler.StepLR(
#             optimizer, step_size=step_size, gamma=gamma
#         )

#     elif policy == "warmupcosine":
#         warmup_frac = trial.suggest_uniform('warmup_frac', 0.01, 0.1)
#         warmup_epochs = max(1, int(warmup_frac * epochs))
#         warmup = torch.optim.lr_scheduler.LinearLR(
#             optimizer, start_factor=0.1, total_iters=warmup_epochs
#         )
#         cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, T_max=epochs - warmup_epochs
#         )
#         return torch.optim.lr_scheduler.SequentialLR(
#             optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
#         )

#     elif policy == "cosine":
#         eta_min = trial.suggest_loguniform('eta_min', 1e-6, 1e-3)
#         return torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, T_max=epochs, eta_min=eta_min
#         )

#     elif policy == "poly":
#         power = trial.suggest_uniform('poly_power', 0.5, 1.5)
#         return torch.optim.lr_scheduler.PolynomialLR(
#             optimizer, total_iters=30_000, power=power
#         )


def get_objective_optimiser(trial, model):
    """Return an optimizer with hyperparameters chosen by Optuna."""
    optimiser_name = trial.suggest_categorical(
        "optimizer_name",
        [
            "Adafactor",
            "Adadelta",
            "Adagrad",
            "Adam",
            "Adamax",
            "AdamW",
            "ASGD",
            "LBFGS",
            "NAdam",
            "RAdam",
            "RMSprop",
            "Rprop",
            "SGD",
            "SparseAdam",
        ],
    )
    prefix = optimiser_name.lower() + "_"

    # Common hyperparams
    lr = trial.suggest_loguniform(prefix + "lr", 1e-5, 1e-1)
    weight_decay = trial.suggest_loguniform(prefix + "weight_decay", 1e-8, 1e-2)

    params = model.parameters()

    if optimiser_name == "Adafactor":
        scale_parameter = trial.suggest_categorical(
            prefix + "scale_parameter", [True, False]
        )
        relative_step = trial.suggest_categorical(
            prefix + "relative_step", [True, False]
        )
        warmup_init = trial.suggest_categorical(prefix + "warmup_init", [True, False])
        return optim.Adafactor(
            params,
            lr=lr,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )

    elif optimiser_name == "Adadelta":
        rho = trial.suggest_uniform(prefix + "rho", 0.8, 0.999)
        eps = trial.suggest_loguniform(prefix + "eps", 1e-8, 1e-5)
        return optim.Adadelta(
            params, lr=lr, rho=rho, eps=eps, weight_decay=weight_decay
        )

    elif optimiser_name == "Adagrad":
        initial_accumulator_value = trial.suggest_uniform(
            prefix + "init_acc_val", 0.0, 0.1
        )
        eps = trial.suggest_loguniform(prefix + "eps", 1e-10, 1e-5)
        return optim.Adagrad(
            params,
            lr=lr,
            initial_accumulator_value=initial_accumulator_value,
            eps=eps,
            weight_decay=weight_decay,
        )

    elif optimiser_name in ["Adam", "Adamax", "AdamW", "NAdam", "RAdam", "SparseAdam"]:
        betas = (
            trial.suggest_uniform(prefix + "beta1", 0.8, 0.999),
            trial.suggest_uniform(prefix + "beta2", 0.9, 0.999),
        )
        eps = trial.suggest_loguniform(prefix + "eps", 1e-10, 1e-6)
        kwargs = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        if optimiser_name == "Adam":
            return optim.Adam(params, **kwargs)
        elif optimiser_name == "Adamax":
            return optim.Adamax(params, **kwargs)
        elif optimiser_name == "AdamW":
            return optim.AdamW(params, **kwargs)
        elif optimiser_name == "NAdam":
            momentum_decay = trial.suggest_uniform(
                prefix + "momentum_decay", 0.004, 0.1
            )
            return optim.NAdam(params, **kwargs, momentum_decay=momentum_decay)
        elif optimiser_name == "RAdam":
            return optim.RAdam(params, **kwargs)
        elif optimiser_name == "SparseAdam":
            return optim.SparseAdam(params, **kwargs)

    elif optimiser_name == "ASGD":
        lambd = trial.suggest_uniform(prefix + "lambd", 1e-6, 1e-3)
        alpha = trial.suggest_uniform(prefix + "alpha", 0.5, 0.99)
        t0 = trial.suggest_int(prefix + "t0", 1, 1000)
        return optim.ASGD(
            params, lr=lr, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay
        )

    elif optimiser_name == "LBFGS":
        max_iter = trial.suggest_int(prefix + "max_iter", 5, 50)
        history_size = trial.suggest_int(prefix + "history_size", 10, 100)
        line_search_fn = trial.suggest_categorical(
            prefix + "line_search_fn", [None, "strong_wolfe"]
        )
        return optim.LBFGS(
            params,
            lr=lr,
            max_iter=max_iter,
            history_size=history_size,
            line_search_fn=line_search_fn,
        )

    elif optimiser_name == "RMSprop":
        alpha = trial.suggest_uniform(prefix + "alpha", 0.8, 0.999)
        momentum = trial.suggest_uniform(prefix + "momentum", 0.0, 0.99)
        centered = trial.suggest_categorical(prefix + "centered", [True, False])
        return optim.RMSprop(
            params,
            lr=lr,
            alpha=alpha,
            momentum=momentum,
            centered=centered,
            weight_decay=weight_decay,
        )

    elif optimiser_name == "Rprop":
        etas = (
            trial.suggest_uniform(prefix + "eta_min", 1.001, 1.2),
            trial.suggest_uniform(prefix + "eta_max", 1.2, 2.0),
        )
        step_sizes = (
            trial.suggest_loguniform(prefix + "step_size_min", 1e-6, 1e-3),
            trial.suggest_loguniform(prefix + "step_size_max", 0.1, 1.0),
        )
        return optim.Rprop(params, lr=lr, etas=etas, step_sizes=step_sizes)

    elif optimiser_name == "SGD":
        momentum = trial.suggest_uniform(prefix + "momentum", 0.0, 0.99)
        dampening = trial.suggest_uniform(prefix + "dampening", 0.0, 0.5)
        nesterov = trial.suggest_categorical(prefix + "nesterov", [True, False])
        return optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    else:
        raise ValueError(f"Unknown optimiser: {optimiser_name}")


def get_objective_policy(trial, optimiser, epochs):
    """Return an LR scheduler with hyperparameters chosen by Optuna."""
    policy = trial.suggest_categorical(
        "scheduler_policy",
        [
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
            "CyclicLR",
            "CosineAnnealingWarmRestarts",
            "OneCycleLR",
            "PolynomialLR",
            "WarmupCosine",
        ],
    )
    prefix = policy.lower() + "_"

    # MultiplicativeLR
    if policy == "MultiplicativeLR":
        multiplier = trial.suggest_uniform(prefix + "mult_factor", 0.9, 1.0)
        return torch.optim.lr_scheduler.MultiplicativeLR(
            optimiser, lr_lambda=lambda _: multiplier
        )

    # StepLR
    elif policy == "StepLR":
        step_size = trial.suggest_int(prefix + "step_size", 5, 50)
        gamma = trial.suggest_uniform(prefix + "gamma", 0.1, 0.9)
        return torch.optim.lr_scheduler.StepLR(
            optimiser, step_size=step_size, gamma=gamma
        )

    # MultiStepLR
    elif policy == "MultiStepLR":
        num_milestones = trial.suggest_int(prefix + "num_milestones", 1, 5)
        milestones = sorted(
            [
                trial.suggest_int(f"{prefix}milestone_{i}", 2, epochs - 1)
                for i in range(num_milestones)
            ]
        )
        gamma = trial.suggest_uniform(prefix + "gamma", 0.1, 0.9)
        return torch.optim.lr_scheduler.MultiStepLR(
            optimiser, milestones=milestones, gamma=gamma
        )

    # ConstantLR
    elif policy == "ConstantLR":
        factor = trial.suggest_uniform(prefix + "constant_factor", 0.5, 1.0)
        total_iters = trial.suggest_int(prefix + "total_iters", 1, epochs)
        return torch.optim.lr_scheduler.ConstantLR(
            optimiser, factor=factor, total_iters=total_iters
        )

    # LinearLR
    elif policy == "LinearLR":
        start_factor = trial.suggest_uniform(prefix + "start_factor", 0.1, 1.0)
        total_iters = trial.suggest_int(prefix + "total_iters", 1, epochs)
        return torch.optim.lr_scheduler.LinearLR(
            optimiser, start_factor=start_factor, total_iters=total_iters
        )

    # ExponentialLR
    elif policy == "ExponentialLR":
        gamma = trial.suggest_uniform(prefix + "gamma", 0.8, 0.999)
        return torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=gamma)

    # SequentialLR
    elif policy == "SequentialLR":
        # Build two schedulers and chain them
        step_size1 = trial.suggest_int(prefix + "seq_step1", 5, 20)
        gamma1 = trial.suggest_uniform(prefix + "seq_gamma1", 0.1, 0.9)
        step1 = torch.optim.lr_scheduler.StepLR(
            optimiser, step_size=step_size1, gamma=gamma1
        )

        step_size2 = trial.suggest_int(prefix + "seq_step2", 5, 20)
        gamma2 = trial.suggest_uniform(prefix + "seq_gamma2", 0.1, 0.9)
        step2 = torch.optim.lr_scheduler.StepLR(
            optimiser, step_size=step_size2, gamma=gamma2
        )

        milestone = trial.suggest_int(prefix + "seq_milestone", 1, epochs - 1)
        return torch.optim.lr_scheduler.SequentialLR(
            optimiser, schedulers=[step1, step2], milestones=[milestone]
        )

    # CosineAnnealingLR
    elif policy == "CosineAnnealingLR":
        T_max = trial.suggest_int(prefix + "T_max", 5, epochs)
        eta_min = trial.suggest_loguniform(prefix + "eta_min", 1e-6, 1e-3)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=T_max, eta_min=eta_min
        )

    # ChainedScheduler
    elif policy == "ChainedScheduler":
        exp_gamma = trial.suggest_uniform(prefix + "chain_exp_gamma", 0.8, 0.999)
        cosine_T_max = trial.suggest_int(prefix + "chain_T_max", 5, epochs)
        sched1 = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=exp_gamma)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=cosine_T_max
        )
        return torch.optim.lr_scheduler.ChainedScheduler([sched1, sched2])

    # ReduceLROnPlateau
    elif policy == "ReduceLROnPlateau":
        patience = trial.suggest_int(prefix + "plateau_patience", 2, 10)
        factor = trial.suggest_uniform(prefix + "plateau_factor", 0.1, 0.9)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", patience=patience, factor=factor
        )

    # CyclicLR
    elif policy == "CyclicLR":
        base_lr = trial.suggest_loguniform(prefix + "cyclic_base_lr", 1e-6, 1e-3)
        max_lr = trial.suggest_loguniform(prefix + "cyclic_max_lr", 1e-3, 1e-1)
        step_size_up = trial.suggest_int(prefix + "cyclic_step_size_up", 1, epochs)
        mode = trial.suggest_categorical(
            prefix + "cyclic_mode", ["triangular", "triangular2", "exp_range"]
        )
        return torch.optim.lr_scheduler.CyclicLR(
            optimiser,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=step_size_up,
            mode=mode,
            cycle_momentum=False,
        )

    # CosineAnnealingWarmRestarts
    elif policy == "CosineAnnealingWarmRestarts":
        T_0 = trial.suggest_int(prefix + "T_0", 5, epochs)
        T_mult = trial.suggest_int(prefix + "T_mult", 1, 3)
        eta_min = trial.suggest_loguniform(prefix + "eta_min", 1e-6, 1e-3)
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimiser, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )

    # OneCycleLR
    elif policy == "OneCycleLR":
        max_lr = trial.suggest_loguniform(prefix + "onecycle_max_lr", 1e-4, 1e-1)
        pct_start = trial.suggest_uniform(prefix + "pct_start", 0.1, 0.5)
        anneal_strategy = trial.suggest_categorical(
            prefix + "anneal_strategy", ["cos", "linear"]
        )
        return torch.optim.lr_scheduler.OneCycleLR(
            optimiser,
            max_lr=max_lr,
            total_steps=epochs * len(optimiser.param_groups),  # placeholder
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
        )

    # PolynomialLR
    elif policy == "PolynomialLR":
        power = trial.suggest_uniform(prefix + "poly_power", 0.5, 2.0)
        total_iters = trial.suggest_int(prefix + "poly_total_iters", 10, 100)
        return torch.optim.lr_scheduler.PolynomialLR(
            optimiser, total_iters=total_iters, power=power
        )

    # Warmup w/ LinearLr & Cosine Annealing
    elif policy == "WarmupCosine":
        warmup_frac = trial.suggest_uniform(prefix + "warmup_frac", 0.01, 0.1)
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


def get_policy(policy, optimiser, opts):
    if policy == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", patience=3, factor=0.5
        )
    elif policy == "step":
        return torch.optim.lr_scheduler.StepLR(optimiser, step_size=10000, gamma=0.1)
    elif policy == "warmupcosine":
        warmup_epochs = max(1, int(0.05 * opts.epochs))
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimiser, start_factor=0.1, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=opts.epochs - warmup_epochs
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimiser,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    elif policy == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=opts.epochs, eta_min=1e-4
        )
    elif policy == "poly":
        return torch.optim.lr_scheduler.PolynomialLR(
            optimizer=optimiser, total_iters=30e3, power=0.9
        )
    else:
        raise Exception("invalid policy")


def get_optimiser(name, model, opts):
    if name == "adamw":
        return torch.optim.Adam(
            model.parameters(), lr=opts.learning_rate, weight_decay=opts.weight_decay
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=opts.learning_rate, weight_decay=opts.weight_decay
        )
    if name == "sgd":
        return torch.optim.SGD(
            params=[
                {"params": model.encoder.parameters(), "lr": 0.1 * opts.learning_rate},
                {
                    "params": model.segmentation_head.parameters(),
                    "lr": opts.learning_rate,
                },
            ],
            lr=opts.learning_rate,
            momentum=0.9,
            weight_decay=opts.weight_decay,
        )
    if name == "rmsprop":
        return optim.RMSprop(
            model.parameters(),
            lr=opts.learning_rate,
            weight_decay=opts.weight_decay,
            momentum=0.999,
            foreach=True,
        )


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


def get_lossfn():
    return


def objective(trial):
    print("TUNING")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 5

    # Model
    model = get_model("unetplusplus", "resnet50", "imagenet")
    model.to(device)

    # Optimizer & Scheduler from Optuna
    optimiser = get_objective_optimiser(trial, model)
    scheduler = get_objective_policy(trial, optimiser, epochs)
    criterion = smp.losses.DiceLoss(smooth=1.0, mode='multiclass')

    # Data
    train_loader, val_loader = get_dataloader(
        dataset="all",
        batch_size=4,
        num_workers=2,
        pin_memory=True,
        shuffle=True,
    )

    best_vloss = 999
    best_score = 0.0
    cur_itrs = 0

    grad_scaler = torch.amp.GradScaler(device, enabled=True)
    loss_stop_policy = EarlyStopping(patience=10, delta=0.001)  # early stopping policy
    metrics = StreamSegMetrics(len(COLOR_TO_CLASS))

    for epoch in range(epochs):

        elapsed_ttime, avg_tloss = train_epoch(
            model=model,
            loss_fn=criterion,
            optimiser=optimiser,
            scaler=grad_scaler,
            loader=train_loader,
            device=device,
            epochs=(epoch, epochs),
            gradient_clipping=0.1,
            use_amp=True,
        )

        cur_itrs += len(train_loader)

        elapsed_vtime, val_score, avg_vloss = validate_epoch(
            model=model,
            loader=val_loader,
            metrics=metrics,
            epochs=(epoch, epochs),
            loss_fn=criterion,
            device=device,
        )

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_vloss)
        else:
            scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} - Avg Train Loss: {avg_tloss:.4f}, Avg Val Loss: {avg_vloss:.4f}, Mean IoU: {val_score['Mean IoU']:.4f}"
        )
        print(
            f"Training time: {str(datetime.timedelta(seconds=int(elapsed_ttime)))}, ",
            end="",
        )
        print(f"Validation time: {str(datetime.timedelta(seconds=int(elapsed_vtime)))}")

        # save model
        checkpoint = create_ckpt(
            cur_itrs=cur_itrs,
            model=model,
            optimiser=optimiser,
            scheduler=scheduler,
            tloss=avg_tloss,
            vloss=avg_vloss,
            vscore=val_score,
        )

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
        if val_score["Mean IoU"] > best_score:
            best_score = val_score["Mean IoU"]
            save_ckpt(checkpoint, f"checkpoints/{model.name}_{epochs}_best.pth")
        save_ckpt(checkpoint, f"checkpoints/{model.name}_{epochs}_current.pth")

        trial.report(avg_vloss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        torch.cuda.empty_cache()  # clear cache
        loss_stop_policy(avg_vloss)
        if loss_stop_policy.early_stop:
            print("No improvement in mean IoU - terminating.")
            break
    return best_vloss


def main():
    opts = get_args().parse_args()

    model = get_model(opts.model, opts.encoder, opts.weights)
    model.to(DEVICE)

    print("GPUs:", torch.cuda.device_count())
    print("Using", torch.cuda.device_count(), "GPUs")
    print("Model device:", next(model.parameters()).device)

    print("-" * 50)

    print(f"Encoder: {opts.encoder}")
    print(f"Weights: {opts.weights}")
    print(f"Epochs: {opts.epochs}")
    print("Training model:", model.name)
    print(
        f"Optimiser (lr {opts.learning_rate}, wd {opts.weight_decay}): {opts.optimiser}"
    )
    print("Learning Rate Policy: ", opts.policy)

    train_loader, val_loader = get_dataloader(
        dataset=opts.dataset,
        batch_size=opts.batch_size,
        num_workers=opts.num_workers,
        pin_memory=opts.pin_memory,
        shuffle=opts.shuffle,
    )

    loss_fn = CEDiceLoss(ce_weight=0.5, dice_weight=0.5)
    optimiser = get_optimiser(name=opts.optimiser, opts=opts, model=model)
    scheduler = get_policy(policy=opts.policy, optimiser=optimiser, opts=opts)

    # optimiser = optim.RMSprop(
    #     model.parameters(),
    #     lr=opts.learning_rate,
    #     weight_decay=opts.weight_decay,
    #     momentum=0.999,
    #     foreach=True,
    # )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimiser, mode="min", patience=3, factor=0.5
    # )

    train_fn(
        model=model,
        loss_fn=loss_fn,
        optimiser=optimiser,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=opts.epochs,
        device=DEVICE,
        num_classes=len(COLOR_TO_CLASS),
        visualise=False,
        use_amp=True,
    )


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Dice score: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
