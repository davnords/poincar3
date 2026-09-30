from tqdm import tqdm
from .utils import to_cuda, MetricLogger
from poincar3.benchmarks import matchbench
import torch

def log_param_statistics(named_parameters, norm_type = 2):
    named_parameters = list(named_parameters)
    grads = [p.grad for n, p in named_parameters if p.grad is not None]
    names = [n for n,p in named_parameters if p.grad is not None]
    device = grads[0].device
    grad_norms = torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads])
    nans_or_infs = torch.isinf(grad_norms) | torch.isnan(grad_norms)
    nan_inf_names = [name for name, naninf in zip(names, nans_or_infs) if naninf]
    if torch.any(nans_or_infs):
        print(f"These params have nan or inf grads: {nan_inf_names}")

def train_step(train_batch, model, objective, optimizer, grad_scaler, grad_clip_norm = 1.,**kwargs):
    optimizer.zero_grad()
    out = model(train_batch)
    l = objective(out, train_batch)
    grad_scaler.scale(l).backward()
    grad_scaler.unscale_(optimizer)
    log_param_statistics(model.named_parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm) # what should max norm be?
    grad_scaler.step(optimizer)
    grad_scaler.update()
    if grad_scaler._scale < 1.:
        grad_scaler._scale = torch.tensor(1.).to(grad_scaler._scale)
    matchbench.GLOBAL_STEP = matchbench.GLOBAL_STEP + matchbench.STEP_SIZE # increment global step
    return {"train_out": out, "train_loss": l.item()}


def train_k_steps(
    n_0, k, dataloader, model, objective, optimizer, lr_scheduler, grad_scaler, progress_bar=True, grad_clip_norm = 1., warmup = None, ema_model = None, pbar_n_seconds = 1,
):

    metric_logger = MetricLogger(delimiter="  ")
    header = "Training"
    
    for n, batch in enumerate(metric_logger.log_every(
        dataloader,
        10,
        header,
        n_0+k,
        n_0,
    )):
        # batch = next(dataloader)
        model.train(True)
        batch = to_cuda(batch)
        out = train_step(
            train_batch=batch,
            model=model,
            objective=objective,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            grad_scaler=grad_scaler,
            n=n,
            grad_clip_norm = grad_clip_norm,
        )
        metric_logger.update(loss=out["train_loss"])
        if ema_model is not None:
            ema_model.update()
        if warmup is not None:
            with warmup.dampening():
                lr_scheduler.step()
        else:
            lr_scheduler.step()


def train_epoch(
    dataloader=None,
    model=None,
    objective=None,
    optimizer=None,
    lr_scheduler=None,
    epoch=None,
):
    model.train(True)
    print(f"At epoch {epoch}")
    for batch in tqdm(dataloader, mininterval=5.0):
        batch = to_cuda(batch)
        train_step(
            train_batch=batch, model=model, objective=objective, optimizer=optimizer
        )
    lr_scheduler.step()
    return {
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "epoch": epoch,
    }


def train_k_epochs(
    start_epoch, end_epoch, dataloader, model, objective, optimizer, lr_scheduler
):
    for epoch in range(start_epoch, end_epoch + 1):
        train_epoch(
            dataloader=dataloader,
            model=model,
            objective=objective,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
        )