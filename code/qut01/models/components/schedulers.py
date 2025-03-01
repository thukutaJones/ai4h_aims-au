from torch.optim.lr_scheduler import LambdaLR


def get_warmup_decay_scheduler(optimizer, warmup, decay, decay_step):
    """Return a Warmup/decay scheduler."""

    def fn(step: int) -> float:
        """Learning rate decay function."""
        if step <= warmup:
            return step / warmup
        elif step > warmup:
            return decay ** ((step - warmup) / decay_step)

    return LambdaLR(optimizer, fn)


def get_warmup_linear_decay_scheduler(optimizer, warmup, final_step):
    """Return a linear Warmup/decay scheduler."""

    def fn(step: int) -> float:
        """Learning rate decay function."""
        if step <= warmup:
            return step / warmup
        else:
            return max(1.0 - ((step - warmup) / (final_step - warmup)), 0)

    return LambdaLR(optimizer, fn)
