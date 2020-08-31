

class CoustomLR(_LRScheduler):

    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model      = torch.tensor(d_model).to(torch.float32)
        self.warmup_steps = warmup_steps
        super(CoustomLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)


        if (self.last_epoch == 0) or (self.last_epoch %self.warmup_steps!=0):
            return [group['lr'] for group in self.optimizer.param_groups]

        arg1 = torch.rsqrt(self.last_epoch)
        arg2 = self.last_epoch * (self.warmup_steps ** -1.5)

        return [torch.rsqrt(self.d_model) * torch.min(arg1, arg2) for group in self.optimizer.param_groups]
