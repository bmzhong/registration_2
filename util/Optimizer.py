import torch
from torch.optim import lr_scheduler
from warmup_scheduler import GradualWarmupScheduler


class Optimizer:
    def __init__(self, config, model, checkpoint=None):

        self.optimizer = self._get_optim(config["optimizer"]["type"],
                                         config["optimizer"]["params"],
                                         model)

        self.lr_scheduler = self._get_lr_scheduler(config["lr_scheduler"]["type"],
                                                   config["lr_scheduler"]["params"],
                                                   self.optimizer)
        if checkpoint is not None and config["load_checkpoint"]:
            self._load(checkpoint)

    def _load(self, checkpoint):
        state_dict = torch.load(checkpoint)["optim"]
        self.optimizer.load_state_dict(state_dict["optim"])
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])

    def step(self):
        self.optimizer.step()
        self.lr_scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_cur_lr(self):
        return self.lr_scheduler.get_last_lr()[0]

    def _get_optim(self, type: str, params: dict, model):
        """
        :param type: classifier
        :param params: parameters included
        :param model: Model that this optimizer is used
        :return:
        """
        return getattr(torch.optim, type, None)(params=model.parameters(), **params)

    def _get_lr_scheduler(self, type: str, params: dict, optim):
        """
        :param type: lr_scheduler type
        :param params: parameters included这个lr_scheduler含有的参数列表
        :param optim: optimizer that this optmizer is used
        :return:
        """
        new_params = params.copy()
        warmup = new_params.get("warmup", False)

        if warmup:
            warmup_step = new_params.get("warmup_steps", 500)
            new_params.pop("warmup")
            new_params.pop("warmup_steps")
            optim = GradualWarmupScheduler(optim, multiplier=1, total_epoch=warmup_step,
                                           after_scheduler=getattr(lr_scheduler, type, None)(optimizer=optim,
                                                                                             **new_params))

        else:
            new_params.pop("warmup")
            new_params.pop("warmup_steps")
            optim = getattr(lr_scheduler, type, None)(optimizer=optim, **new_params)
        return optim

    def state_dict(self):
        return {
            "optim": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict()
        }
