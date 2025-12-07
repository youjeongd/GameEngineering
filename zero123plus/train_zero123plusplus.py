import os
import sys
import argparse
import shutil
import subprocess
from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn

from src.utils.train_util import instantiate_from_config


@rank_zero_only
def rank_zero_print(*args):
    print(*args)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        default=None,
        help="resume from checkpoint",
    )
    parser.add_argument(
        "--resume_weights_only",
        action="store_true",
        help="only resume model weights",
    )
    parser.add_argument(
        "-b",
        "--base",
        type=str,
        default="configs/zero123plus-finetune.yaml",
        help="path to base configs (zero123++ finetune yaml)",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default="",
        help="experiment name (suffix for logdir)",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="number of nodes to use",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,",
        help="gpu ids to use, e.g. '0,' or '0,1,2,3,'",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging data",
    )
    return parser


class SetupCallback(Callback):
    def __init__(self, resume, logdir, ckptdir, cfgdir, config):
        super().__init__()
        self.resume = resume
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            rank_zero_print("Project config")
            rank_zero_print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(
                self.config,
                os.path.join(self.cfgdir, "project.yaml"),
            )


class CodeSnapshot(Callback):
    """
    Save a snapshot of the current code base to the log dir.
    """

    def __init__(self, savedir):
        super().__init__()
        self.savedir = savedir

    def get_file_list(self):
        return [
            b.decode()
            for b in set(
                subprocess.check_output(
                    'git ls-files -- ":!:configs/*"', shell=True
                ).splitlines()
            )
            | set(
                subprocess.check_output(
                    "git ls-files --others --exclude-standard", shell=True
                ).splitlines()
            )
        ]

    @rank_zero_only
    def save_code_snapshot(self):
        os.makedirs(self.savedir, exist_ok=True)
        for f in self.get_file_list():
            if not os.path.exists(f) or os.path.isdir(f):
                continue
            dst = os.path.join(self.savedir, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(f, dst)

    def on_fit_start(self, trainer, pl_module):
        try:
            self.save_code_snapshot()
        except Exception:
            rank_zero_warn(
                "Code snapshot is not saved. "
                "Please make sure you have git installed and are in a git repository."
            )


if __name__ == "__main__":
    # add cwd so that `src.*` and `zero123plus.*` can be imported
    sys.path.append(os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    # logdir naming
    cfg_fname = os.path.split(opt.base)[-1]
    cfg_name = os.path.splitext(cfg_fname)[0]
    exp_name = "-" + opt.name if opt.name != "" else ""
    logdir = os.path.join(opt.logdir, cfg_name + exp_name)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    codedir = os.path.join(logdir, "code")

    seed_everything(opt.seed)

    # ------------------------
    # load config (zero123plus-finetune.yaml)
    # ------------------------
    config = OmegaConf.load(opt.base)
    lightning_config = config.lightning
    trainer_config = dict(lightning_config.trainer)

    # PL 2.x style: accelerator/devices
    trainer_config["accelerator"] = "gpu"
    rank_zero_print(f"Running on GPUs {opt.gpus}")
    ngpu = len(opt.gpus.strip(",").split(",")) if opt.gpus.strip(",") != "" else 0
    trainer_config["devices"] = ngpu

    lightning_config.trainer = trainer_config

    # ------------------------
    # instantiate model (MVDiffusion with Zero123++ pipeline)
    # config.model.target: zero123plus.model.MVDiffusion
    # ------------------------
    model = instantiate_from_config(config.model)

    # optional: resume weights only
    if opt.resume and opt.resume_weights_only:
        model = model.__class__.load_from_checkpoint(
            opt.resume, **config.model.params
        )

    # logdir needed by MVDiffusion (for saving images)
    model.logdir = logdir

    # ------------------------
    # logger
    # ------------------------
    default_logger_cfg = {
        "target": "pytorch_lightning.loggers.TensorBoardLogger",
        "params": {
            "name": "tensorboard",
            "save_dir": logdir,
            "version": "0",
        },
    }
    logger_cfg = OmegaConf.merge(default_logger_cfg)
    logger = instantiate_from_config(logger_cfg)

    # ------------------------
    # checkpoint callback
    # ------------------------
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "filename": "{step:08}",
            "verbose": True,
            "save_last": True,
            "every_n_train_steps": 1000,
            "save_top_k": -1,  # save all checkpoints
        },
    }

    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg = OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)

    # ------------------------
    # callbacks
    # ------------------------
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "train_zero123plus.SetupCallback",
            "params": {
                "resume": opt.resume,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
            },
        },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
            },
        },
        "code_snapshot": {
            "target": "train_zero123plus.CodeSnapshot",
            "params": {
                "savedir": codedir,
            },
        },
    }
    default_callbacks_cfg["checkpoint_callback"] = modelckpt_cfg

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

    callbacks = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

    # ------------------------
    # trainer kwargs
    # ------------------------
    trainer_kwargs = {}
    trainer_kwargs["logger"] = logger
    trainer_kwargs["callbacks"] = callbacks
    trainer_kwargs["precision"] = "32-true"
    trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=True)

    # ------------------------
    # build Trainer
    # ------------------------
    trainer = Trainer(
        **trainer_config,
        **trainer_kwargs,
        num_nodes=opt.num_nodes,
    )
    trainer.logdir = logdir

    # ------------------------
    # data module (Objaverse Zero123++ views)
    # ------------------------
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup("fit")

    # ------------------------
    # configure learning rate
    # ------------------------
    base_lr = config.model.base_learning_rate
    if "accumulate_grad_batches" in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1

    rank_zero_print(f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    model.learning_rate = base_lr
    rank_zero_print("++++ NOT USING LR SCALING ++++")
    rank_zero_print(f"Setting learning rate to {model.learning_rate:.2e}")

    # ------------------------
    # run training
    # ------------------------
    if opt.resume and not opt.resume_weights_only:
        trainer.fit(model, data, ckpt_path=opt.resume)
    else:
        trainer.fit(model, data)
