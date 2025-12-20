import os
import time
import copy
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from typing import Optional, List
from helpers.config import Settings
from torch.utils.data import DataLoader
from models.lr import LRProviderFactory
from models.lr.LREnums import LREnums
from models.baselines.providers import PersonModelProvider
from models.baselines import BaselinesProviderFactory
from models.baselines.BaselinesEnums import BaselinesEnums
from torch.utils.tensorboard import SummaryWriter


class TrainerController:
    
    def __init__(self, 
                 baseline: BaselinesEnums,
                 lr_scheduler: LREnums,
                 settings: Settings,
                 train_loader: DataLoader, 
                 val_loader: DataLoader, 
                 test_loader: Optional[DataLoader] = None,
                 resnet_pretrained: bool = True,
                 base_finetuned: Optional[PersonModelProvider] = None,
                 base_freeze: bool = True,
                 person_temporal: bool = True,
                 compile: bool = True,
                 tensorboard_track: bool = True,):

        self.settings = settings
        self.tensorboard_track = tensorboard_track

        # setting our dataset loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # getting our learning rate scheduler
        self.scheduler = LRProviderFactory(settings=settings).create(
            provider=lr_scheduler,
        )
        
        if not self.scheduler:
            raise TypeError("invalid lr scheduler!")
        
        # sets the internal precision of float32 matrix multiplications.
        torch.set_float32_matmul_precision(self.settings.MATMUL_PRECISION)

        # initialize our model
        self.model = BaselinesProviderFactory(
            settings=settings
        ).create(
            provider=baseline, 
            resnet_pretrained=resnet_pretrained, 
            base_finetuned=base_finetuned, 
            base_freeze=base_freeze,
            temporal=person_temporal,
        )
        self.best_model = None
        self.best_f1 = [0.0]
        
        if not self.model:
            raise TypeError("invalid model type!")
        
        # move the model to the specified device  
        self.model.to(self.settings.DEVICE)
        
        if compile:
            # compile the model
            self.model.compile()

        # define the optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), fused=torch.cuda.is_available())

    
    def eval_model(self, eval_loader: DataLoader) -> tuple[List[float], List[float], List[float]]:

        # set the model to evaluation mode
        self.model.eval()
        with torch.inference_mode():
            
            running_loss, running_acc, running_f1 = [None], [None], [None]
            for batch in tqdm(eval_loader):
            
                with torch.autocast(device_type=self.settings.DEVICE, dtype=torch.bfloat16):
                    _, loss, acc, f1 = self.model(batch)
                
                if not running_loss[0]:
                    running_loss = [lossi.item() for lossi in loss]
                    running_acc  = [acci.item() for acci in acc]
                    running_f1   = [f1i.item() for f1i in f1]
                else:
                    running_loss = [running_loss[i] + loss[i].item() for i in range(len(running_loss))]
                    running_acc  = [running_acc[i] + acc[i].item() for i in range(len(running_acc))]
                    running_f1   = [running_f1[i] + f1[i].item() for i in range(len(running_f1))]
            
            running_loss = [running_loss[i] / len(eval_loader) for i in range(len(running_loss))]
            running_acc  = [running_acc[i] / len(eval_loader) for i in range(len(running_acc))]
            running_f1   = [running_f1[i] / len(eval_loader) for i in range(len(running_f1))]

        # set the model back to the training mode
        self.model.train()
        
        return running_loss, running_acc, running_f1

    def fit(self) -> nn.Module:

        # set the model to the training mode
        self.model.train()

        # intialize tensorboard SummaryWriter if we want to track the experiment
        if self.tensorboard_track:
            writer = SummaryWriter(log_dir=f"{self.model.tensorboard_path}/run_{time.strftime('%Y%m%d-%H%M%S')}")
        
        step = 0
        running_loss, running_acc, running_f1 = [None], [None], [None]
        val_loss, val_acc, val_f1 = [None], [None], [None]
        for epoch in range(self.settings.NUM_EPOCHS):

            running_loss, loss_accum, running_acc, running_f1 = [None], [None], [None], [None]
            for i, batch in enumerate(tqdm(self.train_loader)):
                
                # do the forward path
                with torch.autocast(device_type=self.settings.DEVICE, dtype=torch.bfloat16):
                    _, loss, acc, f1 = self.model(batch)

                # accumulate loss, accuracy, and f1 values on each mini batch
                if not running_loss[0]:
                    running_loss = [lossi.item() for lossi in loss]
                    running_acc  = [acci.item() for acci in acc]
                    running_f1   = [f1i.item() for f1i in f1]

                # divide the loss by the number of gradient accumulation steps
                loss = [lossi / self.settings.GRAD_ACCUM_STEPS for lossi in loss]

                # accumulate the loss
                if not loss_accum[0]:
                    loss_accum = [lossi.item() for lossi in loss]
                else:
                    loss_accum = [loss_accum[i] + loss[i].item() for i in range(len(loss_accum))]

                # calculate the gradients
                total_loss = torch.tensor(0.0).to(self.settings.DEVICE)
                for lossi in loss:
                    total_loss += lossi
                total_loss.backward()

                if (i + 1) % self.settings.GRAD_ACCUM_STEPS == 0:

                    # getting the updated learning rate
                    lr = self.scheduler.get_lr(step)

                    # updating our learning rate
                    for param in self.optimizer.param_groups:
                        param["lr"] = lr

                    # cliping gradients to avoid exploading gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    # weights updates
                    self.optimizer.step()

                    # zeroing all the gradients
                    self.optimizer.zero_grad()    

                    # increment number of gradient accumulation steps done
                    step += 1

                    # tracking the loss of each evaluation interval in the val
                    if step % self.settings.EVAL_INTERVALS == 0:
                        val_accum_loss, val_accum_acc, val_accum_f1 = self.eval_model(self.val_loader)
                        print(f"step {step}: train_loss: {loss_accum}, val_loss: {val_accum_loss} val_acc: {val_accum_acc}, val_f1: {val_accum_f1}")
                        
                        if self.best_model is None or self.best_f1[0] < val_accum_f1[0]:
                            self.best_f1 = val_accum_f1
                            self.best_model = copy.deepcopy(self.model).cpu()
                    else:
                        print(f"step {step}: train_loss: {loss_accum}")
                        
                    # zeroing loss accumulation after using it
                    loss_accum = [None]

            # handle any remaining gradients after the loop
            if (len(self.train_loader) % self.settings.GRAD_ACCUM_STEPS) != 0:

                lr = self.scheduler.get_lr(step)

                for param in self.optimizer.param_groups:
                    param["lr"] = lr
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                self.optimizer.zero_grad()

                step += 1

            # averaging our values by the number of mini batches
            running_loss = [running_loss[i] / len(train_loader) for i in range(len(running_loss))]
            running_acc  = [running_acc[i] / len(train_loader) for i in range(len(running_acc))]
            running_f1   = [running_f1[i] / len(train_loader) for i in range(len(running_f1))]

            # caclulate the overall loss, accuracy, and f1 on the eval set at the end of each epoch
            val_loss, val_acc, val_f1 = self.eval_model(self.val_loader)
            if self.best_model is None or self.best_f1[0] < val_f1[0]:
                self.best_f1 = val_f1
                self.best_model = copy.deepcopy(self.model).cpu()

            if self.tensorboard_track:
                # tracking losses and metrics values
                writer.add_scalar(tag="loss/train", scalar_value=running_loss, global_step=epoch)
                writer.add_scalar(tag="loss/val", scalar_value=val_loss, global_step=epoch)
                writer.add_scalar(tag="accuracy/train", scalar_value=running_acc, global_step=epoch)
                writer.add_scalar(tag="accuracy/val", scalar_value=val_acc, global_step=epoch)
                writer.add_scalar(tag="f1_score/train", scalar_value=running_f1, global_step=epoch)
                writer.add_scalar(tag="f1_score/val", scalar_value=val_f1, global_step=epoch)


            print(f"Epoch [{epoch + 1}/{self.settings.NUM_EPOCHS}]: train_loss: {running_loss}, train_acc: {running_acc}, train_f1: {running_f1}, val_loss: {val_loss} val_acc: {val_acc}, val_f1: {val_f1}")

        # test model if a test set was given
        if self.test_loader is not None:

            test_loss, test_acc, test_f1 = self.eval_model(eval_loader=self.test_loader)
            print(f"test_loss: {test_loss}, test_acc: {test_acc}, test_f1: {test_f1}")
            
            if self.tensorboard_track:

                # track metrics for each experiment
                writer.add_scalar(tag="loss/test", scalar_value=test_loss, global_step=0)
                writer.add_scalar(tag="accuracy/test", scalar_value=test_acc, global_step=0)
                writer.add_scalar(tag="f1_score/test", scalar_value=test_f1, global_step=0)

        if self.tensorboard_track:

            # save all the hyperparameters to the tensorbaord
            hparam_dict = {
                key: value
                for key, value in self.settings.model_dump().items()
                if (not isinstance(value, dict)) and (not isinstance(value, list))
            }

            writer.add_hparams(
                hparam_dict=hparam_dict,
                metric_dict={
                    "hparam/loss/train": running_loss,
                    "hparam/loss/val": val_loss,
                    "hparam/accuracy/train": running_acc,
                    "hparam/accuracy/val": val_acc,
                    "hparam/f1_score/train": running_f1,
                    "hparam/f1_score/val": val_f1,
                }
            )

            writer.close()

        return self.best_model
