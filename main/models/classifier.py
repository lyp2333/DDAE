import pytorch_lightning as pl
import torch
import torch.nn as nn
from main.util import *


class Classifier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Classifier, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.basic_block = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels, affine=True, track_running_stats=True),
        )
        self.relu2 = nn.ReLU()

    def forward(self, input):
        h = self.basic_block(input)
        return input + self.relu2(h)


class ClsModel(pl.LightningModule):
    def __init__(self,
                 encoder,
                 gae_pl,
                 gsl,
                 lr,
                 weight_decay,
                 d_type,
                 train_mode="gae_out",
                 attr_to_classify='id',
                 eval_attr_to_classify='id',
                 z_mean_std=None,
                 ):
        super(ClsModel, self).__init__()

        self.save_hyperparameters(ignore=["encoder", "gae_pl", "gsl"])
        self.encoder = encoder
        self.gae_pl = gae_pl
        self.GSL = gsl
        self.lr = lr
        self.train_mode = train_mode
        self.weight_decay = weight_decay
        self.z_mean_std = z_mean_std if z_mean_std is not None else None
        self.attr_to_classify = attr_to_classify
        self.eval_attr_to_classify = eval_attr_to_classify
        self.loss_func = nn.CrossEntropyLoss()

        self.acc_add = 0
        # eval
        self.acc_list = []
        if 'ilab' in d_type:
            self.dim_slices = {'id': slice(0, 60),
                               'back': slice(60, 80),
                               'pose': slice(80, 100)}
            if self.eval_attr_to_classify == 'id':
                self.num_classes = 10
            elif self.eval_attr_to_classify == 'back':
                self.num_classes = 111
            elif self.eval_attr_to_classify == 'pose':
                self.num_classes = 6
            else:
                raise AttributeError('Invalid attribute to classify')
            self.dim_slice = self.dim_slices[self.attr_to_classify]
            self.in_channels = self.dim_slice.stop - self.dim_slice.start if self.train_mode != 'encoder_out' else 512

            setattr(self, f'{self.attr_to_classify}_classifier', nn.Linear(self.in_channels, self.num_classes))
        else:
            raise NotImplementedError()
        self.classifier = getattr(self, f'{attr_to_classify}_classifier')

    def state_dict(self, *args, **kwargs):
        out = {}
        for k, v in super().state_dict(*args, **kwargs).items():
            if k.startswith('encoder.'):
                pass
            elif k.startswith('Gae_pl.'):
                pass
            elif k.startswith('GSL.'):
                pass
            else:
                out[k] = v
        return out

    def load_state_dict(self, state_dict, strict: bool = None):
        # change the default strict => False

        strict = False if strict is None else strict

        return super().load_state_dict(state_dict, strict=strict)

    def training_step(self, batch, batch_idx):

        terms = {}
        imgs = batch['img']
        batch_size, *_ = imgs.shape
        labels = batch[f'{self.eval_attr_to_classify}_label']
        with torch.no_grad():
            if self.train_mode == "encoder_out":
                z_original = self.encoder(imgs)
                z_for_classify = z_original
            elif self.train_mode == "gae_out":
                z_original = self.encoder(imgs)
                z_in_gae = (z_original - self.z_mean_std[0]) / self.z_mean_std[1] if self.z_mean_std is not None else z_original
                z_for_classify = self.gae_pl.get_latent_code(z_in_gae)
            elif self.train_mode == "gsl_out":
                z_for_classify = self.GSL.get_latent_code(imgs)
            else:
                raise NotImplementedError()

        pred = self.classifier(z_for_classify) if self.train_mode == 'encoder_out' else self.classifier(z_for_classify[:, self.dim_slices[self.attr_to_classify]])
        train_acc = torch.sum(torch.where(torch.argmax(pred, dim=-1) == labels, 1, 0)) / batch_size
        self.acc_add += train_acc
        loss = self.loss_func(pred, labels)
        terms['loss'] = loss
        terms['acc'] = train_acc
        self.log('loss', loss, prog_bar=True)
        self.log('acc', train_acc, prog_bar=True)
        self.log('avg_acc', self.acc_add / self.global_step, prog_bar=True)
        return terms

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.classifier.parameters(),
                                 lr=self.lr,
                                 weight_decay=self.weight_decay)
        return optim
