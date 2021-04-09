##################################################
# Imports
##################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
import torchvision
from sklearn import metrics
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Custom
from losses import kl_div
from metrics import accuracy
from models.resnet import ResNet34


##################################################
# Utils
##################################################

def squash(s, dim=-1):
    mag_sq = torch.sum(s**2, dim=dim, keepdim=True)
    mag = torch.sqrt(mag_sq)
    v = (mag_sq / (1.0 + mag_sq)) * (s / (mag+1e-8))
    return v


##################################################
# Residual layers
##################################################

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


##################################################
# Decoder
##################################################

class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, out_channels=3, 
                 num_mid_channels=512, lateral_dropout=0.8):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2, 
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self._conv_trans_3 = nn.ConvTranspose2d(in_channels=num_hiddens // 2, 
                                                out_channels=out_channels,
                                                kernel_size=3, 
                                                stride=1, padding=1)
        
        # Lateral connections
        self.conv_l3 = nn.Sequential(
            nn.Dropout(lateral_dropout),
            nn.Conv2d(num_mid_channels // 2, num_hiddens, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )
        self.conv_l2 = nn.Sequential(
            nn.Dropout(lateral_dropout),
            nn.Conv2d(num_hiddens, num_hiddens // 2, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )
        self.conv_l1 = nn.Sequential(
            nn.Dropout(lateral_dropout),
            nn.Conv2d(num_hiddens // 2, num_hiddens // 2, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
        )
        

    def forward(self, inputs, x_l1=None, x_l2=None, x_l3=None):
        x = self._conv_1(inputs)
        if x_l3 is not None:
            x = x + self.conv_l3(x_l3)
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        if x_l2 is not None:
            x = x + self.conv_l2(x_l2)
        
        x = self._conv_trans_2(x)
        x = F.relu(x)
        if x_l1 is not None:
            x = x + self.conv_l1(x_l1)
            
        x = self._conv_trans_3(x)
        return x 


##################################################
# DenseCapsule
##################################################

class DenseCapsule(nn.Module):
    """
    The dense capsule layer. It is similar to Dense (FC) layer. Dense layer has `in_num` inputs, each is a scalar, the
    output of the neuron from the former layer, and it has `out_num` output neurons. DenseCapsule just expands the
    output of the neuron from scalar to vector. So its input size = [None, in_num_caps, in_dim_caps] and output size = \
    [None, out_num_caps, out_dim_caps]. For Dense Layer, in_dim_caps = out_dim_caps = 1.
    :param in_num_caps: number of cpasules inputted to this layer
    :param in_dim_caps: dimension of input capsules
    :param out_num_caps: number of capsules outputted from this layer
    :param out_dim_caps: dimension of output capsules
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, routings=3):
        super(DenseCapsule, self).__init__()
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.routings = routings
        self.weight = nn.Parameter(0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))

    def forward(self, x):
        device = next(self.parameters()).device
        # x.size=[batch, in_num_caps, in_dim_caps]
        # expanded to    [batch, 1,            in_num_caps, in_dim_caps,  1]
        # weight.size   =[       out_num_caps, in_num_caps, out_dim_caps, in_dim_caps]
        # torch.matmul: [out_dim_caps, in_dim_caps] x [in_dim_caps, 1] -> [out_dim_caps, 1]
        # => x_hat.size =[batch, out_num_caps, in_num_caps, out_dim_caps]
        x_hat = torch.squeeze(torch.matmul(self.weight, x[:, None, :, :, None]), dim=-1)

        # In forward pass, `x_hat_detached` = `x_hat`;
        # In backward, no gradient can flow from `x_hat_detached` back to `x_hat`.
        x_hat_detached = x_hat.detach()

        # The prior for coupling coefficient, initialized as zeros.
        # b.size = [batch, out_num_caps, in_num_caps]
        b = Variable(torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps)).to(device)
        #if x.get_device() > 0:
        #    b = b.to(x.get_device())

        assert self.routings > 0, 'The \'routings\' should be > 0.'
        for i in range(self.routings):
            # c.size = [batch, out_num_caps, in_num_caps]
            c = F.softmax(b, dim=1)

            # At last iteration, use `x_hat` to compute `outputs` in order to backpropagate gradient
            if i == self.routings - 1:
                # c.size expanded to [batch, out_num_caps, in_num_caps, 1           ]
                # x_hat.size     =   [batch, out_num_caps, in_num_caps, out_dim_caps]
                # => outputs.size=   [batch, out_num_caps, 1,           out_dim_caps]
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat))  # alternative way
            else:  # Otherwise, use `x_hat_detached` to update `b`. No gradients flow on this path.
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat_detached))  # alternative way

                # outputs.size       =[batch, out_num_caps, 1,           out_dim_caps]
                # x_hat_detached.size=[batch, out_num_caps, in_num_caps, out_dim_caps]
                # => b.size          =[batch, out_num_caps, in_num_caps]
                b = b + torch.sum(outputs * x_hat_detached, dim=-1)

        return torch.squeeze(outputs, dim=-2)


##################################################
# VaeCap class
##################################################

class VaeCap(torch.nn.Module):
    def __init__(self, planes, in_dim_caps, out_num_caps, out_dim_caps, spatial_size=4, t_mu_shift=1.0, z_dim=64, 
                 debug=False):
        super(VaeCap, self).__init__()

        self.in_dim_caps = in_dim_caps
        self.out_dim_caps = out_dim_caps
        self.t_mu_shift = t_mu_shift
        self.conv_pose = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True), # Maybe omit this activation
        )
        self.dense_capsule = DenseCapsule(in_num_caps=(spatial_size ** 2) * (planes // in_dim_caps), 
                                          in_dim_caps=in_dim_caps, out_num_caps=out_num_caps, 
                                          out_dim_caps=out_dim_caps, routings=3)
        self.fc_mean = nn.Sequential(
            nn.Linear(out_dim_caps, 1024),
            nn.ReLU(),
            nn.Linear(1024, z_dim),
            #nn.Linear(out_dim_caps, z_dim),
        )
        self.fc_var = nn.Sequential(
            nn.Linear(out_dim_caps, z_dim),
            nn.Softplus(),
        )
        #self.dense_capsule_mean = DenseCapsule(in_num_caps=(spatial_size ** 2) * (planes // in_dim_caps), 
        #                                  in_dim_caps=in_dim_caps, out_num_caps=out_num_caps, 
        #                                  out_dim_caps=out_dim_caps, routings=3)
        #self.dense_capsule_var = DenseCapsule(in_num_caps=(spatial_size ** 2) * (planes // in_dim_caps), 
        #                                  in_dim_caps=in_dim_caps, out_num_caps=out_num_caps, 
        #                                  out_dim_caps=out_dim_caps, routings=3)
        self.debug = debug
        if self.debug:
            print(f'Num in caps: {(spatial_size ** 2) * (planes // in_dim_caps)}')
            print(f'In caps size: {in_dim_caps}')
            print(f'Num out caps: {out_num_caps}')
            print(f'Out caps size: {out_dim_caps}')

    def forward(self, x):
        device = next(self.parameters()).device
        capconv = self.conv_pose(x) # [batch_size, planes, spatial_size, spatial_size]
        if self.debug:
            print('capconv', capconv.shape)
        batch_size, planes, spatial_size, spatial_size = capconv.shape
        pose = capconv.view(batch_size, planes // self.in_dim_caps, self.in_dim_caps, spatial_size, spatial_size)
        if self.debug:
            print('pose0', pose.shape)
        pose = pose.permute(0, 1, 3, 4, 2).contiguous() # [batch_size, n_groups, spatial_size, spatial_size, in_dim_caps]
        pose = pose.view(batch_size, -1, self.in_dim_caps)
        pose = squash(pose) # [batch_size, (spatial_size ** 2) * (planes // in_dim_caps), in_dim_caps]
        if self.debug:
            print('pose1', pose.shape)
        out = self.dense_capsule(pose) # [bs, num_classes, out_dim_caps]
        if self.debug:
            print(out.shape)

        # Variational part
        z_mu = self.fc_mean(out) # # [bs, num_classes, z_dim]
        z_var = self.fc_var(out) + 1e-8
        #z_mu = self.dense_capsule_mean(pose)
        #z_var = F.softplus(self.dense_capsule_var(pose)) + 1e-8
        #print(pose.shape)

        #z_var = self.fc_var(pose) + 1e-8
        z = self._reparametrization_trick(z_mu, z_var, device=device)
        return z, z_mu, z_var

    def _reparametrization_trick(self, x_mean, x_var, device='cpu'):
        if self.training:
            eps = torch.randn(x_mean.shape).to(device)
            z = x_mean + eps * (x_var.pow(0.5))
        else:
            z = x_mean
        return z


##################################################
# CVAECapOsr Model
##################################################

class CVAECapOSR(pl.LightningModule):
    def __init__(self, num_classes, in_channels, in_dim_caps=16, out_dim_caps=32, t_mu_shift=1.0, 
                 t_var_scale=1.0, z_dim=128, alpha=1.0, beta=100.0, margin=10.0, lr=3e-4):
        super(CVAECapOSR, self).__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.spatial_size = 8 # Spatial size of the middle feature [bs, ch, 8, 8]
        self.enc = ResNet34(in_channels=in_channels)
        self.vae_cap = VaeCap(
            planes=512, in_dim_caps=in_dim_caps, out_num_caps=num_classes, out_dim_caps=out_dim_caps, 
            spatial_size=self.spatial_size, t_mu_shift=t_mu_shift, z_dim=z_dim,
        )
        self.fc = nn.Linear(z_dim * num_classes, 64 * self.spatial_size * self.spatial_size)
        self.dec = Decoder(64, num_hiddens=z_dim, num_residual_layers=2, num_residual_hiddens=32, 
                           out_channels=in_channels)
        self.t_mean = nn.Embedding(self.num_classes, self.num_classes * z_dim)
        self.t_var = nn.Embedding(self.num_classes, self.num_classes * z_dim)
        self.t_mean, self.t_var = self._init_targets(t_mu_shift, t_var_scale)
        
        # Loss
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.lr = lr
        
    def _init_targets(self, t_mu_shift, t_var_scale):
        device = self.t_mean.weight.device
        t_mean_init = 1.0 * F.one_hot(torch.arange(self.num_classes), self.num_classes)
        t_mean_init = t_mean_init.unsqueeze(-1).repeat(1, 1, self.z_dim).view(self.num_classes, -1)
        t_mean_init = t_mean_init * t_mu_shift
        t_mean = nn.Embedding.from_pretrained(t_mean_init.to(device), freeze=False)
        t_var_init = torch.ones(self.num_classes, self.num_classes * self.z_dim)
        t_var_init = t_var_init * t_var_scale
        t_var = nn.Embedding.from_pretrained(t_var_init.to(device), freeze=False)
        return t_mean, t_var
    
    def _cross_kl_div(self, z_mu, z_var, detach_inputs=False, detach_targets=False):
        """
        Compute kl divergence between the variation capsule distribution defined by z_mu and z_var with all the 
        targets.
        
        Args:
            z_mu: tensor of shape [bs, num_classes, z_dim].
            z_var: tensor of shape [bs, num_classes, z_dim].
            detach_inputs: bool.
            detach_targets: bool.
            
        Ouput:
            kl: tensor of shape [bs, num_classes]
        """
        B, num_classes, _ = z_mu.shape
        kl = []
        t_idxs = torch.arange(num_classes).unsqueeze(0).repeat(B, 1).to(z_mu.device)
        t_means = self.t_mean(t_idxs).view(B, num_classes, num_classes, -1) # [bs, num_classes, num_classes, z_dim]
        t_vars = self.t_var(t_idxs).view(B, num_classes, num_classes, -1) # [bs, num_classes, num_classes, z_dim]
        
        # Detach inputs
        if detach_inputs:
            z_mu = z_mu.detach()
            z_var = z_var.detach()
            
        # Detach targets
        if detach_inputs:
            t_means = t_means.detach()
            t_vars = t_vars.detach()
            
        for t_mean_i, t_var_i in zip(t_means.permute(1, 0, 2, 3), t_vars.permute(1, 0, 2, 3)):
            kl_i = kl_div(z_mu, z_var, t_mean_i, t_var_i) # [bs, num_classes]
            kl += [kl_i]
        kl = torch.stack(kl, 1).mean(1) # [bs, num_classes]
        return kl
    
    def cross_kl_div(self, z_mu, z_var, detach_inputs=False, detach_targets=False):
        """
        Compute kl divergence between the variation capsule distribution defined by z_mu and z_var with all the 
        targets.
        
        Args:
            z_mu: tensor of shape [bs, num_classes, z_dim].
            z_var: tensor of shape [bs, num_classes, z_dim].
            detach_inputs: bool.
            detach_targets: bool.
            
        Ouput:
            kl: tensor of shape [bs, num_classes]
        """
        B, num_classes, _ = z_mu.shape
        kl = []
        t_idxs = torch.arange(num_classes).unsqueeze(0).repeat(B, 1).to(z_mu.device)
        t_means = self.t_mean(t_idxs) # [bs, num_classes, num_classes * z_dim]
        t_vars = self.t_var(t_idxs) # [bs, num_classes, num_classes * z_dim]
        
        # Detach inputs
        if detach_inputs:
            z_mu = z_mu.detach()
            z_var = z_var.detach()
            
        # Detach targets
        if detach_targets:
            t_means = t_means.detach()
            t_vars = t_vars.detach()
            
        for t_mean_i, t_var_i in zip(t_means.permute(1, 0, 2), t_vars.permute(1, 0, 2)):
            kl_i = kl_div(torch.flatten(z_mu, 1), torch.flatten(z_var, 1), t_mean_i, t_var_i) # [bs]
            kl += [kl_i]
        kl = torch.stack(kl, 1) # [bs, num_classes]
        return kl
        
    def forward(self, x, y_str=None):
        
        # Process label
        if y_str is None:
            y = None
        else:
            y = np.array([int(y_i.split('_')[-1]) for y_i in y_str])
            y = torch.from_numpy(y).to(x.device)
        
        # Encoding
        enc_out = self.enc(x)
        x_f = enc_out['x_f']
        B, d_c, d_h, d_w = x_f.shape
        
        # Latent
        z, z_mu, z_var = self.vae_cap(x_f)
        
        # Target selection
        kl = self.cross_kl_div(z_mu, z_var, detach_targets=True)
        logits = - torch.log(kl)
        y_hat = F.softmax(logits, -1)
        if y is None:
            z = z + y_hat.unsqueeze(-1).repeat(1, 1, z.shape[-1])
        else:
            z = z + F.one_hot(y, self.num_classes).unsqueeze(-1).repeat(1, 1, z.shape[-1])
        
        # Latent
        z_flat = torch.flatten(z, 1)
        z_flat = F.relu(self.fc(z_flat))
        z_chw = z_flat.view(B, -1, d_h, d_w)
        
        # Decoding
        x_hat = self.dec(z_chw, x_l1=enc_out['x_l1'], x_l2=enc_out['x_l2'], x_l3=enc_out['x_l3'])
        x_hat = torch.sigmoid(x_hat)
        
        # Out
        out = {
            'x_hat': x_hat,
            'logits': logits,
            'z': z,
            'z_mu': z_mu,
            'z_var': z_var,
        }
        return out
    
    def training_step(self, batch, idx_batch):
        x, y_str = batch
        y = torch.from_numpy(np.array([int(y_i.split('_')[-1]) for y_i in y_str])).to(x.device)
        y_oh = F.one_hot(y, self.num_classes)
        
        # Forward with teacher forcing
        preds = self(x, y_str=y_str)
        t_mu, t_var = self.t_mean(y), self.t_var(y)
        t_mu = t_mu.view(preds['z_mu'].shape)
        t_var = t_var.view(preds['z_var'].shape)
        
        # Loss
        loss_kl = kl_div(preds['z_mu'], preds['z_var'], t_mu.detach(), t_var.detach()).mean(-1).mean(0)
        loss_contr = self.cross_kl_div(preds['z_mu'], preds['z_var'], detach_inputs=True)
        loss_contr = torch.where(y_oh < 1, loss_contr, torch.zeros_like(loss_contr))
        loss_contr = F.relu(self.margin - loss_contr)
        loss_contr = (loss_contr / (self.num_classes - 1)).sum(-1).mean(0)
        loss_rec = F.binary_cross_entropy(preds['x_hat'], x) #F.mse_loss(x, preds['x_hat'])
        loss = loss_kl + self.beta * loss_rec + self.alpha * loss_contr
        
        # Metrics
        acc = accuracy(preds['logits'].detach(), y)
        
        # Cache
        self.log('train_loss_kl', loss_kl)
        self.log('train_loss_contr', loss_contr)
        self.log('train_loss_rec', loss_rec)
        self.log('train_loss', loss)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, idx_batch):
        x, y_str = batch
        
        y = torch.from_numpy(np.array([int(y_i.split('_')[-1]) for y_i in y_str])).to(x.device)
        y_oh = F.one_hot(y, self.num_classes)
        
        # Forward with teacher forcing
        preds = self(x)
        t_mu, t_var = self.t_mean(y), self.t_var(y)
        t_mu = t_mu.view(preds['z_mu'].shape)
        t_var = t_var.view(preds['z_var'].shape)
        
        # Loss
        loss_kl = kl_div(preds['z_mu'], preds['z_var'], t_mu.detach(), t_var.detach()).mean(-1).mean(0)
        loss_contr = self.cross_kl_div(preds['z_mu'], preds['z_var'], detach_inputs=True)
        loss_contr = torch.where(y_oh < 1, loss_contr, torch.zeros_like(loss_contr))
        loss_contr = F.relu(self.margin - loss_contr)
        loss_contr = (loss_contr / (self.num_classes - 1)).sum(-1).mean(0)
        loss_rec = F.binary_cross_entropy(preds['x_hat'], x)
        loss = loss_kl + self.beta * loss_rec + self.alpha * loss_contr
        
        # Metrics
        acc = accuracy(preds['logits'].detach(), y)
        
        # Cache
        self.log('validation_loss_kl', loss_kl)
        self.log('validation_loss_contr', loss_contr)
        self.log('validation_loss_rec', loss_rec)
        self.log('validation_loss', loss)
        self.log('validation_acc', acc, prog_bar=True)
        if idx_batch == 0:
            tb_log = self.logger.experiment
            x_hat_grid = torchvision.utils.make_grid(preds['x_hat'][:16], nrow=4, padding=0)
            tb_log.add_image('validation_x_hat', x_hat_grid.detach().cpu(), self.current_epoch)
        return loss
    
    def test_step(self, batch, idx_batch):
        x, y_str = batch
        y_ku = torch.from_numpy(np.array([0 if y_i.split('_')[0] == 'k' else 1 for y_i in y_str])).to(x.device)
        
        # Forward with teacher forcing
        preds = self(x)
        probs = torch.max(F.softmax(preds['logits'], -1), -1)[0]
        
        tb_log = self.logger.experiment
        x_hat_grid = torchvision.utils.make_grid(preds['x_hat'][:16], nrow=4, padding=0)
        tb_log.add_image('test_x_hat', x_hat_grid.detach().cpu(), idx_batch)
        return torch.stack([probs, y_ku], -1)
    
    def test_epoch_end(self, test_outputs):
        auroc = pl.metrics.classification.AUROC(pos_label=0)
        probs, y_ku = torch.cat(test_outputs, 0).T
        y_ku = y_ku.to(torch.int64)
        self.log('test_auroc', auroc(probs, y_ku), prog_bar=True)
        return
    
    def configure_optimizers(self):
        optim = Adam(self.parameters(), self.lr)
        lr_sched = ReduceLROnPlateau(
            optim, mode='max', factor=0.5, patience=5, verbose=1, 
        )
        lr_sched = {
            'scheduler': lr_sched,
            'reduce_on_plateau': True,
            'monitor': 'validation_acc'
        }
        return [optim], [lr_sched]

def get_model(args, data_info):
    model_args = {
        'num_classes': len(args.known_classes), 
        'in_channels': data_info['channels'], 
        'in_dim_caps': args.in_dim_caps, 
        'out_dim_caps': args.out_dim_caps, 
        't_mu_shift': args.t_mu_shift,
        't_var_scale': args.t_var_scale, 
        'z_dim': args.z_dim, 
        'alpha': args.alpha, 
        'beta': args.beta, 
        'margin': args.margin, 
        'lr': args.lr,
    }
    model = CVAECapOSR(**model_args)
    if len(args.checkpoint) > 0:
        model = model.load_from_checkpoint(args.checkpoint, num_classes=model_args['num_classes'], 
                                           in_channels=model_args['in_channels'])
        print(f'Loaded model checkpoint {args.checkpoint}')
    return model
