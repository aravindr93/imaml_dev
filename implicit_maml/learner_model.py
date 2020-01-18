import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import implicit_maml.utils as utils
from   torch.nn import functional as F

class Learner:
    def __init__(self, model, loss_function, inner_lr=1e-3, outer_lr=1e-2, GPU=False, inner_alg='gradient', outer_alg='adam'):
        self.model = model
        self.use_gpu = GPU
        if GPU:
            self.model.cuda()
        assert outer_alg == 'sgd' or 'adam'
        self.inner_opt = torch.optim.SGD(self.model.parameters(), lr=inner_lr)
        if outer_alg == 'adam':
            self.outer_opt = torch.optim.Adam(self.model.parameters(), lr=outer_lr, eps=1e-3)
        else:
            self.outer_opt = torch.optim.SGD(self.model.parameters(), lr=outer_lr)
        self.loss_function = loss_function
        assert inner_alg == 'gradient' # sqp unsupported in this version
        self.inner_alg = inner_alg

    def get_params(self):
        return torch.cat([param.data.view(-1) for param in self.model.parameters()], 0).clone()

    def set_params(self, param_vals):
        offset = 0
        for param in self.model.parameters():
            param.data.copy_(param_vals[offset:offset + param.nelement()].view(param.size()))
            offset += param.nelement()
            
    def set_outer_lr(self, lr):
        for param_group in self.outer_opt.param_groups:
            param_group['lr'] = lr
            
    def set_inner_lr(self, lr):
        for param_group in self.inner_opt.param_groups:
            param_group['lr'] = lr

    def regularization_loss(self, w_0, lam=0.0):
        """
        Add a regularization loss onto the weights
        The proximal term regularizes around the point w_0
        Strength of regularization is lambda
        lambda can either be scalar (type float) or ndarray (numpy.ndarray)
        """
        regu_loss = 0.0
        offset = 0
        regu_lam = lam if type(lam) == float or np.float64 else utils.to_tensor(lam)
        if w_0.dtype == torch.float16:
            try:
                regu_lam = regu_lam.half()
            except:
                regu_lam = np.float16(regu_lam)
        for param in self.model.parameters():
            delta = param.view(-1) - w_0[offset:offset + param.nelement()].view(-1)
            if type(regu_lam) == float or np.float64:
                regu_loss += 0.5 * regu_lam * torch.sum(delta ** 2)
            else:
                # import ipdb; ipdb.set_trace()
                param_lam = regu_lam[offset:offset + param.nelement()].view(-1)
                param_delta = delta * param_lam
                regu_loss += 0.5 * torch.sum(param_delta ** 2)
            offset += param.nelement()
        return regu_loss

    def get_loss(self, x, y, return_numpy=False):
        """
        Assume that x and y are torch tensors -- either in CPU or GPU (controlled externally)
        """
        yhat = self.model.forward(x)
        loss = self.loss_function(yhat, y)
        if return_numpy:
            loss = utils.to_numpy(loss).ravel()[0]
        return loss

    def predict(self, x, return_numpy=False):
        yhat = self.model.forward(utils.to_device(x, self.use_gpu))
        if return_numpy:
            yhat = utils.to_numpy(yhat)
        return yhat

    def learn_on_data(self, x, y, num_steps=10,
                      add_regularization=False,
                      w_0=None, lam=0.0):
        
        assert self.inner_alg == 'gradient' # or 'sqp' or 'adam' # TODO(Aravind): support sqp and adam 
        train_loss = []
        if self.inner_alg == 'gradient':
            for i in range(num_steps):
                self.inner_opt.zero_grad()
                tloss = self.get_loss(x, y)
                loss = tloss + self.regularization_loss(w_0, lam) if add_regularization else tloss
                loss.backward()
                self.inner_opt.step()
                train_loss.append(utils.to_numpy(tloss))

        return train_loss

    def learn_task(self, task, num_steps=10, add_regularization=False, w_0=None, lam=0.0):
        xt, yt = task['x_train'], task['y_train']
        return self.learn_on_data(xt, yt, num_steps, add_regularization, w_0, lam)

    def move_toward_target(self, target, lam=2.0):
        """
        Move slowly towards the target parameter value
        Default value for lam assumes learning rate determined by optimizer
        Useful for implementing Reptile
        """
        # we can implement this with the regularization loss, but regularize around the target point
        # and with specific choice of lam=2.0 to preserve the learning rate of inner_opt
        self.outer_opt.zero_grad()
        loss = self.regularization_loss(target, lam=lam)
        loss.backward()
        self.outer_opt.step()

    def outer_step_with_grad(self, grad, flat_grad=False):
        """
        Given the gradient, step with the outer optimizer using the gradient.
        Assumed that the gradient is a tuple/list of size compatible with model.parameters()
        If flat_grad, then the gradient is a flattened vector
        """
        check = 0
        for p in self.model.parameters():
            check = check + 1 if type(p.grad) == type(None) else check
        if check > 0:
            # initialize the grad fields properly
            dummy_loss = self.regularization_loss(self.get_params())
            dummy_loss.backward()  # this would initialize required variables
        if flat_grad:
            offset = 0
            grad = utils.to_device(grad, self.use_gpu)
            for p in self.model.parameters():
                this_grad = grad[offset:offset + p.nelement()].view(p.size())
                p.grad.copy_(this_grad)
                offset += p.nelement()
        else:
            for i, p in enumerate(self.model.parameters()):
                p.grad = grad[i]
        self.outer_opt.step()

    def matrix_evaluator(self, task, lam, regu_coef=1.0, lam_damping=10.0, x=None, y=None):
        """
        Constructor function that can be given to CG optimizer
        Works for both type(lam) == float and type(lam) == np.ndarray
        """
        if type(lam) == np.ndarray:
            lam = utils.to_device(lam, self.use_gpu)
        def evaluator(v):
            hvp = self.hessian_vector_product(task, v, x=x, y=y)
            Av = (1.0 + regu_coef) * v + hvp / (lam + lam_damping)
            return Av
        return evaluator

    def hessian_vector_product(self, task, vector, params=None, x=None, y=None):
        """
        Performs hessian vector product on the train set in task with the provided vector
        """
        if x is not None and y is not None:
            xt, yt = x, y
        else:
            xt, yt = task['x_train'], task['y_train']
        if params is not None:
            self.set_params(params)
        tloss = self.get_loss(xt, yt)
        grad_ft = torch.autograd.grad(tloss, self.model.parameters(), create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_ft])
        vec = utils.to_device(vector, self.use_gpu)
        h = torch.sum(flat_grad * vec)
        hvp = torch.autograd.grad(h, self.model.parameters())
        hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])
        return hvp_flat


def make_fc_network(in_dim=1, out_dim=1, hidden_sizes=(40,40), float16=False):
    non_linearity = nn.ReLU()
    model = nn.Sequential()
    model.add_module('fc_0', nn.Linear(in_dim, hidden_sizes[0]))
    model.add_module('nl_0', non_linearity)
    model.add_module('fc_1', nn.Linear(hidden_sizes[0], hidden_sizes[1]))
    model.add_module('nl_1', non_linearity)
    model.add_module('fc_2', nn.Linear(hidden_sizes[1], out_dim))
    if float16:
        return model.half()
    else:
        return model

    
def make_conv_network(in_channels, out_dim, task='Omniglot', filter_size=32):
    assert task == 'Omniglot' or 'MiniImageNet'
    model = nn.Sequential()
    
    if task == 'MiniImageNet':
        model = model_imagenet_arch(in_channels, out_dim, filter_size)
        
    elif task == 'Omniglot':
        num_filters = 64
        conv_stride = 2
        pool_stride = None
    
        model.add_module('conv1', nn.Conv2d(in_channels=in_channels, out_channels=num_filters,
                                            kernel_size=3, stride=conv_stride, padding=1))
        model.add_module('BN1', nn.BatchNorm2d(num_filters, track_running_stats=False))
        model.add_module('relu1', nn.ReLU())
        model.add_module('conv2', nn.Conv2d(in_channels=num_filters, out_channels=num_filters,
                                            kernel_size=3, stride=conv_stride, padding=1))
        model.add_module('BN2', nn.BatchNorm2d(num_filters, track_running_stats=False))
        model.add_module('relu2', nn.ReLU())
        model.add_module('pad2', nn.ZeroPad2d((0, 1, 0, 1)))
        model.add_module('conv3', nn.Conv2d(in_channels=num_filters, out_channels=num_filters,
                                            kernel_size=3, stride=conv_stride, padding=1))
        model.add_module('BN3', nn.BatchNorm2d(num_filters, track_running_stats=False))
        model.add_module('relu3', nn.ReLU())
        model.add_module('conv4', nn.Conv2d(in_channels=num_filters, out_channels=num_filters,
                                        kernel_size=3, stride=conv_stride, padding=1))
        model.add_module('BN4', nn.BatchNorm2d(num_filters, track_running_stats=False))
        model.add_module('relu4', nn.ReLU())
        model.add_module('flatten', Flatten())
        model.add_module('fc1', nn.Linear(2*2*num_filters, out_dim))
        
    for layer in [model.conv1, model.conv2, model.conv3, model.conv4, model.fc1]:
        torch.nn.init.xavier_uniform_(layer.weight, gain=1.73)
        try:
            torch.nn.init.uniform_(layer.bias, a=0.0, b=0.05)
        except:
            print("Bias layer not detected for layer:", layer)
            pass
    
    return model


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x

    
def model_imagenet_arch(in_channels, out_dim, num_filters=32, batch_norm=True, bias=True):
    raise NotImplementedError
