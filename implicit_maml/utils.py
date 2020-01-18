import os
import numpy as np
import scipy
import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle
import csv


def to_cuda(x):
    try:
        return x.cuda()
    except:
        return torch.from_numpy(x).float().cuda()


def to_tensor(x):
    if type(x) == np.ndarray:
        return torch.from_numpy(x).float()
    elif type(x) == torch.Tensor:
        return x
    else:
        print("Type error. Input should be either numpy array or torch tensor")
    

def to_device(x, GPU=False):
    if GPU:
        return to_cuda(x)
    else:
        return to_tensor(x)
    
    
def to_numpy(x):
    if type(x) == np.ndarray:
        return x
    else:
        try:
            return x.data.numpy()
        except:
            return x.cpu().data.numpy()
        

def cg_solve(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10, x_init=None):
    """
    Goal: Solve Ax=b equivalent to minimizing f(x) = 1/2 x^T A x - x^T b
    Assumption: A is PSD, no damping term is used here (must be damped externally in f_Ax)
    Algorithm template from wikipedia
    Verbose mode works only with numpy
    """
       
    if type(b) == torch.Tensor:
        x = torch.zeros(b.shape[0]) if x_init is None else x_init
        x = x.to(b.device)
        if b.dtype == torch.float16:
            x = x.half()
        r = b - f_Ax(x)
        p = r.clone()
    elif type(b) == np.ndarray:
        x = np.zeros_like(b) if x_init is None else x_init
        r = b - f_Ax(x)
        p = r.copy()
    else:
        print("Type error in cg")

    fmtstr = "%10i %10.3g %10.3g %10.3g"
    titlestr = "%10s %10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm", "obj fn"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose:
            obj_fn = 0.5*x.dot(f_Ax(x)) - 0.5*b.dot(x)
            norm_x = torch.norm(x) if type(x) == torch.Tensor else np.linalg.norm(x)
            print(fmtstr % (i, r.dot(r), norm_x, obj_fn))

        rdotr = r.dot(r)
        Ap = f_Ax(p)
        alpha = rdotr/(p.dot(Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        newrdotr = r.dot(r)
        beta = newrdotr/rdotr
        p = r + beta * p

        if newrdotr < residual_tol:
            # print("Early CG termination because the residual was small")
            break

    if callback is not None:
        callback(x)
    if verbose: 
        obj_fn = 0.5*x.dot(f_Ax(x)) - 0.5*b.dot(x)
        norm_x = torch.norm(x) if type(x) == torch.Tensor else np.linalg.norm(x)
        print(fmtstr % (i, r.dot(r), norm_x, obj_fn))
    return x


def smooth_vector(vec, window_size=25):
    svec = vec.copy()
    if vec.shape[0] < window_size:
        for i in range(vec.shape[0]):
            svec[i,:] = np.mean(vec[:i, :], axis=0)
    else:   
        for i in range(window_size, vec.shape[0]):
            svec[i,:] = np.mean(vec[i-window_size:i, :], axis=0)
    return svec
    
    
def save_data(agent, train_curve, other_data, save_file, itr=None):
    data = dict(agent=agent,
                losses=train_curve,
                other_data=other_data,
               )
    pickle_file_name = save_file + '.pickle'
    pickle.dump(data, open(pickle_file_name, 'wb'))
    
    plot_file_name = save_file + '.png'
    plt.figure(figsize=(10,6))
    if itr != None:
        plt.plot(smooth_vector(train_curve[:itr]), lw=2)
    else:
        plt.plot(smooth_vector(train_curve), lw=2)
    plt.xlabel('Meta (outer) iterations')
    plt.ylabel('Loss')
    plt.ylim([0.0, 5.0])
    plt.legend(['Train pre-adapt', 'Test pre-adapt', 'Train post-adapt', 'Test post-adapt'], loc=1)
    plt.savefig(plot_file_name, dpi=100)

    
def measure_accuracy(task, model, train=False):
    if train is True:
        x, y = task['x_train'], task['y_train']
    else:
        x, y = task['x_val'], task['y_val']
    y_hat = model.predict(x, return_numpy = True)
    batch_size = y.shape[0]
    predict_label = np.argmax(y_hat, axis=1)
    try:
        correct = np.sum(predict_label == y.cpu().data.numpy())
    except:
        correct = np.sum(predict_label == y.data.numpy())
    return correct * 100.0 / batch_size

    
class DataLog:

    def __init__(self):
        self.log = {}
        self.max_len = 0
        
    def log_exp_args(self, parsed_args):
        args = vars(parsed_args) # makes it a dictionary
        for k in args.keys():
            self.log_kv(k, args[k])

    def log_kv(self, key, value):
        # logs the (key, value) pair
        if key not in self.log:
            self.log[key] = []
        self.log[key].append(value)
        if len(self.log[key]) > self.max_len:
            self.max_len = self.max_len + 1

    def save_log(self, save_path=None):
        save_path = self.log['save_dir'][-1] if save_path is None else save_path
        pickle.dump(self.log, open(save_path+'/log.pickle', 'wb'))
        with open(save_path+'/log.csv', 'w') as csv_file:
            fieldnames = self.log.keys()
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in range(self.max_len):
                row_dict = {}
                for key in self.log.keys():
                    if row < len(self.log[key]):
                        row_dict[key] = self.log[key][row]
                writer.writerow(row_dict)

    def get_current_log(self):
        row_dict = {}
        for key in self.log.keys():
            row_dict[key] = self.log[key][-1]
        return row_dict

    def read_log(self, log_path):
        with open(log_path) as csv_file:
            reader = csv.DictReader(csv_file)
            listr = list(reader)
            keys = reader.fieldnames
            data = {}
            for key in keys:
                data[key] = []
            for row in listr:
                for key in keys:
                    try:
                        data[key].append(eval(row[key]))
                    except:
                        None
        self.log = data
