'''

'''


from abc import abstractmethod
import torch
import numpy as np
import os
from . import speclib
import os

sm = torch.nn.Softmax(dim=1)

class Model():
    @abstractmethod
    def predict(self,data):
        pass

def load_torch_model(model_file, class_names_file=None,class_names_long_file=None, device=0, name=None, cluster_file='default'):

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(['%d' % i for i in dev_ids])
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(['%d' % i for i in range(device+1)])


    from_file = isinstance(model_file,str)

    if name is None and from_file:
        filename, file_extension = os.path.splitext(model_file)
        name = os.path.split(filename)[-1]
        self_name = name
    else:
        self_name = 'torchmodel'

    torch.cuda.set_device(device)
    if not isinstance(device,str):
        self_device = f'cuda:{device}'
    else:
        self_device=device
    
    if from_file:
        model = torch.jit.load(model_file,map_location=self_device)
        model.eval()
    else:
        model=model_file
    self_model = model.to(self_device)
    
    if class_names_file is None and from_file:
        filename, file_extension = os.path.splitext(model_file)
        class_names_file = filename+'_class_names.npy'

    if from_file:
        if os.path.exists(class_names_file):
            class_names = np.load(class_names_file)
            
            long_name_file = class_names_file.replace('class_names','class_names_long')
            if os.path.exists(long_name_file):
                class_names_long = np.load(long_name_file)
            else:
                print('No long name file found. Using short names.')
                class_names_long = class_names
        else:
            print('Cannot find a matching class names file. Expecting {}'.format(class_names_file))
            class_names = None
            class_names_long = None
    else:
        class_names = class_names_file
        if class_names_long_file is not None:
            class_names_long = class_names_long_file
        else:
            class_names_long = class_names

    self_clusters = speclib.LibClusters()
    if cluster_file is not None:
        if cluster_file == 'default':
            cluster_file =    os.path.abspath(os.path.join(os.path.dirname(__file__), 'library_clusters.json'))
        self_clusters.load(cluster_file)
        self_clusters.group([cnl.split('||')+[cn] for cn,cnl in zip(class_names,class_names_long)])
    else:
        self_clusters.from_dict(dict(zip(class_names,[cnl.split('||') for cnl in class_names_long])))

    return Torch_Model(self_model, class_names = class_names, name=self_name, clusters=self_clusters, device=device)

class Torch_Model(Model):
    # def __init__(self, model_file, class_names_file = None, device = 0, name=None, cluster_file=None):
    def __init__(self, model, class_names = None, name=None, clusters=None, device = 'cuda:0'): 
        self.model = model
        self.class_names = class_names
        self.name = name
        self.clusters = clusters
        self.device=device

    @property
    def class_names_long(self):
        return ['||'.join(self.clusters.cdict.inverse[self.clusters.cdict[k]]) for k in self.class_names]

    def predict(self, x, normalize=True):

        if x.ndim==1:
            x=x[np.newaxis,np.newaxis,:,np.newaxis]
        if x.ndim==2:
            x=x[:,np.newaxis,:,np.newaxis]

        if normalize:
            x = x/np.std(x,axis=2,keepdims=True)
            x = x-np.mean(x,axis=2,keepdims=True)

        if x.shape[0]>128:
            split = np.array_split(x, int(x.shape[0]/128),axis=0)
            y = np.concatenate([self._predict_batch(s) for s in split],axis=0)
        else:
            y = self._predict_batch(x)

        return y


    def predict_for_att(self, x, normalize=True):

        if x.ndim==1:
            x=x[np.newaxis,np.newaxis,:,np.newaxis]
        if x.ndim==2:
            x=x[:,np.newaxis,:,np.newaxis]

        if normalize:
            x = x/np.std(x,axis=2,keepdims=True)
            x = x-np.mean(x,axis=2,keepdims=True)

        if x.shape[0]>128:
            split = np.array_split(x, int(x.shape[0]/128),axis=0)
            y = np.concatenate([self._predict_batch_att(s) for s in split],axis=0)
        else:
            y = self._predict_batch_att(x)
        x = torch.Tensor(x)
        x = x.to(self.device)
        return x, y
    
    def _predict_batch_att(self, x):
        x=torch.Tensor(x)
        x=x.to(self.device)
        y=sm(self.model(x))
        return y
    
    
    def _predict_batch(self, x):
        x=torch.Tensor(x)
        x=x.to(self.device)
        y=sm(self.model(x)).detach().cpu().numpy()
        return y
    
    def predict_dict(self, raddict, normalize=True):
        [*keys],[*values] = zip(*raddict.items())
        names,scores = self.predict_names(np.asarray(values),normalize=normalize)
        result = {k:{'names':n,'scores':s} for k,n,s in zip(keys,list(names),list(scores))}
        # result = dict(zip(keys, zip(list(names),list(scores))))
        # result = dict()

        # for k in raddict:
        #     names,scores = self.predict_names(raddict[k],normalize=normalize)
        #     result[k]={'names':names,'scores':scores}
        return result

    
    def predict_names(self, x, normalize=True):
        y = self.predict(np.asarray(x),normalize=normalize)

        I = np.argsort(-y,axis=1)
        names = np.asarray([self.class_names[i] for i in I])
        scores = np.asarray([y1[i] for y1,i in zip(y,I)])
        return names, scores

    def predict_cube(self, cube, normalize=True, keep_top=10, whitener=None, cube_mean=None):
        (nx,ny,nb)=cube.shape
        labels = np.zeros((nx,ny,keep_top),dtype='int')
        scores = np.zeros((nx,ny,keep_top))

        if whitener is not None:
            whitener = whitener.T
        if cube_mean is not None:
            cube = cube - cube_mean

        for iy in range(ny):
            print(iy)
            line = np.squeeze(cube[:,iy,:])
            if whitener is not None:
                line = np.matmul(line,whitener)
            y = self.predict(line,normalize=normalize)
            # I = np.argsort(-y,axis=1)
            I = np.argpartition(y, -keep_top,axis=1)[:,-keep_top:]

            for ix in range(nx):
                I1 = I[ix,np.argsort(-y[ix,I[ix,:]])]
                labels[ix,iy,:]=I1
                scores[ix,iy,:]=y[ix,I1]
        
        return(labels,scores)



    
    def report(self, y, num_detects=5):

        for ii in range(y.shape[0]):
            print(f'Superpixel # {ii}')
            real_scores=y[ii,:]

            # Sort by score
            I = np.argsort(real_scores)[::-1].flatten()

            # Print top scores with target names
            for i in range(num_detects):
                pos = I[i]
                print('*{0}* : {1} : {2}'.format(self.class_names[pos],self.class_names_long[pos],real_scores[pos]))

    def predict_and_report(self, x):
        y=self.predict(x)
        self.report(y)
                
    def change_device(self, device):
        torch.cuda.set_device(self.device)
        self.model = self.model.to(f'cuda:{self.device}')


def get_model_stats(model_file,num_decimal=1):
    info_file = model_file.replace('.pt','_info.npy')
    if not os.path.exists(info_file):
        return dict()
    info = np.load(info_file,allow_pickle=True).item()
    stats = dict()
    stats['name'] = '_'.join(os.path.split(info_file)[-1].split('_')[:-1])
    if 'train_acc_history' in info:
        stats['epoch'] = len(info['train_acc_history'])-1
    else:
        stats['epoch'] = -1
    for k in info:
        if isinstance(info[k],list):
            if len(info[k]) > 0:
                stats[k]=round(float(info[k][-1]),num_decimal)
        else:
            stats[k]=info[k]
    return stats

    
def make_summary_string(model_file):
    info = get_model_stats(model_file)
    name = info['name']
    info_str = name + ' || '
    if 'train_acc_history' in info:
        acc = round(info['train_acc_history'])
        info_str += f'Tr={acc} '
    if 'test_acc_history' in info:
        acc = round(info['test_acc_history'])
        info_str += f'Te={acc} '
    if 'custom_acc_history' in info:
        acc = round(info['custom_acc_history'])
        info_str += f'B={acc} '
    
    return info_str
