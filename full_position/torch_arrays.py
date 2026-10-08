"""Small explicit Torch array adapter for the shared FP64 batched solver."""
import numpy as np


class TorchArrays:
    def __init__(self,device=0):
        import torch
        self.torch=torch; self.device=torch.device("cuda",device)
        self.float64,self.int32,self.int64=torch.float64,torch.int32,torch.int64
        self.inf,self.nan=np.inf,np.nan
        self.linalg=self.Linalg(self)
    def dtype(self,dtype):
        if dtype in (None,float,np.float64): return self.float64
        if dtype in (bool,np.bool_): return self.torch.bool
        if dtype in (int,np.int64): return self.int64
        return dtype
    def asarray(self,value,dtype=None):
        if isinstance(value,self.torch.Tensor):
            return value.to(device=self.device,dtype=self.dtype(dtype) if dtype is not None else value.dtype)
        return self.torch.as_tensor(value,dtype=self.dtype(dtype),device=self.device)
    def asnumpy(self,value): return value.detach().cpu().numpy()
    def zeros(self,shape,dtype=None): return self.torch.zeros(shape,dtype=self.dtype(dtype),device=self.device)
    def ones(self,shape,dtype=None): return self.torch.ones(shape,dtype=self.dtype(dtype),device=self.device)
    def full(self,shape,fill_value,dtype=None): return self.torch.full((shape,) if isinstance(shape,int) else shape,fill_value,dtype=self.dtype(dtype),device=self.device)
    def arange(self,n): return self.torch.arange(n,device=self.device)
    def eye(self,n): return self.torch.eye(n,dtype=self.float64,device=self.device)
    def stack(self,values,axis=0): return self.torch.stack(tuple(values),dim=axis)
    def sum(self,value,axis=None): return self.torch.sum(value,dim=axis)
    def max(self,value,axis=None): return self.torch.amax(value,dim=axis)
    def min(self,value,axis=None): return self.torch.amin(value,dim=axis)
    def argmin(self,value,axis=None): return self.torch.argmin(value,dim=axis)
    def all(self,value,axis=None): return self.torch.all(value,dim=axis)
    def any(self,value,axis=None): return self.torch.any(value,dim=axis)
    def minimum(self,a,b): return self.torch.minimum(self.asarray(a),self.asarray(b))
    def maximum(self,a,b): return self.torch.maximum(self.asarray(a),self.asarray(b))
    def clip(self,x,lo,hi): return self.torch.clamp(x,min=self.asarray(lo),max=self.asarray(hi))
    def where(self,condition,a,b): return self.torch.where(condition,self.asarray(a),self.asarray(b))
    def repeat(self,x,repeats,axis=0): return self.torch.repeat_interleave(x,repeats,dim=axis)
    def take_along_axis(self,x,indices,axis): return self.torch.take_along_dim(x,indices.to(self.int64),dim=axis)
    def diagonal(self,x,axis1=-2,axis2=-1): return self.torch.diagonal(x,dim1=axis1,dim2=axis2)
    def copy(self,x): return x.clone()
    def __getattr__(self,name): return getattr(self.torch,name)

    class Linalg:
        def __init__(self,arrays): self.arrays=arrays
        def __getattr__(self,name): return getattr(self.arrays.torch.linalg,name)
        def svd(self,x,full_matrices=True,compute_uv=True):
            if not compute_uv: return self.arrays.torch.linalg.svdvals(x)
            return self.arrays.torch.linalg.svd(x,full_matrices=full_matrices)
