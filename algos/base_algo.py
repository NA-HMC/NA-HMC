from abc import ABC, abstractmethod

class Base_Algo(ABC):
    def __init__(self, model, H_funcs, sigma_0, cls_fn=None):
        self.model = model
        self.H_funcs = H_funcs
        self.sigma_0 = sigma_0
        self.cls_fn = cls_fn
    
    @abstractmethod
    def cal_x0(self, xt, t, at, at_next, classes):
        pass
    
    @abstractmethod
    def map_back(self, x0_t, y_0, add_up, at_next, at):
        pass
