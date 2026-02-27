import torch
from algos.base_algo import Base_Algo

class Unconditional(Base_Algo):
    def __init__(self, model, H_funcs, sigma_0, cls_fn=None):
        super().__init__(model, H_funcs, sigma_0, cls_fn)

    @ torch.enable_grad()
    def cal_x0(self, xt, t, at, at_next, classes=None):
        #xt.requires_grad_(True)
        if self.cls_fn == None:
            et = self.model(xt, t)
        else:
            et = self.model(xt, t, self.classes)
            et = et[:, :3]
            et = et - (1 - at).sqrt()[0,0,0,0] * self.cls_fn(xt,t,self.classes)
        if et.size(1) == 6:
            et = et[:, :3]
        # et = et.clip(-1, 1)
        self.et = et
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        x0_t = x0_t.clip(-1, 1)
        add_up = (1-at_next[0,0,0,0]).sqrt() * et
        return x0_t, add_up
    
    def map_back(self, x0_t, y_0, add_up, at_next, at):
        xt_next = at_next.sqrt() * x0_t + add_up
        return xt_next
