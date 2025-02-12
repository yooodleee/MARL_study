
import torch



class Transform:

    def transform(self, tensor):
        raise NotImplementedError
    
    
    def infer_output_info(self, vshape_in, dtype_in):
        raise NotImplementedError



