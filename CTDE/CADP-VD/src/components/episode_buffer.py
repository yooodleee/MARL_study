import torch
import numpy as np
from types import SimpleNamespace as SN




class EpisodeBatch:

    def __init__(
            self,
            scheme,
            groups,
            batch_size,
            max_seq_length,
            data=None,
            preprocess=None,
            device="cpu"):
        
        self.scheme = scheme
        self.groups = groups
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.preprocess = {} if preprocess is None else preprocess
        self.device = device

        if data is not None:
            self.data = data
        
        else:
            self.data = SN()
            self.data.transition_data = {}
            self.data.episode_data = {}
            self._setup_data(self.scheme, self.groups, batch_size, max_seq_length, self.preprocess)
    

    def _setup_data(
            self,
            scheme,
            groups,
            batch_size,
            max_seq_length,
            preprocess):
        
        if preprocess is not None:
            for k in preprocess:
                assert k in scheme

                new_k = preprocess[k][0]
                transforms = preprocess[k][1]

                vshape = self.scheme[k]["vshape"]
                dtype = self.scheme[k]["dtype"]

                for transform in transforms:
                    vshape, dtype = transform.infer_output_info(vshape, dtype)
                
                self.scheme[new_k] = {
                    "vshape": vshape,
                    "dtype": dtype
                }

                if "group" in self.scheme[k]:
                    self.scheme[new_k]["group"] = self.scheme[k]["group"]
                
                if "episode_const" in self.scheme[k]:
                    self.scheme[new_k]["episode_const"] = self.scheme[k]["episode_const"]
        

        assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update(
            {
                "filled": {"vshape": (1,), "dtype": torch.long}
            }
        )

        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, "Scheme must define vshape for {}".format(field_key)

            vshape = field_info["vshape"]
            episode_const = field_info.get("episode_const", False)
            group = field_info.get("group", None)
            dtype = field_info.get("dtype", torch.float32)

            if isinstance(vshape, int):
                vshape = (vshape,)
            
            if group:
                assert group in groups, "Group {} must have its number of members defined in _groups_".format(group)
                shape = (groups[group], *vshape)
            
            else:
                shape = vshape
            
            if episode_const:
                self.data.episode_data[field_key] = torch.zeros((batch_size, *shape), dtype=dtype, device=self.device)
            
            else:
                self.data.transition_data[field_key] = torch.zeros((batch_size, max_seq_length, *shape), dtype=dtype, device=self.device)
    

    