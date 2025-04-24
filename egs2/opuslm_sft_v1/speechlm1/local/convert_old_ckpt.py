import torch

old = "exp/opuslm_7b_baseline/205epoch.pth"
new = "exp/opuslm_7b_baseline/205epoch_revised.pth"

old = torch.load(old, map_location="cpu")['model']

old["corelm.lm_head.weight"][5256: 13448] = old["corelm.aux_lm_head.weight"]
old["criterion.lm_head.weight"][5256: 13448] = old["criterion.aux_lm_head.weight"]

del old["criterion.aux_lm_head.weight"]
del old["corelm.aux_lm_head.weight"]

torch.save(old, new)