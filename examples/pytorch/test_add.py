# Note that this doesnt work yet... we have to recompile pytorch from scratch, for RISC-V
# in order to enable this. (We also have to make it possible to compile pytorch against
# VeriGPU, which is non-trivial...)

import torch

a = torch.rand(3)
a = a.cuda()
a = a + 1
