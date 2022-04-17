# pytorch

This is for using the simulated GPU from pytorch, via a HIP API.

This is *very* early days yet...

For more information see [docs/hip.md](/docs/hip.md)

Information on examples in this folder:
- test_create_tensor.py  This runs ok, though doesnt do much except copy some data from main memory into the gpu global memory
- test_add.py  This is a next goal. Far from working just yet
