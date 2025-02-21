import torch_pruning as tp



# 每个剪枝器的重要函数是 regularize
tp.pruner.GrowingRegPruner().regularize()