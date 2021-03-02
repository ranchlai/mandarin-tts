import sys
import torch
infile = sys.argv[1]
model = torch.load(infile)['model']
print('model saved to',infile[:-4])
torch.save(model,infile[:-4])
