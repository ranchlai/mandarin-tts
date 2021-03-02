import torch.nn as nn
import torch.nn.functional as F
import torch
class Discriminator(nn.Module):
    def __init__(self, channels=1,output_size =  128):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [ (64, 2, False),
                                                (128, 2, True),
                                                (256, 2, True),
                                                (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 2, 3, 1, 1))

        self.model = nn.Sequential(*layers)
        self.fc = nn.Linear(output_size,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        output = self.model(img)
        output1 = output.view(output.shape[0],-1)
        output2 = self.fc(output1)
        output3 = self.sigmoid(output2)
        return output3
