import torch
import timm
import torch.nn as nn

class BackboneResNet(nn.Module) :
    def __init__(self, base_model) :
        super(BackboneResNet, self).__init__()
        self.resnet_dict = {'resnet18' : timm.create_model('resnet18', pretrained=True, num_classes=2),
                            'resnet50' : timm.create_model('resnet50', pretrained=True, num_classes=2)}
        
        resnet = self._get_basemodel(base_model)
        modules = list(resnet.children())[:-1]  #fc layer 제거
        
        self.features = nn.Sequential(*modules)
                            
    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x) :
        return self.features(x)