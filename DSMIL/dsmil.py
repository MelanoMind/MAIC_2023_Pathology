import torch
import timm
import torch.nn as nn
import torch.nn.functional as F


class IClassifier(nn.Module) :
    """
    Patch Instacne Classifier 
    ImageNet으로 Pretrained된 backbone에 학습가능한 layer 2개를 추가하여 추츨된 feature가 positive인지 nagative인지 학습.
    binary classifier로, out_dim을 1로 두어서 recurrence에 따라 0,1로 학습.
    """
    def __init__(self, backbone, freeze=True, out_dim=1, nonlinear=True) :
        super(IClassifier, self).__init__()
        self.backbone_head = backbone
        
        if freeze :
            for param in self.backbone_head.parameters() :
                param.requires_grad = False
    
        if nonlinear :
            self.backbone_tail = nn.Sequential(nn.Linear(512, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 64),
                                        nn.ReLU())
        else :
            self.backbone_tail = nn.Sequential(nn.Linear(512, 256),
                                        nn.Linear(256, 64))
        self.classifier = nn.Sequential(nn.Linear(64, out_dim))
            
    def forward(self, x) :
        feature = self.backbone_head(x)
        feature = feature.view(feature.size(0), -1)
        mid_feature = self.backbone_tail(feature)
        mid_feature_ = mid_feature.view(mid_feature.size(0), -1)
        feature_class = self.classifier(mid_feature_)
        return feature_class, mid_feature, feature
    
    
class BClassifier(nn.Module) : #Bag
    def __init__(self, input_size=64, output_class=1, dropout_v=0.2, nonlinear=True): # K, L, N
        super(BClassifier, self).__init__()
        self.attention = nn.Sequential(nn.Linear(input_size, input_size//2),
                                        nn.Tanh(),
                                        nn.Linear(input_size//2, 1))
        self.classifier = nn.Sequential(nn.Linear(input_size, output_class))
#         if nonlinear:
#             self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
#         else:
#             self.q = nn.Linear(input_size, 128)
            
#         self.v = nn.Sequential(
#                 nn.Dropout(dropout_v),
#                 nn.Linear(input_size, input_size),
#                 nn.ReLU())
        
        # self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)
        
        
    def forward(self, feats, c): # N x K, N x C
        '''
        A : attention score
        B : bag representation
        C : prediction bag's class
        '''
        device = feats.device
        A_unnorm = self.attention(feats)
        A = torch.transpose(A_unnorm, 1, 0)
        A = F.softmax(A, dim=1)
        A_ = torch.mm(A, feats)
        B_Logit = self.classifier(A_)
        
        return B_Logit, A_unnorm
        # V = self.v(feats) # N x V, unsorted
        # Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
#         _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
#         m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
#         q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
#         A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
#         A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
#         B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
#         B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
#         C = self.fcc(B) # 1 x C x 1
#         C = C.view(1, -1)
#         C,A,B
        
        
        
    
#         def forward(self, feats, c):
#         device = feats.device
#         V = self.v(feats)  # N x V, unsorted

#         # Use the class scores directly
#         A = F.softmax(c, dim=1)  # Softmax over class scores, A in shape N x C
#         B = torch.mm(A.transpose(0, 1), V)  # Compute bag representation, B in shape C x V

#         B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
#         C = self.fcc(B)  # 1 x C x 1
#         C = C.view(1, -1)
#         return C, A, B
    
class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        
    def forward(self, x):
        classes, mid_feature, feature = self.i_classifier(x)
        prediction_bag, A_unnorm = self.b_classifier(mid_feature, classes)
        
        return classes, prediction_bag, A_unnorm, feature