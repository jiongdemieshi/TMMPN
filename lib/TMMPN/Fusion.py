from BAN.PureT_encoder import Encoder
# from BAN.PureT_decoder import Decoder
import torch
import torch.nn as nn
class Fusion(nn.Module):
    def __init__(self,input_dim):
        super(Fusion, self).__init__()
        self.use_gx=True
        self.ATT_FEATS_NORM=False
        self.ATT_FEATS_EMBED_DIM=1024
        self.encoder = Encoder(
            embed_dim=self.ATT_FEATS_EMBED_DIM,
            input_resolution=(7, 7),
            depth=6,
            num_heads=8,
            window_size=7,
            shift_size=3,
            mlp_ratio=4,
            dropout=0.1,
            use_gx=self.use_gx
        )
        self.att_embed = nn.Sequential(
                nn.Linear(input_dim, self.ATT_FEATS_EMBED_DIM),
                # nn.CELU(1.3),
                # nn.LayerNorm(self.ATT_FEATS_EMBED_DIM) if self.ATT_FEATS_NORM == True else nn.Identity(),
                # nn.Dropout(0.1)
        )
        self.global_embed = nn.Sequential(
                nn.Linear(1536, self.ATT_FEATS_EMBED_DIM),
                # nn.CELU(1.3),
                # nn.LayerNorm(self.ATT_FEATS_EMBED_DIM) if self.ATT_FEATS_NORM == True else nn.Identity(),
                # nn.Dropout(0.1)
        )
        # self.back = nn.Sequential(
        #         nn.Linear(self.ATT_FEATS_EMBED_DIM,1024),
        #         # nn.CELU(1.3),
        #         # nn.LayerNorm(self.ATT_FEATS_EMBED_DIM) if self.ATT_FEATS_NORM == True else nn.Identity(),
        #         # nn.Dropout(0.1)
        # )
       
    
    def forward(self,x,y):
        if x.size(0)!=0:
            x=self.att_embed(x)
            y=self.global_embed(y)
            global_feat,grid_feat=self.encoder(x,y)
            # grid_feat=self.back(grid_feat)
        else:
            grid_feat=torch.empty(0,49,1024).cuda()
            global_feat=torch.empty(0,512).cuda()
        # global_feat=self.back2(global_feat)
       

        return global_feat,grid_feat
    
class Fusion2(nn.Module):
    def __init__(self,input_dim):
        super(Fusion2, self).__init__()
        self.use_gx=True
        self.ATT_FEATS_NORM=False
        self.ATT_FEATS_EMBED_DIM=1024
        self.encoder = Encoder(
            embed_dim=self.ATT_FEATS_EMBED_DIM,
            input_resolution=(7, 7),
            depth=6,
            num_heads=8,
            window_size=7,
            shift_size=3,
            mlp_ratio=4,
            dropout=0.1,
            use_gx=self.use_gx
        )
        
        self.global_embed = nn.Sequential(
                nn.Linear(1536, self.ATT_FEATS_EMBED_DIM),
                # nn.CELU(1.3),
                # nn.LayerNorm(self.ATT_FEATS_EMBED_DIM) if self.ATT_FEATS_NORM == True else nn.Identity(),
                # nn.Dropout(0.1)
        )
        # self.back = nn.Sequential(
        #         nn.Linear(self.ATT_FEATS_EMBED_DIM,1024),
        #         # nn.CELU(1.3),
        #         # nn.LayerNorm(self.ATT_FEATS_EMBED_DIM) if self.ATT_FEATS_NORM == True else nn.Identity(),
        #         # nn.Dropout(0.1)
        # )
       
    
    def forward(self,x,y):
        if x.size(0)!=0:
            y=self.global_embed(y)
            global_feat,grid_feat=self.encoder(x,y)
            # grid_feat=self.back(grid_feat)
        else:
            grid_feat=torch.empty(0,49,1024).cuda()
            global_feat=torch.empty(0,512).cuda()
        # global_feat=self.back2(global_feat)
       

        return global_feat,grid_feat