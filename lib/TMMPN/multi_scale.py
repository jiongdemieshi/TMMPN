
import torch
import torch.nn as nn
from language.language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from network.connect import FCNet
from network.connect import BCNet
from network.counting import Counter
from utils.utils import tfidf_loading
from network.maml import SimpleCNN
from network.auto_encoder import Auto_Encoder_Model
from torch.nn.utils.weight_norm import weight_norm
from language.classify_question import typeAttention 
import clip
import os
from TMMPN.ATT import Mutimodel_fusion
import torch.nn.functional as F
import numpy as np
from TMMPN.Fusion import Fusion,Fusion2
from TMMPN.mca import AttFlat
from transformers import BertModel
from TMMPN.swin_model import swin_large_patch4_window7_224_in22k as create_model






def seperate(v,q,a,att, answer_target, n_unique_close):     #q: b x 12 x 1024  v:  b x 1 x 128 answer_target : 1 x b
    indexs_open = []
    indexs_close = []

    for i in range(len(answer_target)):
        if answer_target[i]==0:
            indexs_close.append(i)
        else:
            indexs_open.append(i)
        if len(q.shape) == 2:  # in case of using clip to encode q
            q = q.unsqueeze(1)
    return v[indexs_open,:,:],v[indexs_close,:,:],q[indexs_open,:,:],\
            q[indexs_close,:,:],a[indexs_open, n_unique_close:],a[indexs_close,:n_unique_close],att[indexs_open,:],att[indexs_close,:], indexs_open, indexs_close
 
def seperate_cma(v,q,q_mask,a, answer_target, n_unique_close):     #q: b x 12 x 1024  v:  b x 1 x 128 answer_target : 1 x b
    indexs_open = []
    indexs_close = []

    for i in range(len(answer_target)):
        if answer_target[i]==0:
            indexs_close.append(i)
        else:
            indexs_open.append(i)
        if len(q.shape) == 2:  # in case of using clip to encode q
            q = q.unsqueeze(1)
    
    return v[indexs_open,:,:],v[indexs_close,:,:],q[indexs_open,:,:],\
            q[indexs_close,:,:],a[indexs_open, n_unique_close:],a[indexs_close,:n_unique_close],q_mask[indexs_open,:,:],q_mask[indexs_close], indexs_open, indexs_close

class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        # self.args = args

        base_model = BertModel.from_pretrained("bert-base-uncased")
        bert_model = nn.Sequential(*list(base_model.children())[0:])
        self.bert_embedding = bert_model[0] 

        # self.word_embedding = nn.Linear(768, 1024, bias=False)
        # nn.init.xavier_normal_(self.word_embedding.weight)
        # self.sen_encode = SentenceTransformer(sen)
        # self.sen_embedding = nn.Linear(768, self.args.hidden_size, bias=False).to(device)  
        # nn.init.xavier_normal_(self.sen_embedding.weight)
        # self.w_s = word_sen(args)

    def forward(self, input_tokens):#, #input_sen, #question_mask):
        word_embedding = self.bert_embedding(input_tokens)
        # word_embedding = self.word_embedding(word_embedding)

        # sen = self.sen_encode.encode(input_sen)
        # sen = torch.tensor(sen)
        # sen = sen.unsqueeze(1).to(device)
        # sen_embedding = self.sen_embedding(sen)

        # embeddings = self.w_s(tokens_embedding, sen_embedding, question_mask)

        return word_embedding

class AnswerMask(nn.Module):
    def __init__(self, num_hid, num_ans, dropout=0.1):
        """ dropout rate for Plain should be 0.0"""
        super(AnswerMask, self).__init__()
        layers = [
            nn.Dropout(dropout),
            nn.Linear(num_hid, num_ans),
        ]

        self.mask = nn.Sequential(*layers)
        # self.mask=nn.Linear(num_hid, num_ans)
    def forward(self, q):
        q = self.mask(q)
        # q = torch.sigmoid(q).clone()
        q = F.softplus(q).clamp(max=1.0)
        return q


# Create BAN model
class BAN_Model(nn.Module):
    def __init__(self, dataset, cfg, device):
        super(BAN_Model, self).__init__()

        self.cfg = cfg
        self.dataset = dataset
        self.device = device
        # self.w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, cfg.TRAIN.QUESTION.CAT)
        # self.q_emb = QuestionEmbedding(600 if cfg.TRAIN.QUESTION.CAT else 300, cfg.TRAIN.QUESTION.HID_DIM, 1, False, .0, cfg.TRAIN.QUESTION.RNN)
        
        self.close_classifier = SimpleClassifier(cfg.TRAIN.QUESTION.CLS_HID_DIM, cfg.TRAIN.QUESTION.CLS_HID_DIM * 2, dataset.num_close_candidates, cfg)
        self.open_classifier = SimpleClassifier(cfg.TRAIN.QUESTION.CLS_HID_DIM, cfg.TRAIN.QUESTION.CLS_HID_DIM * 2, dataset.num_open_candidates, cfg)

        self.w_emb=Embedding()

        path = os.path.join(self.cfg.DATASET.DATA_DIR, "glove6b_init_300d.npy")
        
        
        if cfg.TRAIN.VISION.MAML:
            weight_path = cfg.DATASET.DATA_DIR + '/' + cfg.TRAIN.VISION.MAML_PATH
            print('load initial weights MAML from: %s' % (weight_path))
            self.maml = SimpleCNN(weight_path, cfg.TRAIN.OPTIMIZER.EPS_CNN, cfg.TRAIN.OPTIMIZER.MOMENTUM_CNN)
        # build and load pre-trained Auto-encoder model
        if cfg.TRAIN.VISION.AUTOENCODER:
            self.ae = Auto_Encoder_Model()
            weight_path = cfg.DATASET.DATA_DIR + '/' + cfg.TRAIN.VISION.AE_PATH
            print('load initial weights DAE from: %s' % (weight_path))
            self.ae.load_state_dict(torch.load(weight_path))
            self.convert = nn.Linear(16384, 64)
        # build and load pre-trained CLIP model
        if cfg.TRAIN.VISION.CLIP:
            self.clip, _ = clip.load(cfg.TRAIN.VISION.CLIP_VISION_ENCODER, jit=False)
            if not cfg.TRAIN.VISION.CLIP_ORG:
                checkpoint = torch.load(cfg.TRAIN.VISION.CLIP_PATH)
                self.clip.load_state_dict(checkpoint['state_dict'])
            self.clip = self.clip.float()
        # Loading tfidf weighted embedding
        # if cfg.TRAIN.QUESTION.TFIDF:
        #     self.w_emb = tfidf_loading(cfg.TRAIN.QUESTION.TFIDF, self.w_emb, cfg)
        # Loading the other net
        if cfg.TRAIN.VISION.OTHER_MODEL:
            pass
        
        # self.fuse_layer=nn.Linear(1024+64,512);
      
        self.backbone_open=WSDAN(cfg.CC)
        self.backbone_closed=WSDAN(cfg.CC)
        self.attflat_img_open = AttFlat(cfg.CC)
        self.attflat_lang_open = AttFlat(cfg.CC)

        self.Swin=create_model(num_classes=5).to(device)
        # weights_dict=torch.load('/home/tj/swin_transformer/weights/model-15.pth', map_location=device)
        
        # weights_dict=torch.load('/home/tj/swin_transformer/fine-turn/model-15.pth', map_location=device)
        # self.Swin.load_state_dict(weights_dict, strict=False)
        
        self.grid_global_encoder=Fusion(768)
        self.grid_global_encoder2=Fusion2(1024)
        # self.attflat_img_close = AttFlat(cfg.CC)
        # self.attflat_lang_close = AttFlat(cfg.CC)
        self.img_type_open=nn.Linear(1024, 5)
        self.img_type_close=nn.Linear(1024, 5)
        # self.lstm = nn.LSTM(
        #     input_size=768,
        #     hidden_size=1024,
        #     num_layers=1,
        #     batch_first=True
        # )
        self.mask_open=AnswerMask(1024,dataset.num_open_candidates)
        self.mask_close=AnswerMask(1024,dataset.num_close_candidates)

    def forward(self, v, q, token, answer_mask,a, answer_target):
        """Forward
        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [[batch_size, seq_length], [batch_size, seq_length]]
        return: logits, not probs
        """
        # get visual feature
        batch=q[0].size(0)
        if self.cfg.TRAIN.VISION.MAML:
            maml_v_emb = self.maml(v[0]).unsqueeze(1)
            v_emb = maml_v_emb
        if self.cfg.TRAIN.VISION.AUTOENCODER:
            encoder = self.ae.forward_pass(v[1])
            decoder = self.ae.reconstruct_pass(encoder)
            ae_v_emb = encoder.view(encoder.shape[0], -1)
            ae_v_emb = self.convert(ae_v_emb).unsqueeze(1)
            v_emb = ae_v_emb
        if self.cfg.TRAIN.VISION.CLIP: 
            #clip_v_emb = self.clip.encode_image(v[2]).unsqueeze(1)
            clip_v_emb,_= self.clip.encode_image(v[2])
            v_emb = clip_v_emb

        global_feat=self.Swin(v[3])
        global_feat,v_emb=self.grid_global_encoder(v_emb,global_feat)
        
        mask_feat=v[4]
        mask_feat,v_emb=self.grid_global_encoder2(v_emb,mask_feat)
        # scale_feat=torch.cat((global_feat.unsqueeze(1),mask_feat.unsqueeze(1)),1)
        if self.cfg.TRAIN.VISION.CLIP and self.cfg.TRAIN.VISION.AUTOENCODER:
            ae_v_emb=torch.repeat_interleave(ae_v_emb,49,dim=1)
            v_emb = torch.cat((v_emb, ae_v_emb), 2)#batch,49,1088
           
        
            

       

        q_mask=self.make_mask(token.unsqueeze(2))
        q_emb=self.w_emb(token)
        # q_emb, _ = self.lstm(q_emb)
        
        img_predict=torch.zeros(batch,5).cuda()
        v_emb_open,v_emb_closed,q_open, q_close,a_open, a_close,q_mask_open,q_mask_close,  indexs_open, indexs_close=seperate_cma(v_emb,q_emb,q_mask,a, answer_target, self.dataset.num_close_candidates)
        # cm_feat_open,cm_feat_closed,q_open, q_close,a_open, a_close,  _, _=seperate_cma(cm_feat,q_emb,a, answer_target, self.dataset.num_close_candidates)
        
        answer_mask_open=answer_mask[indexs_open, self.dataset.num_close_candidates:]
        answer_mask_close=answer_mask[indexs_close,:self.dataset.num_close_candidates]
        if v_emb_open.size(0)!=0:
            lang_feat_open, img_feat_open = self.backbone_open(
                q_open,
                v_emb_open,
                None,
                None
            )
           
            # img_feat_open=img_feat_open.sum(1)
            # lang_feat_open=lang_feat_open.sum(1)
            # img_feat_open=torch.cat((img_feat_open,scale_feat[indexs_open]),1)
            img_feat_open=self.attflat_img_open(img_feat_open,None)
            # img_feat_open=torch.cat((img_feat_open.unsqueeze(1),scale_feat[indexs_open]),1).sum(1)
            lang_feat_open=self.attflat_lang_open(lang_feat_open,None)
            proj_feat_open = lang_feat_open + img_feat_open 
            img_pre_open=self.img_type_open(proj_feat_open)
            img_predict[indexs_open]=img_pre_open
            predict_mask_open=self.mask_open(proj_feat_open)
           
            
            
        else:
            proj_feat_open=torch.empty(0,1024).cuda()
            predict_mask_open=torch.empty(0,self.dataset.num_open_candidates).cuda()
        
        if v_emb_closed.size(0)!=0:
            lang_feat_close, img_feat_close = self.backbone_closed(
                q_close,
                v_emb_closed,
                None,
                None
            )
            
            
            # img_feat_close=img_feat_close.sum(1)
            # lang_feat_close=lang_feat_close.sum(1)
            # img_feat_close=torch.cat((img_feat_close,scale_feat[indexs_close]),1)
            img_feat_close=self.attflat_img_open(img_feat_close,None)
            # img_feat_close=torch.cat((img_feat_close.unsqueeze(1),scale_feat[indexs_close]),1).sum(1)
            lang_feat_close=self.attflat_lang_open(lang_feat_close,None)
            proj_feat_close = lang_feat_close + img_feat_close 
            img_pre_close=self.img_type_open(proj_feat_close)
            img_predict[indexs_close]=img_pre_close
            predict_mask_close=self.mask_close(proj_feat_close)

            
            
        else:
            proj_feat_close=torch.empty(0,1024).cuda()
            predict_mask_close=torch.empty(0,self.dataset.num_close_candidates).cuda()
            


        if self.cfg.TRAIN.VISION.AUTOENCODER:
                return proj_feat_close,proj_feat_open,a_close,a_open, decoder,img_predict,predict_mask_open,predict_mask_close,answer_mask_open,answer_mask_close
        return proj_feat_close,proj_feat_open,a_close, a_open

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)
    def classify(self, close_feat, open_feat,open_mask,close_mask):
        return self.close_classifier(close_feat,close_mask), self.open_classifier(open_feat,open_mask)
    def bbn_classify(self, bbn_mixed_feature):
        return self.bbn_classifier(bbn_mixed_feature)
    
    def forward_classify(self,v,q,token,answer_mask,a,classify, n_unique_close):
        # get visual feature
        batch=q[0].size(0)
        
       
        if self.cfg.TRAIN.VISION.MAML:
            maml_v_emb = self.maml(v[0]).unsqueeze(1)
            v_emb = maml_v_emb
        if self.cfg.TRAIN.VISION.AUTOENCODER:
            encoder = self.ae.forward_pass(v[1])
            decoder = self.ae.reconstruct_pass(encoder)
            ae_v_emb = encoder.view(encoder.shape[0], -1)
            ae_v_emb = self.convert(ae_v_emb).unsqueeze(1)
            v_emb = ae_v_emb
        if self.cfg.TRAIN.VISION.CLIP: 
            #clip_v_emb = self.clip.encode_image(v[2]).unsqueeze(1)
            clip_v_emb,_ = self.clip.encode_image(v[2])
            # clip_v_emb=clip_v_emb.view(batch,2048,-1).transpose(1,2)
            v_emb = clip_v_emb
        if self.cfg.TRAIN.VISION.MAML and self.cfg.TRAIN.VISION.AUTOENCODER:
            v_emb = torch.cat((maml_v_emb, ae_v_emb), 2)
        
        global_feat=self.Swin(v[3])
        global_feat,v_emb=self.grid_global_encoder(v_emb,global_feat)
        mask_feat=v[4]
        mask_feat,v_emb=self.grid_global_encoder2(v_emb,mask_feat)
        # scale_feat=torch.cat((global_feat.unsqueeze(1),mask_feat.unsqueeze(1)),1)
        
        if self.cfg.TRAIN.VISION.CLIP and self.cfg.TRAIN.VISION.AUTOENCODER:
            ae_v_emb=torch.repeat_interleave(ae_v_emb,49,dim=1)
            v_emb = torch.cat((v_emb, ae_v_emb), 2)#batch,49,1088
           
        
        if self.cfg.TRAIN.VISION.OTHER_MODEL:
            pass
        
        q_mask=self.make_mask(token.unsqueeze(2))
        q_emb=self.w_emb(token)
        

        # get open & close feature
        answer_target = classify(q)
        _,predicted=torch.max(answer_target,1)

        img_predict=torch.zeros(batch,5).cuda()
        
        v_emb_open,v_emb_closed,q_open, q_close,a_open, a_close,q_mask_open,q_mask_close,indexs_open, indexs_close=seperate_cma(v_emb,q_emb,q_mask,a, predicted, self.dataset.num_close_candidates)
        answer_mask_open=answer_mask[indexs_open, self.dataset.num_close_candidates:]
        answer_mask_close=answer_mask[indexs_close,:self.dataset.num_close_candidates]
        if v_emb_open.size(0)!=0:
            lang_feat_open, img_feat_open = self.backbone_open(
                q_open,
                v_emb_open,
                None,
                None
            )
            # img_feat_open=torch.cat((img_feat_open,scale_feat[indexs_open]),1)
            img_feat_open=self.attflat_img_open(img_feat_open,None)
            
            # img_feat_open=torch.cat((img_feat_open.unsqueeze(1),scale_feat[indexs_open]),1).sum(1)
            lang_feat_open=self.attflat_lang_open(lang_feat_open,None)
            proj_feat_open = lang_feat_open + img_feat_open 
            img_pre_open=self.img_type_open(proj_feat_open)
            img_predict[indexs_open]=img_pre_open
            # predict_mask_open=self.mask_open(proj_feat_open)
        
            
        else:
            proj_feat_open=torch.empty(0,1024).cuda()
            # predict_mask_open=torch.empty(0,self.dataset.num_open_candidates).cuda()
        
        if v_emb_closed.size(0)!=0:
            lang_feat_close, img_feat_close = self.backbone_closed(
                q_close,
                v_emb_closed,
                None,
                None
            )
            
            # img_feat_close=torch.cat((img_feat_close,scale_feat[indexs_close]),1)
            img_feat_close=self.attflat_img_open(img_feat_close,None)
            # img_feat_close=torch.cat((img_feat_close.unsqueeze(1),scale_feat[indexs_close]),1).sum(1)
            lang_feat_close=self.attflat_lang_open(lang_feat_close,None)
            proj_feat_close = lang_feat_close + img_feat_close
            img_pre_close=self.img_type_open(proj_feat_close)
            img_predict[indexs_close]=img_pre_close
            # predict_mask_close=self.mask_close(proj_feat_close)
            
            
        else:
            proj_feat_close=torch.empty(0,1024).cuda()
            # predict_mask_close=torch.empty(0,self.dataset.num_close_candidates).cuda()

        if self.cfg.TRAIN.VISION.AUTOENCODER:
                return proj_feat_close,proj_feat_open ,a_close,a_open, decoder, indexs_open, indexs_close,img_predict
        return proj_feat_close,proj_feat_open ,a_close, a_open, indexs_open, indexs_close


def generate_spatial_batch(N, featmap_H, featmap_W):
    spatial_batch_val = np.zeros((N, 8, featmap_H, featmap_W), dtype=np.float32)
    for h in range(featmap_H):
        for w in range(featmap_W):
            xmin = w / featmap_W * 2 - 1
            xmax = (w+1) / featmap_W * 2 - 1
            xctr = (xmin+xmax) / 2
            ymin = h / featmap_H * 2 - 1
            ymax = (h+1) / featmap_H * 2 - 1
            yctr = (ymin+ymax) / 2
            spatial_batch_val[:, :, h, w] = \
                [xmin, ymin, xmax, ymax, xctr, yctr, 1/featmap_W, 1/featmap_H]
    return spatial_batch_val