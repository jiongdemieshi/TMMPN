
# from transformers import BertModel, BertTokenizer
# import torch
# import torch.nn as nn
# import torch.utils.data as Data

# # bert = BertModel.from_pretrained("bert-base-uncased")
# # print(bert)
# # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# # # text_sentence="What organ system is pictured"
# # # tokens = tokenizer.encode_plus(text=text_sentence)
# # # print(tokens)
# device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# class Embedding(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.args = args

#         base_model = BertModel.from_pretrained("bert-base-uncased")
#         bert_model = nn.Sequential(*list(base_model.children())[0:])
#         self.bert_embedding = bert_model[0] 

#         self.word_embedding = nn.Linear(768, 1024, bias=False)
#         nn.init.xavier_normal_(self.word_embedding.weight)
#         # self.sen_encode = SentenceTransformer(sen)
#         # self.sen_embedding = nn.Linear(768, self.args.hidden_size, bias=False).to(device)  
#         # nn.init.xavier_normal_(self.sen_embedding.weight)
#         # self.w_s = word_sen(args)

#     def forward(self, input_tokens):#, #input_sen, #question_mask):
#         word_embedding = self.bert_embedding(input_tokens)
#         tokens_embedding = self.word_embedding(word_embedding)

#         # sen = self.sen_encode.encode(input_sen)
#         # sen = torch.tensor(sen)
#         # sen = sen.unsqueeze(1).to(device)
#         # sen_embedding = self.sen_embedding(sen)

#         # embeddings = self.w_s(tokens_embedding, sen_embedding, question_mask)

#         return tokens_embedding

# # if __name__ == '__main__':
# #     Question_embed=Embedding()
# #     text_sentence="Is there evidence of a pneumothorax"
# #     text_sentence2="What organ system is pictured"
    
    
# #     tokens = tokenizer.encode_plus(text=text_sentence,max_length=20)

# #     #tensor([  101,  2003,  2045,  3350,  1997,  1037,  1052,  2638,  2819, 29288,
# #     #     2527,  2595,   102])
# #     token=torch.tensor(tokens["input_ids"]).to(device)
# #     token=token.unsqueeze(0)
# #     print(tokens)
# #     a=Question_embed(token)

# maxlen=20
# class MyDataset(Data.Dataset):
#   def __init__(self, sentences, labels=None, with_labels=True,):
#     self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     self.with_labels = with_labels
#     self.sentences = sentences
#     self.labels = labels
#   def __len__(self):
#     return len(self.sentences)
#   def __getitem__(self, index):
#     # Selecting sentence1 and sentence2 at the specified index in the data frame
#     sent = self.sentences[index]
#     # Tokenize the pair of sentences to get token ids, attention masks and token type ids
#     encoded_pair = self.tokenizer(sent,
#                     padding='max_length',  # Pad to max_length
#                     truncation=True,       # Truncate to max_length
#                     max_length=16,
#                     return_tensors='pt')  # Return torch.Tensor objects
#     token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
#     attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
#     token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens
#     if self.with_labels:  # True if the dataset has labels
#       label = self.labels[index]
#       return token_ids#, attn_masks, token_type_ids, label
#     else:
#       return token_ids#, attn_masks, token_type_ids

# if __name__ == '__main__':
#     test_text = ["Is there evidence of a pneumothorax", "What organ system is pictured"]
#     test =MyDataset(test_text, labels=None, with_labels=False)
#     print(test)
#     x = test.__getitem__(0).unsqueeze(0)
#     print(x)
#     y= test.__getitem__(1).unsqueeze(0)
#     input=torch.cat((x,y),dim=0)
#     print(input)
#     # x = tuple(p.unsqueeze(0).to(device) for p in x)
#     # Question_embed=Embedding()
#     # out=Question_embed(input)
#     # print(out.size())

from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
import torch.utils.data as Data

bert = BertModel.from_pretrained("bert-base-uncased")