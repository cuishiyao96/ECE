from layers import ConditionalLayerNorm
from transformers import BertModel, BertTokenizer
from utils import *


class GridModel(nn.Module):
    def __init__(self, hidden_size, type_num, dropout):
        super(GridModel, self).__init__()

        self.hidden_size = hidden_size
        self.cond_norm = ConditionalLayerNorm(self.hidden_size, eps=1e-6)

        self.W1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dp = nn.Dropout(dropout)

        self.cls_layer = nn.Linear(self.hidden_size, type_num)


    def handshake(self, inputs, cond):
        """
        inputs: batch, len, dim
        cond: batch, type_num(cond_len), dim
        """
        batch, len_, dim = inputs.shape
        cond_len = cond.shape[1]
        x_ = inputs.unsqueeze(1).expand(batch, cond_len, len_, dim).reshape(batch * cond_len, len_, dim)
        condition_ = cond.reshape(batch * cond_len, dim) # [batch * cond_len, dim]
        x = self.dp(self.W1(x_))
        condition = self.dp(self.W2(condition_))
        out = self.cond_norm(x = x, condition = condition)
        return out.reshape(batch, cond_len, len_, dim) 

    def forward(self, input_embs, cond):
        """
        input_embs: batch, len_, dim
        """
        batch_size, len_, dim = input_embs.shape

        embedding_matrix = self.handshake(input_embs, cond) #
        outputs = self.cls_layer( embedding_matrix  ) 
        return outputs
        



class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.hidden_size = args.hidden_size

        self.bert_embedding = BertModel.from_pretrained(args.bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_path)
        
        # Grid Representation and Classification for the Cause Table
        self.gridmodel = GridModel(hidden_size=self.hidden_size, type_num=int(len(tt_map) // 2), dropout=args.dropout)
        # Grid Representation and Classification for the Effect Table
        self.gridmodel2 = GridModel(hidden_size=self.hidden_size, type_num=int(len(tt_map) // 2), dropout=args.dropout)
        self.dp = nn.Dropout( args.dropout  )
        
        self.thresh = args.thresh

    def encoding(self, input_ids, segment_ids, input_masks):

        out = self.bert_embedding(input_ids=input_ids, attention_mask=input_masks, token_type_ids=segment_ids)
        input_embs = out.last_hidden_state
        input_embs = self.dp( input_embs  )
        return input_embs

    def obtain_embs(self, input_embs_, label_indexs ):

        etype_embs = torch.index_select(input_embs_, dim = 1, index = label_indexs[0])
        input_embs = input_embs_[:, -max_seq_len: , :]
        return input_embs, etype_embs


    def run(self, input_ids, segment_ids, input_maks, label_indexs):
        batch_size = input_ids.shape[0]

        input_embs_  = self.encoding(input_ids, segment_ids, input_maks)
        input_embs, etype_embs = self.obtain_embs(input_embs_, label_indexs)
        # input_embs: [batch, seq, dim]
        # etype_embs: [batch, type_num, dim]

        tt_outputs_1 = self.gridmodel(input_embs, etype_embs)
        tt_outputs_2 = self.gridmodel2(input_embs, etype_embs)
        # Concat the output of two tables to derive the final loss
        tt_outputs = torch.sigmoid(torch.cat([tt_outputs_1, tt_outputs_2], dim = -1))
        return tt_outputs


    def forward(self, input_ids, segment_ids, input_maks, label_indexs):

        tt_outputs = self.run(input_ids, segment_ids, input_maks, label_indexs)
        return tt_outputs

    
    def inference(self, text_ids, input_ids, segment_ids, input_maks, label_indexs):
        
        result = {'text_id': text_ids[0], 'result': []}
        batch_size = input_ids.shape[0]

        tt_outputs_ = self.run(input_ids, segment_ids, input_maks, label_indexs)
        tt_outputs = tt_outputs_.squeeze(0).detach().cpu().numpy() # [type_num, seq, type_nums]
        
        # Decode ent
        input_ids = input_ids.squeeze(0)[-max_seq_len:]
        sent_len = torch.sum(input_maks.squeeze(0)[-max_seq_len:]).item()
        heads, tails, iids = np.where(tt_outputs > self.thresh)
        ent_dict, event_ent_dict = {}, {}
        reason_dict, result_dict = {}, {}
        

        for (etype_id, token_id, iid) in list(zip(heads, tails, iids)):
            # etype_id: index along the column, namely event types
            # token_id: index along the row, namely the position of token
            # iid: index of the predefined tags
            
            etype = etype_id2type[etype_id]
            tag_type = tt_id2type[iid]
            tag, ent_type, ent_pos = tag_type.split('-')

            
            ### Step1: Argument Span Decoding ###
            # In the Cause Table
            if (tt_map['Rea2Rea-product-H'] <= iid <= tt_map['Rea2Rea-industry-T']) or (tt_map['Rea2Res-product-H'] <= iid <= tt_map['Rea2Res-industry-T']):
                if etype not in reason_dict:
                    reason_dict[etype] = { 'reason':{'product': {'H': [], 'T': []},  'region': {'H': [], 'T': []}, 'industry': {'H': [], 'T': []}}, 
                                           'result':{'product': {'H': [], 'T': []},  'region': {'H': [], 'T': []}, 'industry': {'H': [], 'T': []}}}
                if tag == 'Rea2Rea':
                    reason_dict[etype]['reason'][ent_type][ent_pos].append(token_id) 
                elif tag == 'Rea2Res':
                    reason_dict[etype]['result'][ent_type][ent_pos].append(token_id) 
            # In the Effect Table
            elif (tt_map['Res2Res-product-H'] <= iid <= tt_map['Res2Res-industry-T']) or (tt_map['Res2Rea-product-H'] <= iid <= tt_map['Res2Rea-industry-T']):
                if etype not in result_dict:
                    result_dict[etype] = {'reason':{'product': {'H': [], 'T': []},  'region': {'H': [], 'T': []}, 'industry': {'H': [], 'T': []}}, 
                                          'result':{'product': {'H': [], 'T': []},  'region': {'H': [], 'T': []}, 'industry': {'H': [], 'T': []}}}
                if tag == 'Res2Rea':
                    result_dict[etype]['reason'][ent_type][ent_pos].append(token_id) 
                elif tag == 'Res2Res':
                    # pdb.set_trace()
                    result_dict[etype]['result'][ent_type][ent_pos].append(token_id) 
 
        
        reason_ent_dict, result_ent_dict = {}, {}
        # In the cause table
        for etype in reason_dict:
            reason_ent_dict[etype] = {'reason': [], 'result': []}
            for tag in reason_dict[etype]: # tag: reason(Intra) / result(Inter)
                for key in reason_dict[etype][tag]: # key: product / region / industry
                    for ent_hid in reason_dict[etype][tag][key]['H']:
                        ent_tid_list = [ii for ii in reason_dict[etype][tag][key]['T'] if ii >= ent_hid]
                        if len(ent_tid_list) > 0:
                            ent_tid = min(ent_tid_list)
                            if max(ent_hid, ent_tid) < sent_len:
                                ent_text = "".join( self.tokenizer.convert_ids_to_tokens( input_ids[ent_hid: ent_tid + 1] ) )
                                reason_ent_dict[etype][tag].append((ent_text, key))
        # In the effect table
        for etype in result_dict:
            result_ent_dict[etype] = {'reason': [], 'result': []}
            for tag in result_dict[etype]: # tag: reason(Inter) / result(Intra)
                for key in result_dict[etype][tag]: # key: product / region / industry
                    for ent_hid in result_dict[etype][tag][key]['H']:
                        ent_tid_list = [ii for ii in result_dict[etype][tag][key]['T'] if ii >= ent_hid]                        
                        if len(ent_tid_list) > 0:
                            ent_tid = min(ent_tid_list)
                            if max(ent_hid, ent_tid) < sent_len:
                                ent_text = "".join( self.tokenizer.convert_ids_to_tokens( input_ids[ent_hid: ent_tid + 1] ) )
                                result_ent_dict[etype][tag].append((ent_text, key))

        
        ### Step3: Decode event pair ###
        for reason_type in reason_ent_dict:
            for result_type in result_ent_dict:
                reason_args = [item for item in reason_ent_dict[reason_type]['reason'] if item in result_ent_dict[result_type]['reason']]
                result_args = [item for item in result_ent_dict[result_type]['result'] if item in reason_ent_dict[reason_type]['result']]
                if max( len(reason_args), len(result_args)) != 0:
                    rr_pair = {'reason_type': reason_type, 'result_type': result_type, 
                        'reason_product': set(), 'reason_region': set(), 'reason_industry': set(), 
                        'result_product': set(), 'result_region': set(), 'result_industry': set()}

                    for item in reason_args:
                        ent_text, ent_type = item[-2], 'reason_' + item[-1]
                        rr_pair[ent_type].add(ent_text)
                    for item in result_args:
                        ent_text, ent_type = item[-2], 'result_' + item[-1]
                        rr_pair[ent_type].add(ent_text)

                    for key in ['reason_product', 'reason_region', 'reason_industry', 'result_product', 'result_region', 'result_industry']:
                        rr_pair[key] = ",".join(list(rr_pair[key])) if len(rr_pair[key]) != 0 else ""
                    result['result'].append(rr_pair)         
        return result











        

        
        







