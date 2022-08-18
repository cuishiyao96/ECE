from utils import *

class Data_data(object):
    def __init__(self, data, batch_size, data_type='train'):
        super(Data_data, self).__init__()

        self.data_type = data_type
        self.data = data

        self.corpus_length=len(data)
        self.start = 0
        self.len_data = int(self.corpus_length // batch_size)
        self.batch_size = batch_size
        print("total: ", self.corpus_length, self.len_data, self.batch_size)

    def iter_batch(self):


        for idx in range(self.len_data):
            start = idx * self.batch_size
            current_data = self.data[start: start + self.batch_size]
            text_ids, input_ids, input_masks, segment_ids, label_indexs, result_indexs, loss_masks = [], [], [], [], [], [], []
            # rr_edges, tt_edges = [], []
            if self.data_type == 'train':
                tt_labels = []

            for index, data_item in enumerate(current_data):
                # pdb.set_trace()
                text_ids.append(data_item['text_id'])
                input_ids.append(data_item['input_ids'])
                input_masks.append(data_item['input_masks'])
                segment_ids.append(data_item['segment_ids'])
                label_indexs.append(data_item['label_indexs'])
 
                if self.data_type == 'train':                    
                    tt_label = np.zeros((len(etype_map), max_seq_len, len(tt_map) ))
                    # pdb.set_trace()
                    for (i, j, k) in data_item['tt_labels']:
                        tt_label[i][j][k] = 1
                    tt_labels.append(tt_label.tolist())


            input_ids = trans_to_cuda(torch.LongTensor(input_ids))
            input_masks = trans_to_cuda(torch.FloatTensor(input_masks))
            segment_ids = trans_to_cuda(torch.LongTensor(segment_ids))
            label_indexs = trans_to_cuda(torch.LongTensor(label_indexs))
            if self.data_type == 'train':
                tt_labels = trans_to_cuda(torch.FloatTensor(tt_labels))
                yield [text_ids, input_ids, input_masks, segment_ids, label_indexs, tt_labels]
            else:
                yield [text_ids, input_ids, input_masks, segment_ids, label_indexs, None]
                

if __name__ == '__main__':

    filename = '../data/processed_data/train.pickle'
    train_data_ = load_data(filename, debug=True)
    train_data = Data_data(train_data_, batch_size=2, data_type='train')
    for (text_ids, input_ids, input_masks, segment_ids, label_indexs, tt_labels) in train_data.iter_batch():
        pdb.set_trace()
    # predict_res_list = [{"text_id": "1882370", "result": [{"reason_type": "需求增加", "reason_region": "南方", "reason_product": "水稻", "reason_industry": "", "result_type": "市场价格提升"    , "result_region": "", "result_product": "尿素", "result_industry": ""}]}]
    # gold_res_list = [{"text_id": "1882370", "text": "尿素：随着天气转暖，春耕、北方小麦返青肥、南方水稻用肥需求增加，价格稳中上涨", "result": [{"reason_type": "需求增加", "reason_product": "小麦返青肥,水稻用肥", "reason_region": "北方,南方", "result_region": "", "result_industry": "", "result_type": "市场价格提升", "reason_industry": "", "result_product": "尿素"}]}]
    # p, r, f = metric(predict_res_list, gold_res_list)
    # print(p, r, f)
