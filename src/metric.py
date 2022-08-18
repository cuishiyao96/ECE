import copy, pdb


def get_multi_ids(gold_data):
    single_ids, multi_ids = [], []
    multi_gold_data, single_gold_data = [], []
    for line in gold_data:
        if len(line['result']) > 1:
            multi_ids.append(line['text_id'])
            multi_gold_data.append(line)
        else:
            single_ids.append( line['text_id']   )
            single_gold_data.append(line)
    # print("Single: ", len(single_ids), "Multi: ", len(multi_ids), "Total: ", len(gold_data))
    return single_ids, multi_ids, single_gold_data, multi_gold_data


def get_data(inputs, multi_ids, single_ids):
    multi_data = []
    single_data = []
    for line in inputs:
        newline = eval(line)
        if newline['text_id'] in multi_ids:
            multi_data.append(line)
        else:
            single_data.append(line)
    return single_data, multi_data


def get_events(input_res):
    event_list = []
    for item in input_res['result']:
        rea_event = {'event_type': item['reason_type'], 'product': item['reason_product'], 'region': item['reason_region'], 'industry': item['reason_industry']}
        event_list.append(rea_event)
        res_event = {'event_type': item['result_type'], 'product': item['result_product'], 'region': item['result_region'], 'industry': item['result_industry']}
        event_list.append(res_event)
    return event_list

def event_metric(predict_res_list, gold_res_list):
    """
    Evaluation for EAE
    """
    tp, fp, fn = 0, 0, 0
    ttp, tfp, tfn = 0, 0, 0

    for (pred_res, gold_res) in zip(predict_res_list, gold_res_list):
        pred_res = eval(pred_res)
        assert ( pred_res['text_id'] == gold_res['text_id'])

        pred_events = get_events(pred_res)
        gold_events = get_events(gold_res)
        for pred_event_item in pred_events:
            pred_type = pred_event_item['event_type']

            
            candidate_gold_item_list = []
            for gold_event_item in gold_events:
                gold_type = gold_event_item['event_type']
                if gold_type == pred_type:
                    candidate_gold_item_list.append(gold_event_item)

            score_list = []
            target_gold_event = None
            if len(candidate_gold_item_list) > 0:
                for candidate_gold_item in candidate_gold_item_list:
                    score = 0
                    for key in candidate_gold_item:
                        if candidate_gold_item[key] == pred_event_item[key]:
                            score += 1
                    score_list.append(score)
                index = score_list.index(max( score_list) )
                target_gold_event = candidate_gold_item_list[index]
                gold_events.remove(target_gold_event)
     
            if target_gold_event is not None:
                assert (pred_event_item['event_type'] == target_gold_event['event_type'])
                for key in ['product', 'region', 'industry']:
                    pred_args, gold_args = pred_event_item[key].split(","), target_gold_event[key].split(',')
                    for arg in pred_args: 
                        if arg != '' and arg in gold_args:
                            tp += 1
                            gold_args.remove(arg)
                        elif arg != '' and arg not in gold_args:
                            fp += 1
                        for aarg in gold_args:
                            fn = fn + 1 if aarg != "" else fn
            else:
                for key in ['product', 'region', 'industry']:
                    pred_args = pred_event_item[key].split(",")
                    for arg in pred_args:
                        fp = fp + 1 if arg != '' else fp
        
        for gold_event_item in gold_events:
            for key in ['product', 'region', 'industry']:
                gold_args = gold_event_item[key].split(',')
                for arg in gold_args:
                    fn = fn + 1 if arg != '' else fn 

    p = (tp) / (tp + fp) if (tp != 0) else 0.0
    r = (tp) / (tp + fn) if (tp != 0) else 0.0
    f1 = (2 * p * r) / (p + r) if ((p+r) != 0 ) else 0.0

    return p * 100.00, r * 100.00, f1 * 100.00, tp, fp, fn


def metric(predict_res_list, gold_res_list):
    """
    Evaluation for CET and ECE
    :param predict_res_list: [{id:x, result:[]}, ... ]
    :param gold_res_list: [{id:x, result:[]}, ...]
    :return:
    """
    tp, fp, fn = 0, 0, 0 # ECE
    ttp, tfp, tfn = 0, 0, 0  # CET


    for (pred_res, gold_res) in zip(predict_res_list, gold_res_list):
        # pred_res/gold_res refer to predicted/golden content of one input sentence
        pred_res = eval(pred_res)
        assert ( pred_res['text_id'] == gold_res['text_id'])
        
        for pred_item in pred_res['result']:
            pred_reason_type, pred_result_type = pred_item['reason_type'], pred_item['result_type']
            
            target_gold_item = None
            candidate_gold_item_list = []
            for gold_item in gold_res['result']:
                if (gold_item['reason_type'] == pred_reason_type) and (gold_item['result_type'] == pred_result_type):
                    candidate_gold_item_list.append( gold_item )
          
            score_list = []
            if len(candidate_gold_item_list) != 0:
                # find the best matched event
                for candidate_gold_item in candidate_gold_item_list:
                    score = 0
                    for key in candidate_gold_item:
                        if candidate_gold_item[key] == pred_item[key]:
                            score += 1
                    score_list.append(score)
                index = score_list.index(max( score_list) )
                target_gold_item = candidate_gold_item_list[index]
                gold_res['result'].remove(target_gold_item)
            
            if target_gold_item is not None:
                assert (pred_reason_type == target_gold_item['reason_type'] and pred_result_type == target_gold_item['result_type'])
                ttp += 1

                for key in ['reason_product', 'reason_region', 'reason_industry']:
                    pred_args, gold_args = pred_item[key].split(","), target_gold_item[key].split(",")
                    for arg in pred_args:
                        if arg != "" and arg in gold_args:
                            tp += 1
                            gold_args.remove(arg)
                        elif arg != "" and arg not in gold_args:
                            fp += 1
                    for aarg in gold_args:
                        fn = fn + 1 if aarg != "" else fn


                for key in ['result_product', 'result_region', 'result_industry']:
                    pred_args, gold_args = pred_item[key].split(","), target_gold_item[key].split(",")
                    for arg in pred_args:
                        if arg != "" and arg in gold_args:
                            tp += 1
                            gold_args.remove(arg)
                        elif arg != "" and arg not in gold_args:
                            fp += 1
                    for aarg in gold_args:
                        fn = fn + 1 if aarg != "" else fn
            else: # arguments under the wrong cause-effect event types are all counted as false positive

                tfp += 1 # FP for CET
                for key in pred_item:
                    if 'type' not in key:
                        pred_args = pred_item[key].split(",")
                        for arg in pred_args:
                            fp = fp + 1 if arg != "" else fp

        # cause-effect event types and arguments which are not predicted
        for gold_item in gold_res['result']:
            tfn += 1
            for key in gold_item:
                if 'type' not in key:
                    gold_args = gold_item[key].split(",")
                    for aarg in gold_args:
                        fn = fn + 1 if aarg != "" else fn

    p = (tp) / (tp + fp) if (tp != 0) else 0.0
    r = (tp) / (tp + fn) if (tp != 0) else 0.0
    f1 = (2 * p * r) / (p + r) if ((p+r) != 0 ) else 0.0

    t_p = (ttp) / (ttp + tfp) if (ttp != 0) else 0.0
    t_r = (ttp) / (ttp + tfn) if (ttp != 0) else 0.0
    t_f1 = (2 * t_p * t_r) / (t_p + t_r) if ( (t_p + t_r) != 0) else 0.0 
    return p * 100.00, r * 100.00, f1 * 100.00, tp, fp, fn, t_p * 100, t_r * 100, t_f1 * 100, ttp, tfp, tfn


def metric_multi_instances(predict_res_list, gold_res_list):
    single_ids, multi_ids, single_gold_res_list, multi_gold_res_list = get_multi_ids(gold_res_list)
    single_predict_res_list, multi_predict_res_list = get_data(predict_res_list, multi_ids, single_ids)
    assert (len(multi_predict_res_list) == len(multi_gold_res_list))
    ep, er, ef, etp, efp, efn = event_metric(copy.deepcopy(multi_predict_res_list), copy.deepcopy(multi_gold_res_list))
    p, r, f, tp, fp, fn, t_p, t_r, t_f, ttp, tfp, tfn = metric(copy.deepcopy(multi_predict_res_list), copy.deepcopy(multi_gold_res_list))
    print("========================= Multi instances ========================================")
    print("EAE: ", "P: ", ep, "R: ", er, "F1: ", ef, "tp: ", etp, "fp: ", efp, "fn: ", efn)
    print("CET: ","P: ", t_p, "R: ", t_r, "F1: ", t_f, "tp: ", ttp, "fp: ", tfp, "fn: ", tfn)
    print("ECE: ", "P: ", p, "R: ", r, "F1: ", f, "tp: ", tp, "fp: ", fp, "fn: ", fn)
    ep, er, ef, etp, efp, efn = event_metric(copy.deepcopy(single_predict_res_list), copy.deepcopy(single_gold_res_list))
    p, r, f, tp, fp, fn, t_p, t_r, t_f, ttp, tfp, tfn = metric(copy.deepcopy(single_predict_res_list), copy.deepcopy(single_gold_res_list))
    print("========================= Single instances ========================================")
    print("EAE: ", "P: ", ep, "R: ", er, "F1: ", ef, "tp: ", etp, "fp: ", efp, "fn: ", efn)
    print("CET: ","P: ", t_p, "R: ", t_r, "F1: ", t_f, "tp: ", ttp, "fp: ", tfp, "fn: ", tfn)
    print("ECE: ", "P: ", p, "R: ", r, "F1: ", f, "tp: ", tp, "fp: ", fp, "fn: ", fn)
    return multi_predict_res_list, single_predict_res_list



if __name__ == '__main__':

    gold_res_list = [{'text_id': 1, 'result':[{'reason_type':1, 'result_type':2, 
                                              'reason_product': '1,6', 'reason_region': '2', 
                                              'reason_industry': '', 'result_product': '3',
                                              'result_region': '4', 'result_industry': ''},
                                              {'reason_type':3, 'result_type':4,
                                              'reason_product': '1', 'reason_region': '6', 
                                              'reason_industry': '', 'result_product': '',
                                              'result_region': '7', 'result_industry': '8'}]}]
    pred_res_list = [str({'text_id': 1, 'result':[{'reason_type':1, 'result_type':2, 
                                              'reason_product': '1', 'reason_region': '2', 
                                              'reason_industry': '3', 'result_product': '',
                                              'result_region': '4', 'result_industry': ''},
                                              {'reason_type':3, 'result_type':4,
                                              'reason_product': '1', 'reason_region': '', 
                                              'reason_industry': '', 'result_product': '',
                                              'result_region': '2', 'result_industry': ''},
                                              {'reason_type':1, 'result_type':4,
                                              'reason_product': '9', 'reason_region': '', 
                                              'reason_industry': '', 'result_product': '',
                                              'result_region': '10', 'result_industry': ''}]})]

    # gold_res_list = [{"text_id": "1660587", "text": "2）不锈钢下跌或受纯镍成本端回落所致；不锈钢供给：短期，受钢厂润下滑影响9月排产环比8月下滑5", "result": [{"reason_type": "运营成本下降", "reason_product": "纯镍", "reason_region": "", "result_region": "", "result_industry": "", "result_type": "市场价格下降", "reason_industry": "", "result_product": "不锈钢"}]},
    # {"text_id": "1615959", "text": "受新疆、河南猪价低迷影响，19Q1业绩下降显著2019年第一季度，公司预计归母净利润约为1381.31万元至3038.89万元，同比下降约45%至75%，主要受非洲猪瘟疫情影响，猪价降幅较大，导致利润大幅下降", "result": [{"reason_type": "猪瘟", "reason_product": "", "reason_region": "", "result_region": "新疆,河南", "result_industry": "", "result_type": "市场价格下降", "reason_industry": "", "result_product": "猪"}]},
    # {"text_id": "131378", "text": "我们认为中国政府很有可能在2016年再次上调新能源附加费,以补充新能源资金,同时可能下调2016/17年光伏电价,或将导致紧急装机和光伏材料价格大幅上升", "result": [{"reason_type": "市场价格下降", "reason_product": "光伏电", "reason_region": "", "result_region": "", "result_industry": "", "result_type": "市场价格提升", "reason_industry": "", "result_product": "紧急装机,光伏材料"}]}]
    # pred_res_list = [{'text_id': '1660587', 'result': [{'reason_type': '运营成本下降', 'result_type': '供给减少', 'reason_product': '', 'reason_region': '', 'reason_industry': '', 'result_product': '不锈钢', 'result_region': '', 'result_industry': ''}]},
    # {'text_id': '1615959', 'result': [{'reason_type': '市场价格下降', 'result_type': '产品利润下降', 'reason_product': '', 'reason_region': '', 'reason_industry': '', 'result_product': '猪', 'result_region': '', 'result_industry': ''}]},
    # {'text_id': '131378', 'result': [{'reason_type': '市场价格提升', 'result_type': '市场价格提升', 'reason_product': '光伏电', 'reason_region': '', 'reason_industry': '', 'result_product': '紧急装机,光伏材料', 'result_region': '', 'result_industry': ''}]}]
    item = metric(pred_res_list, copy.deepcopy(gold_res_list))
    print(item)
    event_item = event_metric(pred_res_list, copy.deepcopy(gold_res_list))
    print(event_item)
