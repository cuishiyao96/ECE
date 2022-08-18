from model import Model
from log import Log
from metric import metric, event_metric, metric_multi_instances
from data_loader import Data_data
from utils import *


def train(model, train_data, dev_data, test_data, dev_answer_list):

    

    num_training_steps = int(train_data.corpus_length / (args.batch_size )) * args.epoches
    warmup_steps = int(args.warmup_rate * num_training_steps)
    optimizer, scheduler = get_optimizer(model, args, warmup_steps, num_training_steps)
    optimizer.zero_grad()
    best_f1 = -1

    for epoch in range(args.epoches):
        loss = train_epoch(model, train_data, optimizer, scheduler)
        log_handler.info("Epoch: {}, Loss: {} ".format(epoch + 1, loss))
        torch.save(model.state_dict(), "../models/{}/{}.pickle".format(args.task_name, str(epoch + 1)))
        dev_result = inference(model, dev_data)
        
        # torch.save(model.state_dict(), "../models/{}/{}.pickle".format(args.task_name, str(epoch + 1)))
        write_predicted_results(args.task_name, 'dev', epoch + 1, dev_result)
        curr_p, curr_r, curr_f1, tp, fp, fn, type_p, type_r, type_f1, ttp, tfp, tfn = metric(dev_result, copy.deepcopy(dev_answer_list))
        log_handler.info("Epoch: {}, type precision: {}, type recall: {}, type f1: {}, type tp: {}, type fp: {}, type fn: {}".format(epoch + 1, type_p, type_r, type_f1, ttp, tfp, tfn))
        log_handler.info("Epoch: {}, precision: {}, recall: {}, f1: {}, tp: {}, fp: {}, fn:{} \n".format(epoch + 1, curr_p, curr_r, curr_f1, tp, fp, fn))
            
        


def inference(model, eval_data):
     
    result = []
    model.eval()
    for (text_ids, input_ids, input_masks, segment_ids, label_indexs, _, ) in eval_data.iter_batch():
        one_result = model.inference(text_ids, input_ids,segment_ids,  input_masks, label_indexs)
        result.append(str(one_result))

    return result


def train_epoch(model, train_data, optimizer, scheduler):
    model.train()    
    tt_loss_list, rr_loss_list, loss_list = [], [], []
    step, start = 0, time.time()
    counter = 0
    for (text_ids, input_ids, input_masks, segment_ids, label_indexs, tt_labels) in train_data.iter_batch():
        
        counter += 1
        tt_outputs = model(input_ids, segment_ids, input_masks, label_indexs)
        # pdb.set_trace()                               
        dim = tt_outputs.shape[-1]
        masks = input_masks[:, -max_seq_len:].unsqueeze(1).expand(-1, len(etype_map), max_seq_len).bool()  
        tt_outputs_used = torch.masked_select(tt_outputs, masks.bool().unsqueeze(-1).repeat(1,1,1,len(tt_map))).reshape(-1, len(tt_map))
        tt_labels_used = torch.masked_select(tt_labels, masks.unsqueeze(-1).repeat(1, 1, 1, len(tt_map))).reshape(-1, len(tt_map))
        loss = loss_func(tt_outputs_used.pow(args.pow), tt_labels_used)

        loss_list.append(loss.item())
        loss.backward()
        if args.clip_norm > 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step == 0:
            step_time = time.time() - start
        if step % 200 == 0:
            print("Loss: ", loss.item())
            print("Remain Time: {} min".format( str(step_time * (train_data.len_data - step) / 60) ) )
        step += 1

    return sum(loss_list) / len(loss_list)




parser = argparse.ArgumentParser(description='test')
parser.add_argument('--task_name', default='no', type=str)
parser.add_argument('--model_name', default='no', type=str)
parser.add_argument('--data_path', default='../data/', type=str)
parser.add_argument('--train_data_path', default="../data/processed_data/train.pickle", type=str)
parser.add_argument('--dev_data_path', default="../data/processed_data/dev.pickle", type=str)
parser.add_argument('--test_data_path', default="../data/processed_data/test.pickle", type=str)
parser.add_argument('--debug', default=1, type=int)
parser.add_argument('--training', default=1, type=int)
parser.add_argument('--bert_path', default='/home/cuishiyao/data/bert-base-zh', type=str)
parser.add_argument('--epoches', default=10, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--pow', default=2, type=int)
parser.add_argument('--l2_penalty', default=0.00001, type=float)
parser.add_argument('--clip_norm', default=5, type=float)
parser.add_argument('--lr', default=0.00005, type=float)
parser.add_argument('--dropout', default=0.0, type=float)
parser.add_argument('--warmup_rate', default=0.1, type=float)
parser.add_argument('--reduc', default='sum',type=str)
parser.add_argument('--hidden_size', default=768, type=int)
parser.add_argument('--thresh', default=0.5, type=float)
args = parser.parse_args()

log = Log(args.task_name + ".log")
log_handler = log.getLog()
if args.training:
    for arg in vars(args):
        log_handler.info("{}: {}".format(arg, getattr(args, arg)))
    log_handler.info("\n")

os.environ['PYTHONHASHSEED'] = str(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

Mkdir("../res/{}".format(args.task_name))
Mkdir("../models/{}".format(args.task_name))

train_data_ = load_data(args.train_data_path, debug=args.debug)
train_data = Data_data(train_data_, batch_size=args.batch_size, data_type='train')
dev_data_ = load_data(args.dev_data_path, debug=args.debug)
dev_data = Data_data(dev_data_, batch_size=1, data_type='dev')
test_data_ = load_data(args.test_data_path, debug=args.debug)
test_data = Data_data(test_data_, batch_size=1, data_type='test')
dev_answer_list, test_answer_list, train_answer_list = load_gold_answer(args.data_path)


loss_func = nn.BCELoss(reduction=args.reduc)
model = Model(args)
model = trans_to_cuda(model)

if args.training:
    train(model, train_data, dev_data, test_data, dev_answer_list)
else:
    model.load_state_dict(torch.load("../models/{}/{}.pickle".format(args.task_name, args.model_name)))
    eval_data, eval_answer_list = test_data, test_answer_list
    final_results = inference(model, eval_data)
    write_predicted_results(args.task_name, 'eval', args.model_name, final_results)
    ep, er, ef, etp, efp, efn = event_metric(copy.deepcopy(final_results), copy.deepcopy(eval_answer_list))
    p, r, f, tp, fp, fn, t_p, t_r, t_f, ttp, tfp, tfn = metric(copy.deepcopy(final_results), copy.deepcopy(eval_answer_list))
    print("EAE: ", "P: ", ep, "R: ", er, "F1: ", ef)
    print("CET: ","P: ", t_p, "R: ", t_r, "F1: ", t_f)
    print("ECE: ", "P: ", p, "R: ", r, "F1: ", f)
    multi_predict_res_list, single_pred_res_list = metric_multi_instances(copy.deepcopy(final_results), copy.deepcopy(eval_answer_list))
