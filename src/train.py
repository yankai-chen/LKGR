import random
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import *
from time import time
from utility.data_loader import *
from utility.log_helper import *

CURVATURE_THRESHOLD = 0.01


def topk_evaluate(model, n_item, user_list, train_record, test_record, k_list, device):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}

    with torch.no_grad():
        model.eval()
        test_pos_item_binary = torch.zeros([len(user_list), n_item], dtype=torch.float32)
        user_index = torch.LongTensor(user_list)
        item_index = torch.LongTensor(np.arange(n_item))
        user_index = user_index.to(device)
        item_index = item_index.to(device)
        score = model('batch_score', user_index, item_index)
        # score = model('get_rank_score', user_index, item_index)
        for i in range(len(user_list)):
            user = user_list[i]
            train_pos_item_list = torch.LongTensor(list(train_record[user]))
            test_pos_item_list = torch.LongTensor(list(test_record[user]))
            score[i][train_pos_item_list] = 0
            test_pos_item_binary[i][test_pos_item_list] = 1

            _, rank_indices = torch.sort(score, descending=True)

        for i in range(len(user_list)):
            user = user_list[i]
            binary_hit = test_pos_item_binary[i][rank_indices[i]]
            binary_hit = binary_hit.numpy()
            for k in k_list:
                hit_k = binary_hit[:k]
                hit_num = np.sum(hit_k)
                precision_list[k].append(hit_num / k)
                recall_list[k].append(hit_num / len(test_record[user]))
                dcg = np.sum((2 ** hit_k - 1) / np.log2(np.arange(2, k + 2)))
                sorted_hits_k = np.flip(np.sort(binary_hit))[:k]
                idcg = np.sum((2 ** sorted_hits_k - 1) / np.log2(np.arange(2, k + 2)))
                # idcg[idcg == 0] = np.inf
                ndcg_list[k].append(dcg / idcg)

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    ndcg = [np.mean(ndcg_list[k]) for k in k_list]

    return precision, recall, ndcg


def get_total_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def topk_setting(train_data, test_data, n_item):
    user_num = 100
    k_list = [1, 5, 10, 20, 50, 100]
    train_record = get_user_record(train_data, True)
    test_record = get_user_record(test_data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    if len(user_list) > user_num:
        user_list = np.random.choice(user_list, size=user_num, replace=False)
    item_set = set(list(range(n_item)))
    return user_list, train_record, test_record, item_set, k_list


def exp_i(args, train_file, eval_file, test_file, logging):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    train_data, eval_data, test_data, kg_dict, user_item_dict, item_user_dict, n_relation, n_entity, n_triplet \
        = load_data(args, train_file, eval_file, test_file)

    n_user = max(max(train_data[:, 0]), max(eval_data[:, 0]), max(test_data[:, 0])) + 1
    n_item = max(max(train_data[:, 1]), max(eval_data[:, 1]), max(test_data[:, 1])) + 1

    # gpu/cpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    model = LKGR(args, n_user, n_item, n_entity, n_relation).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight)


    logging.info('n_users:           %d' % n_user)
    logging.info('n_items:           %d' % n_item)
    logging.info('n_entities:         %d' % n_entity)
    logging.info('n_users_entities:   %d' % (n_entity + n_user))
    logging.info('n_relations:        %d' % n_relation)

    eval_precision_list = []
    eval_recall_list = []
    eval_ndcg_list = []

    test_precision_list = []
    test_recall_list = []
    test_ndcg_list = []

    best_eval_recall = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_epoch = [0, 0, 0, 0, 0, 0]

    adj_u2i, adj_i2u, adj_i2e, adj_relation = construct_adj(args, n_user, n_item, n_entity, kg_dict, user_item_dict,
                                                            item_user_dict)
    adj_u2i = adj_u2i.to(device)
    adj_i2u = adj_i2u.to(device)
    adj_i2e = adj_i2e.to(device)
    adj_relation = adj_relation.to(device)
    model.set_adj_matrix(adj_u2i=adj_u2i, adj_i2u=adj_i2u, adj_entity=adj_i2e, adj_relation=adj_relation)

    for epoch in range(1, args.n_epochs + 1):
        start = 0
        iter = 0
        total_loss = 0
        np.random.shuffle(train_data)

        # currently, we are using the dynamic sampling for neighbors

        model.train()
        time0 = time()
        # train
        while start + args.batch_size <= len(train_data):

            optimizer.zero_grad()
            user_indices, item_indices, labels = get_feed(train_data, start, start + args.batch_size)
            user_indices = user_indices.to(device)
            item_indices = item_indices.to(device)
            labels = labels.to(device)
            loss = model('cal_loss', user_indices, item_indices, labels)

            loss.backward()

            if model.c.requires_grad and model.c.item() < CURVATURE_THRESHOLD:
                model.clamp_curvature(CURVATURE_THRESHOLD)

            optimizer.step()
            total_loss += loss.item()

            iter += 1
            start += args.batch_size
        time_train = time() - time0

        #  # top-K evaluation
        time0 = time()
        user_list, train_record, eval_record, item_set, k_list = topk_setting(train_data, eval_data, n_item)
        eval_precision, eval_recall, eval_ndcg = topk_evaluate(model, n_item, user_list, train_record,
                                                                   eval_record, k_list, device)

        user_list, train_record, test_record, item_set, k_list = topk_setting(train_data, test_data, n_item)
        test_precision, test_recall, test_ndcg = topk_evaluate(model, n_item, user_list, train_record,
                                                                   test_record, k_list, device)
        time1 = time() - time0

        eval_precision_list.append(eval_precision)
        eval_recall_list.append(eval_recall)
        eval_ndcg_list.append(eval_ndcg)

        test_precision_list.append(test_precision)
        test_recall_list.append(test_recall)
        test_ndcg_list.append(test_ndcg)

        for i, _ in enumerate(k_list):
            if eval_recall[i] > best_eval_recall[i]:
                best_eval_recall[i] = eval_recall[i]
                best_epoch[i] = epoch

        line1 = 'Eval P:['

        for i in eval_precision:
            line1 = line1 + '%.4f ' % i
        line1 = line1 + '] | R:['
        for i in eval_recall:
            line1 = line1 + '%.4f ' % i
        line1 = line1 + '] | NDCG: ['
        for i in eval_ndcg:
            line1 = line1 + '%.4f ' % i
        line1 += ']'

        line2 = 'Test P:['
        for i in test_precision:
            line2 = line2 + '%.4f ' % i
        line2 = line2 + '] | R:'
        for i in test_recall:
            line2 = line2 + '%.4f ' % i
        line2 = line2 + '] | NDCG:'
        for i in test_ndcg:
            line2 = line2 + '%.4f ' % i
        line2 += ']'

        logging.info('Epoch %d  Loss: %.4f  Train-Time: %.3fs  Eval-Time: %.3fs '
                         % (epoch, total_loss / iter, time_train, time1))

        logging.info(line1)
        logging.info(line2)
        logging.info('')

    total_num = get_total_parameters(model)
    logging.info('Total: %d ' % total_num)

    k_list = [1, 5, 10, 20, 50, 100]
    return best_epoch, k_list, \
           eval_precision_list, eval_recall_list, eval_ndcg_list, \
           test_precision_list, test_recall_list, test_ndcg_list


def Exp_run(args):
    log_name = create_log_name(args.saved_dir)
    log_config(path=args.saved_dir, name=log_name, level=logging.DEBUG, console_level=logging.DEBUG, console=True)
    logging.info(args)
    logging.info('Run-exp-TopK_recommendation')

    files = [1, 2, 3, 4, 5]
    recall_list = [[] for _ in range(6)]
    ndcg_list = [[] for _ in range(6)]
    for i in files:
        train_file = 'train_' + str(i) + '.txt'
        eval_file = 'eval_' + str(i) + '.txt'
        test_file = 'test_' + str(i) + '.txt'
        logging.info(train_file)
        best_epoch, k_list, \
        eval_precision_list, eval_recall_list, eval_ndcg_list, \
        test_precision_list, test_recall_list, test_ndcg_list = exp_i(args, train_file, eval_file, test_file, logging)

        logging.info('')

        logging.info('Top@k Evaluation')
        for j, k in enumerate(k_list):
            idx = best_epoch[j]
            recall_list[j].append(test_recall_list[idx - 1][j])
            ndcg_list[j].append(test_ndcg_list[idx - 1][j])
            logging.info('Top@%d:  Best epoch: %d  corresponding Test R: %.4f    corresponding Test NDCG: %.4f'
                         % (k, best_epoch[j], test_recall_list[idx - 1][j], test_ndcg_list[idx - 1][j]))
        logging.info('--------------------------------')
        logging.info('')

    for i, k in enumerate(k_list):
        logging.info('Top@%d recommendation:   Avg-best-R %.4f | Avg-best-NDCG: %.4f ' %
                     (k, np.mean(recall_list[i]), np.mean(ndcg_list[i])))

    logging.info('********************************************************************************************')
    logging.info('********************************************************************************************')

