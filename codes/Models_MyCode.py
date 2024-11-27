# coding: utf-8

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from utility.parser import parse_args
args = parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LGMRec(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats, text_feats, inter_matrix, uu, ii, uu_norm, ii_norm):
        super().__init__()
        self.v_feat = image_feats
        self.t_feat = text_feats

        self.embedding_dim = embedding_dim
        self.feat_embed_dim = embedding_dim

        self.n_personal_layer = 1
        self.n_common_layer = 1
        self.n_ui_layers = 3
        self.reg_weight = 0.000001
        self.alpha_1 = 0.3
        self.alpha_2 = 0.8  #best 0.8
        self.alpha_3 = 0.3
        self.alpha_4 = 0.8

        self.topk_uu = 10  #best 5
        self.topk_ii = 10

        self.keep_rate = 0.5
        self.alpha = 0.3
        self.cl_weight = 0.0001
        self.tau = 0.2
        self.ssm_reg = 0.04

        self.n_users = n_users
        self.n_items = n_items
        self.n_nodes = n_users + n_items

        # load dataset info
        self.interaction_matrix = inter_matrix.tocoo().astype(np.float32)
        self.ii = ii.tocoo().astype(np.float32)
        self.uu = uu.tocoo().astype(np.float32)
        self.ii_norm = ii_norm.tocoo().astype(np.float32)
        self.uu_norm = uu_norm.tocoo().astype(np.float32)

        self.adj = self.scipy_matrix_to_sparse_tenser(self.interaction_matrix, torch.Size((self.n_users, self.n_items)))
        self.popularity_adj_ii = self.scipy_matrix_to_sparse_tenser(self.ii, torch.Size((self.n_items, self.n_items)))
        self.popularity_adj_uu = self.scipy_matrix_to_sparse_tenser(self.uu, torch.Size((self.n_users, self.n_users)))
        self.popularity_adj_ii_norm = self.scipy_matrix_to_sparse_tenser(self.ii_norm, torch.Size((self.n_items, self.n_items)))
        self.popularity_adj_uu_norm = self.scipy_matrix_to_sparse_tenser(self.uu_norm, torch.Size((self.n_users, self.n_users)))

        self.num_inters, self.norm_adj = self.get_norm_adj_mat()
        self.num_inters = torch.FloatTensor(1.0 / (self.num_inters + 1e-7)).cuda()

        # init user and item ID embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.drop = nn.Dropout(p=1 - self.keep_rate)
        self.linear1 = nn.Linear(self.n_items, 64)
        self.linear2 = nn.Linear(64, self.n_items)

        # load item modal features

        self.image_embedding = nn.Embedding.from_pretrained(torch.Tensor(self.v_feat), freeze=True)
        self.item_image_trs = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.v_feat.shape[1], self.feat_embed_dim)))
        self.item_image_gate = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.feat_embed_dim, self.feat_embed_dim)))
        self.item_image_bias = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, self.feat_embed_dim)))
        self.item_image_gate_1 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.feat_embed_dim, self.feat_embed_dim)))
        self.item_image_bias_1 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, self.feat_embed_dim)))

        self.text_embedding = nn.Embedding.from_pretrained(torch.Tensor(self.t_feat), freeze=True)
        self.item_text_trs = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.t_feat.shape[1], self.feat_embed_dim)))
        self.item_text_gate = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.feat_embed_dim, self.feat_embed_dim)))
        self.item_text_bias = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, self.feat_embed_dim)))
        self.item_text_gate_1 = nn.Parameter( nn.init.xavier_uniform_(torch.zeros(self.feat_embed_dim, self.feat_embed_dim)))
        self.item_text_bias_1 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, self.feat_embed_dim)))

        image_adj_ii = self.build_sim(self.image_embedding.weight.detach())
        image_adj_ii = self.build_knn_neighbourhood(image_adj_ii, topk=self.topk_ii)
        image_adj_ii = self.compute_normalized_laplacian(image_adj_ii)

        text_adj_ii = self.build_sim(self.text_embedding.weight.detach())
        text_adj_ii = self.build_knn_neighbourhood(text_adj_ii, topk=self.topk_ii)
        text_adj_ii = self.compute_normalized_laplacian(text_adj_ii)

        self.text_content_adj_ii = text_adj_ii.cuda()
        self.image_content_adj_ii = image_adj_ii.cuda()

        ori_user_embs_v = torch.sparse.mm(self.adj, self.image_embedding.weight.cuda()) * self.num_inters[:self.n_users]
        image_adj_uu = self.build_sim(ori_user_embs_v.detach())
        image_adj_uu = self.build_knn_neighbourhood(image_adj_uu, topk=self.topk_uu)
        image_adj_uu = self.compute_normalized_laplacian(image_adj_uu)

        ori_user_embs_t = torch.sparse.mm(self.adj, self.text_embedding.weight.cuda()) * self.num_inters[ :self.n_users]
        text_adj_uu = self.build_sim(ori_user_embs_t.detach())
        text_adj_uu = self.build_knn_neighbourhood(text_adj_uu, topk=self.topk_uu)
        text_adj_uu = self.compute_normalized_laplacian(text_adj_uu)

        # self.text_content_adj_uu = text_adj_uu.cuda()
        # self.image_content_adj_uu = image_adj_uu.cuda()
        #
        # common_ii = self.popularity_adj_ii_norm.cuda()
        # # content_ii = self.text_content_adj_ii.cuda() #beauty  toys
        # # content_ii = self.image_content_adj_ii.cuda()
        # self.text_content_adj_ii.cuda()
        # self.image_content_adj_ii.cuda()
        # content_ii = torch.zeros_like(self.text_content_adj_ii).cuda()
        # # non_zero_positions = (self.text_content_adj_ii != 0) & (self.image_content_adj_ii != 0)
        # non_zero_positions = self.text_content_adj_ii != 0
        # content_ii[non_zero_positions] = self.text_content_adj_ii[non_zero_positions]
        #
        #
        # # 使用稠密张量来存储结果
        # common_ii_dense = common_ii.to_dense().clone()
        # # 创建一个与common_ii_dense形状相同的随机数张量，用于比较
        # rand_probs = torch.rand_like(content_ii)
        # # 应用条件逻辑（在content_ii的概率下将common_ii_dense的非零元素置为0）
        # with torch.no_grad():
        #     # common_ii_dense[common_ii_dense > 0] *= (
        #     #             rand_probs[common_ii_dense > 0] >= content_ii[common_ii_dense > 0]).float()
        #     positive_indices = common_ii_dense > 0
        #     # 计算rand_probs在positive_indices位置上大于等于content_ii在相同位置的值的布尔张量
        #     condition = rand_probs[positive_indices] >= (content_ii[positive_indices] - 0.15)
        #     # 将布尔张量转换为浮点张量，以便进行乘法操作
        #     condition_float = condition.float()
        #     # 更新common_ii_dense矩阵，只更新满足条件的元素
        #     common_ii_dense[positive_indices] *= condition_float
        #
        # # 提取非零元素的索引和值
        # indices = torch.nonzero(common_ii_dense, as_tuple=False)
        # values = common_ii_dense[indices[:, 0], indices[:, 1]]
        # # 重新创建稀疏张量
        # self.common_ii_t = torch.sparse_coo_tensor(indices.t(), values, common_ii.size())
        #
        #
        # common_uu = self.popularity_adj_uu_norm.cuda()
        # # content_uu = self.text_content_adj_uu.cuda()
        # # content_uu = self.image_content_adj_uu.cuda()
        #
        # self.text_content_adj_uu.cuda()
        # self.image_content_adj_uu.cuda()
        # content_uu = torch.zeros_like(self.text_content_adj_uu).cuda()
        # non_zero_positions = (self.text_content_adj_uu != 0) & (self.image_content_adj_uu != 0)
        # content_uu[non_zero_positions] = self.text_content_adj_uu[non_zero_positions]
        #
        # common_uu_dense = common_uu.to_dense().clone()
        # rand_probs = torch.rand_like(content_uu)
        # with torch.no_grad():
        #     common_uu_dense[common_uu_dense > 0] *= (
        #             rand_probs[common_uu_dense > 0] >= content_uu[common_uu_dense > 0]).float()
        # indices = torch.nonzero(common_uu_dense, as_tuple=False)
        # values = common_uu_dense[indices[:, 0], indices[:, 1]]
        # self.common_uu = torch.sparse_coo_tensor(indices.t(), values, common_uu.size())


        '''
        cpu
        '''
        # self.text_content_adj_uu = self.text_content_adj_uu.cpu()
        # self.image_content_adj_uu = self.image_content_adj_uu.cpu()
        # content_uu = torch.zeros_like(self.text_content_adj_uu)
        # content_uu = content_uu.cpu()
        # non_zero_positions = (self.text_content_adj_uu != 0) & (self.image_content_adj_uu != 0)
        # content_uu[non_zero_positions] = self.text_content_adj_uu[non_zero_positions]
        #
        # common_uu_dense = common_uu.to_dense().clone()
        # common_uu_dense = common_uu_dense.cpu()
        # rand_probs = torch.rand_like(content_uu)
        # with torch.no_grad():
        #     common_uu_dense[common_uu_dense > 0] *= (
        #             rand_probs[common_uu_dense > 0] >= content_uu[common_uu_dense > 0]).float()
        #     # positive_indices = common_uu_dense > 0
        #     # condition = rand_probs[positive_indices] >= (content_uu[positive_indices] + 0.1)
        #     # condition_float = condition.float()
        #     # common_uu_dense[positive_indices] *= condition_float
        #
        # indices = torch.nonzero(common_uu_dense, as_tuple=False)
        # values = common_uu_dense[indices[:, 0], indices[:, 1]]
        # self.common_uu = torch.sparse_coo_tensor(indices.t(), values, common_uu.size())
        # self.common_uu = self.common_uu.cuda()
        # self.text_content_adj_uu = self.text_content_adj_uu.cuda()
        # self.image_content_adj_uu = self.image_content_adj_uu.cuda()

        self.text_content_adj_uu = text_adj_uu.cuda()
        self.image_content_adj_uu = image_adj_uu.cuda()

        common_ii = self.popularity_adj_ii_norm.cuda()
        self.text_content_adj_ii.cuda()
        self.image_content_adj_ii.cuda()
        content_ii_t = torch.zeros_like(self.text_content_adj_ii).cuda()
        content_ii_v = torch.zeros_like(self.image_content_adj_ii).cuda()
        non_zero_positions = self.text_content_adj_ii != 0
        content_ii_t[non_zero_positions] = self.text_content_adj_ii[non_zero_positions]
        non_zero_positions_v = self.image_content_adj_ii != 0
        content_ii_v[non_zero_positions] = self.image_content_adj_ii[non_zero_positions_v]


        common_ii_dense_t = common_ii.to_dense().clone()
        common_ii_dense_v = common_ii.to_dense().clone()
        rand_probs_t = torch.rand_like(content_ii_t)
        rand_probs_v = torch.rand_like(content_ii_v)
        # 应用条件逻辑（在content_ii的概率下将common_ii_dense的非零元素置为0）
        with torch.no_grad():
            positive_indices = common_ii_dense_t > 0
            condition = rand_probs_t[positive_indices] >= (content_ii_t[positive_indices])
            # 将布尔张量转换为浮点张量，以便进行乘法操作
            condition_float = condition.float()
            # 更新common_ii_dense矩阵，只更新满足条件的元素
            common_ii_dense_t[positive_indices] *= condition_float

        # 提取非零元素的索引和值
        indices = torch.nonzero(common_ii_dense_t, as_tuple=False)
        values = common_ii_dense_t[indices[:, 0], indices[:, 1]]
        # 重新创建稀疏张量
        self.common_ii_t = torch.sparse_coo_tensor(indices.t(), values, common_ii.size())

        with torch.no_grad():
            positive_indices = common_ii_dense_v > 0
            condition = rand_probs_v[positive_indices] >= (content_ii_v[positive_indices])
            condition_float = condition.float()
            common_ii_dense_v[positive_indices] *= condition_float
        indices = torch.nonzero(common_ii_dense_v, as_tuple=False)
        values = common_ii_dense_v[indices[:, 0], indices[:, 1]]
        self.common_ii_v = torch.sparse_coo_tensor(indices.t(), values, common_ii.size())

        common_uu = self.popularity_adj_uu_norm.cpu()
        self.text_content_adj_uu = self.text_content_adj_uu.cpu()
        self.image_content_adj_uu = self.image_content_adj_uu.cpu()
        content_uu_t = torch.zeros_like(self.text_content_adj_uu).cpu()
        content_uu_v = torch.zeros_like(self.image_content_adj_uu).cpu()
        non_zero_positions = self.text_content_adj_uu != 0
        non_zero_positions = non_zero_positions.cpu()
        self.text_content_adj_uu =  self.text_content_adj_uu.cpu()
        content_uu_t[non_zero_positions] = self.text_content_adj_uu[non_zero_positions]
        non_zero_positions_v = self.image_content_adj_uu != 0
        non_zero_positions_v =non_zero_positions_v.cpu()
        self.image_content_adj_uu = self.image_content_adj_uu.cpu()
        content_uu_v[non_zero_positions] = self.image_content_adj_uu[non_zero_positions_v]

        common_uu_dense_t = common_uu.to_dense().clone()
        common_uu_dense_v = common_uu.to_dense().clone()
        common_uu_dense_t = common_uu_dense_t.cpu()
        common_uu_dense_v = common_uu_dense_v.cpu()
        rand_probs_t = torch.rand_like(content_uu_t)
        rand_probs_v = torch.rand_like(content_uu_v)
        rand_probs_t = rand_probs_t.cpu()
        rand_probs_t = rand_probs_t.cpu()
        with torch.no_grad():
            positive_indices = common_uu_dense_t > 0
            condition = rand_probs_t[positive_indices] >= (content_uu_t[positive_indices])
            condition_float = condition.float()
            common_uu_dense_t[positive_indices] *= condition_float

        indices = torch.nonzero(common_uu_dense_t, as_tuple=False)
        values = common_uu_dense_t[indices[:, 0], indices[:, 1]]
        self.common_uu_t = torch.sparse_coo_tensor(indices.t(), values, common_uu.size())
        self.common_uu_t = self.common_uu_t.cuda()

        with torch.no_grad():
            positive_indices = common_uu_dense_v > 0
            condition = rand_probs_v[positive_indices] >= (content_uu_v[positive_indices])
            condition_float = condition.float()
            common_uu_dense_v[positive_indices] *= condition_float
        indices = torch.nonzero(common_uu_dense_v, as_tuple=False)
        values = common_uu_dense_v[indices[:, 0], indices[:, 1]]
        self.common_uu_v = torch.sparse_coo_tensor(indices.t(), values, common_uu.size())
        self.common_uu_v = self.common_uu_v.cuda()




    def scipy_matrix_to_sparse_tenser(self, matrix, shape):
        row = matrix.row
        col = matrix.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(matrix.data)
        return torch.sparse.FloatTensor(i, data, shape).cuda()

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        return sumArr, self.scipy_matrix_to_sparse_tenser(L, torch.Size((self.n_nodes, self.n_nodes)))

    def build_knn_neighbourhood(self, adj, topk):
        knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
        weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        return weighted_adjacency_matrix

    def compute_normalized_laplacian(self,adj):
        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
        L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return L_norm

    def build_sim(self, context):
        context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        return sim

    # collaborative graph embedding
    def behavior(self):
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        cge_embs = []
        for _ in range(self.n_ui_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            cge_embs += [ego_embeddings]
        cge_embs = torch.stack(cge_embs, dim=1)
        cge_embs = cge_embs.mean(dim=1, keepdim=False)
        return cge_embs


    def self_gating(self, str='v'):
        if str == 'v':
            item_feats = torch.mm(self.image_embedding.weight, self.item_image_trs)
            common_item_feats = torch.mul(F.sigmoid(torch.mm(item_feats, self.item_image_gate)+self.item_image_bias), item_feats)
            personal_item_feats = torch.mul(F.sigmoid(torch.mm(item_feats-common_item_feats, self.item_image_gate_1) + self.item_image_bias_1),item_feats)

        elif str == 't':
            item_feats = torch.mm(self.text_embedding.weight, self.item_text_trs)
            common_item_feats = torch.mul(F.sigmoid(torch.mm(item_feats, self.item_text_gate) + self.item_text_bias), item_feats)
            personal_item_feats = torch.mul(F.sigmoid(torch.mm(item_feats-common_item_feats, self.item_text_gate_1) + self.item_text_bias_1), item_feats)


        return personal_item_feats, common_item_feats




    def forward(self):
        # behavior-driven
        id_embs = self.behavior()

        # obtain disentangled embeddings
        personal_item_embs_v, common_item_embs_v = self.self_gating('v')
        personal_item_embs_t, common_item_embs_t = self.self_gating('t')

        # personal-driven
        personal_user_embs_v = torch.sparse.mm(self.adj, personal_item_embs_v) * self.num_inters[:self.n_users]
        personal_user_embs_t = torch.sparse.mm(self.adj, personal_item_embs_t) * self.num_inters[:self.n_users]

        for _ in range(self.n_personal_layer):
            personal_item_embs_v = torch.sparse.mm(self.image_content_adj_ii, personal_item_embs_v)
            personal_item_embs_t = torch.sparse.mm(self.text_content_adj_ii, personal_item_embs_t)
            self.image_content_adj_uu = self.image_content_adj_uu.cuda()
            personal_user_embs_v = torch.sparse.mm(self.image_content_adj_uu, personal_user_embs_v)
            self.text_content_adj_uu = self.text_content_adj_uu.cuda()
            personal_user_embs_t = torch.sparse.mm(self.text_content_adj_uu, personal_user_embs_t)

        personal_embs_v = torch.cat([personal_user_embs_v, personal_item_embs_v], dim=0)
        personal_embs_t = torch.cat([personal_user_embs_t, personal_item_embs_t], dim=0)


        personal_embs = self.alpha_1 * F.normalize(personal_embs_v) + self.alpha_2 * F.normalize(personal_embs_t)
        # personal_embs = F.normalize(personal_embs_v)

        # common-driven
        common_user_embs_v = torch.sparse.mm(self.adj, common_item_embs_v) * self.num_inters[:self.n_users]
        common_user_embs_t = torch.sparse.mm(self.adj, common_item_embs_t) * self.num_inters[:self.n_users]

        #MLP → self.adj
        # x = torch.relu(self.linear1(self.adj))
        # x = x.cuda()
        # new_adj = self.linear2(x)
        # common_user_embs_v = torch.sparse.mm(new_adj, common_item_embs_v) * self.num_inters[:self.n_users]
        # common_user_embs_t = torch.sparse.mm(new_adj, common_item_embs_t) * self.num_inters[:self.n_users]


        for _ in range(self.n_common_layer):
            # Uncensored edge
            common_item_embs_v = torch.sparse.mm(self.popularity_adj_ii_norm, common_item_embs_v)
            common_item_embs_t = torch.sparse.mm(self.popularity_adj_ii_norm, common_item_embs_t)
            common_user_embs_v = torch.sparse.mm(self.popularity_adj_uu_norm, self.user_embedding.weight)
            common_user_embs_t = torch.sparse.mm(self.popularity_adj_uu_norm, self.user_embedding.weight)

            common_item_embs_v = torch.sparse.mm(self.common_ii_v, common_item_embs_v)
            common_item_embs_t = torch.sparse.mm(self.common_ii_t, common_item_embs_t)
            common_user_embs_v = torch.sparse.mm(self.common_uu_v, self.user_embedding.weight)
            common_user_embs_t = torch.sparse.mm(self.common_uu_t, self.user_embedding.weight)


        common_embs_v = torch.cat([common_user_embs_v, F.normalize(common_item_embs_v)], dim=0)
        common_embs_t = torch.cat([common_user_embs_t, F.normalize(common_item_embs_t)], dim=0)

        # common_embs = self.alpha_3 * common_embs_v + self.alpha_4 * common_embs_t
        common_embs = self.alpha_3 * F.normalize(common_embs_v) + self.alpha_4 * F.normalize(common_embs_t)
        # common_embs = F.normalize(common_embs_v)

        # all_embs = torch.cat((id_embs, personal_embs, common_embs), dim=1)
        # all_embs = torch.cat((id_embs, personal_embs), dim=1)
        # all_embs = id_embs + personal_embs + 0.5*common_embs
        all_embs = id_embs + personal_embs + 0.5*common_embs
        self.id_embs = id_embs
        self.personal_embs = personal_embs
        self.common_embs = common_embs


        # Do not split multimodes
        adj_ii = self.text_content_adj_ii + self.image_content_adj_ii
        adj_uu = self.text_content_adj_uu + self.image_content_adj_uu

        item_embs_v = torch.mm(self.image_embedding.weight, self.item_image_trs)
        item_embs_t = torch.mm(self.text_embedding.weight, self.item_text_trs)
        user_embs_v = torch.sparse.mm(self.adj, item_embs_v) * self.num_inters[:self.n_users]
        user_embs_t = torch.sparse.mm(self.adj, item_embs_t) * self.num_inters[:self.n_users]

        for _ in range(self.n_personal_layer):
            item_embs_v = torch.sparse.mm(adj_ii, item_embs_v)
            item_embs_t = torch.sparse.mm(adj_ii, item_embs_t)
            user_embs_v = torch.sparse.mm(adj_uu, user_embs_v)
            user_embs_t = torch.sparse.mm(adj_uu, user_embs_t)

        embs_v = torch.cat([user_embs_v, item_embs_v], dim=0)
        embs_t = torch.cat([user_embs_t, item_embs_t], dim=0)
        multimodal_embs = self.alpha_1 * F.normalize(embs_v) + self.alpha_2 * F.normalize(embs_t)
        # all_embs = multimodal_embs + id_embs

        '''
        '''
        #MLP
        # all_embs = torch.cat((id_embs, personal_embs, common_embs), dim=1)
        # x = torch.relu(self.linear1(all_embs))
        # x.to(device)
        # all_embs = self.linear2(x)
        '''
        '''

        u_embs, i_embs = torch.split(all_embs, [self.n_users, self.n_items], dim=0)


        return u_embs, i_embs, [personal_user_embs_v, personal_item_embs_v, personal_user_embs_t, personal_item_embs_t], [common_user_embs_v, common_item_embs_v, common_user_embs_t, common_item_embs_t]

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        return bpr_loss

    def ssl_triple_loss(self, emb1, emb2, all_emb):
        norm_emb1 = F.normalize(emb1)
        norm_emb2 = F.normalize(emb2)
        norm_all_emb = F.normalize(all_emb)
        pos_score = torch.exp(torch.mul(norm_emb1, norm_emb2).sum(dim=1) / self.tau)
        ttl_score = torch.exp(torch.matmul(norm_emb1, norm_all_emb.T) / self.tau).sum(dim=1)
        ssl_loss = -torch.log(pos_score / ttl_score).sum()
        return self.cl_weight*ssl_loss

    def reg_loss(self, *embs):
        reg_loss = 0
        for emb in embs:
            reg_loss += torch.norm(emb, p=2)
        reg_loss /= embs[-1].shape[0]
        return self.reg_weight * reg_loss

    def ssm_loss(self, users, pos_items):
        pos_user_norm = F.normalize(users)
        pos_item_norm = F.normalize(pos_items)

        pos_score = torch.sum(pos_user_norm * pos_item_norm, dim=1)
        neg_score = torch.matmul(pos_user_norm, pos_item_norm.t())

        pos_score = torch.exp(pos_score / 0.2)
        neg_score = torch.sum(torch.exp(neg_score / 0.2), dim=1)

        ssm_loss = (-1) * torch.log(pos_score / neg_score)
        ssm_loss = torch.mean(ssm_loss)

        return self.ssm_reg * ssm_loss

    def get_embeddings(self):
        id_embs = self.id_embs
        personal_embs = self.personal_embs
        common_embs = self.common_embs

        user_id_embeddings, item_id_embeddings = torch.split(id_embs, [self.n_users, self.n_items])
        user_personal_embeddings, item_personal_embeddings = torch.split(personal_embs, [self.n_users, self.n_items])
        user_common_embeddings, item_common_embeddings = torch.split(common_embs, [self.n_users, self.n_items])

        return user_id_embeddings, item_id_embeddings, user_personal_embeddings, item_personal_embeddings, user_common_embeddings, item_common_embeddings


