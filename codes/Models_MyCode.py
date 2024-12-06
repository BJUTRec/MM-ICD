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
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats, text_feats, inter_matrix, uu, ii, uu_norm, ii_norm, P_i, C_u):
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

        self.topk_uu = 5  #best 5
        self.topk_ii = 5

        self.keep_rate = 0.5
        self.alpha = 0.3
        self.cl_weight = 0.0001
        self.tau = 0.2
        self.ssm_reg = 0.04 #best 0.04？

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

        self.P_i = torch.reshape(torch.Tensor(P_i), shape=[self.n_items, 1]).cuda()
        self.C_u = torch.reshape(torch.Tensor(C_u), shape=[self.n_users, 1]).cuda()
        self.att = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(2, 1)))

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
        self.item_text_gate_1 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.feat_embed_dim, self.feat_embed_dim)))
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

        self.text_content_adj_uu = text_adj_uu.cuda()
        self.image_content_adj_uu = image_adj_uu.cuda()
########################################################################################################################
        common_ii = self.popularity_adj_ii_norm.cuda()
        self.text_content_adj_ii.cuda()
        self.image_content_adj_ii.cuda()
        content_ii = torch.zeros_like(self.text_content_adj_ii).cuda()
        non_zero_positions = (self.text_content_adj_ii != 0) & (self.image_content_adj_ii != 0)
        content_ii[non_zero_positions] = self.text_content_adj_ii[non_zero_positions]

        common_ii_dense = common_ii.to_dense().clone()
        rand_probs = torch.rand_like(content_ii)
        # How to remove edges
        with torch.no_grad():
            positive_indices = common_ii_dense > 0
            condition = rand_probs[positive_indices] >= (content_ii[positive_indices] - 0.15)
            condition_float = condition.float()
            common_ii_dense[positive_indices] *= condition_float
        indices = torch.nonzero(common_ii_dense, as_tuple=False)
        values = common_ii_dense[indices[:, 0], indices[:, 1]]

        self.common_ii = torch.sparse_coo_tensor(indices.t(), values, common_ii.size())


########################################################################################################################
        common_uu = self.popularity_adj_uu_norm.cuda()
        self.text_content_adj_uu.cuda()
        self.image_content_adj_uu.cuda()
        content_uu = torch.zeros_like(self.text_content_adj_uu).cuda()
        non_zero_positions = (self.text_content_adj_uu != 0) & (self.image_content_adj_uu != 0)
        content_uu[non_zero_positions] = self.text_content_adj_uu[non_zero_positions]

        common_uu_dense = common_uu.to_dense().clone()
        rand_probs = torch.rand_like(content_uu)
        with torch.no_grad():
            common_uu_dense[common_uu_dense > 0] *= (
                    rand_probs[common_uu_dense > 0] >= content_uu[common_uu_dense > 0]).float()
        indices = torch.nonzero(common_uu_dense, as_tuple=False)
        values = common_uu_dense[indices[:, 0], indices[:, 1]]
        self.common_uu = torch.sparse_coo_tensor(indices.t(), values, common_uu.size())


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
            personal_user_embs_v = torch.sparse.mm(self.image_content_adj_uu, personal_user_embs_v)
            personal_user_embs_t = torch.sparse.mm(self.text_content_adj_uu, personal_user_embs_t)

        personal_embs_v = torch.cat([personal_user_embs_v, personal_item_embs_v], dim=0)
        personal_embs_t = torch.cat([personal_user_embs_t, personal_item_embs_t], dim=0)

        personal_embs = self.alpha_1 * F.normalize(personal_embs_v) + self.alpha_2 * F.normalize(personal_embs_t)

        # common-driven
        common_user_embs_v = torch.sparse.mm(self.adj, common_item_embs_v) * self.num_inters[:self.n_users]
        common_user_embs_t = torch.sparse.mm(self.adj, common_item_embs_t) * self.num_inters[:self.n_users]


        for _ in range(self.n_common_layer):
            # Uncensored edge
            # common_item_embs_v = torch.sparse.mm(self.popularity_adj_ii_norm, common_item_embs_v)
            # common_item_embs_t = torch.sparse.mm(self.popularity_adj_ii_norm, common_item_embs_t)
            # common_user_embs_v = torch.sparse.mm(self.popularity_adj_uu_norm, self.user_embedding.weight)
            # common_user_embs_t = torch.sparse.mm(self.popularity_adj_uu_norm, self.user_embedding.weight)

            common_item_embs_v = torch.sparse.mm(self.common_ii, common_item_embs_v)
            common_item_embs_t = torch.sparse.mm(self.common_ii, common_item_embs_t)
            common_user_embs_v = torch.sparse.mm(self.common_uu, self.user_embedding.weight)
            common_user_embs_t = torch.sparse.mm(self.common_uu, self.user_embedding.weight)

        common_embs_v = torch.cat([common_user_embs_v, F.normalize(common_item_embs_v)], dim=0)
        common_embs_t = torch.cat([common_user_embs_t, F.normalize(common_item_embs_t)], dim=0)

        common_embs = self.alpha_3 * F.normalize(common_embs_v) + self.alpha_4 * F.normalize(common_embs_t)
        all_embs = id_embs + personal_embs + 0.5*common_embs

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


