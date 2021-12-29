import torch
import torch.nn as nn
from src.aggregator import *
from manifold.hyperboloid import Hyperboloid
import numpy as np

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class LKGR(torch.nn.Module):
    def __init__(self, args, n_user, n_item, n_entity, n_relation):
        super(LKGR, self).__init__()
        self.min_norm = 1e-6
        self.max_norm = 1e6
        self.n_user = n_user
        self.n_item = n_item
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.n_layer = args.n_layer
        self.batch_size = args.batch_size
        self.node_dim = args.node_dim + 1

        self.sample_size = args.sample_size

        self.agg_type = args.agg_type
        self.l2_weight = args.l2_weight
        self.dropout = args.dropout


        self.set_curvature(args)

        self.user_embedding = torch.nn.Embedding(self.n_user, self.node_dim)
        self.entity_embedding = torch.nn.Embedding(self.n_entity, self.node_dim)
        self.W_R = nn.Parameter(torch.FloatTensor(self.n_relation + 1, self.node_dim, self.node_dim),
                                requires_grad=True)

        nn.init.xavier_uniform_(self.user_embedding.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.entity_embedding.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))

        self.manifold = Hyperboloid()

        self.user_aggregator = UserAggregator(self.batch_size, self.sample_size, self.node_dim, self.agg_type,
                                              self.dropout, self.manifold, self.c)
        self.kg_aggregator = KGAggregator(self.batch_size, self.sample_size, self.node_dim,
                                          self.agg_type, self.dropout, self.manifold, self.c)

    def set_curvature(self, args):
        self.c = nn.Parameter(torch.tensor([args.curvature], dtype=torch.float), requires_grad=True)

    def clamp_curvature(self, threshold):
        self.c.data = torch.clamp(self.c.data, min=threshold)


    def set_adj_matrix(self, adj_u2i, adj_i2u, adj_entity, adj_relation):
        """
        set the sample neighbor adjacency matrix
        """
        self.adj_u2i = adj_u2i
        self.adj_i2u = adj_i2u
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation

    def _get_user_ngh(self, item_index):
        """
        :param  item_index:         [batch_size]
        :return:                    [batch_size, sample_size]
        """
        return self.adj_i2u[item_index]

    def _get_item_ngh(self, user_index):
        """
        :param  user_index:         [batch_size]
        :return:                    [batch_size, sample_size]
        """
        return self.adj_u2i[user_index]

    def _get_entity_ngh(self, item_index):
        """
        :param item_index:              [batch_size]
        :return:            entity:    {[batch_size, 1], [batch_size, sample_size], [batch_size, sample_size^2], ..., [batch_size, sample_size^n_layer]}
                            relation:  {[batch_size, sample_size], [batch_size, sample_size^2], ..., [batch_size, sample_size^n_layer]}
        """
        item_index = item_index.view((-1, 1))
        entities = [item_index]
        relations = []

        for h in range(self.n_layer):
            neighbor_entities = self.adj_entity[entities[h]].view((self.batch_size, -1))
            neighbor_relations = self.adj_relation[entities[h]].view((self.batch_size, -1))

            entities.append(neighbor_entities)
            relations.append(neighbor_relations)

        return entities, relations

    def _aggregate_for_user(self, user_index):
        """
        :param  user_index:             [batch_size]
        :return:                        [batch_size, n_dim]
        """
        item_ngh_index = self._get_item_ngh(user_index)
        user_vectors = self.user_embedding(user_index)
        user_vectors = self.manifold.proj_tan0(user_vectors, self.c)

        user_vectors = self.manifold.expmap0(user_vectors, self.c)
        item_ngh_vectors = self.entity_embedding(item_ngh_index)
        item_ngh_vectors = self.manifold.proj_tan0(item_ngh_vectors, self.c)
        item_ngh_vectors = self.manifold.expmap0(item_ngh_vectors, self.c)

        W_ui = self.W_R[self.n_relation]

        output = self.user_aggregator(user_vectors, item_ngh_vectors, W_ui, act_f=torch.tanh)
        return output

    def _aggregate_for_item(self, item_embeddings, user_ngh, entities, relations):
        """
        Make item embeddings by aggregating neighbor vectors
        :param user_ngh:                [batch_size, sample_size]
        :param user_embeddings:         [batch_size, dim]
        :param item_embeddings:         [batch_size, dim]
        :param entities:                {[batch_size, 1], [batch_size, sample_size], [batch_size, sample_size^2], ..., [batch_size, sample_size^n_layer]}
        :param relations:               {[batch_size, sample_size], [batch_size, sample_size^2], ..., [batch_size, sample_size^n_layer]}
        :return:                        [batch_size, dim]
        """
        item_embeddings = self.manifold.proj_tan0(item_embeddings, self.c)
        item_embeddings = self.manifold.expmap0(item_embeddings, self.c)

        # [batch_size, sample_size, dim]
        user_ngh_vector = self.user_embedding(user_ngh)
        user_ngh_vector = self.manifold.proj_tan0(user_ngh_vector, self.c)
        user_ngh_vector = self.manifold.expmap0(user_ngh_vector, self.c)

        entity_vectors = []

        for entity in entities:
            entity_embedding = self.entity_embedding(entity)
            entity_embedding = self.manifold.proj_tan0(entity_embedding, self.c)
            entity_embedding = self.manifold.expmap0(entity_embedding, self.c)
            entity_vectors.append(entity_embedding)

        is_item_layer = False
        for i in range(self.n_layer):
            if i == self.n_layer - 1:
                act_f = torch.tanh
                is_item_layer = True
            else:
                act_f = torch.relu

            entity_vectors_next_iter = []
            for hop in range(self.n_layer - i):
                tmp_index = relations[hop].view(self.batch_size, -1, self.sample_size)
                # [batch_size, -1, sample_size, dim, dim]
                Wr = self.W_R[tmp_index]
                W_ui = self.W_R[self.n_relation]
                para = [Wr, W_ui]

                vector = self.kg_aggregator(
                    self_vectors=entity_vectors[hop],
                    ngh_user_vectors=user_ngh_vector,
                    ngh_entity_vectors=entity_vectors[hop + 1].view(
                        (self.batch_size, -1, self.sample_size, self.node_dim)),
                    item_embeddings=item_embeddings,
                    para=para,
                    act_f=act_f,
                    is_item_layer=is_item_layer)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        return entity_vectors[0].view((self.batch_size, self.node_dim))

    def _get_score(self, user_index, item_index):
        """
        :param user_index:                  [batch_size]
        :param item_index:                  [batch_size]
        :return:
        """
        batch_size = item_index.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size

        # [batch_size, dim]
        new_user_embeddings = self._aggregate_for_user(user_index)
        user_ngh = self._get_user_ngh(item_index)
        entities, relations = self._get_entity_ngh(item_index)
        item_embeddings = self.entity_embedding(item_index)

        new_item_embeddings = self._aggregate_for_item(item_embeddings, user_ngh, entities, relations)

        user_embeddings = self.manifold.logmap0(new_user_embeddings, self.c)
        item_embeddings = self.manifold.logmap0(new_item_embeddings, self.c)
        # [batch_size, dim] * [batch_size, dim] -> [batch_size]
        score = torch.sum(user_embeddings * item_embeddings, dim=-1)
        score = torch.sigmoid(score)
        score = torch.clamp(score, min=self.min_norm, max=self.max_norm)

        # to avoid nan
        score[torch.isnan(score)] = 0
        return score

    def _batch_score(self, user_index, item_index):
        self.batch_size = user_index.shape[0]
        new_all_user_embedding = self._aggregate_for_user(user_index)

        self.batch_size = item_index.shape[0]
        user_ngh = self._get_user_ngh(item_index)
        entities, relations = self._get_entity_ngh(item_index)
        all_item_embedding = self.entity_embedding(item_index)

        new_all_item_embeddings = self._aggregate_for_item(all_item_embedding, user_ngh, entities, relations)

        new_all_user_embedding = self.manifold.logmap0(new_all_user_embedding, self.c)

        score = torch.mm(new_all_user_embedding, new_all_item_embeddings.t())
        all_score = torch.sigmoid(score)
        all_score = torch.clamp(all_score, min=self.min_norm, max=self.max_norm)

        # to avoid nan
        all_score[torch.isnan(all_score)] = 0
        return all_score

    def _cal_loss_CE(self, user_index, item_index, labels):
        # using cross entropy loss with negative sampling
        score = self._get_score(user_index, item_index)

        criterion = torch.nn.BCELoss()
        base_loss = criterion(score, labels)

        l2_loss = _L2_loss_mean(self.user_embedding(user_index)) + _L2_loss_mean(self.entity_embedding(item_index))
        l2_loss += _L2_loss_mean(self.user_aggregator.linear.get_weight())
        for linear in self.kg_aggregator.get_linear():
            l2_loss += _L2_loss_mean(linear.get_weight())

        base_loss += self.l2_weight * l2_loss
        return base_loss

    def forward(self, mode, *input):
        if mode == 'cal_loss':
            return self._cal_loss_CE(*input)
        elif mode == 'get_score':
            return self._get_score(*input)
        elif mode == 'batch_score':
            return self._batch_score(*input)
        else:
            raise NotImplementedError






