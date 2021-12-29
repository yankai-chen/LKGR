import torch
import torch.nn as nn
import torch.nn.functional as F
from manifold.hyperboloid import Hyp_Linear

class UserAggregator(torch.nn.Module):
    """
    Aggregator class
    Mode in ['sum', 'concat', 'neighbor']
    """
    def __init__(self, batch_size, sample_size, dim, agg_type, dropout, manifold, curvature):
        super(UserAggregator, self).__init__()
        self.batch_size = batch_size
        self.dim = dim
        self.agg_type = agg_type
        self.dropout = dropout
        self.c = curvature
        self.sample_size = sample_size
        self.manifold = manifold

        if self.agg_type == 'concat':
            self.linear = Hyp_Linear(2 * self.dim, self.dim, self.c, self.dropout, use_bias=True)
        elif self.agg_type in ['sum', 'ngh']:
            self.linear = Hyp_Linear(self.dim, self.dim, self.c, self.dropout, use_bias=True)
        else:
            raise NotImplementedError

    def compute_att(self, self_embeddings, item_ngh_embedding, W_ui):
        """
        :param self_embeddings:             [batch_size, dim]
        :param item_ngh_embedding:          [batch_size, sample_size, dim]
        :param W_ui:                        [dim, dim]
        :return:                            [batch_size, sample_size, 1]
        """
        # [1， dim, dim]
        W_ui = W_ui.expand(1, self.dim, self.dim)
        # [batch_size, 1, dim]
        self_embeddings = self.manifold.logmap0(self_embeddings, self.c).view(self.batch_size, 1, self.dim)
        item_ngh_embedding = self.manifold.logmap0(item_ngh_embedding, self.c)

        # [batch_size, 1, dim] * [1, dim, dim] -> [batch_size, dim] -> [batch_size, 1, dim]
        user_mut_W_ui = torch.sum(self_embeddings * W_ui, dim=-1).view(self.batch_size, 1, self.dim)

        # [batch_size, 1, dim] * [batch_size, sample_size, dim] -> [batch_size, sample_size]
        att = torch.mean(user_mut_W_ui * item_ngh_embedding, dim=-1)

        att_norm = F.softmax(att, dim=-1)
        # [batch_size, sample_size, 1]
        att_norm = att_norm.view(self.batch_size, -1, 1)
        return att_norm

    def aggregate_item_ngh(self, self_vectors, item_ngh_vectors, W_ui):
        """
        :param self_vectors:                [batch_size, dim]
        :param item_ngh_vectors:            [batch_size, sample_size, dim]
        :param W_ui:                        [dim, dim]
        :return:                            [batch_size, dim]
        """
        # [batch_size, sample_size, 1]
        att_norm = self.compute_att(self_vectors, item_ngh_vectors, W_ui)

        # [batch_size, dim] -> [batch_size, sample_size, dim]
        tmp_self_vectors = self_vectors.unsqueeze(dim=1).expand(self.batch_size, self.sample_size, self.dim)
        item_ngh_vectors = self.manifold.logmap(tmp_self_vectors, item_ngh_vectors, self.c)
        # [batch_size, sample_size, 1] * [batch_size, sample_size, dim] -> [batch_size, 1, dim]
        ngh_aggregated = torch.sum(att_norm * item_ngh_vectors, dim=1)
        ngh_aggregated = self.manifold.expmap(ngh_aggregated, self_vectors, self.c)

        return ngh_aggregated

    def forward(self, self_vectors, item_ngh_vectors, W_ui, act_f):
        """
        :param self_vectors:                [batch_size, dim]
        :param item_ngh_vectors:            [batch_size, sample_size, dim]
        :return:                            [batch_size, dim]
        """
        batch_size = self_vectors.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size

        # [batch_size, dim] * [batch_size, sample_size, dim] -> [batch_size, dim]
        ngh_agg = self.aggregate_item_ngh(self_vectors, item_ngh_vectors, W_ui)
        if self.agg_type == 'sum':
            # [batch_size, dim]
            output = self.manifold.add(self_vectors, ngh_agg, self.c)

        elif self.agg_type == 'concat':
            # [batch_size, 2*dim]
            self_vectors = self.manifold.logmap0(self_vectors, self.c)
            ngh_agg = self.manifold.logmap0(ngh_agg, self.c)
            output = torch.cat((self_vectors, ngh_agg), dim=-1)
            output = self.manifold.expmap0(output, self.c)

        elif self.agg_type == 'ngh':
            # [batch_size, dim]
            output = ngh_agg

        else:
            raise NotImplementedError
        output = self.linear(output)
        # [batch_size, dim]
        output = self.manifold.logmap0(output, self.c)
        output = act_f(output)
        output = self.manifold.expmap0(output, self.c)
        return output


class KGAggregator(torch.nn.Module):
    """
    Aggregator class
    Mode in ['sum', 'concat', 'neighbor']
    """
    def __init__(self, batch_size, sample_size, dim, agg_type, dropout, manifold, curvature):
        super(KGAggregator, self).__init__()
        self.batch_size = batch_size
        self.dim = dim
        self.agg_type = agg_type
        self.dropout = dropout
        self.c = curvature
        self.sample_size = sample_size
        self.manifold = manifold

        if self.agg_type == 'concat':
            self.linear1 = Hyp_Linear(2 * self.dim, self.dim, self.c, self.dropout, use_bias=True)
            self.linear2 = Hyp_Linear(3 * self.dim, self.dim, self.c, self.dropout, use_bias=True)
        elif self.agg_type in ['sum', 'ngh']:
            self.linear = Hyp_Linear(self.dim, self.dim, self.c, self.dropout, use_bias=True)
        else:
            raise NotImplementedError

    def compute_att_user_side(self, self_embeddings, user_ngh_embeddings, W_ui):
        """"
        :param self_embeddings:             [batch_size, dim]
        :param user_ngh_embeddings:         [batch_size, sample_size, dim]
        :param W_ui:                        [dim, dim]
        :return:                            [batch_size, sample_size, 1]
        """

        self_embeddings = self.manifold.logmap0(self_embeddings, self.c)
        user_ngh_embeddings = self.manifold.logmap0(user_ngh_embeddings, self.c)
        # [batch_size, dim] * [dim, dim] -> [batch_size, dim] -> [batch_size, 1, dim]
        item_mut_W_ui = torch.sum(self_embeddings * W_ui, dim=-1).view(self.batch_size, 1, self.dim)

        # [batch_size, 1, dim] * [batch_size, sample_size, dim] -> [batch_size, sample_size]
        att = torch.sum(item_mut_W_ui * user_ngh_embeddings, dim=-1)
        att_norm = F.softmax(att, dim=-1)

        # [batch_size, sample_size, 1]
        att_norm = att_norm.view(self.batch_size, -1, 1)
        return att_norm

    def compute_att_entity_side(self, self_embeddings, entity_ngh_embeddings, Wr):
        """
        :param self_embeddings:             [batch_size, -1, dim]
        :param entity_ngh_embeddings:       [batch_size, -1, sample_size, dim]
        :param Wr:                          [batch_size, -1, sample_size, dim, dim]
        :return:                            [batch_size, -1, sample_size]
        """
        # [batch_size, -1, dim] -> [batch_size, -1, sample_size, 1, dim]
        self_embeddings = self_embeddings.view(self.batch_size, -1, 1, 1, self.dim)
        self_embeddings = self_embeddings.expand(self.batch_size, -1, self.sample_size, 1, self.dim)
        self_embeddings = self.manifold.logmap0(self_embeddings, self.c)

        # [batch_size, -1, sample_size, dim]
        entity_ngh_embeddings = self.manifold.logmap0(entity_ngh_embeddings, self.c)

        # [batch_size, -1, sample_size, 1, dim] * [batch_size, -1, sample_size, dim, dim]
        # -> [batch_size, -1, sample_size, dim]
        self_mut_Wr = torch.sum(self_embeddings * Wr, dim=-1)

        # [batch_size, -1, sample_size, dim] * [batch_size, -1, sample_size, dim]
        # -> [batch_size, -1, sample_size]
        att = torch.sum(self_mut_Wr * entity_ngh_embeddings, dim=-1)

        att_norm = F.softmax(att, dim=-1)
        # [batch_size, -1, sample_size]
        return att_norm

    def aggregate_user_ngh(self, self_vectors, user_ngh_embeddings, W_ui):
        """"
        :param self_vectors:                [batch_size, dim]
        :param user_ngh_embeddings:         [batch_size, sample_size, dim]
        :param W_ui:                        [dim, dim]
        :return:                            [batch_size, dim]
        """
        # [batch_size, dim] -> [batch_size, 1, dim]
        self_vectors = self_vectors.view(self.batch_size, 1, self.dim)
        # [batch_size, sample_size, 1]
        att_norm = self.compute_att_user_side(self_vectors, user_ngh_embeddings, W_ui)

        # [batch_size, dim] -> [batch_size, sample_size, dim]
        tmp_self_vectors = self_vectors.expand(-1, self.sample_size, -1)
        item_ngh_vectors = self.manifold.logmap(tmp_self_vectors, user_ngh_embeddings, self.c)
        # [batch_size, sample_size, 1] * [batch_size, sample_size, dim] -> [batch_size, 1, dim]
        ngh_aggregated = torch.sum(att_norm * item_ngh_vectors, dim=1, keepdim=True)
        ngh_aggregated = self.manifold.expmap(ngh_aggregated, self_vectors, self.c)
        # [batch_size, dim]
        ngh_aggregated = ngh_aggregated.squeeze(dim=1)

        return ngh_aggregated

    def aggregate_entity_ngh(self, self_vectors, entity_ngh_vectors, Wr):
        """
        This aims to aggregate neighbor vectors
        :param self_vectors:                    [batch_size, -1, dim]
        :param entity_ngh_vectors:              [batch_size, -1, sample_size, dim]
        :param Wr:                              [batch_size, -1, sample_size, dim, dim]
        :return:                                [batch_size, -1, dim]
        """
        # [batch_size, -1, sample_size, dim] -> [batch_size, -1, sample_size]
        att_norm = self.compute_att_entity_side(self_vectors, entity_ngh_vectors, Wr)
        # [batch_size, -1, sample_size] -> [batch_size, -1, sample_size, 1]
        att_norm = att_norm.unsqueeze(dim=-1)
        # [batch_size, -1, dim] - > [batch_size, -1, 1, dim]
        self_vectors = self_vectors.unsqueeze(dim=2)

        # [batch_size, -1, 1, dim] - > [batch_size, -1, sample_size, dim]
        tmp_self_vectors = self_vectors.expand(-1, -1, self.sample_size, -1)
        ngh_vectors = self.manifold.logmap(tmp_self_vectors, entity_ngh_vectors, self.c)

        # [batch_size, -1, sample_size, 1] * [batch_size, -1, sample_size, dim] -> [batch_size, -1, 1, dim]
        neighbors_aggregated = torch.sum(att_norm * ngh_vectors, dim=2, keepdim=True)

        neighbors_aggregated = self.manifold.expmap(neighbors_aggregated, self_vectors, self.c)
        neighbors_aggregated = neighbors_aggregated.squeeze(dim=2)

        return neighbors_aggregated

    def forward(self, self_vectors, ngh_user_vectors, ngh_entity_vectors,
                item_embeddings, para, act_f, is_item_layer):
        """
        :param self_vectors:                [batch_size, -1, dim]
        :param ngh_user_vectors:            [batch_size, sample_size, dim]
        :param ngh_entity_vectors:          [batch_size, -1, sample_size, dim]
        :param user_embeddings:             [batch_size, dim]
        :param item_embeddings:             [batch_size, dim]
        :param Wr:                          [batch_size, -1, sample_size, dim, dim]
        :return:                            [batch_size, -1, dim]
        """
        Wr, W_ui = para

        batch_size = item_embeddings.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size

        if not is_item_layer:
            '''
            If it's not the layer for items, we only aggregate the entity neighbors
            '''
            # [batch_size, -1, dim]
            kg_ngh_agg = self.aggregate_entity_ngh(self_vectors, ngh_entity_vectors, Wr)

            if self.agg_type == 'sum':
                # [batch_size*-1, dim]
                output = self.manifold.add(self_vectors, kg_ngh_agg, self.c).view(-1, self.dim)
                output = self.linear(output).view((self.batch_size, -1, self.dim))

            elif self.agg_type == 'concat':
                # [batch_size, -1, 2*dim]
                self_vectors = self.manifold.logmap0(self_vectors, self.c)
                kg_ngh_agg = self.manifold.logmap0(kg_ngh_agg, self.c)
                output = torch.cat((self_vectors, kg_ngh_agg), dim=-1)
                output = self.manifold.expmap0(output, self.c)

                # [batch_size*-1, 2*dim]
                output = output.view((-1, 2 * self.dim))
                output = self.linear1(output).view((self.batch_size, -1, self.dim))

            elif self.agg_type == 'ngh':
                # [batch_size*-1, dim]
                output = kg_ngh_agg.view((-1, self.dim))
                output = self.linear(output).view((self.batch_size, -1, self.dim))

            else:
                raise NotImplementedError

            # [batch_size, -1, dim]
            output = self.manifold.logmap0(output, self.c)
            output = act_f(output)
            output = self.manifold.expmap0(output, self.c)
            return output

        else:
            '''
            If it is the layer for items, we aggregate the user part and entity part together
            '''
            # [batch_size, -1, dim]
            kg_ngh_agg = self.aggregate_entity_ngh(self_vectors, ngh_entity_vectors, Wr)
            _, size, _ = kg_ngh_agg.shape

            # [batch_size, dim]
            user_ngh_agg = self.aggregate_user_ngh(item_embeddings, ngh_user_vectors, W_ui)
            # [batch_size, -1, dim]
            user_ngh_agg = user_ngh_agg.view(self.batch_size, size, self.dim)

            if self.agg_type == 'sum':
                # [batch_size*-1, dim]
                output = self.manifold.add(self_vectors, kg_ngh_agg, self.c)
                output = self.manifold.add(output, user_ngh_agg, self.c).view(-1, self.dim)
                # [batch_size, -1, dim]
                output = self.linear(output).view((self.batch_size, -1, self.dim))

            elif self.agg_type == 'concat':
                # [batch_size, -1, 3*dim]
                self_vectors = self.manifold.logmap0(self_vectors, self.c)
                kg_ngh_agg = self.manifold.logmap0(kg_ngh_agg, self.c)
                user_ngh_agg = self.manifold.logmap0(user_ngh_agg, self.c)
                output = torch.cat((self_vectors, kg_ngh_agg, user_ngh_agg), dim=-1)
                output = self.manifold.expmap0(output, self.c)

                # [batch_size*-1, 3*dim]
                output = output.view((-1, 3 * self.dim))
                # [batch_size, -1, 3*dim]
                output = self.linear2(output).view((self.batch_size, -1, self.dim))

            elif self.agg_type == 'ngh':
                # [batch_size*-1, dim]
                output = self.manifold.add(user_ngh_agg, kg_ngh_agg, self.c)
                output = output.view((-1, self.dim))

                # [batch_size, -1, dim]
                output = self.linear(output).view((self.batch_size, -1, self.dim))

            else:
                raise NotImplementedError

            # [batch_size, -1, dim]
            output = self.manifold.logmap0(output, self.c)
            output = act_f(output)
            output = self.manifold.expmap0(output, self.c)
            return output

    def get_linear(self):
        if self.agg_type == 'concat':
            return [self.linear1, self.linear2]

        elif self.agg_type in ['sum', 'ngh']:
            return [self.linear]