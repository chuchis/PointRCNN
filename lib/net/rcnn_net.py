import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_lib.pointnet2.pointnet2_modules import PointnetSAModule
from lib.rpn.proposal_target_layer import ProposalTargetLayer
import pointnet2_lib.pointnet2.pytorch_utils as pt_utils
import lib.utils.loss_utils as loss_utils
from lib.config import cfg
import numpy as np
# from PIL import Image
# import png

import lib.utils.kitti_utils as kitti_utils
import lib.utils.roipool3d.roipool3d_utils as roipool3d_utils
from gcn_lib.dense import BasicConv, GraphConv2d, ResDynBlock2d, DenseDynBlock2d, DenseDilatedKnnGraph, ResBlock2d
from torch.nn import Sequential as Seq


class DenseDeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DenseDeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.kernel_size
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.stochastic
        conv = opt.conv
        c_growth = channels
        self.n_blocks = opt.n_blocks
        self.head_xyz=opt.head

        self.knn = DenseDilatedKnnGraph(k, 1, stochastic, epsilon)
        self.head = GraphConv2d(opt.in_channels, channels, conv, act, norm, bias)
        if opt.constant_dilation:
            self.dilation = lambda x: 1
        else:
            if opt.linear_dilation:
                self.dilation = lambda x: x+1
            else:
                self.dilation = lambda x: (x%4)+1
        if opt.block.lower() == 'res':
            self.backbone = Seq(*[ResDynBlock2d(channels, k, self.dilation(i), conv, act, norm, bias, stochastic, epsilon)
                                  for i in range(self.n_blocks-1)])
        elif opt.block.lower() == 'dense':
            self.backbone = Seq(*[DenseDynBlock2d(channels+c_growth*i, c_growth, k, self.dilation(i), conv, act,
                                                  norm, bias, stochastic, epsilon)
                                  for i in range(self.n_blocks-1)])
        elif opt.block.lower() == 'res_fixed':
            self.backbone = Seq(*[ResBlock2d(channels, k, self.dilation(i), conv, act, norm, bias, stochastic, epsilon)
                                  for i in range(self.n_blocks-1)])
        else:
            raise NotImplementedError('{} is not implemented. Please check.\n'.format(opt.block))
        self.block = opt.block.lower()
        self.fusion_block = BasicConv([channels+c_growth*(self.n_blocks-1), 1024], act, norm, bias)
        self.channel_out = 1024+channels+c_growth*(self.n_blocks-1)

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        # print(inputs.shape)
        #(B,C,N,1)
        if self.head_xyz:
            knn_input = inputs[:, 0:3]
        else:
            knn_input = inputs
        feats = [self.head(inputs, self.knn(knn_input))]
        # print(inputs.shape)
        for i in range(self.n_blocks-1):
            # print(feats[-1].shape)
            # print(i)
            if self.block == 'res_fixed':
                feats.append(self.backbone[i](feats[-1], knn_input))
            else:
                feats.append(self.backbone[i](feats[-1]))
        feats = torch.cat(feats, dim=1)

        fusion = torch.max_pool2d(self.fusion_block(feats), kernel_size=[feats.shape[2], feats.shape[3]])
        fusion = torch.repeat_interleave(fusion, repeats=feats.shape[2], dim=2)
        return torch.cat((fusion, feats), dim=1)

class DenseOpts():
    def __init__(self):
        self.n_filters = cfg.RCNN.DEEPGCN_CONFIG.N_FILTERS
        self.kernel_size = cfg.RCNN.DEEPGCN_CONFIG.KERNEL_SIZE
        self.act = 'relu'
        self.norm = 'batch'
        self.bias = True
        self.epsilon = 0.2
        self.stochastic = True
        self.conv = cfg.RCNN.DEEPGCN_CONFIG.CONV # edge, mr
        self.n_blocks = cfg.RCNN.DEEPGCN_CONFIG.N_BLOCKS
        self.in_channels = 3
        self.block = cfg.RCNN.DEEPGCN_CONFIG.BLOCK
        self.head = cfg.RCNN.DEEPGCN_CONFIG.HEAD
        self.constant_dilation = cfg.RCNN.DEEPGCN_CONFIG.CONSTANT_DILATION
        self.linear_dilation = cfg.RCNN.DEEPGCN_CONFIG.LINEAR_DILATION

class DenseRCNN(nn.Module):
    def __init__(self, num_classes, input_channels=0, use_xyz=True):
        super().__init__()
        opt = DenseOpts()
        self.backbone = DenseDeepGCN(opt)

        # classification layer
        cls_channel = 1 if num_classes == 2 else num_classes
        cls_layers = []
        channel_in = self.backbone.channel_out
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.CLS_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, cls_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            cls_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.cls_layer = nn.Sequential(*cls_layers)

        if cfg.RCNN.LOSS_CLS == 'SigmoidFocalLoss':
            self.cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=cfg.RCNN.FOCAL_ALPHA[0],
                                                                           gamma=cfg.RCNN.FOCAL_GAMMA)
        elif cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':
            self.cls_loss_func = F.binary_cross_entropy
        elif cfg.RCNN.LOSS_CLS == 'CrossEntropy':
            cls_weight = torch.from_numpy(cfg.RCNN.CLS_WEIGHT).float()
            self.cls_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduce=False, weight=cls_weight)
        else:
            raise NotImplementedError

        # regression layer
        per_loc_bin_num = int(cfg.RCNN.LOC_SCOPE / cfg.RCNN.LOC_BIN_SIZE) * 2
        loc_y_bin_num = int(cfg.RCNN.LOC_Y_SCOPE / cfg.RCNN.LOC_Y_BIN_SIZE) * 2
        reg_channel = per_loc_bin_num * 4 + cfg.RCNN.NUM_HEAD_BIN * 2 + 3
        reg_channel += (1 if not cfg.RCNN.LOC_Y_BY_BIN else loc_y_bin_num * 2)

        reg_layers = []
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.REG_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.REG_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, reg_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.reg_layer = nn.Sequential(*reg_layers)

        self.proposal_target_layer = ProposalTargetLayer()
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layer[-1].conv.weight, mean=0, std=0.001)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, input_data):
        """
        :param input_data: input dict
        :return:
        """
        if cfg.RCNN.ROI_SAMPLE_JIT:
            if self.training:
                with torch.no_grad():
                    target_dict = self.proposal_target_layer(input_data)

                pts_input = torch.cat((target_dict['sampled_pts'], target_dict['pts_feature']), dim=2)
                target_dict['pts_input'] = pts_input
            else:
                rpn_xyz, rpn_features = input_data['rpn_xyz'], input_data['rpn_features']
                batch_rois = input_data['roi_boxes3d']
                if cfg.RCNN.USE_INTENSITY:
                    pts_extra_input_list = [input_data['rpn_intensity'].unsqueeze(dim=2),
                                            input_data['seg_mask'].unsqueeze(dim=2)]
                else:
                    pts_extra_input_list = [input_data['seg_mask'].unsqueeze(dim=2)]

                if cfg.RCNN.USE_DEPTH:
                    pts_depth = input_data['pts_depth'] / 70.0 - 0.5
                    pts_extra_input_list.append(pts_depth.unsqueeze(dim=2))
                pts_extra_input = torch.cat(pts_extra_input_list, dim=2)

                pts_feature = torch.cat((pts_extra_input, rpn_features), dim=2)
                pooled_features, pooled_empty_flag = \
                        roipool3d_utils.roipool3d_gpu(rpn_xyz, pts_feature, batch_rois, cfg.RCNN.POOL_EXTRA_WIDTH,
                                                      sampled_pt_num=cfg.RCNN.NUM_POINTS)

                # canonical transformation
                batch_size = batch_rois.shape[0]
                roi_center = batch_rois[:, :, 0:3]
                pooled_features[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2)
                for k in range(batch_size):
                    pooled_features[k, :, :, 0:3] = kitti_utils.rotate_pc_along_y_torch(pooled_features[k, :, :, 0:3],
                                                                                        batch_rois[k, :, 6])

                pts_input = pooled_features.view(-1, pooled_features.shape[2], pooled_features.shape[3])
        else:
            pts_input = input_data['pts_input'].view(-1,512,133)
            target_dict = {}
            target_dict['pts_input'] = input_data['pts_input'].view(-1,512,133)
            target_dict['roi_boxes3d'] = input_data['roi_boxes3d'].view(-1,7)
            if self.training:
                target_dict['cls_label'] = input_data['cls_label'].view(-1)
                target_dict['reg_valid_mask'] = input_data['reg_valid_mask'].view(-1)
                target_dict['gt_of_rois'] = input_data['gt_boxes3d_ct'].view(-1,7)

        xyz, features = self._break_up_pc(pts_input)
        # print(xyz)
        # print(xyz.shape)
        pt_features = self.backbone(xyz.transpose(1,2).contiguous().unsqueeze(3))
        features = torch.max(pt_features, dim=2)[0]
        # print(features.shape)

        rcnn_cls = self.cls_layer(features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layer(features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)
        ret_dict = {'rcnn_cls': rcnn_cls, 'rcnn_reg': rcnn_reg}

        if self.training:
            ret_dict.update(target_dict)
        return ret_dict


class DenseFeatRCNN(nn.Module):
    def __init__(self, num_classes, input_channels=0, use_xyz=True):
        super().__init__()

        if cfg.RCNN.USE_RPN_FEATURES:
            self.rcnn_input_channel = 3 + int(cfg.RCNN.USE_INTENSITY) + int(cfg.RCNN.USE_MASK) + int(cfg.RCNN.USE_DEPTH)
            self.xyz_up_layer = pt_utils.SharedMLP([self.rcnn_input_channel] + cfg.RCNN.XYZ_UP_LAYER,
                                                   bn=cfg.RCNN.USE_BN)
            c_out = cfg.RCNN.XYZ_UP_LAYER[-1]
            self.merge_down_layer = pt_utils.SharedMLP([c_out * 2, c_out], bn=cfg.RCNN.USE_BN)

        opt = DenseOpts()
        opt.in_channels = input_channels
        self.backbone = DenseDeepGCN(opt)
        # classification layer
        cls_channel = 1 if num_classes == 2 else num_classes
        cls_layers = []
        channel_in = self.backbone.channel_out
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.CLS_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, cls_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            cls_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.cls_layer = nn.Sequential(*cls_layers)

        if cfg.RCNN.LOSS_CLS == 'SigmoidFocalLoss':
            self.cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=cfg.RCNN.FOCAL_ALPHA[0],
                                                                           gamma=cfg.RCNN.FOCAL_GAMMA)
        elif cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':
            self.cls_loss_func = F.binary_cross_entropy
        elif cfg.RCNN.LOSS_CLS == 'CrossEntropy':
            cls_weight = torch.from_numpy(cfg.RCNN.CLS_WEIGHT).float()
            self.cls_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduce=False, weight=cls_weight)
        else:
            raise NotImplementedError

        # regression layer
        per_loc_bin_num = int(cfg.RCNN.LOC_SCOPE / cfg.RCNN.LOC_BIN_SIZE) * 2
        loc_y_bin_num = int(cfg.RCNN.LOC_Y_SCOPE / cfg.RCNN.LOC_Y_BIN_SIZE) * 2
        reg_channel = per_loc_bin_num * 4 + cfg.RCNN.NUM_HEAD_BIN * 2 + 3
        reg_channel += (1 if not cfg.RCNN.LOC_Y_BY_BIN else loc_y_bin_num * 2)

        reg_layers = []
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.REG_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.REG_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, reg_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.reg_layer = nn.Sequential(*reg_layers)

        self.proposal_target_layer = ProposalTargetLayer()
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layer[-1].conv.weight, mean=0, std=0.001)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, input_data):
        """
        :param input_data: input dict
        :return:
        """
        if cfg.RCNN.ROI_SAMPLE_JIT:
            if self.training:
                with torch.no_grad():
                    target_dict = self.proposal_target_layer(input_data)

                pts_input = torch.cat((target_dict['sampled_pts'], target_dict['pts_feature']), dim=2)
                target_dict['pts_input'] = pts_input
            else:
                rpn_xyz, rpn_features = input_data['rpn_xyz'], input_data['rpn_features']
                batch_rois = input_data['roi_boxes3d']
                if cfg.RCNN.USE_INTENSITY:
                    pts_extra_input_list = [input_data['rpn_intensity'].unsqueeze(dim=2),
                                            input_data['seg_mask'].unsqueeze(dim=2)]
                else:
                    pts_extra_input_list = [input_data['seg_mask'].unsqueeze(dim=2)]

                if cfg.RCNN.USE_DEPTH:
                    pts_depth = input_data['pts_depth'] / 70.0 - 0.5
                    pts_extra_input_list.append(pts_depth.unsqueeze(dim=2))
                pts_extra_input = torch.cat(pts_extra_input_list, dim=2)

                pts_feature = torch.cat((pts_extra_input, rpn_features), dim=2)
                pooled_features, pooled_empty_flag = \
                        roipool3d_utils.roipool3d_gpu(rpn_xyz, pts_feature, batch_rois, cfg.RCNN.POOL_EXTRA_WIDTH,
                                                      sampled_pt_num=cfg.RCNN.NUM_POINTS)

                # canonical transformation
                batch_size = batch_rois.shape[0]
                roi_center = batch_rois[:, :, 0:3]
                pooled_features[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2)
                for k in range(batch_size):
                    pooled_features[k, :, :, 0:3] = kitti_utils.rotate_pc_along_y_torch(pooled_features[k, :, :, 0:3],
                                                                                        batch_rois[k, :, 6])

                pts_input = pooled_features.view(-1, pooled_features.shape[2], pooled_features.shape[3])
        else:
            pts_input = input_data['pts_input'].view(-1,512,133)
            target_dict = {}
            target_dict['pts_input'] = input_data['pts_input'].view(-1,512,133)
            target_dict['roi_boxes3d'] = input_data['roi_boxes3d'].view(-1,7)
            if self.training:
                target_dict['cls_label'] = input_data['cls_label'].view(-1)
                target_dict['reg_valid_mask'] = input_data['reg_valid_mask'].view(-1)
                target_dict['gt_of_rois'] = input_data['gt_boxes3d_ct'].view(-1,7)

        xyz, features = self._break_up_pc(pts_input)

        if cfg.RCNN.USE_RPN_FEATURES:
            xyz_input = pts_input[..., 0:self.rcnn_input_channel].transpose(1, 2).unsqueeze(dim=3)
            xyz_feature = self.xyz_up_layer(xyz_input)

            rpn_feature = pts_input[..., self.rcnn_input_channel:].transpose(1, 2).unsqueeze(dim=3)

            merged_feature = torch.cat((xyz_feature, rpn_feature), dim=1)
            merged_feature = self.merge_down_layer(merged_feature)
            l_xyz, l_features = [xyz], [merged_feature.squeeze(dim=3)]
        else:
            l_xyz, l_features = [xyz], [features]
        # print(xyz)
        # print(xyz.shape)
        # print(l_xyz[-1].shape, l_features[-1].shape)
        # input_features = torch.cat((l_xyz[-1], l_features[-1].transpose(1,2).contiguous()), dim=2)
        # print(l_features[-1].shape)
        pt_features = self.backbone(l_features[-1].unsqueeze(3))
        features = torch.max(pt_features, dim=2)[0]
        # print(features.shape)

        rcnn_cls = self.cls_layer(features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layer(features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)
        ret_dict = {'rcnn_cls': rcnn_cls, 'rcnn_reg': rcnn_reg}

        if self.training:
            ret_dict.update(target_dict)
        return ret_dict


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size, num_dims, num_points = x.size()
    device = torch.device('cuda')
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)
    feature = feature[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
    # print(feature.shape)
    return feature

def get_graph_feature_spatial(xyz, feature, k=20, idx=None):
    batch_size, num_dims, num_points = feature.size()
    device = torch.device('cuda')
    xyz = xyz.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(xyz, k=k)   # (batch_size, num_points, k)
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    feature = feature.view(batch_size, -1, num_points)
    x = feature.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)
    feature = feature[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)
    # print(feature.shape)
    return feature

def batch_process(input, fun, num_batches=5):
    num_data = input.shape[0]
    data_per_batch = np.ceil(num_data/num_batches).astype(int)
    for i in range(num_batches):
        if i == 0:
            out = fun(input[:data_per_batch])
        else:
            start = data_per_batch*i
            end = min(data_per_batch*(i+1), num_data)
            out = torch.cat((out,fun(input[start:end])), axis=0)
        # print(out.shape)
    return out

class GCNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = 'GCNLayer'
        self.units = out_channels
        self.in_channels = in_channels*2
        self.last_layer = last_layer
        if use_norm:
            BatchNorm2d = nn.BatchNorm2d(self.units,
                eps=1e-3, momentum=0.01)
            Conv2d = nn.Conv2d(self.in_channels, self.units, kernel_size=1, bias=False)
        else:
            BatchNorm2d = nn.Identity()
            Conv2d = nn.Conv2d(self.in_channels, self.units, kernel_size=1, bias=True)

        self.seq = nn.Sequential(Conv2d,
                                 BatchNorm2d,
                                 nn.LeakyReLU(negative_slope=0.2))
        self.k = 8

    def forward(self, inputs, xyz=None):
        # x = get_graph_feature(inputs.transpose(1,2).contiguous(), k=self.k)
        x = get_graph_feature_spatial(xyz, inputs.transpose(1,2).contiguous(), k=self.k)
        # print(x.shape)
        x = self.seq(x)
        # print(x.shape)
        x = x.max(dim=-1, keepdim=False)[0].transpose(1,2).contiguous()
        # print(x.shape)
        if self.last_layer:
            x = x.max(dim=1, keepdim=True)[0].transpose(1,2).contiguous()
        return x


class GCNNet(nn.Module):
    def __init__(self, num_classes, input_channels=0, use_xyz=True):
        super().__init__()

        self.gcn_layers = nn.ModuleList()
        channel_in = input_channels

        if cfg.RCNN.USE_RPN_FEATURES:
            self.rcnn_input_channel = 3 + int(cfg.RCNN.USE_INTENSITY) + int(cfg.RCNN.USE_MASK) + int(cfg.RCNN.USE_DEPTH)
            self.xyz_up_layer = pt_utils.SharedMLP([self.rcnn_input_channel] + cfg.RCNN.XYZ_UP_LAYER,
                                                   bn=cfg.RCNN.USE_BN)
            c_out = cfg.RCNN.XYZ_UP_LAYER[-1]
            self.merge_down_layer = pt_utils.SharedMLP([c_out * 2, c_out], bn=cfg.RCNN.USE_BN)

        for k in range(cfg.RCNN.GCN_CONFIG.FILTERS.__len__()):
            in_channels = 3 if k==0 else cfg.RCNN.GCN_CONFIG.FILTERS[k-1]
            self.gcn_layers.append(
                GCNLayer(
                    in_channels=in_channels,
                    out_channels=cfg.RCNN.GCN_CONFIG.FILTERS[k],
                    use_norm=cfg.RCNN.USE_BN,
                    last_layer=k==cfg.RCNN.GCN_CONFIG.FILTERS.__len__()-1
                )
            )

        # classification layer
        cls_channel = 1 if num_classes == 2 else num_classes
        cls_layers = []
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.CLS_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, cls_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            cls_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.cls_layer = nn.Sequential(*cls_layers)

        if cfg.RCNN.LOSS_CLS == 'SigmoidFocalLoss':
            self.cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=cfg.RCNN.FOCAL_ALPHA[0],
                                                                           gamma=cfg.RCNN.FOCAL_GAMMA)
        elif cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':
            self.cls_loss_func = F.binary_cross_entropy
        elif cfg.RCNN.LOSS_CLS == 'CrossEntropy':
            cls_weight = torch.from_numpy(cfg.RCNN.CLS_WEIGHT).float()
            self.cls_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduce=False, weight=cls_weight)
        else:
            raise NotImplementedError

        # regression layer
        per_loc_bin_num = int(cfg.RCNN.LOC_SCOPE / cfg.RCNN.LOC_BIN_SIZE) * 2
        loc_y_bin_num = int(cfg.RCNN.LOC_Y_SCOPE / cfg.RCNN.LOC_Y_BIN_SIZE) * 2
        reg_channel = per_loc_bin_num * 4 + cfg.RCNN.NUM_HEAD_BIN * 2 + 3
        reg_channel += (1 if not cfg.RCNN.LOC_Y_BY_BIN else loc_y_bin_num * 2)

        reg_layers = []
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.REG_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.REG_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, reg_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.reg_layer = nn.Sequential(*reg_layers)

        self.proposal_target_layer = ProposalTargetLayer()
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layer[-1].conv.weight, mean=0, std=0.001)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, input_data):
        """
        :param input_data: input dict
        :return:
        """
        if cfg.RCNN.ROI_SAMPLE_JIT:
            if self.training:
                with torch.no_grad():
                    target_dict = self.proposal_target_layer(input_data)

                pts_input = torch.cat((target_dict['sampled_pts'], target_dict['pts_feature']), dim=2)
                target_dict['pts_input'] = pts_input
            else:
                rpn_xyz, rpn_features = input_data['rpn_xyz'], input_data['rpn_features']
                batch_rois = input_data['roi_boxes3d']
                if cfg.RCNN.USE_INTENSITY:
                    pts_extra_input_list = [input_data['rpn_intensity'].unsqueeze(dim=2),
                                            input_data['seg_mask'].unsqueeze(dim=2)]
                else:
                    pts_extra_input_list = [input_data['seg_mask'].unsqueeze(dim=2)]

                if cfg.RCNN.USE_DEPTH:
                    pts_depth = input_data['pts_depth'] / 70.0 - 0.5
                    pts_extra_input_list.append(pts_depth.unsqueeze(dim=2))
                pts_extra_input = torch.cat(pts_extra_input_list, dim=2)

                pts_feature = torch.cat((pts_extra_input, rpn_features), dim=2)
                pooled_features, pooled_empty_flag = \
                        roipool3d_utils.roipool3d_gpu(rpn_xyz, pts_feature, batch_rois, cfg.RCNN.POOL_EXTRA_WIDTH,
                                                      sampled_pt_num=cfg.RCNN.NUM_POINTS)

                # canonical transformation
                batch_size = batch_rois.shape[0]
                roi_center = batch_rois[:, :, 0:3]
                pooled_features[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2)
                for k in range(batch_size):
                    pooled_features[k, :, :, 0:3] = kitti_utils.rotate_pc_along_y_torch(pooled_features[k, :, :, 0:3],
                                                                                        batch_rois[k, :, 6])

                pts_input = pooled_features.view(-1, pooled_features.shape[2], pooled_features.shape[3])
        else:
            pts_input = input_data['pts_input']
            target_dict = {}
            target_dict['pts_input'] = input_data['pts_input']
            target_dict['roi_boxes3d'] = input_data['roi_boxes3d']
            if self.training:
                target_dict['cls_label'] = input_data['cls_label']
                target_dict['reg_valid_mask'] = input_data['reg_valid_mask']
                target_dict['gt_of_rois'] = input_data['gt_boxes3d_ct']

        xyz, features = self._break_up_pc(pts_input)

        if cfg.RCNN.USE_RPN_FEATURES:
            xyz_input = pts_input[..., 0:self.rcnn_input_channel].transpose(1, 2).unsqueeze(dim=3)
            xyz_feature = self.xyz_up_layer(xyz_input)

            rpn_feature = pts_input[..., self.rcnn_input_channel:].transpose(1, 2).unsqueeze(dim=3)

            merged_feature = torch.cat((xyz_feature, rpn_feature), dim=1)
            merged_feature = self.merge_down_layer(merged_feature)
            l_xyz, l_features = [xyz], [merged_feature.squeeze(dim=3)]
        else:
            l_xyz, l_features = [xyz], [features]

        for i in range(len(self.gcn_layers)):
            if i == 0:
                li_features = self.gcn_layers[i](l_xyz[i], xyz=l_xyz[0])
            else:
                li_features = self.gcn_layers[i](l_features[i], xyz=l_xyz[0])
            # print(li_features.shape)
            l_features.append(li_features)
        # print(l_features[-1].shape, rpn_feature.shape)
        rcnn_cls = self.cls_layer(torch.cat((l_features[-1],rpn_feature.max(dim=2)[0]), dim=1)).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layer(torch.cat((l_features[-1],rpn_feature.max(dim=2)[0]), dim=1)).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)
        ret_dict = {'rcnn_cls': rcnn_cls, 'rcnn_reg': rcnn_reg}

        if self.training:
            ret_dict.update(target_dict)
        return ret_dict

def get_num_rot(degree_res):
    return int(np.ceil(360/degree_res))

def create_rot_mat(degree_res):
    angles = np.radians([(i * degree_res) for i in range(get_num_rot(degree_res))])
    # print(angles, angles.shape)
    cosas = np.cos(angles); sinas = np.sin(angles);
    # x_rot = (x - cx) * cosa + (z - cz) * (-sina);
    # z_rot = (x - cx) * sina + (z - cz) * cosa;
    # print(zip(cosas, sinas))
    rot_mat = [[[cosa, 0, -sina],
                         [   0, 1,    0],
                         [sina, 0, cosa]]
                for cosa, sina in zip(cosas, sinas)]
    # print(rot_mat[1])
    return rot_mat

class RotProjNet(nn.Module):
    def __init__(self, degree_res):
        super().__init__()
        self.degree_res = degree_res
        self.num_rot = get_num_rot(degree_res)
        self.rot_mat = torch.tensor(create_rot_mat(self.degree_res)).float().cuda()
        self.pixel_size = 0.0625 #pixel size in meters
        self.im_size_meters = np.array([4,4]) #image size in meters.
        self.im_size = (self.im_size_meters / self.pixel_size).astype(int)
        # print(self.im_size)

    def forward(self, xyz):
        #PARAM xyz: (B, N, 3)
        batch_size, num_pts, _ = xyz.shape
        xyz_rot = xyz.repeat_interleave(self.num_rot, dim=0).contiguous().transpose(1,2).contiguous()
        rot_mat = self.rot_mat.repeat(batch_size, 1, 1)
        xyz_rot = torch.bmm(rot_mat, xyz_rot)
        xyz_rot = xyz_rot.view(batch_size, self.num_rot, 3, num_pts).contiguous().transpose(2,3).contiguous() # (B, M, N, 3) with M different views
        xyz_proj = xyz_rot[:,:,:,:2] + torch.tensor([self.im_size_meters[0]/2, self.im_size_meters[0]/2]).cuda()  #(B, M, N, 3) with M different views
        # print(xyz_proj.shape)
        xyz_proj = torch.round(xyz_proj/self.pixel_size).long()
        # xyz_proj[:,:,:,1] = xyz_proj[:,:,:,1]
        # print(xyz_proj.shape)
        xyz_proj_mask = (xyz_proj[:,:,:,0] >= 0) & (xyz_proj[:,:,:,0] < self.im_size[0]) & (xyz_proj[:,:,:,1] >= 0) & (xyz_proj[:,:,:,1] < self.im_size[1])
        # print(xyz_proj_mask.shape)
        xyz_proj = xyz_proj * xyz_proj_mask.unsqueeze(-1).long()
        # print(xyz_proj)
        image = torch.zeros(batch_size, self.num_rot, self.im_size[0], self.im_size[1]).cuda()
        # print(xyz_proj[:,:,:,0].shape)
        batch = np.arange(batch_size)
        rots = np.arange(self.num_rot)
        pts = np.arange(num_pts)
        B, M, N = np.meshgrid(batch, rots, pts)
        B = B.flatten()
        M = M.flatten()
        N = N.flatten()
        # for b in range(batch_size):
        #     for m in range(self.num_rot):
        #         for n in range(num_pts):
        #             image[b,m, xyz_proj[b,m,n,0], xyz_proj[b,m,n,1]] = 1 #occupied voxelization
        # print(B.shape)
        if cfg.RCNN.ROT_CONFIG.OCCUPANCY:
            image[B,M,xyz_proj[B,M,N,1], xyz_proj[B,M,N,0]] = 1 # occupied voxelization
        else:
            image[B,M,xyz_proj[B,M,N,1], xyz_proj[B,M,N,0]] = xyz_rot[B,M,N,2]/10 # distance voxelization
        # for i in range(self.num_rot):
        #     f = open("views/image"+str(i)+".png", 'wb')      # binary mode is important
        #     w = png.Writer(64, 64, greyscale=True)
        #     w.write(f,image[0,i].detach().cpu().numpy().astype(np.uint8)*255)
        #     f.close()
        # print(image)
        return image

class RotRefModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_bn, dropout):
        super().__init__()
        # layer_size = [1024, 512, 256, 128]
        if use_bn:
            batch_norm = nn.BatchNorm2d
        else:
            batch_norm = nn.Identity
        bias = not use_bn
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias),
                                   batch_norm(out_channels),
                                   nn.Dropout(p=dropout),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

    def forward(self, img):
        return self.conv(img)
        
class RotRCNN(nn.Module):
    def __init__(self, num_classes, input_channels=0, use_xyz=True):
        super().__init__()

        self.rot_net = RotProjNet(cfg.RCNN.ROT_CONFIG.DEGREE_RES)
        self.ref_modules = nn.ModuleList()
        channel_in = input_channels

        for k in range(cfg.RCNN.ROT_CONFIG.NFILTERS.__len__()):
            channel_out = cfg.RCNN.ROT_CONFIG.NFILTERS[k] if cfg.RCNN.ROT_CONFIG.NFILTERS[k]!= -1 else None
            kernel_size = cfg.RCNN.ROT_CONFIG.KERNEL_SIZE[k]
            stride = cfg.RCNN.ROT_CONFIG.STRIDE[k]
            self.ref_modules.append(
                RotRefModule(
                    in_channels=channel_in,
                    out_channels=channel_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    use_bn=cfg.RCNN.USE_BN,
                    dropout=cfg.RCNN.ROT_CONFIG.DROPOUT
                )
            )
            channel_in = cfg.RCNN.ROT_CONFIG.NFILTERS[k]

        # classification layer
        cls_channel = 1 if num_classes == 2 else num_classes
        cls_layers = []
        pre_channel = channel_in * cfg.RCNN.ROT_CONFIG.CONV_FEAT_MULTIPLIER
        for k in range(0, cfg.RCNN.CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.CLS_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, cls_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            cls_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.cls_layer = nn.Sequential(*cls_layers)

        if cfg.RCNN.LOSS_CLS == 'SigmoidFocalLoss':
            self.cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=cfg.RCNN.FOCAL_ALPHA[0],
                                                                           gamma=cfg.RCNN.FOCAL_GAMMA)
        elif cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':
            self.cls_loss_func = F.binary_cross_entropy
        elif cfg.RCNN.LOSS_CLS == 'CrossEntropy':
            cls_weight = torch.from_numpy(cfg.RCNN.CLS_WEIGHT).float()
            self.cls_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduce=False, weight=cls_weight)
        else:
            raise NotImplementedError

        # regression layer
        per_loc_bin_num = int(cfg.RCNN.LOC_SCOPE / cfg.RCNN.LOC_BIN_SIZE) * 2
        loc_y_bin_num = int(cfg.RCNN.LOC_Y_SCOPE / cfg.RCNN.LOC_Y_BIN_SIZE) * 2
        reg_channel = per_loc_bin_num * 4 + cfg.RCNN.NUM_HEAD_BIN * 2 + 3
        reg_channel += (1 if not cfg.RCNN.LOC_Y_BY_BIN else loc_y_bin_num * 2)

        reg_layers = []
        pre_channel = channel_in * cfg.RCNN.ROT_CONFIG.CONV_FEAT_MULTIPLIER
        for k in range(0, cfg.RCNN.REG_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.REG_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, reg_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.reg_layer = nn.Sequential(*reg_layers)

        self.proposal_target_layer = ProposalTargetLayer()
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layer[-1].conv.weight, mean=0, std=0.001)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, input_data):
        """
        :param input_data: input dict
        :return:
        """
        if cfg.RCNN.ROI_SAMPLE_JIT:
            if self.training:
                with torch.no_grad():
                    target_dict = self.proposal_target_layer(input_data)

                pts_input = torch.cat((target_dict['sampled_pts'], target_dict['pts_feature']), dim=2)
                target_dict['pts_input'] = pts_input
            else:
                rpn_xyz, rpn_features = input_data['rpn_xyz'], input_data['rpn_features']
                batch_rois = input_data['roi_boxes3d']
                if cfg.RCNN.USE_INTENSITY:
                    pts_extra_input_list = [input_data['rpn_intensity'].unsqueeze(dim=2),
                                            input_data['seg_mask'].unsqueeze(dim=2)]
                else:
                    pts_extra_input_list = [input_data['seg_mask'].unsqueeze(dim=2)]

                if cfg.RCNN.USE_DEPTH:
                    pts_depth = input_data['pts_depth'] / 70.0 - 0.5
                    pts_extra_input_list.append(pts_depth.unsqueeze(dim=2))
                pts_extra_input = torch.cat(pts_extra_input_list, dim=2)

                pts_feature = torch.cat((pts_extra_input, rpn_features), dim=2)
                pooled_features, pooled_empty_flag = \
                        roipool3d_utils.roipool3d_gpu(rpn_xyz, pts_feature, batch_rois, cfg.RCNN.POOL_EXTRA_WIDTH,
                                                      sampled_pt_num=cfg.RCNN.NUM_POINTS)

                # canonical transformation
                batch_size = batch_rois.shape[0]
                roi_center = batch_rois[:, :, 0:3]
                pooled_features[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2)
                for k in range(batch_size):
                    pooled_features[k, :, :, 0:3] = kitti_utils.rotate_pc_along_y_torch(pooled_features[k, :, :, 0:3],
                                                                                        batch_rois[k, :, 6])

                pts_input = pooled_features.view(-1, pooled_features.shape[2], pooled_features.shape[3])
        else:
            pts_input = input_data['pts_input']
            target_dict = {}
            target_dict['pts_input'] = input_data['pts_input']
            target_dict['roi_boxes3d'] = input_data['roi_boxes3d']
            if self.training:
                target_dict['cls_label'] = input_data['cls_label']
                target_dict['reg_valid_mask'] = input_data['reg_valid_mask']
                target_dict['gt_of_rois'] = input_data['gt_boxes3d_ct']

        xyz, features = self._break_up_pc(pts_input)
        # print(xyz)

        l_features = [self.rot_net(xyz)]
        # print(l_features)
        for layer in self.ref_modules:
            l_features.append(layer(l_features[-1]))
            # print(l_features.shape)

        features = l_features[-1].view(l_features[-1].shape[0], -1).unsqueeze(2)
        # print(features.shape)
        rcnn_cls = self.cls_layer(features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layer(features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)
        ret_dict = {'rcnn_cls': rcnn_cls, 'rcnn_reg': rcnn_reg}

        if self.training:
            ret_dict.update(target_dict)
        return ret_dict

class RCNNNet(nn.Module):
    def __init__(self, num_classes, input_channels=0, use_xyz=True):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        if cfg.RCNN.USE_RPN_FEATURES:
            self.rcnn_input_channel = 3 + int(cfg.RCNN.USE_INTENSITY) + int(cfg.RCNN.USE_MASK) + int(cfg.RCNN.USE_DEPTH)
            self.xyz_up_layer = pt_utils.SharedMLP([self.rcnn_input_channel] + cfg.RCNN.XYZ_UP_LAYER,
                                                   bn=cfg.RCNN.USE_BN)
            c_out = cfg.RCNN.XYZ_UP_LAYER[-1]
            self.merge_down_layer = pt_utils.SharedMLP([c_out * 2, c_out], bn=cfg.RCNN.USE_BN)

        for k in range(cfg.RCNN.SA_CONFIG.NPOINTS.__len__()):
            mlps = [channel_in] + cfg.RCNN.SA_CONFIG.MLPS[k]

            npoint = cfg.RCNN.SA_CONFIG.NPOINTS[k] if cfg.RCNN.SA_CONFIG.NPOINTS[k] != -1 else None
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=npoint,
                    radius=cfg.RCNN.SA_CONFIG.RADIUS[k],
                    nsample=cfg.RCNN.SA_CONFIG.NSAMPLE[k],
                    mlp=mlps,
                    use_xyz=use_xyz,
                    bn=cfg.RCNN.USE_BN
                )
            )
            channel_in = mlps[-1]

        # classification layer
        cls_channel = 1 if num_classes == 2 else num_classes
        cls_layers = []
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.CLS_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, cls_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            cls_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.cls_layer = nn.Sequential(*cls_layers)

        if cfg.RCNN.LOSS_CLS == 'SigmoidFocalLoss':
            self.cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=cfg.RCNN.FOCAL_ALPHA[0],
                                                                           gamma=cfg.RCNN.FOCAL_GAMMA)
        elif cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':
            self.cls_loss_func = F.binary_cross_entropy
        elif cfg.RCNN.LOSS_CLS == 'CrossEntropy':
            cls_weight = torch.from_numpy(cfg.RCNN.CLS_WEIGHT).float()
            self.cls_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduce=False, weight=cls_weight)
        else:
            raise NotImplementedError

        # regression layer
        per_loc_bin_num = int(cfg.RCNN.LOC_SCOPE / cfg.RCNN.LOC_BIN_SIZE) * 2
        loc_y_bin_num = int(cfg.RCNN.LOC_Y_SCOPE / cfg.RCNN.LOC_Y_BIN_SIZE) * 2
        reg_channel = per_loc_bin_num * 4 + cfg.RCNN.NUM_HEAD_BIN * 2 + 3
        reg_channel += (1 if not cfg.RCNN.LOC_Y_BY_BIN else loc_y_bin_num * 2)

        reg_layers = []
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.REG_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.REG_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, reg_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.reg_layer = nn.Sequential(*reg_layers)

        self.proposal_target_layer = ProposalTargetLayer()
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layer[-1].conv.weight, mean=0, std=0.001)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, input_data):
        """
        :param input_data: input dict
        :return:
        """
        if cfg.RCNN.ROI_SAMPLE_JIT:
            if self.training:
                with torch.no_grad():
                    target_dict = self.proposal_target_layer(input_data)

                pts_input = torch.cat((target_dict['sampled_pts'], target_dict['pts_feature']), dim=2)
                target_dict['pts_input'] = pts_input
            else:
                rpn_xyz, rpn_features = input_data['rpn_xyz'], input_data['rpn_features']
                batch_rois = input_data['roi_boxes3d']
                if cfg.RCNN.USE_INTENSITY:
                    pts_extra_input_list = [input_data['rpn_intensity'].unsqueeze(dim=2),
                                            input_data['seg_mask'].unsqueeze(dim=2)]
                else:
                    pts_extra_input_list = [input_data['seg_mask'].unsqueeze(dim=2)]

                if cfg.RCNN.USE_DEPTH:
                    pts_depth = input_data['pts_depth'] / 70.0 - 0.5
                    pts_extra_input_list.append(pts_depth.unsqueeze(dim=2))
                pts_extra_input = torch.cat(pts_extra_input_list, dim=2)

                pts_feature = torch.cat((pts_extra_input, rpn_features), dim=2)
                pooled_features, pooled_empty_flag = \
                        roipool3d_utils.roipool3d_gpu(rpn_xyz, pts_feature, batch_rois, cfg.RCNN.POOL_EXTRA_WIDTH,
                                                      sampled_pt_num=cfg.RCNN.NUM_POINTS)

                # canonical transformation
                batch_size = batch_rois.shape[0]
                roi_center = batch_rois[:, :, 0:3]
                pooled_features[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2)
                for k in range(batch_size):
                    pooled_features[k, :, :, 0:3] = kitti_utils.rotate_pc_along_y_torch(pooled_features[k, :, :, 0:3],
                                                                                        batch_rois[k, :, 6])

                pts_input = pooled_features.view(-1, pooled_features.shape[2], pooled_features.shape[3])
        else:
            pts_input = input_data['pts_input'].view(-1,512,133)
            target_dict = {}
            target_dict['pts_input'] = input_data['pts_input'].view(-1,512,133)
            target_dict['roi_boxes3d'] = input_data['roi_boxes3d'].view(-1,7)
            if self.training:
                target_dict['cls_label'] = input_data['cls_label'].view(-1)
                target_dict['reg_valid_mask'] = input_data['reg_valid_mask'].view(-1)
                target_dict['gt_of_rois'] = input_data['gt_boxes3d_ct'].view(-1,7)

        xyz, features = self._break_up_pc(pts_input)

        if cfg.RCNN.USE_RPN_FEATURES:
            xyz_input = pts_input[..., 0:self.rcnn_input_channel].transpose(1, 2).unsqueeze(dim=3)
            xyz_feature = self.xyz_up_layer(xyz_input)

            rpn_feature = pts_input[..., self.rcnn_input_channel:].transpose(1, 2).unsqueeze(dim=3)

            merged_feature = torch.cat((xyz_feature, rpn_feature), dim=1)
            merged_feature = self.merge_down_layer(merged_feature)
            l_xyz, l_features = [xyz], [merged_feature.squeeze(dim=3)]
        else:
            l_xyz, l_features = [xyz], [features]


        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        # print(l_features[-1].shape)

        rcnn_cls = self.cls_layer(l_features[-1]).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layer(l_features[-1]).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)
        # print(rcnn_cls.shape)
        ret_dict = {'rcnn_cls': rcnn_cls, 'rcnn_reg': rcnn_reg}

        if self.training:
            ret_dict.update(target_dict)
        return ret_dict



class RefineRCNNNet(nn.Module):
    def __init__(self, num_classes, input_channels=0, use_xyz=True):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels
        opt = DenseOpts()
        opt.head=False
        opt.in_channels = 512 + 7 * cfg.RCNN.REF_CONFIG.USE_PROPOSALS + 128 * cfg.RCNN.REF_CONFIG.USE_RPN_FEATS
        # opts.constant_dilation=True
        opt.n_blocks = cfg.RCNN.REF_CONFIG.N_BLOCKS
        opt.kernel_size = cfg.RCNN.REF_CONFIG.KERNEL_SIZE
        opt.n_filters = cfg.RCNN.REF_CONFIG.N_FILTERS
        opt.conv = cfg.RCNN.REF_CONFIG.CONV
        opt.constant_dilation=cfg.RCNN.REF_CONFIG.CONSTANT_DILATION
        opt.linear_dilation=cfg.RCNN.REF_CONFIG.LINEAR_DILATION
        self.backbone = DenseDeepGCN(opt)
        if cfg.RCNN.USE_RPN_FEATURES:
            self.rcnn_input_channel = 3 + int(cfg.RCNN.USE_INTENSITY) + int(cfg.RCNN.USE_MASK) + int(cfg.RCNN.USE_DEPTH)
            self.xyz_up_layer = pt_utils.SharedMLP([self.rcnn_input_channel] + cfg.RCNN.XYZ_UP_LAYER,
                                                   bn=cfg.RCNN.USE_BN)
            c_out = cfg.RCNN.XYZ_UP_LAYER[-1]
            self.merge_down_layer = pt_utils.SharedMLP([c_out * 2, c_out], bn=cfg.RCNN.USE_BN)

        for k in range(cfg.RCNN.SA_CONFIG.NPOINTS.__len__()):
            mlps = [channel_in] + cfg.RCNN.SA_CONFIG.MLPS[k]

            npoint = cfg.RCNN.SA_CONFIG.NPOINTS[k] if cfg.RCNN.SA_CONFIG.NPOINTS[k] != -1 else None
            self.SA_modules.append(
                PointnetSAModule(
                    npoint=npoint,
                    radius=cfg.RCNN.SA_CONFIG.RADIUS[k],
                    nsample=cfg.RCNN.SA_CONFIG.NSAMPLE[k],
                    mlp=mlps,
                    use_xyz=use_xyz,
                    bn=cfg.RCNN.USE_BN
                )
            )
            channel_in = mlps[-1]

        # classification layer
        cls_channel = 1 if num_classes == 2 else num_classes
        cls_layers = []
        channel_in = self.backbone.channel_out
        if cfg.RCNN.REF_CONFIG.USE_RCNN_FEATS:
            channel_in += opt.in_channels

        pre_channel = channel_in
        for k in range(0, cfg.RCNN.CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.CLS_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, cls_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            cls_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.cls_layer = nn.Sequential(*cls_layers)

        if cfg.RCNN.LOSS_CLS == 'SigmoidFocalLoss':
            self.cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=cfg.RCNN.FOCAL_ALPHA[0],
                                                                           gamma=cfg.RCNN.FOCAL_GAMMA)
        elif cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':
            self.cls_loss_func = F.binary_cross_entropy
        elif cfg.RCNN.LOSS_CLS == 'CrossEntropy':
            cls_weight = torch.from_numpy(cfg.RCNN.CLS_WEIGHT).float()
            self.cls_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduce=False, weight=cls_weight)
        else:
            raise NotImplementedError

        # regression layer
        per_loc_bin_num = int(cfg.RCNN.LOC_SCOPE / cfg.RCNN.LOC_BIN_SIZE) * 2
        loc_y_bin_num = int(cfg.RCNN.LOC_Y_SCOPE / cfg.RCNN.LOC_Y_BIN_SIZE) * 2
        reg_channel = per_loc_bin_num * 4 + cfg.RCNN.NUM_HEAD_BIN * 2 + 3
        reg_channel += (1 if not cfg.RCNN.LOC_Y_BY_BIN else loc_y_bin_num * 2)

        reg_layers = []
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.REG_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.REG_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, reg_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.reg_layer = nn.Sequential(*reg_layers)

        self.proposal_target_layer = ProposalTargetLayer()
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layer[-1].conv.weight, mean=0, std=0.001)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, input_data):
        """
        :param input_data: input dict
        :return:
        """
        if cfg.RCNN.ROI_SAMPLE_JIT:
            if self.training:
                with torch.no_grad():
                    target_dict = self.proposal_target_layer(input_data)

                pts_input = torch.cat((target_dict['sampled_pts'], target_dict['pts_feature']), dim=2)
                target_dict['pts_input'] = pts_input
            else:
                rpn_xyz, rpn_features = input_data['rpn_xyz'], input_data['rpn_features']
                batch_rois = input_data['roi_boxes3d']
                if cfg.RCNN.USE_INTENSITY:
                    pts_extra_input_list = [input_data['rpn_intensity'].unsqueeze(dim=2),
                                            input_data['seg_mask'].unsqueeze(dim=2)]
                else:
                    pts_extra_input_list = [input_data['seg_mask'].unsqueeze(dim=2)]

                if cfg.RCNN.USE_DEPTH:
                    pts_depth = input_data['pts_depth'] / 70.0 - 0.5
                    pts_extra_input_list.append(pts_depth.unsqueeze(dim=2))
                pts_extra_input = torch.cat(pts_extra_input_list, dim=2)

                pts_feature = torch.cat((pts_extra_input, rpn_features), dim=2)
                pooled_features, pooled_empty_flag = \
                        roipool3d_utils.roipool3d_gpu(rpn_xyz, pts_feature, batch_rois, cfg.RCNN.POOL_EXTRA_WIDTH,
                                                      sampled_pt_num=cfg.RCNN.NUM_POINTS)

                # canonical transformation
                batch_size = batch_rois.shape[0]
                roi_center = batch_rois[:, :, 0:3]
                pooled_features[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2)
                for k in range(batch_size):
                    pooled_features[k, :, :, 0:3] = kitti_utils.rotate_pc_along_y_torch(pooled_features[k, :, :, 0:3],
                                                                                        batch_rois[k, :, 6])

                pts_input = pooled_features.view(-1, pooled_features.shape[2], pooled_features.shape[3])
        else:
            pts_input = input_data['pts_input'].view(-1,512,133)
            target_dict = {}
            target_dict['pts_input'] = input_data['pts_input'].view(-1,512,133)
            target_dict['roi_boxes3d'] = input_data['roi_boxes3d'].view(-1,7)
            if self.training:
                target_dict['cls_label'] = input_data['cls_label'].view(-1)
                target_dict['reg_valid_mask'] = input_data['reg_valid_mask'].view(-1)
                target_dict['gt_of_rois'] = input_data['gt_boxes3d_ct'].view(-1,7)

        xyz, features = self._break_up_pc(pts_input)

        if cfg.RCNN.USE_RPN_FEATURES:
            xyz_input = pts_input[..., 0:self.rcnn_input_channel].transpose(1, 2).unsqueeze(dim=3)
            xyz_feature = self.xyz_up_layer(xyz_input)

            rpn_feature = pts_input[..., self.rcnn_input_channel:].transpose(1, 2).unsqueeze(dim=3)

            merged_feature = torch.cat((xyz_feature, rpn_feature), dim=1)
            merged_feature = self.merge_down_layer(merged_feature)
            l_xyz, l_features = [xyz], [merged_feature.squeeze(dim=3)]
        else:
            l_xyz, l_features = [xyz], [features]


        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        # print(input_data.shape)
        # print(l_features[-1].shape)
        if self.training:
            num_proposals = cfg.RCNN.ROI_PER_IMAGE
        else:
            num_proposals = cfg.TEST.RPN_POST_NMS_TOP_N
        # print(l_features[-1].shape)
        
            # print(l_features[-1].shape)
            # print(proposals)
        if cfg.BATCH_SIZE == 1:
            features = l_features[-1].view(1,-1,l_features[-1].shape[1],1).contiguous().transpose(1,2).contiguous()
        else:
            features = l_features[-1].view(-1,num_proposals,l_features[-1].shape[1],1).contiguous().transpose(1,2).contiguous()
        
        #features = l_features[-1].view(cfg.BATCH_SIZE,num_proposals,l_features[-1].shape[1],1).contiguous().transpose(1,2).contiguous()

        if cfg.RCNN.REF_CONFIG.USE_PROPOSALS:
            proposals = input_data['roi_boxes3d'].view(-1,7)
            prop_feat = torch.zeros_like(proposals)
            prop_feat[:,0] = proposals[:,0]/80 + 0.5
            prop_feat[:,1] = proposals[:,1]/10 + 0.5
            prop_feat[:,2] = proposals[:,2]/70
            prop_feat[:,3] = proposals[:,3]/5
            prop_feat[:,4] = proposals[:,4]/10
            prop_feat[:,5] = proposals[:,5]/5
            prop_feat[:,6] = proposals[:,6]/(2*np.pi) + 0.5
            # print(features.shape, prop_feat.shape)
            features = torch.cat((features, prop_feat.transpose(0,1).contiguous().unsqueeze(0).unsqueeze(-1)), dim=1)

        if cfg.RCNN.REF_CONFIG.USE_RPN_FEATS:
            rpn_feats = pts_input[..., self.rcnn_input_channel:].transpose(1, 2).unsqueeze(dim=3)
            # print(rpn_feats.shape)
            rpn_feats = torch.max(rpn_feats, dim=2)[0].transpose(0,1).contiguous().unsqueeze(0)
            # print(features.shape, rpn_feats.shape)
            features = torch.cat((features, rpn_feats), dim=1)
        # print(features.shape)
        # print(xyz.shape)
        
        features = self.backbone(features).transpose(1,2).contiguous().view(-1,self.refine.channel_out,1).contiguous()
        # print(features.shape)
        # if cfg.RCNN.REF_CONFIG.USE_RCNN_FEATS:
        #     #print(features.shape, l_features[-1].shape)
        #     features = torch.cat((features, l_features[-1]), dim=1)
            # print(features.shape)
        rcnn_cls = self.cls_layer(features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layer(features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)
        # print(rcnn_cls.shape)
        ret_dict = {'rcnn_cls': rcnn_cls, 'rcnn_reg': rcnn_reg}

        if self.training:
            ret_dict.update(target_dict)
        return ret_dict



class RefineDeepRCNNNet(nn.Module):
    def __init__(self, num_classes, input_channels=0, use_xyz=True):
        super().__init__()

        opt = DenseOpts()
        self.backbone = DenseDeepGCN(opt)
        channel_in = input_channels
        opt = DenseOpts()
        opt.head=False
        opt.in_channels = self.backbone.channel_out + 7 * cfg.RCNN.REF_CONFIG.USE_PROPOSALS + 128 * cfg.RCNN.REF_CONFIG.USE_RPN_FEATS
        # opts.constant_dilation=True
        opt.n_blocks = cfg.RCNN.REF_CONFIG.N_BLOCKS
        opt.kernel_size = cfg.RCNN.REF_CONFIG.KERNEL_SIZE
        opt.n_filters = cfg.RCNN.REF_CONFIG.N_FILTERS
        opt.conv = cfg.RCNN.REF_CONFIG.CONV
        opt.constant_dilation=cfg.RCNN.REF_CONFIG.CONSTANT_DILATION
        opt.linear_dilation=cfg.RCNN.REF_CONFIG.LINEAR_DILATION
        self.refine = DenseDeepGCN(opt)
       
        # classification layer
        cls_channel = 1 if num_classes == 2 else num_classes
        cls_layers = []
        channel_in = self.refine.channel_out

        pre_channel = channel_in
        for k in range(0, cfg.RCNN.CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.CLS_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, cls_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            cls_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.cls_layer = nn.Sequential(*cls_layers)

        if cfg.RCNN.LOSS_CLS == 'SigmoidFocalLoss':
            self.cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=cfg.RCNN.FOCAL_ALPHA[0],
                                                                           gamma=cfg.RCNN.FOCAL_GAMMA)
        elif cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':
            self.cls_loss_func = F.binary_cross_entropy
        elif cfg.RCNN.LOSS_CLS == 'CrossEntropy':
            cls_weight = torch.from_numpy(cfg.RCNN.CLS_WEIGHT).float()
            self.cls_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduce=False, weight=cls_weight)
        else:
            raise NotImplementedError

        # regression layer
        per_loc_bin_num = int(cfg.RCNN.LOC_SCOPE / cfg.RCNN.LOC_BIN_SIZE) * 2
        loc_y_bin_num = int(cfg.RCNN.LOC_Y_SCOPE / cfg.RCNN.LOC_Y_BIN_SIZE) * 2
        reg_channel = per_loc_bin_num * 4 + cfg.RCNN.NUM_HEAD_BIN * 2 + 3
        reg_channel += (1 if not cfg.RCNN.LOC_Y_BY_BIN else loc_y_bin_num * 2)

        reg_layers = []
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.REG_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.REG_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, reg_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.reg_layer = nn.Sequential(*reg_layers)

        self.proposal_target_layer = ProposalTargetLayer()
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layer[-1].conv.weight, mean=0, std=0.001)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, input_data):
        """
        :param input_data: input dict
        :return:
        """
        if cfg.RCNN.ROI_SAMPLE_JIT:
            if self.training:
                with torch.no_grad():
                    target_dict = self.proposal_target_layer(input_data)

                pts_input = torch.cat((target_dict['sampled_pts'], target_dict['pts_feature']), dim=2)
                target_dict['pts_input'] = pts_input
            else:
                rpn_xyz, rpn_features = input_data['rpn_xyz'], input_data['rpn_features']
                batch_rois = input_data['roi_boxes3d']
                if cfg.RCNN.USE_INTENSITY:
                    pts_extra_input_list = [input_data['rpn_intensity'].unsqueeze(dim=2),
                                            input_data['seg_mask'].unsqueeze(dim=2)]
                else:
                    pts_extra_input_list = [input_data['seg_mask'].unsqueeze(dim=2)]

                if cfg.RCNN.USE_DEPTH:
                    pts_depth = input_data['pts_depth'] / 70.0 - 0.5
                    pts_extra_input_list.append(pts_depth.unsqueeze(dim=2))
                pts_extra_input = torch.cat(pts_extra_input_list, dim=2)

                pts_feature = torch.cat((pts_extra_input, rpn_features), dim=2)
                pooled_features, pooled_empty_flag = \
                        roipool3d_utils.roipool3d_gpu(rpn_xyz, pts_feature, batch_rois, cfg.RCNN.POOL_EXTRA_WIDTH,
                                                      sampled_pt_num=cfg.RCNN.NUM_POINTS)

                # canonical transformation
                batch_size = batch_rois.shape[0]
                roi_center = batch_rois[:, :, 0:3]
                pooled_features[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2)
                for k in range(batch_size):
                    pooled_features[k, :, :, 0:3] = kitti_utils.rotate_pc_along_y_torch(pooled_features[k, :, :, 0:3],
                                                                                        batch_rois[k, :, 6])

                pts_input = pooled_features.view(-1, pooled_features.shape[2], pooled_features.shape[3])
        else:
            pts_input = input_data['pts_input'].view(-1,512,133)
            target_dict = {}
            target_dict['pts_input'] = input_data['pts_input'].view(-1,512,133)
            target_dict['roi_boxes3d'] = input_data['roi_boxes3d'].view(-1,7)
            if self.training:
                target_dict['cls_label'] = input_data['cls_label'].view(-1)
                target_dict['reg_valid_mask'] = input_data['reg_valid_mask'].view(-1)
                target_dict['gt_of_rois'] = input_data['gt_boxes3d_ct'].view(-1,7)

        xyz, features = self._break_up_pc(pts_input)

        pt_features = self.backbone(xyz.transpose(1,2).contiguous().unsqueeze(3))
        features = torch.max(pt_features, dim=2)[0]
        # print(input_data.shape)
        # print(l_features[-1].shape)

        if self.training:
            num_proposals = cfg.RCNN.ROI_PER_IMAGE
        else:
            num_proposals = cfg.TEST.RPN_POST_NMS_TOP_N
        
        if cfg.BATCH_SIZE == 1:
            ref_features_prep = features.view(1,-1,features.shape[1],1).contiguous().transpose(1,2).contiguous()
        else:
            ref_features_prep = features.view(-1,num_proposals,features.shape[1],1).contiguous().transpose(1,2).contiguous()
        #ref_features_prep = features.view(cfg.BATCH_SIZE,num_proposals,features.shape[1],1).contiguous().transpose(1,2).contiguous()

        if cfg.RCNN.REF_CONFIG.USE_PROPOSALS:
            proposals = input_data['roi_boxes3d'].view(-1,7)
            prop_feat = torch.zeros_like(proposals)
            prop_feat[:,0] = proposals[:,0]/80 + 0.5
            prop_feat[:,1] = proposals[:,1]/10 + 0.5
            prop_feat[:,2] = proposals[:,2]/70
            prop_feat[:,3] = proposals[:,3]/5
            prop_feat[:,4] = proposals[:,4]/10
            prop_feat[:,5] = proposals[:,5]/5
            prop_feat[:,6] = proposals[:,6]/(2*np.pi) + 0.5
            # print(features.shape, prop_feat.shape)
            ref_features_prep = torch.cat((ref_features_prep, prop_feat.transpose(0,1).contiguous().unsqueeze(0).unsqueeze(-1)), dim=1)

        if cfg.RCNN.REF_CONFIG.USE_RPN_FEATS:
            rpn_feats = pts_input[..., self.rcnn_input_channel:].transpose(1, 2).unsqueeze(dim=3)
            # print(rpn_feats.shape)
            rpn_feats = torch.max(rpn_feats, dim=2)[0].transpose(0,1).contiguous().unsqueeze(0)
            # print(features.shape, rpn_feats.shape)
            ref_features_prep = torch.cat((ref_features_prep, rpn_feats), dim=1)

        # print(features.shape)
        # print(xyz.shape)

        ref_features = self.refine(ref_features_prep).transpose(1,2).contiguous().view(-1,self.refine.channel_out,1).contiguous()

        # if cfg.RCNN.REF_CONFIG.USE_RCNN_FEATS:
        #     #print(features.shape, l_features[-1].shape)
        #     ref_features = torch.cat((ref_features, l_features[-1]), dim=1)
        # print(features.shape)


        rcnn_cls = self.cls_layer(ref_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layer(ref_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)
        # print(rcnn_cls.shape)
        ret_dict = {'rcnn_cls': rcnn_cls, 'rcnn_reg': rcnn_reg}

        if self.training:
            ret_dict.update(target_dict)
        return ret_dict


class DenseFeatRefineRCNN(nn.Module):
    def __init__(self, num_classes, input_channels=0, use_xyz=True):
        super().__init__()

        if cfg.RCNN.USE_RPN_FEATURES:
            self.rcnn_input_channel = 3 + int(cfg.RCNN.USE_INTENSITY) + int(cfg.RCNN.USE_MASK) + int(cfg.RCNN.USE_DEPTH)
            self.xyz_up_layer = pt_utils.SharedMLP([self.rcnn_input_channel] + cfg.RCNN.XYZ_UP_LAYER,
                                                   bn=cfg.RCNN.USE_BN)
            c_out = cfg.RCNN.XYZ_UP_LAYER[-1]
            self.merge_down_layer = pt_utils.SharedMLP([c_out * 2, c_out], bn=cfg.RCNN.USE_BN)

        opt = DenseOpts()
        opt.in_channels = input_channels
        self.backbone = DenseDeepGCN(opt)

        # channel_in = input_channels
        opt = DenseOpts()
        opt.head=False
        opt.in_channels = self.backbone.channel_out + 7 * cfg.RCNN.REF_CONFIG.USE_PROPOSALS + 128 * cfg.RCNN.REF_CONFIG.USE_RPN_FEATS
        # opts.constant_dilation=True
        opt.n_blocks = cfg.RCNN.REF_CONFIG.N_BLOCKS
        opt.kernel_size = cfg.RCNN.REF_CONFIG.KERNEL_SIZE
        opt.n_filters = cfg.RCNN.REF_CONFIG.N_FILTERS
        opt.conv = cfg.RCNN.REF_CONFIG.CONV
        opt.constant_dilation=cfg.RCNN.REF_CONFIG.CONSTANT_DILATION
        opt.linear_dilation=cfg.RCNN.REF_CONFIG.LINEAR_DILATION
        self.refine = DenseDeepGCN(opt)

        # classification layer
        cls_channel = 1 if num_classes == 2 else num_classes
        cls_layers = []
        channel_in = self.refine.channel_out
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.CLS_FC.__len__()):
            cls_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.CLS_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.CLS_FC[k]
        cls_layers.append(pt_utils.Conv1d(pre_channel, cls_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            cls_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.cls_layer = nn.Sequential(*cls_layers)

        if cfg.RCNN.LOSS_CLS == 'SigmoidFocalLoss':
            self.cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=cfg.RCNN.FOCAL_ALPHA[0],
                                                                           gamma=cfg.RCNN.FOCAL_GAMMA)
        elif cfg.RCNN.LOSS_CLS == 'BinaryCrossEntropy':
            self.cls_loss_func = F.binary_cross_entropy
        elif cfg.RCNN.LOSS_CLS == 'CrossEntropy':
            cls_weight = torch.from_numpy(cfg.RCNN.CLS_WEIGHT).float()
            self.cls_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduce=False, weight=cls_weight)
        else:
            raise NotImplementedError

        # regression layer
        per_loc_bin_num = int(cfg.RCNN.LOC_SCOPE / cfg.RCNN.LOC_BIN_SIZE) * 2
        loc_y_bin_num = int(cfg.RCNN.LOC_Y_SCOPE / cfg.RCNN.LOC_Y_BIN_SIZE) * 2
        reg_channel = per_loc_bin_num * 4 + cfg.RCNN.NUM_HEAD_BIN * 2 + 3
        reg_channel += (1 if not cfg.RCNN.LOC_Y_BY_BIN else loc_y_bin_num * 2)

        reg_layers = []
        pre_channel = channel_in
        for k in range(0, cfg.RCNN.REG_FC.__len__()):
            reg_layers.append(pt_utils.Conv1d(pre_channel, cfg.RCNN.REG_FC[k], bn=cfg.RCNN.USE_BN))
            pre_channel = cfg.RCNN.REG_FC[k]
        reg_layers.append(pt_utils.Conv1d(pre_channel, reg_channel, activation=None))
        if cfg.RCNN.DP_RATIO >= 0:
            reg_layers.insert(1, nn.Dropout(cfg.RCNN.DP_RATIO))
        self.reg_layer = nn.Sequential(*reg_layers)

        self.proposal_target_layer = ProposalTargetLayer()
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layer[-1].conv.weight, mean=0, std=0.001)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, input_data):
        """
        :param input_data: input dict
        :return:
        """
        if cfg.RCNN.ROI_SAMPLE_JIT:
            if self.training:
                with torch.no_grad():
                    target_dict = self.proposal_target_layer(input_data)

                pts_input = torch.cat((target_dict['sampled_pts'], target_dict['pts_feature']), dim=2)
                target_dict['pts_input'] = pts_input
            else:
                rpn_xyz, rpn_features = input_data['rpn_xyz'], input_data['rpn_features']
                batch_rois = input_data['roi_boxes3d']
                if cfg.RCNN.USE_INTENSITY:
                    pts_extra_input_list = [input_data['rpn_intensity'].unsqueeze(dim=2),
                                            input_data['seg_mask'].unsqueeze(dim=2)]
                else:
                    pts_extra_input_list = [input_data['seg_mask'].unsqueeze(dim=2)]

                if cfg.RCNN.USE_DEPTH:
                    pts_depth = input_data['pts_depth'] / 70.0 - 0.5
                    pts_extra_input_list.append(pts_depth.unsqueeze(dim=2))
                pts_extra_input = torch.cat(pts_extra_input_list, dim=2)

                pts_feature = torch.cat((pts_extra_input, rpn_features), dim=2)
                pooled_features, pooled_empty_flag = \
                        roipool3d_utils.roipool3d_gpu(rpn_xyz, pts_feature, batch_rois, cfg.RCNN.POOL_EXTRA_WIDTH,
                                                      sampled_pt_num=cfg.RCNN.NUM_POINTS)

                # canonical transformation
                batch_size = batch_rois.shape[0]
                roi_center = batch_rois[:, :, 0:3]
                pooled_features[:, :, :, 0:3] -= roi_center.unsqueeze(dim=2)
                for k in range(batch_size):
                    pooled_features[k, :, :, 0:3] = kitti_utils.rotate_pc_along_y_torch(pooled_features[k, :, :, 0:3],
                                                                                        batch_rois[k, :, 6])

                pts_input = pooled_features.view(-1, pooled_features.shape[2], pooled_features.shape[3])
        else:
            pts_input = input_data['pts_input'].view(-1,512,133)
            target_dict = {}
            target_dict['pts_input'] = input_data['pts_input'].view(-1,512,133)
            target_dict['roi_boxes3d'] = input_data['roi_boxes3d'].view(-1,7)
            if self.training:
                target_dict['cls_label'] = input_data['cls_label'].view(-1)
                target_dict['reg_valid_mask'] = input_data['reg_valid_mask'].view(-1)
                target_dict['gt_of_rois'] = input_data['gt_boxes3d_ct'].view(-1,7)

        xyz, features = self._break_up_pc(pts_input)

        if cfg.RCNN.USE_RPN_FEATURES:
            xyz_input = pts_input[..., 0:self.rcnn_input_channel].transpose(1, 2).unsqueeze(dim=3)
            xyz_feature = self.xyz_up_layer(xyz_input)

            rpn_feature = pts_input[..., self.rcnn_input_channel:].transpose(1, 2).unsqueeze(dim=3)

            merged_feature = torch.cat((xyz_feature, rpn_feature), dim=1)
            merged_feature = self.merge_down_layer(merged_feature)
            l_xyz, l_features = [xyz], [merged_feature.squeeze(dim=3)]
        else:
            l_xyz, l_features = [xyz], [features]


        pt_features = self.backbone(l_features[-1].unsqueeze(3))
        features = torch.max(pt_features, dim=2)[0]

        if self.training:
            num_proposals = cfg.RCNN.ROI_PER_IMAGE
        else:
            num_proposals = cfg.TEST.RPN_POST_NMS_TOP_N
        
        if cfg.BATCH_SIZE == 1:
            ref_features_prep = features.view(1,-1,features.shape[1],1).contiguous().transpose(1,2).contiguous()
        else:
            ref_features_prep = features.view(-1,num_proposals,features.shape[1],1).contiguous().transpose(1,2).contiguous()
        #ref_features_prep = features.view(cfg.BATCH_SIZE,num_proposals,features.shape[1],1).contiguous().transpose(1,2).contiguous()

        if cfg.RCNN.REF_CONFIG.USE_PROPOSALS:
            proposals = input_data['roi_boxes3d'].view(-1,7)
            # print(target_dict['gt_of_rois'].shape)
            prop_feat = torch.zeros_like(proposals)
            prop_feat[:,0] = proposals[:,0]/80 + 0.5
            prop_feat[:,1] = proposals[:,1]/10 + 0.5
            prop_feat[:,2] = proposals[:,2]/70
            prop_feat[:,3] = proposals[:,3]/5
            prop_feat[:,4] = proposals[:,4]/10
            prop_feat[:,5] = proposals[:,5]/5
            prop_feat[:,6] = proposals[:,6]/(2*np.pi) + 0.5
            # print(ref_features_prep.shape, prop_feat.shape)
            ref_features_prep = torch.cat((ref_features_prep, prop_feat.transpose(0,1).contiguous().unsqueeze(0).unsqueeze(-1)), dim=1)

        if cfg.RCNN.REF_CONFIG.USE_RPN_FEATS:
            rpn_feats = pts_input[..., self.rcnn_input_channel:].transpose(1, 2).unsqueeze(dim=3)
            # print(rpn_feats.shape)
            rpn_feats = torch.max(rpn_feats, dim=2)[0].transpose(0,1).contiguous().unsqueeze(0)
            # print(features.shape, rpn_feats.shape)
            ref_features_prep = torch.cat((ref_features_prep, rpn_feats), dim=1)

        # print(features.shape)
        # print(xyz.shape)

        ref_features = self.refine(ref_features_prep).transpose(1,2).contiguous().view(-1,self.refine.channel_out,1).contiguous()

        rcnn_cls = self.cls_layer(ref_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layer(ref_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)
        ret_dict = {'rcnn_cls': rcnn_cls, 'rcnn_reg': rcnn_reg}

        if self.training:
            ret_dict.update(target_dict)
        return ret_dict