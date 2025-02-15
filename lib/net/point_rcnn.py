import torch
import torch.nn as nn
from lib.net.rpn import RPN
from lib.net.rcnn_net import RCNNNet, GCNNet, RotRCNN, get_num_rot, DenseRCNN, RefineRCNNNet, RefineDeepRCNNNet, DenseFeatRCNN, DenseFeatRefineRCNN
from lib.config import cfg


class PointRCNN(nn.Module):
    def __init__(self, num_classes, use_xyz=True, mode='TRAIN'):
        super().__init__()

        assert cfg.RPN.ENABLED or cfg.RCNN.ENABLED

        if cfg.RPN.ENABLED:
            self.rpn = RPN(use_xyz=use_xyz, mode=mode)

        if cfg.RCNN.ENABLED:
            rcnn_input_channels = 128  # channels of rpn features
            if cfg.RCNN.BACKBONE == 'pointnet':
                self.rcnn_net = RCNNNet(num_classes=num_classes, input_channels=rcnn_input_channels, use_xyz=use_xyz)
            elif cfg.RCNN.BACKBONE == 'pointsift':
                pass 
            elif cfg.RCNN.BACKBONE == 'edgeconv':
                rcnn_input_channels = cfg.RCNN.GCN_CONFIG.FILTERS[-1]*2
                self.rcnn_net = GCNNet(num_classes=num_classes, input_channels=rcnn_input_channels, use_xyz=use_xyz)
            elif cfg.RCNN.BACKBONE == 'rotnet':
                rcnn_input_channels = get_num_rot(cfg.RCNN.ROT_CONFIG.DEGREE_RES)
                self.rcnn_net = RotRCNN(num_classes=num_classes, input_channels=rcnn_input_channels, use_xyz=use_xyz)
            elif cfg.RCNN.BACKBONE == 'deepgcn':
                # rcnn_input_channels = 256
                self.rcnn_net = DenseRCNN(num_classes=num_classes, input_channels=rcnn_input_channels, use_xyz=use_xyz)
                
            elif cfg.RCNN.BACKBONE == 'refine':
                # rcnn_input_channels = 256
                self.rcnn_net = RefineRCNNNet(num_classes=num_classes, input_channels=rcnn_input_channels, use_xyz=use_xyz)
            elif cfg.RCNN.BACKBONE == 'deeprefine':
                # rcnn_input_channels = 256
                self.rcnn_net = RefineDeepRCNNNet(num_classes=num_classes, input_channels=rcnn_input_channels, use_xyz=use_xyz)
            elif cfg.RCNN.BACKBONE == 'deepfeats':
                # rcnn_input_channels = 256
                self.rcnn_net = DenseFeatRCNN(num_classes=num_classes, input_channels=rcnn_input_channels, use_xyz=use_xyz)
            elif cfg.RCNN.BACKBONE == 'deepfeatsrefine':
                # rcnn_input_channels = 256
                self.rcnn_net = DenseFeatRefineRCNN(num_classes=num_classes, input_channels=rcnn_input_channels, use_xyz=use_xyz)
            else:
                raise NotImplementedError
        if cfg.CGCN.ENABLED:
            self.cgcn_net = None
        print(sum(p.numel() for p in self.rcnn_net.parameters()))

    def forward(self, input_data):
        if cfg.RPN.ENABLED:
            output = {}
            # rpn inference
            with torch.set_grad_enabled((not cfg.RPN.FIXED) and self.training):
                if cfg.RPN.FIXED:
                    self.rpn.eval()
                rpn_output = self.rpn(input_data)
                output.update(rpn_output)

            # rcnn inference
            if cfg.RCNN.ENABLED:
                with torch.no_grad():
                    rpn_cls, rpn_reg = rpn_output['rpn_cls'], rpn_output['rpn_reg']
                    backbone_xyz, backbone_features = rpn_output['backbone_xyz'], rpn_output['backbone_features']

                    rpn_scores_raw = rpn_cls[:, :, 0]
                    rpn_scores_norm = torch.sigmoid(rpn_scores_raw)
                    seg_mask = (rpn_scores_norm > cfg.RPN.SCORE_THRESH).float()
                    pts_depth = torch.norm(backbone_xyz, p=2, dim=2)

                    # proposal layer
                    rois, roi_scores_raw = self.rpn.proposal_layer(rpn_scores_raw, rpn_reg, backbone_xyz)  # (B, M, 7)

                    output['rois'] = rois
                    output['roi_scores_raw'] = roi_scores_raw
                    output['seg_result'] = seg_mask

                rcnn_input_info = {'rpn_xyz': backbone_xyz,
                                   'rpn_features': backbone_features.permute((0, 2, 1)),
                                   'seg_mask': seg_mask,
                                   'roi_boxes3d': rois,
                                   'pts_depth': pts_depth}
                if self.training:
                    rcnn_input_info['gt_boxes3d'] = input_data['gt_boxes3d']

                rcnn_output = self.rcnn_net(rcnn_input_info)
                output.update(rcnn_output)

        elif cfg.RCNN.ENABLED:
            output = self.rcnn_net(input_data)
        else:
            raise NotImplementedError
        # print(output['roi_boxes3d'])
        # print()
        return output
