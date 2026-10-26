import torch

from lane_detection_ai.model.model.backbone import ResNetBackbone
from lane_detection_ai.model.model.seg_model import SegHead
from lane_detection_ai.model.utils.common import initialize_weights
from lane_detection_ai.model.utils.config import Config

# Width of the hidden layer of the classification MLP. Older models were trained
# with 512; set `mlp_mid_dim = 512` in their model config to load them.
DEFAULT_MLP_MID_DIM = 2048


class ParsingNet(torch.nn.Module):
    """Ultra Fast Lane Detection v2 network (row and column anchors)."""

    def __init__(
        self,
        *,
        pretrained: bool = False,
        backbone: str = "50",
        num_grid_row: int,
        num_cls_row: int,
        num_grid_col: int,
        num_cls_col: int,
        num_lane_on_row: int,
        num_lane_on_col: int,
        use_aux: bool = False,
        input_height: int,
        input_width: int,
        fc_norm: bool = False,
        mlp_mid_dim: int = DEFAULT_MLP_MID_DIM,
    ) -> None:
        """Initialize the network.

        Args:
            pretrained: Initialize the backbone with the ImageNet weights.
            backbone: The ResNet variant of the backbone.
            num_grid_row: Number of grid cells of the row anchors.
            num_cls_row: Number of row anchors.
            num_grid_col: Number of grid cells of the column anchors.
            num_cls_col: Number of column anchors.
            num_lane_on_row: Number of lanes on the row anchors.
            num_lane_on_col: Number of lanes on the column anchors.
            use_aux: Add the auxiliary segmentation head (training only).
            input_height: Height of the input image.
            input_width: Width of the input image.
            fc_norm: Add a layer norm in front of the classification MLP.
            mlp_mid_dim: Width of the hidden layer of the classification MLP.
        """
        super().__init__()
        self.num_grid_row = num_grid_row
        self.num_cls_row = num_cls_row
        self.num_grid_col = num_grid_col
        self.num_cls_col = num_cls_col
        self.num_lane_on_row = num_lane_on_row
        self.num_lane_on_col = num_lane_on_col
        self.use_aux = use_aux
        self.dim1 = self.num_grid_row * self.num_cls_row * self.num_lane_on_row
        self.dim2 = self.num_grid_col * self.num_cls_col * self.num_lane_on_col
        self.dim3 = 2 * self.num_cls_row * self.num_lane_on_row
        self.dim4 = 2 * self.num_cls_col * self.num_lane_on_col
        self.total_dim = self.dim1 + self.dim2 + self.dim3 + self.dim4
        self.input_dim = input_height // 32 * input_width // 32 * 8

        self.model = ResNetBackbone(backbone, pretrained=pretrained)
        self.cls = torch.nn.Sequential(
            torch.nn.LayerNorm(self.input_dim) if fc_norm else torch.nn.Identity(),
            torch.nn.Linear(self.input_dim, mlp_mid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_mid_dim, self.total_dim),
        )
        pool_channels = 512 if backbone in {"34", "18", "9"} else 2048
        self.pool = torch.nn.Conv2d(pool_channels, 8, 1)
        if self.use_aux:
            self.seg_head = SegHead(backbone, num_lane_on_row + num_lane_on_col)
        initialize_weights(self.cls)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict the lane locations and existences for a batch of images."""
        x2, x3, fea = self.model(x)
        seg_out = self.seg_head(x2, x3, fea) if self.use_aux else None
        out = self.cls(self.pool(fea).view(-1, self.input_dim))

        pred_dict = {
            "loc_row": out[:, : self.dim1].view(
                -1, self.num_grid_row, self.num_cls_row, self.num_lane_on_row
            ),
            "loc_col": out[:, self.dim1 : self.dim1 + self.dim2].view(
                -1, self.num_grid_col, self.num_cls_col, self.num_lane_on_col
            ),
            "exist_row": out[
                :, self.dim1 + self.dim2 : self.dim1 + self.dim2 + self.dim3
            ].view(-1, 2, self.num_cls_row, self.num_lane_on_row),
            "exist_col": out[:, -self.dim4 :].view(
                -1, 2, self.num_cls_col, self.num_lane_on_col
            ),
        }
        if seg_out is not None:
            pred_dict["seg_out"] = seg_out

        return pred_dict


def get_model(cfg: Config) -> ParsingNet:
    """Create the network described by the model config.

    Args:
        cfg: The model config.

    Returns:
        The network on the CPU (with random weights).
    """
    return ParsingNet(
        backbone=cfg.backbone,
        num_grid_row=cfg.num_cell_row,
        num_cls_row=cfg.num_row,
        num_grid_col=cfg.num_cell_col,
        num_cls_col=cfg.num_col,
        num_lane_on_row=cfg.num_lanes,
        num_lane_on_col=cfg.num_lanes,
        use_aux=cfg.use_aux,
        input_height=cfg.train_height,
        input_width=cfg.train_width,
        fc_norm=cfg.fc_norm,
        mlp_mid_dim=getattr(cfg, "mlp_mid_dim", DEFAULT_MLP_MID_DIM),
    ).cpu()
