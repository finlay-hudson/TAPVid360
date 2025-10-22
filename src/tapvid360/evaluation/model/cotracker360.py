# Modified from https://github.com/facebookresearch/co-tracker/blob/main/cotracker/models/core/cotracker/cotracker3_offline.py

import roma

import torch
import torch.nn.functional as F

from cotracker.models.core.cotracker.cotracker import EfficientUpdateFormer
from cotracker.models.core.cotracker.cotracker3_online import CoTrackerThreeBase, posenc

from src.tapvid360.training.utils import CheckpointSaver

torch.manual_seed(0)


class CoTracker360(CoTrackerThreeBase):
    def __init__(self, **args):
        super(CoTracker360, self).__init__(**args)
        self.updateformer = EfficientUpdateFormer(
            space_depth=3,
            time_depth=3,
            input_dim=self.input_dim,
            hidden_size=384,
            output_dim=11,
            mlp_ratio=4.0,
            num_virtual_tracks=self.num_virtual_tracks,
            add_space_attn=True,
            linear_layer_for_vis_conf=self.linear_layer_for_vis_conf,
        )
        self.checkpoint_name = "cotracker360"

    @staticmethod
    def load_pretrained_state_dict(checkpoint_path=None):
        if checkpoint_path is None:
            raise ValueError("need a checkpoint path")
        with open(checkpoint_path, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
        return state_dict

    def load_weights(self, checkpoint_path, ignored_layers=None, strict=False):
        state_dict = self.load_pretrained_state_dict(checkpoint_path)
        if self.checkpoint_name in state_dict:
            state_dict = state_dict[self.checkpoint_name]
        filtered_state_dict = state_dict
        if ignored_layers is not None:
            filtered_state_dict = {k: v for k, v in state_dict.items() if not any(l in k for l in ignored_layers)}
        self.load_state_dict(filtered_state_dict, strict=strict)

    def save_checkpoint(self, checkpointer: CheckpointSaver, current_epoch: int, eval_dists: dict, losses: dict):
        checkpointer.save(current_epoch, {self.checkpoint_name: self}, eval_dists, losses)

    def forward(
            self,
            video,
            all_queries,
            mapper,
            iters=4,
            is_train=False,
            add_space_attn=True,
            fmaps_chunk_size=200,
    ):
        """Predict tracks

        Args:
            video (FloatTensor[B, T, 3]): input videos.
            all_queries: (List[FloatTensor[B, N, 3]], FloatTensor[B, N, 2]):
                - queries_uv (FloatTensor[B, N, 3]): initial uv queries.
                - queries (FloatTensor[B, N, 2]): initial xy queries.
            iters (int, optional): number of updates. Defaults to 4.
            is_train (bool, optional): enables training mode. Defaults to False.
        Returns:
            - coords_predicted (FloatTensor[B, T, N, 2]):
            - vis_predicted (FloatTensor[B, T, N]):
            - train_data: `None` if `is_train` is false, otherwise:
                - all_vis_predictions (List[FloatTensor[B, S, N, 1]]):
                - all_coords_predictions (List[FloatTensor[B, S, N, 2]]):
                - mask (BoolTensor[B, T, N]):
        """
        B, T, C, H, W = video.shape
        queries_uv, queries = all_queries
        device = queries.device
        assert H % self.stride == 0 and W % self.stride == 0

        B, N, __ = queries.shape
        # B = batch size
        # S_trimmed = actual number of frames in the window
        # N = number of tracks
        # C = color channels (3 for RGB)
        # E = positional embedding size
        # LRR = local receptive field radius
        # D = dimension of the transformer input tokens

        assert T >= 1  # A tracker needs at least two frames to track something

        video = 2 * (video / 255.0) - 1.0
        dtype = video.dtype
        queried_frames = queries_uv[:, :, 0].long()

        queried_coords = queries
        queried_coords = queried_coords / self.stride

        # We store our predictions here
        all_coords_predictions, all_vis_predictions, all_confidence_predictions = (
            [],
            [],
            [],
        )
        C_ = C
        H4, W4 = H // self.stride, W // self.stride
        # Compute convolutional features for the video or for the current chunk in case of online mode

        if T > fmaps_chunk_size:
            fmaps = []
            for t in range(0, T, fmaps_chunk_size):
                video_chunk = video[:, t: t + fmaps_chunk_size]
                fmaps_chunk = self.fnet(video_chunk.reshape(-1, C_, H, W))
                T_chunk = video_chunk.shape[1]
                C_chunk, H_chunk, W_chunk = fmaps_chunk.shape[1:]
                fmaps.append(fmaps_chunk.reshape(B, T_chunk, C_chunk, H_chunk, W_chunk))
            fmaps = torch.cat(fmaps, dim=1).reshape(-1, C_chunk, H_chunk, W_chunk)
        else:
            fmaps = self.fnet(video.reshape(-1, C_, H, W))
        fmaps = fmaps.permute(0, 2, 3, 1)
        fmaps = fmaps / torch.sqrt(
            torch.maximum(
                torch.sum(torch.square(fmaps), axis=-1, keepdims=True),
                torch.tensor(1e-12, device=fmaps.device),
            )
        )
        fmaps = fmaps.permute(0, 3, 1, 2).reshape(
            B, -1, self.latent_dim, H // self.stride, W // self.stride
        )
        fmaps = fmaps.to(dtype)

        # We compute track features
        fmaps_pyramid = []
        track_feat_pyramid = []
        track_feat_support_pyramid = []
        fmaps_pyramid.append(fmaps)
        for i in range(self.corr_levels - 1):
            fmaps_ = fmaps.reshape(
                B * T, self.latent_dim, fmaps.shape[-2], fmaps.shape[-1]
            )
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            fmaps = fmaps_.reshape(
                B, T, self.latent_dim, fmaps_.shape[-2], fmaps_.shape[-1]
            )
            fmaps_pyramid.append(fmaps)

        for i in range(self.corr_levels):
            track_feat, track_feat_support = self.get_track_feat(
                fmaps_pyramid[i],
                queried_frames,
                queried_coords / 2 ** i,
                support_radius=self.corr_radius,
            )
            track_feat_pyramid.append(track_feat.repeat(1, T, 1, 1))
            track_feat_support_pyramid.append(track_feat_support.unsqueeze(1))

        D_coords = 9

        coord_preds, vis_preds, confidence_preds = [], [], []

        vis = torch.zeros((B, T, N), device=device).float()
        confidence = torch.zeros((B, T, N), device=device).float()
        coords = queried_coords.reshape(B, 1, N, 2).expand(B, T, N, 2).float()

        r = 2 * self.corr_radius + 1

        uv_queries = queries_uv[..., 1:]
        uv_coords = uv_queries.reshape(B, 1, N, 3).expand(B, T, N, 3).float()
        for it in range(iters):
            coords = coords.detach()  # B T N 2
            uv_coords = uv_coords.detach()  # B T N 3
            coords_init = coords.contiguous().view(B * T, N, 2)
            corr_embs = []
            for i in range(self.corr_levels):
                corr_feat = self.get_correlation_feat(
                    fmaps_pyramid[i], coords_init / 2 ** i
                )
                track_feat_support = (
                    track_feat_support_pyramid[i]
                    .view(B, 1, r, r, N, self.latent_dim)
                    .squeeze(1)
                    .permute(0, 3, 1, 2, 4)
                )
                corr_volume = torch.einsum(
                    "btnhwc,bnijc->btnhwij", corr_feat, track_feat_support
                )
                corr_emb = self.corr_mlp(corr_volume.reshape(B * T * N, r * r * r * r))
                corr_embs.append(corr_emb)
            corr_embs = torch.cat(corr_embs, dim=-1)
            corr_embs = corr_embs.view(B, T, N, corr_embs.shape[-1])

            transformer_input = [vis[..., None], confidence[..., None], corr_embs]

            rel_coords_forward = coords[:, :-1] - coords[:, 1:]
            rel_coords_backward = coords[:, 1:] - coords[:, :-1]

            rel_coords_forward = torch.nn.functional.pad(
                rel_coords_forward, (0, 0, 0, 0, 0, 1)
            )
            rel_coords_backward = torch.nn.functional.pad(
                rel_coords_backward, (0, 0, 0, 0, 1, 0)
            )
            scale = (
                    torch.tensor(
                        [self.model_resolution[1], self.model_resolution[0]],
                        device=coords.device,
                    )
                    / self.stride
            )
            rel_coords_forward = rel_coords_forward / scale
            rel_coords_backward = rel_coords_backward / scale

            rel_pos_emb_input = posenc(
                torch.cat([rel_coords_forward, rel_coords_backward], dim=-1),
                min_deg=0,
                max_deg=10,
            )  # batch, num_points, num_frames, 84
            transformer_input.append(rel_pos_emb_input)

            x = (
                torch.cat(transformer_input, dim=-1)
                .permute(0, 2, 1, 3)
                .reshape(B * N, T, -1)
            )

            x = x + self.interpolate_time_embed(x, T)
            x = x.view(B, N, T, -1)  # (B N) T D -> B N T D

            delta = self.updateformer(
                x,
                add_space_attn=add_space_attn,
            )

            delta_coords = delta[..., :D_coords].permute(0, 2, 1, 3)
            delta_vis = delta[..., D_coords].permute(0, 2, 1)
            delta_confidence = delta[..., D_coords + 1].permute(0, 2, 1)

            vis = vis + delta_vis
            confidence = confidence + delta_confidence

            reshaped_delta_coords = delta_coords.view(B, T, N, 3, 3).flatten(0, 2)
            # This requires fp32 accuracy
            with torch.cuda.amp.autocast(enabled=False):
                rot_deltas = roma.special_procrustes(reshaped_delta_coords.to(torch.float32)).to(
                    reshaped_delta_coords.dtype)
                uv_coords = torch.bmm(rot_deltas, uv_coords.flatten(0, 2).unsqueeze(2)).squeeze(2)
                uv_coords = uv_coords.view(B, T, N, 3)
                coords = torch.empty(B, T, N, 2, device=device, dtype=uv_coords.dtype)
                for b, uv_coord in enumerate(uv_coords):
                    coords[b] = mapper[b].point.vc.to_ij(uv_coord.clone())

            coord_preds.append(uv_coords.clone())
            vis_preds.append(torch.sigmoid(vis))
            confidence_preds.append(torch.sigmoid(confidence))

        if is_train:
            all_coords_predictions.append(coord_preds)
            all_vis_predictions.append(vis_preds)
            all_confidence_predictions.append(confidence_preds)

        if is_train:
            train_data = (
                all_coords_predictions,
                all_vis_predictions,
                all_confidence_predictions,
                torch.ones_like(vis_preds[-1], device=vis_preds[-1].device),
            )
        else:
            train_data = None

        return coord_preds[-1], vis_preds[-1], confidence_preds[-1], train_data
