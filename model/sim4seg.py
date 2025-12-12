from typing import List
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h

import numpy as np


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):

    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class SimSegMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(SimSegMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", "/path/to/sam_vit_h_4b8939.pth")
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", "/path/to/sam_vit_h_4b8939.pth")
        print('self.vision_pretrained',self.vision_pretrained)
        self.initialize_simseg_modules(self.config)
        print('initialize_sim4seg_modules successfully')

    
    
    def initialize_simseg_modules(self, config):
        self.visual_model = build_sam_vit_h(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class SimSegModel(SimSegMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(SimSegModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class SimSegForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        print('CONFIG:',config)
        print('kwargs',kwargs)
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "/path/to/clip-vit-large-patch14"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
            print('self.ce_loss_weight:',self.ce_loss_weight)
        else:
            config.mm_vision_tower = config.vision_tower
            print("config.mm_vision_tower:",config.mm_vision_tower)
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        print('self.ce_loss_weight:',self.ce_loss_weight)
            
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        self.default_im_start_token_idx = kwargs.pop("default_im_start_token_idx")

        super().__init__(config)

        self.model = SimSegModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        print('SimSegForCausalLM inited')
        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        **kwargs,
    ):
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx

        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
        
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
            dim=1,
        )
        
        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None

        else:
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )

        seg_token_offset = seg_token_offset[offset]

        pred_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_

        sam_mask_shape_list = []

        for i in range(len(resize_list)):
            input_size = resize_list[i]
            original_size = label_list[i].shape
            sam_mask_shape_list.append((input_size, original_size))
        sim_mask_list = self.similarity4seg(
            last_hidden_state,
            seg_token_mask, 
            offset,
            input_ids,
            masks_list,
            sam_mask_shape_list,
            images_clip,
            kwargs['image_paths']
        )

        multimask_output = False
        pred_masks = []
        for i in range(len(pred_embeddings)):
            input_prompt_encoder_mask = sim_mask_list[i].to(torch.bfloat16)
            input_prompt_encoder_mask = input_prompt_encoder_mask.to('cuda')
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=input_prompt_encoder_mask.unsqueeze(0),
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            save_mask_image_out=False
            if save_mask_image_out:
                mask_image_out_file = "./mask_image_out"
                os.makedirs(mask_image_out_file, exist_ok=True)
                pmm=pred_mask[:, 0]
                pmmmmm=pmm.detach().cpu().numpy()[0]
                pred_mask_visual = pmmmmm > 0
                save_p = "{}/{}_mask.jpg".format(
                    mask_image_out_file,kwargs['image_paths'][i].split("/")[-1].split(".")[0], 
                )
                cv2.imwrite(save_p, pred_mask_visual * 100)
                save_p = "{}/{}_masked_img.jpg".format(
                    mask_image_out_file,kwargs['image_paths'][i].split("/")[-1].split(".")[0],
                )
                i_np = cv2.imread(kwargs['image_paths'][i])
                i_np = cv2.cvtColor(i_np, cv2.COLOR_BGR2RGB)
                s_img = i_np.copy()
                s_img[pred_mask_visual] = (
                    i_np * 0.5
                    + pred_mask_visual[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
                )[pred_mask_visual]
                s_img = cv2.cvtColor(s_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_p, s_img)
            pred_masks.append(pred_mask[:, 0])

        model_output = output
        gt_masks = masks_list

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        output = model_output.logits

        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]
            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        loss = ce_loss + mask_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }
    

    def similarity4seg(
            self, 
            output_hidden_states, 
            seg_token_mask,
            offset,
            input_ids, 
            masks_list,
            sam_mask_shape_list,
            images_clip,
            image_path
    ):
        def get_similarity_map(sm, shape):
            sm = (sm - sm.min(1, keepdim=True)[0]) / (sm.max(1, keepdim=True)[0] - sm.min(1, keepdim=True)[0])
            side = int(sm.shape[1] ** 0.5)
            sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)
            sm = sm.to(torch.float32)

            target_size = 336
            h, w = shape
            scale = target_size / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            sm = torch.nn.functional.interpolate(sm, (target_size, target_size), mode='bilinear')
            pad_h = (new_h - target_size) // 2
            pad_w = (new_w - target_size) // 2
            padded_sm = F.pad(sm, (pad_w, pad_w, pad_h, pad_h))
            sm = torch.nn.functional.interpolate(padded_sm, shape, mode='bilinear')
            sm = sm.permute(0, 2, 3, 1)
            return sm
        
        def compute_similarity_map(
            image_features, 
            text_features, 
            redundant_feats=None
        ):  
            if redundant_feats != None:
                similarity = image_features @ (text_features - redundant_feats).t()
            else:
                image_features = image_features.clone()
                text_features = text_features.clone()
                prob = image_features[:, :1, :] @ text_features.t()
                prob = (prob * 2).softmax(-1)
                w = prob / prob.mean(-1, keepdim=True)
                b, n_t, n_i, c = image_features.shape[0], text_features.shape[0], \
                    image_features.shape[1], image_features.shape[2]
                feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(1, 1, n_t, c)
                feats *= w.reshape(1, 1, n_t, 1)
                similarity = feats.sum(-1)
            return similarity
        
        images_size_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            images_size_list.extend([sam_mask_shape_list[i][1]] * (end_i - start_i))

        seg_token_counts = seg_token_mask.int().sum(-1)
        image_embedding_tokens = output_hidden_states[seg_token_counts==1]
        seg_embedding_tokens = output_hidden_states[seg_token_mask]

        similarity_mask_list = []
        for bs in range(len(image_embedding_tokens)):
            default_im_start_token_idx = torch.where(
                input_ids==self.default_im_start_token_idx
            )[1][0].item()

            similarity = compute_similarity_map(
                image_embedding_tokens [ 
                    bs: bs+1, 
                    default_im_start_token_idx + 1: default_im_start_token_idx + 1 \
                    + self.get_vision_tower().num_patches, :
                ],
                seg_embedding_tokens[bs: bs + 1, ...]
            )

            case_study_sim4mask=True
            if case_study_sim4mask:
                case_study_file = "./case_study"
                os.makedirs(case_study_file, exist_ok=True)
                images_list = []
                for i in range(len(offset) - 1):
                    cv2_img = cv2.imread(image_path[i])
                    start_i, end_i = offset[i], offset[i + 1]
                    images_list.extend([cv2_img] * (end_i - start_i))
                blk_similarity = get_similarity_map(similarity, images_size_list[bs])
                sim_map = blk_similarity[0, ..., 0].detach().cpu().numpy()
                h, w = sim_map.shape

                grid_rows, grid_cols = 16, 16
                cell_height = h // grid_rows
                cell_width = w // grid_cols

                block_avgs = []

                for i in range(grid_rows):
                    for j in range(grid_cols):
                        y1 = i * cell_height
                        y2 = (i + 1) * cell_height
                        x1 = j * cell_width
                        x2 = (j + 1) * cell_width
        
                        cell = sim_map[y1:y2, x1:x2]
                        avg_val = np.mean(cell)
                        block_avgs.append((avg_val, i, j))


                block_avgs.sort(key=lambda x: x[0], reverse=True)
                top_blocks = block_avgs[:36]

                block_avg_map = np.zeros_like(sim_map)
                for i in range(grid_rows):
                    for j in range(grid_cols):
                        y1 = i * cell_height
                        y2 = (i + 1) * cell_height
                        x1 = j * cell_width
                        x2 = (j + 1) * cell_width

                        avg_val = next((val for val, r, c in block_avgs if r == i and c == j), 0)
                        block_avg_map[y1:y2, x1:x2] = avg_val

                mask = np.zeros((h, w), dtype=np.uint8)
                
                prompt_msk = torch.zeros((256, 256), dtype=torch.float32)
                for val, i, j in top_blocks:
                    y1 = i * cell_height
                    y2 = (i + 1) * cell_height
                    x1 = j * cell_width
                    x2 = (j + 1) * cell_width
                    mask[y1:y2, x1:x2] = 255
                    yy1 = i * 16
                    yy2 = (i + 1) * 16
                    xx1 = j * 16
                    xx2 = (j + 1) * 16
                    prompt_msk[yy1:yy2, xx1:xx2] = 1.0
                masks1 = prompt_msk

                block_avg_vis = (block_avg_map * 255).astype('uint8')
                block_avg_vis = cv2.applyColorMap(block_avg_vis, cv2.COLORMAP_JET)
                orig_img = images_list[bs]
                overlay = cv2.addWeighted(orig_img, 0.3, block_avg_vis, 0.7, 0)
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

                save_case_mask=False
                if save_case_mask:
                    image_name = os.path.splitext(os.path.basename(image_path[bs]))[0]

                    block_avg_path = f"{case_study_file}/{image_name}_block_avg.jpg"
                    cv2.imwrite(block_avg_path, overlay)

                    mask_path = f"{case_study_file}/{image_name}_sim4mask.jpg"
                    cv2.imwrite(mask_path, mask)

                similarity_mask_list.append(masks1)
                

            case_study=False
            if case_study:
                case_study_file = "./case_study"
                os.makedirs(case_study_file, exist_ok=True)
                images_list = []
                for i in range(len(offset) - 1):
                    cv2_img = cv2.imread(image_path[i])
                    start_i, end_i = offset[i], offset[i + 1]
                    images_list.extend([cv2_img] * (end_i - start_i))

                vis_similarity_map = get_similarity_map(similarity, images_size_list[bs])
                sim_map = vis_similarity_map[0, ..., 0].detach().cpu().numpy()
                inverted_sim = 1 - sim_map
                vis = (inverted_sim * 255).astype('uint8')
                vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                vis = images_list[bs] * 0.3 + vis * 0.7
                vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)

                vis_similarity_map_save_path = "{}/{}_sim.jpg".format(
                            case_study_file,image_path[bs].split("/")[-1].split(".")[0]
                        )
                cv2.imwrite(vis_similarity_map_save_path, vis)

        return similarity_mask_list

    
    def similarity_map_to_mask(self, sm, shape,n_grid_size=16, top_n_cell=36):
        def get_similarity_map(sm, shape):
            sm = (sm - sm.min(1, keepdim=True)[0]) / (sm.max(1, keepdim=True)[0] - sm.min(1, keepdim=True)[0])
            side = int(sm.shape[1] ** 0.5)
            sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)
            sm = sm.to(torch.float32)

            target_size = 336
            h, w = shape
            scale = target_size / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            sm = torch.nn.functional.interpolate(sm, (target_size, target_size), mode='bilinear')
            pad_h = (new_h - target_size) // 2
            pad_w = (new_w - target_size) // 2
            padded_sm = F.pad(sm, (pad_w, pad_w, pad_h, pad_h))
            sm = torch.nn.functional.interpolate(padded_sm, shape, mode='bilinear')
            sm = sm.permute(0, 2, 3, 1)
            return sm

        origin_sm = get_similarity_map(sm[None, ..., None], shape)
        grid_size = n_grid_size
        h, w = origin_sm.shape[1:3]
        cell_height = h // grid_size
        cell_width = w // grid_size
    
        avg_similarities = []
        for i in range(grid_size):
            for j in range(grid_size):
                cell = origin_sm[0, i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width,:]
                avg_similarity = cell.mean().item()
                avg_similarities.append((avg_similarity, i, j))

        top_cells = sorted(avg_similarities, key=lambda x: x[0], reverse=True)[:top_n_cell]
        mask = torch.zeros((256, 256), dtype=torch.float32)

        for _, i, j in top_cells:
            mask[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width] = 1.0

        return mask


    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences
            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
                    seg_token_mask,
                ],
                dim=1,
            )
            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]
            seg_token_counts = seg_token_mask.int().sum(-1)
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_
            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks
