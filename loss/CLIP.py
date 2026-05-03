import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    def __init__(self, args, dim):
        super().__init__()
        self.rank = args.rank
        self.prev_num_logits = 0
        self.labels = {}
        self.gamma_1=args.gamma_1
        self.gamma_2=args.gamma_2
        self.lambda_1=args.lambda_1
        self.lambda_2=args.lambda_2
        self.sigma=args.sigma

    def forward(self, text_features, image_features, local,logit_scale=2.659):
        lambda_1 =self.lambda_1
        lambda_2= self.lambda_2
        sigma =self.sigma
        gamma_1 = self.gamma_1
        gamma_2 = self.gamma_2

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        device = image_features.device
        sims = image_features @ text_features.T

        num_logits = sims.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
        else:
            labels = self.labels[device]

        logits_per_image = logit_scale * sims
        logits_per_text = logit_scale * sims.T
        local1 = local
        local2 = local.T

        resize_label = labels

        # Calculate the sample-wise global contrastive loss
        loss_i2t = F.cross_entropy(logits_per_image, resize_label, reduction='none')
        loss_t2i = F.cross_entropy(logits_per_text, resize_label, reduction='none')
        total_loss = loss_i2t + loss_t2i

        # Calculate the sample-wise local contrastive loss
        loss_local1 = F.cross_entropy(local1, resize_label, reduction='none')
        loss_local2 = F.cross_entropy(local2, resize_label, reduction='none')
        localloss2 = loss_local1 + loss_local2


        term1 = total_loss + localloss2

        mask1 = term1 < gamma_1
        weight = (torch.cos((torch.pi / 2) * (term1 / gamma_1)) * mask1.float()).detach()
        regularization_term = - (2/torch.pi) * gamma_1 * (weight* torch.arccos(weight) - torch.sqrt(1 - weight**2))
        regularization_term=regularization_term * mask1.float().detach()
        term_1_per_sample = weight * term1 + regularization_term

        mask2 = (term1 >= gamma_1) & (term1 < gamma_2)
        weight2=(torch.cos((torch.pi / 2) * (term1 / gamma_2)) * mask2.float()).detach()
        regularization_term2=- (2/torch.pi) * gamma_2 * (weight2* torch.arccos(weight2) - torch.sqrt(1 - weight2**2))
        regularization_term2 = regularization_term2 * mask2.float().detach()
        term_2_per_sample = (weight2 * term1 + regularization_term2)

        mask3 = term1 >= gamma_2
        term_3_per_sample = self._soft_triplet_loss(sims, sigma, mask3)

        # Combine all stages
        term = term_1_per_sample + lambda_1 * term_2_per_sample + lambda_2 * term_3_per_sample
        total = term.mean()

        return total
    
    def _soft_triplet_loss(self, sims, base_sigma, mask):
        batch_size = sims.shape[0]
        device = sims.device
        eye_mask = torch.eye(batch_size, device=device).bool()
        pos_sim = torch.diag(sims)
        sims_i2t = sims.masked_fill(eye_mask, -float('inf'))
        hardest_neg_text_sim, hardest_neg_text_idx = torch.max(sims_i2t, dim=1)
        sims_t2i = sims.masked_fill(eye_mask, -float('inf'))
        hardest_neg_img_sim, hardest_neg_img_idx = torch.max(sims_t2i, dim=0)
        difficulty_i2t = pos_sim - hardest_neg_text_sim
        difficulty_t2i = pos_sim - hardest_neg_img_sim
        sigma_i2t = base_sigma * (1 + F.relu(-difficulty_i2t))
        sigma_t2i = base_sigma * (1 + F.relu(-difficulty_t2i))
        loss1 = F.relu(sigma_i2t - pos_sim + hardest_neg_text_sim)
        loss2 = F.relu(sigma_t2i - pos_sim + hardest_neg_img_sim)
        soft_triplet_loss = (loss1 + loss2) * mask.float()
        return soft_triplet_loss
