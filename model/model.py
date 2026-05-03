import random
from contextlib import suppress

import open_clip

from loss import NEED_LOGIT_SCALE, NEED_PROTOTYPE_LAYER
from model.clip_model import *
from model.simple_tokenizer import *
from training.distributed import is_master
from training.projection import DINOHead

AVALIABLE_TEXT_MODEL_BUILDER = ['openclip', 'chineseclip', 'huggingface', 'sbert']
AVALIABLE_IMAGE_MODEL_BUILDER = ['openclip', 'chineseclip', 'torchvision', "torchhub"]


def get_model(args):
    logging.info(f'Builing model for rank {args.rank}')

    # === text model === #
    if is_master(args):
        logging.info(
            f'Loading [{args.text_model}] as text model via [{args.text_model_builder}]. Pretrained={args.pretrained_text_model}')

    if args.text_model_builder == 'openclip':
        CLIP_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name=args.text_model,
            pretrained=args.text_model_tag if args.pretrained_text_model else '',
            precision=args.precision,
            device=args.device,
            jit=args.torchscript,
            force_quick_gelu=args.force_quick_gelu,
            cache_dir=os.path.join(args.cache_dir, 'open_clip')
        )
        CLIP_model.visual = None
        text_backbone = CLIP_model
        tokenizer = open_clip.tokenize
        args.text_width, args.text_dim = text_backbone.text_projection.size()
        text_backbone.layers = open_clip.get_model_config(args.text_model)['text_cfg']['layers']

        if args.adapter is not None:
            raise RuntimeError(f'Adapter {args.adapter} is not avaliable for {args.text_model_builder} models!')


    else:
        raise RuntimeError(f'text model builder "{args.text_model_builder}" is not supported.')

    # === image model === #
    if is_master(args):
        logging.info(
            f'Loading [{args.image_model}] as image model via [{args.image_model_builder}]. Pretrained={args.pretrained_image_model}')

    if args.image_model_builder == 'openclip':
        CLIP_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name=args.image_model,
            pretrained=args.image_model_tag if args.pretrained_image_model else '',
            precision=args.precision,
            device=args.device,
            jit=args.torchscript,
            force_quick_gelu=args.force_quick_gelu,
            cache_dir=os.path.join(args.cache_dir, 'open_clip')
        )
        image_backbone = CLIP_model.visual
        args.image_dim = image_backbone.output_dim
        image_backbone.layers = open_clip.get_model_config(args.image_model)['vision_cfg']['layers']
        if type(image_backbone.layers) == list:
            image_backbone.layers = len(image_backbone.layers)
        if 'RN' in args.image_model:
            image_backbone.arch = 'ResNet'
            image_backbone.layers += 2  # stem and attention pooling accont for two layers
        elif 'ViT' in args.image_model:
            image_backbone.arch = 'ViT'
        else:
            raise RuntimeError(f'Unrecognized image backbone architechture')

    else:
        raise RuntimeError(f'image model builder "{args.image_model_builder}" is not supported.')

    # Set 'param.required_grad' to implement partial finetune
    for name, param in text_backbone.named_parameters():
        param.requires_grad = False if args.lock_text_model else True
        if args.lock_text_partial != '':
            for keyword in args.lock_text_partial.split(','):
                if keyword.replace('!', '') in name:
                    if '!' in keyword:
                        param.requires_grad = True
                        if args.lock_text_model:
                            break
                    else:
                        param.requires_grad = False
                        if not args.lock_text_model:
                            break

    for name, param in image_backbone.named_parameters():
        param.requires_grad = False if args.lock_image_model else True
        if args.lock_image_partial != '':
            for keyword in args.lock_image_partial.split(','):
                if keyword.replace('!', '') in name:
                    if '!' in keyword:
                        param.requires_grad = True
                        if args.lock_image_model:
                            break
                    else:
                        param.requires_grad = False
                        if not args.lock_image_model:
                            break

    model = ItraModel(
        text_backbone=text_backbone,
        image_backbone=image_backbone,
        tokenizer=tokenizer,
        args=args
    )

    return model, preprocess_train, preprocess_val, preprocess_val


class ItraModel(nn.Module):
    def __init__(self, text_backbone, image_backbone, tokenizer, args) -> None:
        super().__init__()
        self.device = args.device
        self.text_model = args.text_model
        self.queuesize = 200000
        self.register_buffer("queuesim", torch.randn(self.queuesize))
       
        self.tokenizermlm = SimpleTokenizer(bpe_path=r"/xiaolin/xyq/MyMethod/RRSITR/clip_benchmark/bpe_simple_vocab_16e6.txt.gz")
        self.vocabsize = 49408
        self.embed_dim = 512
        self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                self.embed_dim // 64,
                                                batch_first=True)
        self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                   layers=2,
                                                   heads=self.embed_dim //
                                                         64)
        scale = self.cross_modal_transformer.width ** -0.5

        self.ln_pre_t = LayerNorm(self.embed_dim)
        self.ln_pre_i = LayerNorm(self.embed_dim)
        self.ln_post = LayerNorm(self.embed_dim)

        proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
        attn_std = scale
        fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
        for block in self.cross_modal_transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        
        nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

        self.mlm_head = nn.Sequential(
            OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                         ('gelu', QuickGELU()),
                         ('ln', LayerNorm(self.embed_dim)),
                         ('fc', nn.Linear(self.embed_dim, self.vocabsize))]))  # 这里是vocabsize
        
        nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
        nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

        # text backbone
        self.text_backbone = text_backbone
        self.text_pooler = args.text_pooler
        if self.text_pooler != 'cls':
            self.text_backbone.pooler = nn.Identity()
        self.text_dim = args.text_dim
        self.text_width = args.text_dim
        self.tokenizer1 = tokenizer
        self.tokenizer = SimpleTokenizer(bpe_path=r"/xiaolin/xyq/MyMethod/RRSITR/clip_benchmark/bpe_simple_vocab_16e6.txt.gz")
        self.text_model_builder = args.text_model_builder
        self.image_model_builder = args.image_model_builder
        self.max_seq_length = args.max_seq_length

        self.image_context = torch.no_grad if (
                args.lock_image_model and
                '!' not in args.lock_image_partial
        ) else suppress

        self.text_context = torch.no_grad if (
                args.lock_text_model and
                '!' not in args.lock_text_partial and
                args.adapter is None and
                not args.prompt
        ) else suppress

        if is_master(args):
            logging.info(f'Calculate gradients for image backbone?\t{self.image_context == suppress}')
            logging.info(f'Calculate gradients for text backbone?\t{self.text_context == suppress}')

        # TODO: CoOp text prompt
        if args.prompt:
            assert args.text_model_builder == 'openclip'  # CoOp style prompt only supports OpenCLIP models
            self.prompt = nn.Parameter(torch.empty(args.n_prompt, args.text_width))
            torch.nn.init.normal_(self.prompt, std=0.02)
            self.n_prompt = args.n_prompt
        else:
            self.prompt = None

        # image backbone
        self.image_backbone = image_backbone
        self.image_dim = image_backbone.output_dim
        self.image_model_tag = args.image_model_tag

        # text projection head
        if args.text_head_n_layers > 0 or args.loss in NEED_PROTOTYPE_LAYER:
            if args.image_head_n_layers == 0 and args.joint_projection_dim < 0:
                args.joint_projection_dim = self.image_dim  # adaption layer
            self.text_projection_head = DINOHead(
                in_dim=self.text_dim, out_dim=65536, bottleneck_dim=args.joint_projection_dim,
                nlayers=args.text_head_n_layers, skip_last_layer=args.loss not in NEED_PROTOTYPE_LAYER
            ).to(args.device)

            # DINO & ProtoCPC copy student's learnable prototype to teacher, so teacher's prototype should not be optimized
            if args.loss in NEED_PROTOTYPE_LAYER and args.teacher == 'text':
                for param in self.text_projection_head.parameters():
                    param.requires_grad = False
        else:
            self.text_projection_head = nn.Identity()
            if is_master(args):
                logging.info(
                    'Text backbone do not append projection head, so set args.joint_projection_dim = self.text_dim')
            args.joint_projection_dim = self.text_dim

        # image projection head
        if args.image_head_n_layers > 0 or args.loss in NEED_PROTOTYPE_LAYER:
            if args.text_head_n_layers == 0 and args.joint_projection_dim < 0:
                args.joint_projection_dim = self.text_dim  # adaption layer
            self.image_projection_head = DINOHead(
                in_dim=self.image_dim, out_dim=65536, bottleneck_dim=args.joint_projection_dim,
                nlayers=args.image_head_n_layers, skip_last_layer=args.loss not in NEED_PROTOTYPE_LAYER
            ).to(args.device)
            # FIXME? # DINO & ProtoCPC copy student's learnable prototype to teacher, so teacher's prototype should not be optimized
            if args.loss in NEED_PROTOTYPE_LAYER and args.teacher == 'image':
                for param in self.image_projection_head.parameters():
                    param.requires_grad = False
        else:
            self.image_projection_head = nn.Identity()
            if is_master(args):
                logging.info(
                    'Image backbone do not append projection head so set args.joint_projection_dim = self.image_dim')
            args.joint_projection_dim = self.image_dim

        if args.loss in NEED_LOGIT_SCALE:
            if hasattr(self.text_backbone, 'logit_scale'):
                self.logit_scale = self.text_backbone.logit_scale
                self.text_backbone.logit_scale = None
            else:
                self.logit_scale = torch.autograd.Variable(torch.ones(1) * np.log(1 / args.logit_scale)).to(self.device)
            self.logit_scale = nn.Parameter(self.logit_scale)
            self.logit_scale.requires_grad = True
        else:
            self.logit_scale = torch.zeros(1)
        self.to(self.device)



    def reinit_logit_scale(self, logit_scale):
        self.logit_scale = torch.nn.parameter.Parameter(torch.ones(1) * np.log(1 / logit_scale))  # .to(self.device)
        # self.logit_scale.to(self.device)
        self.to(self.device)

    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q),
            self.ln_pre_i(k),
            self.ln_pre_i(v),
            need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, images, projection=False):
        with self.image_context():
            def _expand_token(token, batch_size: int):
                return token.view(1, 1, -1).expand(batch_size, -1, -1)

            def open_clip_forwardim(image):
                x = self.image_backbone.conv1(image)
                x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
                x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
                x = torch.cat([_expand_token(self.image_backbone.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
                # shape = [*, grid ** 2 + 1, width]
                x = x + self.image_backbone.positional_embedding.to(x.dtype)

                x = self.image_backbone.patch_dropout(x)
                x = self.image_backbone.ln_pre(x)

                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.image_backbone.transformer(x)
                x = x.permute(1, 0, 2)  # LND -> NLD
                x = self.image_backbone.ln_post(x)
                pooled, tokens = self.image_backbone._global_pool(x)
                pooled = pooled @ self.image_backbone.proj
                tokens = tokens @ self.image_backbone.proj
                return pooled, tokens

            image_features, tokens = open_clip_forwardim(images)
        if projection:
            image_features = self.image_projection_head(image_features)
            tokens = self.image_projection_head(tokens)
        return image_features.float(), tokens.float()

    def encode_text(self, texts, projection=False, use_pooler=True):
        with self.text_context():
            if self.text_model_builder in ['openclip']:
                # TODO: support CoOp-style prompting (CoOp for retrieval finetuning?)
                context_length = (77 - self.n_prompt) if self.prompt is not None else 77
                text1 = self.tokenizer1(texts, context_length=context_length).to(self.device)



                def open_clip_forward(texts):
                    x = self.text_backbone.token_embedding(
                        texts)  # [batch_size, n_ctx, d_model] (bs, 77-args.n_prompts, 512)
                    if self.prompt is not None:
                        batch_prompt = self.prompt.unsqueeze(0).expand(x.size(0), -1, -1)
                        x = torch.cat([x[:, :1, :], batch_prompt, x[:, 1:, :]], dim=1)
                    x = x + self.text_backbone.positional_embedding
                    x = x.permute(1, 0, 2)  # NLD -> LND
                    x = self.text_backbone.transformer(x, attn_mask=self.text_backbone.attn_mask)
                    x = x.permute(1, 0, 2)  # LND -> NLD
                    x = self.text_backbone.ln_final(x)  # [batch_size, n_ctx, transformer.width]

                    tokens = x

                    x = x[torch.arange(x.shape[0]), texts.argmax(dim=-1)] @ self.text_backbone.text_projection
                    tokens = tokens @ self.text_backbone.text_projection
                    return x, tokens

                text_features, text_tokens = open_clip_forward(text1)


        if projection:
            text_features = self.text_projection_head(text_features)
            text_tokens = self.text_projection_head(text_tokens)

        return text_features, text_tokens
    def compute_relation(self, text, image):
        text = F.normalize(text, dim=-1)
        image = F.normalize(image, dim=-1)
        if self.training:
            local1 = np.exp(2.996) * torch.einsum('abc,dec->adbe', [text, image])
        else:
            local1 = torch.einsum('abc,dec->adbe', [text, image])
        local2 = local1.permute(0, 1, 3, 2)

        # local2 = torch.max(local2,dim=3)[0]
        local2 = torch.norm(local2, p=2, dim=-1)
        local2 = torch.norm(local2, p=2, dim=-1)

        return local2

    def _build_attribute_masked_tokens_and_labels(self, tokens, attribute_position):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizermlm.encoder["<|mask|>"]

        labels = torch.zeros(len(tokens), dtype=torch.long)
        for i, position in enumerate(attribute_position):
            labels[position] = tokens[position]
            tokens[position] = mask

        return torch.tensor(tokens), torch.tensor(labels)

    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizermlm.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizermlm.encoder) - 3))  # 1 ~ 49405

        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)

        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)

    def forward(self, images, texts, text_only):
        """
        images: torch.tensor (batchs_size, preprocessed image)
        texts:  torch.tensor (batchs_size, token_indexs)
        """

        text_features, text_tokens= self.encode_text(texts, projection=True)

        if text_only:  # skip image forward for efficient teacher caching
            image_features = text_features
        else:
            image_features, image_tokens = self.encode_image(images, projection=True)

        local2 = self.compute_relation(text_tokens, image_tokens)

        return image_features, text_features, self.logit_scale.exp(), text_tokens, image_tokens, local2


def mean_pooling(hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    return torch.sum(hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


from torch import Tensor


def cos_similar(p: Tensor, q: Tensor):
    # sim_matrix = p.matmul(q.transpose(-2, -1))
    p = l2norm(p, dim=-1)
    q = l2norm(q, dim=-1)
    sim_matrix = torch.einsum('abc,dec->adbe', [p, q])
    sim_matrix = torch.where(torch.isnan(sim_matrix), torch.full_like(sim_matrix, 0), sim_matrix)
    return sim_matrix


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X
