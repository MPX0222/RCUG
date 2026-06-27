import open_clip
from open_clip.tokenizer import HFTokenizer
from random import sample
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.CLIP_Base_Net import CLIP_Base_Net
import copy
from model.backbone.clip.clip import load, tokenize
from model.backbone.MedCLIP.model import MedCLIPModel, MedCLIPVisionModelViT
from model.backbone.Adapter import Adapter
from utils.functions import *


class CLIP_task_adapter_Net(CLIP_Base_Net):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.prompt_length = config.prompt_length
        self.inc = config.increment_steps[0]
        self.img_adapter_list = nn.ModuleList([])
        self.img_final_adapter = nn.ModuleList([])
        # self.img_final_adapter_list = nn.ModuleList([])
        self.text_adapter_list = None
        self.text_final_adapter = None
        self.use_ood = False

    def model_init(self):
        if self.config.backbone == "CLIP":
            self.backbone, _ = load(self.config.pretrained_path, jit=False)
        elif self.config.backbone == "OpenCLIP":
            self.backbone, _ = open_clip.create_model_from_pretrained("ViT-B-16", pretrained=self.config.pretrained_path+"/open_clip_pytorch_model.bin")
            # self.backbone = open_clip.create_model("ViT-B-16-SigLIP")
            # self.backbone.load_state_dict(
            #     torch.load(self.config.pretrained_path + "/open_clip_pytorch_model.bin")
            # )
            self.backbone.float()
            self.backbone.vision_layers = 12
            self.backbone.vision_width = 768
            self.backbone.transformer_layers = 12
            self.backbone.transformer_width = 512
        elif self.config.backbone == "MedCLIP":
            self.backbone = MedCLIPModel(MedCLIPVisionModelViT)
            self.backbone.from_pretrained(self.config.pretrained_path)
        self.output_dim = 512
        # self.output_dim = 512
        self.logger.info("model loaded!")

        for i in range(3):
            self.img_final_adapter.append(Adapter(d_model=self.output_dim,
                                                  dropout=0,
                                                  # nonlinear=nn.Identity,
                                                  bottleneck=64,
                                                  is_ss=False,
                                                  init_option="lora",
                                                  adapter_scalar="0.1",
                                                  adapter_layernorm_option=None))
        self.img_router = nn.Parameter(torch.zeros(self.output_dim, 3))
        self.img_noise_w = nn.Parameter(torch.zeros(self.output_dim, 3))

        self.cur_img_adapter = None
        # self.img_final_adapter = Adapter(d_model=self.output_dim,
        #                                 dropout=0,
        #                                 # nonlinear=nn.Identity,
        #                                 bottleneck=64,
        #                                 is_ss=False,
        #                                 init_option="lora",
        #                                 adapter_scalar="0.1",
        #                                 adapter_layernorm_option=None)

        for module in self.img_final_adapter.modules():
            if type(module) == nn.Linear:
                nn.init.kaiming_uniform_(module.weight)

        # self.img_final_adapter = nn.Sequential(
        #     nn.Linear(512, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 512),
        # )

        # self.text_final_adapter = Adapter(d_model=512,
        #                                  dropout=0,
        #                                  bottleneck=64,
        #                                  is_ss=False,
        #                                  init_option="lora",
        #                                  adapter_scalar="0.1",
        #                                  adapter_layernorm_option=None)

        # if self.use_ood:
        #     self.ood_text = tokenize(["a photo of a ood class"]).cuda()
        #     self.ood_prompt = tensor_prompt(1, 6, 512)
        if self.prompt_length > 0:
            self.text_prompts = tensor_prompt(sum(self.config.increment_steps), self.prompt_length, 512, ortho=False)

    def update_model(self, task_id):
        new_classes = self.config.increment_steps[task_id]
        known_classes = sum(self.config.increment_steps[:task_id])
        cur_classes = new_classes + known_classes
        if task_id > 0:
            self.img_adapter_list.append(copy.deepcopy(self.cur_img_adapter))
            for param in self.img_adapter_list.parameters():
                param.requires_grad = False
            self.cur_img_adapter = nn.ModuleList([])
            for i in range(self.backbone.vision_layers):  # self.backbone.vision_layers
                img_adapter = Adapter(d_model=self.backbone.vision_width,
                                      dropout=0.1,
                                      bottleneck=64,
                                      init_option="lora",
                                      adapter_scalar="0.1",
                                      adapter_layernorm_option=None)
                self.cur_img_adapter.append(img_adapter)
        else:
            self.cur_img_adapter = nn.ModuleList([])
            for i in range(self.backbone.vision_layers):  # self.backbone.vision_layers
                img_adapter = Adapter(d_model=self.backbone.vision_width,
                                      dropout=0.1,
                                      bottleneck=64,
                                      init_option="lora",
                                      adapter_scalar="0.1",
                                      adapter_layernorm_option=None)
                self.cur_img_adapter.append(img_adapter)

        # for module in self.img_final_adapter.modules():
        #     if type(module) == nn.Linear:
        #         nn.init.kaiming_uniform_(module.weight)

    def moe_adapter(self, x, train=True):
        clean_logits = x @ self.img_router
        if train:
            raw_noise_stddev = x @ self.img_noise_w
            noise_stddev = F.softplus(raw_noise_stddev) + 1e-2
            logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        else:
            logits = clean_logits

        logits, indices = torch.topk(logits, k=3, dim=-1)
        mask = F.softmax(logits, dim=-1)

        all_out = torch.stack([adapter(x, add_residual=False) for adapter in self.img_final_adapter], dim=1)
        selected_out = torch.gather(all_out, dim=1, index=indices.unsqueeze(-1).expand(-1, -1, all_out.shape[-1]))
        adapt_out = (selected_out * mask.unsqueeze(-1)).sum(dim=1)
        out = adapt_out+x

        # out = self.img_final_adapter(x)

        return out

    def forward_train(self, image, text_features_normed, task_id=None):
        class_num = text_features_normed.shape[0]
        B = image.shape[0]
        image_features = self.backbone.encode_image(image, adapter_list=self.cur_img_adapter)
        image_features = self.moe_adapter(image_features, train=True)
        image_features_normed = image_features / image_features.norm(dim=-1, keepdim=True)

        logits_global = self.backbone.logit_scale.exp() * image_features_normed @ text_features_normed.t()
        logits_global = logits_global.softmax(dim=-1)

        ood_logits = None
        # if task_id > 0:
        #     with torch.no_grad():
        #         ood_features = self.get_ood_vectors(image, task_id)
        #         sampled_ood_features = ood_features[torch.randint(len(ood_features), (B,))]
        #         ood_inputs = torch.cat([image_features, sampled_ood_features], dim=0)
        #     # ood_inputs = self.moe_adapter(ood_inputs, train=True)
        #     ood_logits = self.ood_detector(ood_inputs)

        return {"logits": logits_global, "ood_logits": ood_logits, "features": image_features}

    def mahalanobis_distance(self, X, mu, Sigma):
        # 正则化协方差矩阵
        Sigma_reg = Sigma  # 添加正则化项
        Sigma_inv = torch.inverse(Sigma_reg)  # (n, d, d)
        # 计算 (X - μ) 的差值
        X_mu = X - mu.unsqueeze(0)  # (b, n, d)
        # 计算马氏距离
        distances = torch.sqrt(torch.einsum("bnj,bnj->bn", torch.einsum('bni,nij->bnj', X_mu, Sigma_inv), X_mu))  # (b, n)

        return distances

    def feature_aggregate(self, img_fe, text_features_normed, labels=None):
        image_features, image_features1 = img_fe
        # text_features_normed = text_features / text_features.norm(dim=-1, keepdim=True)
        # image_features1 = self.moe_adapter(image_features, train=False)
        # image_features_normed = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features_normed1 = image_features1 / image_features1.norm(dim=-1, keepdim=True)
        # all_logits_global = self.backbone.logit_scale.exp() * image_features_normed @ text_features_normed.transpose(-2, -1)
        all_logits_global1 = self.backbone.logit_scale.exp() * image_features_normed1 @ text_features_normed.transpose(-2, -1)
        # all_logits_global_unsm = all_logits_global1
        # all_logits_local = self.backbone.logit_scale.exp() * image_features_normed.unsqueeze(-2) @ text_features_normed.view(image_features_normed.shape[1], -1, 512).transpose(-2, -1)
        # all_logits_local = all_logits_local.squeeze(-2).view(image_features_normed.shape[0], -1)

        # all_logits_global_softmax = all_logits_global.softmax(dim=-1)
        # entropy = torch.sum(-all_logits_global_softmax * torch.log(all_logits_global_softmax), dim=-1)

        all_logits_global1_softmax = all_logits_global1.softmax(dim=-1)
        entropy1 = torch.sum(-all_logits_global1_softmax * torch.log(all_logits_global1_softmax), dim=-1)
        delta_entropy = None
        # s = torch.zeros_like(delta_entropy, device=delta_entropy.device)
        # s[delta_entropy <= 0] = -100
        # neg = torch.where(delta_entropy>=0, torch.zeros_like(delta_entropy), delta_entropy)
        # s = torch.where(neg < 0, torch.min(neg,dim=-1, keepdim=True)[0]-neg, torch.ones_like(delta_entropy))
        w = entropy1

        idx = torch.min(w, dim=-1, keepdim=True)[1]
        # true_idx = torch.div(labels, self.config.increment_steps[0], rounding_mode="floor").unsqueeze(-1)

        logits = torch.gather(all_logits_global1, dim=1,
                                index=idx.unsqueeze(-1).expand(-1, -1, all_logits_global1.shape[-1])).squeeze()

        return logits, idx, (delta_entropy, entropy1)

    def logit_aggregate(self, all_logits_global, all_proto_logits=None, labels=None):
        flag = "entropy"
        if flag == "entropy":
            all_logits_global = all_logits_global.softmax(dim=-1)
            entropy = torch.sum(-all_logits_global * torch.log(all_logits_global), dim=-1)
            w = (torch.max(entropy, dim=-1, keepdim=True)[0] - entropy).softmax(dim=-1)

            # entropy weighted
            # logits = (w.unsqueeze(-1).expand(-1, -1, all_logits_global.shape[-1])*all_logits_global).sum(dim=1)

            # entropy top1
            idx = torch.max(w, dim=-1, keepdim=True)[1]
            # idx = torch.div(labels, 20, rounding_mode="floor").unsqueeze(-1)
            logits = torch.gather(all_logits_global, dim=1,
                                  index=idx.unsqueeze(-1).expand(-1, -1, all_logits_global.shape[-1])).squeeze()
            if all_proto_logits is not None:
                all_proto_logits = all_proto_logits.softmax(dim=-1)
                proto_logits = torch.gather(all_proto_logits, dim=1, index=idx.unsqueeze(-1).expand(-1, -1,
                                                                                                    all_proto_logits.shape[
                                                                                                        -1])).squeeze()
                logits = logits + proto_logits
        elif flag == "max":
            # all_logits_global = all_logits_global.softmax(dim=-1)
            idx = torch.max(all_logits_global.max(dim=-1)[0], dim=-1, keepdim=True)[1]
            logits = torch.gather(all_logits_global, dim=1,
                                  index=idx.unsqueeze(-1).expand(-1, -1, all_logits_global.shape[-1])).squeeze()
        elif flag == "energy":
            # all_logits_global = all_logits_global.softmax(dim=-1)
            energies = -torch.logsumexp(all_logits_global, dim=-1)
            idx = torch.min(energies, dim=-1, keepdim=True)[1]
            logits = torch.gather(all_logits_global, dim=1,
                                  index=idx.unsqueeze(-1).expand(-1, -1, all_logits_global.shape[-1])).squeeze()
        else:
            all_logits_global = torch.where(
                (all_logits_global.argmax(dim=-1) == all_logits_global.shape[-1] - 1).unsqueeze(-1).expand(-1, -1,
                                                                                                           all_logits_global.shape[
                                                                                                               -1]), 0,
                all_logits_global)
            logits = all_logits_global[:, :, :-1].mean(dim=1)

        return logits, idx


    def forward_test(self, image, text_features=None, task_id=None, img_proto=None, labels=None):
        # all_logits_global = torch.tensor([]).cuda()
        all_img_features = torch.tensor([]).cuda()
        all_img_features1 = torch.tensor([]).cuda()
        for i in range(task_id):
            image_features = self.backbone.encode_image(image, adapter_list=self.img_adapter_list[i])
            image_features1 = self.moe_adapter(image_features, train=False)
            # text_features = self.backbone.encode_text(text_tokens, adapter_list=self.text_adapter_list,
            #                                           prompt=self.text_prompts[:(task_id + 1) * self.inc] if self.text_prompts is not None else None)
            all_img_features = torch.cat([all_img_features, image_features.unsqueeze(1)], dim=1)
            all_img_features1 = torch.cat([all_img_features1, image_features1.unsqueeze(1)], dim=1)

        image_features = self.backbone.encode_image(image, adapter_list=self.cur_img_adapter)
        image_features1 = self.moe_adapter(image_features, train=False)

        all_img_features = torch.cat([all_img_features, image_features.unsqueeze(1)], dim=1)
        all_img_features1 = torch.cat([all_img_features1, image_features1.unsqueeze(1)], dim=1)

        logits, idx, delta_entropy = self.feature_aggregate((all_img_features, all_img_features1), text_features, labels)

        assert logits.shape[-1] == len(text_features)
        text_features = None

        return {"logits": logits, "features": image_features, "task_id": idx, "delta_entropy": delta_entropy}

    def forward(self, image, text_features=None, train=False, task_id=None, img_proto=None, labels=None):
        if text_features is None:
            image_features = self.backbone.encode_image(image, adapter_list=self.cur_img_adapter)
            image_features1 = self.moe_adapter(image_features, train=False)
            # image_features_normed = image_features / image_features.norm(dim=1, keepdim=True)
            return {"features": image_features, "features1": image_features1}
        else:
            if train:
                out = self.forward_train(image, text_features, task_id)
            else:
                out = self.forward_test(image, text_features, task_id, img_proto=img_proto, labels=labels)

            return out

    # def get_ood_vectors(self, image, task_id):
    #     ood_features = torch.tensor([]).cuda()
    #     for i in range(task_id):
    #         image_features = self.backbone.encode_image(image, adapter_list=self.img_adapter_list[i])
    #         # image_features_normed = image_features / image_features.norm(dim=1, keepdim=True)
    #         ood_features = torch.cat([ood_features, image_features], dim=0)
    #
    #     return ood_features

    def forward_with_vectors(self, x, text_features_normed, ood_vectors=None):
        # x = x.type(self.backbone.dtype)
        # x = x.to(self.backbone.transformer.get_cast_dtype())
        image_features = x
        image_features1 = self.moe_adapter(x, train=True)
        image_features_normed = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features_normed1 = image_features1 / image_features1.norm(dim=-1, keepdim=True)

        logits = self.backbone.logit_scale.exp() * image_features_normed @ text_features_normed.t()
        logits = logits.softmax(dim=-1)
        logits1 = self.backbone.logit_scale.exp() * image_features_normed1 @ text_features_normed.t()
        logits1 = logits1.softmax(dim=-1)
        return {"logits": logits, "logits1": logits1, "ood_logits": None}

    def select_desc(self, new_class_names, old_desc_tokens, all_descs, class_means=None):
        if old_desc_tokens is not None:
            old_desc_features = self.backbone.encode_text(old_desc_tokens)
            old_desc_features = old_desc_features / old_desc_features.norm(dim=-1, keepdim=True)
        if class_means is not None:
            class_means = class_means.cuda()
            class_means = class_means / class_means.norm(dim=-1, keepdim=True)
        new_desc_tokens = torch.LongTensor([]).cuda()
        for class_name in new_class_names:
            assert class_name in all_descs, "Class_name not in prompt_json!"
            # cls_desc = ""
            # for desc in all_descs[class_name]:
            #     flag = True
            #     for i in ["red", "blue", "green", "yellow", "black", "white", "orange", "grey", "brown"]:
            #         if i in desc:
            #             flag = False
            #     if flag:
            #         cls_desc += desc
            #
            # new_cls_tokens = tokenize([self.config.prompt_template.format(class_name) + cls_desc]).cuda()
            # new_desc_tokens = torch.cat([new_desc_tokens, new_cls_tokens], dim=0)

            if old_desc_tokens is not None:
                cls_desc = all_descs[class_name]
                for desc in cls_desc:
                    for i in ["red", "blue", "green", "yellow", "black", "white", "orange", "grey", "brown"]:
                        if i in desc:
                            cls_desc.remove(desc)
                            break
                new_cls_tokens = tokenize([self.config.prompt_template.format(class_name)+item for item in cls_desc]).cuda()    # self.config.prompt_template.format(class_name)+
                new_cls_features = self.backbone.encode_text(new_cls_tokens)
                new_cls_features = new_cls_features / new_cls_features.norm(dim=-1, keepdim=True)
                idx = torch.min((new_cls_features @ old_desc_features.t()).sum(dim=-1), dim=-1)[1]
                new_desc_tokens = torch.cat([new_desc_tokens, new_cls_tokens[idx].unsqueeze(0)], dim=0)

            # if class_means is not None:
            #     cls_desc = all_descs[class_name]
            #     new_cls_tokens = tokenize([item for item in cls_desc]).cuda()  # self.config.prompt_template.format(class_name)+
            #     new_cls_features = self.backbone.encode_text(new_cls_tokens)
            #     new_cls_features = new_cls_features / new_cls_features.norm(dim=-1, keepdim=True)
            #     idx = torch.min((new_cls_features @ class_means.t()).sum(dim=-1), dim=-1)[1]
            #     new_desc_tokens = torch.cat([new_desc_tokens, new_cls_tokens[idx].unsqueeze(0)], dim=0)
            else:
                # cls_desc = max(all_descs[class_name], key=len)
                cls_desc = sample(all_descs[class_name], 1)[0]
                new_cls_tokens = tokenize([self.config.prompt_template.format(class_name)+cls_desc]).cuda()       # self.config.prompt_template.format(class_name)+
                new_desc_tokens = torch.cat([new_desc_tokens, new_cls_tokens], dim=0)
        cur_desc_tokens = torch.cat([old_desc_tokens, new_desc_tokens], dim=0) if old_desc_tokens is not None else new_desc_tokens

        return new_desc_tokens, cur_desc_tokens

    def freeze_adapter(self):
        for name, param in self.named_parameters():
            if "cur_img_adapter" in name or "text_prompts" in name:
                param.requires_grad = False

        # for module in self.img_final_adapter.modules():
        #     if type(module) == nn.Linear:
        #         nn.init.kaiming_uniform_(module.weight)

    def unfreeze_prompt(self):
        self.text_prompts.requires_grad = True
