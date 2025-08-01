"""
REMAKRKS:
This file contains the model BLAP2. It is highly inspired by the BLIP2 model and therefore contains code snippets from BLIP2's implementation (GitHub: https://github.com/salesforce/LAVIS/tree/main/projects/blip2)
"""

import logging
import soundfile as sf
import json
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch import nn
from torch.utils.data import DataLoader
from transformers import Blip2QFormerConfig, Blip2QFormerModel, BertTokenizer, BertConfig, T5TokenizerFast, AutoTokenizer
from accelerate import Accelerator

from blap.config.BLAP2_Config import BLAP2_Stage1_Config, BLAP2_Stage2_Config
from blap.dataset.dataset import MusicCaps, ShutterStock
from blap.model.BLAP2.BLAP2Outputs import BlapOutput, BlapIntermediateOutput
from blap.model.BLAP2.QFormer import BertLMHeadModel
from blap.model.BLAP2.modeling_t5 import T5Config, T5ForConditionalGeneration

from clamp3.code.utils import *
from clamp3.code.config import *

from Adapters.bottleneck import *




# Lavis Utilities
"""
The following functions are based on the BLIP2 implementation by Salesforce.
Copyright (c) 2022 Salesforce, Inc.
All rights reserved.
"""
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [
            torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """

    if not is_dist_avail_and_initialized():
        return tensors
    
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    # tensor_all = GatherLayer.apply(tensors)
    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)

class BLAP2_Base(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
    
    @property
    def device(self):
        return list(self.parameters())[0].device
    
    @classmethod
    def init_tokenizer(cls, truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer
    
    @classmethod
    def init_clamp3(self, audio_config, symbolic_config):
        accelerator = Accelerator()
        device = accelerator.device
        print("Using device:", device)
        
        model = CLaMP3Model(audio_config=audio_config,
                            symbolic_config=symbolic_config,
                            text_model_name=TEXT_MODEL_NAME,
                            hidden_size=CLAMP3_HIDDEN_SIZE,
                            load_m3=CLAMP3_LOAD_M3)
        clamp3 = model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME) # environnmnt in Multi language is more efficient than BERT base tokenizer in BLAP2 
        patchilizer = M3Patchilizer()
        print("Total Parameter Number: "+str(sum(p.numel() for p in model.parameters())))
        return clamp3, tokenizer , patchilizer

    @classmethod
    def init_adapter(self, input_dim=768, output_dim=1408):
        adapter = Adapter(input_dim
                        , output_dim) # For enter Q-Former, input_dim 768 from Clamp3 and output_dim 1408 from cross-attention layers in Q-Former
        return adapter
    
    @classmethod
    def initQformer(cls, num_query_token, audio_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = audio_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token

        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens


class BLAP2_Stage1(BLAP2_Base):
    """
    This stage deals with the training of the QFormer.
    """
    def __init__(self, config: BLAP2_Stage1_Config, trainConfig, forbidden_words=[], forbidden_drop=0.3) -> None:
        super().__init__()

        # Qualitative Analysis Audio, Caption Tuples
        self.forbidden_words = forbidden_words
        self.forbidden_drop = forbidden_drop
        is_only_eng = False
        # Training Configuration File
        self.trainConfig = trainConfig

        # Used for tracking previous validation step results -> compute mean over it after validation
        self.validation_step_outputs = []

        self.adapter = self.init_adapter(input_dim=768
                                        , output_dim=1408)

        audio_config = BertConfig(vocab_size=1,
                        hidden_size=AUDIO_HIDDEN_SIZE,
                        num_hidden_layers=AUDIO_NUM_LAYERS,
                        num_attention_heads=AUDIO_HIDDEN_SIZE//64,
                        intermediate_size=AUDIO_HIDDEN_SIZE*4,
                        max_position_embeddings=MAX_AUDIO_LENGTH)
        symbolic_config = BertConfig(vocab_size=1,
                                    hidden_size=M3_HIDDEN_SIZE,
                                    num_hidden_layers=PATCH_NUM_LAYERS,
                                    num_attention_heads=M3_HIDDEN_SIZE//64,
                                    intermediate_size=M3_HIDDEN_SIZE*4,
                                    max_position_embeddings=PATCH_LENGTH)

        #  Setup Audio Encoder known as CLaMP3, tokenizer and patchilizer
        # This tokenizer's name is FacebookAI/xlm-roberta-base
        self.clamp3, self.tokenizer, self.patchilizer = self.init_clamp3(self, audio_config, symbolic_config)

        # Freeze Audio Encoder Weights
        for param in self.clamp3.parameters():
            param.requires_grad = False

        # Restore AudioEncoder Weights if applicable
        #if hasattr(config.audio_encoder, 'pretrained'):
        #    weights = torch.load(config.audio_encoder.pretrained)
        #    self.audio_encoder.load_state_dict(weights)

        # Tokenizer
        if is_only_eng:
            self.tokenizer: BertTokenizer = self.init_tokenizer()

        # Setup Q Former
        self.qformer, self.query_tokens = self.initQformer(config.num_query_tokens, config.audio_encoder.embed_dim_audio)
        self.qformer.resize_token_embeddings(len(self.tokenizer))

        state_dict = self.qformer.state_dict()
        for name, param in self.qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        logging.debug(f"qformer config hidden size is : {self.qformer.config.hidden_size}\nconfg embedding dimension is : {config.embed_dim}")
        # TODO: Instead of using below code, input data, which is musical modality and text extract features from encoder being each clamp3 and t5_model.encoder
        # before they must be gone via adapter or projection  
        #self.musical_feature = self.adapter.forward(self.qformer.config.hidden_size, config.embed_dim)
        #self.text_proj = self.adapter.forward(self.qformer.config.hidden_size, config.embed_dim)

        self.atm_head = nn.Linear(self.qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = config.max_txt_len

    @classmethod
    def from_json(cls, modelConfig, trainConfig):
        if trainConfig == "":
            train_Config = {}
        else: 
            with open(trainConfig, "r")as f:
                train_Config = json.load(f)
        config = BLAP2_Stage1_Config.from_file(modelConfig)
        return cls(config, train_Config)

    @classmethod
    def from_checkpoint(cls, checkpoint_path, modelConfig, trainConfig):
        # Load configurations from files
        config = BLAP2_Stage1_Config.from_file(modelConfig)
        if trainConfig == "":
            train_Config = {}
        else: 
            with open(trainConfig, "r")as f:
                train_Config = json.load(f)

        # Load the model from the checkpoint
        map_location = None

        if not torch.cuda.is_available():
            print("No GPU found, loading on CPU")
            map_location = torch.device('cpu')
        model = cls.load_from_checkpoint(checkpoint_path, config=config, trainConfig=train_Config, map_location=map_location)

        return model

    
    def get_audioFeatures(self, audio, device=None):
        """
            Deprecated
            Returns the embedding provided by the audio encoder
        """
        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        audio_embeds = self.audio_encoder(audio, device) # Default width 512 ()
        return audio_embeds
    
    def get_queryFeatures(self, audio: torch.Tensor):
        audio_embeds = self.ln_audio(F.avg_pool2d( self.audio_encoder(audio)[2]["fine_grained_embedding"], (32, 1)))
        audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long)

        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)

        query_output = self.qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=audio_atts,
            use_cache=True,
            return_dict=True,
        )
        
        return query_output
    
    @torch.no_grad()
    def get_audioEmbeddings(self, audios, device=None):
        """
            Returns the audio embedding of the Q-Former (Sometimes called audio features)

            -> Returns a BatchxQueryTokensxEmbedding
        """
        
        # Set Device to GPU if available
        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        audio_embeds: torch.Tensor = self.ln_audio(F.avg_pool2d( self.audio_encoder(audios)[2]["fine_grained_embedding"], (32, 1)))
        audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)

        query_output = self.qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=audio_atts,
            use_cache=True,
            return_dict=True,
        )

        music_feats: torch.Tensor = F.normalize(
            self.audio_proj(query_output.last_hidden_state), dim=-1
        )

        return audio_feats
    
    @torch.no_grad()
    def get_captionEmbeddings(self, captions, device=None):
        """
            Returns the text embedding of the Q-Former (Somtimes called text features)

            -> Returns BatchSizexEmbedding
        """
        # Set Device to GPU if available
        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        caption_tokens = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        text_output = self.qformer.bert(
            caption_tokens.input_ids,
            attention_mask=caption_tokens.attention_mask,
            return_dict=True,
        )

        # Take CLS token as text embedding
        text_feat: torch.Tensor = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        return text_feat
    
    @torch.no_grad()
    def extract_feature(self, filename, get_global=True):
        accelerator = Accelerator()
        device = accelerator.device
        if not filename.endswith(".npy"):
            with open(filename, "r", encoding="utf-8") as f:
                item = f.read()

        if filename.endswith(".txt"):
            item = list(set(item.split("\n")))
            item = "\n".join(item)
            item = item.split("\n")
            item = [c for c in item if len(c) > 0]
            item = self.tokenizer.sep_token.join(item)
            input_data = self.tokenizer(item, return_tensors="pt")
            input_data = input_data['input_ids'].squeeze(0)
            max_input_length = MAX_TEXT_LENGTH
        elif filename.endswith(".abc") or filename.endswith(".mtf"):
            input_data = self.patchilizer.encode(item, add_special_patches=True)
            input_data = torch.tensor(input_data)
            max_input_length = PATCH_LENGTH
        elif filename.endswith(".npy"):
            input_data = np.load(filename)
            input_data = torch.tensor(input_data)
            input_data = input_data.reshape(-1, input_data.size(-1))
            zero_vec = torch.zeros((1, input_data.size(-1)))
            input_data = torch.cat((zero_vec, input_data, zero_vec), 0)
            max_input_length = MAX_AUDIO_LENGTH
        else:
            raise ValueError(f"Unsupported file type: {filename}, only support .txt, .abc, .mtf, .npy files")

        segment_list = []
        for i in range(0, len(input_data), max_input_length):
            segment_list.append(input_data[i:i+max_input_length])
        segment_list[-1] = input_data[-max_input_length:]

        last_hidden_states_list = []

        for input_segment in segment_list:
            input_masks = torch.tensor([1]*input_segment.size(0))
            if filename.endswith(".txt"):
                pad_indices = torch.ones(MAX_TEXT_LENGTH - input_segment.size(0)).long() * self.tokenizer.pad_token_id
            elif filename.endswith(".abc") or filename.endswith(".mtf"):
                pad_indices = torch.ones((PATCH_LENGTH - input_segment.size(0), PATCH_SIZE)).long() * self.patchilizer.pad_token_id
            else:
                pad_indices = torch.ones((MAX_AUDIO_LENGTH - input_segment.size(0), AUDIO_HIDDEN_SIZE)).float() * 0.
            input_masks = torch.cat((input_masks, torch.zeros(max_input_length - input_segment.size(0))), 0)
            input_segment = torch.cat((input_segment, pad_indices), 0)

            if filename.endswith(".txt"):
                last_hidden_states = self.clamp3.get_text_features(text_inputs=input_segment.unsqueeze(0).to(device),
                                                            text_masks=input_masks.unsqueeze(0).to(device),
                                                            get_global=get_global)
            elif filename.endswith(".abc") or filename.endswith(".mtf"):
                last_hidden_states = self.clamp3.get_symbolic_features(symbolic_inputs=input_segment.unsqueeze(0).to(device),
                                                            symbolic_masks=input_masks.unsqueeze(0).to(device),
                                                            get_global=get_global)
            else:
                last_hidden_states = self.clamp3.get_audio_features(audio_inputs=input_segment.unsqueeze(0).to(device),
                                                            audio_masks=input_masks.unsqueeze(0).to(device),
                                                            get_global=get_global)
            if not get_global:
                last_hidden_states = last_hidden_states[:, :input_masks.sum().long().item(), :]
            last_hidden_states_list.append(last_hidden_states)

        if not get_global:
            last_hidden_states_list = [last_hidden_states[0] for last_hidden_states in last_hidden_states_list]
            last_hidden_states_list[-1] = last_hidden_states_list[-1][-(len(input_data)%max_input_length):]
            last_hidden_states_list = torch.concat(last_hidden_states_list, 0)
        else:
            full_chunk_cnt = len(input_data) // max_input_length
            remain_chunk_len = len(input_data) % max_input_length
            if remain_chunk_len == 0:
                feature_weights = torch.tensor([max_input_length] * full_chunk_cnt, device=device).view(-1, 1)
            else:
                feature_weights = torch.tensor([max_input_length] * full_chunk_cnt + [remain_chunk_len], device=device).view(-1, 1)
            
            last_hidden_states_list = torch.concat(last_hidden_states_list, 0)
            last_hidden_states_list = last_hidden_states_list * feature_weights
            last_hidden_states_list = last_hidden_states_list.sum(dim=0) / feature_weights.sum()

            features = last_hidden_states_list.unsqueeze(0)
            features = features.detach().cpu().numpy()
        return features


    """
    Function is based on salesforce's implementation of the BLIP2 model
    Copyright (c) 2022 Salesforce, Inc.
    All rights reserved.
    """
    # only provides the loss for training
    def forward(self, music_files, captions, device=None):
        # Get Audio Embeddings
        # Fix device if not provided
        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # On-the-fly feature extraction
        # 1. Extract features from file paths, which returns a list of numpy arrays
        # 2. Convert numpy arrays to a single tensor
        with torch.no_grad():
            music_features_list = [torch.from_numpy(self.extract_feature(music_file)) for music_file in music_files]
            music_features = torch.cat(music_features_list).to(device)

        # Pass features through the adapter
        music_embeds = self.adapter(music_features) # (batch, 768) -> (batch, 1408)
        # Reshape for Q-former cross-attention
        music_embeds = music_embeds.unsqueeze(1) # (batch, 1, 1408)
        music_atts = torch.ones(music_embeds.size()[:-1], dtype=torch.long).to(device)


        query_tokens = self.query_tokens.expand(music_embeds.shape[0], -1, -1)

        query_output = self.qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=music_embeds,
            encoder_attention_mask=music_atts,
            use_cache=True,
            return_dict=True,
        )

        music_feats: torch.Tensor = F.normalize(
            self.audio_proj(query_output.last_hidden_state), dim=-1
        )

        caption_tokens = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        text_output = self.qformer.bert(
            caption_tokens.input_ids,
            attention_mask=caption_tokens.attention_mask,
            return_dict=True,
        )

        # Take CLS token as text embedding
        text_feat: torch.Tensor = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )
        
        #-------------------------------------------------------------------#
        #--------------------Audio Text Contrastive Loss--------------------#
        #-------------------------------------------------------------------#

        music_feats_all = concat_all_gather(
            music_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        sim_q2t: torch.Tensor = torch.matmul(
            music_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # audio-text similarity: aggregate across all query tokens
        sim_a2t, _ = sim_q2t.max(-1)
        sim_a2t = sim_a2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), music_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-audio similarity: aggregate across all query tokens
        sim_t2a, _ = sim_t2q.max(-1)
        sim_t2a = sim_t2a / self.temp  
        try:
            rank = dist.get_rank()
        except RuntimeError:
            print("Not part of cluster, rank not initalised")
            rank = 0 
        bs = len(music_files)

        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            music_embeds.device
        )

        # compute matchting matrix
        loss_atc = (
            F.cross_entropy(sim_a2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2a, targets, label_smoothing=0.1)
        ) / 2

        #----------------------------------------------------------------#    
        #--------------------Audio Text Matching Loss--------------------#
        #----------------------------------------------------------------#

        text_input_ids_world = concat_all_gather(caption_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(caption_tokens.attention_mask)
        music_embeds_world = all_gather_with_grad(music_embeds)

        with torch.no_grad():
            sim_t2a[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
            sim_a2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
            weights_t2i = F.softmax(sim_t2a, dim=1)
            weights_i2t = F.softmax(sim_a2t, dim=1)

        music_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            music_embeds_neg.append(music_embeds_world[neg_idx])
        music_embeds_neg = torch.stack(music_embeds_neg, dim=0)

        # select a negative text for each audio
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [caption_tokens.input_ids, caption_tokens.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg

        text_atts_all = torch.cat(
            [caption_tokens.attention_mask, caption_tokens.attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_atm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_atm.size()[:-1], dtype=torch.long).to(
            device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        music_embeds_all = torch.cat(
            [music_embeds, music_embeds_neg, music_embeds], dim=0
        )

        music_atts_all = torch.ones(music_embeds_all.size()[:-1], dtype=torch.long).to(
            device
        )

        output_atm = self.qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_atm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=audio_embeds_all,
            encoder_attention_mask=audio_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_atm.last_hidden_state[:, : query_tokens_atm.size(1), :]
        vl_output = self.atm_head(vl_embeddings)
        logits: torch.Tensor = vl_output.mean(dim=1)

        atm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(device)


        loss_atm = F.cross_entropy(logits, atm_labels)

        #---------------------------------------------------------------#
        #--------------------Language Modelling Loss--------------------#
        #---------------------------------------------------------------#

        decoder_input_ids = caption_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            device
        )
        attention_mask = torch.cat([query_atts, caption_tokens.attention_mask], dim=1)
        lm_output = self.qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss
        
        return BlapOutput(
            intermediate_output=BlapIntermediateOutput(
                atm_logits=logits.detach(),
                atm_labels=atm_labels.detach(),
                vl_embedding=vl_embeddings.detach(),
                music_embeds=music_embeds.detach(),
                music_features=music_feats.detach(),
                text_embeds=text_feat.detach(),
                decoder_output=lm_output,
                text_features=text_feat
            ),
            loss=loss_atc + loss_atm + loss_lm,
            loss_atc=loss_atc,
            loss_atm=loss_atm,
            loss_lm=loss_lm,
        )
    
    
    #------------------------------------------------#
    #----------------PyTorch Lightning---------------#
    #------------------------------------------------#
    def setup(self, stage=None) -> None:
        trainSetConfig = self.trainConfig["trainSet"]
        set_type = trainSetConfig["type"]
        dataset = ShutterStock(trainSetConfig['data'], trainSetConfig['music']) if set_type == 'ShutterStock' else MusicCaps(trainSetConfig['data'], trainSetConfig['music'])
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.trainConfig["learn_rate"])

    def training_step(self, batch, batch_idx):
        music_files, caption = batch
        loss: BlapOutput = self(music_files, caption)

        # Logging
        self.log('loss_atc_train', loss.loss_atc.item())
        self.log('loss_atm_train', loss.loss_atm.item())
        self.log('loss_lm_train', loss.loss_lm.item())
        self.log("loss_train", loss.loss.item())
        return loss.loss

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.trainConfig["batch_size"], shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.trainConfig["batch_size"], shuffle=False, drop_last=True)

    def validation_step(self, batch, batch_idx):
        music_files, caption = batch

        loss: BlapOutput = self(music_files, caption)
        
        # Logging
        self.log('loss_atc_val', loss.loss_atc.item(), sync_dist=True)
        self.log('loss_atm_val', loss.loss_atm.item(), sync_dist=True)
        self.log('loss_lm_val', loss.loss_lm.item(), sync_dist=True)
        self.log("val_loss", loss.loss_lm.item() + loss.loss_atc.item() + loss.loss_atm.item(), sync_dist=True)
        self.validation_step_outputs.append( {'loss_atc_val': loss.loss_atc.item(), 'loss_atm_val': loss.loss_atm.item(), "loss_lm_val": loss.loss_lm.item()})

    def on_validation_epoch_end(self):
        avg_atc = np.stack([x['loss_atc_val'] for x in self.validation_step_outputs]).mean()
        avg_atm = np.stack([x['loss_atm_val'] for x in self.validation_step_outputs]).mean()
        avg_lm = np.stack([x['loss_lm_val'] for x in self.validation_step_outputs]).mean()

        self.log('avg_atc', avg_atc, sync_dist=True)
        self.log('avg_atm', avg_atm, sync_dist=True)
        self.log('avg_lm', avg_lm, sync_dist=True)
        self.log('avg_loss', avg_atc + avg_atm + avg_lm, sync_dist=True)
        self.validation_step_outputs = []
        return {'avg_atc': avg_atc, 'avg_atm': avg_atm, "avg_lm": avg_lm, "avg_loss": avg_atc + avg_atm + avg_lm}
    
# T5 text encoder / decoder model
class BLAP2_Stage2(BLAP2_Base):
    def __init__(self, config: BLAP2_Stage2_Config, trainConfig, forbidden_words=["Upbeat", "upbeat", "Upbeat,", "upbeat,", "The low quality recording", "This is a low-quality"], forbidden_drop=0.6):
        super().__init__()

        self.forbidden_words = forbidden_words
        self.forbidden_drop = forbidden_drop

        self.trainConfig = trainConfig

        self.tokenizer: BertTokenizer = self.init_tokenizer()
        # Setup Audio Encoder
        self.audio_encoder = AudioEncoder(embed_dim=config.audio_encoder.embed_dim_audio, audio_cfg=config.audio_encoder.audio_cfg)
        self.ln_audio = torch.nn.LayerNorm(config.audio_encoder.embed_dim_audio)

        # Freeze Audio Encoder Weights
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        # Restore AudioEncoder Weights if applicable
        if hasattr(config.audio_encoder, 'pretrained') and config.audio_encoder.pretrained != "":
            weights = torch.load(config.audio_encoder.pretrained)
            self.audio_encoder.load_state_dict(weights)
        
        if config.ln_audio != "":
            print("Init LN-audio from CKPT")
            weights = torch.load(config.ln_audio)
            self.ln_audio.load_state_dict(weights)

        # Setup Q Former
        self.qformer, self.query_tokens = self.initQformer(
            config.num_query_tokens, config.audio_encoder.embed_dim_audio
        )
        self.qformer.resize_token_embeddings(len(self.tokenizer))

        # Check if a Checkpoint is provided
        if config.qFormer_ckpt != "":
            # Load qFormer from checkpoint
            weights = torch.load(config.qFormer_ckpt)
            self.qformer.load_state_dict(weights)

        if config.qTokens != "":
            self.query_tokens = torch.load(config.qTokens)

        self.qformer.cls = None
        self.qformer.bert.embeddings.word_embeddings = None
        self.qformer.bert.embeddings.position_embeddings = None
        for layer in self.qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        # Setup Language Model
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(config.LLM.t5_model)
        t5_config = T5Config.from_pretrained(config.LLM.t5_model)
        t5_config.dense_act_fn = "gelu"
        t5_config.repetition_penalty = config.LLM.repetition_penalty
        if config.atRandom:
            print("Caution, init LLM at random, if you init from a checkpoint you can ignore this message")
            self.t5_model = T5ForConditionalGeneration(config=t5_config)
        else:
            self.t5_model = T5ForConditionalGeneration.from_pretrained(
                config.LLM.t5_model, config=t5_config
            )

        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False

        self.t5_proj = nn.Linear(
            self.qformer.config.hidden_size, self.t5_model.config.hidden_size
        )

        self.max_txt_len = config.max_txt_len
        self.prompt = config.prompt

        self._apply_lemmatizer = config.apply_lemmatizer
        self._lemmatizer = None

        self.validation_step_outputs = []

    @classmethod
    def from_json(cls, modelConfig, trainConfig):
        with open(trainConfig, "r") as f:
            trainConfigData = json.load(f)
        config = BLAP2_Stage2_Config.from_file(modelConfig)
        return cls(config, trainConfigData)

    @classmethod
    def from_checkpoint(cls, checkpoint_path, modelConfig, trainConfig=""):
        # Load configurations from files
        config = BLAP2_Stage2_Config.from_file(modelConfig)
        if trainConfig != "":
            with open(trainConfig, "r") as f:
                train_Config = json.load(f)
        else:
            print("Caution No training information provided, for inference you can ignore this message")
            train_Config = {}
        # Load the model from the checkpoint
        map_location = None

        if not torch.cuda.is_available():
            print("No GPU found, loading on CPU")
            map_location = torch.device('cpu')
        model = cls.load_from_checkpoint(checkpoint_path,map_location=map_location, config=config, trainConfig=train_Config)

        return model
    
    def remove_forbidden_phrases_with_chance(self, captions, forbidden_phrases, chance=0.3):
        # Remove occurrences of the forbidden phrases based on a random chance
        processed_captions = []
        for cap in captions:
            modified_caption = cap
            for phrase in forbidden_phrases:
                while phrase in modified_caption:
                    if torch.rand(1) < chance:
                        modified_caption = modified_caption.replace(phrase, "", 1)
                    else:
                        # Break the loop if the condition is not met to avoid infinite loop
                        break
            processed_captions.append(modified_caption.strip())
        return processed_captions

    """
    Function is based on salesforce's implementation of the BLIP2 model
    Copyright (c) 2022 Salesforce, Inc.
    All rights reserved.
    """
    def forward(self, audios, captions, device=None, qualitative_testing=True):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        captions = self.remove_forbidden_phrases_with_chance(captions, self.forbidden_words, self.forbidden_drop)


        # Compute Audio Emebedding via Audio Encoder
        audio_embeds = self.ln_audio(F.avg_pool2d( self.audio_encoder(audios)[2]["fine_grained_embedding"], (32, 1)))
        audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)

        # Obtain output of Q-Former 
        query_output = self.qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=audio_atts,
            use_cache=True,
            return_dict=True,
        )

        # Map Q-Former output to T5 representation
        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(device)

        # Generate Prompt tokens
        input_tokens = self.t5_tokenizer(
                [self.prompt] * audio_embeds.size()[0],
                # captions,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
        ).to(device)

        # Generate Target Tokens
        output_tokens = self.t5_tokenizer(
                captions,
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)
        
        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        targets = output_tokens.input_ids.masked_fill(
                output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
        )

        # Compute Final Token input embedding
        inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

        # Compute Output including loss
        outputs = self.t5_model(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            decoder_attention_mask=output_tokens.attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        # ****************************************************** #
        # generate text predictions for small subset of the data #
        # ****************************************************** #
        if qualitative_testing:
            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds[:3],
                attention_mask=encoder_atts[:3],
                do_sample=False,
                num_beams=10,
                max_new_tokens=35,
                min_length=20,
                length_penalty=0.5,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

            if self._apply_lemmatizer:
                output_text = self._lemmatize(output_text)

            print(output_text)
        return {"loss": loss}
    
    def predict_answers(
        self,
        audios,
        prompt,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        
        length_penalty=-1,
        device=None
    ):
        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        audio_embeds = self.ln_audio(F.avg_pool2d( self.audio_encoder(audios)[2]["fine_grained_embedding"], (32, 1)))
        audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.query_tokens.expand(audio_embeds.shape[0], -1, -1)

        query_output = self.qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_embeds,
            encoder_attention_mask=audio_atts,
            use_cache=True,
            return_dict=True,
        )
        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(device)

        input_tokens = self.t5_tokenizer(
            [prompt] * audio_embeds.size()[0], padding="longest", return_tensors="pt"
        ).to(device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
        inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

        outputs = self.t5_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            do_sample=False,
            num_beams=num_beams,
            max_new_tokens=max_len,
            min_length=min_len,
            length_penalty=length_penalty,
        )
        output_text = self.t5_tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

        if self._apply_lemmatizer:
            output_text = self._lemmatize(output_text)

        return output_text


    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer
    
    #------------------------------------------------#
    #----------------PyTorch Lightning---------------#
    #------------------------------------------------#

    def setup(self, stage=None) -> None:
        trainSetConfig = self.trainConfig["trainSet"]
        set_type = trainSetConfig["type"]
        dataset = ShutterStock(trainSetConfig['data'], trainSetConfig['music']) if set_type == 'ShutterStock' else MusicCaps(trainSetConfig['data'], trainSetConfig['music'])
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        if set_type == 'ShutterStock':
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        else:
            self.train_dataset, self.val_dataset = dataset.createSplit()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.trainConfig["learn_rate"])

    def training_step(self, batch, batch_idx):
        music_files, caption = batch
        loss = self(music_files, list(caption), qualitative_testing=False)

        # Logging
        self.log('loss_train', loss["loss"].item(), sync_dist=True)
       
        return loss

    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, shuffle=True)
        return  DataLoader(
            self.train_dataset, 
            batch_size=self.trainConfig["batch_size"], 
            sampler=sampler, 
            drop_last=True
        )

    def val_dataloader(self):

        sampler = DistributedSampler(self.val_dataset)

        return DataLoader(
            self.val_dataset,
            batch_size=self.trainConfig["batch_size"],
            sampler= sampler,
            drop_last=True
        )

    def validation_step(self, batch, batch_idx):
        music_files, caption = batch
        loss = self(music_files, list(caption))
        
        
        self.validation_step_outputs.append(loss["loss"].item())

    def on_validation_epoch_end(self):
        val_loss = np.array(self.validation_step_outputs).mean()
       
       
        self.log('val_loss', val_loss, sync_dist=True)
        self.validation_step_outputs = []
        return {
            'val_loss', val_loss
        }
    