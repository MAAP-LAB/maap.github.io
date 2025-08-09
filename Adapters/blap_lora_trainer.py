"""
BLAP LoRA Trainer - Simplified Version
Q-Former and T5 fine-tuning with LoRA for MusicQA
"""

import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
import argparse
from typing import Dict, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Add paths
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/blap")

from transformers import BertConfig, T5TokenizerFast, T5Config, BertTokenizer
from blap.model.BLAP2.modeling_t5 import T5ForConditionalGeneration
from blap.model.BLAP2.QFormer import BertLMHeadModel
from blap.config.BLAP2_Config import BLAP2_Stage2_Config

# PEFT/LoRA imports
from peft import LoraConfig, get_peft_model, TaskType

# PyTorch 2.6+ compatibility
import numpy.core.multiarray
torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])


class MusicQADataset(Dataset):
    """Dataset for MusicQA with *.npy files"""
    
    def __init__(self, json_path: str, npy_base_path: str = f"{PROJECT_ROOT}/Assets/npys"):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.npy_base_path = Path(npy_base_path)
        self.data = [item for item in self.data if item['audio_name'].endswith('.npy')]
        
        print(f"Loaded {len(self.data)} samples with .npy audio files")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = self.npy_base_path / item['audio_name']
        
        conversation = item['conversation']
        question = conversation[0]['value'] if conversation else "Describe the audio"
        answer = conversation[1]['value'] if len(conversation) > 1 else "Unknown"
        
        return {
            'audio_path': str(audio_path),
            'question': question,
            'answer': answer
        }


class BLAPLoRATrainer(nn.Module):
    """BLAP trainer with LoRA adapters for Q-Former and T5"""
    
    def __init__(self, 
                 blap_checkpoint_path: str,
                 config_path: str,
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Load BLAP configuration
        self.blap_config = BLAP2_Stage2_Config.from_file(config_path)
        
        # Initialize models
        self._init_qformer_with_checkpoint(blap_checkpoint_path)
        self._init_t5()
        
        # Simple projection for dimension matching
        self.audio_projection = nn.Linear(768, 1024)  # CLaMP3 → Q-Former
        
        # Apply LoRA
        self._apply_lora_to_qformer()
        self._apply_lora_to_t5()
        
        # Freeze base models
        self._freeze_base_models()
        
        # Setup tokenizers
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="right")
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(self.blap_config.LLM.t5_model)
        
        self.max_txt_len = self.blap_config.max_txt_len
        self.prompt = self.blap_config.prompt
    
    def _init_qformer_with_checkpoint(self, checkpoint_path: str):
        """Initialize Q-Former from BLAP checkpoint"""
        print("🤖 Initializing Q-Former...")
        
        # Q-Former config
        qformer_config = BertConfig.from_pretrained("bert-base-uncased")
        qformer_config.encoder_width = 1024
        qformer_config.add_cross_attention = True
        qformer_config.cross_attention_freq = 2
        qformer_config.query_length = self.blap_config.num_query_tokens
        
        self.qformer_config = qformer_config
        self.qformer = BertLMHeadModel(qformer_config)
        
        # Query tokens
        self.query_tokens = nn.Parameter(
            torch.zeros(1, self.blap_config.num_query_tokens, qformer_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=qformer_config.initializer_range)
        
        # Load checkpoint
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint.get('state_dict', checkpoint)
            
            qformer_state_dict = {}
            for key, value in state_dict.items():
                if 'qformer' in key:
                    new_key = key.replace('qformer.', '')
                    qformer_state_dict[new_key] = value
                elif 'query_tokens' in key:
                    self.query_tokens.data.copy_(value)
            
            if qformer_state_dict:
                self.qformer.load_state_dict(qformer_state_dict, strict=False)
                print("✅ Q-Former loaded from checkpoint")
    
    def _init_t5(self):
        """Initialize T5 model"""
        print("📝 Initializing T5...")
        
        t5_config = T5Config.from_pretrained(self.blap_config.LLM.t5_model)
        t5_config.dense_act_fn = "gelu"
        
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            self.blap_config.LLM.t5_model,
            config=t5_config
        )
        
        # T5 projection
        self.t5_proj = nn.Linear(
            self.qformer_config.hidden_size,
            self.t5_model.config.hidden_size
        )
    
    def _apply_lora_to_qformer(self):
        """Apply LoRA to Q-Former"""
        print("🔧 Applying LoRA to Q-Former...")
        
        qformer_lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=[
            "attention.self.query","attention.self.key","attention.self.value",
            "attention.output.dense",
            "crossattention.self.query","crossattention.self.key","crossattention.self.value",
            "crossattention.output.dense",
            ],
            bias="none"
        )
        
        self.qformer = get_peft_model(self.qformer, qformer_lora_config)
        self.qformer.print_trainable_parameters()
    
    def _apply_lora_to_t5(self):
        """Apply LoRA to T5"""
        print("🔧 Applying LoRA to T5...")
        
        t5_lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=["q", "k", "v", "o"],  # Only Q and V for efficiency
            bias="none"
        )
        
        self.t5_model = get_peft_model(self.t5_model, t5_lora_config)
        self.t5_model.print_trainable_parameters()
    
    def _freeze_base_models(self):
        """Freeze non-LoRA parameters"""
        self.query_tokens.requires_grad = False
        self.t5_proj.requires_grad = False
        
        # Audio projection stays trainable
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    def load_npy_features(self, audio_paths: List[str], max_length: int = 128):
        """Load pre-extracted CLaMP3 features"""
        features_list = []
        masks_list = []
        
        for audio_path in audio_paths:
            features = np.load(audio_path)
            features = torch.from_numpy(features).float()
            
            if features.ndim == 1:
                features = features.unsqueeze(0)
            elif features.ndim == 3:
                features = features.view(-1, features.shape[-1])
            
            seq_len, feat_dim = features.shape[0], features.shape[1]
            
            if seq_len > max_length:
                features = features[:max_length]
                mask = torch.ones(max_length)
            elif seq_len < max_length:
                padding = torch.zeros(max_length - seq_len, feat_dim)
                features = torch.cat([features, padding], dim=0)
                mask = torch.zeros(max_length)
                mask[:seq_len] = 1
            else:
                mask = torch.ones(max_length)
            
            features_list.append(features)
            masks_list.append(mask)
        
        return torch.stack(features_list).to(self.device), torch.stack(masks_list).to(self.device)
    
    def forward(self, batch: Dict):
        """Forward pass with LoRA-enabled models"""
        audio_paths = batch['audio_path']
        questions = batch['question']
        answers = batch['answer']
        
        batch_size = len(audio_paths)
        
        # 1. Load CLaMP3 features (768 dim)
        clamp3_features, audio_masks = self.load_npy_features(audio_paths)
        
        # 2. Project to Q-Former dimension (1024 dim)
        audio_features = self.audio_projection(clamp3_features)
        
        # 3. Q-Former with LoRA
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        query_output = self.qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_features,  # Now 1024 dim
            encoder_attention_mask=audio_masks,
            return_dict=True,
        )
        
        # 4. Project to T5
        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long, device=self.device)
        
        # 5. T5 with LoRA
        input_tokens = self.t5_tokenizer(
            [self.prompt.format(q) for q in questions],
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        
        output_tokens = self.t5_tokenizer(
            answers,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        
        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
        
        with torch.no_grad():
            inputs_embeds = self.t5_model.get_input_embeddings()(input_tokens.input_ids)
        
        inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
        
        targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
        )
        
        outputs = self.t5_model(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            decoder_attention_mask=output_tokens.attention_mask,
            return_dict=True,
            labels=targets,
        )
        
        return {'loss': outputs.loss, 'logits': outputs.logits}