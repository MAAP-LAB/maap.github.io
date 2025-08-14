"""
SABA Base Models
Base 클래스들과 공통 기능을 정의합니다.
"""

import torch
import torch.nn as nn
from blap.blap.model.BLAP2.BLAP2_Pretrain import BLAP2_Stage2
import pytorch_lightning as pl
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import sys

# Add project paths
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.append(PROJECT_ROOT)

from saba.SABA_qformer import SABA_QFormer
from saba.SABA_outputs import SABAOutput, SABAStage1Output, SABAStage2Output

from transformers import BertConfig, AutoTokenizer, BertTokenizer
from saba.qformer import BertLMHeadModel, BertModel
    

class SABABase(pl.LightningModule):
    """
    SABA 모델들의 공통 베이스 클래스
    QFormer와 공통 기능들을 제공
    """
    
    def __init__(self,
                 qformer_config: Dict = None,
                 blap_checkpoint_path: str = None,
                 num_query_tokens: int = 32,
                 embed_dim: int = 256,
                 device_name: str = "cpu") -> None:
        super().__init__()
        
        # Initialize SABA QFormer if checkpoint path is provided
        if blap_checkpoint_path:
            self.saba_qformer = SABA_QFormer(
                blap_checkpoint_path=blap_checkpoint_path,
                config=qformer_config,
                num_query_tokens=num_query_tokens,
                embed_dim=embed_dim,
                device=device_name
            )
            self.tokenizer = self.saba_qformer.tokenizer
        else:
            self.saba_qformer = None
            self.tokenizer = self.init_tokenizer()
    
    @property
    def device(self):
        return list(self.parameters())[0].device
    
    @classmethod
    def init_tokenizer(cls, truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer
    
    @classmethod
    def init_Qformer(cls, checkpoint_path, modelConfig_path, trainConfig):
        blap_model = BLAP2_Stage2.from_checkpoint(
        checkpoint_path=checkpoint_path,
        modelConfig=modelConfig_path)

        return blap_model.Qformer, blap_model.query_tokens


    
    def get_parameter_count(self) -> int:
        """총 파라미터 수를 반환"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self) -> int:
        """훈련 가능한 파라미터 수를 반환"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_clamp3(self):
        """CLaMP3 파라미터를 동결 (서브클래스에서 구현)"""
        if hasattr(self, 'clamp3'):
            for param in self.clamp3.parameters():
                param.requires_grad = False
            print("🧊 CLaMP3 frozen")
    
    def freeze_qformer(self):
        """QFormer 파라미터를 동결"""
        if hasattr(self, 'saba_qformer') and self.saba_qformer:
            self.saba_qformer.freeze()
        elif hasattr(self, 'qformer'):
            for param in self.qformer.parameters():
                param.requires_grad = False
            print("🧊 QFormer frozen")


