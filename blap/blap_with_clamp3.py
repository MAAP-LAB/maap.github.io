import sys
import json
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
import soundfile as sf
import numpy as np
import subprocess
import os
import shutil

# Add clamp3 and blap to the python path
sys.path.append(str(Path('C:/Users/hyoun/maap.github.io')))
sys.path.append(str(Path('C:/Users/hyoun/maap.github.io/clamp3/code')))
sys.path.append(str(Path('C:/Users/hyoun/maap.github.io/blap')))

from blap.model.BLAP2.BLAP2_Pretrain import BLAP2_Base
from blap.config.BLAP2_Config import BLAP2_Stage2_Config
from clamp3.code.utils import CLaMP3Model
from transformers import BertConfig, T5TokenizerFast, T5Config
from blap.model.BLAP2.modeling_t5 import T5ForConditionalGeneration
from Adapters.adapter_module import Adapter

# Helper function (기존과 동일)
def _get_clamp3_embedding_from_wav(wav_path: str, device: torch.device) -> torch.Tensor:
    # ... (이 함수는 기존 코드와 동일하므로 생략합니다) ...
    # ... (실제 코드에서는 이 부분을 그대로 유지해야 합니다) ...
    pass # Placeholder for brevity

class BLAP2_Adapter_Trainer(BLAP2_Base):
    """
    This class is designed specifically to train the Adapter module.
    It uses a dual-loss system:
    1. Feature-level MSE loss between Clamp3 output and Adapter output.
    2. End-to-end QA Cross-Entropy loss from the final T5 output.

    All other components (Clamp3, Q-Former, T5) are frozen.
    """
    def __init__(self, config: BLAP2_Stage2_Config, clamp3_model_path: str, clamp3_audio_config: BertConfig, alpha: float = 0.5, beta: float = 0.5):
        """
        Args:
            config: Configuration object for the model.
            clamp3_model_path: Path to the pre-trained Clamp3 model weights.
            clamp3_audio_config: Configuration for the Clamp3 audio model.
            alpha (float): Weight for the feature-level MSE loss.
            beta (float): Weight for the end-to-end QA loss.
        """
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.alpha = alpha
        self.beta = beta

        # --- 1. Audio Encoder (Clamp3) ---
        print("Initializing and freezing audio encoder from clamp3...")
        symbolic_config = BertConfig(vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, max_position_embeddings=512)
        clamp3_full_model = CLaMP3Model(audio_config=clamp3_audio_config, symbolic_config=symbolic_config)
        
        checkpoint = torch.load(clamp3_model_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        if next(iter(state_dict)).startswith('module.'):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        clamp3_full_model.load_state_dict(state_dict)

        self.audio_encoder = clamp3_full_model.audio_model
        self.ln_audio = torch.nn.LayerNorm(clamp3_audio_config.hidden_size)
        
        # FREEZE Clamp3
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        for param in self.ln_audio.parameters():
            param.requires_grad = False
        print("Clamp3 audio encoder is FROZEN.")

        # --- 2. Bottleneck Adapter (TRAIN TARGET) ---
        print("Initializing Bottleneck Adapter for training...")
        self.bottleneck_adapter = Adapter(
            input_dim=clamp3_audio_config.hidden_size, # 768
            output_dim=config.embed_dim # Should be 1408
        )
        print("Adapter is set to be TRAINED.")

        # --- 3. Q-Former ---
        print("Initializing and freezing Q-Former...")
        self.qformer, self.query_tokens = self.initQformer(
            config.num_query_tokens, config.embed_dim
        )
        self.qformer.resize_token_embeddings(len(self.tokenizer))

        if config.qFormer_ckpt:
            weights = torch.load(config.qFormer_ckpt)
            self.qformer.load_state_dict(weights)
        
        # ## LoRA 적용 가이드 (1/2) ##
        # 나중에 LoRA를 적용하려면 아래 주석을 해제하고, peft 라이브러리를 사용하세요.
        # from peft import LoraConfig, get_peft_model, TaskType
        # lora_config_qformer = LoraConfig(r=8, lora_alpha=16, target_modules=["query", "value"], ...)
        # self.qformer = get_peft_model(self.qformer, lora_config_qformer)
        # print("Q-Former is now a PEFT model for LoRA fine-tuning.")
        
        # FREEZE Q-Former (for now)
        for param in self.qformer.parameters():
            param.requires_grad = False
        print("Q-Former is FROZEN.")

        # --- 4. Language Model (Flan-T5) ---
        print("Initializing and freezing Language Model (Flan-T5)...")
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(config.LLM.t5_model)
        t5_config = T5Config.from_pretrained(config.LLM.t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(config.LLM.t5_model, config=t5_config)

        self.t5_proj = nn.Linear(
            self.qformer.config.hidden_size, self.t5_model.config.hidden_size
        )

        # ## LoRA 적용 가이드 (2/2) ##
        # 나중에 LoRA를 적용하려면 아래 주석을 해제하세요.
        # lora_config_t5 = LoraConfig(r=16, lora_alpha=32, target_modules=["q", "v"], task_type=TaskType.SEQ_2_SEQ_LM, ...)
        # self.t5_model = get_peft_model(self.t5_model, lora_config_t5)
        # print("T5 model is now a PEFT model for LoRA fine-tuning.")

        # FREEZE T5 and Projection Layer
        for param in self.t5_model.parameters():
            param.requires_grad = False
        for param in self.t5_proj.parameters():
            param.requires_grad = False
        print("T5 Model and Projection Layer are FROZEN.")

        self.max_txt_len = config.max_txt_len
        self.prompt = config.prompt

    def forward(self, audios, questions, answer_labels):
        """
        Performs a forward pass and calculates the dual loss for training the adapter.
        
        Args:
            audios (torch.Tensor): Input audio embeddings.
            questions (list[str]): List of questions corresponding to the audios.
            answer_labels (torch.Tensor): Tokenized and prepared labels for the T5 model.
        
        Returns:
            dict: A dictionary containing total_loss, and the individual losses.
        """
        device = self.bottleneck_adapter.parameters().__next__().device

        # --- Audio Feature Extraction ---
        audio_atts = torch.ones(audios.size()[:-1], dtype=torch.long, device=device)
        
        # 1. Get original feature from Clamp3 (within no_grad context)
        with torch.no_grad():
            original_feature = self.audio_encoder(inputs_embeds=audios, attention_mask=audio_atts)[0]
            original_feature = self.ln_audio(original_feature)

        # 2. Get adapted feature from the Adapter (gradients will flow from here)
        adapted_feature = self.bottleneck_adapter(original_feature)

        # --- Loss 1: Feature-level MSE Loss ---
        # Compare Adapter's output with its input (original_feature)
        # Use .detach() on the target to prevent gradients from flowing back into Clamp3
        loss1_feature_mse = F.mse_loss(adapted_feature, original_feature.detach())

        # --- Loss 2: End-to-End QA Cross-Entropy Loss ---
        # The rest of the forward pass uses the adapted_feature
        # Gradients from this loss will flow back through T5, Q-Former, and finally to the Adapter
        with torch.no_grad(): # Q-Former and T5 are frozen, but ops need to be in grad context
            query_tokens = self.query_tokens.expand(adapted_feature.shape[0], -1, -1)
            query_output = self.qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=adapted_feature,
                encoder_attention_mask=audio_atts,
                return_dict=True,
            )
            
            inputs_t5 = self.t5_proj(query_output.last_hidden_state)
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(device)

            input_tokens = self.t5_tokenizer(
                [self.prompt.format(question) for question in questions], # Integrate question into prompt
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)

            encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_attention_mask=answer_labels.attention_mask,
                return_dict=True,
                labels=answer_labels.input_ids.masked_fill(
                    answer_labels.input_ids == self.t5_tokenizer.pad_token_id, -100
                ),
            )
            loss2_qa_ce = outputs.loss

        # --- Combine Losses ---
        total_loss = (self.alpha * loss1_feature_mse) + (self.beta * loss2_qa_ce)

        return {
            "total_loss": total_loss,
            "loss1_feature_mse": loss1_feature_mse.detach(),
            "loss2_qa_ce": loss2_qa_ce.detach()
        }

if __name__ == '__main__':
    # This is a demonstration of how to use the new trainer class.
    # You must provide real paths to your config and model files.

    # --- Configuration (Use your actual paths and configs) ---
    blap_config_path = 'dummy_blap_config.json'
    clamp3_model_weights_path = 'dummy_clamp3_weights.pth'
    
    # Create dummy files if they don't exist
    if not Path(blap_config_path).exists():
        dummy_blap_config_data = {
            "LLM": {"t5_model": "google/flan-t5-base"},
            "num_query_tokens": 32, "embed_dim": 1408, "max_txt_len": 32, 
            "prompt": "Question: {} Answer:", "qFormer_ckpt": ""
        }
        with open(blap_config_path, 'w') as f: json.dump(dummy_blap_config_data, f)

    clamp3_audio_conf = BertConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, max_position_embeddings=128)
    if not Path(clamp3_model_weights_path).exists():
        symbolic_conf = BertConfig(hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, max_position_embeddings=512)
        dummy_clamp3_model = CLaMP3Model(audio_config=clamp3_audio_conf, symbolic_config=symbolic_conf)
        torch.save({'model': dummy_clamp3_model.state_dict()}, clamp3_model_weights_path)
    
    print("Loading configurations...")
    blap_config = BLAP2_Stage2_Config.from_file(blap_config_path)

    print("Initializing the new model: BLAP2_Adapter_Trainer")
    # You can adjust alpha and beta here
    model = BLAP2_Adapter_Trainer(
        config=blap_config,
        clamp3_model_path=clamp3_model_weights_path,
        clamp3_audio_config=clamp3_audio_conf,
        alpha=0.4, # Example weight for feature loss
        beta=0.6   # Example weight for QA loss
    )
    
    # --- Verify Trainable Parameters ---
    print("--- Trainable Parameters ---")
    total_params = 0
    trainable_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(name)
    print(f"Total params: {total_params}")
    print(f"Trainable params: {trainable_params} (Should only be the adapter's parameters)")
    print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    print("--------------------------")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train() # Set model to training mode

    # --- Example forward pass ---
    print("Performing a test forward pass...")
    # Dummy data for a batch size of 2
    batch_size = 2
    dummy_audio_embeds = torch.randn(batch_size, 128, 768).to(device) # (batch, seq_len, hidden_size)
    dummy_questions = ["What instrument is playing?", "What is the genre of this music?"]
    dummy_answers = ["A solo piano.", "This is a classical piece."]

    # Prepare labels for T5
    labels = model.t5_tokenizer(
        dummy_answers,
        padding="longest",
        return_tensors="pt"
    ).to(device)

    try:
        # The optimizer would typically be configured like this:
        # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        # optimizer.zero_grad()
        
        output = model.forward(dummy_audio_embeds, dummy_questions, labels)
        
        # loss = output['total_loss']
        # loss.backward()
        # optimizer.step()
        
        print(f"Forward pass successful.")
        print(f"Returned losses: {output}")

    except Exception as e:
        print(f"An error occurred during the forward pass: {e}")
        import traceback
        traceback.print_exc()