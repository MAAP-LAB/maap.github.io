import json

from blap.config.config import Config

class AudioEncoder_Config:
    def __init__(self, jsonData):
        if "pretrained" in jsonData:
            self.pretrained = jsonData["pretrained"]
        self.audio_cfg = Config(jsonData["audio_cfg"])
        self.embed_dim_audio: int = jsonData["embed_dim_audio"]
        if self.audio_cfg.model_type == "Clamp3":
            self.hidden_size = jsonData["hidden_size"]
            self.num_layers = jsonData["num_layers"]
            self.max_length = jsonData["max_length"]
            self.clamp3_weights_path = jsonData["clamp3_weights_path"]
        # self.class_num: int = jsonData["class_num"]

    @classmethod
    def from_file(cls, jsonFile):
        with open(jsonFile, "r") as f:
            data = json.load(f)
        
        return cls(data)