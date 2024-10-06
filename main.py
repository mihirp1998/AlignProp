import ipdb
st = ipdb.set_trace
import builtins
import time
import os
builtins.st = ipdb.set_trace
from dataclasses import dataclass, field
import prompts as prompts_file
import numpy as np
from transformers import HfArgumentParser

from config.alignprop_config import AlignPropConfig
from alignprop_trainer import AlignPropTrainer
from sd_pipeline import DiffusionPipeline
from trl.models.auxiliary_modules import aesthetic_scorer


@dataclass
class ScriptArguments:
    pretrained_model: str = field(
        default="runwayml/stable-diffusion-v1-5", metadata={"help": "the pretrained model to use"}
    )
    pretrained_revision: str = field(default="main", metadata={"help": "the pretrained model revision to use"})
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})




def image_outputs_logger(image_pair_data, global_step, accelerate_logger):
    # For the sake of this example, we will only log the last batch of images
    # and associated data
    result = {}
    images, prompts = [image_pair_data["images"], image_pair_data["prompts"]]
    for i, image in enumerate(images[:4]):
        prompt = prompts[i]
        result[f"{prompt}"] = image.unsqueeze(0).float()
    accelerate_logger.log_images(
        result,
        step=global_step,
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, AlignPropConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    project_dir = f"alignprop_{int(time.time())}"
    os.makedirs(f"checkpoints/{project_dir}", exist_ok=True)
    
    training_args.project_kwargs = {
        "logging_dir": "./logs",
        "automatic_checkpoint_naming": True,
        "total_limit": 5,
        "project_dir": f"checkpoints/{project_dir}",
    }

    prompt_fn = getattr(prompts_file, training_args.prompt_fn)
    
    pipeline = DiffusionPipeline(
        script_args.pretrained_model,
        pretrained_model_revision=script_args.pretrained_revision,
        use_lora=script_args.use_lora,
    )
    trainer = AlignPropTrainer(
        training_args,
        prompt_fn,
        pipeline,
        image_samples_hook=image_outputs_logger,
    )

    trainer.train()