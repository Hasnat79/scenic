import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
import ml_collections
from scenic.projects.vid2seq.configs.yttemporal import get_config
from flax.training import checkpoints
# from scenic.projects.vid2seq.datasets.dense_video_captioning_tfrecord_dataset import get_datasets


# def eval_only(
#     rng: np.ndarray, config: ml_collections.ConfigDict, *, workdir: str,
#     writer: Any, model_cls, dataset_dict
# ) -> Tuple[train_utils.TrainState, Dict[str, Any], Dict[str, Any]]:
#     return 
config = get_config()


print(config.model_name)

ds_name, cfg = config.datasets.items()[0]
print(ds_name)

modalities = config.get('modalities', ['features'])
print(modalities)

#restore checkpoint
checkpoint_path  = config.get('init_from.encoder.checkpoint_path')
print(checkpoint_path)
# train_state = checkpoints.restore_checkpoint(
#       checkpoint_path, train_state, step
#   )
#   return train_state, int(train_state.global_step)