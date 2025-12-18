
## Training Example

### 1. training example for t2i sft or lora

- Data format

You need to create a jsonl file with key-values in the table below:

|    key_word       | Required  |  Description     |   Example   |
|:---------------:| :------:  |:----------------:|:-----------:|
|   `img_path`  | Required  |  image path               |     `./data_example/images/0.png`        | 
|   `prompt`     | Required  |    text               |  `A lovely little girl.` | 
|   `width`       | Required  |    image width    |     ` 1024 `       | 
|   `height`      | Required  |    image height   |     ` 1024 `       | 



- Tainging Scripts

```bash

bash ./train_examples/sft/train.sh

# All training setting in train_config.yaml

# --data_csv_root:  data csv_filepath
# --aspect_ratio_type: data bucketing strategy, mar_256、mar_512、mar_1024
# --pretrained_model_name_or_path: root directory of the model
# --diffusion_pretrain_weight: if a specified diffusion weight path is provided, load the model parameters from the current directory.
# --work_dir: the save root directory for ckpt and logs
# --resume_from_checkpoint: If 'resume_from_checkpoint' is set to 'latest', load the most recent step checkpoint. If a specific directory is provided, resume training from that directory.

```


### 2. training example for t2i dpo

- Data format

You need to create a txt file with key-values in the table below:


|    key_word       | Required  |  Description     |   Example   |
|:---------------:| :------:  |:----------------:|:-----------:|
|   `img_path_win`  | Required  |  win image path               |     `./data_example/images/0.png`        | 
|   `img_path_lose`  | Required  |  lose image path               |     `./data_example/images/1.png`        | 
|   `prompt`     | Required  |    text               |  `A lovely little girl.` | 
|   `width`       | Required  |    image width    |     ` 1024 `       | 
|   `height`      | Required  |    image height   |     ` 1024 `       | 


- Tainging Scripts

```bash

bash ./train_examples/dpo/train.sh

# All training setting in train_config.yaml

# --data_txt_root:  data txt_filepath
# --aspect_ratio_type: data bucketing strategy, mar_256、mar_512、mar_1024
# --pretrained_model_name_or_path: root directory of the model
# --diffusion_pretrain_weight: if a specified diffusion weight path is provided, load the model parameters from the current directory.
# --work_dir: the save root directory for ckpt and logs
# --resume_from_checkpoint: If 'resume_from_checkpoint' is set to 'latest', load the most recent step checkpoint. If a specific directory is provided, resume training from that directory.

```

### 3. training example for edit sft or lora

- Data format

You need to create a txt file with key-values in the table below:


|    key_word       | Required  |  Description     |   Example   |
|:---------------:| :------:  |:----------------:|:-----------:|
|   `img_path`  | Required  |  edited image path      |     `./data_example/images/0_edited.png`        | 
|   `ref_img_path`  | Required  |  raw image path     |     `./data_example/images/0.png`        | 
|   `prompt`     | Required  |    edit instruction     |  `change the dog to cat.` | 
|   `width`       | Required  |    image width    |     ` 1024 `       | 
|   `height`      | Required  |    image height   |     ` 1024 `       | 


- Tainging Scripts

```bash

bash ./train_examples/edit_sft/train.sh

# All training setting in train_config.yaml

# --data_txt_root:  data txt_filepath
# --aspect_ratio_type: data bucketing strategy, mar_256、mar_512、mar_1024
# --pretrained_model_name_or_path: root directory of the model
# --diffusion_pretrain_weight: if a specified diffusion weight path is provided, load the model parameters from the current directory.
# --work_dir: the save root directory for ckpt and logs
# --resume_from_checkpoint: If 'resume_from_checkpoint' is set to 'latest', load the most recent step checkpoint. If a specific directory is provided, resume training from that directory.

```

### 4. training example for edit dpo

- Data format

You need to create a txt file with key-values in the table below:


|    key_word       | Required  |  Description     |   Example   |
|:---------------:| :------:  |:----------------:|:-----------:|
|   `img_path_win`  | Required  |  win image path               |     `./data_example/images/0_win.png`        | 
|   `img_path_lose`  | Required  |  lose image path               |     `./data_example/images/0_lose.png`        | 
|   `ref_img_path`  | Required  |  ref image path     |     `./data_example/images/0_ref.png`        | 
|   `prompt`     | Required  |    text               |  `change the dog to cat.` | 
|   `width`       | Required  |    image width    |     ` 1024 `       | 
|   `height`      | Required  |    image height   |     ` 1024 `       | 


- Tainging Scripts

```bash

bash ./train_examples/edit_dpo/train.sh

# All training setting in train_config.yaml

# --data_txt_root:  data txt_filepath
# --aspect_ratio_type: data bucketing strategy, mar_256、mar_512、mar_1024
# --pretrained_model_name_or_path: root directory of the model
# --diffusion_pretrain_weight: if a specified diffusion weight path is provided, load the model parameters from the current directory.
# --work_dir: the save root directory for ckpt and logs
# --resume_from_checkpoint: If 'resume_from_checkpoint' is set to 'latest', load the most recent step checkpoint. If a specific directory is provided, resume training from that directory.

```
