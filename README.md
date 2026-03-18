# image_lora
基于图像模型的LoRA训练


### 运行
1. 缓存Latents  
预计算所有图像的latent表示，将结果保存在cache_directory中

```
python qwen_image_cache_latents.py \
    --dataset_config /path/to/dataset_config.toml \
    --vae /path/to/qwen_image_vae.safetensors \
    --model_version edit-2511
```

2. 缓存文本编码器输出  
预计算文本编码器的输出
```
python qwen_image_cache_text_encoder_outputs.py \ 
    --dataset_config /path/to/dataset_config.toml \
    --text_encoder /path/to/qwen_2.5_vl_7b.safetensors \  
    --batch_size 1 \
    --model_version edit-2511 \


```

3. 启动 LoRA 训练

```
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    qwen_image_train_network.py \
    --dit /path/to/qwen_image_edit_2511_bf16.safetensors \
    --vae /path/to/qwen_image_vae.safetensors \
    --text_encoder /path/to/qwen_2.5_vl_7b.safetensors \
    --model_version edit-2511 \
    --dataset_config /path/to/dataset_config.toml \
    --sdpa \
    --mixed_precision bf16 \
    --timestep_sampling shift \
    --discrete_flow_shift 2.2 \
    --weighting_scheme none \
    --optimizer_type adamw8bit \
    --learning_rate 5e-5 \
    --gradient_checkpointing \
    --max_data_loader_n_workers 2 \
    --persistent_data_loader_workers \
    --network_module networks.lora_qwen_image \
    --network_dim 16 \
    --max_train_epochs 16 \
    --save_every_n_epochs 1 \
    --seed 42 \
    --output_dir /path/to/output \
    --output_name my_lora_name
```


