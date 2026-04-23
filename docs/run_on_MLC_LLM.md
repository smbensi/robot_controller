1st command
```
 jetson-containers run \
  --env HUGGINGFACE_TOKEN=<HF_key> \
  -p 9000:9000 \
  -v /mnt/nvme/cache:/root/.cache \
  $(autotag mlc) \
  sudonim serve \
    --model mlc-ai/Qwen2.5-7B-Instruct-q4f16_1-MLC \
    --quantization q4f16_1 \
    --max-batch-size 1 \
    --prefill-chunk 2048 \
    --host 0.0.0.0 \
    --port 9000
```