gpus=0,1
name=semi+kd

mkdir -p ./recon_xray/logs/$name/

CUDA_VISIBLE_DEVICES=$gpus nohup python -u -m torch.distributed.launch \
    --master_port 1255 \
    --nproc_per_node 2 \
    ./recon_xray/train.py \
    --name $name \
    --ct_name semi \
    --dist \
    >> ./recon_xray/logs/$name/train.log 2>&1 &
