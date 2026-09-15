CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --master_port=25902 train.py \
--config ./configs/retrieval_cuhk.yaml \
--output_dir output/PCRNet_OCC \
--batch_size_test 32 \
--k_test 32 \
--pretrained ./output/PCRNet_OCC/checkpoint_best.pth \
--evaluate
