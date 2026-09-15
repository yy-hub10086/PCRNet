CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --master_port=25902 train.py \
--config ./configs/retrieval_icfg.yaml \
--output_dir output/PCRNet_OCC_ICFG \
--max_epoch 40 \
--batch_size_train 16 \
--batch_size_test 32 \
--init_lr 1e-5  \
--k_test 32 \
--epoch_eval 1 \
--distributed True

