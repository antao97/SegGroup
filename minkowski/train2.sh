export BATCH_SIZE=8;
./scripts/train_scannet.sh 0 \
        -default \
        "--scannet_path /data4/taoan/Documents/Datasets/ScanNet_processed/mainseg_2
        --resume outputs/ScannetVoxelization2cmDataset/Res16UNet34C-b8-120000--default/2022-02-28_13-35-31"