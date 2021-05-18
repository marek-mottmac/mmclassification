# dataset settings
dataset_type = 'CocoStyle'
img_norm_cfg = dict(
    mean=[128.0,128.0,128.0], std=[64.0,64.0,64.0], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='Resize', size=(256, -1)),
    #dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_prefix= r'D:\ash_dev\data\cropped_trees_gopro',
        ann_file= r'D:\ash_dev\data\annotation_files\cropped_trees_gopro_train_210518.json',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix=r'D:\ash_dev\data\cropped_trees_gopro',
        ann_file= r'D:\ash_dev\data\annotation_files\cropped_trees_gopro_val_210518.json',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix=r'D:\ash_dev\data\cropped_trees_gopro',
        ann_file= r'D:\ash_dev\data\annotation_files\cropped_trees_gopro_test_210518.json',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')
