from monai.networks.nets import SwinUNETR

def create_model(in_channels=4, out_channels=3, img_size=(128,128,128), feature_size=48, use_checkpoint=False):
    model = SwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
    )
    return model
