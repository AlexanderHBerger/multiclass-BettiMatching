from argparse import ArgumentParser
import json
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from monai.metrics import DiceMetric
import wandb
from typing import Tuple
import monai
from monai.data import list_data_collate
import yaml

from datasets.acdc import ACDC_ShortAxisDataset
from datasets.octa import Octa500Dataset
from datasets.platelet import PlateletDataset
from metrics.betti_error import BettiNumberMetric
from metrics.cldice import ClDiceMetric

parser = ArgumentParser()
parser.add_argument('--config',
                    default=None,
                    help='config file (.yaml) containing the hyper-parameters for training and dataset specific info.')
parser.add_argument('--model', default=None, help='checkpoint of the pretrained model')

class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)
        
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

def evaluate_model(
        model: torch.nn.Module, 
        data_loader: DataLoader, 
        device: torch.device, 
        logging: bool = False,
        mask_background: bool = False # This is only used for cases where the ground truth does not contain annotations for every pixel
    ) -> Tuple[float, float, float, float, float, float]:

    dice_metric = DiceMetric(include_background=False,
                                reduction="mean",
                                get_not_nans=False)
    clDice_metric = ClDiceMetric()
    betti_number_metric = BettiNumberMetric()
    
    model.eval()
    with torch.no_grad():
        test_images = None
        test_labels = None
        test_outputs = None

        for test_data in tqdm(data_loader):
            test_images, test_labels = test_data["img"].to(device), test_data["seg"].to(device)
            # convert meta tensor back to normal tensor
            if isinstance(test_images, monai.data.meta_tensor.MetaTensor): # type: ignore
                test_images = test_images.as_tensor()
                test_labels = test_labels.as_tensor()

            test_outputs = model(test_images)

            # Get the class index with the highest value for each pixel
            pred_indices = torch.argmax(test_outputs, dim=1)

            if mask_background:
                # Set all pixels to 0 where the ground truth is 0
                pred_indices[torch.argmax(test_labels, dim=1) == 0] = 0

            # Convert to onehot encoding
            one_hot_pred = torch.nn.functional.one_hot(pred_indices, num_classes=test_outputs.shape[1])

            # Move channel dimension to the second dim
            one_hot_pred = one_hot_pred.permute(0, 3, 1, 2)

            # compute metric for current iteration
            dice_metric(y_pred=one_hot_pred, y=test_labels)
            clDice_metric(y_pred=one_hot_pred, y=test_labels)
            betti_number_metric(y_pred=one_hot_pred, y=test_labels)

        # aggregate the final mean dice result
        dice_score = dice_metric.aggregate().item()
        clDice_score = clDice_metric.aggregate().item()
        b0, b1, bm, norm_bm = betti_number_metric.aggregate()

        if logging and test_images is not None and test_labels is not None and pred_indices is not None:
            class_labels = {
                0: "Zero",
                1: "One",
                2: "Two",
                3: "Three",
                4: "Four",
                5: "Five",
                6: "Six",
                7: "Seven",
                8: "Eight",
                9: "Nine",
                10: "Background"
            }
            mask_img = wandb.Image(test_images[0].cpu(), masks={
                "predictions": {"mask_data": pred_indices.cpu()[0].numpy(), "class_labels": class_labels},
                "ground_truth": {"mask_data": torch.argmax(test_labels[0].cpu(), dim=0).numpy(), "class_labels": class_labels},
            })
            wandb.log({
                "test/test_mean_dice": dice_score,
                "test/test_mean_cldice": clDice_score,
                "test/test_b0_error": b0,
                "test/test_b1_error": b1,
                "test/test_bm_loss": bm,
                "test/test_normalized_bm_loss": norm_bm,
                "test/test image": mask_img,
            })

        return dice_score, clDice_score, b0.item(), b1.item(), bm.item(), norm_bm.item()
    

if __name__ == "__main__":
    args = parser.parse_args()

    # if no model path is given, throw error
    if args.config is None:
        raise ValueError("Config file is required")
    
    # if no config file is given, throw error
    if args.model is None:
        raise ValueError("Pretrained model is required")

    # Load the config files
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2obj(config)

    # Set seeds
    torch.manual_seed(config.TRAIN.SEED)
    torch.cuda.manual_seed(config.TRAIN.SEED)

    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on", device)

    # Load the dataset
    if config.DATA.DATASET == 'octa500_6mm':
        test_dataset = Octa500Dataset(
            img_dir=config.DATA.TEST_PATH, 
            artery_dir=config.DATA.ARTERY_PATH, 
            vein_dir=config.DATA.VEIN_PATH, 
            augmentation=False, 
            max_samples=-1,
            rotation_correction=False # The 6mm dataset is not rotated
        )
    elif config.DATA.DATASET == 'topcow':
        raise Exception('ERROR: Dataset not implemented')
    elif config.DATA.DATASET == 'ACDC_sa':
        test_dataset = ACDC_ShortAxisDataset(
            img_dir=config.DATA.TEST_PATH,
            patient_ids=list(range(101, 151)),
            mean=74.29, 
            std=81.47, 
            augmentation=False, 
            max_samples=-1,
        )
    elif config.DATA.DATASET == 'platelet':
        test_dataset = PlateletDataset(
            img_file=os.path.join(config.DATA.DATA_PATH, "eval-images.tif"),
            label_file=os.path.join(config.DATA.DATA_PATH, "eval-labels.tif"),
            frame_ids=[],
            augmentation=False,
            patch_width=config.DATA.IMG_SIZE[0],
            patch_height=config.DATA.IMG_SIZE[1],
        )
    else:
        raise Exception('ERROR: Dataset not implemented')
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=config.TRAIN.NUM_WORKERS,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
        sampler=None,
        drop_last=False
    ) 

    # Create model
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=config.DATA.OUT_CHANNELS,
        channels=config.MODEL.CHANNELS,
        strides=[2] + [1 for _ in range(len(config.MODEL.CHANNELS) - 2)],
        num_res_units=config.MODEL.NUM_RES_UNITS,
    ).to(device)

    # Start from pretrained model
    dic = torch.load(args.model, map_location=device)
    model.load_state_dict(dic['model'], strict=True)
    
    dice_score, clDice_score, b0, b1, bm, norm_bm = evaluate_model(model, test_loader, device, logging=False, mask_background=False)

    # print results to file in model folder
    with open(os.path.join(os.path.dirname(args.model), "test_results.txt"), "w") as f:
        f.write(f"Dice score: {dice_score}\n")
        f.write(f"CLDice score: {clDice_score}\n")
        f.write(f"B0 error: {b0}\n")
        f.write(f"B1 error: {b1}\n")
        f.write(f"BM loss: {bm}\n")
        f.write(f"Normalized BM loss: {norm_bm}\n")

    # print results to console
    print(f"Dice score: {dice_score}")
    print(f"CLDice score: {clDice_score}")
    print(f"B0 error: {b0}")
    print(f"B1 error: {b1}")
    print(f"BM loss: {bm}")
    print(f"Normalized BM loss: {norm_bm}")