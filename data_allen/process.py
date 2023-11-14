import _pickle as cpickle
from pathlib import Path
from loguru import logger
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_dataset", type=str, default="mini_imagenet")
    parser.add_argument("--tgt_dataset", type=str, default="mini_imagenet")
    parser.add_argument("--backbone", type=str, default="resnet12") #wrn2810
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_source", type=str, default="feat")
    parser.add_argument("--training", type=str, default="standard")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--override", type=str2bool, default="True")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--keep_all_train_features", type=bool, default=False)
    parser.add_argument("--debug", type=str2bool, default="False")

    args = parser.parse_args()
    return args


def main(args):
    stem = f"{args.backbone}_{args.src_dataset}_{args.model_source}"  # used for saving features downstream
    pickle_name = Path(stem).with_suffix(f".pickle").name
    output_file = (
        Path("./")
        / "features"
        / args.src_dataset
        / args.tgt_dataset
        / args.split
        / args.training
        / pickle_name
    )
    print(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # First checking whether those features already exist
    if output_file.exists():
        logger.info(f"File {output_file} already exists.")
        if args.override:
            logger.warning("Overriding.")
        else:
            logger.warning("Not overriding.")
            return
    else:
        logger.info(f"File {output_file} does not exist. Performing extraction.")
    logger.info("Computing features...")
    # features = None
    # labels = None

    # # if output_file is None:


    # packed_features = {
    #         class_integer_label: features[labels == class_integer_label]
    #         for class_integer_label in labels.unique()
    #     }
    import torch
    import numpy as np
    with open("/home/yifei/workspace/few-shot-open-set/data_allen/miniImagenet/resnet12_S2M2_R/last/output.plk", "rb") as f:
    # with open("/home/yifei/workspace/few-shot-open-set/data_allen/miniImagenet/WideResNet28_10_S2M2_R/last/output.plk", "rb") as f:
        packed_features = cpickle.load(f)
        new_packed_features = {}
        for k, v in packed_features.items():
            vt = torch.FloatTensor(np.array(v))
            new_packed_features[k] = vt

    with open(output_file, "wb") as stream:
            cpickle.dump(new_packed_features, stream, protocol=-1)
   
    logger.info(f"Dumped features in {output_file}")
    import pickle
    with open(output_file, "rb") as f:
            features = pickle.load(f)
            # print(type(features[96]))
            # print(type(features[96][0]))


if __name__ == "__main__":
    args = parse_args()
    main(args)

