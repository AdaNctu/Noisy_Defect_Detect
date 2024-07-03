from .input_ksdd2 import KSDD2Dataset
from .input_pcb import PCBDataset
from config import Config
from torch.utils.data import DataLoader
from typing import Optional


def get_dataset(kind: str, cfg: Config) -> Optional[DataLoader]:
    if kind == "VAL" and not cfg.VALIDATE:
        return None
    if kind == "VAL" and cfg.VALIDATE_ON_TEST:
        kind = "TEST"
    if cfg.DATASET == "KSDD2":
        ds = KSDD2Dataset(kind, cfg)
    elif cfg.DATASET == "PCB":
        ds = PCBDataset(kind, cfg)
    else:
        raise Exception(f"Unknown dataset {cfg.DATASET}")

    shuffle = kind == "TRAIN"
    batch_size = cfg.BATCH_SIZE if kind == "TRAIN" else 1
    num_workers = 0
    drop_last = kind == "TRAIN"
    pin_memory = False

    return DataLoader(dataset=ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, pin_memory=pin_memory)
