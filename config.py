class Config:
    GPU = None

    RUN_NAME = None

    DATASET = None  # DAGM, KSDD2
    DATASET_PATH = None

    EPOCHS = None

    LEARNING_RATE = None

    BATCH_SIZE = None

    WEIGHTED_SEG_LOSS = None
    WEIGHTED_DEFECT = 1.0

    # Default values
    FOLD = None
    NUM_SEGMENTED = None
    RESULTS_PATH = "./RESULTS" # TODO use when releasing
    # RESULTS_PATH = "/home/jakob/outputs/WEAKLY_LABELED/PC_DEBUG" if "CONTAINER_NODE" in os.environ else "/opt/workspace/host_storage_hdd/REWRITE/v2"
    SPLITS_PATH = None

    VALIDATE = True
    VALIDATE_ON_TEST = True
    VALIDATION_N_EPOCHS = 5
    USE_BEST_MODEL = False

    ON_DEMAND_READ = False
    REPRODUCIBLE_RUN = True
    MEMORY_FIT = 1
    SAVE_IMAGES = True

    # Auto filled
    INPUT_WIDTH = None
    INPUT_HEIGHT = None
    INPUT_CHANNELS = None
    
    #Noisy
    NOISE_RATE = 0.5
    CLEAN_TRAIN = False
    #Co-train
    COTRAIN=False
    DROP_RATE = 0.5
    GMM_SINGLE = False
    #ADL
    ADL=False

    def init_extra(self):
        if self.NUM_SEGMENTED is None:
            raise Exception("Missing NUM_SEGMENTED!")
        if self.DATASET == 'KSDD2':
            self.INPUT_WIDTH = 232
            self.INPUT_HEIGHT = 640
            self.INPUT_CHANNELS = 3
            if self.NUM_SEGMENTED is None:
                raise Exception("Missing NUM_SEGMENTED for KSDD2 dataset!")
        elif self.DATASET == 'PCB':
            self.INPUT_WIDTH = 224
            self.INPUT_HEIGHT = 256
            self.INPUT_CHANNELS = 3
        elif self.DATASET == 'DAGM':
            self.INPUT_WIDTH = 256
            self.INPUT_HEIGHT = 256
            self.INPUT_CHANNELS = 1
        else:
            raise Exception('Unknown dataset {}'.format(self.DATASET))

    def merge_from_args(self, args):
        self.GPU = args.GPU
        self.RUN_NAME = args.RUN_NAME
        self.DATASET = args.DATASET
        self.DATASET_PATH = args.DATASET_PATH
        self.EPOCHS = args.EPOCHS
        self.LEARNING_RATE = args.LEARNING_RATE
        self.BATCH_SIZE = args.BATCH_SIZE
        self.WEIGHTED_SEG_LOSS = args.WEIGHTED_SEG_LOSS
        self.NUM_SEGMENTED = args.NUM_SEGMENTED

        if args.FOLD is not None: self.FOLD = args.FOLD
        if args.RESULTS_PATH is not None: self.RESULTS_PATH = args.RESULTS_PATH
        if args.VALIDATE is not None: self.VALIDATE = args.VALIDATE
        if args.VALIDATE_ON_TEST is not None: self.VALIDATE_ON_TEST = args.VALIDATE_ON_TEST
        if args.VALIDATION_N_EPOCHS is not None: self.VALIDATION_N_EPOCHS = args.VALIDATION_N_EPOCHS
        if args.USE_BEST_MODEL is not None: self.USE_BEST_MODEL = args.USE_BEST_MODEL
        if args.ON_DEMAND_READ is not None: self.ON_DEMAND_READ = args.ON_DEMAND_READ
        if args.REPRODUCIBLE_RUN is not None: self.REPRODUCIBLE_RUN = args.REPRODUCIBLE_RUN
        if args.MEMORY_FIT is not None: self.MEMORY_FIT = args.MEMORY_FIT
        if args.SAVE_IMAGES is not None: self.SAVE_IMAGES = args.SAVE_IMAGES
        if args.NOISE_RATE is not None: self.NOISE_RATE = args.NOISE_RATE
        if args.CLEAN_TRAIN is not None: self.CLEAN_TRAIN = args.CLEAN_TRAIN
        if args.WEIGHTED_DEFECT is not None: self.WEIGHTED_DEFECT = args.WEIGHTED_DEFECT
        if args.COTRAIN is not None: self.COTRAIN = args.COTRAIN
        if args.DROP_RATE is not None: self.DROP_RATE = args.DROP_RATE
        if args.GMM_SINGLE is not None: self.GMM_SINGLE = args.GMM_SINGLE
        if args.ADL is not None: self.ADL = args.ADL

    def get_as_dict(self):
        params = {
            "GPU": self.GPU,
            "DATASET": self.DATASET,
            "DATASET_PATH": self.DATASET_PATH,
            "EPOCHS": self.EPOCHS,
            "LEARNING_RATE": self.LEARNING_RATE,
            "BATCH_SIZE": self.BATCH_SIZE,
            "WEIGHTED_SEG_LOSS": self.WEIGHTED_SEG_LOSS,
            "WEIGHTED_DEFECT": self.WEIGHTED_DEFECT,
            "FOLD": self.FOLD,
            "NUM_SEGMENTED": self.NUM_SEGMENTED,
            "RESULTS_PATH": self.RESULTS_PATH,
            "VALIDATE": self.VALIDATE,
            "VALIDATE_ON_TEST": self.VALIDATE_ON_TEST,
            "VALIDATION_N_EPOCHS": self.VALIDATION_N_EPOCHS,
            "USE_BEST_MODEL": self.USE_BEST_MODEL,
            "ON_DEMAND_READ": self.ON_DEMAND_READ,
            "REPRODUCIBLE_RUN": self.REPRODUCIBLE_RUN,
            "MEMORY_FIT": self.MEMORY_FIT,
            "INPUT_WIDTH": self.INPUT_WIDTH,
            "INPUT_HEIGHT": self.INPUT_HEIGHT,
            "INPUT_CHANNELS": self.INPUT_CHANNELS,
            "SAVE_IMAGES": self.SAVE_IMAGES,
            "NOISE_RATE": self.NOISE_RATE,
            "CLEAN_TRAIN": self.CLEAN_TRAIN,
            "COTRAIN": self.COTRAIN,
            "DROP_RATE": self.DROP_RATE,
            "GMM_SINGLE": self.GMM_SINGLE,
            "ADL": self.ADL,
        }
        return params


def load_from_dict(dictionary):
    cfg = Config()

    cfg.GPU = dictionary.get("GPU", None)
    cfg.DATASET = dictionary.get("DATASET", None)
    cfg.DATASET_PATH = dictionary.get("DATASET_PATH", None)
    cfg.EPOCHS = dictionary.get("EPOCHS", None)
    cfg.LEARNING_RATE = dictionary.get("LEARNING_RATE", None)
    cfg.BATCH_SIZE = dictionary.get("BATCH_SIZE", None)
    cfg.WEIGHTED_SEG_LOSS = dictionary.get("WEIGHTED_SEG_LOSS", None)
    cfg.WEIGHTED_DEFECT = dictionary.get("WEIGHTED_DEFECT", None)
    cfg.FOLD = dictionary.get("FOLD", None)
    cfg.NUM_SEGMENTED = dictionary.get("NUM_SEGMENTED", None)
    cfg.RESULTS_PATH = dictionary.get("RESULTS_PATH", None)
    cfg.VALIDATE = dictionary.get("VALIDATE", None)
    cfg.VALIDATE_ON_TEST = dictionary.get("VALIDATE_ON_TEST", None)
    cfg.VALIDATION_N_EPOCHS = dictionary.get("VALIDATION_N_EPOCHS", None)
    cfg.USE_BEST_MODEL = dictionary.get("USE_BEST_MODEL", None)
    cfg.ON_DEMAND_READ = dictionary.get("ON_DEMAND_READ", None)
    cfg.REPRODUCIBLE_RUN = dictionary.get("REPRODUCIBLE_RUN", None)
    cfg.MEMORY_FIT = dictionary.get("MEMORY_FIT", None)
    cfg.INPUT_WIDTH = dictionary.get("INPUT_WIDTH", None)
    cfg.INPUT_HEIGHT = dictionary.get("INPUT_HEIGHT", None)
    cfg.INPUT_CHANNELS = dictionary.get("INPUT_CHANNELS", None)
    cfg.SAVE_IMAGES = dictionary.get("SAVE_IMAGES", None)
    cfg.NOISE_RATE = dictionary.get("NOISE_RATE", None)
    cfg.CLEAN_TRAIN = dictionary.get("CLEAN_TRAIN", None)
    cfg.COTRAIN = dictionary.get("COTRAIN", None)
    cfg.DROP_RATE = dictionary.get("DROP_RATE", None)
    cfg.GMM_SINGLE = dictionary.get("GMM_SINGLE", None)
    cfg.ADL = dictionary.get("ADL", None)

    return cfg
