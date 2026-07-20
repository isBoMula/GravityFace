class cfg:
    # Margin Base Softmax
    margin_list = (1.0, 0.5, 0.0)

    # Partial FC
    sample_rate = 0.5
    interclass_filtering_threshold = 0
    bottle_neck = 32
    fp16 = True

    # For AdamW
    # optimizer = "adamw"
    # lr = 0.001
    # weight_decay = 0.1

    verbose = 16000
    frequent = 10

    # For Large Sacle Dataset, such as WebFace42M
    dali = False 
    dali_aug = False

    # Gradient ACC
    gradient_acc = 1

    # setup seed
    seed = 3407  #3407

    # For SGD 
    optimizer = "sgd"
    lr = 0.02
    momentum = 0.9
    weight_decay = 5e-4
    
    # dataload numworkers
    num_workers = 4
    batch_size = 128  # 128
    embedding_size = 512

    image_size = (112, 112)
    network = "r100"
    resume = False
    save_all_states = False
    device = "cuda"
    output = "Output/" 
    rec = "E:\Dataset\ms1mv3\ms1mv3.pickle"  #"E:\Dataset\ms1mv3\ms1mv3.pickle"   "C:\MS1MV2\ms1mv2.pkl"  "C:\WebFace\WebFace260M\WebFace4M.pkl"  "C:\CASIA-WebFace\casia_webface.pkl"
    val = "../Data/test"
    num_classes = 93431 #93431_v3   85742_v2  205990_4m  10572
    num_image = 5179510 #5179510_v3   5822653_v2  4235242_4m  490623
    num_epoch = 5
    steps_per_epoch = num_image // batch_size
    total_step = steps_per_epoch * num_epoch
    warmup_epoch = 0
    # ["lfw", "cplfw", "calfw", "cfp_ff", "cfp_fp", "agedb_30", "vgg2_fp"]
    # val_targets = ["lfw", "cfp_fp", "agedb_30"]
