class params():
    def __init__(self):
        self.init_dataset()
        self.init_training_hyperparams()
        self.init_model()

    def init_dataset(self):
        self.dataset_name = 'mp3d'
        self.image_root_dir = "data/mp3d_skybox"
        self.fov = 90
        self.rot = 45
        self.resolution = 512
        self.crop_size = 512
        self.seed = 0

    def init_training_hyperparams(self):
        self.lr = 0.0002
        self.max_epochs = None
        self.batch_size = None

    def init_model(self):
        self.model_type = 'pano_generation'
        self.guidance_scale = 9.0
        self.model_id = 'stabilityai/stable-diffusion-2-base'
        self.single_image_ft = False
        self.diff_timestep = 50
        self.model_ckpt_path = 'weights/model.ckpt'
