
class Config:
    caption_data_path = ''
    img_path = ''
    img_festure_path = 'results.pth'
    scale_size = 300
    img_size = 224
    batch_size = 8
    shuffle = True
    num_workers = 4
    rnn_hidden = 256
    embedding_dim = 256
    num_layers = 2
    share_embedding_weights = False

    prefix = 'checkpoints/caption'

    env = 'caption'
    plot_every = 10
    debug_file = ''

    model_ckpt = None
    lr = 1e-3
    use_gpu = True
    epoch = 1
    test_img = 'img/example.jpg'