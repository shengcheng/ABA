def add_basic_args(parser):
    parser.add_argument('--data_name', type=str, default='pacs',
        choices=['digits', 'pacs', 'cifar10', 'officehome', 'living17', 'imagenet', 'wilds', 'camelyon17'],
        help='name of ssdg benchmark')
    parser.add_argument('--source', '-sc', type=str.lower, default='photo',
        help='souce domain for training')
    parser.add_argument('--feat', type=str, default='none',
        help='extractor for feature loss')
    parser.add_argument('--lr', '-lr', default=0.0001, type=float,
        help='learning rate')
    parser.add_argument('--gpu_ids', '-g', type=int, default=0,
        help='ids of GPUs to use')
    parser.add_argument('--n_epoch', '-ne', type=int, default=100,
        help='number of trainning epochs')
    # set training iterations when epoch does not exist
    parser.add_argument('--n_iter', '-ni', type=int, default=10000,
        help='number of total trainning iterations')
    parser.add_argument('--val_iter', '-vi', type=int, default=250,
        help='number of training iterations between two validations')
    parser.add_argument('--val_freq', type=int, default=3,
        help='validate every val_freq epochs')
    parser.add_argument('--viz_freq', type=int, default=100,
        help='visualize inputs/aug/batches after every 100 batches')
    parser.add_argument('--batch_size', '-bs', type=int, default=32,
        help='ids of GPUs to use')
    parser.add_argument('--rand_seed', '-rs', type=int,  default=1,
        help='random seed')
    parser.add_argument('--net', '-net', type=str, default='resnet18',
        help='network')
    parser.add_argument('--trans', '-trans', type=str, default='fcn',
        help='which transformation module to use')
    parser.add_argument('--activation', type=str, default='lrelu',
        help='non-linear activation in transnet')

    parser.add_argument('--grey', '-gr', action='store_true',
        help='using gray scale images')
    parser.add_argument('--SGD', '-sgd', action='store_true',
        help='use optimizer')
    parser.add_argument('--nesterov', '-nest', action='store_true',
        help='use nesterov momentum')
    parser.add_argument('--weight_decay', '-wd', default=1e-4, type=float,
        help='weight decay')
    parser.add_argument('--momentum', '-mmt', default=0.9, type=float,
        help='momentum')
    # parser.add_argument('--drop', default=0.5, type=float,
    #     help='dropout probability')

    parser.add_argument('--multi_aug', '-ma', action='store_true',
        help='strong data augmentations')
    parser.add_argument('--colorjitter', action='store_true', help='use cj')

    parser.add_argument('--scheduler', '-sch', type=str, default='',
        help='type of lr scheduler, StepLR/MultiStepLR/CosLR')
    parser.add_argument('--step_size', '-stp', type=int, default=30,
        help='fixed step size for StepLR')
    parser.add_argument('--milestones', type=int, nargs='+',
        help='milestone for MultiStepLR')
    parser.add_argument('--gamma', '-gm', type=float,  default=0.2,
        help='reduce rate for step scheduler')
    parser.add_argument('--power', '-power', default=0.9,
        help='power for poly scheduler')

    parser.add_argument('--image_size', type=int, default=224,
        help='resize input image size, -1 means keep original size')
    parser.add_argument('--n_classes', '-nc', type=int, default=7,
        help='number of classes')
    parser.add_argument('--name', type=str, default="debug")

    ### adv training
    parser.add_argument('--K', type=int, default=5,
        help='interval between augmentation epochs')
    parser.add_argument('--lr_adv', '-lr_adv', default=0.00001, type=float,
        help='learning rate')
    parser.add_argument('--fd_coeff', '-fd', default=0.05, type=float)
    parser.add_argument('--pre_epoch', type=int, default=2,
        help='number of epochs to pretrain on the source domains')
    parser.add_argument('--post_epoch', '-pe', type=int, default=15,
        help='number of epochs to use after augmentation is finished')
    parser.add_argument('--aug_percent', '-augp', default=0.2, type=float)
    parser.add_argument('--adv_steps', '-nadv', default=10, type=int)
    parser.add_argument('--tiny', action='store_true',
        help='for debugging')
    parser.add_argument('--test', '-test', action='store_true',
        help='run testing only')

    ## add_rand_layer_args
    parser.add_argument('--bal', action='store_true',
        help='use BCNN ALT for training')
    parser.add_argument('--elbo_beta', '-beta', default=0.1, type=float)
    parser.add_argument('--alt', action='store_true',
        help='use ALT for training')
    parser.add_argument('--append', action='store_true',
        help='append multiple augmented datasets')
    parser.add_argument('--combine', action='store_true',
        help='combine random and learned weights')
    parser.add_argument('--randconv', action='store_true',
                        help='use Randconv')
    parser.add_argument('--channel_size', '-chs', type=int, default=3,
        help='Number of output channel size  random layers, '
                        )
    parser.add_argument('--kernel_size', '-ks', type=int, default=[1, 3, 5, 7],
        nargs='+',
        help='kernal size for random layer, could be multiple kernels for multiscale mode')
    parser.add_argument('--distribution', '-db', type=str,
        default='kaiming',
        help='distribution of random sampling')
    parser.add_argument('--clamp_output', '-clamp', action='store_true',
        help='clamp value range of randconv outputs to a range (as in original image)')
    parser.add_argument('--mixing', '-mix', action='store_true',
        help='mix the output of rand conv layer with the original input')
    parser.add_argument('--affine', action='store_true',
        help='affine transformation of transformed input')
    # parser.add_argument('--alpha', type=float, default=None,
    #     help='mixing weight')
    parser.add_argument('--identity_prob', '-idp', type=float, default=0.0,
        help='the probability that the rand conv is a identity map, '
            'in this case, the output and input must have the same channel number')
    parser.add_argument('--rand_freq', '-rf', type=int, default=1,
        help='frequency of randomize weights of random layers (every n steps)')
    parser.add_argument('--trans_depth', type=int, default=4,
        help='get outputs from a random intermediate layer of transnet')
    parser.add_argument('--augmix', '-am', action='store_true',
        help='aug_mix mode, only raw data is used to compute cls loss')
    parser.add_argument('--n_val', '-nv', type=int, default=1,
        help='repeat validation with different randconv')
    parser.add_argument('--val_with_rand', '-vwr', action='store_true',
        help='validation with random conv;')
    parser.add_argument('--test_latest', '-tl', action='store_true',
        help='test the last saved model instead of the best one')
    parser.add_argument('--test_target', '-tt', action='store_true',
        help='test the best model on target domain')
    parser.add_argument('--ensemble', type=int, default=0,
        help='number of alpha, beta, nu ensembles to create')
    parser.add_argument('--ens_pert', action='store_true')
    parser.add_argument('--ens_dist', type=str, default='uniform')
    parser.add_argument('--ensw', type=int, default=10)
    parser.add_argument('--alpha_init', type=float, default=0.5)
    parser.add_argument('--cl', action='store_true', help='consistency loss')
    parser.add_argument('--clw', type=float, default=1.0,
        help='weight for invariant loss')
    parser.add_argument('--toss', type=float, default=0.0)
    parser.add_argument('--wr', type=float, default=1.0)
    parser.add_argument('--temp', type=float, default=1.0,
        help='temperature scaling for g() softmax')
    parser.add_argument('--sag', action='store_true',
                        help='use Sagnet')
    parser.add_argument('--sag_w_adv', type=float, default=0.1)
    parser.add_argument('--sag_clip_adv', type=float, default=0.1)
    parser.add_argument('--with_randconv', action='store_true', help='add randconv for adv')

    parser.add_argument('--augmax', action='store_true', help='use augmax?')
    parser.add_argument('--mixture_width', type=int, default=3, help='Number of augmentation chains to mix per augmented example')
    parser.add_argument('--mixture_depth', type=int, default=-1,
                        help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
    parser.add_argument('--aug_severity', type=int, default=3, help='Severity of base augmentation operators')