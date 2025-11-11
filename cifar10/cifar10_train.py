import os
import sys
current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)
from network import kan

# Train on MNIST
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import datetime
import time
from optim import TJU_v1,TJU_v3,TJU_v4


# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR10')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--eval_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for eval (default: 1000)')
    parser.add_argument('--width', type=str, default='32*32*3, 128, 10', help='network width (comma-separated list)')
    parser.add_argument('--grid_size', type=int, default=5, help='grid size')
    parser.add_argument('--spline_order', type=int, default=3, help='parameter k')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--cuda', type=bool, default=True, help='Whether to use cuda')
    parser.add_argument('--device', type=str, default='cuda', help='device to use for training')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train')
    parser.add_argument('--opts', type=str, nargs='+', choices=['sgd', 'adam', 'adamW','TJU_v1','TJU_v3','TJU_v4'],
                                                default=['adamW'], help='optimizers to use')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate', required=False)
    parser.add_argument('--warmup_updates', type=int, default=500, metavar='N',
                        help='number of updates to warm up (default: 0)')
    parser.add_argument('--init_lr', type=float, default=1e-5, help='initial learning rate')
    parser.add_argument('--opt_h1', type=float, default=0.9, help='momentum for SGD, beta1 of Adam or beta for QGS')
    parser.add_argument('--opt_h2', type=float, default=0.999, help='beta2 of Adam or RAdam')
    parser.add_argument('--eps', type=float, default=1e-4, help='eps of Adam')
    parser.add_argument('--weight_decay', type=float, default=2.5e-4, help='weight for l2 norm decay')
    parser.add_argument('--weight_decay_type', choices=['L2', 'decoupled', 'stable'], default='L2', help='type of weight decay')
    parser.add_argument('--rebound', choices=['constant', 'belief'], default='constant', help='type of recified bound of diagonal hessian')
    parser.add_argument('--dataset', choices=['mnist', 'cifar10', 'cifar100'],default='cifar10', required=False)
    parser.add_argument('--workers', default=2, type=int, metavar='N', help='number of data loading workers (default: 2)')
    parser.add_argument('--log', default=10, type=int, metavar='N', help='Print the log information during training')
    parser.add_argument('--log_interval', default=10, type=int, metavar='N', help='Print the log information during training')
    parser.add_argument('--data_path', help='path for data file.', default='data/', required=False)
    parser.add_argument('--model_path', help='path for saving model file.',
                        default='data/model-' + str(
                            datetime.datetime.now().strftime('%Y-%m-%d-%H')) + '/', required=False)

    return parser.parse_args()

def write_result(data,file_name,txt_name):
    file_path = os.path.join(file_name, f'{txt_name}.txt')
    # Ensure that the directory exists
    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    r=str(data) + '\n'
    with open('{}/{}.txt'.format(file_name,txt_name),'w') as wr:
        wr.write(r)

def parse_width(width_str):
    # allowed_names = {"__builtins__": None}
    # return [eval(dim, allowed_names) for dim in width_str.split(',')]
    network_width = []
    for dim in width_str.split(','):
        if '*' in dim:
            factors = dim.split('*')
            product = 1
            for factor in factors:
                product *= int(factor.strip())
            network_width.append(product)
        else:
            network_width.append(int(dim.strip()))
    return network_width

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def logging(info, logfile=None):
    print(info)
    if logfile is not None:
        print(info, file=logfile)
        logfile.flush()

# Obtain the optimizer function
def get_optimizer(opt, learning_rate, parameters, hyper1, hyper2, eps, rebound,
                    weight_decay, weight_decay_type,
                  warmup_updates, init_lr):
    if opt == 'sgd':
        optimizer = optim.SGD(parameters, lr=learning_rate, momentum=hyper1, weight_decay=weight_decay, nesterov=True)
        opt = 'momentum=%.1f, ' % (hyper1)
        weight_decay_type = 'L2'
    elif opt == 'adam':
        optimizer = optim.Adam(parameters, lr=learning_rate, betas=(hyper1, hyper2), eps=eps, weight_decay=weight_decay)
        opt = 'betas=(%.1f, %.3f), eps=%.1e, ' % (hyper1, hyper2, eps)
        weight_decay_type = 'decoupled'
    elif opt == 'TJU_v3':
        # TJU_v3
        optimizer = TJU_v3(parameters, lr=learning_rate, betas=(hyper1, hyper2),
                           eps=eps, rebound=rebound, warmup=warmup_updates, init_lr=init_lr,
                           weight_decay=weight_decay, weight_decay_type=weight_decay_type)
    elif opt == 'adamW':
        optimizer = optim.AdamW(parameters,lr=learning_rate,betas=(hyper1, hyper2),eps=eps,weight_decay=weight_decay)
    elif opt=='TJU_v4':
        # TJU_v4 jzj
        # weight_decay = 0.01
        optimizer = TJU_v4.TJU_v4(parameters, lr=learning_rate, betas=(hyper1, hyper2),
                           eps=eps, rebound=rebound, warmup=warmup_updates, init_lr=init_lr,
                           weight_decay=weight_decay, weight_decay_type='AdamW')
        opt = f'betas=({hyper1:.1f}, {hyper2:.3f}), eps={eps:.1e}, '
        weight_decay_type = 'AdamW'
    elif opt == 'TJU_v1':
        # TJU_v1 jzj
        optimizer = TJU_v1(parameters, lr=learning_rate, eps=eps, rebound=rebound, warmup=warmup_updates, init_lr=init_lr,
                           weight_decay=weight_decay, weight_decay_type=weight_decay_type)
        #
        opt = f'betas=({hyper1:.1f}, {hyper2:.3f}), eps={eps:.1e}, '
        weight_decay_type = 'decoupled'
    else:
        raise ValueError(f'Unknown optimizer: {opt}')
    
    opt += 'warmup={}, init_lr={:.1e}, wd={:.1e} ({})'.format(warmup_updates, init_lr, weight_decay, weight_decay_type)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    return optimizer, scheduler, opt

class KAN_CIFAR_model(nn.Module):
    def __init__(self, input_size=None, width=None, device=None) -> None:
        super(KAN_CIFAR_model, self).__init__()
        self.input_size = input_size
        self.width = width
        self.device = device  
        
        # Build a KAN network using the passed width parameter
        self.model = torch.nn.Sequential(
            kan.KAN(width)  #
        ).to(device)
        
    def forward(self, X):
        X = X.view(-1, self.input_size)
        return self.model(X)

def setup(args):
    dataset = args.dataset
    data_path = args.data_path

    model_path = args.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    args.cuda = torch.cuda.is_available()
    device = torch.device('cuda', 0) if args.cuda else torch.device('cpu')

    if args.cuda:
        torch.cuda.set_device(device)

    if dataset == 'mnist':
        dataset = datasets.MNIST
        num_classes = 10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = dataset(data_path, train=True, download=True,
                       transform=transform)
        valset = dataset(data_path, train=False, download=False,
                     transform=transform)

        # Analyze the network architecture list
        width = parse_width(args.width)
        model = kan.KAN(layers_hidden=width, grid_size=args.grid_size, spline_order=args.spline_order)
        model.to(device)
        args.device = device

        return args, (trainset, valset, len(trainset), len(valset)), model

    elif dataset == 'cifar10':
        dataset = datasets.CIFAR10
        num_classes = 10
        transform = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  
        ])
        trainset = dataset(data_path, train=True, download=True,
                       transform=transform)
        valset = dataset(data_path, train=False, download=False,
                     transform=transform)
        
        # Analyze the network architecture list
        width = parse_width(args.width)
        model = KAN_CIFAR_model(input_size=width[0],width=width,device=device)
        model.to(device)
        args.device = device

        return args, (trainset, valset, len(trainset), len(valset)), model
    else:
        dataset = datasets.CIFAR100
        num_classes = 100
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        trainset = dataset(data_path, train=True, download=True,
                       transform=transform)
        valset = dataset(data_path, train=False, download=False,
                     transform=transform)
        
        # Analyze the network architecture list
        width = parse_width(args.width)
        model = KAN_CIFAR_model(input_size=width[0],width=width,device=device)
        model.to(device)
        args.device = device

        return args, (trainset, valset, len(trainset), len(valset)), model
    

def init_dataloader(args, trainset, valset):
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(valset, batch_size=args.eval_batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader

def train(args, trainloader,  model, criterion, optimizer, num_train):
    model.train()
    start_time = time.time()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    num_back = 0

    device = args.device
    dataset = args.dataset

    if args.cuda:
        torch.cuda.empty_cache()

    for i, (images, labels) in enumerate(trainloader):
        if dataset == 'mnist':
            images = images.view(-1, 28 * 28).to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        else:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        # images = images.view(-1, 28 * 28).to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, labels.data, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        top5.update(prec5.item(), images.size(0))

        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            sys.stdout.write("\b" * num_back)
            sys.stdout.write(" " * num_back)
            sys.stdout.write("\b" * num_back)
            log_info = '[{}/{} ({:.0f}%)] loss: {:.4f}, top1: {:.2f}%, top5: {:.2f}%'.format(
                losses.count, num_train, 100. * losses.count / num_train,
                losses.avg, top1.avg, top5.avg)
            sys.stdout.write(log_info)
            sys.stdout.flush()
            num_back = len(log_info)

    sys.stdout.write("\b" * num_back)
    sys.stdout.write(" " * num_back)
    sys.stdout.write("\b" * num_back)
    # logging('Train_loss_average: {:.4f}, top1: {:.2f}%, top5: {:.2f}%, time: {:.1f}s'.format(
    # losses.avg, top1.avg, top5.avg, time.time() - start_time), args.log)

    return losses.avg, top1.avg, top5.avg

def eval(args, val_loader, model, criterion):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    device = args.device
    dataset = args.dataset
    if args.cuda:
        torch.cuda.empty_cache()

    for data, y in val_loader:
        if dataset == 'mnist':
            data = data.view(-1, 28 * 28).to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
        else:
            data = data.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
        outputs = model(data)
        loss = criterion(outputs, y)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, y.data, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(prec1.item(), data.size(0))
        top5.update(prec5.item(), data.size(0))

    logging('Test_loss_average:  {:.4f}, top1: {:.2f}%, top5: {:.2f}%'.format(
        losses.avg, top1.avg, top5.avg), None)

    return losses.avg, top1.avg, top5.avg


# main
def main(args,model_path):

    args.model_path = model_path
    args, (trainset, valset, num_train, num_val), model = setup(args)

    criterion = nn.CrossEntropyLoss()
    model_name = 'model_weights.pth'
    filt_name = os.path.join(args.model_path, model_name)
    torch.save(model.state_dict(), filt_name)

    train_loader, val_loader = init_dataloader(args, trainset, valset)
    
    epochs = args.epochs
    # log = args.log

    #opt = args.opts
    lr_warmup = args.warmup_updates
    init_lr = args.init_lr
    hyper1 = args.opt_h1
    hyper2 = args.opt_h2
    eps = args.eps
    rebound = args.rebound
    weight_decay = args.weight_decay
    weight_decay_type = args.weight_decay_type

    result = {}
    for idx, opt in enumerate(args.opts):
        model.load_state_dict(torch.load(filt_name))
        logging('# of Parameters: %d' % sum([param.numel() for param in model.parameters()]), None)
        numbers = {'train loss': [], 'train acc': [], 'test loss': [], 'test acc': []}

        optimizer, scheduler, opt_param = get_optimizer(opt, args.lr, model.parameters(), hyper1, hyper2, eps, rebound,
                                                        weight_decay=weight_decay, weight_decay_type=weight_decay_type,
                                                        warmup_updates=lr_warmup, init_lr=init_lr)

        best_epoch = 0
        best_top1 = 0
        best_top5 = 0
        best_loss = 0
        ###load Netpara
        # dic = torch.load(args.model_path + 'model_QGS.pkl')
        # model.load_state_dict(dic)

        for epoch in range(1, epochs + 1):
            lr = scheduler.get_last_lr()[0]
            logging('Epoch: {}/{} ({}, lr={:.6f}, {})'.format(epoch, epochs, opt, lr, opt_param), None)

            train_loss, train_top1, train_top5 = train(args, train_loader, model , criterion, optimizer,num_train)
            scheduler.step()

            with torch.no_grad():
                loss, top1, top5 = eval(args, val_loader, model, criterion)

            if top1 > best_top1:
                best_top1 = top1
                best_top5 = top5
                best_loss = loss
                best_epoch = epoch

            logging('Best_Test_loss:     {:.4f}, top1: {:.2f}%, top5: {:.2f}%, epoch: {}'.format(
                best_loss, best_top1, best_top5, best_epoch))

            numbers['train loss'].append(train_loss)
            numbers['test loss'].append(loss)
            numbers['train acc'].append(train_top1)
            numbers['test acc'].append(top1)

        result[opt] = numbers

        #save data in txt---JAshen
        write_result(result[opt],'result',f'{opt}')
        print(f'{opt}_file saved')

        current_time = datetime.datetime.now().strftime('%H-%M-%S')
        torch.save(model.state_dict(), args.model_path + 'mnist_{}_model_{}.pkl'.format(opt, current_time))
            #json.dump(numbers, open(os.path.join(args.model_path, 'values.run{}.json'.format(args.run)), 'w'))
            #####save model
            # if epoch % 50 == 0:
            #     torch.save(model.state_dict(), args.model_path + 'model.pkl')
    current_time = datetime.datetime.now().strftime('%H-%M-%S')
    torch.save(result, args.model_path + 'mnist_result_{}.pkl'.format(current_time))

    #torch.save(model.state_dict(), args.model_path + 'cifar10_model.pkl')

    #print(numbers['train loss'])
  


if __name__ == '__main__':
    args = parse_args()
    args.epochs = 300
    model_path = args.model_path
    main(args,model_path)

