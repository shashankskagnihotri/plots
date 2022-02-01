import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy
from pruneshift.datamodules import ShiftDataModule as dm
from torch.utils.data import DataLoader

from pruneshift.networks import create_network

from torchvision.datasets import ImageFolder
from torchvision import transforms

from scipy.stats import rankdata
from video_loader import VideoFolder




class_ids=['n02395406','n02108089','n02356798','n02106662','n01983481','n02112018','n01784675','n02138441','n02088238','n02113624',
           'n01770393','n01616318','n02130308','n01644373','n02106550','n02364673','n01498041','n02009912','n02098286','n02268443',
           'n02110341','n02206856','n02110185','n02088466','n01531178','n02092339','n02190166','n02398521','n01882714','n02112137',
           'n02088094','n02119022','n01534433','n02128385','n02134084','n02077923','n01518878','n01484850','n02099601','n01855672',
           'n02106030','n02317335','n02346627','n02096585','n02109525','n02056570','n02066245','n01820546','n01860187','n02102318',
           'n02071294','n02091134','n02423022','n01833805','n02108915','n02123045','n02279972','n02114367','n01806143','n01514859',
           'n02391049','n02236044','n01910747','n01677366','n01986214','n01774750','n02226429','n02110958','n01630670','n02051845',
           'n01843383','n01632777','n02086240','n02097298','n02106166','n02007558','n02088364','n01694178','n02129604','n02410509',
           'n02128757','n01494475','n02117135','n01748264','n02219486','n02165456','n01944390','n02094433','n02085620','n02091032',
           'n02099712','n02113799','n01443537','n01847000','n02129165','n02363005','n01614925','n02325366','n02113023','n02233338']



path="/work/dlclarge2/agnihotr-shashank-pruneshift/hoffmaja-pruneshift/datasets/ILSVRC2012-100/val"

path_corrupted="/work/dlclarge2/agnihotr-shashank-pruneshift/hoffmaja-pruneshift/datasets/ILSVRC2012-100/corrupted"

path_pertubations="/work/dlclarge1/agnihotr-ensemble/ImageNet-100-P/"

pertubations=['snow','shot_noise','scale','gaussian_noise', 'tilt',
              'motion_blur','brightness','rotate','zoom_blur', 'translate']

data_transform = transforms.Compose([transforms.Resize(256),
        transforms.CenterCrop(224),        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


datasets = ImageFolder(root=path, transform=data_transform)
dataset_loader = DataLoader(datasets, batch_size=32, shuffle=False, num_workers=4, drop_last=False, pin_memory=False)

dataset_loader_test = DataLoader(datasets, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=False)

distortions_list = [
        "brightness",
        "contrast",
        "defocus_blur",
        "elastic_transform",
        "fog",
        "frost",
        "gaussian_noise",
        "glass_blur",
        "impulse_noise",
        "jpeg_compression",
        "motion_blur",
        "pixelate",
        "shot_noise",
        "snow",
        "zoom_blur",
    ]


path_b1="/work/dlclarge2/agnihotr-shashank-pruneshift/hoffmaja-pruneshift/datasets/ILSVRC2012-100/corrupted/brightness/1"
datasets_b1 = ImageFolder(root=path_b1, transform=data_transform)
dataset_loader_b1 = DataLoader(datasets_b1, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=False)


path_b3="/work/dlclarge2/agnihotr-shashank-pruneshift/hoffmaja-pruneshift/datasets/ILSVRC2012-100/corrupted/brightness/3"
datasets_b3 = ImageFolder(root=path_b3, transform=data_transform)
dataset_loader_b3 = DataLoader(datasets_b3, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=False)


path_b5="/work/dlclarge2/agnihotr-shashank-pruneshift/hoffmaja-pruneshift/datasets/ILSVRC2012-100/corrupted/brightness/5"
datasets_b5 = ImageFolder(root=path_b5, transform=data_transform)
dataset_loader_b5 = DataLoader(datasets_b5, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=False)


path_r="/work/dlclarge2/agnihotr-shashank-pruneshift/hoffmaja-pruneshift/datasets/ILSVRC2012-100/renditions"
datasets_r = ImageFolder(root=path_r, transform=data_transform)
dataset_loader_r = DataLoader(datasets_r, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=False)


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
device = torch.device(dev)
cuda = torch.device('cuda')

def dist(sigma, mode='top5'):
    if mode == 'top5':
        return np.sum(np.abs(cum_sum_top5[:5] - cum_sum_top5[sigma-1][:5]))
    elif mode == 'zipf':
        return np.sum(np.abs(recip - recip[sigma-1])*recip)


def ranking_dist(ranks, noise_perturbation=False, mode='top5'):
    result = 0
    step_size = 1

    for vid_ranks in ranks:
        result_for_vid = []

        for i in range(step_size):
            perm1 = vid_ranks[i]
            perm1_inv = numpy.argsort(perm1)

            for rank in vid_ranks[i::step_size][1:]:
                perm2 = rank
                result_for_vid.append(dist(perm2[perm1_inv], mode))
                if not noise_perturbation:
                    perm1 = perm2
                    perm1_inv = numpy.argsort(perm1)

        result += numpy.mean(result_for_vid) / len(ranks)

    return result


def flip_prob(predictions, noise_perturbation= False):
    result = 0
    step_size = 1

    for vid_preds in predictions:
        result_for_vid = []

        for i in range(step_size):
            prev_pred = vid_preds[i]

            for pred in vid_preds[i::step_size][1:]:
                result_for_vid.append(int(prev_pred != pred))
                if not noise_perturbation: prev_pred = pred

        result += numpy.mean(result_for_vid) / len(predictions)

    return result



def calculate_on_per(net, multiheaded):
    mean_flip_prob = 0
    mean_ece = 0
    
    mean_accu=0
    length=0
    net.eval()
    net.to(device)
    net.half()
    for per in pertubations:
        
        predictions, ranks = [], []        
        temp=path_pertubations + per
        print(temp)
        #print(temp.__class__)
        
        with torch.no_grad():
            loader = torch.utils.data.DataLoader(VideoFolder(root=temp,
                                                             transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)])),
                                                 batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

            mean_acc = []
            size=[]
            mean_diff= []
            
            for data, target in loader:
                num_vids = data.size(0)
                data = data.view(-1,3,224,224).cuda()
                target = target.cuda()
                target.half()        
                output = net(data.half())
                if multiheaded:
                    output=output[-1]        
                confidence = 0
                accuracy = 0
        
                for vid, label in zip(output.view(num_vids, -1, 100), target):
                    predictions.append(vid.argmax(1).to('cpu').numpy())
                    prob = F.softmax(vid, dim=1)
                    top_p, top_class = prob.topk(1, dim = 1)                    
                    pred=torch.mode(top_class).values
                    accuracy += (vid.argmax(1).mode().values==label).sum() #/ float(len(loader.dataset))
                    confidence += top_p.data.max()#/float(len(loader.dataset))                       
                mean_accuracy=accuracy/num_vids
                #mean_accuracy = torch.FloatTensor(mean_accuracy)
                mean_acc.append(mean_accuracy)
                mean_confidence = confidence/num_vids
                mean_diff.append(torch.abs((mean_accuracy - mean_confidence)))
                size.append(num_vids)
        
            assert len(mean_diff)==len(size)
            multiplied = []
            for i in range(len(size)):
                multiplied.append(mean_diff[i]*size[i])
            ece = torch.FloatTensor(multiplied).sum()/torch.FloatTensor(size).sum()
            mean_ece += ece
            mean_acc = torch.FloatTensor(mean_acc).cuda()
            mean_accu += mean_acc.mean()
        
        #import pdb;pdb.set_trace()
        #ranks = numpy.asarray(ranks)
        mean_flip_prob += flip_prob(predictions)
        length +=1
        #print('Computing Metrics\n')
        
        #print('Flipping Prob\t{:.5f}'.format(flip_prob(predictions)))
        #print('Top5 Distance\t{:.5f}'.format(ranking_dist(ranks, mode='top5')))
        #print('Zipf Distance\t{:.5f}'.format(ranking_dist(ranks, mode='zipf')))
    print('error: ', 1-(mean_accu/length))
    print('mFR: ', mean_flip_prob/length)
    print('mECE: ', mean_ece/length)
    alex_flip_prob=0.0512405
    mean_flip_rate= mean_flip_prob/length
    return mean_flip_rate, mean_ece/length, 1-(mean_accu/length)

def calculate_ece_corr(net, multiheaded):
    length=0
    mean_ece = 0
    mean_acc = 0
    net.eval()
    net.to(device)
    net.half()
    mean_accuracy=[]
    
    for corr in distortions_list:
        for i in range(1,6):
            path_dataset = path_corrupted + '/'+ corr + '/' + str(i)
            dataset_corrupted = ImageFolder(root=path_dataset, transform=data_transform)
            dataset_loader_corr = DataLoader(dataset_corrupted, batch_size=32, shuffle=False, num_workers=4, drop_last=False, pin_memory=False)
            
            mean_diff = []
            size=[]
            
            for data, target in dataset_loader_corr:
                accuracy = 0
                confidence = 0
                with torch.no_grad():
                    data, target = data.cuda(), target.cuda()
                    target.half()
                    #data, target = Variable(data, volatile=True), Variable(target)            
                    output = net(data.half())
                if multiheaded:
                    output=output[-1]
                pred = output.data.max(1, keepdim = True)
                prob = F.softmax(output, dim=1)
                top_p, top_class = prob.topk(1, dim = 1)
                accuracy = (top_class.view_as(target)==target).sum()# / float(len(dataset_loader_corr.dataset))
                confidence = top_p.data.sum()#/float(len(dataset_loader_corr.dataset))
                mean_diff.append(torch.abs(accuracy - confidence))
                size.append(data.shape[0])
                mean_accuracy.append(accuracy/data.shape[0])
            assert len(mean_diff)==len(size)
            multiplied = []
            for i in range(len(size)):
                multiplied.append(mean_diff[i]*size[i])
            ece = torch.FloatTensor(multiplied).sum()/torch.FloatTensor(size).sum()
            #print('path: ', path_dataset)
            #print('length: ', length)
            mean_ece += ece
            #mean_acc += accuracy
            length += 1
    print('mCE: ', 1- (torch.FloatTensor(mean_accuracy).mean()))
    return mean_ece/length, 1- (torch.FloatTensor(mean_accuracy).mean())

def calculate_ece(net, multiheaded):
    net.eval()
    net.to(device)
    net.half()
    mean_diff = []
    size=[]
    mean_accuracy=[]    
    for data, target in dataset_loader:
        accuracy = 0
        confidence = 0
        with torch.no_grad():
            data, target = data.cuda(), target.cuda()
            target.half()
            #data, target = Variable(data, volatile=True), Variable(target)            
            output = net(data.half())
        if multiheaded:
            output=output[-1]
        pred = output.data.max(1, keepdim = True)
        prob = F.softmax(output, dim=1)
        top_p, top_class = prob.topk(1, dim = 1)
        accuracy = (top_class.view_as(target)==target).sum()# / float(len(dataset_loader.dataset))
        confidence = top_p.data.sum()#/float(len(dataset_loader.dataset))
        mean_diff.append(torch.abs(accuracy - confidence))
        mean_accuracy.append(accuracy/data.shape[0])
        size.append(data.shape[0])
    #import ipdb;ipdb.set_trace()
    assert len(mean_diff)==len(size)
    multiplied = []
    for i in range(len(size)):
        multiplied.append(mean_diff[i]*size[i])
    ece =torch.FloatTensor(multiplied).sum()/torch.FloatTensor(size).sum()
    print('ece: ', ece)
    print('err: ', 1-(torch.FloatTensor(mean_accuracy).mean()))
    return ece, 1-(torch.FloatTensor(mean_accuracy).mean())

def top_perf(net,multiheaded):
    net.eval()
    net.to(device) 
    net.half()
    with torch.no_grad():
        for data_test, target_test in dataset_loader_test:
            data_test, target_test = data_test.cuda(), target_test.cuda()
            target_test.half()
            output_test = net(data_test.half())
            break
        
        for data_b1, target_b1 in dataset_loader_b1:
            data_b1, target_b1 = data_b1.cuda(), target_b1.cuda()
            target_b1.half()
            output_b1 = net(data_b1.half())
            break
        
        for data_b3, target_b3 in dataset_loader_b3:
            data_b3, target_b3 = data_b3.cuda(), target_b3.cuda()
            target_b3.half()
            output_b3 = net(data_b3.half())
            break
        
        
        for data_b5, target_b5 in dataset_loader_b5:
            data_b5, target_b5 = data_b5.cuda(), target_b5.cuda()
            target_b5.half()
            output_b5 = net(data_b5.half())
            break
        
        for data_r, target_r in dataset_loader_r:
            data_r, target_r = data_r.cuda(), target_r.cuda()
            target_r.half()
            output_r = net(data_r.half())
            break
        
    if multiheaded:
        output_test=output_test[-1]
        output_b1=output_b1[-1]
        output_b3=output_b3[-1]
        output_b5=output_b5[-1]
        output_r=output_r[-1]
        
    pred_test = output_test.data.max(1, keepdim = True)
    prob_test = F.softmax(output_test, dim=1)
    top_p_test, top_class_test = prob_test.topk(100, dim = 1)
    top_class_test=top_class_test.cpu().numpy()
    pos = numpy.where(top_class_test==target_test.item())[1][0]
    target_p_test = top_p_test[0][pos]
    
    
    
    pred_b1 = output_b1.data.max(1, keepdim = True)
    prob_b1 = F.softmax(output_b1, dim=1)
    top_p_b1, top_class_b1 = prob_b1.topk(100, dim = 1)
    top_class_b1=top_class_b1.cpu().numpy()
    pos = numpy.where(top_class_b1==target_b1.item())[1][0]
    target_p_b1 = top_p_b1[0][pos]
    
    
    
    pred_b3 = output_b3.data.max(1, keepdim = True)
    prob_b3 = F.softmax(output_b3, dim=1)
    top_p_b3, top_class_b3 = prob_b3.topk(100, dim = 1)
    top_class_b3=top_class_b3.cpu().numpy()
    pos = numpy.where(top_class_b3==target_b3.item())[1][0]
    target_p_b3 = top_p_b3[0][pos]
    
    
    
    pred_b5 = output_b5.data.max(1, keepdim = True)
    prob_b5 = F.softmax(output_b5, dim=1)
    top_p_b5, top_class_b5 = prob_b5.topk(100, dim = 1)
    top_class_b5=top_class_b5.cpu().numpy()
    pos = numpy.where(top_class_b5==target_b5.item())[1][0]
    target_p_b5 = top_p_b5[0][pos]
    
    
    
    pred_r = output_r.data.max(1, keepdim = True)
    prob_r = F.softmax(output_r, dim=1)
    top_p_r, top_class_r = prob_r.topk(100, dim = 1)
    top_class_r=top_class_r.cpu().numpy()
    pos = numpy.where(top_class_r==target_r.item())[1][0]
    target_p_r = top_p_r[0][pos]
    
    
    
    return top_p_test[0][0:3], top_class_test[0][0:3], target_p_test.item(), target_test.item(), top_p_b1[0][0:3], top_class_b1[0][0:3], target_p_b1.item(), target_b1.item(), top_p_b3[0][0:3], top_class_b3[0][0:3], target_p_b3.item(), target_b3.item(), top_p_b5[0][0:3], top_class_b5[0][0:3], target_p_b5.item(), target_b5.item(), top_p_r[0][0:3], top_class_r[0][0:3], target_p_r.item(), target_r.item() 

        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
'''
>>> import os
>>> import pathlib
>>> import glob
>>>
>>> os.getcwd()
'/misc/lmbraid19/agnihotr/datasets/ImageNet-P'

>>> a = glob.glob(path_source+'/*/*')
>>> a[0]
'/misc/lmbraid19/agnihotr/datasets/ImageNet-P/snow/n02137549'
>>> pathlib.Path(a[0]).parent
PosixPath('/misc/lmbraid19/agnihotr/datasets/ImageNet-P/snow')
>>> pathlib.Path(a[0]).parent.name
'snow'
>>> pathlib.Path(a[0]).name



>>> for name in a:
...     for class_n in class_ids:
...             if class_n in name:
...                     path_copy=path_target+str(pathlib.Path(name).parent.name)
...                     command = 'cp -r ' + name + ' ' + path_copy + '/'
...                     if not os.path.isdir(path_copy):
...                             os.makedirs(path_copy)
...                     os.system(command)


'''
