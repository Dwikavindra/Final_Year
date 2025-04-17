import argparse
import os, sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
import pickle
from data_load import mnist, svhn, usps,election
from PIL import ImageOps
import torch.multiprocessing as mp
import csv
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
import cv2
from PIL import Image,ImageOps

def write_row_to_csv(file_path, columns, values):
    assert len(columns) == len(values), "Columns and values must be the same length."
    folder = os.path.dirname(file_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    df = pd.DataFrame([dict(zip(columns, values))])
    df.to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))

mp.set_start_method('spawn', force=True)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def image_processing_no_effect(image):
    """Processes the image to match LeNetBase input: grayscale, invert, resize, normalize."""
    


    # Apply transformations
    transform = transforms.Compose([
        # transforms.Resize((28, 28)),  # Resize to 28x28 for LeNetBase
        transforms.ToTensor(),        # Convert to tensor (C, H, W)
        transforms.Normalize((0.5,), (0.5,)),  # Normalize after ToTensor()
    ])
    
    return transform(image)  # Apply transformations


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def digit_load(args): 
    train_bs = args.batch_size
    if args.dset == 's2m':
        train_source = svhn.SVHN('./data/svhn/', split='train', download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))
        test_source = svhn.SVHN('./data/svhn/', split='test', download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))  
        train_target = mnist.MNIST_idx('./data/mnist/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))      
        test_target = mnist.MNIST('./data/mnist/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.Lambda(lambda x: x.convert("RGB")),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]))
    elif args.dset == 'u2m':
        train_source = usps.USPS('./data/usps/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(28, padding=4),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
        test_source = usps.USPS('./data/usps/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(28, padding=4),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))    
        train_target = mnist.MNIST_idx('./data/mnist/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))    
        test_target = mnist.MNIST('./data/mnist/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
    elif args.dset == 'm2u':
        train_source = mnist.MNIST('./data/mnist/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
        test_source = mnist.MNIST('./data/mnist/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))

        train_target = usps.USPS_idx('./data/usps/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
        test_target = usps.USPS('./data/usps/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
    elif args.dset == 'mnistelection':
        train_source = mnist.MNIST('./data/mnist/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
        test_source = mnist.MNIST('./data/mnist/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))

        train_target = election.ElectionDatasetSirekapMethod('../data/batch_inference.csv', train=True,dset_size=args.dset_size,
                transform=None)
        test_target = election.ElectionDatasetSirekapMethod('../data/batch_inference.csv', train=False,dset_size=args.dset_size, 
                transform=None) 

    dset_loaders = {}
    dset_loaders["source_tr"] = DataLoader(train_source, batch_size=train_bs, shuffle=True, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["source_te"] = DataLoader(test_source, batch_size=train_bs*2, shuffle=True, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["target"] = DataLoader(train_target, batch_size=train_bs, shuffle=True, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["target_te"] = DataLoader(train_target, batch_size=train_bs, shuffle=False, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["test"] = DataLoader(test_target, batch_size=train_bs*2, shuffle=False, 
        num_workers=args.worker, drop_last=False)
    return dset_loaders

def cal_acc(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0].to(device)  # Move to MPS or CPU
            labels = data[1].to(device)

            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float().cpu()), 0)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size(0))
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().item()
    return accuracy * 100, mean_ent

def train_source(args):
    dset_loaders = digit_load(args)

    if args.dset in ['u2m', 'm2u','mnistelection']:
        netF = network.LeNetBase().to(device)
    elif args.dset == 's2m':
        netF = network.DTNBase().to(device)

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).to(device)
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).to(device)

    param_group = []
    learning_rate = args.lr
    for model in [netF, netB, netC]:
        for k, v in model.named_parameters():
            param_group += [{'params': v, 'lr': learning_rate}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 10
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()

    while iter_num < max_iter:
        print(f'Train Source Iteration: {iter_num} out of {max_iter}')
        try:
            inputs_source, labels_source = next(iter_source)
        except:
            iter_source = iter(dset_loaders["source_tr"])
            inputs_source, labels_source = next(iter_source)

        if inputs_source.size(0) == 1:
            continue

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_source, labels_source = inputs_source.to(device), labels_source.to(device)
        outputs_source = netC(netB(netF(inputs_source)))
        classifier_loss = loss.CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(outputs_source, labels_source)

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            acc_s_tr, _ = cal_acc(dset_loaders['source_tr'], netF, netB, netC)
            acc_s_te, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC)
            log_str = f'Task: {args.dset}, Iter:{iter_num}/{max_iter}; Accuracy = {acc_s_tr:.2f}%/ {acc_s_te:.2f}%' #not reallly accuracy but a comparison of accuracy from source data for training
        # and source testing , so calcualte the model against the training sample and test sample of the original mnist, ok now see the ratio between them if its low then the model accuracy on source testing increases
        # and if its low the training dataset is higher ,, ideally a model should be able to identify its training dataset better so this ratio would tell us as we apply domain shift
        # how much does the model change its features so it can accomodate new entry. Higher is better because we would like to work for training data and the newly inputted data
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

            if acc_s_te >= acc_init:
                acc_init = acc_s_te
                best_netF = copy.deepcopy(netF.state_dict())
                best_netB = copy.deepcopy(netB.state_dict())
                best_netC = copy.deepcopy(netC.state_dict())

            netF.train()
            netB.train()
            netC.train()

    torch.save(best_netF, osp.join(args.output_dir, "source_F.pt"))
    torch.save(best_netB, osp.join(args.output_dir, "source_B.pt"))
    torch.save(best_netC, osp.join(args.output_dir, "source_C.pt"))

    return netF, netB, netC
def test_target(args):
    dset_loaders = digit_load(args)
    ## set base network
    if args.dset == 'u2m':
        netF = network.LeNetBase().to(device)
    elif args.dset == 'm2u':
        netF = network.LeNetBase().to(device)  
    elif args.dset == 's2m':
        netF = network.DTNBase().to(device)
    elif args.dset == 'mnistelection':
        netF = network.LeNetBase().to(device) 

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).to(device)
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).to(device)

    args.modelpath = args.output_dir + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath, map_location=device))
    args.modelpath = args.output_dir + '/source_B.pt'   
    netB.load_state_dict(torch.load(args.modelpath, map_location=device))
    args.modelpath = args.output_dir + '/source_C.pt'   
    netC.load_state_dict(torch.load(args.modelpath, map_location=device))
    netF.eval()
    netB.eval()
    netC.eval()

    acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC)
    log_str = 'Task: {}, Accuracy = {:.2f}%'.format(args.dset, acc)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def saveResultTestAccuracy(accuracy,f1,precision,recall,iteration, dset_size,iter_num,max_iter,fileName, filePath):

    
    file_path = os.path.join(filePath, f"{fileName}.csv")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    is_file_exist = os.path.exists(file_path)


    with open(file_path, mode="a", newline="") as f: 
        writer = csv.writer(f)
        if not is_file_exist:
            writer.writerow(["iteration", "accuracy", "f1","precision","recall","iter_num","max_iter","dset_size"])
        writer.writerow([iteration, accuracy,f1,precision,recall,iter_num,max_iter,dset_size])
    print(f"Result saved: Iteration {iteration},Accuracy {accuracy}, Dset_size: {dset_size}")


def cal_acc_key_metrics(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0].to(device)
            labels = data[1].to(device)

            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.cpu()), 0)

    _, predict = torch.max(all_output, 1)

    accuracy = (torch.sum(predict == all_label).item() / all_label.size(0)) * 100

    y_true = all_label.numpy()
    y_pred = predict.numpy()

    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    class_accuracies = {}
    for label in set(y_true):
        class_accuracies[label] = accuracy_score(y_true[y_true == label], y_pred[y_true == label])

    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().item()

    return accuracy, mean_ent, precision, recall, f1, class_accuracies

def train_target(args):
    dset_loaders = digit_load(args)
    
    
    ## Set base network
    if args.dset == 'u2m':
        netF = network.LeNetBase().to(device)
    elif args.dset == 'm2u':
        netF = network.LeNetBase().to(device)  
    elif args.dset == 's2m':
        netF = network.DTNBase().to(device)
    elif args.dset =='mnistelection':
        netF= network.LeNetBase().to(device)

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).to(device)
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).to(device)

    args.modelpath = args.output_dir + '/source_F.pt'   
    netF.load_state_dict(torch.load(args.modelpath, map_location=device))
    args.modelpath = args.output_dir + '/source_B.pt'   
    netB.load_state_dict(torch.load(args.modelpath, map_location=device))
    args.modelpath = args.output_dir + '/source_C.pt'    
    netC.load_state_dict(torch.load(args.modelpath, map_location=device))
    netC.eval()
    
    for k, v in netC.named_parameters():
        v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    for k, v in netB.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = len(dset_loaders["target"])
    interval_iter_tracker=1
    iter_num = 0

    while iter_num < max_iter:
        print(f'Train Target Function Iteration: {iter_num} out of {max_iter}')
        optimizer.zero_grad()
        try:
            inputs_test, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = next(iter_test)

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            mem_label = obtain_label(dset_loaders['target_te'], netF, netB, netC, interval_iter_tracker,args)
            interval_iter_tracker+=1
            mem_label = torch.from_numpy(mem_label).to(device)
            netF.train()
            netB.train()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        inputs_test = inputs_test.to(device)
        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)

        if args.cls_par > 0:
            pred = mem_label[tar_idx]
            classifier_loss = args.cls_par * nn.CrossEntropyLoss()(outputs_test, pred)
        else:
            classifier_loss = torch.tensor(0.0).to(device)

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            acc, _ ,precision, recall, f1, class_accuracies= cal_acc_key_metrics(dset_loaders['test'], netF, netB, netC) # Real accuracy after running n times
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.dset, iter_num, max_iter, acc)
            saveResultTestAccuracy(acc, f1,precision,recall,args.iteration,args.dset_size,iter_num,max_iter,"SHOT_evaluation_metrics","../saved_result/adaptation")    
            columns = ['Class', 'Accuracy',"iter_num","max_iter","dset_size","iteration"]
    
            for class_label, accuracy in class_accuracies.items():
                write_row_to_csv(f".../saved_result/adaptation/per_class_acc/SHOT_per_class_accuracy.csv", columns, [class_label, accuracy,iter_num,max_iter,args.dset_size,args.iteration])
        
            print(log_str+'\n')
            netF.train()
            netB.train()

    if args.issave:
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))

    return netF, netB, netC
def saveResultObtainLabel(accuracy_before,accuracy_after, iteration, dset_size,interval_iter_tracker,fileName, filePath):

    
    file_path = os.path.join(filePath, f"{fileName}.csv")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    
    file_exists = os.path.exists(file_path)
    
    with open(file_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["iteration", "accuracy_before", "accuracy_after", "dset_size","interval_iter_tracker"])

        # Write the data row
        writer.writerow([iteration, accuracy_before, accuracy_after,dset_size,interval_iter_tracker])

    print(f"✅ Result saved: Iteration {iteration},Accuracy_before {accuracy_before}, Accuracy_after {accuracy_after}, Dset_size: {dset_size}")

def obtain_label(loader, netF, netB, netC,interval_iter_tracker, args, c=None):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0].to(device)
            labels = data[1].to(device)
            feas = netB(netF(inputs))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu() #this should be the matrix of the last layer of net b 
                all_output = outputs.float().cpu()
                all_label = labels.float().cpu()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float().cpu()), 0)
    
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size(0))
    
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t() # say batch size is 11
    all_fea = all_fea.float().cpu().numpy() # (11,257) added 

    K = all_output.size(1)#11,10 batch size 11 10 rows 
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)#10x11 * 11*257 = 10*257
    #p values 
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea) # len fea is 11, so it goes to 11 samples and see which one is thesame 

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    # saveResultObtainLabel(accuracy,acc,args.iteration,args.dset_size, interval_iter_tracker,"obtain_label_accuracy_with_interval_iter_tracker/key_metrics","../saved_result/adaptation/")
    # args.out_file.write(log_str + '\n')
    # args.out_file.flush()
    print(log_str + '\n')
    return pred_label.astype('int')
def build_model(dset_size):
    netF = network.LeNetBase().to(device)
    netB = network.feat_bottleneck(type="bn", feature_dim=netF.in_features, bottleneck_dim=256).to(device)
    netC = network.feat_classifier(type="wn", class_num=10, bottleneck_dim=256).to(device)
    modelpath = "../digit/ckps_digits/seed2020/mnistelection" + f'/target_F_par_0.1dset_size{dset_size}randomized.pt'   
    netF.load_state_dict(torch.load(modelpath, map_location=device))
    modelpath = "../digit/ckps_digits/seed2020/mnistelection" + f'/target_B_par_0.1dset_size{dset_size}randomized.pt'   
    netB.load_state_dict(torch.load(modelpath, map_location=device))
    modelpath = "../digit/ckps_digits/seed2020/mnistelection" + f'/target_C_par_0.1dset_size{dset_size}randomized.pt'   
    netC.load_state_dict(torch.load(modelpath, map_location=device))
    netF.eval()
    netB.eval()
    netC.eval()
    #   outputs = netC(netB(netF(inputs)))
    return netC,netB,netF
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--max_epoch', type=int, default=30, help="maximum epoch")
    parser.add_argument('--iteration', type=float, default=1.0, help="for scripting iteration")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset_size', type=float, default=None, help="percentage of dataset size ")
    parser.add_argument('--dset', type=str, default='s2m', choices=['u2m', 'm2u', 's2m','mnistelection'])
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)   
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--issave', type=bool, default=True)
    args = parser.parse_args()
    args.class_num = 10

    # Set device for computation
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # Set seed for reproducibility
    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    args.output_dir = osp.join(args.output, 'seed' + str(args.seed), args.dset)
    os.makedirs(args.output_dir, exist_ok=True)

    if not osp.exists(osp.join(args.output_dir, 'source_F.pt')):
        args.out_file = open(osp.join(args.output_dir, 'log_src.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_source(args)
        test_target(args)
    
    args.savename = 'par_' + str(args.cls_par)
    if args.dset_size != None:
        args.savename+= 'dset_size'+ str(args.dset_size)+'randomized' #randomzied train data 
    args.out_file = open(osp.join(args.output_dir, f'log_tar_{args.savename}.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()

    train_target(args)
    target_mnist = mnist.MNIST('./data/mnist/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ]))
    dataloader= DataLoader(target_mnist, batch_size=64*2, shuffle=False, 
        num_workers=4, drop_last=False)
    netC,netB,netF=build_model(args.dset_size)
    acc, _ ,precision, recall, f1, class_accuracies=cal_acc_key_metrics(dataloader,netF,netB,netC)
    columns = ['Accuracy', 'precision',"recall","f1","iteration","dset_size"]
    write_row_to_csv(f"../saved_result/inference_key_metrics/inference_mnist_after_performance.csv", columns, [acc,precision,recall,f1,args.iteration,args.dset_size])
    columns = ['Class', 'Accuracy',"iteration","dset_size"]
  
