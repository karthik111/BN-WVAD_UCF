import torch
#import torch_xla
#import torch_xla.core.xla_model as xm

#device = xm.xla_device()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(net, normal_loader, abnormal_loader, optimizer, criterion):
    net.train()
    net.flag = "Train"
    ninput, nlabel = next(normal_loader)        #                       (64, 200, 1024)
    ainput, alabel = next(abnormal_loader)      #                       (64, 200, 1024)
    _data = torch.cat((ninput, ainput), 0)      # cat normal & abnormal (128, 200, 1024)
    _label = torch.cat((nlabel, alabel), 0)
    #_data = _data.cuda()
    #_label = _label.cuda()
    _data = _data.to(device)
    _label = _label.to(device)

    res = net(_data)
    cost, loss = criterion(res)

    optimizer.zero_grad()

    cost.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
    optimizer.step()
    #xm.optimizer_step(optimizer)
    
    return loss