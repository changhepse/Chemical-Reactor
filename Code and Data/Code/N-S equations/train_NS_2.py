###########################
#强制连续性NS,1个网络
###########################
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.utils.data import TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
import pandas as pd
import geo3

from torch.utils.tensorboard import SummaryWriter
dtype = torch.float32
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_dtype(torch.float32)

#固定随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #torch.backends.cudnn.deterministic = True
setup_seed(2022)

###############
#定义神经网络
###############
class layer(nn.Module):
    def __init__(self, n_in, n_out, activation=None):
        super().__init__()
        self.layer = nn.Linear(n_in, n_out)
        self.activation = activation

    def forward(self, x):
        x = self.layer(x)
        if self.activation:
            x = self.activation(x)
        return x

#隐藏层节点数
h_n=64
#学习率
learning_rate = 0.001
#批次
batchsize = 512

class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            layer(2, h_n, nn.Tanh()),
            layer(h_n, h_n, nn.Tanh()),
            #layer(h_n, h_n, nn.Tanh()),
            layer(h_n, h_n, nn.Tanh()),
            layer(h_n, h_n, nn.Tanh()),
            layer(h_n, h_n, nn.Tanh()),
            layer(h_n, 2),
            )

    def forward(self, xy):
        return self.net(xy)

net = DNN().to(device)

#初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
        #nn.init.zeros_(m.bias)
net.apply(init_normal)

#优化器
#optimizer1 = optim.Adam(net.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
optimizer1 = optim.SGD(net.parameters(), lr=learning_rate)
#scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.9,patience=10,verbose=True)
#scheduler =CosineAnnealingLR(optimizer, 1000)
scheduler1 = ExponentialLR(optimizer1, gamma=0.99985)
optimizer2 = optim.LBFGS(
    net.parameters(),
    lr=1.0,
    max_iter=20000,
    max_eval=20000,
    history_size=50,
    tolerance_grad=1e-5,
    tolerance_change=1.0 * np.finfo(float).eps,
    line_search_fn="strong_wolfe")
#####################
#准备数据集
#######################
#物性
L=0.1#m
uin = 5e-4#m/s
rho=997#kg/m3
mu = 8.5e-4#Pa*s
Re = (L*uin*rho)/mu
#内部点
x_c = geo3.XY_c_sdf[:, 0:1]
y_c = geo3.XY_c_sdf[:, 1:2]
sdf = geo3.XY_c_sdf[:, 2:3]
x_c = torch.as_tensor(x_c, dtype=dtype).to(device)
y_c = torch.as_tensor(y_c, dtype=dtype).to(device)
sdf = torch.as_tensor(sdf, dtype=dtype).to(device)
#无滑移边界
x_wall = geo3.NoSWALL[:, 0:1]
y_wall = geo3.NoSWALL[:, 1:2]
x_wall = torch.as_tensor(x_wall, dtype=dtype).to(device)
y_wall = torch.as_tensor(y_wall, dtype=dtype).to(device)
#入口
x_inlet = geo3.INLET_uv[:, 0:1]
y_inlet = geo3.INLET_uv[:, 1:2]
u_inlet_bc = geo3.INLET_uv[:, 2:3]
x_inlet = torch.as_tensor(x_inlet, dtype=dtype).to(device)
y_inlet = torch.as_tensor(y_inlet, dtype=dtype).to(device)
u_inlet_bc = torch.as_tensor(u_inlet_bc, dtype=dtype).to(device)
#出口
x_outlet = geo3.OUTLET[:, 0:1]
y_outlet = geo3.OUTLET[:, 1:2]
x_outlet = torch.as_tensor(x_outlet, dtype=dtype).to(device)
y_outlet = torch.as_tensor(y_outlet, dtype=dtype).to(device)

#整合数据集
#dataset = TensorDataset(X,Y,sdf)
#dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True,drop_last=True)

##############
#定义损失函数
##############
def criterion(x,y,sdf):

    w = sdf

    x = x.requires_grad_()
    y = y.requires_grad_()

    net_in = torch.cat((x,y),1)
    uv = net(net_in)[:,0:1]
    u = grad(uv.sum(), y, create_graph=True)[0]
    v = - grad(uv.sum(), x, create_graph=True)[0]
    p = net(net_in)[:,1:2]

    u_x = grad(u.sum(), x, create_graph=True)[0]
    u_y = grad(u.sum(), y, create_graph=True)[0]
    v_x = grad(v.sum(), x, create_graph=True)[0]
    v_y = grad(v.sum(), y, create_graph=True)[0]
    p_x = grad(p.sum(), x, create_graph=True)[0]
    p_y = grad(p.sum(), y, create_graph=True)[0]
    u_xx = grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = grad(u_y.sum(), y, create_graph=True)[0]
    v_xx = grad(v_x.sum(), x, create_graph=True)[0]
    v_yy = grad(v_y.sum(), y, create_graph=True)[0]

    #NS方程
    fu = w*(u*u_x + v*u_y + p_x - (1/Re)*(u_xx + u_yy))
    fv = w*(u*v_x + v*v_y + p_y - (1/Re)*(v_xx + v_yy))

    loss_fu = torch.mean(torch.square(fu))
    loss_fv = torch.mean(torch.square(fv))

    #入口边界条件：u=抛物线,v=0
    global x_inlet
    global y_inlet
    x_inlet = x_inlet.requires_grad_()
    y_inlet = y_inlet.requires_grad_()

    net_inlet = torch.cat((x_inlet,y_inlet),1)
    uv_inlet_pred = net(net_inlet)[:,0:1]
    u_inlet_pred = grad(uv_inlet_pred.sum(), y_inlet, create_graph=True)[0]
    v_inlet_pred = - grad(uv_inlet_pred.sum(), x_inlet, create_graph=True)[0]

    loss_inlet = torch.mean(torch.square(u_inlet_pred - u_inlet_bc)) + torch.mean(torch.square(v_inlet_pred))

    #壁面，加热筒边界条件：u=0,v=0
    global x_wall
    global y_wall
    x_wall = x_wall.requires_grad_()
    y_wall = y_wall.requires_grad_()

    net_wall = torch.cat((x_wall,y_wall),1)
    uv_wall_pred = net(net_wall)[:,0:1]
    u_wall_pred = grad(uv_wall_pred.sum(), y_wall, create_graph=True)[0]
    v_wall_pred = - grad(uv_wall_pred.sum(), x_wall, create_graph=True)[0]

    loss_wall = torch.mean(torch.square(u_wall_pred)) + torch.mean(torch.square(v_wall_pred))

    #出口边界条件：p=0
    global x_outlet
    global y_outlet
    x_outlet = x_outlet.requires_grad_()
    y_outlet = y_outlet.requires_grad_()

    net_outlet = torch.cat((x_outlet,y_outlet),1)
    p_outlet_pred = net(net_outlet)[:,1:2]

    loss_outlet = torch.mean(torch.square(p_outlet_pred))


    #总损失函数
    loss =  loss_fu + loss_fv + 30*loss_wall + loss_inlet + loss_outlet

    return loss,loss_fu,loss_fv,loss_wall,loss_inlet,loss_outlet


##############
#训练
##############
#CFD验证集
Val = pd.read_csv('f_data_0.5.csv')

x_val = Val['x']
x_val = x_val.tolist()
x_val = np.array(x_val)
x_val = x_val.flatten()[:, None]

y_val = Val['y']
y_val = y_val.tolist()
y_val = np.array(y_val)
y_val = y_val.flatten()[:, None]

u_val = Val['u']
u_val = u_val.tolist()
u_val = np.array(u_val)
u_val = u_val.flatten()[:, None]

v_val = Val['v']
v_val = v_val.tolist()
v_val = np.array(v_val)
v_val = v_val.flatten()[:, None]

p_val = Val['p']
p_val = p_val.tolist()
p_val = np.array(p_val)
p_val = p_val.flatten()[:, None]

x_val = torch.as_tensor(x_val, dtype=dtype).to(device)#验证集的x坐标
y_val = torch.as_tensor(y_val, dtype=dtype).to(device)#验证集的y坐标
u_val = torch.as_tensor(u_val, dtype=dtype).to(device)#验证集的u
v_val = torch.as_tensor(v_val, dtype=dtype).to(device)#验证集的v
p_val = torch.as_tensor(p_val, dtype=dtype).to(device)#验证集的p

print("--------------------开始训练(๑•̀ㅂ•́)و✧-----------------------")

iter = 0

#计算相对误差
def val(x_val,y_val):

    x_val = x_val.requires_grad_()
    y_val = y_val.requires_grad_()
    net_val_in = torch.cat((x_val,y_val),1)
    uv_pred = net(net_val_in)[:,0:1]
    u_pred = grad(uv_pred.sum(),y_val, create_graph=True)[0]
    v_pred = - grad(uv_pred.sum(),x_val, create_graph=True)[0]
    p_pred = net(net_val_in)[:,1:2]

    u_diff = (u_pred - u_val)/u_val
    mask = torch.isfinite(u_diff)
    u_diff = u_diff[mask]
    error_u = torch.mean(torch.abs(u_diff))

    v_diff = (v_pred - v_val)/v_val
    mask = torch.isfinite(v_diff)
    v_diff = v_diff[mask]
    error_v = torch.mean(torch.abs(v_diff))

    error_p = torch.mean(torch.abs((p_pred - p_val)/p_val))

    return error_u, error_v,error_p


def closure():

    global iter
    net.zero_grad()
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    loss = criterion(x_c,y_c,sdf)[0]
    loss_fu = criterion(x_c,y_c,sdf)[1]
    loss_fv = criterion(x_c,y_c,sdf)[2]
    loss_wall = criterion(x_c,y_c,sdf)[3]
    loss_inlet = criterion(x_c,y_c,sdf)[4]
    loss_outlet = criterion(x_c,y_c,sdf)[5]
    loss.backward()

    error_u, error_v,error_p = val(x_val,y_val)


    if iter % 100 == 0:

        writer = SummaryWriter('log_SGD/loss',flush_secs=60)
        writer.add_scalar('总损失/Loss',loss,iter)
        writer.add_scalar('函数损失/Loss_fu',loss_fu,iter)
        writer.add_scalar('函数损失/Loss_fv',loss_fv,iter)
        writer.add_scalar('边界损失/Loss_wall',loss_wall,iter)
        writer.add_scalar('边界损失/Loss_inlet',loss_inlet,iter)
        writer.add_scalar('边界损失/Loss_outlet',loss_outlet,iter)

        writer = SummaryWriter('log_SGD/error',flush_secs=60)
        writer.add_scalar('相对误差/error_u',error_u,iter)
        writer.add_scalar('相对误差/error_v',error_v,iter)
        writer.add_scalar('相对误差/error_p',error_p,iter)


    msg = f"\r{iter}, " + \
          f"loss : {loss.item():.3e}, " + \
          f"error_u : {error_u.item():.2%}, "+ \
          f"error_v : {error_v.item():.2%}, "+ \
          f"error_p : {error_p.item():.2%} "

    print(msg, end = "")

    if iter % 1000 == 0:
        print("")

    iter += 1

    return loss

for epoch in range(40000):
    closure()
    optimizer1.step()
    scheduler1.step()
#optimizer2.step(closure)


#保存网络参数
torch.save(net.state_dict(), "./pkl/NS_wb_SGD.pkl")

#可视化
#tensorboard --logdir=D:\Project\code\2022\20220301\2\2.2.1基础框架\log_Adam

