import torch
import numpy as np
import torch.nn as nn
from torch.autograd import grad
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from torch.optim.lr_scheduler import ExponentialLR,CosineAnnealingLR
import geo3
from torch.utils.tensorboard import SummaryWriter
dtype = torch.float32
torch.set_default_dtype(torch.float32)

#设定随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #random.seed(seed)
    #torch.backends.cudnn.deterministic = True
# 设置随机数种子
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
            layer(h_n, h_n, nn.Tanh()),
            #layer(h_n, h_n, nn.Tanh()),
            #layer(h_n, h_n, nn.Tanh()),
            layer(h_n, h_n, nn.Tanh()),
            layer(h_n, h_n, nn.Tanh()),
            layer(h_n, 1),
        )

    def forward(self, xy):
        return self.net(xy)

device = torch.device("cuda")
net = DNN().to(device)

#初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)

net.apply(init_normal)

#优化器
optimizer1 = optim.Adam(net.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
#scheduler =CosineAnnealingLR(optimizer1, 2000)
scheduler = ExponentialLR(optimizer1, gamma=0.99975)
#optimizer2 = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)

optimizer2 = optim.LBFGS(
    net.parameters(),
    lr=1.0,
    max_iter=20000,
    max_eval=20000,
    history_size=50,
    tolerance_grad=1e-10,
    tolerance_change=1.0 * np.finfo(float).eps,
    line_search_fn="strong_wolfe")

#####################
#数据集
#######################
#物性
rho=997#kg/m3
L=0.1#m
uin = 5e-4#m/s
#内部点
x_c = geo3.XY_c_sdf[:, 0:1]
y_c = geo3.XY_c_sdf[:, 1:2]
sdf = geo3.XY_c_sdf[:, 2:3]
x_c = torch.as_tensor(x_c, dtype=dtype).to(device)
y_c = torch.as_tensor(y_c, dtype=dtype).to(device)
sdf = torch.as_tensor(sdf, dtype=dtype).to(device)
#上下边界
x_wall = geo3.WALL[:, 0:1]
y_wall = geo3.WALL[:, 1:2]
x_wall = torch.as_tensor(x_wall, dtype=dtype).to(device)
y_wall = torch.as_tensor(y_wall, dtype=dtype).to(device)
#入口边界
x_inlet = geo3.INLET_uv[:, 0:1]
y_inlet = geo3.INLET_uv[:, 1:2]
x_inlet = torch.as_tensor(x_inlet, dtype=dtype).to(device)
y_inlet = torch.as_tensor(y_inlet, dtype=dtype).to(device)
#出口边界
x_outlet = geo3.OUTLET[:, 0:1]
y_outlet = geo3.OUTLET[:, 1:2]
x_outlet = torch.as_tensor(x_outlet, dtype=dtype).to(device)
y_outlet = torch.as_tensor(y_outlet, dtype=dtype).to(device)
#圆环边界
x_cyl = geo3.CYLD[:, 0:1]
y_cyl = geo3.CYLD[:, 1:2]
x_cyl = torch.as_tensor(x_cyl, dtype=dtype).to(device)
y_cyl = torch.as_tensor(y_cyl, dtype=dtype).to(device)

#整合数据集
#dataset = TensorDataset(X,Y,sdf)
#dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True,drop_last = True)

##############
#定义损失函数
##############

#读取NS方程数据
h_ns=64

class DNN_NS(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            layer(2, h_ns, nn.Tanh()),
            #layer(h_ns, h_ns, nn.Tanh()),
            layer(h_ns, h_ns, nn.Tanh()),
            layer(h_ns, h_ns, nn.Tanh()),
            layer(h_ns, h_ns, nn.Tanh()),
            layer(h_ns, h_ns, nn.Tanh()),
            layer(h_ns, 2),
        )

    def forward(self, xy):
        return self.net(xy)

net_ns = DNN_NS().to(device)
net_ns.load_state_dict(torch.load('./pkl/NS_wb_0.5.pkl'))



#损失函数
def criterion(x,y,sdf):

    w = 1

    x = x.requires_grad_()
    y = y.requires_grad_()

    net_in = torch.cat((x,y),1)
    T = net(net_in)

    T_x = grad(T.sum(), x, create_graph=True)[0]
    T_y = grad(T.sum(), y, create_graph=True)[0]
    T_xx = grad(T_x.sum(), x, create_graph=True)[0]
    T_yy = grad(T_y.sum(), y, create_graph=True)[0]

    #导入速度场
    uv = net_ns(net_in)[:,0:1]
    u = grad(uv.sum(),y, create_graph=True)[0]
    v = - grad(uv.sum(),x, create_graph=True)[0]
    with torch.no_grad():
        u = u
        v = v


    #温度相关物理数
    Tn = T*300
    k = -0.869083936 + 0.00894880345*Tn - 1.58366345E-5*Tn**2 + 7.97543259E-9*Tn**3 #W/(m*k)
    cp = 12010.1471 - 80.4072879*Tn + 0.309866854*Tn**2 - 5.38186884E-4*Tn**3 + 3.62536437E-7*Tn**4 #J/(kg*k)
    alpha = k/(rho*cp)
    Pe = L*uin/alpha

    #传热方程
    heat_f = w*(u*T_x + v*T_y - (1/Pe)*(T_xx + T_yy))

    loss_heat_f = torch.mean(torch.square(heat_f))

    #边界条件
    #加热筒：T= 1.083
    net_cyl = torch.cat((x_cyl,y_cyl),1)
    T_cyc_pred = net(net_cyl)

    loss_heat_cyc = torch.mean(torch.square(T_cyc_pred - 1.083))

    #入口：T=1
    net_inlet = torch.cat((x_inlet,y_inlet),1)
    T_inlet_pred = net(net_inlet)

    loss_heat_inlet = torch.mean(torch.square(T_inlet_pred - 1))

    #壁面：dT/dy=0
    global y_wall
    y_wall = y_wall.requires_grad_()

    net_wall = torch.cat((x_wall,y_wall),1)
    T_wall_pred = net(net_wall)

    T_wall_pred_y = grad(T_wall_pred.sum(), y_wall, create_graph = True)[0]

    loss_heat_wall = torch.mean(torch.square(T_wall_pred_y))

    #出口：dT/dx=0
    global x_outlet
    x_outlet = x_outlet.requires_grad_()

    net_outlet = torch.cat((x_outlet,y_outlet),1)
    T_outlet_pred = net(net_outlet)

    T_outlet_pred_x = grad(T_outlet_pred.sum(), x_outlet, create_graph = True)[0]

    loss_heat_outlet = torch.mean(torch.square(T_outlet_pred_x))


    #传热损失函数
    loss_heat = loss_heat_f + loss_heat_cyc + 5*loss_heat_inlet + loss_heat_wall + loss_heat_outlet



    return loss_heat, loss_heat_f,loss_heat_cyc,loss_heat_inlet,loss_heat_wall ,loss_heat_outlet


##############
#训练
##############

#CFD验证集
Val = pd.read_csv('h_r_data.csv')

x_val = Val['x']
x_val = x_val.tolist()
x_val = np.array(x_val)
x_val = x_val.flatten()[:, None]

y_val = Val['y']
y_val = y_val.tolist()
y_val = np.array(y_val)
y_val = y_val.flatten()[:, None]

T_val = Val['T']
T_val = T_val.tolist()
T_val = np.array(T_val)
T_val = T_val.flatten()[:, None]

x_val = torch.as_tensor(x_val, dtype=dtype).to(device)#验证集的x坐标
y_val = torch.as_tensor(y_val, dtype=dtype).to(device)#验证集的y坐标
T_val = torch.as_tensor(T_val, dtype=dtype).to(device)#验证集的T

###############
#训练
###############
print("----------开始训练(๑•̀ㅂ•́)و✧------------")

iter = 0

#计算相对误差
def val(x_val,y_val):

    net_val_in = torch.cat((x_val,y_val),1)
    T_pred = net(net_val_in)

    error_heat = torch.mean(torch.abs((T_pred - T_val)/T_val))

    return error_heat


def closure():

    global iter
    net.zero_grad()
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    loss_heat = criterion(x_c,y_c,sdf)[0]
    loss_heat_f = criterion(x_c,y_c,sdf)[1]
    loss_heat_cyc = criterion(x_c,y_c,sdf)[2]
    loss_heat_inlet = criterion(x_c,y_c,sdf)[3]
    loss_heat_wall = criterion(x_c,y_c,sdf)[4]
    loss_heat_outlet = criterion(x_c,y_c,sdf)[5]
    loss_heat.backward()

    error_heat = val(x_val,y_val)


    if iter % 100 == 0:

        writer = SummaryWriter('log_h2/loss', flush_secs=60)
        writer.add_scalar('总损失/Loss',loss_heat,iter)
        writer.add_scalar('函数损失/Loss_heat_f',loss_heat_f,iter)
        writer.add_scalar('边界损失/Loss_heat_cyc',loss_heat_cyc,iter)
        writer.add_scalar('边界损失/Loss_heat_inlet',loss_heat_inlet,iter)
        writer.add_scalar('边界损失/Loss_heat_wall',loss_heat_wall,iter)
        writer.add_scalar('边界损失/Loss_heat_outlet',loss_heat_outlet,iter)

        writer = SummaryWriter('log_h2/error', flush_secs=60)
        writer.add_scalar('相对误差/error_heat',error_heat,iter)


    msg = f"\r{iter}, " + \
          f"loss : {loss_heat.item():.3e}, " + \
          f"error_heat : {error_heat.item():.3%}"
    print(msg, end = "")

    if iter % 1000 == 0:
        print("")

    iter += 1

    return loss_heat

for epoch in range(5000):
    closure()
    optimizer1.step()
    scheduler.step()
optimizer2.step(closure)

##############
#保存网络
##############
torch.save(net.state_dict(), "./pkl2/h_wb2.pkl")

#可视化
#tensorboard --logdir=D:\Project\code\2022\20220301\3\H_R\log_h2