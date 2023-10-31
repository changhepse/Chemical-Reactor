import torch
import numpy as np
import torch.nn as nn
from torch.autograd import grad
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from torch.optim.lr_scheduler import ExponentialLR,ReduceLROnPlateau
import geo3
from torch.utils.tensorboard import SummaryWriter
dtype = torch.float64
torch.set_default_dtype(torch.float64)

#设定随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #random.seed(seed)
    #torch.backends.cudnn.deterministic = True
# 设置随机数种子
#setup_seed(2022)


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
h_n1=64#传热
h_n2=64#反应

#学习率
learning_rate = 0.001
#批次
batchsize = 512

class DNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            layer(2, h_n1, nn.ReLU()),
            layer(h_n1, h_n1, nn.ReLU()),
            layer(h_n1, h_n1, nn.ReLU()),
            layer(h_n1, h_n1, nn.ReLU()),
            layer(h_n1, h_n1, nn.ReLU()),
            layer(h_n1, 1),
        )

    def forward(self, xy):
        return self.net(xy)

class DNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            layer(2, h_n2, nn.ReLU()),
            layer(h_n2, h_n2, nn.ReLU()),
            layer(h_n2, h_n2, nn.ReLU()),
            layer(h_n2, h_n2, nn.ReLU()),
            layer(h_n2, h_n2, nn.ReLU()),
            layer(h_n2, 1),
        )

    def forward(self, xy):
        return self.net(xy)

device = torch.device("cuda")
net1 = DNN1().to(device)
net2 = DNN2().to(device)

#初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
        #nn.init.zeros_(m.bias)
net1.apply(init_normal)
net2.apply(init_normal)
#net1.load_state_dict(torch.load('./pkl/h_wbt.pkl'))
#net2.load_state_dict(torch.load('./pkl/r_wbt.pkl'))

#优化器
optimizer1 = optim.Adam(net1.parameters(), lr=0.001, betas = (0.9,0.99),eps = 10**-15)
optimizer2	= optim.Adam(net2.parameters(), lr=0.001, betas = (0.9,0.99),eps = 10**-15)
#scheduler1 =CosineAnnealingLR(optimizer1, 1000)
#scheduler2 =CosineAnnealingLR(optimizer2, 1000)
scheduler1 = ExponentialLR(optimizer1, gamma=0.99975)
#scheduler2 = ExponentialLR(optimizer2, gamma=0.9995)
scheduler2 = ReduceLROnPlateau(optimizer2, 'min',factor=0.99,patience=50,verbose=False)
optimizer1_2 = optim.LBFGS(
    net1.parameters(),
    lr=1.0,
    max_iter=100000,
    max_eval=100000,
    history_size=50,
    tolerance_grad=1e-10,
    tolerance_change=1.0 * np.finfo(float).eps,
    line_search_fn="strong_wolfe")
optimizer2_2 = optim.LBFGS(
    net2.parameters(),
    lr=1.0,
    max_iter=100000,
    max_eval=100000,
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
#上下壁面
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
cos_alpha = geo3.CYLD_cos[:, 2:3]
cos_beta = geo3.CYLD_cos[:, 3:4]
x_cyl = torch.as_tensor(x_cyl, dtype=dtype).to(device)
y_cyl = torch.as_tensor(y_cyl, dtype=dtype).to(device)
cos_alpha = torch.as_tensor(cos_alpha, dtype=dtype).to(device)
cos_beta = torch.as_tensor(cos_beta, dtype=dtype).to(device)

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

    w1 = 1
    w2 = 1

    x = x.requires_grad_()
    y = y.requires_grad_()

    net_in = torch.cat((x,y),1)
    T = net1(net_in)
    c = net2(net_in)

    T_x = grad(T.sum(), x, create_graph=True)[0]
    T_y = grad(T.sum(), y, create_graph=True)[0]
    c_x = grad(c.sum(), x, create_graph=True)[0]
    c_y = grad(c.sum(), y, create_graph=True)[0]
    T_xx = grad(T_x.sum(), x, create_graph=True)[0]
    T_yy = grad(T_y.sum(), y, create_graph=True)[0]
    c_xx = grad(c_x.sum(), x, create_graph=True)[0]
    c_yy = grad(c_y.sum(), y, create_graph=True)[0]

    #导入速度场
    uv = net_ns(net_in)[:,0:1]
    u = grad(uv.sum(),y, create_graph=True)[0]
    v = - grad(uv.sum(),x, create_graph=True)[0]
    with torch.no_grad():
        u = u
        v = v

    # 反应速率R
    A = 1e10#1/s
    A = torch.as_tensor(A)
    E = 72e3#j/mol
    E = torch.as_tensor(E)
    Rg = 8.314#j/mol/k
    Rg = torch.as_tensor(Rg)
    Tr = T*300#有量纲
    cr = c*1000#有量纲
    k = A*torch.exp(-E/Rg/Tr)
    R = k*cr

    # 温度相关物理数
    Tn = T * 300
    k = -0.869083936 + 0.00894880345 * Tn - 1.58366345E-5 * Tn ** 2 + 7.97543259E-9 * Tn ** 3  # W/(m*k)
    cp = 12010.1471 - 80.4072879 * Tn + 0.309866854 * Tn ** 2 - 5.38186884E-4 * Tn ** 3 + 3.62536437E-7 * Tn ** 4  # J/(kg*k)
    alpha = k / (rho * cp)
    Pe = L * uin / alpha


    #传热方程
    heat_f = w1*(u*T_x + v*T_y - (1/Pe)*(T_xx + T_yy) - 0.0159*R)

    loss_heat_f = torch.mean(torch.square(heat_f))

    #边界条件
    #加热筒：T= 1.083
    global x_cyl
    global y_cyl

    net_cyl = torch.cat((x_cyl,y_cyl),1)
    T_cyc_pred = net1(net_cyl)

    loss_heat_cyc = torch.mean(torch.square(T_cyc_pred - 1.083))

    #入口：T=1
    net_inlet = torch.cat((x_inlet,y_inlet),1)
    T_inlet_pred = net1(net_inlet)

    loss_heat_inlet = torch.mean(torch.square(T_inlet_pred - 1))

    # 壁面：dT/dy=0
    global y_wall
    y_wall = y_wall.requires_grad_()

    net_wall = torch.cat((x_wall, y_wall), 1)
    T_wall_pred = net1(net_wall)

    T_wall_pred_y = grad(T_wall_pred.sum(), y_wall, create_graph=True)[0]

    loss_heat_wall = torch.mean(torch.square(T_wall_pred_y))

    # 出口：dT/dx=0
    global x_outlet
    x_outlet = x_outlet.requires_grad_()

    net_outlet = torch.cat((x_outlet, y_outlet), 1)
    T_outlet_pred = net1(net_outlet)

    T_outlet_pred_x = grad(T_outlet_pred.sum(), x_outlet, create_graph=True)[0]

    loss_heat_outlet = torch.mean(torch.square(T_outlet_pred_x))

    # 传热损失函数
    loss_heat = loss_heat_f + loss_heat_cyc + 5 * loss_heat_inlet + loss_heat_wall + loss_heat_outlet
    '''
    #梯度计算
    w0 = list(net1.parameters())[0]
    w5 = list(net1.parameters())[10]

    loss_bc = loss_heat_cyc + 5*loss_heat_inlet + loss_heat_wall + loss_heat_outlet

    grad_f1_f = grad(loss_heat_f.sum(),w0,create_graph=True)[0]
    grad_f1_bc = grad(loss_bc.sum(),w0,create_graph=True)[0]

    grad_f6_f = grad(loss_heat_f.sum(),w5,create_graph=True)[0]
    grad_f6_bc = grad(loss_bc.sum(),w5,create_graph=True)[0]

    #if iter % 10 == 0:
    print('grad_f_layer1_f:',grad_f1_f)
    print('grad_f_layer1_bc:',grad_f1_bc)
    print('grad_f_layer6_f:',grad_f6_f)
    print('grad_f_layer6_bc:',grad_f6_bc)
    '''

    #----------------------------------------------------------------------------

    #对流扩散方程
    rec_f = w2*(u*c_x + v*c_y - (1/25000)*(c_xx + c_yy) + 0.2*R)
    loss_rec_f = torch.mean(torch.square(rec_f))

    #边界条件
    #入口：c=1
    c_inlet_pred = net2(net_inlet)

    loss_rec_inlet = torch.mean(torch.square(c_inlet_pred - 1))

    # 出口：dc/dx=0
    x_outlet = x_outlet.requires_grad_()

    net_outlet = torch.cat((x_outlet, y_outlet), 1)
    c_outlet_pred = net2(net_outlet)

    c_outlet_pred_x = grad(c_outlet_pred.sum(), x_outlet, create_graph=True)[0]

    loss_rec_outlet = torch.mean(torch.square(c_outlet_pred_x))

    # 壁面：dc/dy=0
    y_wall = y_wall.requires_grad_()

    net_wall = torch.cat((x_wall, y_wall), 1)
    c_wall_pred = net2(net_wall)

    c_wall_pred_y = grad(c_wall_pred.sum(), y_wall, create_graph=True)[0]

    loss_rec_wall = torch.mean(torch.square(c_wall_pred_y))

    # 加热筒：(dc/dx)*cos(a)+(dc/dy)*cos(b)
    x_cyl = x_cyl.requires_grad_()
    y_cyl = y_cyl.requires_grad_()

    net_cyl = torch.cat((x_cyl, y_cyl), 1)
    c_cyl_pred = net2(net_cyl)

    c_cyl_pred_x = grad(c_cyl_pred.sum(), x_cyl, create_graph=True)[0]
    c_cyl_pred_y = grad(c_cyl_pred.sum(), y_cyl, create_graph=True)[0]

    c_cyl_pred_grad = cos_alpha * c_cyl_pred_x + cos_beta * c_cyl_pred_y

    loss_rec_cyl = torch.mean(torch.square(c_cyl_pred_grad))

    # 对流扩散损失函数
    loss_rec = 10 * loss_rec_f + loss_rec_inlet + loss_rec_outlet + loss_rec_wall + loss_rec_cyl



    return loss_heat, loss_rec,loss_heat_f ,loss_heat_cyc,loss_heat_inlet,loss_heat_wall,loss_heat_outlet,loss_rec_f, loss_rec_inlet,loss_rec_outlet,loss_rec_wall,loss_rec_cyl


##############
#训练
##############

#CFD验证集
Val = pd.read_csv('hr_data.csv')

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

c_val = Val['c']
c_val = c_val.tolist()
c_val = np.array(c_val)
c_val = c_val.flatten()[:, None]

x_val = torch.as_tensor(x_val, dtype=dtype).to(device)#验证集的x坐标
y_val = torch.as_tensor(y_val, dtype=dtype).to(device)#验证集的y坐标
T_val = torch.as_tensor(T_val, dtype=dtype).to(device)#验证集的T
c_val = torch.as_tensor(c_val, dtype=dtype).to(device)#验证集的c

print("----------开始训练(๑•̀ㅂ•́)و✧------------")

iter = 0

#计算相对误差
def val(x_val,y_val):

    net_val_in = torch.cat((x_val,y_val),1)
    T_pred = net1(net_val_in)
    c_pred = net2(net_val_in)

    error_heat = torch.mean(torch.abs((T_pred - T_val)/T_val))
    error_rec = torch.mean(torch.abs((c_pred - c_val)/c_val))

    return error_heat,error_rec


def closure1():

    net1.zero_grad()
    optimizer1.zero_grad()
    optimizer1_2.zero_grad()

    loss_heat = criterion(x_c,y_c,sdf)[0]
    loss_heat.backward()

    return loss_heat


def closure2():

    global iter

    net2.zero_grad()
    optimizer2.zero_grad()
    optimizer2_2.zero_grad()

    loss_rec = criterion(x_c,y_c,sdf)[1]
    loss_rec.backward()

    error_heat,error_rec = val(x_val,y_val)

    #loss_heat = closure1()

    msg =  f'\r{iter},' + \
            f'loss_rec : {loss_rec.item():.3e}, ' + \
            f'error_heat : {error_heat.item():.2%} '+\
            f'error_rec : {error_rec.item():.2%} '

    print(msg, end = '')

    if iter % 1000 == 0:
        print("")

    iter += 1

    return loss_rec

#Adam求解器
for epoch in range(30000):
    closure1()
    optimizer1.step()

    closure2()
    optimizer2.step()

    scheduler1.step()
    scheduler2.step(criterion(x_c,y_c,sdf)[1])

#optimizer1_2.step(closure1)
optimizer2_2.step(closure2)
##############
#保存网络
##############
#torch.save(net1.state_dict(), "pkl2/hr_wb_h2.pkl")
#torch.save(net2.state_dict(), "pkl2/hr_wb_r2.pkl")

#可视化
#tensorboard --logdir=C:\Users\pse\PycharmProjects\limj\2022\20220301\HR\log_hr