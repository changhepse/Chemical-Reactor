import torch
import numpy as np
import torch.nn as nn
from torch.autograd import grad
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from torch.optim.lr_scheduler import ExponentialLR,CosineAnnealingLR,ReduceLROnPlateau
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
            #layer(h_n, h_n, nn.Tanh()),
            #layer(h_n, h_n, nn.Tanh()),
            layer(h_n, h_n, nn.Tanh()),
            layer(h_n, h_n, nn.Tanh()),
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
        #nn.init.zeros_(m.bias)
net.apply(init_normal)

#优化器
optimizer1 = optim.Adam(net.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
#scheduler =CosineAnnealingLR(optimizer1, 5000)
scheduler = ReduceLROnPlateau(optimizer1, 'min',factor=0.99,patience=50,verbose=False)
#scheduler = ExponentialLR(optimizer1, gamma=0.99975)
optimizer2 = optim.LBFGS(
    net.parameters(),
    lr=1.0,
    max_iter=80000,
    max_eval=80000,
    history_size=50,
    tolerance_grad=1e-10,
    tolerance_change=1.0 * np.finfo(float).eps,
    line_search_fn="strong_wolfe")
#####################
#数据集
#######################
#物性
L=0.1#m
uin=5e-4#m/s
D=2e-9#m2/s
Pe = L*uin/D
print('Pe:',Pe)
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
x_cyl = geo3.CYLD_cos[:, 0:1]
y_cyl = geo3.CYLD_cos[:, 1:2]
cos_alpha = geo3.CYLD_cos[:, 2:3]
cos_beta = geo3.CYLD_cos[:, 3:4]
x_cyl = torch.as_tensor(x_cyl, dtype=dtype).to(device)
y_cyl = torch.as_tensor(y_cyl, dtype=dtype).to(device)
cos_alpha = torch.as_tensor(cos_alpha, dtype=dtype).to(device)
cos_beta = torch.as_tensor(cos_beta, dtype=dtype).to(device)
#整合数据集
#dataset = TensorDataset(x_c,y_c,sdf)
#dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True,drop_last = True)

##############
#定义损失函数
##############

#读取NS方程
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


#读取H方程
h_nh=64
class DNN_h(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            layer(2, h_nh, nn.Tanh()),
            #layer(h_nh, h_nh, nn.Tanh()),
            layer(h_nh, h_nh, nn.Tanh()),
            layer(h_nh, h_nh, nn.Tanh()),
            layer(h_nh, h_nh, nn.Tanh()),
            layer(h_nh, h_nh, nn.Tanh()),
            layer(h_nh, 1),
        )

    def forward(self, xy):
        return self.net(xy)

net_h = DNN_h().to(device)
net_h.load_state_dict(torch.load('./pkl2/h_wbt.pkl'))

################
#损失函数作图
##################
import matplotlib.pyplot as plt
def plot(f):

    global x_c
    global y_c

    x_c = x_c.cpu().detach().numpy()
    y_c = y_c.cpu().detach().numpy()
    loss_f = f.cpu().detach().numpy()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    xmin=0
    xmax=1.0
    ymin=0
    ymax=0.2
    alpha=0.5
    s=3
    marker='o'


    cf = ax.scatter(x_c, y_c, c=loss_f, alpha=alpha-0.1, edgecolors='none', cmap='rainbow', marker=marker, s=int(s))
    ax.axis('square')

    for key, spine in ax.spines.items():
        if key in ['right','top','left','bottom']:
            spine.set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_title(r'$Loss-f$ ')
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    plt.savefig('./loss_r/loss'+str(iter)+'.png', dpi=300)

    x_c = torch.as_tensor(x_c, dtype=dtype).to(device)
    y_c = torch.as_tensor(y_c, dtype=dtype).to(device)


#损失函数
def criterion(x,y,sdf):

    w = 1

    x = x.requires_grad_()
    y = y.requires_grad_()

    net_in = torch.cat((x,y),1)
    c = net(net_in)

    c_x = grad(c.sum(), x, create_graph=True)[0]
    c_y = grad(c.sum(), y, create_graph=True)[0]
    c_xx = grad(c_x.sum(), x, create_graph=True)[0]
    c_yy = grad(c_y.sum(), y, create_graph=True)[0]


    #导入速度场
    uv = net_ns(net_in)[:,0:1]
    u = grad(uv.sum(),y, create_graph=True)[0]
    v = - grad(uv.sum(),x, create_graph=True)[0]
    with torch.no_grad():
        u = u
        v = v
        T = net_h(net_in)

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


    #对流扩散方程
    rec_f = w*(u*c_x + v*c_y - (1/Pe)*(c_xx + c_yy) + 0.2*R)
    loss_rec_f = torch.mean(torch.square(rec_f))


    #边界条件
    #入口：c=1
    net_inlet = torch.cat((x_inlet,y_inlet),1)
    c_inlet_pred = net(net_inlet)

    loss_rec_inlet = torch.mean(torch.square(c_inlet_pred - 1))


    #出口：dc/dx=0
    global x_outlet
    x_outlet = x_outlet.requires_grad_()

    net_outlet = torch.cat((x_outlet,y_outlet),1)
    c_outlet_pred = net(net_outlet)

    c_outlet_pred_x = grad(c.sum(), x_outlet, create_graph = True)[0]

    loss_rec_outlet = torch.mean(torch.square(c_outlet_pred_x))

    #壁面：dc/dy=0
    global y_wall
    y_wall = y_wall.requires_grad_()

    net_wall = torch.cat((x_wall,y_wall),1)
    c_wall_pred = net(net_wall)

    c_wall_pred_y = grad(c_wall_pred.sum(), y_wall, create_graph = True)[0]

    loss_rec_wall = torch.mean(torch.square(c_wall_pred_y))

    #加热筒：(dc/dx)*cos(a)+(dc/dy)*cos(b)
    global x_cyl
    global y_cyl
    x_cyl = x_cyl.requires_grad_()
    y_cyl = y_cyl.requires_grad_()

    net_cyl = torch.cat((x_cyl,y_cyl),1)
    c_cyl_pred = net(net_cyl)

    c_cyl_pred_x = grad(c_cyl_pred.sum(), x_cyl, create_graph = True)[0]
    c_cyl_pred_y = grad(c_cyl_pred.sum(), y_cyl, create_graph = True)[0]

    c_cyl_pred_grad = cos_alpha*c_cyl_pred_x + cos_beta*c_cyl_pred_y

    loss_rec_cyl = torch.mean(torch.square(c_cyl_pred_grad))

    #对流扩散损失函数
    loss_rec = 10*loss_rec_f + loss_rec_inlet+ loss_rec_outlet + loss_rec_wall + loss_rec_cyl

    #if iter % 2000==0 :
        #plot(f=rec_square_f)

    return loss_rec,loss_rec_f,loss_rec_inlet,loss_rec_outlet,loss_rec_wall ,loss_rec_cyl


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

c_val = Val['c']
c_val = c_val.tolist()
c_val = np.array(c_val)
c_val = c_val.flatten()[:, None]

x_val = torch.as_tensor(x_val, dtype=dtype).to(device)#验证集的x坐标
y_val = torch.as_tensor(y_val, dtype=dtype).to(device)#验证集的y坐标
c_val = torch.as_tensor(c_val, dtype=dtype).to(device)#验证集的c

print("-------开始训练(๑•̀ㅂ•́)و✧-------")

iter = 0

#计算相对误差
def val(x_val,y_val):

    net_val_in = torch.cat((x_val,y_val),1)
    c_pred = net(net_val_in)

    error_rec = torch.mean(torch.abs((c_pred - c_val)/c_val))

    return error_rec


def closure():

    global iter
    net.zero_grad()
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    loss_rec = criterion(x_c,y_c,sdf)[0]

    loss_rec_f = criterion(x_c,y_c,sdf)[1]
    loss_rec_inlet = criterion(x_c,y_c,sdf)[2]
    loss_rec_outlet = criterion(x_c,y_c,sdf)[3]
    loss_rec_wall = criterion(x_c,y_c,sdf)[4]
    loss_rec_cyl = criterion(x_c,y_c,sdf)[5]

    loss_rec.backward()

    error_rec = val(x_val,y_val)

    '''
    if iter % 100 == 0:

        writer = SummaryWriter('log_r/loss', flush_secs=60)
        writer.add_scalar('总损失/Loss',loss_rec,iter)
        writer.add_scalar('函数损失/Loss_f',loss_rec_f,iter)
        writer.add_scalar('边界损失/Loss_inlet',loss_rec_inlet ,iter)
        writer.add_scalar('边界损失/Loss_outlet',loss_rec_outlet,iter)
        writer.add_scalar('边界损失/Loss_wall',loss_rec_wall,iter)
        writer.add_scalar('边界损失/Loss_cyl',loss_rec_cyl,iter)

        #writer = SummaryWriter('log_r/error', flush_secs=60)
        #writer.add_scalar('相对误差/error_rec',error_rec,iter)
    '''
    msg = f"\r{iter}, " + \
          f"loss : {loss_rec.item():.3e}, " + \
          f"error_rec : {error_rec.item():.2%} "
    print(msg, end = "")

    if iter % 1000 == 0:
        print("")

    iter += 1

    return loss_rec

for epoch in range(30000):
    closure()
    optimizer1.step()
    scheduler.step(criterion(x_c,y_c,sdf)[0])

optimizer2.step(closure)

##############
#保存网络
##############
torch.save(net.state_dict(), "./pkl2/r_wbt2.pkl")

#可视化
#tensorboard --logdir=D:\Project\code\2022\20220301\3\H_R\log_r