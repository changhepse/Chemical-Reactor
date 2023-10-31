###########################################
#画几何、采样、SDF(修改）
##########################################

from pyDOE import lhs
import matplotlib.pyplot as plt
import numpy as np

#反应器参数
l = 1.0#长度，无量纲后，m
h = 0.2#高度，无量纲后，m
r = 0.04#加热筒半径，无量纲后，m
xc = 0.5#加热筒x位置

#取点数
wall_p = 400#上下壁
cyl_p = 400#加热筒
inlet_p = 400#入口
outlet_p = 200#出口
coll_p = 5000#内部点

lb = np.array([0, 0])
ub = np.array([l, h])

#边界(上下左右)
wall_up = [0, h] + [l, 0]* lhs(2, wall_p)#上边界
wall_lw = [0, 0] + [l, 0] * lhs(2, wall_p)#下边界
WALL = np.concatenate((wall_up, wall_lw), 0)#上下边界总

#入口
INLET = [0, 0] + [0.0, h] * lhs(2, inlet_p)#入口
U_max = 1
y_INLET = INLET[:,1:2]
u_INLET = 4.0 * y_INLET * (h - y_INLET) / (h ** 2)
v_INLET = 0*y_INLET
INLET_uv = np.concatenate((INLET, u_INLET, v_INLET), 1)#数组拼接
plt.scatter(INLET_uv[:, 1:2], INLET_uv[:, 2:3], marker='o', alpha=0.2, color='red')
plt.show()

#出口
OUTLET = [l, 0] + [0, h] * lhs(2, outlet_p)#出口

#圆环
theta = [0.0] + [2*np.pi] * lhs(1, cyl_p)
x_CYLD = np.multiply(r, np.cos(theta))+ xc
y_CYLD = np.multiply(r, np.sin(theta))+ h/2
CYLD = np.concatenate((x_CYLD, y_CYLD), 1)#数组拼接，1表示行拼接
#圆环计算角度
cos_alpha = (2*x_CYLD-1)#/np.sqrt((2*x_CYLD-1)**2 + (2*y_CYLD-0.2)**2)
cos_beta = (2*y_CYLD-0.2)#/np.sqrt((2*x_CYLD-1)**2 + (2*y_CYLD-0.2)**2)
#合并
CYLD_cos = np.concatenate((x_CYLD, y_CYLD,cos_alpha,cos_beta), 1)
#print(CYLD_cos)

#无滑移边界
NoSWALL = np.concatenate((WALL,CYLD),0)

#内部点（减去圆）
XY_c = lb + (ub - lb) * lhs(2, coll_p)
def DelCylPT(XY_c, xc, yc, r):
    dst = np.array([((xy[0] - xc) ** 2 + (xy[1] - yc) ** 2) ** 0.5 for xy in XY_c])
    return XY_c[dst>r,:]
XY_c = DelCylPT(XY_c, xc=xc, yc=h/2, r=r)


#采样的所有点
#XY = np.concatenate((INLET,OUTLET, WALL, CYLD, XY_c),0)#0列拼接，1行拼接


#作图
fig, ax = plt.subplots()
ax.set_aspect('equal')
plt.scatter(INLET_uv[:, 0:1], INLET_uv[:, 1:2], s=0.5,marker='o', alpha=1, color='red')#入口
plt.scatter(OUTLET[:, 0:1], OUTLET[:, 1:2], s=0.5,marker='o', alpha=1, color='red')#出口
plt.scatter(WALL[:,0:1], WALL[:,1:2],s=0.5,marker='o', alpha=1 ,color='red')#上下壁面
plt.scatter(CYLD[:,0:1], CYLD[:,1:2], s=0.5,marker='o', alpha=1 ,color='orange')#加热筒
plt.scatter(XY_c[:,0:1], XY_c[:,1:2], s=0.5,marker='o', alpha=0.1 ,color='blue')#内部域
#plt.scatter(XY[:,0:1], XY[:,1:2], s=0.5,marker='o', alpha=0.1 ,color='blue')#所有点
plt.show()

#计算SDF
def sdf(x,y):

    sdf_input = np.abs(x-0)
    sdf_wall_up = np.abs(y-h)
    sdf_wall_low = np.abs(y-0)
    sdf_c = np.sqrt((x-xc)**2 + (y-h/2)**2) - r

    sdf = min(sdf_input,sdf_wall_up,sdf_wall_low,sdf_c)

    return sdf



SDF = []

for idx, (x,y) in enumerate(XY_c):

    sdf_p = sdf(x,y)

    SDF.append(sdf_p)

SDF = np.array(SDF)
SDF = SDF.flatten()[:, None]
SDF = SDF*10
#print('SDF:',SDF.shape)

#内部点+sdf
XY_c_sdf = np.concatenate((XY_c, SDF),1)
#print(XY_c_sdf)

#sdf作图
x = XY_c_sdf[:,0:1]
y = XY_c_sdf[:,1:2]
SDF = XY_c_sdf[:,2:3]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
fig.subplots_adjust(hspace=0.2, wspace=0.2)
xmin=0
xmax=1.0
ymin=0
ymax=0.2
s=3
marker='o'

# Plot MIXED result
#T
cf = ax.scatter(x, y, c=SDF, alpha=0.5, edgecolors='none', cmap='rainbow', marker=marker, s=int(s))
ax.axis('square')

for key, spine in ax.spines.items():
    if key in ['right','top','left','bottom']:
        spine.set_visible(False)

ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
ax.set_title(r'$SDF$ ')
fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
plt.show()