# https://blog.csdn.net/algorithmPro/article/details/103982466
# 在测试阶段
# 初始化模型，并设置模型处于测试阶段
import torch
import torch.nn as nn
import copy

m3 = nn.BatchNorm2d(3, eps=0, momentum=0.5, affine=True, track_running_stats=True).cuda()
# 测试阶段
m3.eval()
# 为了方便验证，设置模型参数的值
m3.running_mean = (torch.ones([3]) * 4).cuda()  # 设置模型的均值是4
m3.running_var = (torch.ones([3]) * 2).cuda()  # 设置模型的方差是2

# 查看模型参数的值
print('trainning:', m3.training)
print('running_mean:', m3.running_mean)
print('running_var:', m3.running_var)
# gamma对应模型的weight，默认值是1
print('weight:', m3.weight)
# gamma对应模型的bias，默认值是0
print('bias:', m3.bias)

# # >
# trainning: False
# running_mean: tensor([4., 4., 4.], device='cuda:0')
# running_var: tensor([2., 2., 2.], device='cuda:0')
# weight: Parameter
# containing:
# tensor([1., 1., 1.], device='cuda:0', requires_grad=True)
# bias: Parameter
# containing:
# tensor([0., 0., 0.], device='cuda:0', requires_grad=True)

# 初始化输入数据，并计算输入数据的均值和方差
# 生成通道3，416行416列的输入数据
torch.manual_seed(21)
input3 = torch.randn(1, 3, 416, 416).cuda()
# 输入数据的均值
obser_mean = torch.Tensor([input3[0][i].mean() for i in range(3)]).cuda()
# 输入数据的方差
obser_var = torch.Tensor([input3[0][i].var() for i in range(3)]).cuda()
# 打印
print('obser_mean:', obser_mean)
print('obser_var:', obser_var)

# >
# obser_mean: tensor([0.0047, 0.0029, 0.0014], device='cuda:0')
# obser_var: tensor([1.0048, 0.9898, 1.0024], device='cuda:0')

ex_old = copy.deepcopy(m3.running_mean)
var_old = copy.deepcopy(m3.running_var)
print('*' * 30)
print('程序计算bn前的均值ex_new:', ex_old)
print('程序计算bn前的方差var_new:', var_old)

# 数据归一化
output3 = m3(input3)
# 输出归一化后的第一个通道的数据


print('*' * 30)
print('程序计算bn后的均值ex_new:', m3.running_mean)
print('程序计算bn后的方差var_new:', m3.running_var)

# 归一化函数实现
output3_calcu = torch.zeros_like(input3)
for channel in range(input3.shape[1]):
    output3_calcu[0][channel] = (input3[0][channel] - m3.running_mean[channel]) / (
        pow(m3.running_var[channel] + m3.eps, 0.5))

print("程序计算的输出bn：")
print(output3)
print("手动计算的输出bn：")
print(output3_calcu)

# 由结果可知，执行测试阶段的froward函数后，模型的running_mean和running_var不改变。