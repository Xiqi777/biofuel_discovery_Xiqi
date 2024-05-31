import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils

# 读取CSV文件
data = pd.read_csv('dataset Xiqi.csv')

# 将数据转换为合适的类型
# 假设你的特征列从第二列开始
features = data.iloc[:, 4:].values.astype(np.float32)

# 创建PyTorch张量
features_tensor = torch.tensor(features)

# 创建数据加载器
batch_size = 64
data_loader = data_utils.DataLoader(features_tensor, batch_size=batch_size, shuffle=True)


# 定义生成器
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.map1(x))
        return torch.sigmoid(self.map2(x))


# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.map1(x))
        return torch.sigmoid(self.map2(x))


# 初始化生成器和判别器
input_size = features.shape[1]  # 特征的维度
hidden_size = 128
output_size = input_size

generator = Generator(input_size, hidden_size, output_size)
discriminator = Discriminator(input_size, hidden_size, 1)

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二分类交叉熵损失
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

# 训练GAN模型
generated_data = []

num_epochs = 100
for epoch in range(num_epochs):
    for data_batch in data_loader:
        # 训练判别器
        real_data = data_batch
        fake_data = generator(torch.randn(batch_size, input_size))
        dis_optimizer.zero_grad()
        dis_real_output = discriminator(real_data)
        dis_fake_output = discriminator(fake_data.detach())
        dis_loss_real = criterion(dis_real_output, torch.ones_like(dis_real_output))
        dis_loss_fake = criterion(dis_fake_output, torch.zeros_like(dis_fake_output))
        dis_loss = dis_loss_real + dis_loss_fake
        dis_loss.backward()
        dis_optimizer.step()

        # 训练生成器
        gen_optimizer.zero_grad()
        fake_data = generator(torch.randn(batch_size, input_size))
        dis_fake_output = discriminator(fake_data)
        gen_loss = criterion(dis_fake_output, torch.ones_like(dis_fake_output))
        gen_loss.backward()
        gen_optimizer.step()

        # 保存生成的数据
        generated_data.append(fake_data.detach().numpy())

# 将生成的数据保存到文件中
generated_data = np.concatenate(generated_data, axis=0)
np.savetxt('generated_data.csv', generated_data, delimiter=',')


