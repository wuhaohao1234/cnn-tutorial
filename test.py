import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 定义神经网络模型
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x

# 加载模型
def load_model(model, weights_path):
    model.load_state_dict(torch.load(weights_path))
    model.eval()

# 预处理测试图片
def preprocess_image(image_path):
    # to_tensor = transforms.Compose([transforms.ToTensor()])
    test_image = Image.open(image_path)
    # 获取图像的宽度和高度

    # 转换为灰度图像或者保留三个通道
    # gray_image = test_image.convert('L')  # 转换为灰度图
    resized_image = test_image.resize((28, 28)).convert('L')

    # 进行预处理
    to_tensor = transforms.Compose([transforms.ToTensor()])
    processed_image = to_tensor(resized_image).view(1, -1)
    # 显示预处理后的形状
    print("Processed image shape:", processed_image.shape)
    return processed_image

# 运行预测
def run_prediction(model, processed_image):
    with torch.no_grad():
        output = model(processed_image)
        predicted_class = torch.argmax(output[0]).item()
    return predicted_class

def main():
    # 创建神经网络模型
    net = Net()

    # 加载之前训练好的模型权重
    load_model(net, 'model_weights_epoch5.pth')

    # 定义测试图片路径
    test_image_path = 'img/test.png'

    # 预处理测试图片
    processed_image = preprocess_image(test_image_path)

    # # 运行预测
    predicted_class = run_prediction(net, processed_image)
    print(predicted_class)
    # 显示测试图片和预测结果
    test_image = Image.open(test_image_path)
    # plt.imshow(test_image, cmap='gray')
    # plt.title("Prediction: {}".format(predicted_class))
    # plt.show()

if __name__ == "__main__":
    main()