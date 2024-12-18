import os
import math
import torch

# 获取文件大小的函数
def get_file_size(file_path):
    file_size = os.path.getsize(file_path)  # 获取文件大小，单位是字节
    return file_size

# 将字节转换为更易读的单位
def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, 2)
    s = round(size_bytes / p, 5)
    return s 

import torch

def quantize_to_int4(tensor):
    # 将 tensor 映射到 [0, 15] 范围，并四舍五入
    tensor = torch.round(tensor * 15).clamp(0, 15).to(torch.int8)
    return tensor

def get_size_MB(file_path):
    size_in_bytes = get_file_size(file_path)
    size_in_MB = convert_size(size_in_bytes)
    return size_in_MB

if __name__ == '__main__':
    # file_path = "models/ldm/cin256-v2/model.ckpt"  # 替换为你的文件路径
    # size_in_bytes = get_file_size(file_path)
    # print(f"File Size: {convert_size(size_in_bytes)}")

    w4a8_ckpt = torch.load('reproduce/step20_quantw4a8_ldm_brecq_dntc_1000classes.pth', map_location='cpu')
    fp32_ckpt = torch.load('models/ldm/cin256-v2/model.ckpt', map_location='cpu')
    # x = w4a8_ckpt

    # for key in fp32_ckpt['state_dict']:
    #     if 'weight' in key:
    #         fp32_ckpt['state_dict'][key] = fp32_ckpt['state_dict'][key].to(torch.int8)

    # # 如果需要保存转换后的检查点
    # torch.save(fp32_ckpt, 'test/test_w8a8_weights.pth')

    # calculate w8a8_weight size
    # file_path = 'test/test_w8a8_weights.pth'
    # size_in_bytes = get_file_size(file_path)
    # size_in_MB = convert_size(size_in_bytes)
    # print("w8a8 size: ", size_in_MB, " MB")

    # calculate w4a8_weight size
    # save_dict_int8 = {}
    # save_dict_int16 = {}
    # keys_to_delete = []
    # for key in fp32_ckpt['state_dict']:
    #     if 'weight' in key:
    #         x_int8 = fp32_ckpt['state_dict'][key].to(torch.int8)
    #         x_int16 = fp32_ckpt['state_dict'][key].to(torch.int16)

    #         save_dict_int8[key] = x_int8
    #         save_dict_int16[key] = x_int16
    #         keys_to_delete.append(key)

    # for key in keys_to_delete:
    #     del fp32_ckpt['state_dict'][key]

    # torch.save(fp32_ckpt, 'test/weight_fp32_del_w.pth')
    # torch.save(save_dict_int16, 'test/int16_weights.pth')
    # torch.save(save_dict_int8, 'test/int8_weights.pth')

    fp32_mb = get_size_MB('test/weight_fp32_del_w.pth')
    # int8w_mb = get_size_MB('test/int8_weights.pth')
    # int16w_mb = get_size_MB('test/int16_weights.pth')
    # fp32_org_mb = get_size_MB('models/ldm/cin256-v2/model.ckpt')

    # print('fp32 org size: ', fp32_org_mb, ' MB')
    # print('fp32 del w size: ', fp32_mb, ' MB')
    # print('int16 w size: ', int16w_mb, ' MB')
    # print('int8 w size: ', int8w_mb, ' MB')

    def pack_int8_to_int4(tensor):
        # Flatten the tensor to 1D
        flat_tensor = tensor.flatten()

        # Ensure the length is even to allow pairing
        if flat_tensor.size(0) % 2 != 0:
            flat_tensor = torch.cat([flat_tensor, torch.tensor([0], dtype=torch.int8)])

        # Pack every two int8 values into one int8 (which simulates two int4)
        packed_tensor = (flat_tensor[::2] << 4) | (flat_tensor[1::2] & 0x0F)

        return packed_tensor

    def get_size_MB(filepath):
        size_bytes = os.path.getsize(filepath)
        return size_bytes / (1024 * 1024)

    # 加载 int8 权重的 pth 文件
    int8_ckpt = torch.load('test/int8_weights.pth')

    # 创建一个新的字典来保存 int4 权重
    int4_ckpt = {}

    # 遍历 int8 权重
    for key, value in int8_ckpt.items():
        # 将 int8 拼接成 int4
        int4_ckpt[key] = pack_int8_to_int4(value)

    # 保存为新的 pth 文件
    torch.save(int4_ckpt, 'test/int4_weights.pth')

    # 计算并输出文件大小
    int4w_mb = get_size_MB('test/int4_weights.pth')
    total_mb = fp32_mb + int4w_mb
    print(f"int4 weights size: {total_mb} MB")
