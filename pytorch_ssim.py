import h5py

def inspect_h5_file_limited(file_path, max_items=5):
    """
    读取 .h5 文件，只输出前 max_items 个字段的名称和形状（如果是数据集）。
    
    参数:
        file_path (str): .h5 文件的路径
        max_items (int): 最多输出多少个字段
    """
    with h5py.File(file_path, 'r') as f:
        print(f"HDF5 文件前 {max_items} 项结构: {file_path}")
        print("=" * 60)
        
        count = 0
        
        def print_name_and_shape(name, obj):
            nonlocal count
            if count >= max_items:
                return  # 超出数量则不处理
            
            if isinstance(obj, h5py.Dataset):
                print(f"字段路径: {name}")
                print(f"  形状: {obj.shape}")
                print(f"  数据类型: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"组路径: {name} (Group)")
            
            count += 1

        # 递归遍历，但通过 count 控制输出数量
        f.visititems(print_name_and_shape)

# 使用示例
if __name__ == "__main__":
    h5_file_path = './VLFDataset_h5/MSRS_train.h5'  # 替换为你的 .h5 文件路径
    inspect_h5_file_limited(h5_file_path, max_items=5)