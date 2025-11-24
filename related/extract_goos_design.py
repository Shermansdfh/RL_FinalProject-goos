"""
從 GOOS 優化結果中提取設計網格的示例腳本

這個腳本展示了如何：
1. 從 GOOS 優化計劃中提取設計參數
2. 將設計參數轉換為二進制網格（供 MEEP 使用）
3. 可視化設計網格
4. 保存設計網格供後續使用
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from spins import goos
from spins.goos_sim import maxwell


def extract_design_from_plan(save_folder, step=None, var_name="design_var"):
    """
    從 GOOS 優化計劃中提取設計參數
    
    Args:
        save_folder: 優化計劃的保存文件夾
        step: 要讀取的步驟（None 表示最新步驟）
        var_name: 設計變數的名稱
    
    Returns:
        design_params: 設計參數（連續值，通常在 [0, 1] 範圍內）
        design_var: 設計變數節點（用於進一步操作）
    """
    with goos.OptimizationPlan() as plan:
        # 加載優化計劃
        plan.load(save_folder)
        
        # 獲取設計變數
        try:
            design_var = plan.get_node(var_name)
        except KeyError:
            print(f"警告: 找不到名為 '{var_name}' 的變數")
            print(f"可用的變數: {list(plan._node_map.keys())}")
            return None, None
        
        # 讀取檢查點（如果指定）
        if step is not None:
            checkpoint_file = os.path.join(save_folder, f"step{step}.pkl")
            if os.path.exists(checkpoint_file):
                plan.read_checkpoint(checkpoint_file)
                print(f"已讀取檢查點: step{step}")
            else:
                print(f"警告: 檢查點文件不存在: {checkpoint_file}")
        else:
            # 嘗試讀取最新的檢查點
            try:
                latest_step = goos.util.get_latest_log_step(save_folder)
                checkpoint_file = os.path.join(save_folder, f"step{latest_step}.pkl")
                if os.path.exists(checkpoint_file):
                    plan.read_checkpoint(checkpoint_file)
                    print(f"已讀取最新檢查點: step{latest_step}")
            except:
                print("警告: 無法讀取檢查點，使用當前變數值")
        
        # 獲取設計參數值
        design_params = plan.get_var_value(design_var)
        
        print(f"設計參數形狀: {design_params.shape}")
        print(f"設計參數範圍: [{design_params.min():.3f}, {design_params.max():.3f}]")
        print(f"設計參數均值: {design_params.mean():.3f}")
        
        return design_params, design_var


def extract_epsilon_grid(save_folder, eps_node_name="eps_rendered"):
    """
    從 GOOS 優化計劃中提取渲染後的介電常數網格
    
    Args:
        save_folder: 優化計劃的保存文件夾
        eps_node_name: 渲染節點的名稱
    
    Returns:
        eps_grid: 介電常數網格（通常是 z 分量）
    """
    with goos.OptimizationPlan() as plan:
        plan.load(save_folder)
        
        try:
            eps_rendered = plan.get_node(eps_node_name)
            eps_data = eps_rendered.get().array
            # 對於 2D 問題，通常使用 z 分量
            if isinstance(eps_data, list) and len(eps_data) >= 3:
                eps_grid = eps_data[2]  # z 分量
            else:
                eps_grid = eps_data
            return eps_grid
        except KeyError:
            print(f"警告: 找不到名為 '{eps_node_name}' 的渲染節點")
            return None


def convert_to_binary_grid(design_params, threshold=0.5):
    """
    將連續的設計參數轉換為二進制網格
    
    Args:
        design_params: 連續設計參數（通常在 [0, 1] 範圍內）
        threshold: 二值化閾值
    
    Returns:
        binary_grid: 二進制網格（0 = 空氣，1 = 矽）
    """
    binary_grid = (design_params > threshold).astype(int)
    return binary_grid


def convert_to_meep_format(binary_grid, flatten_2d=True):
    """
    將二進制網格轉換為 MEEP 使用的格式
    
    Args:
        binary_grid: 二進制網格
        flatten_2d: 如果是 2D 問題，是否展平為 2D 數組
    
    Returns:
        meep_grid: MEEP 格式的網格
    """
    if len(binary_grid.shape) == 3 and flatten_2d:
        # 對於 3D 數組，取中間層（通常是 z 方向）
        z_mid = binary_grid.shape[2] // 2
        meep_grid = binary_grid[:, :, z_mid]
    else:
        meep_grid = binary_grid
    
    return meep_grid


def visualize_design(design_params, binary_grid=None, save_path=None):
    """
    可視化設計網格
    
    Args:
        design_params: 連續設計參數
        binary_grid: 二進制網格（可選）
        save_path: 保存路徑（可選）
    """
    # 處理 3D 數組
    if len(design_params.shape) == 3:
        z_mid = design_params.shape[2] // 2
        design_2d = design_params[:, :, z_mid]
        if binary_grid is not None:
            binary_2d = binary_grid[:, :, z_mid]
    else:
        design_2d = design_params
        binary_2d = binary_grid
    
    fig, axes = plt.subplots(1, 2 if binary_grid is not None else 1, 
                             figsize=(12, 5))
    
    if binary_grid is not None:
        axes = axes.flatten()
    
    # 繪製連續設計參數
    ax = axes[0] if binary_grid is not None else axes
    im1 = ax.imshow(design_2d, cmap='viridis', origin='lower')
    ax.set_title('連續設計參數')
    ax.set_xlabel('X (像素)')
    ax.set_ylabel('Y (像素)')
    plt.colorbar(im1, ax=ax, label='設計參數值')
    
    # 繪製二進制網格
    if binary_grid is not None:
        im2 = axes[1].imshow(binary_2d, cmap='binary', origin='lower')
        axes[1].set_title('二進制網格 (閾值=0.5)')
        axes[1].set_xlabel('X (像素)')
        axes[1].set_ylabel('Y (像素)')
        plt.colorbar(im2, ax=axes[1], label='材料 (0=空氣, 1=矽)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"設計可視化已保存到: {save_path}")
    
    plt.show()


def load_from_checkpoint(save_folder, step=None):
    """
    直接從檢查點文件加載設計數據
    
    Args:
        save_folder: 優化計劃的保存文件夾
        step: 要讀取的步驟（None 表示最新步驟）
    
    Returns:
        data: 檢查點數據字典
    """
    if step is None:
        try:
            step = goos.util.get_latest_log_step(save_folder)
        except:
            print("無法確定最新步驟，請手動指定 step 參數")
            return None
    
    checkpoint_file = os.path.join(save_folder, f"step{step}.pkl")
    
    if not os.path.exists(checkpoint_file):
        print(f"錯誤: 檢查點文件不存在: {checkpoint_file}")
        return None
    
    with open(checkpoint_file, "rb") as fp:
        data = pickle.load(fp)
    
    print(f"已從檢查點加載數據: step{step}")
    print(f"檢查點包含的鍵: {list(data.keys())}")
    
    # 顯示變數值
    if "var_values" in data:
        print(f"\n變數值:")
        for var_name, var_value in data["var_values"].items():
            if isinstance(var_value, np.ndarray):
                print(f"  {var_name}: shape={var_value.shape}, "
                      f"range=[{var_value.min():.3f}, {var_value.max():.3f}]")
            else:
                print(f"  {var_name}: {var_value}")
    
    # 顯示監控數據
    if "monitor_data" in data:
        print(f"\n監控數據:")
        for monitor_name in data["monitor_data"].keys():
            print(f"  {monitor_name}")
    
    return data


def main():
    """
    主函數：演示如何提取和使用 GOOS 設計網格
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="從 GOOS 優化結果中提取設計網格"
    )
    parser.add_argument(
        "save_folder",
        type=str,
        help="GOOS 優化計劃的保存文件夾"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="要讀取的步驟（默認為最新步驟）"
    )
    parser.add_argument(
        "--var-name",
        type=str,
        default="design_var",
        help="設計變數的名稱"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="二值化閾值"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="輸出文件路徑（保存設計網格）"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="可視化設計網格"
    )
    
    args = parser.parse_args()
    
    # 檢查文件夾是否存在
    if not os.path.exists(args.save_folder):
        print(f"錯誤: 文件夾不存在: {args.save_folder}")
        return
    
    print("=" * 60)
    print("從 GOOS 優化結果提取設計網格")
    print("=" * 60)
    
    # 方法 1: 從優化計劃提取
    print("\n[方法 1] 從優化計劃提取設計參數")
    design_params, design_var = extract_design_from_plan(
        args.save_folder,
        step=args.step,
        var_name=args.var_name
    )
    
    if design_params is None:
        print("無法提取設計參數，嘗試從檢查點直接加載...")
        data = load_from_checkpoint(args.save_folder, args.step)
        if data and "var_values" in data:
            if args.var_name in data["var_values"]:
                design_params = data["var_values"][args.var_name]
            else:
                print(f"檢查點中沒有 '{args.var_name}' 變數")
                print("可用的變數:", list(data["var_values"].keys()))
                return
        else:
            return
    
    # 轉換為二進制網格
    print("\n[轉換] 將連續參數轉換為二進制網格")
    binary_grid = convert_to_binary_grid(design_params, threshold=args.threshold)
    print(f"二進制網格形狀: {binary_grid.shape}")
    print(f"矽像素數: {binary_grid.sum()} / {binary_grid.size} "
          f"({100 * binary_grid.sum() / binary_grid.size:.1f}%)")
    
    # 轉換為 MEEP 格式
    print("\n[轉換] 轉換為 MEEP 格式")
    meep_grid = convert_to_meep_format(binary_grid, flatten_2d=True)
    print(f"MEEP 網格形狀: {meep_grid.shape}")
    
    # 保存設計網格
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(args.save_folder, "extracted_design_grid.npy")
    
    np.save(output_path, meep_grid)
    print(f"\n設計網格已保存到: {output_path}")
    
    # 同時保存連續參數（如果需要）
    continuous_path = output_path.replace(".npy", "_continuous.npy")
    np.save(continuous_path, design_params)
    print(f"連續設計參數已保存到: {continuous_path}")
    
    # 可視化
    if args.visualize:
        print("\n[可視化] 顯示設計網格")
        vis_path = output_path.replace(".npy", "_visualization.png")
        visualize_design(design_params, binary_grid, save_path=vis_path)
    
    # 嘗試提取介電常數網格
    print("\n[方法 2] 嘗試提取介電常數網格")
    eps_grid = extract_epsilon_grid(args.save_folder)
    if eps_grid is not None:
        print(f"介電常數網格形狀: {eps_grid.shape}")
        eps_path = output_path.replace(".npy", "_epsilon.npy")
        np.save(eps_path, eps_grid)
        print(f"介電常數網格已保存到: {eps_path}")
    
    print("\n" + "=" * 60)
    print("提取完成！")
    print("=" * 60)
    print(f"\n生成的文件:")
    print(f"  - 二進制網格: {output_path}")
    print(f"  - 連續參數: {continuous_path}")
    if eps_grid is not None:
        print(f"  - 介電常數: {eps_path}")
    if args.visualize:
        print(f"  - 可視化: {vis_path}")


if __name__ == "__main__":
    main()

