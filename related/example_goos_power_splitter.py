"""
GOOS 功率分束器優化示例

這個示例展示如何使用 GOOS 優化一個功率分束器，
將輸入光分成兩個輸出，並達到目標功率分佈。

優化完成後，可以使用 extract_goos_design.py 提取設計網格。
"""

import numpy as np
from spins import goos
from spins.goos_sim import maxwell


def optimize_power_splitter(
    save_folder="power_splitter_goos",
    target_power_ratio=0.5,
    design_size=(2000, 2000),  # nm
    pixel_size=40,  # nm
    max_iters=50
):
    """
    優化一個功率分束器
    
    Args:
        save_folder: 保存文件夾
        target_power_ratio: 目標功率比（輸出1的功率 / 總功率）
        design_size: 設計區域大小 (width, height) in nm
        pixel_size: 像素大小 in nm
        max_iters: 最大迭代次數
    
    Returns:
        design_var: 設計變數節點
        eps_rendered: 渲染後的介電常數節點
    """
    goos.util.setup_logging(save_folder)
    
    design_width, design_height = design_size
    thickness = 220  # nm
    
    with goos.OptimizationPlan(save_path=save_folder) as plan:
        # 創建設計變數
        def initializer(size):
            np.random.seed(42)
            return np.random.random(size) * 0.2 + 0.5
        
        var, design = goos.pixelated_cont_shape(
            initializer=initializer,
            pos=goos.Constant([0, 0, 0]),
            extents=[design_width, design_height, thickness],
            material=goos.material.Material(index=1.0),  # 空氣
            material2=goos.material.Material(index=3.45),  # 矽
            pixel_size=[pixel_size, pixel_size, thickness],
            var_name="design_var"
        )
        
        # 輸入波導
        wg_in = goos.Cuboid(
            pos=goos.Constant([-2000, 0, 0]),
            extents=goos.Constant([3000, 400, thickness]),
            material=goos.material.Material(index=3.45)
        )
        
        # 輸出波導 1（上方）
        wg_out1 = goos.Cuboid(
            pos=goos.Constant([0, 2000, 0]),
            extents=goos.Constant([400, 3000, thickness]),
            material=goos.material.Material(index=3.45)
        )
        
        # 輸出波導 2（下方）
        wg_out2 = goos.Cuboid(
            pos=goos.Constant([0, -2000, 0]),
            extents=goos.Constant([400, 3000, thickness]),
            material=goos.material.Material(index=3.45)
        )
        
        # 完整結構
        eps = goos.GroupShape([wg_in, wg_out1, wg_out2, design])
        
        # 創建渲染節點（用於可視化和提取）
        eps_rendered = maxwell.RenderShape(
            design,
            region=goos.Box3d(
                center=[0, 0, 0],
                extents=[design_width, design_height, 0]
            ),
            mesh=maxwell.UniformMesh(dx=pixel_size),
            wavelength=1550,
            name="eps_rendered"
        )
        
        # 設置仿真
        sim = maxwell.fdfd_simulation(
            name="sim",
            wavelength=1550,
            eps=eps,
            solver=maxwell.DirectSolver(),
            sources=[
                maxwell.WaveguideModeSource(
                    center=[-1400, 0, 0],
                    extents=[0, 2500, 1000],
                    normal=[1, 0, 0],
                    mode_num=0,
                    power=1
                )
            ],
            simulation_space=maxwell.SimulationSpace(
                mesh=maxwell.UniformMesh(dx=pixel_size),
                sim_region=goos.Box3d(
                    center=[0, 0, 0],
                    extents=[4000, 4000, 40]
                ),
                pml_thickness=[400, 400, 400, 400, 0, 0]
            ),
            background=goos.material.Material(index=1.0),
            outputs=[
                maxwell.Epsilon(name="eps"),
                maxwell.ElectricField(name="field"),
                # 輸出 1 的重疊（上方）
                maxwell.WaveguideModeOverlap(
                    name="overlap1",
                    center=[0, 1400, 0],
                    extents=[2500, 0, 1000],
                    normal=[0, 1, 0],
                    mode_num=0,
                    power=1
                ),
                # 輸出 2 的重疊（下方）
                maxwell.WaveguideModeOverlap(
                    name="overlap2",
                    center=[0, -1400, 0],
                    extents=[2500, 0, 1000],
                    normal=[0, -1, 0],
                    mode_num=0,
                    power=1
                )
            ]
        )
        
        # 計算功率
        power1 = goos.abs(sim["overlap1"])
        power2 = goos.abs(sim["overlap2"])
        total_power = power1 + power2
        
        # 計算功率比
        power_ratio = power1 / (total_power + 1e-10)  # 避免除零
        
        # 目標函數
        target_ratio = goos.Constant(target_power_ratio)
        ratio_error = (power_ratio - target_ratio)**2
        power_loss = 1 - total_power
        
        # 組合目標：最小化功率比誤差和功率損耗
        obj = goos.rename(
            ratio_error + 0.1 * power_loss,
            name="objective"
        )
        
        # 運行優化
        print(f"開始優化功率分束器...")
        print(f"目標功率比: {target_power_ratio}")
        print(f"設計區域: {design_width} x {design_height} nm")
        print(f"像素大小: {pixel_size} nm")
        print(f"最大迭代次數: {max_iters}")
        
        goos.opt.scipy_minimize(
            obj,
            "L-BFGS-B",
            monitor_list=[sim["eps"], sim["field"], power1, power2, obj],
            max_iters=max_iters,
            name="optimization"
        )
        
        plan.save()
        plan.run()
        
        # 提取最終結果
        final_design = plan.get_var_value(var)
        final_power1 = power1.get().array
        final_power2 = power2.get().array
        final_ratio = final_power1 / (final_power1 + final_power2 + 1e-10)
        
        print("\n" + "=" * 60)
        print("優化完成！")
        print("=" * 60)
        print(f"最終功率 1: {final_power1:.6f}")
        print(f"最終功率 2: {final_power2:.6f}")
        print(f"總功率: {final_power1 + final_power2:.6f}")
        print(f"實際功率比: {final_ratio:.6f}")
        print(f"目標功率比: {target_power_ratio:.6f}")
        print(f"功率比誤差: {abs(final_ratio - target_power_ratio):.6f}")
        print(f"\n設計已保存到: {save_folder}")
        print(f"使用以下命令提取設計網格:")
        print(f"  python extract_goos_design.py {save_folder} --visualize")
        
        return var, eps_rendered


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="使用 GOOS 優化功率分束器"
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        default="power_splitter_goos",
        help="保存文件夾"
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=0.5,
        help="目標功率比（輸出1的功率 / 總功率）"
    )
    parser.add_argument(
        "--design-width",
        type=int,
        default=2000,
        help="設計區域寬度 (nm)"
    )
    parser.add_argument(
        "--design-height",
        type=int,
        default=2000,
        help="設計區域高度 (nm)"
    )
    parser.add_argument(
        "--pixel-size",
        type=int,
        default=40,
        help="像素大小 (nm)"
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=50,
        help="最大迭代次數"
    )
    
    args = parser.parse_args()
    
    optimize_power_splitter(
        save_folder=args.save_folder,
        target_power_ratio=args.target_ratio,
        design_size=(args.design_width, args.design_height),
        pixel_size=args.pixel_size,
        max_iters=args.max_iters
    )

