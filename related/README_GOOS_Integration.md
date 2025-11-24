# GOOS 與 RL 環境整合指南

## 快速開始

### 1. 使用 GOOS 優化功率分束器

```bash
# 運行 GOOS 優化
python example_goos_power_splitter.py --save-folder power_splitter_goos --target-ratio 0.5

# 從優化結果提取設計網格
python extract_goos_design.py power_splitter_goos --visualize
```

### 2. 提取的設計網格文件

提取腳本會生成以下文件：
- `extracted_design_grid.npy`: 二進制網格（供 MEEP 使用）
- `extracted_design_grid_continuous.npy`: 連續設計參數
- `extracted_design_grid_epsilon.npy`: 介電常數網格（如果可用）
- `extracted_design_grid_visualization.png`: 可視化圖像

### 3. 在 MEEP 環境中使用設計網格

```python
import numpy as np
from meep_env import MeepSimulation

# 加載 GOOS 優化的設計網格
design_grid = np.load("power_splitter_goos/extracted_design_grid.npy")

# 創建 MEEP 仿真
sim = MeepSimulation()

# 將設計網格轉換為 MEEP 幾何
# 注意：需要根據實際的網格尺寸調整
sim.pattern = design_grid
sim.set_geometry()
sim.set_sources()
sim.set_simulation()
sim.set_flux_monitors()
sim.run_simulation(until=200)

# 獲取功率分佈
power_dist, y_pos = sim.get_power_distribution()
```

## 文件說明

### 文檔
- `GOOS_Guide.md`: 完整的 GOOS 使用指南，包括：
  - GOOS 工作流程
  - API 接口說明
  - 如何提取設計網格
  - 應用到 RL 環境的經驗總結

### 示例腳本
- `example_goos_power_splitter.py`: GOOS 功率分束器優化示例
- `extract_goos_design.py`: 從 GOOS 結果提取設計網格的工具

### 現有文件
- `meep_env.py`: MEEP 仿真環境（需要與設計網格整合）
- `black_box_test.py`: MEEP 測試腳本

## GOOS 工作流程總結

```
1. 定義設計變數 (Variable)
   ↓
2. 創建形狀 (Shape) - 使用 pixelated_cont_shape 或 cubic_param_shape
   ↓
3. 組合完整結構 (GroupShape)
   ↓
4. 設置仿真 (fdfd_simulation)
   ↓
5. 定義目標函數 (objective)
   ↓
6. 運行優化 (scipy_minimize)
   ↓
7. 提取設計網格 (get_var_value 或 RenderShape)
```

## 關鍵概念對應

| GOOS | RL 環境 | 說明 |
|------|---------|------|
| Variable | Observation/Action | 設計參數 |
| Shape | Design Grid | 結構表示 |
| fdfd_simulation | MeepSimulation | 仿真器 |
| objective | reward | 優化目標 |
| scipy_minimize | RL Algorithm | 優化方法 |

## 設計網格格式轉換

### GOOS → MEEP

```python
# GOOS 設計參數（連續，[0, 1]）
design_params = plan.get_var_value(design_var)

# 轉換為二進制網格
threshold = 0.5
binary_grid = (design_params > threshold).astype(int)

# 轉換為 MEEP 格式（2D）
if len(binary_grid.shape) == 3:
    meep_grid = binary_grid[:, :, binary_grid.shape[2]//2]
else:
    meep_grid = binary_grid
```

### MEEP → GOOS（如果需要）

```python
# MEEP 二進制網格
meep_grid = np.load("meep_design.npy")

# 轉換為連續參數（可選，用於初始化）
continuous_params = meep_grid.astype(float) + np.random.random(meep_grid.shape) * 0.1

# 在 GOOS 中使用
var, design = goos.pixelated_cont_shape(
    initializer=lambda size: continuous_params.flatten()[:np.prod(size)].reshape(size),
    ...
)
```

## 常見問題

### Q: 如何確定設計變數的名稱？

A: 在優化腳本中，`var_name` 參數指定了變數名稱。默認為 `"design_var"`。如果不知道名稱，可以：

```python
with goos.OptimizationPlan() as plan:
    plan.load("save_folder")
    print("可用的節點:", list(plan._node_map.keys()))
```

### Q: 設計網格的尺寸不匹配怎麼辦？

A: GOOS 和 MEEP 可能使用不同的網格尺寸。需要：

1. 檢查兩者的像素大小和設計區域大小
2. 使用插值調整尺寸：
```python
from scipy.ndimage import zoom
resized_grid = zoom(original_grid, (new_h/old_h, new_w/old_w), order=1)
```

### Q: 如何比較 GOOS 和 RL 的結果？

A: 建議：

1. 使用相同的目標函數/獎勵函數
2. 使用相同的設計區域和像素大小
3. 在相同的仿真條件下評估（波長、邊界條件等）
4. 比較功率分佈和效率

## 下一步

1. **整合到 RL 環境**：
   - 修改 `meep_env.py` 以支持加載 GOOS 設計網格
   - 實現設計網格到 MEEP 幾何的轉換

2. **基準測試**：
   - 使用 GOOS 優化結果作為 RL 的初始化
   - 比較兩種方法的性能

3. **可視化工具**：
   - 創建工具來並排比較 GOOS 和 RL 的設計
   - 比較功率分佈和場分佈

## 參考資源

- GOOS 文檔：`docs/goos/`
- GOOS 示例：`examples/goos/`
- MEEP 文檔：https://meep.readthedocs.io/
- SPINS 源碼：`spins/goos/`

