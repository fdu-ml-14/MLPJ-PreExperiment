# Results 目录说明

- `Result.md`：预实验汇总、解释与限制。
- `episodes/extended_models_episodes.csv`：三模型 160 条逐 episode 记录（π0-Bridge 80 + SpatialVLA 40 + Octo 40）。
- `episodes/extended_models_results.csv`：三模型 16 个“模型 × 任务 × action_step”分组汇总，含 Wilson 95% CI。
- `episodes/pi0_base_episodes.csv`：通用 π0 80 条逐 episode 记录。
- `episodes/pi0_base_results.csv`：通用 π0 8 个分组汇总，含 Wilson 95% CI。
- `episodes/action_step_T1_T4_episodes.csv`：早期 T1–T4 × action_step 的 8 条单集冒烟记录。
- `episodes/action_step_T1_T4_results.csv`：早期 8 条记录的汇总。
- `episodes/representatives_extended.json`：三模型代表视频的选择规则与 episode 编号。
- `episodes/representatives_pi0_base.json`：通用 π0 代表视频的选择规则（仓库只收录其中 4 个 T1/T2 视频）。
- `videos/`：20 个代表性 MP4，文件名格式为 `<model>_<task>_a<action_step>_ep<episode>_<timestamp>.mp4`。

## 代表性视频

| 文件 | 模型 | 任务 | action_step | episode |
|---|---|---|---:|---:|
| `pi0_bridge_widowx_spoon_on_towel_a1_ep2_20261023_130722.mp4` | π0-Bridge | T1 | 1 | 2 |
| `pi0_bridge_widowx_spoon_on_towel_a4_ep0_20261023_130840.mp4` | π0-Bridge | T1 | 4 | 0 |
| `pi0_bridge_widowx_put_eggplant_in_basket_a1_ep5_20261023_130947.mp4` | π0-Bridge | T2 | 1 | 5 |
| `pi0_bridge_widowx_put_eggplant_in_basket_a4_ep0_20261023_131104.mp4` | π0-Bridge | T2 | 4 | 0 |
| `pi0_bridge_widowx_carrot_on_plate_a1_ep0_20261023_131211.mp4` | π0-Bridge | T3 | 1 | 0 |
| `pi0_bridge_widowx_carrot_on_plate_a4_ep2_20261023_131319.mp4` | π0-Bridge | T3 | 4 | 2 |
| `pi0_bridge_widowx_stack_cube_a1_ep3_20261023_131430.mp4` | π0-Bridge | T4 | 1 | 3 |
| `pi0_bridge_widowx_stack_cube_a4_ep1_20261023_131547.mp4` | π0-Bridge | T4 | 4 | 1 |
| `spatialvla_bridge_widowx_spoon_on_towel_a1_ep0_20261023_131605.mp4` | SpatialVLA | T1 | 1 | 0 |
| `spatialvla_bridge_widowx_spoon_on_towel_a4_ep0_20261023_131640.mp4` | SpatialVLA | T1 | 4 | 0 |
| `spatialvla_bridge_widowx_put_eggplant_in_basket_a1_ep0_20261023_131702.mp4` | SpatialVLA | T2 | 1 | 0 |
| `spatialvla_bridge_widowx_put_eggplant_in_basket_a4_ep0_20261023_131728.mp4` | SpatialVLA | T2 | 4 | 0 |
| `octo_base_widowx_spoon_on_towel_a1_ep0_20261023_131750.mp4` | Octo-Base | T1 | 1 | 0 |
| `octo_base_widowx_spoon_on_towel_a4_ep5_20261023_131937.mp4` | Octo-Base | T1 | 4 | 5 |
| `octo_base_widowx_put_eggplant_in_basket_a1_ep0_20261023_132032.mp4` | Octo-Base | T2 | 1 | 0 |
| `octo_base_widowx_put_eggplant_in_basket_a4_ep0_20261023_132211.mp4` | Octo-Base | T2 | 4 | 0 |
| `pi0_base_widowx_spoon_on_towel_a1_ep0_20261023_150444.mp4` | 通用 π0 | T1 | 1 | 0 |
| `pi0_base_widowx_spoon_on_towel_a4_ep0_20261023_150716.mp4` | 通用 π0 | T1 | 4 | 0 |
| `pi0_base_widowx_put_eggplant_in_basket_a1_ep0_20261023_151503.mp4` | 通用 π0 | T2 | 1 | 0 |
| `pi0_base_widowx_put_eggplant_in_basket_a4_ep0_20261023_151725.mp4` | 通用 π0 | T2 | 4 | 0 |
