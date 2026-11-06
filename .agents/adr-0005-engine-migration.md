# ADR-0005: 引擎迁移决策（Godot 4.7 → Bevy 0.20 + bevy_elements）

状态：接受

日期：2026-10-19（新增 ADR，记录本次引擎迁移的决策与映射）

## 上下文

原项目用 Godot 4.7 独立开发，加入 AI Agent 后进度大幅提升，但 Godot 的测试效率较低（引擎内手动验证为主，Agent 闭环困难）。已有游戏迁移到 Rust/Bevy 后 Agent 测试体验明显更好（类型系统、cargo test 闭环、headless 渲染）。本游戏需要写实的天空大气效果，需确认 Bevy 能力后再迁移。

## 决策

### 1. 迁移方案：A（Bevy 0.20 + bevy_elements）

三个候选：

- **A. Bevy 0.20 + bevy_elements（接受）**：官方内置物理基大气（0.16→0.20 打磨四个大版本：0.16 散射、0.18 遮蔽+PBR 整合+通用散射介质、0.20 Solari 光照支持 Atmosphere）；bevy_elements 0.5.3 是截至 2026-10-19 唯一已确认兼容 0.20 的完整天气/云/雾 crate，功能最全（体积云+云影+大气联动配色）。0.20 的 `Ready` 事件、调度随机化（property testing）、headless 渲染对 Agent 自动化测试友好。
- **B. Bevy 0.19 稳妥路线（否决）**：`bevy_weather`/`bevy_fast_clouds` 可用，但 0.19→0.20 迁移迟早要做，二次升级成本高；0.19/0.20 在大气/云上无实质差距。
- **C. 混合方案 / Godot 保留（否决）**：核心逻辑共享 crate、渲染绑定单引擎，天空效果仍被单引擎锁死，长期收益不如整体迁移。

### 2. 碰撞实现：自研运动学控制器（接受）

- **A. 自研运动学控制器 + 半平面裁剪数学（接受）**：玩家无跳跃无跑步、只在负形街道中行走，碰撞测试是"点在凸棱柱内"的纯 2D/2.5D 数学，CPU 开销极小；原 `inset_polygon_clip` 数学可移植。确定性、零外部依赖。
- **B. avian3d（否决）**：完整物理引擎对本游戏（无跳跃/无投掷/无动态刚体）是过度引入，且引入了非确定性物理步进风险。
- **C. bevy-tnua（否决）**：面向浮空角色控制器，超出散步玩法需求。

### 3. Godot → Bevy 映射表（工程层面的关键决策）

| 原实现 | 新实现 | 说明 |
|--------|--------|------|
| GDScript 脚本 | Rust systems（`SystemParam` 分模块） | 模块划分沿用原 Manager 划分：WorldManager → world 系统集，SeasonManager → season 模块，SaveManager → save Resource |
| signal | Bevy Message/Observer | `player_chunk_changed` → Message；`world_exploration_completed` → Message |
| autoload 单例 | `Resource` + Plugin | SaveManager、DuskSky 均为 Resource |
| `global_shader_uniform` | 共享 storage buffer | 见 ADR-0004 第 6 节 |
| CanvasLayer 代码构建 UI | Bevy UI（em/rem） | 菜单/HUD 用 Bevy UI 代码构建，沿用原 M4 结构 |
| StaticBody3D + WorldBoundaryShape3D | 移除 | 负形街道模型已取代边界长墙 |
| CharacterBody3D 玩家控制器 | 自研运动学控制器 | 视角钳制（±10°/22.55–60°）、速度 1.2–2.0 m/s 可调 |
| .blend/.tscn 资产 | glTF（.glb） | 树与建筑模型统一 glTF 导出链路 |
| 调试可视化（F 键 tilemap） | Bevy Gizmos | Block 分布/合并规模/碰撞范围可视化保留 |
| JSON 存档（user://saves） | serde JSON（本地存档目录） | 结构不变：seed+size+season+explored+player_pose |

### 4. 后期效果栈映射

| 效果 | Bevy 0.20 实现 |
|------|----------------|
| Bloom | 内置 `Bloom::NATURAL` |
| 色调映射 | `Tonemapping::AcesFitted` |
| 景深 DoF | 内置 DepthOfField |
| 色差 CA | 无内置 → 0.20 新 `FullscreenMaterial` trait 自写全屏 shader（工作量约 1 天，保留为 ADR-0003 方案 B 的子项） |

### 5. 版本与生态策略

- 精确锁定 `bevy = "=0.20.1"`（补丁位，2026-11-05 由 0.20.0 升级）、`bevy_elements = "0.5.3"`（详见根 AGENTS.md）
- Bevy 0.20 淘汰自研 WGSL 方言、改用 WESL：本项目自写 shader 一律 `.wesl`；不使用带 `#import` 的旧 `.wgsl` 教程代码
- 引擎升级只作为独立里程碑：先核对 bevy_elements 是否跟进新版 Bevy，再做迁移

## 备选方案

- **留在 Godot 仅优化测试**（如 gdUnit/CI headless）：未解决写实大气（Godot 仅 HDRI 天空盒 + 有限散射）与 Agent 闭环的结构性问题——否决
- **其他引擎（Unreal/Unity）**：放弃 Rust 生态与 Agent 工作流优势——否决

## 后果

- 需要重写全部 gameplay 代码（GDScript → Rust），设计决策通过 ADR/agents 文档继承，算法逻辑移植
- 资产链路重建：.blend → glTF、.tscn → BSN/glTF 场景、.gdshader → WESL
- 树的顶点色摆动接口必须在 WESL 重写中保留（艺术资产可复用）
- 建筑模块（Blender 制作）仍是 Feihei 负责的外部依赖，R3 先以 BoxMesh 占位
- Agent 测试闭环升级为：cargo test（确定性/几何/存档）+ headless 截图冒烟 + 调度随机化 property testing + 引擎内人工验证（沿用原"待用户验证"清单）
- 声景系统（原 M5 未完成）在新引擎从零实现，任务顺延至 R5
