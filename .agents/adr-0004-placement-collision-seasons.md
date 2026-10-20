# ADR-0004: 放置、碰撞与季节系统（Bevy 迁移版）

状态：接受

日期：2026-10-19（自原 Godot 版 ADR-0004 2026-07-11（2026-08-26 修订）迁移改写；Block 优先生成、负形街道碰撞、建筑参数表、季节系统全部保留，实现层做 Bevy 映射）

> 本 ADR 保留原 ADR-0004 的全部算法决策，包括 grill 决策推翻薄墙方案的最终形态（负形街道碰撞模型）。原算法历史（网格扰动→边→识别 Block）不再重复，只描述最终形态。

## 上下文

M3 阶段的架构决策在迁移时原样保留：Block 尺寸差异化、建筑沿 edge 放置、负形街道碰撞、季节系统。实现语言改为 Rust/Bevy。

## 决策

### 1. Block 优先生成算法（不变）

**算法**：网格 → 生成 Block → 剩余 cell 自然成街道（**Block 为正形，街道为负形**）

#### Block 合并规模

以基础网格单元（50×50m）为最小单位，按 BlockType 决定合并规模：

| BlockType | 合并规模 N×N | 合并后尺寸 | 数量估算 |
|-----------|------------|----------|-------------------|
| 老城区 OLD_CITY | 1×1 | ~50×50m | 多数 |
| 公寓 APARTMENT | 2×2 或 3×3（随机） | ~100-150m | 少量 |
| 厂房 FACTORY | 4×4 或 5×5（随机） | ~200-250m | 极少 |
| 广场 PLAZA | 1×1 | ~50×50m | 少量 |
| 树林 FOREST | 1×1（边缘过渡带） | ~50×50m | 边缘 |

#### 内部道路删除

合并后的 Block 内部网格边被物理删除（不再作为道路），玩家无法进入大街区内部。

### 2. WorldGrid 2D 中间数据结构（Rust 实现）

```
seed → perturbed_grid()
     → merge_blocks()        # 生成 Block
     → WorldGrid（2D 网格，记录每个 cell 的 block_id）
     → LayoutGraph（从 WorldGrid 提取 edges/blocks）
```

**职责**：
- 记录每个 cell 的归属 Block
- 标记 cell 是否为道路、路口
- 提供 `get_cell(row, col)` 快速查询
- 支持存档序列化（serde，比多边形顶点紧凑）
- 调试可视化直观（Bevy Gizmos 绘制 2D tilemap，替代原 Godot tilemap 调试视图）

### 3. 碰撞策略：不可见实心街区碰撞体（负形街道模型）——最终形态，不变

> 保留原 grill 决策：薄墙方案在街区合并为 N×N 大地块时，转角与路口处墙体交叉/错位，且街宽随机后墙体无法保证缝隙连续，故被推翻。

**废弃**（迁移时不再实现）：
- 每个建筑/树带独立碰撞体（Draw Call 高、可穿树）
- 沿 Block 多边形每条边生成 0.2m 厚碰撞墙
- 4 条边界长墙（由外侧街区自动围合取代）

**新策略（负形街道模型，原样保留）**：
- 街区本身是**实心凸多边形碰撞实体**（凸棱柱，底 y=0、顶 y=BLOCK_SOLID_HEIGHT=2.0m），玩家被挡在街区之间
- 街区之间的缝隙 = 完整街道宽度（负形），可步行
- 碰撞体**不可见**：无可见网格，视觉完全由建筑/树木/地面提供
- 街道宽度随机：每街区 `STREET_WIDTH_MIN=6`/`MAX=10`、inset=width/2，顶点带 jitter（街道宽度有自然变化）

#### 碰撞规则（按 BlockType + is_outer）

| BlockType | is_outer | 是否有碰撞 | 玩家可进入 |
|-----------|----------|-----------|-----------|
| 老城区 OLD_CITY | 任意 | 是 | 否 |
| 公寓 APARTMENT | 任意 | 是 | 否 |
| 厂房 FACTORY | 任意 | 是 | 否 |
| 广场 PLAZA | 任意 | 否 | 是（开放空间） |
| 树林 FOREST | 否（内部） | 否 | 是（开放绿地，可穿树丛） |
| 树林 FOREST | 是（边缘） | 是 | 否（强制带碰撞，形成围墙） |

#### 边界阻挡 / 玩家围合

- 地图最外侧 1 圈 cell（`OUTER_BAND_CELLS=1`）= 外侧街区（OLD_CITY/APARTMENT 各 50%），全为建筑碰撞实体，形成环绕围墙
- 玩家**无法触碰地图边缘、也看不到空边缘视野**
- 不生成"4 条边界长墙"，由外侧街区自动围合

#### 重叠保证（几何层面）

- 路网先生成（扰动网格，非真实 Voronoi），街区边界由路网内缩得到；相邻街区共享同一中心线
- 内缩用自写半平面裁剪（`inset_polygon_clip`，Rust 移植），数学上保证相邻街区实体**不交叠**（原验证 OVERLAPS=0 必须保留为 Rust 单元测试）
- 验证测试：ADJ_PAIRS / OVERLAPS=0 / MAX_OVERLAP 断言

### 4. 建筑放置：沿 edge 统一算法（不变）

#### 建筑参数表

| 类型 | 沿 edge 长度 | 垂直 edge 深度 | 高度 | 间距 | setback |
|------|-----------|------------|------|------|---------|
| 老城区 | 4-6m（方形） | 4-6m | 3-6m | 0.1m | 1m |
| 公寓 | 25-40m | 12-14m | 18m | - | 5m |
| 厂房 | 50m | 50m | 24m | - | 10m |

#### 去重策略

- 建筑放置前检查 `edge 中点` 是否在当前 Chunk 内（原 `_chunk_edge_index` 跨 Chunk 重复放置教训）
- 树木放置同样用 edge 中点或 block 中心去重

#### 朝向

建筑正面朝向街道：`rotation.y = atan2(edge_dir.x, edge_dir.y)`

#### 建筑模型

- R3 初期以 BoxMesh（Bevy 内置 `Mesh3d` 长方体）占位，真实 Blender 模块待 Feihei 制作后导出 glTF 替换
- 程序化做旧材质：`StandardMaterial` 扩展（脏污、褪色、风化）

### 5. 树木放置（不变）

#### 树资产

5 种乔木（aspen_1 / tree_2 / tree_3 / tree_3_1 榆树复叶 / tree_3_3 羽状复叶），从 .blend 重新导出为 glTF（吸取 Godot 版 .tscn 嵌入 >17MB 的教训）。

#### 懒加载

首次需要放树时加载（Bevy AssetServer 惰性加载），避免启动卡顿。

#### 放置规则

- **建筑 Block 边缘（setback 内）**：行道树，按 BlockType 差异化概率
  - 老城区：20% 概率（setback 1m 窄，偶尔放）
  - 公寓：60% 概率（setback 5m 宽）
  - 厂房：80% 概率（setback 10m 最宽）
- **FOREST Block 内部**：高密度放树（约 8m 间距），玩家被墙挡住远观剪影
- **PLAZA Block 内部**：少量放树（玩家可进入近看）

**不带碰撞体**：玩家可穿行道树（视觉上树在街道边缘，碰撞实体在更外侧）。

### 6. 季节系统（参数表不变，实现映射为共享 uniform）

#### 季节参数

| 季节 | leaf_density | leaf_color | sway_strength |
|------|------------|----------|---------------|
| 春 SPRING | 0.7 | Color(0.6, 0.8, 0.3) | 0.03 |
| 夏 SUMMER | 1.0 | Color(0.2, 0.5, 0.2) | 0.03 |
| 秋 AUTUMN | 0.6 | Color(0.8, 0.5, 0.1) | 0.04 |
| 冬 WINTER | 0.0 | Color(0.5, 0.45, 0.4) | 0.05 |

#### 实现：共享 uniform buffer（替代 Godot global_shader_uniform）

- 【迁移改写】原 Godot `global_shader_uniform` + `RenderingServer.set_shader_parameter` 对应 Bevy 的**共享 uniform/storage buffer**：`SeasonState` 为 `Resource`（含 Season 枚举 + 三参数），一个共享 storage buffer 承载 3 个全局参数（leaf_density / leaf_color / sway_strength），树材质的 WESL 顶点/片段着色器直接读取
- **禁止**每帧遍历树材质实例更新参数（沿用原设计：季节切换零开销）
- 冬季 leaf_density=0 → 树叶透明，树枝骨架可见
- 树的 `MaterialExtension` WESL 着色器保留原艺术接口：顶点色 R 通道控制每顶点摆动强度，`sway_speed` / `sway_strength` / `sway_phase_len` uniform

#### 季节切换

- 开发期快捷键（1/2/3/4 切换春夏秋冬）便于验证
- 菜单系统就绪后由菜单选择替代（ADR-0001：开局时指定，单局不变）

## 备选方案

- **保留薄墙碰撞**：转角/路口交叉错位，已否决（grill 决策，沿用）
- **厂房特殊路径（Block 遍历）**：算法分两条路径，不优雅，与"统一沿 edge 放置"冲突——已否决（沿用）
- **AABB 冲突避让**：Block 合并方案自然解决溢出——已否决（沿用）
- **物理引擎做碰撞（avian3d/bevy-tnua）**：见 ADR-0005 决策

## 后果

### 需要实现（Rust）

- `world_grid.rs` — 2D 网格中间数据结构（serde）
- `layout_graph.rs` / `path_network.rs` — Block 优先生成（移植原 GDScript 逻辑）
- `placement.rs` — 沿 edge 放置 + 去重 + 朝向 + 差异化参数表
- `collision.rs` — 负形街道模型（凸棱柱 + 半平面内缩裁剪）
- `season.rs` — SeasonState Resource + 共享 storage buffer
- 树/建筑 `MaterialExtension` WESL 着色器（顶点色摆动接口 + 全局季节参数）

### 性能影响

- 碰撞实体数量 = 每 Block 一个不可见凸棱柱（几何体不渲染，无 Draw Call；玩家控制器的碰撞测试为纯数学，CPU 开销极小）
- 树用真实 glTF 实例，Draw Call 略升，但视觉质量大幅提升
- 共享 storage buffer 无需遍历材质，季节切换零开销

### 对后续阶段的影响

- R4 存档：WorldGrid 可直接 serde 序列化，比多边形顶点紧凑；季节仅需存枚举值
- R5 润色：碰撞策略已优化性能，LOD 可专注于建筑模型本身
