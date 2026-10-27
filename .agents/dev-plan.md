# 开发计划（Bevy 迁移版）

状态：草稿

日期：2026-10-19

> 计划以 checkbox 分阶段呈现。原 Godot 版 M0–M4 已完成，其成果以设计决策（ADR）形式继承；M5（声音与润色）未完成，顺延至 R5。每个阶段完成后：build + fmt + clippy + test 全绿，更新本文件状态，小步提交（Conventional Commits）。

---

## 阶段 R0: 项目骨架与依赖锁定（预计 1-2 天）

**目标**：搭建 Bevy 0.20 workspace 骨架，验证版本匹配与工具链。

- [x] 0.1 创建 Rust workspace：`Cargo.toml` 精确锁定 `bevy = "=0.20.0"`、`bevy_elements = "0.5.3"`
- [x] 0.2 `cargo check` 验证 bevy_elements 0.5.3 的 Bevy 依赖确为 0.20；不匹配则回报并回退评估（ADR-0005 方案 B）
- [x] 0.3 阅读并记录 bevy_elements 的 Cloud/Celestial/Fog/Climate 模块源码要点（无人工审计 crate，先读再信）
- [x] 0.4 项目脚手架：Plugin 结构（SkyPlugin / WorldPlugin / SeasonPlugin / SavePlugin / MenuPlugin）、`Result` 错误处理约定、CI 工作流（fmt/clippy/test 门）
- [x] 0.5 树资产链路验证：aspen_1 等 5 种树从 .blend 导出 glTF，Bevy 加载正常（⚠️ 原项目仅 3 种非备份树 .blend：aspen_1/tree_2/tree_3，已导出并通过 headless 加载测试；第 4/5 种树待 Feihei 提供 .blend）

### 边界方案（2026-10-21 用户决策，偏离原版）

- **最外圈不再是 Block**：原版 `OUTER_BAND_CELLS=1` 把最外一圈按街区生成（含 is_outer 建筑、碰撞）。改为：最外一圈 cell 不分配 Block（`SCENERY_BAND_CELL`），只放景物（建筑背面/树），由同一个 Chunk 系统流式加载；真正挡人的是一圈实心碰撞环。
- **理由**：外带仅 50m 深，扣掉半街宽剩 46m，无法吸收内部街区扰动；直线边界与扰动顶点必然相撞。改为景物带后环路可沿格点走直线。
- **几何**：环路中心线在内部第一圈格点上（`x=50`），路辐 4m → 可走 `x=46..54`；景物带 `x=0..46`，靠环路 12m 布置剪影。
- **待办**：R3 需补景物带内容（建筑背面/树）与实心碰撞环；`is_outer` 相关放置规则随之调整。

### 可验证

- `cargo run` 空窗口启动，无 warning；fmt/clippy/test 全绿
- glTF 树在场景中正确显示

---

## 阶段 R1: 天空系统迁移（预计 2-3 天）

**目标**：用 Bevy 0.20 Atmosphere + bevy_elements 复现原 M1 的黄昏氛围。

- [x] 1.1 集成 bevy_elements：`ElementsCamera` + Atmosphere/EnvironmentMapLight/DepthPrepass/VolumetricFog/Exposure 自动挂接
- [x] 1.2 编写 `HeadsUpSky` Resource：时间锁定 17:30（`WorldClock` time_scale=0 + paused）、天气锁定晴朗（`WeatherDirector::force(WeatherPreset::Clear, Duration::ZERO)`）、黄昏参数接口（cloud_cover/cloud_layer 覆盖 + 曝光基准；星星可见度为保留接口——星场内置于 CelestialPlugin 无公开 API，黄金时刻被亮天空掩盖属预期）
- [x] 1.3 配置黄昏色调：物理散射 + `Exposure { ev100: 13.0 }` + ACES + Bloom（intensity 0.3）；选定 date=2026-03-21 + lon=125.0 → 17:30 太阳 3.5°/月亮 36°；实测：太阳盘+光晕+暖地平线带+云剪影暖镶边+冷蓝顶渐变（橙红饱和度/天顶紫调待 Feihei 人眼迭代，估 60%）
- [x] 1.4 裁剪 bevy_elements 未用模块（Wind/Precipitation/Lightning/Dust/Terrain/Hydrology/Water/GroundCover/Weathered 共 9 个 disable，保留 Climate/Celestial/Fog/Cloud）
- [x] 1.5 天空动态变化：世界坐标噪声（sum-of-sines，确定性）驱动 exposure 微扰 ±0.15 ev100；ScatteringMedium 密度乘子受 GPU LUT 资产限制（每帧重建不可行），记录在笔记中
- [x] 1.6 后期效果：Bloom/ACES 内置配置；CA 用 `FullscreenMaterial` 自写（延后至 R5 润色）
- [x] 1.7 headless 截图冒烟测试：`tests/sky_screenshot.rs` 黄昏天空帧断言（HDR 带分析：渐变/暖地平线/非黑；本地 GPU `--ignored` 运行，CI GPU-less runner 不能跑）

### 可验证

- 抬头看到动态黄昏天空（物理散射，非 HDRI）
- 天空中有云、星星、月亮
- 黄昏色调自然，后期效果好看；同一 seed 场景截图逐帧一致

---

## 阶段 R2: 布局图与 Chunk 系统（预计 3-5 天）

**目标**：移植原 M2/M3 生成算法到 Rust，实现 Chunk 流式加载。

> ⚠️ **状态修正（2026-10-21）**：2.1–2.8 交付的是"规整网格基线"，**未复刻原 Godot 的扰动路网**。已在原项目源码 `E:/play/Godot/heads_up/scripts/` 中逐项核对，差异列入下方 2.9–2.15。这些项完成前，R2 不算完整移植。

- [x] 2.1 `layout_graph.rs`：LayoutGraph 数据结构（节点/边/宽度、Block 标签、边界，serde）；plaza 块中心作为图节点（放置锚点）
- [x] 2.2 `world_grid.rs`：2D 网格中间结构（cell 记录 block_id/is_road/is_junction，`get_cell` 快速查询，serde）；is_road=贴街 cell，is_junction=含图节点的 cell
- [x] 2.3 `path_network.rs`：Block 优先生成。**修订（2026-10-21）：改为完整矩形剖分**——行序扫描取「区域内最大自由方块」，装不进目标类型的最小尺寸就降级为 1×1 老城区。原 carve+回退会留下 L 形残余块，其凹角在扰动+内缩下翻转朝外，实测 280 对穿插 / 最深 20.8m；剖分保证每格恰属一个矩形，无残余即无凹角。类型按面积配额抽样（见 2.12）
- [x] 2.4 种子系统：`rand_chacha::StdRng::seed_from_u64(seed)`，同 seed bit-exact 单元测试；`thread_rng` 不进入生成路径
- [x] 2.5 `chunks.rs`：50m×50m Chunk（16×16=256，与 cell 1:1 对齐），不对称半径（前方 4 / 后方 2），窗口最坏 8×8=64 格流式 spawn/despawn（2026-10-26 地图缩至 0.8km 后仍为 16×16=256 Chunk）（0.20 递归 `despawn` 组删除）+ 每 Chunk 地面 tile（seed 着色）供流式反馈 ⚠️ 原版为「切比雪夫距离 + 前后点积判定」，当前为旋转矩形窗口，略多加载；对齐见 2.15
- [x] 2.6 几何验证单元测试：半平面内缩裁剪（`inset_polygon_clip`）+ SAT；OVERLAPS=0 断言（4 个 seed 全块对）
- [x] 2.7 调试可视化：Bevy Gizmos 绘制 Chunk 边界/街道中心线/Block 轮廓（按类型着色）+ Text2d Block 标签；G 键开关（debug 构建默认开）
- [x] 2.8 系统排序显式声明（`.before()/.after()` 链）+ 调度随机化 property testing（16 种子随机注册顺序 + 歧义检测 = Error）

### 修正项（复刻原 Godot 算法，2.9–2.15）

> 对照 `scripts/path_network_generator.gd`（466 行）/ `object_placer.gd` / `chunk_manager.gd`。按依赖顺序实施，**必须在 R3.1 放置之前完成**（放置沿 Block 边缘排布，边界不扰动则建筑会是整齐直线）。

- [x] 2.9 **扰动网格**：内部格点随机偏移，边界节点两轴均固定（原版只锁单轴，上/左边界仍会滑动，导致外墙off-border）。`PERTURB_MAX` **取 4m**（非原版 12m）：50m 格距下 12m 双向扰动使相邻格点相对偏移达 24m、共享边偏角 53°，按半街宽内缩会产生 5.4m 横向误差 > 街宽 → 实测 280+ 对穿插、最深 20.8m；4m 下偏角约 22°、误差 1.6m，街道仍有可见摆动（格边 34-68m）。**修订（2026-10-24，R3.8 步行测试发现）**：第一圈内部格点（环路中线，row/col 1 与 axis-2）同样钉死——它们被扰动会让环路内侧块越过中线，自由带宽实测低至 0.3m 甚至为负。**另修**：`LayoutGraph` 节点坐标改用扰动后格点（原用基准格点，街道中线与块体内缩错位，同样挤占街道）；`street_unit_edges` 的单位中线与 `associate_edges_with_blocks` 的左右归属随之改按扰动格点/块体几何侧判定
- [x] 2.10 **沿周长构多边形**：按 上→右→下→左 取全部边界节点（合并块 `2*(rows+cols)` 顶点）。**前置条件改为矩形剖分**（见 2.3 修订）：只有块是纯矩形时，周长才是无凹陷的凸四边形；L 形残余块的凹角会让内缩翻转朝外（实测吞掉相邻块 10m）
- [x] 2.11 **内缩**：四边统一 `w/2`，按**顶点角平分线**偏移（`1/cos(θ/2)`，上限 1.5 防尖角外冲），保证扰动下仍严格内缩。原「逐边半平面裁剪 + 斜接」在扰动多边形上会把角落推出中心线，已替换
- [x] 2.12 **类型比例修正**：按**面积**配额而非块数——改前厂房仅占 8% 块数却吃掉 45-50% 面积（ADR-0001 的 50/30/20 指观感面积）。现为 广场 6% / 树林 20% 抽签，建筑类按面积偏置（老城 0.87 / 公寓 0.10 / 厂房 0.03）；实测老城 46-50% / 公寓 21-32% / 厂房 10-21% 面积占比
- [x] 2.13 **Block 元数据补齐**：`min_y/max_y` 层数范围（老城 1-3 / 公寓 4-6 / 厂房 6-10 / 广场·树林 0）、`left_block/right_block` 边->Block 关联（`_associate_edges_with_blocks`）、`world_grid` 挂到 graph（供放置阶段查询）
- [x] 2.14 **测试加固**：新增格点扰动断言（内部 95%+ 移动、边界双轴钉死、同 seed 复现）；OVERLAPS=0 判据从 SAT 深度改为**真实穿插**（顶点包含）——相邻扰动边相距毫米级时 SAT 会假阳性（实测 6-7 对假阳性、真实 0）；边界断言改为「景物带 = 最外圈、内部 cell 必属某块」
- [x] 2.15 **Chunk 窗口对齐**：改为切比雪夫半径 + 前后点积（`dot >= 0` 用 FRONT=4，否则 BACK=2），替换旋转矩形窗口；并补 `CHUNKS_PER_FRAME = 1` 分帧加载/卸载（`pending_load` / `pending_unload` 队列，就近优先），避免瞬移时单帧建 60+ Chunk。窗口随朝向改变（正对 +X 时东侧达 4 格、西侧仅 2 格），单测锁定

### 可验证

- 调试模式可见路径网络与 Block 分布
- 移动时周围物体出现/消失，无明显卡顿
- 同一 seed 每次生成的地图完全一致（bit-exact 测试）
- 几何测试 OVERLAPS=0
- **（2.9 之后）顶视图不再是笔直曼哈顿网格，街区边界有自然错位；`examples/world_map.rs` 可直接出图比对**

### 实现备注

- ~~原始 GDScript 源不可得~~ **更正（2026-10-21）**：原项目源码在 `E:/play/Godot/heads_up/scripts/`，已逐项核对。2.1–2.8 是按 ADR 不变量重实现的规整网格版，2.9–2.15 负责补齐与原版的一致性
- 无死胡同的构造性保证：完整矩形 partition 下内部格点街道度数不可能为 1，band 内部无街道使边界格点度数为 0；扰动不改变拓扑（只移动节点），该保证依然成立
- R3 碰撞可直接用 Block inset 多边形：扰动后为凸四边形（扰动矩形），半平面裁剪与 SAT 逻辑复用；**但前提是 2.10 的周长取点已落地**，否则相邻块共享边不重合，碰撞体会留缝或穿插
- 顶视图工具：`cargo run --example world_map`（CPU 光栅化，4000×4000，2px=1m，不依赖 GPU）

---

## 阶段 R3: 放置、碰撞与季节（预计 3-4 天）

**目标**：移植原 M3 的放置/碰撞/季节决策（负形街道模型最终形态）。

> **前置依赖（硬性）**：3.1 与 3.3 依赖 R2.9–R2.11（扰动网格 + 沿周长构多边形 + 逐边 inset）。建筑沿 Block 边缘排布、碰撞体取 Block 多边形，若 Block 边界仍是未扰动的直线，放置结果与碰撞围合都会偏离原版。**必须先完成 2.9–2.11 再开 R3**。

- [x] 3.1 `placement.rs`：沿 edge 统一放置算法 + 差异化参数表（老城区 4-6m / 公寓 25-40m×12-14m / 厂房 50×50m）+ setback（1/5/10m）+ 朝向（atan2）+ edge 中点去重；层数取 2.13 的 `min_y/max_y`
- [x] 3.2 树木放置：5 种 glTF 懒加载，行道树概率 20/60/80%，FOREST 密 8m，PLAZA 稀疏，树不带碰撞
- [x] 3.3 `collision.rs`：负形街道模型——每个实心 Block 一个不可见凸棱柱（`block.polygon`，y=0..`BLOCK_SOLID_HEIGHT`=2m），按 50m chunk 分桶索引（`CollisionIndex`，与 chunk 流式同步注册/注销）；`resolve(pos, PLAYER_RADIUS)` 沿最浅穿透轴推出，迭代 16 次 × 3×3 邻桶以处理角点连锁。规则：PLAZA 跳过、内部 FOREST 跳过、其余生成（940 块中 804 块实心）。景物带另由逐 chunk 实心环封闭。调试可视化按类型着色画出碰撞轮廓。验证：可走面积 26.2%，从街道朝 721 个实心块内部行走 0 次成功进入

> **与 ADR-0004 的差异**：ADR 描述的是「原版逐 Block 生成 `StaticBody3D` + `ConvexPolygonShape3D`」。本项目按 ADR-0005 不引入物理引擎，改为纯数学碰撞（凸多边形包含 + 最小平移推出），由 R3.4 的运动学控制器消费；语义等价（不可见、实心、不渲染、无 Draw Call）。
- [x] 3.4 自研运动学玩家控制器（`src/player/mod.rs`）：A/D 以 60°/s 转身、鼠标仅纵向控制视角并钳制仰角 22.55–60°（底边掠过地平线；鼠标水平移动不控制旋转/朝向）、W/S 沿朝向行走、E/Q 调速 1.2–2.0（0.1 步进，加速 8 m/s²）、视高 1.65m、无跳跃/跑步/FOV。运动按 ½ 半径细分子步后交 `CollisionIndex::resolve`，高速不穿墙。Esc 释放/重新捕获鼠标。替换 R2 临时调试移动；删除 R1 的 `aim_at_sun_once`（两个系统写同一 Transform 会构成隐式排序）

> **碰撞修复**：合并块内缩后的周长会出现轻微凹陷（实测顶点 3/5/7 的叉积为负），`push_out` 依赖凸性，导致 770 次高速穿墙尝试中有 13 次进入块内。改为对 `convex_hull(polygon)` 做碰撞后降为 0 次；代价是碰撞体比可见轮廓大出凹陷深度（±4m 扰动下可忽略）。已加回归测试。
- [x] 3.1 放置偏差（2026-10-22）：**完整放置**——原版 ADR footprint（公寓 25-40×12-14m / 厂房 50×50m）只有块深 ≥ `w/2 + setback + depth` 时成立，本项目剖分可给「厂房」单 cell（内缩后仅 ~39m），参数表尺寸放不下。改为：逐边采样整排建筑可用深度（`block_depth_along_pub`，从外部入射的射线扫描，非凸多边形安全），先试参数表尺寸，放不下降级为 4-6m 模块节奏，整排放不下再丢弃。`LayoutEdge` 补 `left_block/right_block`（2.13 遗留）；`street_segments` 改为**每 unit edge 一条 segment**（原合并成 1900m 长边会让放置只有 1 个锚点，整条街正面永远放不出）。三重剔除保留（走廊 / block 凸多边形 + 凸包 / 地图边界）；树资产仅 3 种，按可用集合抽样
- [x] 3.2 树木放置偏差：3 种 glTF 懒加载（`WorldAssetRoot` + `GltfAssetLabel::Scene(0)`），行道树概率 20/60/80%、FOREST 8m 密、PLAZA 稀疏（上限 2 棵）；树不带碰撞；资产未就绪时该 chunk 无树（`trees_ready` 标记，重 plan）
- [ ] 3.5 `season.rs` + 树 `MaterialExtension` WESL：共享 storage buffer 承载 leaf_density/leaf_color/sway_strength；保留顶点色 R 通道摆动接口（sway_speed/sway_strength/sway_phase_len）——**顺延至 R4 之后（2026-10-24 用户决策）**：当前剪影+天空观感下季节仅有叶子有无差异，投入产出低
- [ ] 3.6 建筑做旧材质：BoxMesh 占位 + `StandardMaterial` 扩展（脏污/褪色/风化）——**当前下一步**
- [ ] 3.9 **程序化树形态原型（步骤 1–2）**：现状是半透明 alpha 卡片，剪影偏“纸片”。新建 `world/tree_shape.rs`（纯数据 L-system/BFS 生长模型，`StdRng` 确定性）产出「枝条骨架 + 叶簇球」；`plugins/tree_preview.rs` 用 Gizmos 在出生点前方摆 3 物种 × 4 变体画廊，F7 开关；`examples/tree_gallery.rs` headless 渲染对照图（自写 PPM→PNG，无新增依赖），`examples/tree_stats.rs` 输出高度/宽度/枝数/代理三角数。
  - [x] 步骤 1：骨架生成器 + 确定性/预算单元测试（4 个测试：同 seed 复现、跨 id/物种不同、预算区间、逐级变细）
  - [x] 步骤 2：Gizmos 画廊 + F7 开关 + headless 对照图
  - [x] 步骤 3（2026-10-27）**程序化树接入 chunk 流式**：新建 `world/tree_mesh.rs`，把 `TreeSkeleton` 烘焙成**单棵树一个 Mesh**（`ConicalFrustumMeshBuilder` 5 边锥台作枝 + 20 面 icosphere 作叶簇，`Mesh::merge` 合并），黑色剪影材质（roughness 1 / reflectance 0）。`chunks.rs` 的 `PlacementAssets` 改为持有 3 物种 × 4 变体 = **12 个共享 mesh**，替换原 glTF `WorldAssetRoot`；树随 chunk root spawn/despawn
  - [ ] 步骤 4：LOD（近全几何 / 中主枝 / 远单椭球）+ 每 Chunk 三角数预算
  - **实测（seed 0x48454144，4 变体均值）**：杨 73 枝/208 簇/约 5.0k 代理三角、冠高 10.1–11.9 m；行道树 46 枝/135 簇/约 3.3k、冠高 6.6–8.9 m；宽叶 27 枝/168 簇/约 3.7k、冠高 8.1–11.2 m。均为不透明几何预算，远低于 alpha 卡片的 overdraw 成本。
  - **调参轮（2026-10-27，用户选方案 A：先定形态再做实例化）**：
    - 对照图改为**每物种一张等轴测图**（无灭点，4 变体按列对齐），列距 = 实测冠宽 + 2.5 m；此前单一透视相机把三行压成一团、宽叶冠互相粘连
    - 新增 `crown_floor`（按每棵树自身冠底切掉下部簇，消除平直“裙边”）与 `leaf_density`（按位置哈希确定性抽稀，留出透气缝）
    - 叶簇改为**沿枝轴放置**（原为绕枝尖散布，抽稀后最外圈会变成悬浮碎块）；新增 `leaf_spread` 控制离轴漂移上限
    - 抽稀引入**邻域密度加成**：孤立球最难保留（消悬浮碎块），密集处优先保留
    - **修正 `crown_width()` 不含叶簇球半径**：原只统计枝条端点，低估每侧最多一个球径；对照图按该低估值排列距，导致最宽物种仍相互接触（已加回归测试）
    - 所有物种开启 `leaf_run_spacing`（沿枝身布簇，上限 6/枝），让叶团跟随枝条深入冠内
    - 实测（4 变体均值）：杨 72 枝/190 簇/4.7k、行道树 40 枝/130 簇/3.1k、宽叶 26 枝/112 簇/2.6k 代理三角
    - 测试 84 通过（新增 `crown_width_includes_leaf_radius`、`thinning_keeps_crowns_dense_and_grounded`）
  - **第二轮（2026-10-27，Feihei 反馈：线框看不到真实效果 + 树干太长、树冠要更贴近相机）**：
    - **预览改为实体网格**：`TreePreviewPlugin` 不再用 `Gizmos`，每根枝一个低模锥台（6 边，`ConicalFrustumMeshBuilder`），每个叶簇一个低模二十面球（`ico(1)`），近黑剪影材质（roughness 1、reflectance 0）。每棵树一个实体，画廊挂在单一根实体下，关 F7 一次性 despawn
    - **画廊贴近相机**：首排 9 m（原 16 m），行距 17 m / 列距 15 m
    - **树干缩短 + 冠底下压**（Feihei 核心诉求）：杨 `trunk_length` 0.42→0.17、`leaf_min_level`→2；行道树 0.34→0.24；新增 `leaf_start`（叶从枝条多深处开始）
    - **修总预算不守恒**：`level_share` 是每层量，但深层枝数远多于浅层（行道树 L2 8 支 vs L3 25 支），按层预算分摊导致深层枝反而更长（实测 L2 0.96 m vs L3 1.02 m）。改为补偿枝数增长的平方根
    - **实测冠底**（眼高 1.65 m）：杨 1.8–2.9×、行道树 1.1–1.6×、宽叶 1.0–1.9×；原杨为 3.3×，玩家需仰角 60° 以上才能看到冠
    - 三角代理：杨 4.5k、行道树 2.0–3.9k、宽叶 1.8–2.6k（每棵）
  - **取舍记录**：杨树的「露出 Y 形分叉」与「低冠底」在本生长模型下互斥——露出分叉要占 2–3 m 高度，会把冠底推回 3× 眼高以上。**按 Feihei 的明确要求优先压低树冠**，放弃强制露分叉（宽叶因低分叉设定天然可见）
  - **第四轮（2026-10-27，Feihei 反馈：树不随 chunk 加载，走出范围就没树，要按两次 F7 才刷出来）**：
    - 根因：R3.9 步骤 2 的画廊是**一次性快照**（按 F7 当帧玩家位置建好，之后不再更新），它只是形态调试工具，本来就不该承担世界树的职责
    - 修复：把程序化树接入 `chunks.rs`，成为 chunk 内容的一部分——chunk root despawn 时树一起消失
    - **网格预算**：初版叶簇 80 面 + 枝 6 边 → 全窗口约 **262 万三角面**，超 ADR 性能预算。改为叶簇 20 面（ico 0 级）+ 枝 5 边 + 剔除半径 <3.5 cm 的细枝 → **约 72 万**，最重一棵 4392 面，13 棵树/chunk = 5.7 万面
    - **去掉 glTF 依赖**：`build_placement_assets` 不再需要 `AssetServer`（树是程序化烘焙的），顺带让 headless 测试第一次能真正验证树的流式加载
    - 新增 `procedural_trees_stream_with_their_chunks`：断言树在出生点生成、每棵只有 1 个 `Mesh3d`、网格非空、**传送到 300 m 外后树跟着走且没有一棵留在原地或脱离 chunk root**。已验证**把“树挂到 chunk root 外”的 bug 放回去该测试即失败**（报 184 trees not attached）
  - **第五轮（2026-10-27，Feihei 反馈：仍需按两次 F7 才加载新区域的树；并清理临时文件）**：
    - **真实原因：Feihei 看到的一直是 F7 画廊，不是世界里的树**。实测确认 chunk 树一直正常流式（走 100 m 后 chunk 范围 4..11 → 6..14，树 191 → 230）；而画廊是固定快照，走出范围就空、按两次才会在新位置重建——现象完全吻合
    - **处置：F7 画廊从 `HeadsUpPlugins` 注销**。它已完成使命（形态定稿），世界树由 chunk 提供；模块保留作为 `examples/tree_gallery.rs` 的形态参照
    - **顺带修掉两个每帧泄漏（严重）**：
      1. `build_placement_assets` **没有 guard，每帧重跑**——每帧重烘 12 个树 mesh、重建全部建筑材质、替换 `PlacementAssets` 资源。实测 240 帧：mesh 13 → 3242、材质 15 → 3411。加 `existing.is_some()` 早退后稳定在 64/65
      2. `spawn_chunk_content` **每个建筑都 `meshes.add(Cuboid)`**——改为共享单位立方体 + 每实例缩放，`spawn_chunk_content` 不再需要 `Assets<Mesh>`
    - 新增 `placement_assets_and_meshes_are_allocated_once`：跑 180 帧断言 mesh/材质数量不增长。已验证**去掉 guard 该测试即失败**（报 "mesh assets grow every frame"）
  - **第六轮（2026-10-27，Feihei 反馈：F7 无变化、走动旋转都看不到树，建筑盒子也不见了）——这是真正的根因**：
    - **`ChunkRoot` 实体没有 `Transform`/`Visibility`**。它是所有 chunk 内容的父节点（地面 tile / 建筑 / 树），而 Bevy 的层级传播要求父实体带 `Transform`（→`GlobalTransform`）。缺了它，**所有子节点拿不到世界变换，一个都渲染不出来**。而 gizmo 线框不走层级，所以「天空 + 彩色线框正常、网格全空」——症状完全吻合
    - 发现方式：抓真实游戏日志，看到 **1120 条 `warning[B0004]`**（"parent ... without [Transform]"），远早于这次改动就存在——**之前一直是 glTF `WorldAssetRoot` 树 + 本轮之前没人看日志**才没暴露
    - 修复：`ChunkRoot` 补 `Transform::default()` + `Visibility::default()`。实测 25 s 运行：B0004 从 1120 → **0**，无其它告警
    - 新增 `chunk_root_can_propagate_transforms_to_its_children`：断言每个 chunk root 都有 `Transform`、每个子节点都拿到 `GlobalTransform`。已验证**去掉修复该测试即失败**（报 30 chunk roots have no Transform）
    - 顺带删掉 `PlayerBody::clamp_yaw_offset` 及其测试（鼠标横向转向已移除，该函数不再被调用；`yaw_offset` 字段保留，相机旋转仍用）。至此 `cargo clippy -D warnings` 全绿、零告警
  - **仍未解决**：冠内枝干在渲染中不可见（叶簇球覆盖枝段，需实例化阶段用材质/明暗区分）；行道树 V2 树干偏长、组内一致性差
  - **第三轮（2026-10-27，Feihei 反馈：按 F7 后连线框都没有）——两个叠加 bug，都不是材质/背面剔除问题**：
    - **Bug A：空缓存提前返回**。`sync_tree_preview` 里 `if !roots.is_empty() || data.trees.is_empty() { return; }`——打开 F7 的当帧 `data.trees` 必然为空，条件成立直接返回，画廊从未被建造。删掉 `|| data.trees.is_empty()`
    - **Bug B：锚在世界原点**。`TreePreviewGalleryData::generate(Vec3::ZERO)` 固定用世界原点，而玩家出生在地图中心 400 m 处 → 树在 588 m 外。改为以玩家位置 + 水平朝向为锚（抽出 `gallery_origin()` 便于测试）；变体沿玩家右手方向横排，不再朝前堆叠
    - **为什么之前“验证过 618 个 mesh 存在”却仍是空的**：headless 测试只统计了实体数量，没检查它们在世界中的位置；而且那两个 bug 都在 `sync_tree_preview` 里，测试却只调 `generate()`，属于“测了错的层”
    - 新增 `f7_toggle_actually_builds_the_gallery_in_front_of_the_player`：**驱动 `sync_tree_preview` 本身**，断言能建成、距玩家 <80 m、在朝向前方、立于地面、变体横向排开。已验证**把任一 bug 放回去该测试即失败**（Bug A 报“built no gallery”，Bug B 报“588 m from the player”）
  - **验证缺口（重要）**：`tests/tree_preview_screenshot.rs` 的 `frame` 在本机 headless 精简插件栈下恒为清屏色，即使换成 HDR 目标 + 完整 `GoldenHourCamera` bundle 也不出画面，与 `sky_screenshot.rs` 同栈却正常，差异未定位。**因此「游戏内观感」只有 Feihei 本人 `cargo run` + F7 才是有效证据**；本文所有形态结论均来自 `examples/tree_gallery.rs`（纯 CPU 光栅化）。
  - **已知验证缺口**：`tests/tree_preview_screenshot.rs` 在精简插件栈下能跑通（F7 生效、12 棵树生成），但截图为空——HDR 相机在这套栈里没有产出画面（`sky_screenshot.rs` 的栈可正常出图，差异未定位）。**桥接测试只覆盖逻辑层，不覆盖“游戏内可见”**；游戏内观感待 Feihei 本地按 F7 人工确认。
- [ ] 3.7 季节切换快捷键（1/2/3/4，开发期）——随 3.5 顺延
- [x] 3.8 headless 验证（碰撞围合部分）：`walking_every_street_reaches_the_far_end`——沿全部路网边（2112 条 × 4 seed）按 0.5m 步长模拟步行，断言不卡死、不穿块、不出世界、单调前进。**该测试暴露并驱动修复**：环路中线格点扰动、图节点坐标错位、左右归属按基准格点判定、row_fits 与放置偏移不一致（完整链路见 2.9 修订）。修复后全城建筑 2787 → 10091（此前静默丢弃 ~90% 放置）。四季树形态部分随 3.5 顺延

### 可验证

- 街道行走看到建筑与树沿路排列，风格比例约 50/30/20
- 被负形街道碰撞实体限制，不能进入建筑区；可进入 PLAZA
- **环绕地图一周不被卡住、也不漏进边缘**（验证 2.11 的外侧围墙零缝）
- 切换季节树叶密度/颜色/摆动变化
- 同一 seed 每次生成完全一致；碰撞实体不渲染（无 Draw Call）

---

## 阶段 R4: 存档与菜单（预计 2-3 天）

**目标**：复现原 M4 存档/菜单系统。

- [x] 4.1 探索位图：`ExploredBitmap`（chunk 键 HashSet + 进度/完成判定 + 排序键序列化接口）+ `PlayerChunkChanged` 消息，`track_player_chunk` 挂在流式系统之后（同为 Update）
- [x] 4.2 存档序列化：`SaveData`（version/seed/season/explored 排序键/pos/yaw，serde JSON）+ `saves/slot_N.json` 落盘；season 为占位枚举（字段先占格式，R3.5 落地无需迁移）；round-trip 位精确 + 版本字段可检测试
- [x] 4.3 存档恢复：`LoadRequest` 消息 → `handle_load_request`（Update，先于 Streaming）：版本检查拒绝未来格式、seed 重建布局图（实测 <1s）、清空全部 chunk 实体/流式队列/碰撞索引、恢复位图与玩家位姿（yaw + 默认 pitch）
- [x] 4.4 多存档：`SaveGameRequest` 手动写入 + 自动存档（chunk 变化后节流 30s，停留不写盘）；`list_slots()` 元数据列表（seed/season/探索数/mtime，损坏文件跳过）；槽 0 自动 + 槽 1-3 用户
- [x] 4.5 菜单 UI（bevy_ui，键盘驱动极简风）：主菜单（seed 键入+Enter 开始/L 读档/Esc 退出）、Esc 暂停菜单（继续/S 存槽 1/M 回菜单）、HUD（探索 N/1600 %）；`MenuUiRoot` 按阶段重建面板；光标按阶段持有（Playing=Locked，菜单=None）；可点击按钮留 R5.6 润色
- [x] 4.6 菜单驱动开局 + `is_game_active` 门控 + 全探索完成消息：`GamePhase`（MainMenu/Playing/Paused）为唯一事实来源；开局复用 R4.3 恢复流（Start = 带 seed 的新鲜 LoadRequest）；玩家移动/注视/调速、流式、探索全部按 phase 门控；玩家侧 Esc 光标切换移除（菜单接管光标）；`WorldExplorationCompleted` 消息全探索时触发一次（待完成弹窗接线 R5.6）

### 可验证

- 菜单 → 散步 → Esc 保存 → 读档回到同一位置/季节/进度
- 同一 seed 地图结构完全一致；多存档互不干扰
- 全探索触发完成提示

---

## 阶段 R5: 声音与润色（预计 2-3 天，原 M5 顺延）

**目标**：环境音、配乐与细节打磨（原 M5 任务原样继承）。

- [ ] 5.1 脚步声系统：按步行速度播放不同节奏
- [ ] 5.2 风声音效：与 sway_strength 联动
- [ ] 5.3 环境音：鸟叫、远处城市低语，随机触发
- [ ] 5.4 克制配乐：短时间片段，偶尔触发，不持续播放
- [ ] 5.5 性能优化：LOD（远处建筑低模）、Draw Call 控制，确保 2-4GB VRAM 流畅 30fps+
- [ ] 5.6 总体打磨：色调微调、速度手感、UI 美化、Bug 修复
- [ ] 5.7 跨平台验证：Windows + 至少一个其他后端（Vulkan/Metal）核验大气/云/TAA

### 可验证

- 完整体验：菜单 → 散步 → 存档 → 读档 → 再散步
- 声音自然不突兀；低端 GPU 稳定 30fps+

---

## 进度总览

```
R0: 骨架与依赖锁定   [██████████] 完成（2026-10-20）
R1: 天空系统迁移     [████████░░] 完成（2026-10-20，色调饱和度待人眼迭代）
R2: 布局图与Chunk    [██████████] 完成（2.1–2.15，2026-10-21）
R3: 放置碰撞与季节   [████████░░] 基本完成（3.1-3.4, 3.6, 3.8 碰撞围合 done 2026-10-24；3.5/3.7 顺延）
R4: 存档与菜单       [██████████] 完成（4.1-4.6，2026-10-24；可点击按钮/完成弹窗接线留 R5.6）
R5: 声音与润色       [░░░░░░░░░░] 待开始（原 M5 顺延）
R3.9 程序化树         [██████████] 步骤 1-3 完成 + 层级传播根因修复（2026-10-27，实体已能在游戏内渲染）；
                      步骤 4（LOD / 每 chunk 三角数预算）待开始
```

**下一步**：R5 声音与润色（原 M5 顺延：脚步声/风声/环境音/配乐/LOD/打磨/跨平台）。季节（3.5/3.7）顺延至 R4 后。2026-10-24：R3.8 步行测试暴露的街道几何一致性修复见 2.9 修订（环路格点钉死、图节点用扰动坐标、归属按几何侧、curb 逐街宽）。

## 风险与依赖

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| bevy_elements 0.5.3 依赖不匹配 0.20 | R0 阻塞 | `cargo check` 先行；不匹配回退 ADR-0005 方案 B（0.19 或纯内置） |
| bevy_elements 无人工审计 | 隐藏 bug/行为不符 | R0.3 读源码；关键参数测试断言 |
| Bevy 生态版本迭代快 | 升级成本 | 精确锁版本；升级独立里程碑 |
| Blender 建筑模块制作耗时 | R3 建筑占位 | BoxMesh 占位先行（沿用原项目策略） |
| 树 .blend → glTF 导出质量 | 视觉质量 | R0.5 先行验证 |
| WESL 重写顶点色摆动 | 树动画损失 | 接口保留在 ADR-0004，R3.5 重点验证 |
| **扰动后相邻街区交叠** | 街道缝隙错位、碰撞穿插 | 原版实测过只取 4 角点导致 **7.47m 交叠**；必须沿完整周长取 `(n+1)×4` 顶点（2.10），OVERLAPS=0 测试覆盖多 seed |
| **外侧围墙带留缝** | 玩家从地图边缘走出/看到空洞 | 逐边 inset 判定（2.11）：邻居越界或外侧则 `inset=0` 直接跳过裁剪；新增共享顶点距离断言 |
| **生成算法与原版偏离** | 玩法观感不符（过整/过碎） | 已建立可复现比对路径：`cargo run --example world_map` 出顶视图，与原项目 `bake_city.tscn` 输出比对 |
