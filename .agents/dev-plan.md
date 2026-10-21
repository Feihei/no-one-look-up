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
- [x] 2.5 `chunks.rs`：50m×50m Chunk（40×40=1600，与 cell 1:1 对齐），不对称半径（前方 4 / 后方 2 / 侧 3，窗口最坏 8×8=64）流式 spawn/despawn（0.20 递归 `despawn` 组删除）+ 每 Chunk 地面 tile（seed 着色）供流式反馈 ⚠️ 原版为「切比雪夫距离 + 前后点积判定」，当前为旋转矩形窗口，略多加载；对齐见 2.15
- [x] 2.6 几何验证单元测试：半平面内缩裁剪（`inset_polygon_clip`）+ SAT；OVERLAPS=0 断言（4 个 seed 全块对）
- [x] 2.7 调试可视化：Bevy Gizmos 绘制 Chunk 边界/街道中心线/Block 轮廓（按类型着色）+ Text2d Block 标签；G 键开关（debug 构建默认开）
- [x] 2.8 系统排序显式声明（`.before()/.after()` 链）+ 调度随机化 property testing（16 种子随机注册顺序 + 歧义检测 = Error）

### 修正项（复刻原 Godot 算法，2.9–2.15）

> 对照 `scripts/path_network_generator.gd`（466 行）/ `object_placer.gd` / `chunk_manager.gd`。按依赖顺序实施，**必须在 R3.1 放置之前完成**（放置沿 Block 边缘排布，边界不扰动则建筑会是整齐直线）。

- [x] 2.9 **扰动网格**：内部格点随机偏移，边界节点两轴均固定（原版只锁单轴，上/左边界仍会滑动，导致外墙off-border）。`PERTURB_MAX` **取 4m**（非原版 12m）：50m 格距下 12m 双向扰动使相邻格点相对偏移达 24m、共享边偏角 53°，按半街宽内缩会产生 5.4m 横向误差 > 街宽 → 实测 280+ 对穿插、最深 20.8m；4m 下偏角约 22°、误差 1.6m，街道仍有可见摆动（格边 34-68m）
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

- [ ] 3.1 `placement.rs`：沿 edge 统一放置算法 + 差异化参数表（老城区 4-6m / 公寓 25-40m×12-14m / 厂房 50×50m）+ setback（1/5/10m）+ 朝向（atan2）+ edge 中点去重；层数取 2.13 的 `min_y/max_y`
- [ ] 3.2 树木放置：5 种 glTF 懒加载，行道树概率 20/60/80%，FOREST 密 8m，PLAZA 稀疏，树不带碰撞
- [ ] 3.3 `collision.rs`：负形街道模型——每个 Block 一个不可见实心凸棱柱（y=0..BLOCK_SOLID_HEIGHT=2m），取 `hull(block.polygon)` 保证凸性；校准至原版 `object_placer.gd::_place_solid_block`。规则：PLAZA 跳过、内部 FOREST 跳过、其余（建筑类 / 外侧围墙带 / 边缘 FOREST）生成。原版另有 `DEBUG_SHOW_COLLISION` 半透明可见棱柱，作为 `is_outer` / 凸性核对的调试开关一并移植
- [ ] 3.4 自研运动学玩家控制器：视角钳制（±10°/40–60°）、速度 1.2-2.0 m/s 可调、无跳跃/跑步/FOV；边界阻挡
- [ ] 3.5 `season.rs` + 树 `MaterialExtension` WESL：共享 storage buffer 承载 leaf_density/leaf_color/sway_strength；保留顶点色 R 通道摆动接口（sway_speed/sway_strength/sway_phase_len）
- [ ] 3.6 建筑做旧材质：BoxMesh 占位 + `StandardMaterial` 扩展（脏污/褪色/风化）
- [ ] 3.7 季节切换快捷键（1/2/3/4，开发期）
- [ ] 3.8 headless 截图测试：四季树形态（冬=透明枝骨架）、街区碰撞围合

### 可验证

- 街道行走看到建筑与树沿路排列，风格比例约 50/30/20
- 被负形街道碰撞实体限制，不能进入建筑区；可进入 PLAZA
- **环绕地图一周不被卡住、也不漏进边缘**（验证 2.11 的外侧围墙零缝）
- 切换季节树叶密度/颜色/摆动变化
- 同一 seed 每次生成完全一致；碰撞实体不渲染（无 Draw Call）

---

## 阶段 R4: 存档与菜单（预计 2-3 天）

**目标**：复现原 M4 存档/菜单系统。

- [ ] 4.1 探索位图：`explored` 记录 + `player_chunk_changed` Message 标记
- [ ] 4.2 存档序列化：serde JSON（seed + season 枚举 + total_chunks + explored + player_pos + player_yaw），本地存档目录
- [ ] 4.3 存档恢复：从 seed 重建布局图（<1s），恢复位图与玩家位姿；round-trip 单元测试
- [ ] 4.4 多存档：自动存档槽 + 3 用户槽位，列表（seed/季节/进度/时间）
- [ ] 4.5 菜单 UI（Bevy UI）：主菜单（seed 输入+季节选择/读档/退出）、Esc 暂停菜单、保存面板、探索完成弹窗、HUD
- [ ] 4.6 菜单驱动开局 + `is_game_active` 输入门控 + 全探索 `world_exploration_completed` Message

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
R2: 布局图与Chunk    [████████░░] 基线完成（2.1–2.8）｜待扰动修正（2.9–2.15）
R3: 放置碰撞与季节   [░░░░░░░░░░] 待开始（硬依赖 R2.9–R2.11）
R4: 存档与菜单       [░░░░░░░░░░] 待开始
R5: 声音与润色       [░░░░░░░░░░] 待开始（原 M5 顺延）
```

**下一步**：R2.9–R2.15（复刻原 Godot 扰动路网），完成后进 R3。

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
