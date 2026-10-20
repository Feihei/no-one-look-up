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

- [ ] 2.1 `layout_graph.rs`：LayoutGraph 数据结构（节点、边、宽度、Block 标签、边界，serde）
- [ ] 2.2 `world_grid.rs`：2D 网格中间结构（block_id/道路/路口，get_cell 快速查询，serde）
- [ ] 2.3 `path_network.rs`：Block 优先生成算法（合并规模：老城区 1×1 / 公寓 2-3 / 厂房 4-5 / 广场 1×1 / 树林边缘），内部道路物理删除，环路无死胡同
- [ ] 2.4 种子系统：`rand_chacha::StdRng::seed_from_u64(seed)`，同 seed bit-exact 单元测试
- [ ] 2.5 `chunks.rs`：50m×50m Chunk 划分，不对称半径（前方 4 / 后方 2，峰值约 55 Chunk）流式 spawn/despawn（`despawn_all`）
- [ ] 2.6 几何验证单元测试：半平面内缩裁剪（`inset_polygon_clip`）+ OVERLAPS=0 断言
- [ ] 2.7 调试可视化：Bevy Gizmos 绘制 Chunk 边界/路径网络/Block 标签（开发模式可开关）
- [ ] 2.8 系统排序显式声明（`.before()/.after()`）+ 调度随机化 property testing

### 可验证

- 调试模式可见路径网络与 Block 分布
- 移动时周围物体出现/消失，无明显卡顿
- 同一 seed 每次生成的地图完全一致（bit-exact 测试）
- 几何测试 OVERLAPS=0

---

## 阶段 R3: 放置、碰撞与季节（预计 3-4 天）

**目标**：移植原 M3 的放置/碰撞/季节决策（负形街道模型最终形态）。

- [ ] 3.1 `placement.rs`：沿 edge 统一放置算法 + 差异化参数表（老城区 4-6m / 公寓 25-40m×12-14m / 厂房 50×50m）+ setback（1/5/10m）+ 朝向（atan2）+ edge 中点去重
- [ ] 3.2 树木放置：5 种 glTF 懒加载，行道树概率 20/60/80%，FOREST 密 8m，PLAZA 稀疏，树不带碰撞
- [ ] 3.3 `collision.rs`：负形街道模型——实心不可见凸棱柱（y=0..2m），街宽 6-10m + jitter，外圈 cell（OUTER_BAND_CELLS=1）自动围合；按 BlockType+is_outer 碰撞规则表
- [ ] 3.4 自研运动学玩家控制器：视角钳制（±10°/40–60°）、速度 1.2-2.0 m/s 可调、无跳跃/跑步/FOV；边界阻挡
- [ ] 3.5 `season.rs` + 树 `MaterialExtension` WESL：共享 storage buffer 承载 leaf_density/leaf_color/sway_strength；保留顶点色 R 通道摆动接口（sway_speed/sway_strength/sway_phase_len）
- [ ] 3.6 建筑做旧材质：BoxMesh 占位 + `StandardMaterial` 扩展（脏污/褪色/风化）
- [ ] 3.7 季节切换快捷键（1/2/3/4，开发期）
- [ ] 3.8 headless 截图测试：四季树形态（冬=透明枝骨架）、街区碰撞围合

### 可验证

- 街道行走看到建筑与树沿路排列，风格比例约 50/30/20
- 被负形街道碰撞实体限制，不能进入建筑区；可进入 PLAZA
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
R2: 布局图与Chunk    [░░░░░░░░░░] 待开始
R3: 放置碰撞与季节   [░░░░░░░░░░] 待开始
R4: 存档与菜单       [░░░░░░░░░░] 待开始
R5: 声音与润色       [░░░░░░░░░░] 待开始（原 M5 顺延）
```

## 风险与依赖

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| bevy_elements 0.5.3 依赖不匹配 0.20 | R0 阻塞 | `cargo check` 先行；不匹配回退 ADR-0005 方案 B（0.19 或纯内置） |
| bevy_elements 无人工审计 | 隐藏 bug/行为不符 | R0.3 读源码；关键参数测试断言 |
| Bevy 生态版本迭代快 | 升级成本 | 精确锁版本；升级独立里程碑 |
| Blender 建筑模块制作耗时 | R3 建筑占位 | BoxMesh 占位先行（沿用原项目策略） |
| 树 .blend → glTF 导出质量 | 视觉质量 | R0.5 先行验证 |
| WESL 重写顶点色摆动 | 树动画损失 | 接口保留在 ADR-0004，R3.5 重点验证 |
