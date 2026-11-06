# 即使独自漫步你也很少看天

<a id="zh" name="zh"></a>

[English version ↓](#english) · *No One Looks Up Even Alone*（简称 **No One**）· Rust + Bevy 0.20

## 序

> 向晚意不适，驱车登古原。夕阳无限好，只是近黄昏。 ————李商隐《乐游原》

我小时候晚饭后喜欢四处散步，看黄昏的天空被各种景物裁剪出来的形状，直到天色暗下。

多年后，这看似日常的喜好却变得难以实现————日常的傍晚变成了下班后在城市的喧嚣中的行色匆匆，还要不时低头看信息，即使假日的傍晚有幸散散步，却难逃阴雨和雾霾。

我希望有一个永不结束的黄昏，在里面可以无尽随意的散步，这便是这个游戏的由来。

黄昏，一条没有名字的街。你唯一被要求做的事，是抬头。

## 关于

这是一个第一人称散步游戏。地图 0.8 km × 0.8 km，随机生成，时间锁在日落前的黄金时刻。没有任务、没有敌人、没有失败，也没有终点——走完一圈，你可以继续走，或者换一个 seed 重新开始。

真正定义它的是视角。仰角只能在 22.55°–60° 之间，所以你看到的城市只有上半截：檐口、女儿墙、屋顶水塔、电线横担、被枝叶剪碎的树冠，以及它们背后那片一直在变的天空。地面不在画面里，所以街道不是用来「通过」的，它只是让你慢慢走过去。

天空是物理散射算出来的，不是一张 HDRI。太阳落在离地平线约 3° 的位置：地平线一侧是金橙到橙红，抬头到天顶就是深蓝紫。卷云按需铺开，星被明亮的天色压掉，月亮已经升起，三层远山压在天际线尽头。天气锁定晴朗——这个游戏不打算用雨雪制造变化，变化来自光本身。

世界由 seed 决定：同一个 seed，同一座城，逐位一致（bit-exact）。

## 操作

| 键 | 作用 |
|---|---|
| `W` / `S` | 前进 / 后退 |
| `A` / `D` | 左转 / 右转（60°/s） |
| 鼠标 | 只控制俯仰（22.55°–60°），不改变朝向 |
| `E` / `Q` | 加速 / 减速（1.2–2.0 m/s，步进 0.1） |
| `Esc` | 暂停菜单：继续 / 存档 / 回主菜单 |
| `Tab` | 主菜单切换季节；开局后不再改变 |
| `Enter` | 开始散步（主菜单） |
| `L` | 读取一个已有存档（主菜单） |
| `S` | 存到 1 号槽（仅在暂停菜单中） |

开发用按键：`G` 开关调试线框，`F5` 打开天空调参面板（云种 / 云量 / 湿度 / 曝光 / Bloom / 卷云），`1`–`4` 在非游戏状态切换季节。

## 它怎么运转

**世界。** 开局由 seed 一次性生成路径拓扑，行走时再按 chunk 流式实例化物体。街区优先生成：先把地图剖分成完整矩形，每个矩形分配一种用途（老城 / 公寓 / 厂房 / 广场 / 树林），街道因此成为「负形」——不是画出来的路，而是街区之间剩下的缝。内部格点会做 ±4 m 的确定性扰动（边界与环路中线钉死），街区边界随之错位；内缩按顶点角平分线计算，保证相邻街区永不交叠（`OVERLAPS = 0` 是硬性测试）。地图外圈是景物带，玩家被一圈不可见的实心棱柱围在里面，看得到界外，走不出去。

**碰撞。** 没有物理引擎。每个街区就是一个不可见的凸棱柱（y = 0–2 m），行走时对凸多边形做包含测试并沿最浅穿透轴推出。街道宽度是算出来的，不是摆出来的。

**树。** 骨架是确定性的广度优先分枝生长结果（纯数据，可 headless 断言），叶子是贴了程序化蒙版的小卡片，布在树冠壳层而不是体积内部——视线穿过树冠时是「叶—空—叶」，而不是一团实心绿。三个物种（塔形杨、行道树、宽叶）× 四个变体 × 三级 LOD（40 / 110 / 220 m），叶片蒙版分别是杨、槐、柳；LOD 之间用 dither 交叉淡出。四季只改一个 32 字节的共享 uniform：叶密度、叶色、摆动强度，切换时全城的树一起变，没有逐材质遍历。

**建筑。** 三种风格各有自己的开间与层高（老城 2.0 m 开间 / 3 m 层高，公寓 3.3 m，厂房 6.0 m），立面贴图按米制 UV 程序化烘焙，可平铺且接缝处不出现台阶；玻璃近黑、低粗糙度，靠掠射角的天空 IBL 产生反射，不需要 SSR。亮窗按窗格哈希点亮，暖白给住宅、冷白给厂房。屋顶只做剪影：双坡、女儿墙、锯齿，一个风格一种。

**天空。** Bevy 0.20 内置大气散射（Rayleigh / Mie）+ bevy_elements 的云 / 天体 / 雾模块，外面包一层 `DuskSky` 把时间锁在 17:30、天气锁在晴朗。卷云走一张 900 m 半径的穹顶网格，由像素反投影出方向后采样噪声，因此不会随视角漂移；远山是三圈不透明 unlit 山脊带，颜色随距离变浅。

## 技术

| 项 | 值 |
|---|---|
| 引擎 | Bevy 0.20.1（精确锁版本） |
| 语言 | Rust，edition 2024，最低 1.97.1 |
| 地图 | 0.8 km × 0.8 km，16 × 16 格，格边 50 m |
| Chunk | 50 m × 50 m，共 256 个；切比雪夫半径前 4 后 2，每帧最多各 1 个加载 / 卸载 |
| 视高 | 1.65 m；无跳跃、无奔跑、无 FOV 变化 |
| 步速 | 1.2–2.0 m/s（默认 1.5），加速 8 m/s²，60°/s 转身 |
| 后期 | ACES 色调映射、Bloom、体积雾、距离雾 |
| 目标硬件 | 2–4 GB VRAM 的中低端独显，30 fps+ |
| 测试 | 210 个库测试 + 2 个集成测试；后者之一需真实 GPU（`--ignored`），另一个需未入库的 glTF |

确定性是硬约束：所有程序化生成走 `rand_chacha`，禁止全局 `thread_rng` 进入生成路径；系统之间的顺序全部显式声明，并有一层调度随机化的 property test 来暴露隐式依赖。

## 构建与运行

需要 Rust 1.97.1+ 与支持 Vulkan / DX12 的显卡。

```bash
cargo run --release      # 散步
cargo test               # 单元与集成测试
cargo fmt --check && cargo clippy --all-targets -- -D warnings
```

首次编译 Bevy 较慢。开发调试请用 `cargo run`（dev profile 已带 opt-level 1 / 依赖 opt-level 2），迭代远快于 release。

几个不需要开窗的诊断工具：

```bash
cargo run --example world_map        # 顶视图 PNG：路网、街区、边界（纯 CPU 光栅化）
cargo run --example tree_gallery     # 三物种等轴测对照图
cargo run --example tree_stats       # 冠高、枝数、叶片卡片数、冠内开口率
cargo run --example facade_gallery   # 立面贴图对照表
cargo run --example roof_gallery     # 屋顶剪影对照表
cargo run --example placement_stats  # 各类物体的放置产量与拒绝率
```

## 项目结构

```
src/
  main.rs            应用装配（窗口、相机、玩家实体）
  lib.rs             库入口，便于 headless 测试驱动
  core/              GamePhase、输入动作常量、错误类型
  player/            运动学控制器（转向、俯仰钳制、速度、子步碰撞）
  plugins/           天空、云、远山、季节、存档、菜单、天空调参面板
  world/             路网生成、街区剖分、放置、碰撞、立面、屋顶、树、Chunk 流式
assets/shaders/      树摆动、立面 UV、卷云（WESL）
vendor/bevy_elements 打过 3 个云 shader 补丁的 vendored 依赖
.agents/             ADR 与开发计划（中文）
examples/            离线诊断与对照图工具
tests/               资产链路、天空截图、顶视图
```

## 设计文档

`.agents/` 下是这套东西为什么长成这样的完整记录：

| 文档 | 内容 |
|---|---|
| `adr-0001-game-design.md` | 核心体验、视觉风格、世界与存档的设计约束 |
| `adr-0002-world-generation.md` | 两阶段生成、扰动路网、Chunk 流式 |
| `adr-0003-sky-system.md` | 大气散射选型、云与天体的方案取舍 |
| `adr-0004-placement-collision-seasons.md` | 放置、负形街道碰撞、季节参数表 |
| `adr-0005-engine-migration.md` | Godot → Bevy 的映射与被否决的选项 |
| `dev-plan.md` | 分阶段 checkbox 计划与逐轮实测记录 |

## 状态

| 阶段 | 内容 | 状态 |
|---|---|---|
| R0 | 项目骨架、依赖锁定、树资产链路 | 完成 |
| R1 | 天空系统（大气、黄昏、云、星月） | 完成 |
| R2 | 路网与 Chunk 流式 | 完成 |
| R3 | 放置、碰撞、季节、程序化树、立面、屋顶、街道家具 | 完成，部分观感待人工验收 |
| R4 | 存档与菜单 | 完成 |
| R5 | 声音（脚步 / 风 / 鸟鸣 / 配乐）与润色 | 未开始（天空调参面板与云种已提前落在 R5.1–5.5） |

## 已知缺口

- `tests/assets_load.rs` 需要 `assets/models/` 下的三个 glTF，而该目录未入库；新克隆的仓库跑 `cargo test` 时只有这一个测试会因缺文件失败（报错信息会直接指出缺少哪个文件）。程序化树不依赖这些资产，游戏本身可以正常运行。
- 仓库目前没有 `LICENSE` 文件，`Cargo.toml` 声明的是 `MIT OR Apache-2.0`。
- 树种的生长参数（塔形杨 / 行道树 / 宽叶）与三张叶片蒙版（杨 / 槐 / 柳）尚未完全一一对齐，观感上能区分，命名上还没统一。
- 云的边缘仍有 32–64 m 尺度的块状感，根因是噪声分辨率，属已知边界。
- 声音系统完全未实现；当前版本是无声的。

---

<a id="english" name="english"></a>

# No One Looks Up Even Alone

[中文版（Chinese） ↑](#zh)

You can also call it **No One** · Rust + Bevy 0.20.

> Pitch is clamped to 22.55°–60°. The bottom edge of the frame just grazes the horizon, so the ground never enters the view.

## Preface

> Toward evening, ill at ease, I drove up onto the old plain.<br>
> The setting sun is beautiful without limit —<br>
> only that it is so near to dusk.<br>
> — Li Shangyin, *Leyou Plain*

As a child I liked to wander after dinner, watching the dusk sky get cut into shapes by whatever stood between it and me, until the light went.

Years later that ordinary-seeming pleasure became hard to reach. The evening turned into a hurried walk home through the noise of the city, head down every few minutes to check a screen; and on the rare holiday evening with time to walk, there was rain, or smog, or both.

I wanted a dusk that never ends, and a street to walk in for as long as I liked. That is where this game comes from.

Dusk, an unnamed street, and the only thing you are asked to do is look up.

## About

A first-person walking game. The map is 0.8 km × 0.8 km, procedurally generated, and the clock is frozen at golden hour. No objectives, no enemies, no failure, no finish line — when you have walked the whole thing you can keep walking, or start over with a different seed.

What actually defines it is the camera. Pitch only exists between 22.55° and 60°, so you see the upper half of the city and nothing else: eaves, parapets, rooftop water tanks, crossarms and overhead wires, tree crowns cut into lace by their own leaves, and the sky changing behind all of it. The ground is never in frame, so the street is not something you "get through" — it is just what you walk along.

The sky is physically scattered, not an HDRI. The sun sits about 3° above the horizon: gold to orange-red along the horizon, deep blue-violet straight overhead. Cirrus spreads when you ask for it, the stars are washed out by the bright sky, the moon is already up, and three layers of distant hills sit on the skyline. The weather is locked clear — this game does not use rain or snow to make things happen. The change comes from the light.

The world is a seed. Same seed, same city, bit-exact.

## Controls

| Key | Action |
|---|---|
| `W` / `S` | Walk forward / back |
| `A` / `D` | Turn left / right (60°/s) |
| Mouse | Pitch only (22.55°–60°); does not steer |
| `E` / `Q` | Faster / slower (1.2–2.0 m/s, steps of 0.1) |
| `Esc` | Pause menu: resume / save / main menu |
| `Tab` | Cycle season in the main menu; fixed once the walk starts |
| `Enter` | Start walking (main menu) |
| `L` | Load an existing save (main menu) |
| `S` | Save to slot 1 (pause menu only) |

Developer keys: `G` toggles the debug overlay, `F5` opens the sky panel (cloud kind / amount / humidity / exposure / bloom / cirrus), `1`–`4` switch season outside a run.

## How it works

**The world.** The street topology is generated once from the seed at startup; objects are then streamed in per chunk as you walk. Blocks come first: the map is partitioned into whole rectangles, each assigned a use (old town / apartments / factory / plaza / grove), which makes the streets *negative space* — not drawn roads, but what is left between blocks. Interior grid points are displaced by a deterministic ±4 m (the border and the ring road's centreline stay pinned), so block edges fall out of alignment; the inset is computed along vertex angle bisectors, which keeps neighbouring blocks from ever overlapping (`OVERLAPS = 0` is an enforced test). The outer ring is a scenery band: the player is enclosed by invisible solid prisms, able to see past the edge but never walk off it.

**Collision.** There is no physics engine. Each block *is* an invisible convex prism (y = 0–2 m); walking means a point-in-convex-polygon test followed by a push along the shallowest penetration axis. Street widths are computed, not authored.

**Trees.** Skeletons come from a deterministic breadth-first branching model (pure data, assertable headless); leaves are small alpha-masked cards placed on the *shell* of the crown rather than throughout its volume, so a sightline through a tree reads leaf–air–leaf instead of a solid green mass. Three species (columnar aspen, street tree, broadleaf) × four variants × three LOD tiers (40 / 110 / 220 m), wearing poplar, sophora and willow leaf masks; the tiers are crossfaded with dithering. A season is one 32-byte shared uniform — leaf density, tint, sway strength — so the whole city changes at once with no per-material walk.

**Buildings.** Each style has its own bay and floor height (old town: 2.0 m bays, 3 m floors; apartments: 3.3 m; factory: 6.0 m). Facade textures are baked procedurally in metric UV space, tileable with no step at the seam; glass is near-black with low roughness and gets its reflection from the sky IBL at grazing angles, so no SSR is needed. Lit windows are picked by pane hash — warm white for housing, cool white for the factory. Roofs are silhouette only: gable, parapet, sawtooth, one per style.

**Sky.** Bevy 0.20's built-in atmospheric scattering (Rayleigh / Mie) plus bevy_elements for clouds, celestial bodies and fog, wrapped in a `DuskSky` resource that pins the clock to 17:30 and the weather to clear. Cirrus is a 900 m dome mesh whose direction is reconstructed per pixel from its own fragment, so the pattern does not drift with the camera; the distant hills are three opaque unlit ridge bands, paler with distance.

## Tech

| Item | Value |
|---|---|
| **Engine** | Bevy 0.20.1 (pinned exactly) |
| **Language** | Rust, edition 2024, MSRV 1.97.1 |
| **Map** | 0.8 km × 0.8 km, 16 × 16 cells of 50 m |
| **Chunks** | 50 m × 50 m, 256 total; Chebyshev radius 4 ahead / 2 behind, at most one load and one unload per frame |
| **Eye height** | 1.65 m; no jump, no sprint, no FOV change |
| **Walk speed** | 1.2–2.0 m/s (1.5 default), 8 m/s² acceleration, 60°/s turn |
| **Post** | ACES tonemapping, bloom, volumetric and distance fog |
| **Target** | 30 fps+ on a low-end discrete GPU with 2–4 GB VRAM |
| **Tests** | 210 lib tests + 2 integration tests; one needs a real GPU (`--ignored`), the other needs untracked glTF files |

Determinism is a hard constraint: every procedural path goes through `rand_chacha`, and no global `thread_rng` may enter generation. System order is declared explicitly everywhere, and a schedule-randomisation property test exists to flush out implicit dependencies.

## Build and run

Rust 1.97.1+ and a GPU with Vulkan or DX12.

```bash
cargo run --release      # walk
cargo test               # unit and integration tests
cargo fmt --check && cargo clippy --all-targets -- -D warnings
```

The first build of Bevy is slow. For day-to-day work use `cargo run` — the dev profile is opt-level 1 with dependencies at 2, which iterates far faster than release.

Windowless diagnostics:

```bash
cargo run --example world_map        # top-down PNG: streets, blocks, border (pure CPU)
cargo run --example tree_gallery     # isometric contact sheets, one per species
cargo run --example tree_stats       # crown size, branch/card counts, crown opening rate
cargo run --example facade_gallery   # facade texture sheets
cargo run --example roof_gallery     # roof silhouette sheets
cargo run --example placement_stats  # placement yield and rejection rate by kind
```

## Layout

```
src/
  main.rs            app assembly (window, camera, player entity)
  lib.rs             library entry so tests can drive the app headless
  core/              GamePhase, input action constants, error type
  player/            kinematic controller (turn, pitch clamp, speed, substepped collision)
  plugins/           sky, clouds, hills, seasons, saves, menus, sky tuning panel
  world/             path network, block partition, placement, collision, facades, roofs, trees, chunk streaming
assets/shaders/      tree sway, facade UV, cirrus (WESL)
vendor/bevy_elements vendored dependency with three cloud shader patches
.agents/             ADRs and the development plan (Chinese)
examples/            offline diagnostics and contact sheets
tests/               asset chain, sky screenshot, top view
```

## Design docs

Everything about *why* it is shaped this way lives in `.agents/` (in Chinese):

| Doc | Contents |
|---|---|
| `adr-0001-game-design.md` | Core experience, visual style, world and save constraints |
| `adr-0002-world-generation.md` | Two-stage generation, perturbed network, chunk streaming |
| `adr-0003-sky-system.md` | Atmospheric scattering choice, clouds and celestial trade-offs |
| `adr-0004-placement-collision-seasons.md` | Placement, negative-space collision, season tables |
| `adr-0005-engine-migration.md` | Godot → Bevy mapping and the options that were rejected |
| `dev-plan.md` | Staged checkbox plan with per-round measurement notes |

## Status

| Stage | Scope | State |
|---|---|---|
| R0 | Skeleton, pinned dependencies, tree asset chain | Done |
| R1 | Sky (atmosphere, dusk, clouds, stars and moon) | Done |
| R2 | Path network and chunk streaming | Done |
| R3 | Placement, collision, seasons, procedural trees, facades, roofs, street furniture | Done; some look-and-feel awaits a human pass |
| R4 | Saves and menus | Done |
| R5 | Sound and polish | Not started (the sky panel and cloud kinds landed early, in R5.1–5.5) |

## Known gaps

- `tests/assets_load.rs` needs the three glTF files under `assets/models/`, which are not tracked. In a fresh clone that one test fails on a missing file (the assertion names the file). Procedural trees do not use them, so the game itself runs fine.
- There is no `LICENSE` file yet, though `Cargo.toml` declares `MIT OR Apache-2.0`.
- The growth profiles (columnar aspen / street tree / broadleaf) and the three leaf masks (poplar / sophora / willow) are not fully aligned one-to-one yet: they read as distinct species, but the naming is not consistent.
- Cloud edges still read as blocky at a 32–64 m scale; the cause is noise resolution and it is a known boundary.
- Sound is entirely unimplemented. The current build is silent.
