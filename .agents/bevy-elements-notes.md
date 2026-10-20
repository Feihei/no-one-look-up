# bevy_elements 0.5.3 源码阅读笔记

> 目的：这是一个 "vibe coded"（AI 辅助编写）crate，引入前通读关键模块源码确认真实行为。
> 源码位置（只读副本）：`.box-agent-scratch/3310a92473bb44cca9905c080b106f57/bevy_elements/src/`
> 标注约定：**[已确认]** = 直接读到源码；**[未确认]** = 基于源码/文档注释推断，未逐行验证。
> 阅读范围：`plugin.rs`、`celestial/clock.rs`、`celestial/mod.rs`、`climate/weather.rs`、`climate/season.rs`、`climate/mod.rs`、`climate/model.rs`、`clouds/mod.rs`、`clouds/model.rs`（部分）、`fog/mod.rs`、`lib.rs`。

---

## 1. 模块概览

| 模块 | 用途 | 关键类型 |
|---|---|---|
| `plugin` | 插件组、SystemSet、相机、画质 | `ElementsPlugins`、`ElementsSystems`、`ElementsCamera`、`ElementsQuality` |
| `celestial` | 时钟、地理位置、日月、曝光 | `WorldClock`、`CivilDate`、`GeoLocation`、`SunState`、`Sun`、`Moon`、`CelestialPlugin` |
| `climate` | 气候模型、天气调度、季节 | `Climate`、`Weather`、`WeatherDirector`、`WeatherPreset`、`Season`、`SeasonState`、`ClimatePlugin` |
| `clouds` | 体积云（GPU compute 管线） | `CloudPlugin`、`CloudLayer`、`CloudQuality`、`CloudShadows`、`CloudWeatherMap` |
| `fog` | 距离雾 + 相机跟随地面雾体积 | `FogPlugin`、`GroundFog`、常量 `KOSCHMIEDER`/`CLEAR_VISIBILITY_M` 等 |
| 其他 | 风/降水/闪电/尘土/地形/水文/地被/风化 | `WindPlugin`、`PrecipitationPlugin`、`LightningPlugin`、`DustPlugin`、`TerrainPlugin`、`HydrologyPlugin`、`WaterPlugin`、`GroundCoverPlugin`、`WeatheredPlugin` |
| `query` | 游戏玩法只读查询 | `Environment`、`EnvironmentSnapshot` |

**[已确认]** `lib.rs` 公开导出面：`ElementsPlugins`/`ElementsCamera`/`ElementsQuality`/`ElementsSystems`、`ElementsReadback`、`Environment(Snapshot)`、celestial 全套（含 `astro` 星历模块）、climate 全套、`WindPlugin` 系列、`PrecipitationPlugin` 系列、fog 全套（含 `distance_fog_falloff`、`sky_air_extinction` 等函数）、clouds 四件套、lightning、terrain、dust、hydrology、cover、weathered。`ElementsView`、`ensure_gpu_core` 等为 crate 内部，不公开。`#![forbid(unsafe_code)]`。

---

## 2. ElementsPlugins：默认注册的子插件（能否裁剪）

**[已确认]** `ElementsPlugins::build()` 默认注册顺序：

```text
ClimatePlugin, CelestialPlugin, WindPlugin, TerrainPlugin, HydrologyPlugin,
WaterPlugin, GroundCoverPlugin, DustPlugin, FogPlugin, CloudPlugin,
LightningPlugin, PrecipitationPlugin, WeatheredPlugin
// + VoxelPlugin（仅 isomesh feature）
```

**[已确认]** 文档注释明确说明每个成员插件都是 public 的，游戏可以按需去掉某个；官方给出的裁剪方式是：

```rust
ElementsPlugins.build().disable::<FogPlugin>()
```

即用 Bevy `PluginGroupBuilder::disable::<T>()`，无需 fork。

**结论（适配本项目）**：可以裁剪。本项目不编译雨雪/雷电/风场调度，可 `disable::<WindPlugin>()`、`disable::<LightningPlugin>()`、`disable::<PrecipitationPlugin>()`、`disable::<DustPlugin>()`，以及按需 `disable::<HydrologyPlugin>()`、`disable::<WaterPlugin>()`、`disable::<GroundCoverPlugin>()`、`disable::<WeatheredPlugin>()`。**注意 `ClimatePlugin` 仍需保留**（它持有 `WorldClock` 调度与 `Weather`/`WeatherDirector` 资源），但必须配合 `WeatherDirector::force(WeatherPreset::Clear, ...)` 锁死天气（见 §5）。

**[已确认]** `ElementsSystems`（Update 内 SystemSet 链）：`Clock → Weather → Celestial → Simulation → Readback`。云、雾、风的模拟用 `Time<Virtual>`（真实 delta），**世界时钟的 time_scale 不影响云漂移/风偏移的推进速度**——云的漂移按真实秒计算（`DRIFT_WIND_FACTOR · wind · real_delta`）。对本项目（time_scale=0）意味着：云仍会按真实时间缓慢漂移，不会因暂停世界而冻结；如需完全静止的云，需自行覆盖 `Weather.wind_speed_ms = 0` 或接受缓慢漂移。

### ElementsCamera

**[已确认]** `pub struct ElementsCamera { pub auto_exposure: bool }`，默认 `auto_exposure: true`；`#[require(Hdr)]`。spawn 时插入：`AtmosphereSettings`、`AtmosphereEnvironmentMapLight`、`DepthPrepass`、`VolumetricFog`（`ambient_intensity: 0.0`，`step_count` 来自 `quality.fog_steps()`）、`DistanceFog`、`Exposure::SUNLIGHT`、`AmbientLight { brightness: 0.0 }`。

**[已确认] 默认强制 `Msaa::Off`**：云 pass 读取非多重采样的 depth prepass。如需 MSAA，须在 spawn 后自行插入 `Msaa` 组件覆盖（风险自担）。

**[已确认]** 无 `ElementsCamera` 时：风/雾/云模拟 idle，但 clock、weather 调度、query 仍继续。存在多个 `ElementsCamera` 时跟随 `Camera::order` 最低者并 warn 一次。

### ElementsQuality

**[已确认]** 枚举 `Low / Medium / High(#[default]) / Ultra`。`High` 档云参数：`block=2, steps=64, light_samples=6, window=4`。

---

## 3. WorldClock：时间暂停的准确 API

**[已确认]** `WorldClock` 是 Resource，字段：

```rust
pub struct WorldClock {
    pub date: CivilDate,
    pub hour: f64,
    pub time_scale: f32,   // 默认 60：每真实秒推进 60 游戏秒
    pub paused: bool,
}
```

- **[已确认]** 每帧 `advance_clock` 按 `Time<Virtual>` delta × `time_scale` / 3600 推进 `hour`；`paused == true` 时直接 `return`，时间完全停止。
- **[已确认]** `time_scale = 0.0` 同样等效暂停（`hours == 0.0` 时不推进）。两种方式都可行；`paused = true` 语义更明确。

**[已确认]** 构造与 builder：

```rust
WorldClock::new(date: CivilDate)          // 正午 12.0、time_scale 60、运行中
    .at_hour(h: f64)                      // 设定小时；超出 [0,24) 会翻动日期
    .with_time_scale(scale: f32)          // 设定 time_scale
```

**本项目适配（17:30 黄昏 + 暂停）**：

```rust
WorldClock::new(CivilDate::new(y, m, d)).at_hour(17.5).with_time_scale(0.0)
// 或 spawn 后：clock.paused = true;
```

**[未确认]** `GeoLocation { latitude_deg: 45.0, longitude_deg: 0.0, utc_offset_hours: 0 }` 为默认值（读到默认字段）。北京需 `latitude_deg ≈ 39.9, longitude_deg ≈ 116.4, utc_offset_hours: 8`——具体字段写法以源码为准，此处为推断。
---

## 4. Celestial：Sun / Moon / SunState / 曝光

**[已确认]** `SunState`（Resource，每帧由星历更新）：`direction`、`altitude_deg`、`azimuth_deg`、`moon_direction`、`moon_altitude_deg`、`moon_phase`（0 新月 / 0.5 满月）、`moon_illuminated_fraction`；默认太阳在头顶。

**[已确认]** `CelestialPlugin`：Startup spawn 一个 `Sun`（illuminance = `lux::RAW_SUNLIGHT`、shadow enabled、`VolumetricLight`、shadow cascade max 600 m）+ 一个 `Moon`（illuminance 0、shadow disabled、`MOON_TINT (0.88, 0.92, 1.0)`、`SunDisk::OFF`）。Update 链：`update_sky → update_ambient → auto_exposure`。

**[已确认] 关键常量（celestial/mod.rs）**：

- `SUN_OFF_DEG = -2.0`：太阳低于 -2° 天穹光淡出。
- `MOON_OFF_DEG = -0.5`；`MOON_SHADOWS_BELOW_SUN_DEG = -6.0`（太阳低于 -6° 才交给月亮投影）。
- `DUSK_TINT = (1.0, 0.55, 0.3)`；`DUSK_ONSET_DEG = 10.0`：太阳低于 10° 开始向日落橙 lerp，约 -6° 淡出。
- `CIVIL_TWILIGHT_DEG = 6.0`。
- `OVERCAST_DIFFUSE = 0.25`（Kasten & Czeplak：`total = 0.25 + 0.75·T̄`，T̄ = beam_transmittance）。
- `beam_transmittance(cover) = 1 − cover^3.4`。
- `EXPOSURE_TAU_S = 1.5 s`：自动曝光的时间常数。
- `sky_gain(&sun, cloud)`：白天恢复天穹亮度；`dusk_tint(sun_altitude_deg)` 为 pub(crate) 函数，被 fog 复用。

**关于"云 + 星 + 月齐备"**：**[已确认]** Sun 与 Moon 两个实体由 CelestialPlugin 默认 spawn，月亮照度由 `moon_lux`/`moon_phase` 驱动，云的照明在太阳低于 handover 高度时切换为月光（见 §6 `pick_light`）。**星星未在本次阅读范围中找到专门模块/组件——[未确认]**，可能内置于 Bevy Atmosphere 或 crate 的 sky shader，需引入后运行验证。

---

## 5. Climate / Weather / WeatherDirector / Season

### WeatherPreset（10 个）

**[已确认]** `Clear(#[default]), PartlyCloudy, Overcast, Fog, Drizzle, Rain, Thunderstorm, Snow, Blizzard, DustStorm`。
`Clear` 目标值：`cover = 0.05, mm_h = 0, rh = 0.50, wind = 3.0, gust = 0.20, visibility = 40_000, storm = 0, dust = 0, delta_t = 2.0`。

### Weather resource

**[已确认]** 每帧由 `WeatherDirector` 重算：`cloud_cover, precipitation_mm_h, temperature_c(默认 15), relative_humidity, wind_speed_ms, wind_direction(Vec2), gustiness, visibility_m, storminess, dust`。
**[已确认]** `snow_fraction()`：≤ −1°C 为 1，≥ 2°C 为 0，之间线性。

### WeatherDirector：强制晴朗的准确 API

**[已确认]** `WeatherDirector` 是 Resource（保存天气状态）：

```rust
director.force(preset: WeatherPreset, transition: Duration)  // 覆盖 scheduler，直到 release
director.release(transition: Duration)
director.current() -> WeatherPreset
director.is_forced() -> bool
```

- **强制晴朗：`director.force(WeatherPreset::Clear, Duration::from_secs(0))`**（transition 时长为游戏时间，可为 0 立即生效——Duration 单位/即刻语义 [未确认]）。
- force 期间 scheduler 继续步进但被覆盖；release 后混合到当时未强制的 preset。
- 强制 `Snow`/`Blizzard` 时温度封顶 −2°C。
- 天气切换发 `WeatherChanged { from, to, forced }` message。

**[未确认]** scheduler 的自动天气节奏细节（湿时概率、风暴 share 等来自 `Climate`），本项目 force Clear 后不依赖它。

### Climate（默认 `Climate::temperate()`）

**[已确认]** 字段：`mean_temperature_c 11, annual_amplitude_c 9, diurnal_amplitude_c 5（峰值 15:00）, warmest_day 200, mean_humidity 0.75, wet_hour_probability 0.35, storm_share 0.05, fog_share 0.08, dust_share 0, prevailing_wind Vec2::X, mean_wind_ms 4, reference_elevation_m 0, seed 0`。
方法：`with_seed(u64)`、`base_temperature_c(doy, hour, southern)`、`lapse_c(elevation)`；`LAPSE_RATE_C_PER_M = -0.0065`。另有 `alpine / desert / tropical / tundra` 预设。

### Season

**[已确认]** 枚举 `Spring / Summer(#[default]) / Autumn / Winter`（北半球 3-5/6-8/9-11/12-2 月），`Season::of_month(month, southern: bool)`。
`SeasonState`：`season, year_fraction, autumn, leaf_fall, blossom`（0-1 smoothstep 曲线；knots：autumn 9/1→10/15，leaf_fall 10/15→11/30，blossom 4/1→4/20→5/10）。切换发 `SeasonChanged` message。

**[已确认]** `ClimatePlugin`：init `Climate / WeatherDirector / SeasonState / SurfaceWeathering`，注册 `WeatherChanged / SeasonChanged` messages；Update 链 `update_season → run_weather → update_weathering`。

---

## 6. Clouds：CloudLayer / 质量 / 天气图 / 阴影

### CloudLayer 与低云量的设置方式

**[已确认]** `CloudLayer` 是 Resource，每帧由 `update_layer` 系统从 `Weather` 派生（`CloudLayer::from_weather(&weather)`），字段 `base_m`（默认 1500，范围 600–1500）、`top_m`（默认 3300）。

**[已确认] 低云触发逻辑（from_weather）**：

```text
low = max(
    smoothstep(0.9, 0.98, rh)  · (1 − smoothstep(0.1, 0.5, rain)),
    smoothstep(0.1, 0.3, rain) · (1 − smoothstep(0.8, 1.5, rain)),
)
base_m = 1500 + (600 − 1500) · low
kind  = 0 stratus / 0.5 cumulus / 1 cumulonimbus（kind_from_weather）
```

**关键适配点（必须如实记录）**：低云量由 `Weather` 的 RH/降水**派生**，`CloudLayer` 本身**没有公开的"设置低云量"API**；`from_weather` 的文档注明该 resource 对其他系统 read-only。本项目"晴朗低云量"需求的可行路径：

1. **[已确认]** 晴朗 `Clear`（rh=0.50、rain=0）→ `low = 0` → `base_m = 1500 m`、厚 1800 m cumulus，**不会自动压低云底**。
2. **[已确认]** 覆盖入口是 `Weather` 字段：`WeatherDirector::force` 无法直接改 `Weather`（每帧被重算覆盖）。若要人工压低云底，需要在 `ClimatePlugin` 的 `update_weather` 系统之后（`ElementsSystems::Weather` 集之后、`ElementsSystems::Simulation` 之前）加自定义系统改写 `Weather.relative_humidity` 或直接改写 `CloudLayer`，否则每帧被 `update_layer` 覆盖。**[未确认]** 具体插入 SystemSet 的写法需引入后验证。

### CloudQuality / GPU 管线

**[已确认]** `CloudQuality::of(ElementsQuality)` 提供 `block / steps / light_samples / window`； marched/histogram/resolve 纹理按跟随相机的物理分辨率重建。
**[已确认]** 云的照明：`pick_light`——太阳高于 `SUN_HANDOVER_DEG = -4°` 时用太阳（亮度随 smoothstep 淡入），低于则切月光（`MOON_TINT · moon_lux`）。
**[已确认]** 云的天空漫射：`sky_share = 1 + (0.15 − 1) · smoothstep(−2°, 10°, altitude)`——黄昏时几乎全漫射，正午 15%。

### CloudWeatherMap / CloudShadows

**[已确认]** `CloudWeatherMap`：512² rgba8unorm，覆盖相机周围 32 km，R 通道覆盖、G 降水、B 云型；`uv = (world_xz − origin) / extent`，bilinear 重复采样。
**[已确认]** `CloudShadows`：102² 阴影图覆盖 8 km，`camera_transmittance` 初始 1.0，读回后按 `CAMERA_TRANSMITTANCE_TAU_S` 平滑；无读回时用 `beam_transmittance(cover) = 1 − cover^3.4`。
**[已确认]** `CloudShadowSnapshot`：CPU 侧阴影查询网格，每 31.25 m 一个 cell，`transmittance_at(position)` 沿光方向投影采样，图外返回 1。
---

## 7. Fog：FogPlugin / GroundFog / 可见度与消光

**[已确认] 公开常量（fog/mod.rs）**：

- `KOSCHMIEDER: f32 = 3.912`（Koschmieder 2% 对比阈值）。
- `CLEAR_VISIBILITY_M: f32 = 40_000.0`：Clear preset 的可见度；所有天气可见度以它为基准（非真空）。
- `CLEAR_AIR_EXTINCTION = KOSCHMIEDER / CLEAR_VISIBILITY_M`：所有天气可见度的下限（空气消光）。
- `GROUND_FOG_EXTENT = Vec3(240, 120, 240)`（体积保持小，因 Bevy 按包围球半径衰减 fog 采样光照）。
- `GROUND_FOG_TEXELS = UVec3(128, 32, 128)`。

**[已确认] 公开函数**：

```rust
extinction_for_visibility(visibility_m) -> f32       // KOSCHMIEDER / max(v, 1)
sky_air_extinction(fog_extinction_m) -> f32          // (extinction − CLEAR_AIR_EXTINCTION).max(0)
distance_fog_falloff(visibility_m) -> FogFalloff     // Atmospheric，extinction == inscattering
fog_hue(sun_altitude_deg, dust) -> Vec3              // 雾色，随 dusk_tint/尘土着色
volume_light_compensation(visibility, scale_h, cam_h, sun_alt, extent) -> f32
```

**[已确认] 行为要点**：

- `sky_air_extinction`：天穹只添加超出 Clear 空气的过量消光（Bevy Atmosphere 已画 clear air）；≥ 40 km 时为 0，晴空天穹不被雾污染。
- 距离雾写在每个 `ElementsCamera` 上，颜色 = `fog_hue × scene_illuminance/π × exposure`，与天空/网格保持同一套雾。
- 地面雾 `GroundFog`（marker component）：Startup spawn 的相机跟随 `FogVolume`，密度由 compute kernel 每帧写入；`FogVolume` 初始 `density_factor: 0.0`、`absorption = 1 − 0.95`、`scattering = 0.95`。
- `FogPreset` 时 scale height 25 m，否则 haze 60 m，过渡 τ = 600 游戏秒（游戏时间）。
- 体积光照补偿 `volume_light_compensation`：替换 Bevy 的 `exp(−σR)` 包围球衰减为沿太阳的雾层光学深度，clamp 1–100。
- `volume_density_share` / `volume_light_gain`：Bevy 无法照亮的体积不遮挡（避免 Blizzard 时天空变黑洞），能照亮的按场景 total 光散射。
- 系统排序：`update_volume.after(CloudGpuUpdate)`、`update_distance_fog.before(CloudGpuUpdate)`（雾色先写，避免天空滞后一帧）。

**本项目适配**：晴朗 `Clear` 时 `visibility = 40 km` → `sky_air_extinction = 0`、`volume_density_share = 1.0`（beam 全光），地面雾基本透明——无需额外处理；`FogPlugin` 保留即可。

---

## 8. 与本项目用法的适配总结

项目目标：锁定 17:30 黄昏、晴朗低云量、time_scale=0、云+星+月齐备、不编译雨雪/雷电/风场调度。

1. **时钟与黄昏**：`WorldClock::new(CivilDate::new(y,m,d)).at_hour(17.5).with_time_scale(0.0)`；日落色调由 `dusk_tint` 在太阳高度 < 10° 时启动（DUSK_ONSET_DEG = 10.0）——17:30 的太阳高度取决于日期 + 纬度，**[未确认]** 北京纬度 39.9°N 下 17:30 是否落在 < 10° 区间，需运行验证或先算星历。
2. **暂停时间**：`clock.paused = true` 或 `time_scale = 0.0`，两者均已确认等效暂停小时推进；但云漂移/风偏移按真实时间推进（`Time<Virtual>`），如需完全静止可把 `Weather.wind_speed_ms` 压 0（需自定义系统，见 §6）。
3. **晴朗**：`weather_director.force(WeatherPreset::Clear, Duration)` 锁死调度，避免 `ClimatePlugin` 的 scheduler 自动变化。
4. **低云量**：无直接 API；晴朗下 base = 1500 m。若要压低云底需自定义系统在每帧派生后改写 `Weather`/`CloudLayer`（见 §6 适配点）。
5. **云 + 星 + 月**：云（CloudPlugin 默认注册）、月亮（CelestialPlugin 默认 spawn）已确认；星星未检到专门模块——[未确认]，引入后验证。
6. **裁剪**：`ElementsPlugins.build().disable::<WindPlugin>().disable::<LightningPlugin>().disable::<PrecipitationPlugin>().disable::<DustPlugin>()`（已确认 disable 机制）；保留 ClimatePlugin + force Clear。注意：即使裁掉 PrecipitationPlugin，fog 的降水 shaft 逻辑在 `update_volume` 中仍读 `Weather.precipitation_mm_h`（晴朗为 0，安全——[已确认] 读的是 Weather 字段，不是 precipitation 状态资源）。
7. **MSAA**：ElementsCamera 强制 `Msaa::Off`；如需 MSAA 自行覆盖并自担风险。
8. **画质**：默认 `ElementsQuality::High`（steps=64），性能敏感可降 Medium/Low。

## 9. 风险与待验证项

- **[未确认]** 星星（stars）的实现位置与开关方式。
- **[未确认]** 北京坐标下 17:30 的太阳高度是否触发 dusk_tint（< 10°）；可先用 `celestial::astro` 星历函数预计算。
- **[未确认]** `force(preset, Duration)` 传 `Duration::ZERO` 是否立即生效（transition 以游戏时间计）。
- **[未确认]** 自定义改写 `CloudLayer`/`Weather` 的系统插入点（SystemSet 边界）的实际写法。
- 云、雾均依赖 GPU compute 管线（storage texture / readback），wgpu 环境下需实机验证。

## R0 补充：headless 测试环境要点（2026-10-20，已实测验证）

R0.5 的 headless 资产加载测试（`tests/assets_load.rs`）踩坑记录，R1.7 截图冒烟测试直接复用：

- **纯资产加载也必须挂 `RenderPlugin`**：`bevy_image` 只 *pre-register*（reserve）ImageLoader 占位，真正的 loader 由渲染器注册（压缩纹理格式依赖 GPU 能力）。没有 RenderPlugin 时所有资产（含 PNG）永远停在 `Pending` → `loader.get().await` 无限等待，`load_state` 一直 Loading、无任何错误日志。
- **必须手动调 `app.finish()`**：渲染资源（`DeviceErrorHandler`、真实 loader 注册）在 `finish()` 中解包；`app.run()` 会自动调用，测试里手动 `update()` 不会。
- **headless 渲染栈最小插件组合**（无窗口）：`MinimalPlugins + AssetPlugin + RenderPlugin + WindowPlugin{primary_window: None, exit_condition: DontExit, primary_cursor_options: None} + ImagePlugin + MeshPlugin + CameraPlugin + CorePipelinePlugin + WorldSerializationPlugin + GltfPlugin`。缺一会在 extract/系统验证阶段报 "Resource/Message not initialized"（无 debug feature 时系统名被隐藏，临时 `cargo add --dev bevy_ecs --features debug` 可定位）。
- **GltfLoader 需要的资产类型**：`WorldAsset`（WorldSerializationPlugin）、`AnimationClip`、`StandardMaterial`——纯资产加载测试可直接 `app.init_asset::<T>()`，不必挂完整 Animation/Pbr 插件。
- **AssetServer 路径相对 `assets/` 根目录**：`server.load("models/aspen_1.glb")`，不要带 `assets/` 前缀（文件存在性检查用 `assets/models/...`）。
- 本机冒烟：RTX 2070 / Vulkan / driver 610.47，窗口创建与 GPU preprocessing 正常；日志中 RenderDoc JSON 报错是机器上的 Vulkan layer 残留（`E:\Program Files\RenderDoc_1.30_64\renderdoc.json` 缺失），与项目无关。
- 首次并行 `cargo check` 曾在多个 proc-macro crate 上无诊断失败（derive_more/serde_derive/zerofrom 等），单 crate 重试即过，判断为临时故障（疑似杀软/文件锁）；CI 上需留意偶发。

## R1 补充：headless 渲染/截图测试要点（2026-10-20，已实测验证）

R1.7 截图冒烟测试（`tests/sky_screenshot.rs`）踩坑记录，后续截图类测试直接复用：

- **测试帧必须节流（delta 饿死）**：测试循环 600 帧只跑 ~1.5s，`Time<Virtual>` delta≈0，bevy_elements 的时间平滑/读回 cadence 全部饿死，天空渲染成均匀平色（各值与清屏色不同但无内容）。测试循环每帧 `sleep(16ms)` 模拟真实节奏后，完整天空（渐变+太阳盘+云）立即正常。这是 headless 截图测试的第一嫌疑。
- **`Hdr` 相机（ElementsCamera required）+ LDR 图片目标 = 渲染被跳过**：输出停在清屏色 `srgb_u8(43,44,47)`，无任何错误。必须 `Rgba16Float` 目标；断言用曝光不变的比值/分带分析（线性值含曝光标度，阈值放宽）。
- **MinimalPlugins 不含 TransformPlugin**：GlobalTransform 永不传播，破坏 bevy_elements 的相机跟随雾与大气视图（平色帧）。测试插件组合最终清单：
  `MinimalPlugins + TransformPlugin + AssetPlugin + RenderPlugin + WindowPlugin{None} + ImagePlugin + MeshPlugin + CameraPlugin + CorePipelinePlugin` + `PostProcessPlugin + GltfPlugin(先于 PbrPlugin!) + PbrPlugin + LightPlugin + GizmoPlugin + HeadsUpPlugins + LogPlugin`。
- **Bevy 0.20 API 变化**：`RenderTarget` 是独立 Component（不再是 `Camera.target` 字段）；`Camera` 无 `fov` 字段（用 `Projection::Perspective`）；`Exposure { ev100 }` 在 prelude；Bloom 在 `bevy::post_process::bloom`；TextureFormat 在 `bevy::render::render_resource`；PbrPlugin 非 unit struct（用 `::default()`）。
- **黄昏取景**：pitch 50° + 默认 fov 45° → 画面覆盖 27°–72° 仰角，完全看不到地平线带（易误判"天空没渲染"）。验证/截图应对准太阳方向或 pitch ≤ 15°。
- **太阳高度调参经验**：北京纬度 + 3 月 21 日，经度每 +1° → 17:30 太阳高度约 −0.78°。116.4°→10.06°、121.8°→5.95°、125.0°→3.5°（dusk_tint 线性：alt/10° 即因子，10° 起始、0° 全橙、−6° 淡出）。
- **月亮可见性**：date 决定月相与太阳高度（固定时刻下唯一自由参数）。Mar 21 = 2 日细月牙（月亮 36° 高但很淡）；月亮形态调校 = 换 date，会联动太阳高度。
- **黄昏视觉现状（估 60% 达标）**：太阳盘+光晕、暖地平线带、云剪影暖橙镶边、顶部冷蓝渐变、无过曝；短板：橙红中间色阶缺失、天顶偏灰蓝缺紫调、云暗部过暗。待 Feihei 人眼迭代（候选：更低太阳 / Bloom 阈值 / ScatteringMedium 调色 / FullscreenMaterial 黄昏分级 pass）。
