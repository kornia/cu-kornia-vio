# cu-kornia-vio

[Copper](https://github.com/copper-project/copper-rs) tasks wrapping
[kornia-slam](https://github.com/kornia/kornia-slam) stereo visual-inertial odometry.

A stereo source feeds `StereoVio`, which runs in the background and publishes one camera pose per
solved frame. IMU samples reach it through a shared resource bundle, so the tracker keeps a single
input edge and can run with `background: true`.

## Crates

- **`cu-kornia-vio`**: the tasks (`StereoVio`, `ImuFeed`, `WheelFeed`, `MountFeed`,
  `VioTapSource`) and the `VioBus` resource bundle.
- **`cu-stereo-payloads`**: the wire types alone. It depends on cu29, the upstream copper
  payload crates, bincode and serde and nothing else, so a stereo driver can name the types it
  produces without depending on `cu-kornia-vio` and, through it, on kornia-slam. That is the whole
  reason it is a separate crate rather than a module.

Inside `cu-kornia-vio` there are two layers:

- **`track`**: the composition root over kornia-slam. kornia-slam ships stereo matching, the
  map, map-projection PnP and bundle adjustment; this is the orchestration over them, written for
  a live camera. It uses no copper runtime or graph types (no `CuMsg`, no tasks), only the cu29
  unit types, `CuTime` and the logging macros, so the estimation is testable without a graph.
- **`task`**: the `CuTask` shell. It owns only the payload-to-image conversion and the
  world-to-camera inversion and adds no estimation logic of its own. Anything it computed would be
  a second implementation to keep in step with `track`.

## Payload types

All from `cu-stereo-payloads`. Time rides on the message `tov`, never in the payload.

| Type | Carries | `tov` |
|---|---|---|
| `StereoPair` | two `CuImage` eyes (`CuHandle`-backed buffers) plus `RectifiedStereo`: intrinsics and baseline of the rectified pair | capture time of the pair |
| `ImuBatch<N>` | up to `N` `ImuSample`s (an upstream `cu_sensor_payloads::ImuPayload` plus its own capture time), oldest first, a cumulative `dropped` counter and a `camera_aligned` flag | `ImuBatch::tov()`, a `Tov::Range` over the samples |
| `VioPose` | a `cu_spatial_payloads::Pose<f64>` (camera in world) plus `VioStatus`: keyframe flag, reset epoch, landmark count, `localization` (`Tracking`, `Coasting`, `NotLocalized`); `covariance`, the pose's 6x6 covariance as the world left perturbation of the camera pose (`-1` in `[0][0]` for none; see `cu_kornia_vio::covariance`); and `map_points`, an optional map snapshot, `Option<CuHandle<Vec<Landmark>>>`. Without wheel odometry an untracked frame carries no payload | the `tov` of the frame it was computed from |
| `WheelOdom` | a planar wheel-odometry reading: the CUMULATIVE base pose `x`, `y`, `theta` in the odometry frame (REP-103) and the producer's row-major 3x3 `covariance` | capture time of the reading |
| `CameraMount` | `cam_in_base: Pose<f64>`, the rectified left camera's pose in the base frame, for a camera on a moving head | time the mount is valid from |
| `Landmark` | one map point: world-frame `position: [f32; 3]` in metres and its map `index: u32` | element of a `VioPose` snapshot, no time of its own |

`Landmark::index` is a `u32`, not a float in a fourth column: an `f32` is exact only to 2^24,
past which two points would share an index.

The eye geometry rides on `StereoPair`, so `StereoVio` is never told what the source already
knows: the tracker is built on the first frame, and every later frame must match its size.

## Using it

Neither crate is on crates.io yet (kornia-slam is still consumed from git), so depend on the repo.
Name both crates with the identical spec string; see [the spec-string rule](#the-spec-string-rule).

```toml
[dependencies]
cu-kornia-vio = { git = "https://github.com/kornia/cu-kornia-vio", branch = "main" }
cu-stereo-payloads = { git = "https://github.com/kornia/cu-kornia-vio", branch = "main" }
cu29 = { git = "https://github.com/copper-project/copper-rs", rev = "fe2061dc10539868334f6ded55a9e75feb0f2b62" }
```

A stereo driver that only produces frames needs `cu-stereo-payloads` alone. It must fill a
`StereoPair` with rectified GRAY8 eyes and stamp the message `tov` with the capture time; an
untimed frame (`Tov::None`) is refused.

## Wiring

```ron
resources: [ ( id: "bus", provider: "cu_kornia_vio::VioBus" ) ],
tasks: [
    ( id: "imu_feed", type: "cu_kornia_vio::ImuFeed<32>", resources: { "imu": "bus.imu" } ),
    ( id: "vio", type: "cu_kornia_vio::StereoVio", background: true,
      config: { "on_tracking_loss": "reset_map", "max_keyframes": 120,
                "orb_keypoints": 400, "pnp_lm_iterations": 5 },
      resources: { "imu": "bus.imu", "epoch": "bus.reset_epoch" } ),
],
cnx: [
    ( src: "stereo", dst: "vio", msg: "cu_stereo_payloads::StereoPair" ),
    ( src: "imu", dst: "imu_feed", msg: "cu_stereo_payloads::ImuBatch<32>" ),
    ( src: "vio", dst: "poses", msg: "cu_stereo_payloads::VioPose" ),
],
```

- **`VioBus`** is a resource bundle with four members: `imu`, a bounded queue of IMU samples,
  `reset_epoch`, a counter a caller bumps with `ResetEpoch::request` to drop the map on the next
  frame, `wheel`, a bounded queue of wheel readings and camera mounts, and `tap`, the latest
  frame's diagnostics for `VioTapSource` (below). Each bus instance is independent, so two
  estimators in one graph need two buses.
- **`ImuFeed<N>`** is an inline sink that pushes each `ImuBatch<N>` into the bus queue. Its `N`
  must match the source's batch capacity. The queue is armed only when `StereoVio` has an
  `inertial` config block; without one the push is a single atomic load.
- **`StereoVio`** takes a `StereoPair` and emits a `VioPose`, draining the bus queue once per
  frame. Unknown config keys are refused at startup rather than ignored. The recognised keys are
  `on_tracking_loss` (`keep_map`, the default, or `reset_map` for a long-running graph),
  `max_keyframes`, `at_keyframe_bound` (`drop_map`, the default, or `evict_oldest`), `orb_keypoints`, `search_radius_px`, `max_covisible_keyframes`,
  `pnp_lm_iterations`, `inertial`, `landmark_buffers`, `wheel_odom`, `per_frame_fusion`,
  `keyframe_policy` (insertion thresholds, parallax, a time bound), `keyframe_culling`,
  `stationary_edges` and `motion_veto` (on when absent; see Motion veto);
  `cu_kornia_vio::config::CONFIG_KEYS` lists them. Its `Freezable` is a no-op (the kornia-slam
  map has no serialised form), so a resim of a `background: true` graph reproduces it only from
  the start of a log; see Known limitations.
- **`VioTapSource`** is an inline source that emits what the tracker saw and concluded on each
  processed frame: `VioDiagnostics` (tracking state, failure reason, inertial stage and biases,
  the motion check, the fusion verdict, pose sigmas) and `VioFrame` (the solved `StereoPair`, a
  refcount on the camera's buffers, plus its classified keypoints). It is the only place a lost,
  recovering or vision-rejected frame shows up, because an untracked frame carries no `VioPose`.
  Bind `"tap": "bus.tap"` on both `StereoVio` and the source; the tap is inert (one atomic load
  per frame) until a source arms it. Its one config key, `frames` (default `true`), set to
  `false` emits the diagnostics alone and retains no camera buffer.

```ron
( id: "vio_tap", type: "cu_kornia_vio::VioTapSource", config: { "frames": false },
  resources: { "tap": "bus.tap" } ),
// cnx: ( src: "vio_tap", dst: "status", msg: "cu_stereo_payloads::VioDiagnostics" ),
//      ( src: "vio_tap", dst: "viewer", msg: "cu_stereo_payloads::VioFrame" ),
```

Why the IMU does not ride a second input: `CuAsyncTask`, which `background: true` wraps a task
in, accepts a single input message. And while a solve is running the background task refuses the
arriving input, so an IMU batch bundled into the frame payload would be dropped along with every
refused frame. Measured on a Jetson Orin with an OAK-D at 640x400 and 15 fps
(`orb_keypoints: 400`, `pnp_lm_iterations: 5`), tracking takes p50 53 ms / p99 91 ms, and a
frame that inserts a keyframe p50 93 ms, against a 66 ms frame interval, so poses come out at
about 10.8 Hz and about 28 % of frames are refused. Those two keys are what the Wiring snippet
sets; without them the tracker runs 1000 ORB keypoints (`TrackerConfig::new`) and kornia-slam's
50 PnP LM iterations, which are markedly slower and refuse more.

`examples/stereo_vio.ron` is a complete graph on synthetic input:

```sh
cargo run --release --example stereo_vio -- [ITERATIONS] [LOG_PATH]
cargo run --release --example stereo_vio_logreader -- <LOG_PATH> extract-copperlists
```

The second command dumps each copperlist as JSON: the `stereo` frame and the `vio` pose, each with
its `tov`. Because the tracker runs in the background, a pose lands one or more copperlists after
the frame it was computed from, and its `tov` is that frame's `tov`.

### Map points

`StereoVio` attaches a map snapshot, `VioPose::map_points`, to the pose of every frame that
inserted a keyframe, and to no other: a keyframe is when points are created, culled, fused and
moved by local bundle adjustment, while a plain tracking frame only reads the map.

A snapshot is a bounded, newest-window view, not the whole map: the live points among the newest
`MAX_LANDMARKS` (8192) map slots, refreshed on every keyframe, so its cost does not grow with the
map. Each snapshot is complete for that window, and a consumer REPLACES what it displays with
it. Culled and fused points therefore drop out, and after a `VioStatus::reset_epoch` change the
next snapshot describes the new world whether the map was rebuilt or kept. `Landmark::index` is
the point's map index, stable within an epoch: use it to correlate a point across snapshots, not
to accumulate them. `VioStatus::landmarks` counts every live point, so it can exceed a
snapshot's length once the map outgrows the window.

The buffers come from a `CuHostMemoryPool` owned by the task, `landmark_buffers` of them
(default 16, at most 256, 128 KiB each, all allocated at startup). A snapshot stays checked out
while anything holds its handle; when none is free the pose is published without one, and the
stop line counts how often.

## Inertial path

Off by default: without an `inertial` block `StereoVio` runs stereo-only and the IMU queue is
never armed. To enable it, add the block to the `vio` config. Every field is required and none
has a default, because a guessed extrinsic or noise density yields a plausible but wrong map:

```ron
config: {
    "inertial": (
        t_bc_rotation: [/* R_BC, row-major, rectified-left-camera axes into IMU axes */],
        t_bc_translation: [/* lever arm in metres, IMU origin relative to that camera */],
        gyro_noise: 0.0,        // rad/s/sqrt(Hz), must be > 0
        accel_noise: 0.0,       // m/s^2/sqrt(Hz), must be > 0
        gyro_bias_noise: 0.0,   // rad/s^2/sqrt(Hz), must be > 0
        accel_bias_noise: 0.0,  // m/s^3/sqrt(Hz), must be > 0
        rate_hz: 200.0,
    ),
},
```

The doc comments on `InertialRon` in `src/config.rs` explain each field and why no default is
defensible (the type is crate-private, so `cargo doc` does not render them).

### Initializing while parked

The default initializer needs the robot to move (>= 5 cm over >= 10 keyframes): it finds gravity
from how the IMU's integration differs from the visual trajectory. A stereo map is already
metric, so a still window is enough. Add `static_init` inside the `inertial` block:

```ron
"inertial": ( /* ... */, enable_inertial_ba: true,
    static_init: (enabled: true),   // optional: window_s, accel_rms_max, gyro_rms_max,
                                    // gyro_mean_max, gravity_tolerance
),
```

On each keyframe before initialization, the tracker looks for >= `window_s` (1.2 s) of contiguous
IMU edges ending at it in which the accelerometer and gyroscope spreads stay under four times
what the configured noise densities predict, the mean rate stays under a plausible gyro bias,
the wheels (when `wheel_odom` is on) report no net motion and the keyframes agree on the camera
pose. Finding one, it takes gravity from the mean specific force, the gyro bias from the mean
rate and zero velocity, and rotates the map onto gravity exactly as a motion initialization
does, bumping `reset_epoch`. The accelerometer bias across gravity is unobservable at rest and
stays at its prior; along gravity it is corrected by `|f| - g`. Keyframes of the window before
the still one get their finite-difference visual velocity. A window that is not still falls
through to the motion initializer on the same keyframe; one the map refuses is not retried for
`init_retry`. The camera must hold its attitude to 0.25 deg across the window, which is what
keeps a head pan or a slow turn the wheels cannot see out of the gyro bias.

After a static initialization the two refinements wait until the camera has moved 0.2 m or
turned 10 deg, and are then scheduled (5 s, 15 s) from that moment: on parked data they would only
rotate the world again and bump `reset_epoch` for nothing.

A map drop that keeps the world (with wheels, or a loss with `keep_map`) keeps the alignment: the
next map is inertial from its first keyframe, with no new rotation and no extra `reset_epoch`
bump. Its anchor velocity is measured from the last 0.2 s of wheel odometry at the re-bootstrap,
never carried from before the drop; with no wheel velocity available the alignment is dropped
and the inertial path initializes again. A drop that restarts the world at identity runs the
still window (or the motion solve) again.

## Wheel odometry

Off by default, and off means the tracker behaves exactly as without it. For a ground robot,
add a `wheel_odom` block with `enabled: true`, bind the bus's `wheel` on `StereoVio` (the binding
is optional otherwise), and feed it:

```ron
( id: "wheel_feed", type: "cu_kornia_vio::WheelFeed", resources: { "wheel": "bus.wheel" } ),
( id: "mount_feed", type: "cu_kornia_vio::MountFeed", resources: { "wheel": "bus.wheel" } ),
( id: "vio", type: "cu_kornia_vio::StereoVio", background: true,
  config: { "wheel_odom": (
      enabled: true,
      cam_in_base_rotation: [/* camera in base, row-major */],
      cam_in_base_translation: [/* metres */],
      translation_noise: 0.02,  // m per sqrt(m) of path
      rotation_noise: 0.05,     // rad per sqrt(rad) turned
      // optional: z_sigma, tilt_sigma (planar stiffness), huber_chi2, max_gap_s
  ) },
  resources: { "imu": "bus.imu", "epoch": "bus.reset_epoch", "wheel": "bus.wheel" } ),
```

With it on:

- each keyframe pair gets a kornia-slam `WheelOdomFactor` in local BA (an interval whose readings
  have not arrived yet waits for a later keyframe). With the inertial path on, the factors join
  the inertial solve once it is applied; before that, local BA stays visual;
- the per-frame prior is the camera motion the wheels measured, with the gyro's rotation when
  the inertial path is on, instead of constant velocity;
- while vision is lost the pose keeps moving on that prior and is published as `Coasting`, or as
  `NotLocalized` when no reading covers the frame;
- a map drop (`max_keyframes`, or a loss with `reset_map`) seeds the next map at the pose
  propagated on the wheels, so the world frame stays continuous. `reset_epoch` still bumps.

`MountFeed` is only needed when the mount moves (a tilting head); the config's `cam_in_base_*`
is used until the first `CameraMount` arrives.

### Parked keyframes

`wheel_odom.parked_keyframe_interval_s` withholds keyframes while the wheels and the camera are
still, but forces one per interval, joined to the last by an IMU edge. Culling merges IMU edges
only up to `keyframe_culling.max_merged_imu_span_s`, so a robot parked for long enough runs out of
mergeable keyframes and the map is dropped at `max_keyframes` anyway (every ~20 min at 5 s / 11 s
on the robot). `stationary_edges` removes both limits once the inertial path is settled:

```ron
"stationary_edges": (
    enabled: true,              // default false; needs wheel_odom and inertial with
                                // enable_inertial_ba, refused at startup otherwise
    // optional: translation_sigma_m (0.002), rotation_sigma_rad (0.002),
    // velocity_sigma_m_s (0.01), huber_chi2 (21.03), min_span_s (1.0)
),
```

- From each keyframe the tracker verifies stillness as samples arrive: IMU chunks of at least
  0.5 s are checked against the static initialization's spread and mean-rate checks, with the
  gyro spread ceiling raised to 0.07 rad/s (the robot's own 38 Hz head vibration reaches 0.048
  over 0.5 s while parked, against the initialization's 0.0118), and two failing chunks in a row
  end it (one is a knock or footsteps, left out of the gyro mean); the
  wheel path stays under 5 mm / 0.01 rad per 5 s window (the old interval gate's floors) and the
  net displacement since the keyframe under the same. An IMU hole or a wheel discontinuity ends
  it; a gap in the wheel readings (the head/body link going stale) is bridged once the IMU has
  verified that stretch.
- While it is open (verified, or the sensors no more than a 2 s stall behind) and the camera
  stays within 1 cm / 0.5 deg of the last keyframe and of the first keyframe of the stationary
  chain, a keyframe the policy wants only on its time bound (`max_frames_between`,
  `max_interval_s`) is withheld for up to 30 s. Tracking-quality triggers (the tracked ratio
  dropping, parallax, an empty reference) still insert, so a changed scene is still mapped; the
  30 s refresh keyframe feeds the stop's gyro mean to VI-BA, so at most 30 s of it is lost when
  the stop ends in motion.
- Such a keyframe is joined to the last one by a kornia-slam `StationaryFactor` (zero relative
  pose, zero velocity, bias continuity, the stop's mean gyro rate keeping the gyro bias observed)
  instead of an IMU edge. Culling composes stationary factors across a removed keyframe with no
  span limit; a keyframe with an IMU edge on one side and a stationary one on the other is culled
  after the chain is composed, its IMU edge moved onto the still neighbour, so a stop never pins
  keyframes.
- The first keyframe after a stop that ended in motion gets an IMU edge over the tail only, from
  0.5 s before the motion was seen, its bias link widened for the elided time. The edge records
  that time, so a culling merge or a BA repropagation keeps the widening.
- While no keyframe comes, the IMU and wheel rings are pruned every frame to 0.5 s before that
  anchor plus a margin, and the per-frame fusion predicts from it.

Before the inertial path is initialized and while a refinement is pending, the parked interval
gate and IMU edges apply as before (the still window and the refinements are built from IMU
edges). `vio_diag`'s `stationary` record counts edges, merges, culled boundary keyframes, tail
edges, withheld keyframes and still runs closed by the IMU and by the wheels (rising while parked
means false closures).

## Per-frame fusion

Off by default. Without it the IMU and the wheels only seed each frame's PnP and meet vision in
keyframe local BA; the published pose is PnP's unless the motion veto (below) refuses it. With `"per_frame_fusion": (enabled: true)` each
tracked frame's pose is optimized against its map matches (Huber), the IMU preintegrated from the
last keyframe at its bias and velocity (rotation; position once gravity is settled: a still-window
initialization or the second refinement) and the wheels from the last keyframe (the BA factor and
noise model), weighted by the configured noise figures.

Two gates, at `chi2_gate` (a 6-DOF value scaled to each test's degrees of freedom; default
22.458, 99.9 %):

- the IMU against the wheels, at their joint optimum: when they disagree, the one vision agrees
  with less is dropped for the frame (and the wheels are also dropped when their rotation
  disagrees with the gyro's);
- vision against the remaining prediction: past the gate the frame publishes the prediction as
  `Coasting` (`FailureReason::VisionRejected`) and leaves the map untouched. After 15 rejections
  in a row the next frame's vision is accepted, if the motion veto passes it, and the fusion waits
  for the next keyframe.

`VioDiagnostics::fusion` carries the counts (frames, rejects, expired runs, wheel and IMU drops),
this frame's chi2 and whether it was rejected. See the `fusion` module docs for the model and its
assumptions, none of which is measured on the robot yet.

## Motion veto

On by default (`"motion_veto": (enabled: false)` turns it off). It is the last gate on every pose
the tracker is about to adopt and publish: the tracked or fused pose whatever the fusion did, a
relocalization, and a local BA's result. A pose that the gyro (rotation) or the wheels
(translation of the camera centre) contradict is never published and never written to the map.
Each sensor is checked only when it measured the interval, so with neither nothing is checked.

- **Bounds.** 10 deg and 0.3 m, growing by 2 deg/s and 0.1 m/s since the last trusted pose and by
  0.2 of the rotation and distance the sensors measured since (`rotation_deg`, `translation_m`,
  `rotation_deg_per_s`, `translation_m_per_s`, `motion_fraction`).
- **Cumulative.** The candidate is checked against the previous pose carried by this frame's
  gyro and wheel increment, and also against every trusted pose of the last
  `rotation_window_s` (5 s) for rotation and `translation_window_s` (0.5 s) for translation, each
  carried forward by the dead reckoning since. Vision that drifts or freezes below the per-frame
  bound (a frozen view in a slow turn) is therefore still vetoed. The translation window is short
  because over seconds a frozen view and wheels spinning in place look alike.
- **Vetoed frames** publish the gyro and wheel dead reckoning as `Coasting`
  (`FailureReason::VisionVetoed`) with a growing covariance, or as `NotLocalized` when the wheels
  did not measure the frame (the translation is then a constant-velocity guess).
- **Recovery.** After a veto, or a tracking DEGRADED verdict from the keyframe-level motion check,
  vision is accepted again only after `recovery_frames` (5) consecutive frames agree with the dead
  reckoning within half the bounds. Vision itself is checked, not the fusion's prediction. Frames
  held this way are also `VisionVetoed` coasts, with `vio_diag`'s `veto.vetoed` false. If vision
  never agrees within `recovery_timeout_s` (3 s), the map is dropped and re-anchored at the
  dead-reckoned pose (the world stays continuous with wheels). The suspect interval leaving the BA
  window only lets local BA run again; it never recovers the pose in place.
- **BA gate.** A local BA is refused when it moved any keyframe past the bounds from its pre-BA
  pose, or the newest keyframe away from the dead reckoning of the trusted poses (so repeated
  small moves cannot add up). The map already holds the solve, so it is dropped next frame and
  re-anchored at the pre-BA pose, which is the one kept and published.

`vio_diag`'s `veto` record counts vetoes, dead-reckoned frames, recoveries, timeout re-anchors and
refused BA solves.

### The `local BA` line

After each local inertial BA the tracker logs what the solve did, from kornia-slam's
`BaUpdate::solve_stats`: `initial_cost`, `final_cost`, `median_initial_cost` (of the previous 64
solves), `iterations`, `converged`, `depth_rows_gated` (stereo depth rows the solver left out
because the landmark's depth disagreed with them), `reseeded` (landmarks moved back along their
ray to the measured depth), `cheirality_rejections` and `cheirality_held` (steps refused, or
landmark steps held, for moving a landmark behind an observing camera) and `write_back_refused`
(landmarks the solve left behind a camera or flung far, kept at their old position), then
`since_last:` and the same counts summed over every solve since the line last spoke, with
`solves`, `notable` and `spikes`. A solve is notable when any of those counts is non-zero or its
initial cost is past 100 times the running median (`cost_spike=true`). When a solve since the last
line was notable the line is a `warning!`, at most once every 30 s; otherwise it is an `info!` at
most once a minute. Nothing the throttle holds back is lost: its counts are in the next line's
`since_last:` totals. It is logged even when the update is refused afterwards. Log-only: no
`vio_diag` key carries it. The throttle is there because nearly every parked solve after the first
keyframe cull is notable: in a three-hour run of the test below, 230 of the 245 solves after it
gated a few depth rows, held a landmark step or refused a few write-backs (one solve per keyframe,
about every 30 s parked, and at keyframe rate when moving); none tripped the cost spike.

These are the figures of the parked failure kornia/kornia-slam#161 fixes: parked for an hour, a
local BA after a keyframe cull flung landmarks out along their rays, and a later window solved
against them moved a keyframe 80 deg / 2.2 m (a `motion veto: local BA refused` line and a map
drop). The regression test parks the full pipeline for two simulated hours with sensor noise and
requires no refused BA and every BA move under 1 deg across at least two culls. It takes
about 20 minutes in release on the Orin, so it is ignored by default:

```sh
cargo test --release --lib parked_hours_with_sensor_noise -- --ignored --nocapture
```

## Documentation and development

```sh
cargo doc --workspace --no-deps --open   # API docs; every public item is documented
cargo test --workspace                   # unit tests, including golden wire-format fixtures
cargo clippy --workspace --all-targets -- -D warnings
```

## The spec-string rule

There is **no `[patch]` table**. Cargo keys a git source on the exact spec STRING, not on the
resolved commit, so two spellings of one dependency are two packages at the same sha, and the
type error names the same path twice. It is the most expensive mistake available here.

It applies in three directions:

- **copper**: every copper-rs crate (`cu29`, `cu-sensor-payloads`, `cu-spatial-payloads`) is
  pinned by the same `rev =`. A consumer must spell that identical rev, or it gets a second
  `CuMsg` and a second `CuImage`.
- **kornia**: the kornia-rs crates (`kornia-image`, `kornia-3d`, `kornia-algebra`,
  `kornia-imgproc`, and any other a consumer names) come from **crates.io at `"0.2.0"`**, the
  exact requirement kornia-slam uses. A git spelling of kornia-rs is a different
  source even at identical code (v0.1.15 at `c9f9fd7` is the same code as 0.2.0), and splits
  `Image` in two. `kornia-slam` AND `kornia-sensors` (the latter is a workspace member of the
  kornia-slam repo, not of kornia-rs) are on one git `rev =`: kornia-slam
  `feat/evict-oldest-keyframes` at `fbb790c`, which is `main` at `05ea01c` (#153's tracking
  recovery, on top of #161's parked-BA fixes and #158's wheel-odometry factor, stationary IMU
  init, parallax keyframes and culling) plus keyframe eviction. It is re-pinned to the merge
  commit once that lands. They move to crates.io at
  the first release that carries it, and every consumer must move with them.
- **this repo's own two crates**: a consumer naming both `cu-kornia-vio` and `cu-stereo-payloads`
  must give them the identical URL and the identical `branch = "main"`. `cu-kornia-vio` reaches the
  payload crate by path, so it inherits the git source it was itself pulled from; pinning the
  payload line with `rev =` against a `branch =` on the other splits `StereoPair` in two. The
  instinct to pin a payload crate is exactly the trap here. The repo was renamed from
  `kornia/cufx-vio`, and GitHub still redirects that URL, so it resolves: a dependency spelled
  with the old URL next to one spelled with the new is the same split. Use
  `https://github.com/kornia/cu-kornia-vio` everywhere.

`Cargo.lock` is tracked so that a clean clone builds: it holds a kornia-rs / kornia-slam pair
known to compile together (kornia-rs 0.2.0 from crates.io, kornia-slam at `fbb790c`). A consumer's own lock still wins, but a kornia-slam from before its move to crates.io kornia-rs
(`6820744`) names git kornia-rs and splits the graph, and one from before its map-API rework
does not compile against this crate.

## Known limitations

- **Background only.** `StereoVio` is designed for `background: true`: inline, every other task
  in the graph waits behind each solve, and a frame that inserts a keyframe takes p50 93 ms even
  tuned (see [Wiring](#wiring)) against a 66 ms frame interval.
- **No map persistence or resim.** The kornia-slam map has no serialised form, so `StereoVio`'s
  `Freezable` is a no-op. A resim reproduces the task only from the start of a log, never from a
  mid-log keyframe, and a map cannot be saved or reloaded across runs.
- **IMU replay is approximate.** Samples cross into the background worker through the bus queue,
  not a logged input of the tracker, so which samples a given solve saw depends on when the worker
  ran. The `ImuFeed` edge is logged, but a replay is not bit-identical on the inertial path.
- **The map is unbounded unless told otherwise.** Nothing culls keyframes by default, and with
  neither `max_keyframes` nor `on_tracking_loss: "reset_map"` set the map grows until per-frame
  cost stalls the tracker. Set `max_keyframes` on any long-running graph; the map is then dropped
  and rebuilt when it reaches the bound. With `"keyframe_culling": (enabled: true)` the tracker
  first removes redundant keyframes there (ORB-SLAM's rule, IMU and wheel edges merged) and drops
  the map only if that cannot make room. Culling waits for a settled inertial path, and while
  driving few keyframes are redundant: the room comes mostly from stops. With
  `"at_keyframe_bound": "evict_oldest"` the tracker then evicts the oldest keyframes instead of
  dropping the map, so a drive through new space keeps one map as a sliding window; what the
  evicted keyframes constrained is lost with them. With the inertial path
  on, that needs `max_merged_imu_span_s` of at least twice the parked keyframe spacing
  (`parked_keyframe_interval_s` plus 0.25 s for the frame that lands past it), which startup
  checks; `stationary_edges` lifts that limit for a settled inertial path (see
  [Parked keyframes](#parked-keyframes)). Culling bounds keyframes, not landmark
  slots (kornia-slam never compacts those), so the map is still dropped once it holds
  `max_keyframes * orb_keypoints` of them; with eviction that is what bounds a long drive.

## License

Apache-2.0; see [LICENSE](LICENSE). [NOTICE](NOTICE) credits ORB-SLAM3 (Campos et al., IEEE
T-RO 2021) as the algorithmic reference for the tracking pipeline design.
