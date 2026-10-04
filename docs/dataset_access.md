# Public datasets for navigation training: access and licence

Surveyed on 2026-10-04 by three research passes reading each dataset's own pages and licence text. I did not
re-check them line by line; "unclear" means the licence could not be read. Owner-facing version with steps:
https://claude.ai/artifact/WGnLK4P5b8Wa14cCdcTAwT

Publishing rule (owner, 2026-10-04): everything goes on Hugging Face; data rendered in Matterport scenes
goes to a **gated** repo whose access form shows the Matterport academic EULA and records who accepts.

## Credentials status (2026-10-04)

| Item | Status |
|---|---|
| Hugging Face click-throughs (StreamVLN, InternData-N1, Berkeley-FrodoBots-7K, hssd/ai2thor-hab) | accepted by Jinyan0924; file access verified with `HF_TOKEN` |
| Matterport API token (HM3D) | works; `hm3d_minival_v0.2` downloaded as a test. Must be re-entered in `.env` on a new machine |
| MP3D signed terms of use | emailed to matterport3d@googlegroups.com on 2026-10-04; reply (a `download_mp.py` script) pending, usually days. Put it at `/workspace/download_mp.py` |
| nuScenes account | not yet (`NUSCENES_EMAIL` / `NUSCENES_PASSWORD`) |
| Nymeria URL list | not yet (`/workspace/nymeria_download_urls.json`) |

HM3D download: `python -m habitat_sim.utils.datasets_download --username $MATTERPORT_TOKEN_ID --password
$MATTERPORT_TOKEN_SECRET --uids hm3d_train_v0.2 hm3d_val_v0.2 --data-path <dir>` (only the minival uid was
confirmed; train is 32 GB + 8 GB annotations).

## Open, no credential, republishable

| Dataset | Content | Size | Host | Licence |
|---|---|---|---|---|
| SCAND | Jackal + Spot among people; cameras, Kinect, Velodyne, odometry | 8.7 h, 344 GB, rosbags | dataverse.tdl.org doi:10.18738/T8/0PRYRH (needs a browser user-agent) | CC0 |
| MuSoHu | helmet-worn: ZED 2 stereo, Velodyne, 360 camera | 20 h, 100 km, 1,044 GB, rosbags | dataverse.orc.gmu.edu doi:10.13021/orc2020/HZI4LJ | CC0 |
| GND | wheeled robot, 10 campuses; lidar, cameras, traversability maps | 11 h, 876 GB | dataverse.orc.gmu.edu doi:10.13021/orc2020/JUIW5F | CC0 |
| EgoWalk | person-carried ZED: RGB, depth, odometry, language goals | 50 h, 268 GB, parquet + mp4 | HF EgoWalk/trajectories | MIT |
| RECON | Jackal outdoors, front RGB, position | 50 GB, HDF5 | rail.eecs.berkeley.edu/datasets/recon-navigation/recon_dataset.tar.gz | MIT |
| SACSoN / HuRoN | TurtleBot among pedestrians, low-res spherical images, 2D lidar | 75 h, rosbags | rail.eecs.berkeley.edu/datasets/huron/ | MIT |
| GoStanford 2 / 4 | TurtleBot, fisheye cameras, velocities | 16.7 h + 10.3 h | svl.stanford.edu/projects/gonet++/dataset, .../dvmpc/dataset | CC BY-NC-SA 3.0 |
| FrodoBots-2K | sidewalk rovers; front + rear cams, GPS 1 Hz, IMU (no pose) | 2,000 h, 1 TB | HF BitRobot/FrodoBots-2K | CC BY-SA 4.0 |
| Berkeley-FrodoBots-7K | same rovers, re-annotated actions | 7,000 h, 800 GB | HF BitRobot/Berkeley-FrodoBots-7K (gate accepted) | CC BY-SA 4.0 |
| GrandTour | ANYmal: 10 cams, 3 lidars, RTK-GPS | 2.11 TB, Zarr | HF leggedrobotics/grand_tour_dataset | CC BY-SA 4.0 (site) vs MIT (tag): use the stricter |
| RELLIS-3D | Warthog off-road, RGB, lidars, SLAM poses | 5 sequences | Google Drive | CC BY-NC-SA 3.0 |
| NCLT | Segway, omni camera, lidar, RTK | 34.9 h, 147 km | robots.engin.umich.edu/nclt | ODbL |
| TartanDrive 1 / 2 | ATV off-road | 7 h (v2) | scripts in castacks repos | no data licence stated |
| TartanGround | simulated ground robot, pre-rendered | 1.44 M samples | tartanair.org/tartanground | CC BY 4.0 |
| TartanAir V2 | simulated, pre-rendered | unknown | `pip install tartanair` | CC BY 4.0 |
| HSSD | 211 synthetic houses (our simulator) | 12 GB | HF hssd/hssd-hab | CC BY-NC 4.0 |
| ReplicaCAD | apartment variations | 157 MB | HF ai-habitat/ReplicaCAD_dataset | CC BY 4.0 |
| ProcTHOR in Habitat | 12,000 houses | 18.4 GB | HF hssd/ai2thor-hab (gate accepted) | no licence tag: unclear |
| OpenScene | nuPlan sensor data at 2 Hz: cams, lidar, boxes | 120 h, 1.9 TB (mini 144 GB) | HF OpenDriveLab/OpenScene | CC BY-NC-SA 4.0 + nuPlan terms |
| Argoverse 2 Sensor | 9 cams, lidar, poses, 3D boxes | 4.2 h, 1 TB | `s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/sensor/*"` | CC BY-NC-SA 4.0 |
| comma2k19 | highway driving, camera + poses | 33 h, 100 GB | HF commaai/comma2k19 | MIT |
| Yaak L2D | driving, 6 cams, language | 5,000 h, 4.76 TB | HF yaak-ai/L2D | Apache-2.0 |
| ZOD | driving, camera + lidar + pose | 100k frames | HF Zenseact/ZOD | CC BY-SA 4.0 |
| THOR-MAGNI | room mocap of people + robot (not ego navigation) | 3.5 h | zenodo.org/records/10407223 | CC BY 4.0 |
| COMMAND | simulated (Gazebo), sketch instructions | 48 h, 1.13 TB | HF maum-ai/COMMAND | CC BY-NC 4.0 |

Already converted and published by us: CODa (CC BY-NC-SA 4.0), RoboSense (README says CC BY-NC-SA 4.0, HF tag
says cc-by-sa-4.0: use the stricter), JRDB (CC BY-NC-SA 3.0).

## Matterport-based (gated repo required)

| Item | Content | Host | Notes |
|---|---|---|---|
| HM3D v0.2 / HM3D-Semantics | 1,000 scans; train 32 GB, val 4 GB | Matterport API token | academic, non-commercial EULA |
| MP3D | 90 buildings | `download_mp.py` after the signed form | same EULA |
| VLN-CE R2R / RxR-CE episodes | definitions only | Google Drive ids `1T9SjqZWyR2PCLSXYkFckfDeIs6Un0Rjm` (R2R), `145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t` (RxR); also dl.fbaipublicfiles.com/habitat/data/datasets/vln/mp3d/r2r/v1/vln_r2r_mp3d_v1.zip | Matterport terms + CC BY-NC-SA 3.0 |
| ObjectNav HM3D v1 / v2 | episodes | dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip | |
| HM3D-OVON | open-vocabulary object-goal, 379 categories (what Qwen-RobotNav used) | HF nyokoyama/hm3d_ovon, 168 MB | MIT |
| PointNav episodes | Gibson v1/v2, MP3D v1, HM3D v1 | dl.fbaipublicfiles.com (habitat-lab DATASETS.md) | no HM3D v2 exists |
| EVT-Bench (TrackVLA) | tracking episodes STT / DT / AT; avatars | github.com/wsakobe/TrackVLA (`data/datasets/track/`), avatars via `download_humanoid_data.py` | CC BY-NC-SA 4.0; needs HM3D + MP3D |
| GOAT-Bench | multi-goal episodes on HM3D | Google Drive `1N0UbpXK3v7oTphC4LoDqlNeMHbrwkbPe` | licence not stated |
| StreamVLN trajectory data | R2R, RxR, EnvDrop, ScaleVLN subset, with rendered RGB | HF cywan/StreamVLN-Trajectory-Data, 92.5 GB (gate accepted) | CC BY-NC-SA 4.0 |
| InternData-N1 | 240k+ trajectories with frames, LeRobot v2.1 | HF InternRobotics/InternData-N1; mini about 218 GB, full 5 TB+ (gate accepted) | CC BY-NC-SA 4.0 |
| ScaleVLN | 4.9 M instruction-trajectory pairs (HM3D + Gibson, discrete graph) | HF OpenGVLab/ScaleVLN, 52 GB | apache-2.0 tag |

Matterport EULA section 2.4: derived information may be distributed only for academic purposes, with the
agreement linked; distributing "a substantial portion" requires a recorded click-through acceptance by each
recipient. Hence the gated repo.

## Needs an account or form (optional)

- nuScenes: nuscenes.org/sign-up; CC BY-NC-SA 4.0 + Motional terms; converted data may be republished under
  the same licence.
- Nymeria (300 h, 400 km of people walking with Aria glasses, about 80 TB): URL JSON from
  explorer.projectaria.com/nymeria; CC BY-NC 4.0, republishable non-commercially.
- Gibson: forms.gle/36TW9uVpjrE1Mkf9A; licence unread.
- KITTI / KITTI-360: registration at cvlibs.net with institutional email; CC BY-NC-SA 3.0.
- GrandTour rosbags: forms.gle/2qJkGYJ6oxnBvdNq9 (the HF copy is open).
- Unreleased GNM data (Seattle, raw CoryHall): email the GNM authors.
- JRDB: account at jrdb.erc.monash.edu (we used the Stanford mirror).
- R-KNav (sidewalk delivery rover, 10,000 h claimed; 30 min public): form / contact Robot.com.

## Cannot be republished (leave out, or keep private and publish only the converter)

Waymo (all sets), ONCE, CoVLA, NVIDIA PhysicalAI-AV, Ego4D, Aria Everyday Activities, Aria Digital Twin,
SiT (CC BY-NC-ND), ScanNet / ScanNet++, 3D-FRONT (unverified).

YouTube-based (share ids and derived paths only, not frames): CityWalker (about 2,000 h; poses must be
recomputed with DPVO), NaVILA human videos, part of LeLaN.

Unclear: Gibson, Ego-Exo4D, BDD100K, TartanDrive, RUGD (no poses anyway), the three GNM subsets in Open
X-Embodiment (berkeley_gnm_recon / cory_hall / sac_son), GOAT-Bench, REVERIE, SOON, Uni-NaVid subset.

Not released at all: Qwen-RobotNav's generated corpus, NavFoM and NaVid training data.

## Notes from first contact

- **EgoWalk** (one trajectory inspected): `data/<name>.parquet` has per-frame timestamp (ms), `cart_x/y/z`,
  `quat_x/y/z/w`; `video/rgb/<name>__rgb.mp4` 960 x 600, one video frame per row (the container reports
  100 fps; the real rate is 5 fps, `meta/info.json`); `annotations/end2end/<name>.parquet` gives
  (start_frame, end_frame, caption, brief): the caption names the thing reached at the end of the frame
  range. `meta/heights.json` gives camera height (1.25 to 1.34 m), `meta/camera_rgb.json` a rational
  distortion model (fx 367.5, cx 483.3, k1..k6). Pose frame: body x forward, z up (travel direction in the
  body frame averaged (0.81, -0.16, 0.15)). 140 of 650 poses were null, and there are jumps of up to 11 m
  (odometry re-initialisations): cut episodes at nulls and jumps. Median speed 0.78 m/s.
