# Hamster AI detection: label → train → deploy

The whole process takes about one afternoon. Apart from labeling the photos, everything is a button click.

## 0. Collect photos (automatic)
Camera tracking saves photos on the Raspberry Pi by itself:
- **motion_*.jpg** – saved when movement is detected (at most one every 30 s). **Most already come with a pre-drawn box** (a `.txt` file with the same name)
- **idle_*.jpg** – one every 10 minutes when nothing moves (the hamster may be asleep, hidden, or the cage empty)

The **AI hamster detection** card on the dashboard shows the counts. Start once you have **300+ motion photos**. Hamsters are most active at night, so this usually takes 3–7 days.

## 1. Pack and download
Dashboard → AI hamster detection → **打包訓練照片 (Pack training photos)** → **下載 zip (Download zip)**
- Near-identical photos are dropped automatically; at most 600 photos are packed
- Unzip it on your computer, e.g. `Desktop\hamster_dataset\`

## 2. Label (the slow part: about 1–2 hours for 600 photos)
1. Download **X-AnyLabeling** (Windows .exe): https://github.com/CVHub520/X-AnyLabeling/releases
2. Open it → **Open Dir** → choose the `hamster_dataset` folder
3. Menu **Upload → YOLO annotations**, pick the `classes.txt` inside the folder; the pre-drawn boxes are loaded
4. Go through every photo (shortcuts: `D` next, `A` previous, `W` draw a new box):
   - The box should **fit the hamster's body tightly**, head to rump, with no extra empty space
   - Pre-drawn boxes are guesses from motion detection and are often **too big, shifted, or on a reflection with no hamster at all** – fix or delete them
   - **No hamster in the photo** → delete all boxes (the photo becomes a "background" example that teaches the model what is *not* a hamster)
   - **Only a tiny part visible, or too blurry to tell** → box it if you are sure, otherwise **delete the whole photo**
   - Hamster inside the transparent hideout and visible through the plastic → box it
5. When done → menu **Export → YOLO annotations** (into the same folder, overwriting the old `.txt` files)
6. **Zip the whole `hamster_dataset` folder**

> Rule of thumb: fewer photos that are all correct beat more photos with mistakes. Wrong labels hurt the model more than missing photos.

## 3. Train (Google Colab, free GPU, about 20–40 minutes)
1. Open https://colab.research.google.com → **Upload notebook** → pick `train_hamster_colab.ipynb`
2. Menu **Runtime → Change runtime type → T4 GPU**
3. Press ▶ on each cell from top to bottom:
   - Cell 2 shows an upload button → choose your `hamster_dataset.zip`
   - Cell 5 shows the score: **mAP50 of 0.8 or higher works well**; below 0.6 means the labels or photos need improving
   - Cell 6 downloads **`hamster_ncnn_model.zip`** automatically

## 4. Deploy (1 minute)
Dashboard → AI hamster detection → **上傳訓練好的模型 (Upload trained model)** → choose `hamster_ncnn_model.zip`
- The model is **test-run on the Pi first**; if that fails, nothing is changed
- On success the old model is backed up, the tracker switches to **hybrid** mode and restarts

## Detection modes (`config.toml` → `[tracker]` → `detector`)
| Mode | Position comes from | "Active" means | Use when |
|---|---|---|---|
| `motion` | motion detection | something in the picture moves | no model yet |
| `hybrid` (recommended) | **AI box** | motion near the AI box | you have a model – the hamster is located even when still, and kicked-up bedding is not counted |
| `yolo` | AI box | the AI box moves | lighting changes a lot and motion detection gives false alarms |

Other settings: `yolo_conf` (confidence threshold – raise it if other things are mistaken for the hamster, lower it if the hamster is often missed) and `yolo_interval_s` (how often detection runs, default 0.5 s).

## Making it more accurate later
- New cage, new camera angle, or frequent mistakes → collect a new batch of photos and repeat steps 1–4
- In Colab cell 4, change `YOLO('yolo11n.pt')` to `YOLO('best.pt')` (included in the previous model zip) to continue from the old model – faster and usually better
