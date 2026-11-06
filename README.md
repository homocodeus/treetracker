# TreeTracker

A local browser application for annotating tree keypoints, training a custom pose model and analyzing tree movement in video.

## Run the annotation app

```sh
npm run dev
```

Open http://localhost:5173. Python 3 is required. No package installation is needed for the annotation interface.

## Annotate images

- Try the sample tree or upload your own images.
- Select a keypoint from the list and click the image to place it. Drag markers to adjust their positions.
- Define up to 30 keypoints. Use **Fill up to 30 points** to create additional points at once.
- Add named points, edit connections and mark occluded points.
- Download the dataset as a ZIP. All images share the same keypoint schema.
- Include one tree per image. The YOLO bounding box covers the entire image; crop images beforehand if necessary.
- Unmarked points use visibility 0, occluded points use 1 and visible points use 2.
- Prepare training and validation folders using the pipeline below.

## Current limitations

Annotations are stored in memory; export before refreshing the page. The sample image is excluded from exports. The browser does not upload your files to a server. System fonts are used if Google Fonts is unavailable.

Model training and video analysis run through the Python commands below. Raspberry Pi/Coral integration is not implemented. Coral compatibility requires separate validation.

## Checks

`npm run check` checks JavaScript syntax. Browser interactions still require verification in a real browser.

## Training and video analysis

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Extract the annotation ZIP to exported/ first.
python pipeline.py prepare exported --output dataset
python pipeline.py train --data dataset/dataset.yaml --epochs 100 --device cpu
python pipeline.py analyze --model runs/tree-pose/weights/best.pt --video tree.mp4 --schema dataset/schema.json --output analysis
```

Training downloads initial YOLO11 pose weights on first use. Use the actual best.pt path printed by training. Hardware options: `--device mps` for supported Apple hardware, `--device 0` for CUDA. Training disables horizontal/vertical flips to preserve named anatomical points. Do not use the human pretrained model directly for tree measurements.

Prepare validates exported labels and creates a reproducible 80/20 split. At least two images are required for technical operation, but this is insufficient for a useful model. Use diverse data and separate capture sessions for meaningful validation; adjacent frames leak information across a random split. Existing output directories are refused to protect previous work.

Analysis is for a single tree under a stationary camera. Each frame is inferred independently; anatomical identities come from the pose model, with no interpolation over missing detections. Frames with multiple detections are rejected as ambiguous. Low confidence samples remain blank in CSV, not zero motion. Each point/segment uses its first confident observation as its baseline. Segments measure inclination from upward vertical, clockwise positive, and wrapped angle changes in [-180,180). This is segment inclination, not a joint angle between two branches. Displacements use pixels (x right, y up) relative to reference; direction labels use a 1px deadband configurable with `--deadband`. Camera movement is not compensated. No depth or physical length is inferred.

Outputs: annotated.mp4, motion.csv, summary.json. Analysis stops on schema/model mismatch. Use new output directories for repeat runs. Browser panels document these terminal workflows; they do not launch Python jobs.

```sh
python3 -m unittest discover -s tests -v
```

Reference APIs: [Ultralytics pose](https://docs.ultralytics.com/tasks/pose/) and [training](https://docs.ultralytics.com/modes/train/).
