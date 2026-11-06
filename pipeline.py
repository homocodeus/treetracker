"""TreeTracker: dataset preparation, YOLO training and single-tree video analysis."""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import shutil


def segment_angle(a, b):
    if math.hypot(b[0]-a[0], b[1]-a[1]) < 1e-6:
        return None
    return math.degrees(math.atan2(b[0]-a[0], a[1]-b[1]))


def angle_delta(current, baseline):
    return (current-baseline+180) % 360-180


def motion(point, baseline, deadband=1):
    dx, dy = point[0]-baseline[0], baseline[1]-point[1]
    direction = 'stationary' if math.hypot(dx, dy) <= deadband else (
        ('right' if dx > deadband else 'left' if dx < -deadband else '') +
        ('-up' if dy > deadband else '-down' if dy < -deadband else '')).strip('-')
    return dx, dy, math.hypot(dx, dy), direction


def prepare(args):
    root, out = Path(args.source).resolve(), Path(args.output).resolve()
    if out.exists():
        raise ValueError('Output already exists; choose a new directory.')
    schema = json.loads((root/'schema.json').read_text())
    count = len(schema['keypoints'])
    if not 2 <= count <= 30:
        raise ValueError('Schema must contain 2–30 keypoints.')
    pairs = []
    for image in sorted((root/'images').iterdir()):
        if not image.is_file():
            continue
        label = root/'labels'/f'{image.stem}.txt'
        values = label.read_text().split()
        if len(values) != 5+3*count:
            raise ValueError(f'{label.name}: expected one tree with {count} keypoints.')
        numbers = list(map(float, values))
        if not all(math.isfinite(v) for v in numbers) or numbers[0] != 0:
            raise ValueError(f'Invalid class or values in {label.name}')
        if not all(0 <= v <= 1 for v in numbers[1:5]) or min(numbers[3:5]) <= 0:
            raise ValueError(f'Invalid box in {label.name}')
        for i in range(count):
            x, y, visibility = numbers[5+3*i:8+3*i]
            if not (0 <= x <= 1 and 0 <= y <= 1 and visibility in (0, 1, 2)):
                raise ValueError(f'Invalid keypoint in {label.name}')
        pairs.append((image, label))
    if len(pairs) < 2:
        raise ValueError('At least two annotated images are required.')
    random.Random(args.seed).shuffle(pairs)
    nval = max(1, min(len(pairs)-1, round(len(pairs)*args.val)))
    for i, (image, label) in enumerate(pairs):
        split = 'val' if i < nval else 'train'
        for kind, src in [('images', image), ('labels', label)]:
            dst = out/kind/split/src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    shutil.copy2(root/'schema.json', out/'schema.json')
    (out/'dataset.yaml').write_text(f'path: {json.dumps(str(out))}\ntrain: images/train\nval: images/val\nkpt_shape: [{count}, 3]\nnames:\n  0: tree\n')
    print(f'Prepared {len(pairs)-nval} training and {nval} validation images at {out}')


def train(args):
    from ultralytics import YOLO
    data = Path(args.data).resolve()
    if not data.is_file():
        raise ValueError('Dataset YAML not found. Run prepare first.')
    model = YOLO(args.model)
    if model.task != 'pose':
        raise ValueError('Choose a pose model.')
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz,
                device=args.device, project=args.output, name='tree-pose',
                fliplr=0.0, flipud=0.0, degrees=0.0)
    print('Training complete. Use weights/best.pt from the printed run directory.')


def analyze(args):
    import cv2
    from ultralytics import YOLO
    schema = json.loads(Path(args.schema).read_text())
    names = [p['name'] for p in schema['keypoints']]
    edges = schema['edges']
    if not edges or any(len(e) != 2 or any(not isinstance(i, int) or not 0 <= i < len(names) for i in e) for e in edges):
        raise ValueError('Define valid skeleton connections before analysis.')
    model = YOLO(args.model)
    if model.task != 'pose':
        raise ValueError('Analysis requires a trained pose model.')
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError('Cannot open video or read its frame rate.')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    writer = None
    baseline_points, baseline_angles = {}, {}
    rows = 0
    try:
        with (out/'motion.csv').open('w', newline='') as f:
            csvout = csv.writer(f)
            csvout.writerow(['frame','time_s','type','name','x_px','y_px','confidence','dx_px','dy_up_px','displacement_px','direction','angle_deg','change_deg','status'])
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                result = model.predict(frame, conf=args.conf, device=args.device, verbose=False)[0]
                points = {}
                # Fixed-camera, single-tree scenes: never switch between multiple detections.
                if result.keypoints is not None and len(result.keypoints.xy) == 1:
                    xy = result.keypoints.xy[0].cpu().tolist()
                    confidences = result.keypoints.conf
                    if len(xy) != len(names) or confidences is None:
                        raise ValueError('Model keypoints do not match the annotation schema.')
                    scores = confidences[0].cpu().tolist()
                    points = {i: p for i, (p, c) in enumerate(zip(xy, scores)) if c >= args.conf}
                else:
                    scores = [0]*len(names)
                for i, name in enumerate(names):
                    if i not in points:
                        csvout.writerow([frame_idx,frame_idx/fps,'point',name,'','','','','','','','','','missing_or_ambiguous'])
                        continue
                    p = points[i]
                    baseline_points.setdefault(i, p[:])
                    dx, dy, distance, direction = motion(p, baseline_points[i], args.deadband)
                    csvout.writerow([frame_idx,frame_idx/fps,'point',name,*p,scores[i],dx,dy,distance,direction,'','','ok'])
                    cv2.circle(frame, tuple(map(round,p)), 5, (70,220,160), -1)
                    cv2.putText(frame, str(i+1), tuple(map(round,p)), cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                for a,b in edges:
                    angle = segment_angle(points[a],points[b]) if a in points and b in points else None
                    if angle is None:
                        csvout.writerow([frame_idx,frame_idx/fps,'segment',f'{names[a]} -> {names[b]}','','','','','','','','','','missing'])
                        continue
                    baseline_angles.setdefault((a,b),angle)
                    delta = angle_delta(angle, baseline_angles[a,b])
                    csvout.writerow([frame_idx,frame_idx/fps,'segment',f'{names[a]} -> {names[b]}','','','','','','','',angle,delta,'ok'])
                    cv2.line(frame,tuple(map(round,points[a])),tuple(map(round,points[b])),(150,230,210),2)
                    midpoint = tuple(round((x+y)/2) for x,y in zip(points[a],points[b]))
                    cv2.putText(frame,f'{delta:+.1f} deg',midpoint,cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                if writer is None:
                    writer = cv2.VideoWriter(str(out/'annotated.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(frame.shape[1],frame.shape[0]))
                    if not writer.isOpened():
                        raise ValueError('MP4 encoder unavailable.')
                writer.write(frame)
                frame_idx += 1
                rows += len(points)
            if not frame_idx:
                raise ValueError('Video contains no decodable frames.')
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    (out/'summary.json').write_text(json.dumps({'frames':frame_idx,'fps':fps,'valid_point_samples':rows,'reference':'First confident observation of each point/segment','angle':'Degrees from upward vertical; clockwise positive','movement':'Image plane pixels; positive dy is up','camera':'Fixed camera required; camera motion is not compensated','schema':schema},ensure_ascii=False,indent=2))
    print(f'Analysis saved to {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare');p.add_argument('source');p.add_argument('--output',default='dataset');p.add_argument('--val',type=float,default=.2);p.add_argument('--seed',type=int,default=42);p.set_defaults(run=prepare)
    p = commands.add_parser('train');p.add_argument('--data',default='dataset/dataset.yaml');p.add_argument('--model',default='yolo11n-pose.pt');p.add_argument('--epochs',type=int,default=100);p.add_argument('--imgsz',type=int,default=640);p.add_argument('--device',default='cpu');p.add_argument('--output',default='runs');p.set_defaults(run=train)
    p = commands.add_parser('analyze');p.add_argument('--model',required=True);p.add_argument('--video',required=True);p.add_argument('--schema',required=True);p.add_argument('--output',default='analysis');p.add_argument('--conf',type=float,default=.5);p.add_argument('--deadband',type=float,default=1);p.add_argument('--device',default='cpu');p.set_defaults(run=analyze)
    args = parser.parse_args()
    if hasattr(args,'val') and not 0 < args.val < 1: parser.error('--val must be between 0 and 1')
    if hasattr(args,'conf') and not 0 < args.conf <= 1: parser.error('--conf must be in (0,1]')
    if hasattr(args,'deadband') and args.deadband < 0: parser.error('--deadband must be nonnegative')
    if hasattr(args,'epochs') and (args.epochs < 1 or args.imgsz < 32): parser.error('Invalid training settings')
    try:
        args.run(args)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f'Error: {exc}\n')

if __name__ == '__main__':
    main()
