
# Python 2/3 compatibility
from __future__ import print_function
import sys
import json
import datetime
import os
import glob

PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2

# built-in modules
from collections import namedtuple

# local modules
from PyLogoDet import common, video

FLANN_INDEX_KDTREE = 1
FLANN_INDEX_LSH    = 6
flann_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

MIN_MATCH_COUNT = 10

'''
  image     - image to track
  rect      - tracked rectangle (x1, y1, x2, y2)
  keypoints - keypoints detected inside rect
  descrs    - their descriptors
  data      - some user-provided data
'''
PlanarTarget = namedtuple('PlaneTarget', 'image, rect, keypoints, descrs, data')

'''
  target - reference to PlanarTarget
  p0     - matched points coords in target image
  p1     - matched points coords in input frame
  H      - homography matrix from p0 to p1
  quad   - target bounary quad in input frame
'''
TrackedTarget = namedtuple('TrackedTarget', 'target, p0, p1, H, quad')

class PlaneTracker:
    def __init__(self):
        self.detector = cv2.AKAZE_create()
        self.matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
        self.targets = []
        self.frame_points = []

    def add_target(self, image, rect, data=None):
        '''Add a new tracking target.'''
        x0, y0, x1, y1 = rect
        raw_points, raw_descrs = self.detect_features(image)
        points, descs = [], []
        for kp, desc in zip(raw_points, raw_descrs):
            x, y = kp.pt
            if x0 <= x <= x1 and y0 <= y <= y1:
                points.append(kp)
                descs.append(desc)
        descs = np.uint8(descs)
        self.matcher.add([descs])
        target = PlanarTarget(image = image, rect=rect, keypoints = points, descrs=descs, data=data)
        self.targets.append(target)

    def clear(self):
        '''Remove all targets'''
        self.targets = []
        self.matcher.clear()

    def track(self, frame):
        '''Returns a list of detected TrackedTarget objects'''
        self.frame_points, frame_descrs = self.detect_features(frame)
        if len(self.frame_points) < MIN_MATCH_COUNT:
            return []
        matches = self.matcher.knnMatch(frame_descrs, k = 2)
        matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < m[1].distance * 0.75]
        if len(matches) < MIN_MATCH_COUNT:
            return []
        matches_by_id = [[] for _ in xrange(len(self.targets))]
        for m in matches:
            matches_by_id[m.imgIdx].append(m)
        tracked = []
        for imgIdx, matches in enumerate(matches_by_id):
            if len(matches) < MIN_MATCH_COUNT:
                continue
            target = self.targets[imgIdx]
            p0 = [target.keypoints[m.trainIdx].pt for m in matches]
            p1 = [self.frame_points[m.queryIdx].pt for m in matches]
            p0, p1 = np.float32((p0, p1))
            H, status = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
            status = status.ravel() != 0
            if status.sum() < MIN_MATCH_COUNT:
                continue
            p0, p1 = p0[status], p1[status]

            x0, y0, x1, y1 = target.rect
            quad = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            quad = cv2.perspectiveTransform(quad.reshape(1, -1, 2), H).reshape(-1, 2)

            track = TrackedTarget(target=target, p0=p0, p1=p1, H=H, quad=quad)
            tracked.append(track)
        tracked.sort(key = lambda t: len(t.p0), reverse=True)
        return tracked

    def detect_features(self, frame):
        '''detect_features(self, frame) -> keypoints, descrs'''
        keypoints, descrs = self.detector.detectAndCompute(frame, None)
        if descrs is None:  # detectAndCompute returns descs=None if not keypoints found
            descrs = []
        return keypoints, descrs


class App:
    #cap is created
    def __init__(self, src):
        self.vidOrImg = -1
        if not src == 0 and not src == 1:
            self.src = src
            splitsrc = src.split(".")
            ext = splitsrc[len(splitsrc) - 1].lower()
            exts = ["jpg", "jpeg", "bmp", "png"]

            if ext in exts:
                self.frame = cv2.imread(src)
                shape = self.frame.shape
                if shape[0] > 4000:
                    self.frame = cv2.resize(self.frame, (0, 0), fx=0.2, fy=0.2)
                    self.vidOrImg = 0
                elif shape[0] > 3000:
                    self.frame = cv2.resize(self.frame, (0, 0), fx=0.3, fy=0.3)
                    self.vidOrImg = 0
                elif shape[0] > 2000:
                    self.frame = cv2.resize(self.frame, (0, 0), fx=0.4, fy=0.4)
                    self.vidOrImg = 0
                elif shape[0] > 1500:
                    self.frame = cv2.resize(self.frame, (0, 0), fx=0.5, fy=0.5)
                    self.vidOrImg = 0
                elif shape[0] > 1000:
                    self.frame = cv2.resize(self.frame, (0, 0), fx=0.8, fy=0.8)
                    self.vidOrImg = 0
                self.vidOrImg = 0

            else:
                self.cap = video.create_capture(src)
                self.vidOrImg = 1
                self.frame = None
        else:
            self.cap = video.create_capture(src)
            self.vidOrImg = 2
            self.frame = None

        self.paused = False
        self.tracker = PlaneTracker()
        cv2.namedWindow('plane')

        ###############
        self.rect_sel = common.RectSelector('plane', self.on_rect)
        path = os.path.join(os.getcwd(), "stored")
        types = ["*.jpg", "*.png", "*.jpeg", "*.bmp"]
        files_grabbed = []

        for files in types:
            path2 = os.path.join(path, files)
            files_grabbed.extend(glob.glob(path2))

        i = 0;
        for file in files_grabbed:
            img = cv2.imread(file)

            i = i + 1
            jsonfile = file.split('.')
            jsonfilename = jsonfile[0] + ".json"
            with open(jsonfilename, 'r') as f:
                region = json.load(f)
            region = region['roi']
            hight, width, depth = img.shape
            self.tracker.add_target(img, [int(region[0]), int(region[1]), int(region[2]), int(region[3])])
        ####################

        self.rect_sel = common.RectSelector('plane', self.on_rect)

    def on_rect(self, rect):
        if self.vidOrImg == 0 or self.vidOrImg ==1: #if src is a video or an image
            now = datetime.datetime.now()
            path, file = os.path.split(self.src)
            strnow = str(now).replace("-", "")
            strnow = strnow.replace(":", "")
            strnow = strnow.replace(" ", "")
            strnow = strnow.replace(".", "")
            fileSplit = file.split(".")
            filename = fileSplit[0]
            fileExt = fileSplit[1]
        elif self.vidOrImg == 2: #if src is from webcam
            now = datetime.datetime.now()
            strnow = str(now).replace("-", "")
            strnow = strnow.replace(":", "")
            strnow = strnow.replace(" ", "")
            strnow = strnow.replace(".", "")
            filename = "cam"
            fileExt = "jpg"

        if self.vidOrImg == 0: #if source is an image
            ###############
            dir = os.path.join(os.getcwd(),"stored",filename+strnow+"."+fileExt)

            # roi = {"roi":[str(rect[1]),str(rect[3]), str(rect[0]),str(rect[2])]}
            roi = {"roi": [str(rect[0]), str(rect[1]), str(rect[2]), str(rect[3])]}
            outfilename = os.path.join(os.getcwd(),"stored",filename+strnow+'.json')
            with open(outfilename, 'w') as outfile:
                json.dump(roi, outfile)
            cv2.imwrite(dir,self.frame)
            ################
        elif self.vidOrImg == 1 or  self.vidOrImg == 2: #if source is a video or a webcam
            dir = os.path.join(os.getcwd(), "stored", filename + strnow + ".jpg")

            # roi = {"roi":[str(rect[1]),str(rect[3]), str(rect[0]),str(rect[2])]}
            roi = {"roi": [str(rect[0]), str(rect[1]), str(rect[2]), str(rect[3])]}
            outfilename = os.path.join(os.getcwd(), "stored", filename + strnow + '.json')
            with open(outfilename, 'w') as outfile:
                json.dump(roi, outfile)
            cv2.imwrite(dir,self.frame)
        self.tracker.add_target(self.frame, rect)

    def run(self):

        while True:
            if self.vidOrImg == 0:
                playing = not self.paused and not self.rect_sel.dragging
                if playing or self.frame is None:
                    self.frame = self.frame.copy()

                vis = self.frame.copy()
                if playing:
                    tracked = self.tracker.track(self.frame)
                    for tr in tracked:
                        cv2.polylines(vis, [np.int32(tr.quad)], True, (0, 255, 0), 2)
                        for (x, y) in np.int32(tr.p1):
                            cv2.circle(vis, (x, y), 2, (255, 255, 255))
            elif self.vidOrImg == 1 or self.vidOrImg==2:
                playing = not self.paused and not self.rect_sel.dragging
                if playing or self.frame is None:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    self.frame = frame.copy()

                vis = self.frame.copy()
                if playing:
                    tracked = self.tracker.track(self.frame)
                    for tr in tracked:
                        cv2.polylines(vis, [np.int32(tr.quad)], True, (255, 255, 255), 2)
                        for (x, y) in np.int32(tr.p1):
                            cv2.circle(vis, (x, y), 2, (255, 255, 255))

            self.rect_sel.draw(vis)
            cv2.imshow('plane', vis)
            ch = cv2.waitKey(1) & 0xFF
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == ord('c'):
                self.tracker.clear()
            if ch == 27:
                break

if __name__ == '__main__':
    print(__doc__)

    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    print(video_src)
    App(video_src).run()
