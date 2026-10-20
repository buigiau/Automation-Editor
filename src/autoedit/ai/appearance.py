"""Reference costume/appearance screening, without facial identity matching."""
from pathlib import Path

from .cache import digest

VERSION = 'reference-costume-3-palette-veto'


class ReferenceAppearance:
    def __init__(self, references, people, cache):
        import cv2
        self.people = people
        self.folder = Path(cache) / 'local/reference-appearance' / VERSION
        self.folder.mkdir(parents=True, exist_ok=True)
        self.profiles = []
        self.profile_reference_ids = []
        self.references = []
        self.human_references = False
        for reference in references:
            rgb, data = self.inspect(reference['path'])
            if data['people']:
                self.human_references = True
                profile = self.profile(rgb, data)
                if profile is not None:
                    self.profiles.append(profile)
                    self.profile_reference_ids.append(reference['id'])
            masked = self.masked(reference['path'], rgb, data)
            self.references.append({**reference, 'path': str(masked)})

    def inspect(self, path):
        import cv2
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Cannot read reference appearance image: {path}')
        if image.shape[1] > 960:
            image = cv2.resize(image, (960, max(2, round(image.shape[0]*960/image.shape[1]))))
        rgb = image[:, :, ::-1].copy()
        return rgb, self.people.inspect(rgb, minimum_face=20)

    @staticmethod
    def face_mask(rgb, data):
        import numpy as np
        mask = np.full(rgb.shape[:2], 255, dtype=np.uint8)
        height, width = mask.shape
        faces = [f['face'] for f in data['faces']]
        for x, y, w, h in data['body_boxes']:
            if not any(x <= fx+fw/2 <= x+w and y <= fy+fh/2 <= y+h for fx, fy, fw, fh in faces):
                faces.append([x, y, w, h*.35])
        for x, y, w, h in faces:
            left, top = max(0, int(x-w*.25)), max(0, int(y-h*.25))
            right, bottom = min(width, int(x+w*1.25)), min(height, int(y+h*1.2))
            mask[top:bottom, left:right] = 0
        return mask

    def masked(self, path, rgb=None, data=None):
        import cv2
        record = self.folder / (digest([VERSION, Path(path).read_bytes().hex()])+'.jpg')
        if record.is_file():
            return record
        if rgb is None:
            rgb, data = self.inspect(path)
        masked = rgb.copy()
        masked[self.face_mask(rgb, data) == 0] = 128
        cv2.imwrite(str(record), masked[:, :, ::-1])
        return record

    def profile(self, rgb, data):
        import cv2
        import numpy as np
        if not data['body_boxes']:
            return None
        height, width = rgb.shape[:2]
        x, y, w, h = max(data['body_boxes'], key=lambda box: box[2]*box[3])
        region = np.zeros((height, width), dtype=np.uint8)
        region[max(0, int(y)):min(height, int(y+h)), max(0, int(x)):min(width, int(x+w))] = 255
        region &= self.face_mask(rgb, data)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        # Wardrobe/accessory colours; face pixels never enter this descriptor.
        region[hsv[:, :, 1] < 45] = 0
        if np.count_nonzero(region) < 32:
            return None
        histogram = cv2.calcHist([hsv], [0, 1], region, [16, 4], [0, 180, 0, 256])
        cv2.normalize(histogram, histogram, norm_type=cv2.NORM_L1)
        return histogram

    def preview_score(self, rgb, data):
        import cv2
        scores = []
        for box in data['body_boxes']:
            profile = self.profile(rgb, {**data, 'body_boxes': [box]})
            if profile is not None:
                scores.extend(float(cv2.compareHist(profile, target, cv2.HISTCMP_INTERSECT))
                    for target in self.profiles)
        return max(scores) if scores else None

    @staticmethod
    def palette_compatible(profile, target):
        """Require distinct reference wardrobe colours, rather than one shared hue."""
        import numpy as np
        reference = target.sum(axis=1)
        observed = profile.sum(axis=1)
        anchors = []
        for index in np.argsort(reference)[::-1]:
            if reference[index] < .12:
                break
            if all(min(abs(int(index)-old), 16-abs(int(index)-old)) >= 3 for old in anchors):
                anchors.append(int(index))
            if len(anchors) == 2:
                break
        for index in anchors:
            bins = [(index-1)%16, index, (index+1)%16]
            if (observed[bins].sum() < max(.03, float(reference[bins].sum())*.15) or
                    observed[index] < max(.02, float(reference[index])*.15)):
                return False
        return True

    def compatible_references(self, path):
        if not self.profiles:
            return None
        rgb, data = self.inspect(path)
        compatible = set()
        for box in data['body_boxes']:
            profile = self.profile(rgb, {**data, 'body_boxes': [box]})
            if profile is not None:
                compatible.update(rid for target, rid in zip(self.profiles, self.profile_reference_ids)
                    if self.palette_compatible(profile, target))
        return compatible

    def supported_outfit_colors(self, reference_id):
        bands = {'red': [0, 15], 'orange': [1], 'yellow': [2, 3], 'green': [4, 5, 6],
                 'blue': [7, 8, 9, 10], 'purple': [11, 12], 'pink': [13, 14]}
        colors = set()
        for profile, rid in zip(self.profiles, self.profile_reference_ids):
            if rid == reference_id:
                hue = profile.sum(axis=1)
                colors.update(color for color, bins in bands.items() if float(hue[bins].sum()) >= .12)
        return colors
