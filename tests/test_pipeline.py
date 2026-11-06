import json
import tempfile
import unittest
from pathlib import Path
from argparse import Namespace
from pipeline import angle_delta, segment_angle, motion, prepare

class GeometryTests(unittest.TestCase):
    def test_angles(self):
        self.assertEqual(segment_angle([0,10],[0,0]),0)
        self.assertEqual(segment_angle([0,10],[10,10]),90)
        self.assertIsNone(segment_angle([0,0],[0,0]))
        self.assertEqual(angle_delta(-179,179),2)
    def test_movement(self):
        self.assertEqual(motion([3,-4],[0,0]),(3,4,5,'right-up'))
        self.assertEqual(motion([.1,.1],[0,0])[-1],'stationary')
    def test_prepare(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'source';root.mkdir()
            (root/'images').mkdir();(root/'labels').mkdir()
            (root/'schema.json').write_text(json.dumps({'keypoints':[{'name':'a'},{'name':'b'}],'edges':[[0,1]]}))
            for i in range(4):
                (root/'images'/f'{i}.png').write_bytes(b'fixture')
                (root/'labels'/f'{i}.txt').write_text('0 .5 .5 1 1 .5 .8 2 .5 .2 2')
            out=Path(temp)/'dataset'
            prepare(Namespace(source=str(root),output=str(out),seed=42,val=.25))
            self.assertEqual(len(list((out/'images/train').iterdir())),3)
            self.assertEqual(len(list((out/'images/val').iterdir())),1)
            with self.assertRaises(ValueError):
                prepare(Namespace(source=str(root),output=str(out),seed=42,val=.25))

if __name__=='__main__': unittest.main()
