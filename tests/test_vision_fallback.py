import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import unittest
import numpy as np
from PIL import Image
from khs.vision import FaceAnalyzer

class FallbackTests(unittest.TestCase):
    def test_landmark_miss_calls_optional_cpu_fallback(self):
        analyzer=FaceAnalyzer('missing.task'); analyzer.mode='legacy'
        analyzer.engine=NS(process=lambda pixels:NS(multi_face_landmarks=[]))
        with patch.dict('sys.modules',{'mediapipe':NS()}),patch.object(analyzer,'_fallback',return_value=[{'box':(1,2,3,4)}]) as fallback:
            result=analyzer._detect(Image.new('RGB',(32,32)))
            self.assertEqual(result,[{'box':(1,2,3,4)}]); fallback.assert_called_once()

    def test_yunet_is_cpu_only_and_keeps_pose_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'face_detection_yunet_2023mar.onnx').touch()
            rows=np.array([[8,8,20,24,13,14,23,14,18,20,14,26,23,26,.9]])
            detector=NS(setInputSize=Mock(),detect=Mock(return_value=(None,rows)))
            cv=NS(dnn=NS(DNN_BACKEND_OPENCV=3,DNN_TARGET_CPU=0),FaceDetectorYN_create=Mock(return_value=detector))
            analyzer=FaceAnalyzer(root/'face_landmarker.task')
            with patch.dict('sys.modules',{'cv2':cv}):
                found=analyzer._fallback(Image.new('RGB',(64,64)))
            self.assertEqual(cv.FaceDetectorYN_create.call_args.args[-2:],(3,0))
            self.assertEqual(found[0]['box'],(8,8,28,32))
            self.assertEqual(found[0]['pose_source'],'YuNet 2D estimate')
            self.assertEqual(found[0]['head_h'],24)
            analyzer.close(); self.assertIsNone(analyzer.fallback_detector)

if __name__=='__main__': unittest.main()
