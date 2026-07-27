import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Work3_BEF_SBG.teacher.train_teacher_core import main

if __name__ == "__main__":
    main(["--teacher_type", "cnn", *sys.argv[1:]])
