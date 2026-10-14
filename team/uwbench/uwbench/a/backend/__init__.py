"""A1/A2 acoustic-optic smoother (W3-02 P08, W4-01 P10): GTSAM iSAM2 over raw IMU preintegration, stereo reprojection
and sonar range-bearing. A1 uses fixed sonar weights; A2 adds the bounded scalar weight in weights.py.

cli.py is `uwbench run-a`, diagnose.py is `uwbench diagnose-a` (a read-only hook on the same smoother)."""
