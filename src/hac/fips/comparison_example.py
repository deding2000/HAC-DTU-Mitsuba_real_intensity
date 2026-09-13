# import metrics
from hac.fips import metrics

"""
1. Measure: The first measure is defined as meas1(A,B) = 1 - |A \cap B|/|A \cup B|
2. Measure: The second measure is defined as meas2(A,B) = (|A\B| + |B\A|)/(|A| + |B|)
"""
measure1, measure2 = metrics.relative_volume_difference_voxelized(
    "grid1s.stl", "grid2s.stl", pitch=0.05
)
print("The relative distance with the voxel measure1 is: " + str(measure1))
print("The relative distance with the voxel measure2 is: " + str(measure2))
