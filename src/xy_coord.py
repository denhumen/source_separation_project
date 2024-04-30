import matplotlib.pyplot as plt
import numpy as np

micro_pos = [(1, 1), (3, 2)]
source_pos = [(0, 0), (6, 7)]

micro_x, micro_y = zip(*micro_pos)
source_x, source_y = zip(*source_pos)

plt.scatter(micro_x, micro_y, color='red', label='Microphone Positions')
plt.scatter(source_x, source_y, color='blue', label='Source Positions')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('XY-coordinate System with Points')


plt.axvline(x=0, color='black', linestyle='-')
plt.axhline(y=0, color='black', linestyle='-')

plt.savefig('xy_points.png')
plt.grid(True)
plt.xticks(np.arange(-10, 12, 2))
plt.yticks(np.arange(-10, 12, 2))
plt.show()

