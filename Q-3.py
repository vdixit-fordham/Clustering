import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import sqrt

xPoints = [0,1,1,2,3,5,4,5,6,7]
yPoints = [0,2,6,3,4,1,2,3,2,4]

#circle1 = plt.Circle((0, 0), 1.141, facecolor='none', edgecolor='black')
#circle2 = plt.Circle((1, 2), 1.141, facecolor='none', edgecolor='black')

figure = plt.figure()
fig, ax = plt.subplots()

plt.scatter(xPoints, yPoints)
for index in range(len(xPoints)):
    circle = plt.Circle((xPoints[index], yPoints[index]), sqrt(2), facecolor='none', edgecolor='black')
    ax.add_artist(circle)

#ax.add_artist(circle1)
#ax.add_artist(circle2)
ax.arrow(1, 2, 1, 1, color = 'r')
ax.arrow(2, 3, 1, 1, color = 'b')
ax.arrow(4, 2, 1, 1, color = 'r')
ax.arrow(5, 3, 1, -1, color = 'b')
ax.arrow(6, 2, -1, -1, color = 'r')
ax.arrow(5, 1, -1, 1, color = 'b')
plt.show()

fig.savefig("Plot_Question3.pdf")