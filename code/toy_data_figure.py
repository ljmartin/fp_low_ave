import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import norm
import utils

utils.set_mpl_params()
def rot(xy, radians):
    """Use numpy to build a rotation matrix and take the dot product."""
    x = xy[:,0]
    y = xy[:,1]
    c, s = np.cos(radians), np.sin(radians)
    j = np.array([[c, s], [-s, c]])
    m = np.dot(j, [x, y])
    return m.T



fig, ax = plt.subplots(1,3)
fig.set_figwidth(15*1.2)
fig.set_figheight(5*1.2)



####ONE:
pts = [[0,0],
      [1,-0.25],
      [0.35,0.35],
      [0.65,-0.55]]

labels = ['Test\nActives', 'Test\nInactives', 'Train\nActives', 'Train\nInactives']
offsets = [(50, -10), (-50, -10), (50, -10), (50,-10)]

for pt, m, c, lab, off in zip(pts, ['s', 's', 'o', 'o'], ['C1', 'C0', 'C1', 'C0'], labels, offsets):
    ax[0].scatter(pt[0], pt[1], marker = m, c=c,s=800, zorder=1)
    ax[0].annotate(lab, # this is the text
                 (pt[0],pt[1]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=off, # distance from text to points (x,y)
                 ha='center')
    
# ax[0].plot([0, 0.35], [0, 0.35], c='C2', label='Test Active > Train Active',zorder=0)
# ax[0].plot([0, 0.65], [0, -0.55], c='C4',label='Test Active > Train Inactive', zorder=0)
# ax[0].plot([1, 0.65], [-0.25, -0.55], c='C5',label='Test Inactive > Train Inactive', zorder=0)
# ax[0].plot([1, 0.35], [-0.25, 0.35], c='C6',label='Test Inactive > Train Active', zorder=0)

#clf = LogisticRegression()
#clf.fit(np.array([pts[2], pts[3]]), np.array([0,1]))

line_bias = np.array([-0.107875])
line_w = np.array([[0.1348843], [-0.40462549]])

print(line_bias)
print(line_w)
points_x = np.linspace(-0.1, 1.2, 10)
points_y=[(line_w[0]*x+line_bias)/(-1*line_w[1]) for x in points_x]
ax[0].plot(points_x, points_y, label='Learned hyperplane')

ax[0].plot([0, 0.35], [0, 0.35], c='C2', zorder=0, linestyle='--')
ax[0].plot([0, 0.65], [0, -0.55], c='C4', zorder=0, linestyle='--')
ax[0].plot([1, 0.65], [-0.25, -0.55], c='C5', zorder=0, linestyle='--')
ax[0].plot([1, 0.35], [-0.25, 0.35], c='C6', zorder=0, linestyle='--')
ax[0].set_xlim(-0.1,1.2)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_xlabel('X1')
ax[0].set_ylabel('X2')
ax[0].legend(loc='lower left')





####TWO:
samples = list()
sample = norm([0,0], [2,0.5]).rvs([500,2])
samples.append(rot(sample, np.pi*15/180))
sample = norm([0,0], [2,0.5]).rvs([500,2])
samples.append(rot(sample, np.pi*15/180) + np.array([2,5]))


coords = np.array([0,1]) #keep both dimensions
test_samples = list()
train_samples = list()

test_idx = np.random.choice(range(500), 100, replace=False)
train_idx = np.setdiff1d(range(500), test_idx)

test_samples.append(samples[0][test_idx][:,coords])
train_samples.append(samples[0][train_idx][:,coords])

test_idx = np.random.choice(range(500), 100, replace=False)
train_idx = np.setdiff1d(range(500), test_idx)

test_samples.append(samples[1][test_idx][:,coords])
train_samples.append(samples[1][train_idx][:,coords])



iTest_iTrain_D = cdist(test_samples[1][:,0][:,None], train_samples[1][:,0][:,None]).min(1)
iTest_aTrain_D = cdist(test_samples[1][:,0][:,None], train_samples[0][:,0][:,None]).min(1)

aTest_aTrain_D = cdist(test_samples[0][:,0][:,None], train_samples[0][:,0][:,None]).min(1)
aTest_iTrain_D = cdist(test_samples[0][:,0][:,None], train_samples[1][:,0][:,None]).min(1)

aTest_aTrain_S = np.mean( [ np.mean( aTest_aTrain_D < t ) for t in np.linspace( 0, 6, 50 ) ] )
aTest_iTrain_S = np.mean( [ np.mean( aTest_iTrain_D < t ) for t in np.linspace( 0, 6, 50 ) ] )
iTest_iTrain_S = np.mean( [ np.mean( iTest_iTrain_D < t ) for t in np.linspace( 0, 6, 50 ) ] )
iTest_aTrain_S = np.mean( [ np.mean( iTest_aTrain_D < t ) for t in np.linspace( 0, 6, 50 ) ] )
ave1 = aTest_aTrain_S-aTest_iTrain_S+iTest_iTrain_S-iTest_aTrain_S
print(ave1)

iTest_iTrain_D = cdist(test_samples[1][:,1][:,None], train_samples[1][:,1][:,None]).min(1)
iTest_aTrain_D = cdist(test_samples[1][:,1][:,None], train_samples[0][:,1][:,None]).min(1)

aTest_aTrain_D = cdist(test_samples[0][:,1][:,None], train_samples[0][:,1][:,None]).min(1)
aTest_iTrain_D = cdist(test_samples[0][:,1][:,None], train_samples[1][:,1][:,None]).min(1)

aTest_aTrain_S = np.mean( [ np.mean( aTest_aTrain_D < t ) for t in np.linspace( 0, 6, 50 ) ] )
aTest_iTrain_S = np.mean( [ np.mean( aTest_iTrain_D < t ) for t in np.linspace( 0, 6, 50 ) ] )
iTest_iTrain_S = np.mean( [ np.mean( iTest_iTrain_D < t ) for t in np.linspace( 0, 6, 50 ) ] )
iTest_aTrain_S = np.mean( [ np.mean( iTest_aTrain_D < t ) for t in np.linspace( 0, 6, 50 ) ] )
ave2 = aTest_aTrain_S-aTest_iTrain_S+iTest_iTrain_S-iTest_aTrain_S
print(ave2)


ax[1].scatter(train_samples[0][:,0], train_samples[0][:,1], c='C1', marker='o', label='Train Actives')
ax[1].scatter(test_samples[0][:,0], test_samples[0][:,1], c='C1', marker='s', label='Test Actives')

ax[1].scatter(train_samples[1][:,0], train_samples[1][:,1], c='C0', marker='o', label='Train Inactives')
ax[1].scatter(test_samples[1][:,0], test_samples[1][:,1], c='C0', marker='s', label='Test Inactives')
ax[1].legend(loc= 'lower right', ncol=2)
ax[1].set_xlabel(f'X1, AVE = {np.around(ave1,3)}')
ax[1].set_ylabel(f'X2, AVE = {np.around(ave2,3)}')
ax[1].set_xticks([])
ax[1].set_yticks([])





####Three:
c1 = [0,0] 
c2 = [1,0]
c3 = [1,1]
c4 = [0,1]

c3 = [0,1]
c4 = [1,1]
v = [0.1, 0.1]

s = lambda c, v: norm(c, v).rvs([200,2])

pts = [s(center, v) for center in [c1,c2,c3,c4]]


labels = ['Train Actives', 'Test Actives', 'Train Inactives', 'Test Inactives']

handles_one = list()
for center, color, marker, label in zip(pts, ['C1', 'C1', 'C0', 'C0'], ['o', 's', 'o', 's'], labels):
    handles_one.append(ax[2].scatter(*center.T, c=color, edgecolor='black', marker=marker, label=label))
    
handles_two = list()
for center, color, marker, label in zip(pts, ['C1', 'C1', 'C0', 'C0'], ['o', 's', 's', 'o'], labels):
    handles_two.append(ax[2].scatter([],[], c=color, edgecolor='black', marker=marker, label=label))


    
iTest_iTrain_D = cdist(pts[3], pts[2]).min(1)
iTest_aTrain_D = cdist(pts[3], pts[0]).min(1)

aTest_aTrain_D = cdist(pts[1], pts[0]).min(1)
aTest_iTrain_D = cdist(pts[1], pts[2]).min(1)

aTest_aTrain_S = np.mean( [ np.mean( aTest_aTrain_D < t ) for t in np.linspace( 0, 1.5, 50 ) ] )
aTest_iTrain_S = np.mean( [ np.mean( aTest_iTrain_D < t ) for t in np.linspace( 0, 1.5, 50 ) ] )
iTest_iTrain_S = np.mean( [ np.mean( iTest_iTrain_D < t ) for t in np.linspace( 0, 1.5, 50 ) ] )
iTest_aTrain_S = np.mean( [ np.mean( iTest_aTrain_D < t ) for t in np.linspace( 0, 1.5, 50 ) ] )
ave_one = aTest_aTrain_S-aTest_iTrain_S+iTest_iTrain_S-iTest_aTrain_S
print(ave_one)

iTest_iTrain_D = cdist(pts[2], pts[3]).min(1)
iTest_aTrain_D = cdist(pts[2], pts[0]).min(1)

aTest_aTrain_D = cdist(pts[1], pts[0]).min(1)
aTest_iTrain_D = cdist(pts[1], pts[3]).min(1)

aTest_aTrain_S = np.mean( [ np.mean( aTest_aTrain_D < t ) for t in np.linspace( 0, 1.5, 50 ) ] )
aTest_iTrain_S = np.mean( [ np.mean( aTest_iTrain_D < t ) for t in np.linspace( 0, 1.5, 50 ) ] )
iTest_iTrain_S = np.mean( [ np.mean( iTest_iTrain_D < t ) for t in np.linspace( 0, 1.5, 50 ) ] )
iTest_aTrain_S = np.mean( [ np.mean( iTest_aTrain_D < t ) for t in np.linspace( 0, 1.5, 50 ) ] )
ave_two = aTest_aTrain_S-aTest_iTrain_S+iTest_iTrain_S-iTest_aTrain_S
print(ave_two)
    
    
    
    
    
    
    
    
leg = ax[2].legend(title=f'AVE = {np.around(ave_one, 3)}',handles = handles_one, ncol=2, loc='lower center')
ax[2].add_artist(leg)


ax[2].legend(title=f'AVE = {np.around(ave_two, 3)}', handles = handles_two, ncol=2, loc='upper center')
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[2].set_xlabel('X1')
ax[2].set_ylabel('X2')
ax[2].set_xlim(-0.75,1.75)
ax[2].set_ylim(-0.75, 1.75)


utils.plot_fig_label(ax[0], 'A.')
utils.plot_fig_label(ax[1], 'B.')
utils.plot_fig_label(ax[2], 'C.')

fig.savefig('toy_data.png')
