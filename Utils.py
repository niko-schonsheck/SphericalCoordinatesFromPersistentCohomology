import matplotlib.pyplot as plt
import numpy as np
import math
import sklearn
import tadasets
import scipy.stats
from ripser import ripser
from persim import plot_diagrams
from sklearn.neighbors import NearestNeighbors
from math import remainder
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from PIL import Image
from skimage.io import imread
from skimage.transform import rescale
from skimage.util import crop

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs, lw=2)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def plot_pts(pts, title='None'):
    fig = plt.figure()
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    for i in range(4):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        ax.plot_wireframe(x, y, z, color='b', linewidth=.1)
        # Add points
        for j in range(len(pts)):
            ax.scatter(pts[:][j][0], pts[:][j][1], pts[:][j][2], marker='*', color='r', s=100)
        if i == 1:
            ax.view_init(-70, -70)
        elif i == 2:
            ax.view_init(70, 70)
        elif i == 3:
            ax.view_init(70, 0)
    if title != 'None':
        plt.suptitle(title)
    plt.show()


def plot_data_and_results(data, pts, title='None', c='None'):
    fig = plt.figure()
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    if c == 'None':
        c = np.sum(data, axis=1).tolist()
        c = c - np.min(c)
        c = c/np.max(c)
    for i in range(6):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        if i < 3:
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker='.', c=c, s=100)
            ax.set_box_aspect((np.ptp(data[:, 0]), np.ptp(data[:, 1]), np.ptp(data[:, 2])))
            #ax.set_box_aspect(aspect=(1, 1, 1))
        else:
            ax.plot_wireframe(x, y, z, color='b', linewidth=.1)
            ax.scatter(pts[:,0], pts[:,1], pts[:,2], marker='.', c=c, s=100)
        if i == 1 or i == 4:
            ax.view_init(-70, -70)
        elif i == 2 or i == 5:
            ax.view_init(70, 70)
        if i == 1:
            plt.title('Input Data')
        if i == 4:
            plt.title('Results')
    if title != 'None':
        plt.suptitle(title)
    plt.show()

def drawLine(start, end, ax, c='r', w=1):
    seg = np.array([start, end])
    ax.plot(seg[:,0], seg[:,1], seg[:,2],c=c, linewidth=w)


def plot_graph_with_cocycles(data, simplices, values, c, title='None'):
    fig = plt.figure()
    simplices = np.array(simplices)
    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker='.', c=c, s=100)
        ax.set_box_aspect((np.ptp(data[:, 0]), np.ptp(data[:, 1]), np.ptp(data[:, 2])))
        for j in range(len(simplices)):
            if values[j] == 0:
                drawLine(data[simplices[j, 0],:], data[simplices[j, 1],:], ax, c='k', w=.1)
                drawLine(data[simplices[j, 0],:], data[simplices[j, 2],:], ax, c='k', w=.1)
                drawLine(data[simplices[j, 2],:], data[simplices[j, 1],:], ax, c='k', w=.1)
            else:
                drawLine(data[simplices[j, 0],:], data[simplices[j, 1],:], ax, c='g', w=1.5)
                drawLine(data[simplices[j, 0],:], data[simplices[j, 2],:], ax, c='g', w=1.5)
                drawLine(data[simplices[j, 2],:], data[simplices[j, 1],:], ax, c='g', w=1.5)

        if i == 1:
            ax.view_init(-70, -70)
        if i == 2:
                ax.view_init(70, 70)
    if title != 'None':
        plt.suptitle(title)
    plt.show()


def align_spheres(pts1, pts2):
    x_bar = (1/len(pts1))*np.sum(pts1, axis=0)
    y_bar = (1/len(pts2))*np.sum(pts2, axis=0)
    X = (pts1 - x_bar).T
    Y = (pts2 - y_bar).T
    XYT = np.matmul(X,Y.T)
    U, D, VT = np.linalg.svd(XYT)
    S = np.eye(3)
    #if np.abs(np.linalg.det(U)*np.linalg.det(VT.T) - 1) < 1e-12:
    #    S = np.eye(3)
    #else:
    #    S = np.eye(3)
    #    S[-1,-1] = -1
    R = np.matmul(U, np.matmul(S, VT))
    new_pts1 = np.matmul(pts1, R)
    return R, new_pts1


def coord_plot_sphere(pts, sphere, plot=1):
    p = np.zeros(pts.shape)
    s = np.zeros(p.shape)
    for i in range(len(p)):
        p[i,:] = cart2sph(pts[i,0], pts[i,1], pts[i,2])
        s[i,:] = cart2sph(sphere[i,0], sphere[i,1], sphere[i,2])
    if plot == 1:
        plt.figure()
        plt.subplot(1,2,1)
        plt.scatter(p[:,0], s[:,0], c=p[:,0], cmap='hsv')
        plt.title('Azimuth')
        plt.subplot(1,2,2)
        plt.scatter(p[:,1], s[:,1], c=p[:,1], cmap='hsv')
        plt.title('Elivation')
        plt.show()
    else:
        return p,s

def double_sphere_aligment(input, sphere):
    R1, pts1_rotated = align_spheres(input[:, :3], sphere)
    R2, pts2_rotated = align_spheres(input[:, 3:], sphere)
    p1, s1 = coord_plot_sphere(pts1_rotated, input[:, :3], plot=0)
    p2, s2 = coord_plot_sphere(pts2_rotated, input[:, 3:], plot=0)
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.scatter(p1[:, 0], s1[:, 0], c=p1[:, 0], cmap='hsv')
    plt.title('S1 Azimuth')
    plt.subplot(2, 2, 2)
    plt.scatter(p1[:, 1], s1[:, 1], c=p1[:, 1], cmap='hsv')
    plt.title('S1 Elivation')
    plt.subplot(2, 2, 3)
    plt.scatter(p2[:, 0], s2[:, 0], c=p2[:, 0], cmap='hsv')
    plt.title('S2 Azimuth')
    plt.subplot(2, 2, 4)
    plt.scatter(p2[:, 1], s2[:, 1], c=p2[:, 1], cmap='hsv')
    plt.title('S2 Elivation')
    plt.show()


def short_arc_length(pt1, pt2):
    dot =  np.minimum(1.0, pt1.dot(pt2.T))
    d1 = np.arccos(dot)
    return d1

# def compute_spherical_trg_bary(pt1, pt2, pt3, guess):
#     trg_bary = (1/3)*(pt1+pt2+pt3)
#     b1 = trg_bary/np.linalg.norm(trg_bary)
#     b2 = -b1
#     if  np.abs(b1.dot(guess)) < .5:
#         print('Guess may be bad')
#     if b1.dot(guess) > b2.dot(guess):
#         return b1
#     else:
#         return b2


def compute_spherical_trg_bary(pt1, pt2, pt3, guess):
    num = pt1 + pt2 + pt3
    denom = np.sqrt(3+2*pt1.dot(pt2) + 2*pt1.dot(pt3) + 2*pt2.dot(pt3))
    result = num/denom
    neg = - result
    r_coeff = result.dot(guess)
    n_coeff = neg.dot(guess)
    # if  np.abs(r_coeff) < .5:
    #     print('Guess may be bad')
    if r_coeff > n_coeff:
        return result
    else:
        return neg


def compute_spherical_trg_area(pt1, pt2, pt3, bary):
    a = short_arc_length(pt1, pt2)
    b = short_arc_length(pt2, pt3)
    c = short_arc_length(pt3, pt1)
    s = (a+b+c)/2
    temp = np.tan(s/2)*np.tan((s-a)/2)*np.tan((s-b)/2)*np.tan((s-c)/2)
    E = 4*np.arctan(np.sqrt(np.abs(temp)))
    num = pt1 + pt2 + pt3
    denom = np.sqrt(3+2*pt1.dot(pt2) + 2*pt1.dot(pt3) + 2*pt2.dot(pt3))
    result = num/denom
    if result.dot(bary) > 0:
        return E
    else:
        return 4*np.pi-E


def compute_trangent_grad(pt, bary):
    if (pt == bary).all():
        return np.zeros(pt.shape)
    else:
        perp = np.cross(pt, bary)
        perp = perp/np.linalg.norm(perp+1e-5)
        d1 = np.cross(pt, perp)
        # d = pt - bary
        # proj = pt.dot(d)
        # d = d - proj*pt
        # d = d/np.linalg.norm(d)
        # print(np.linalg.norm(d-d1))
    return d1


def compute_spherical_string_grad_single_trg(pt1, pt2, pt3, bary, orient):
    A = compute_spherical_trg_area(pt1, pt2, pt3, bary)
    if A == 4*np.pi:
        if orient == 1:
            g1 = A*np.array([0, 1, 0])
            g2 = A*np.array([0, -np.sqrt(2)/2, np.sqrt(2)/2])
            g3 = A*np.array([0, -np.sqrt(2)/2, -np.sqrt(2)/2])
        else:
            g1 = A*np.array([0, 1, 0])
            g2 = A*np.array([0, -np.sqrt(2)/2, -np.sqrt(2)/2])
            g3 = A*np.array([0, -np.sqrt(2)/2, np.sqrt(2)/2])
    elif A == 0:
        g1 = 0
        g2 = 0
        g3 = 0
    else:
        g1 = A*compute_trangent_grad(pt1, bary)
        g2 = A*compute_trangent_grad(pt2, bary)
        g3 = A*compute_trangent_grad(pt3, bary)
    return g1, g2, g3, A

def compute_spherical_spring_grad_single_trg(pt1, pt2, pt3, bary, orient, S, K):
    if (pt1 == pt2).all():
        if (pt1 == pt3).all():
            if pt1.dot(bary) > 0:
                    D = K*S
                    if orient == 1:
                        g1 = -1*D*np.array([0, 1, 0])
                        g2 = -1*D*np.array([0, -np.sqrt(2), np.sqrt(2)])
                        g3 = -1*D*np.array([0, -np.sqrt(2), -np.sqrt(2)])
                    else: #orient = =1
                        g1 = -1*D*np.array([0, 1, 0])
                        g2 = -1*D*np.array([0, -np.sqrt(2), -np.sqrt(2)])
                        g3 = -1*D*np.array([0, -np.sqrt(2), np.sqrt(2)])
            else:
                D = K*(4*np.pi-S)
                if orient == 1:
                    g1 = D*np.array([0, 1, 0])
                    g2 = D*np.array([0, -np.sqrt(2), np.sqrt(2)])
                    g3 = D*np.array([0, -np.sqrt(2), -np.sqrt(2)])
                else:
                    g1 = D*np.array([0, 1, 0])
                    g2 = D*np.array([0, -np.sqrt(2), -np.sqrt(2)])
                    g3 = D*np.array([0, -np.sqrt(2), np.sqrt(2)])
        else:
            g1 = 0
            g2 = 0
            g3 = 0
            D = K*S
    else:
        A = compute_spherical_trg_area(pt1, pt2, pt3, bary)
        sign = np.sign(A - S)
        D = K*np.abs(A-S)
        g1 = sign*D*compute_trangent_grad(pt1, bary) #(pt1-bary)
        g2 = sign*D*compute_trangent_grad(pt2, bary)
        g3 = sign*D*compute_trangent_grad(pt3, bary)
    return g1, g2, g3, D


def compute_grad_and_NRG(pts, trg_list, bary, orient, E_type, S=0, K=1):
    grad = np.zeros(pts.shape)
    n_trg = len(trg_list)
    E_list = np.zeros(n_trg)
    A_list = np.zeros(n_trg)
    E = 0
    for i in range(n_trg):
        pt1 = pts[trg_list[i][0]]
        pt2 = pts[trg_list[i][1]]
        pt3 = pts[trg_list[i][2]]
        if E_type == 0:
            g1, g2, g3, A = compute_spherical_string_grad_single_trg(pt1, pt2, pt3, bary[i], orient[i])
        elif E_type == 1:
            g1, g2, g3, A = compute_spherical_spring_grad_single_trg(pt1, pt2, pt3, bary[i], orient[i], S, K)
        grad[trg_list[i][0]] = grad[trg_list[i][0]] + g1
        grad[trg_list[i][1]] = grad[trg_list[i][1]] + g2
        grad[trg_list[i][2]] = grad[trg_list[i][2]] + g3
        E = E + (1/2)*(A**2)
        E_list[i] = (1/2)*(A**2)
        A_list[i] = A
    E = (1/n_trg)*E
    return E, E_list, A_list, grad


def center_mass_mobius(pts, bary, alpha=1):
    c = (1/len(pts))*np.sum(pts, axis=0)
    pts = pts - alpha*c
    bary = bary - alpha*c
    #for i in range(len(pts)):
    #    pts[i,:] = pts[i,:]/np.linalg.norm(pts[i,:])
    pts = sklearn.preprocessing.normalize(pts, norm="l2")
    bary = sklearn.preprocessing.normalize(bary, norm="l2")
    return pts, bary


def compute_grad_step(pts, trg_list, grad, bary, alpha, iter):
    #update pts
    pts = pts - alpha*grad
    pts = sklearn.preprocessing.normalize(pts, norm="l2")

    #Mobius transform
    if iter > 100:
        if iter % 101 == 0:
            pts, bary = center_mass_mobius(pts, bary, alpha=1)
    if iter > 250:
        pts, bary = center_mass_mobius(pts, bary, alpha=.1)


    #update bary
    for i in range(len(trg_list)):
        bary[i,:] = compute_spherical_trg_bary(pts[trg_list[i][0]], pts[trg_list[i][1]], pts[trg_list[i][2]],
                                   guess=bary[i, :])

    return pts, bary




def initial_ripser_calculation(data_set, coefficient, plot=0):
    result = ripser(data_set, coeff=coefficient, do_cocycles=True, maxdim=2)
    diagrams = result['dgms']
    cocycles = result['cocycles']
    D = result['dperm2all']

    if plot == 1:
        fig, ax = plt.subplots()
        for i in range(0, len(cocycles[2][:])):
            birth = result['dgms'][2][i][0]
            death = result['dgms'][2][i][1]
            ax.hlines(y=i, xmin = birth, xmax = death, linewidth = 2, color = 'b')
        plt.title('Bar Codes')
        plt.show()
    return cocycles, D, result


# WARNING: we are counting on ripser to always give simplices in descending order, which it does seem to do
def dim_2_get_edges_and_cohomology_info(distance_matrix, selected_cocycle, max_nhood_size):
    all_2_simplices = []
    all_values = []
    num_points = len(distance_matrix)
    for i in range(num_points):
        for j in range(i):
            for k in range(j):
                if distance_matrix[i, j] <= max_nhood_size and distance_matrix[i, k] <= max_nhood_size and \
                        distance_matrix[j, k] <= max_nhood_size:
                    all_2_simplices.append([i, j, k])
                    value = 0
                    for [v_1, v_2, v_3, val] in selected_cocycle:
                        if i == v_1 and j == v_2 and k == v_3:
                            value = value + val
                    all_values.append(value)
    return all_2_simplices, all_values

def generate_noisy_sphere(num_points = 50, radius = 1, noise = 0.0):
    random_float_array = np.random.uniform(-1.0, 1.0, [num_points, 3])
    #print(random_float_array)
    for i in range(num_points):
        norm = np.sqrt(random_float_array[i][0]**2 + random_float_array[i][1]**2 + random_float_array[i][2]**2)
        for j in range(3):
            random_float_array[i][j] = (radius+noise*np.random.randn())*(random_float_array[i][j]/norm) #(radius+np.random.uniform(-noise, noise))*(random_float_array[i][j]/norm)
    return random_float_array


def drawLineColored(X, C):
    for i in range(X.shape[0] - 1):
        plt.plot(X[i:i + 2, 0], X[i:i + 2, 1], c=C[i, :], linewidth=1)


def drawLineColored_2(X):
    for i in range(X.shape[0] - 1):
        plt.plot(X[i:i + 2, 0], X[i:i + 2, 1], c='r', linewidth=1.5)


def drawLine(start, end):
    for i in range(start.shape[0]):
        plt.plot(start[i:i + 2], end[i:i + 2], 'ro-')


def drawLineOrient(start, end, orient):
    for i in range(start.shape[0]):
        if orient[i] == -1:
            C = 'red'
        else:
            C = 'blue'
        dx = end[i, 0] - start[i, 0]
        dy = end[i, 1] - start[i, 1]
        plt.arrow(start[i, 0], start[i, 1], dx, dy, color=C, head_width=.1)

        # plt.plot(start[i:i + 2], end[i:i + 2], '-', c=C)


def plotCocycle2D(D, X, cocycle, thresh):
    """
    Given a 2D point cloud X, display a cocycle projected
    onto edges under a given threshold "thresh"
    """
    # Plot all edges under the threshold
    N = X.shape[0]
    t = np.linspace(0, 1, 10)
    c = plt.get_cmap('Greys')
    C = c(np.array(np.round(np.linspace(0, 255, len(t))), dtype=np.int32))
    C = C[:, 0:3]

    for i in range(N):
        for j in range(N):
            if D[i, j] <= thresh:
                Y = np.zeros((len(t), 2))
                Y[:, 0] = X[i, 0] + t * (X[j, 0] - X[i, 0])
                Y[:, 1] = X[i, 1] + t * (X[j, 1] - X[i, 1])
                drawLineColored(Y, C)
    # Plot cocycle projected to edges under the chosen threshold
    for k in range(cocycle.shape[0]):
        [i, j, val] = cocycle[k, :]
        if D[i, j] <= thresh:
            [i, j] = [min(i, j), max(i, j)]
            a = 0.5 * (X[i, :] + X[j, :])
            plt.text(a[0], a[1], '%g' % val, color='b', weight="bold")
    # Plot vertex labels
    if N < 50:
        for i in range(N):
            plt.text(X[i, 0], X[i, 1], '%i' % i, color='r', weight="bold")
        plt.axis('equal')


def plotCocycle2D_2(D, X, cocycle, thresh):
    """
    Given a 2D point cloud X, display a cocycle projected
    onto edges under a given threshold "thresh"
    """
    # Plot all edges under the threshold
    N = X.shape[0]
    t = np.linspace(0, 1, 10)
    c = plt.get_cmap('Greys')
    C = c(np.array(np.round(np.linspace(0, 255, len(t))), dtype=np.int32))
    C = C[:, 0:3]

    for i in range(N):
        for j in range(N):
            if D[i, j] <= thresh:
                Y = np.zeros((len(t), 2))
                Y[:, 0] = X[i, 0] + t * (X[j, 0] - X[i, 0])
                Y[:, 1] = X[i, 1] + t * (X[j, 1] - X[i, 1])
                drawLineColored(Y, C)
    for k in range(cocycle.shape[0]):
        [i, j, val] = cocycle[k, :]
        if D[i, j] <= thresh:
            if val != 0:
                Y = np.zeros((len(t), 2))
                Y[:, 0] = X[i, 0] + t * (X[j, 0] - X[i, 0])
                Y[:, 1] = X[i, 1] + t * (X[j, 1] - X[i, 1])
                drawLineColored_2(Y)


def getEdgeInfo(x, field, dataType, plot=1, max=1, return_ripser=0):
    if dataType == 0:
        edges = [[0, 1], [2, 1], [0, 2]]
        values = [1, 0, 0]
        return edges, values, 0, 0, 0
    else:
        number_of_points = len(x)

        # Get Ripser
        print('Doing ripser stuff')
        result = ripser(x, coeff=field, do_cocycles=True)  # 5
        diagrams = result['dgms']
        cocycles = result['cocycles']
        D = result['dperm2all']

        # PD diagram
        dgm1 = diagrams[1]
        if max == 1:
            idx = np.argmax(dgm1[:, 1] - dgm1[:, 0])
        if max == -1:
            idx = np.argmin(dgm1[:, 1] - dgm1[:, 0])
        if plot == 1:
            plot_diagrams(diagrams, show=False)
            plt.scatter(dgm1[idx, 0], dgm1[idx, 1], 20, 'k', 'x')
            plt.title("Max 1D birth = %.3g, death = %.3g" % (dgm1[idx, 0], dgm1[idx, 1]))
            plt.show()

        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        # Compute Cycles
        print('Doing Niko Stuff')
        cocycle = cocycles[1][idx]
        # I'm going to change this 6/1 to a lower threshold so we have fewer edges
        # Original: thresh = dgm1[idx, 1] - 0.0001 # Project cocycle onto edges STRICLY LESS THAN death time
        thresh = (1 - .4) * dgm1[idx, 0] + .4 * dgm1[idx, 1]
        if plot == 1:
            plotCocycle2D(D, x, cocycle, thresh)
            plt.title("1-Form Thresh=%g" % thresh)
            plt.show()

        # Fix cycles
        fixed_cocycle = []
        for x in range(len(cocycle)):
            i = cocycle[x][0]
            j = cocycle[x][1]
            if D[i, j] <= thresh:
                fixed_cocycle.append([i, j, cocycle[x][2]])
        # print(fixed_cocycle)
        fixed_cocycle_values = []
        for (x, y, z) in fixed_cocycle:
            fixed_cocycle_values.append(z)

        edges = []
        for i in range(number_of_points):
            for j in range(i):
                if D[i, j] <= thresh:
                    edges.append([i, j])

        # print("Edges with nontrivial cohomology value:")
        nontrivial_edges = []
        for (x, y, z) in fixed_cocycle:
            nontrivial_edges.append([x, y])
        # print(nontrivial_edges)

        # print('Trivial Edges')
        trivial_edges = edges.copy()
        for x in edges:
            for k in range(len(nontrivial_edges)):
                if x == nontrivial_edges[k]:
                    trivial_edges.remove(x)
        # print(trivial_edges)

        if len(trivial_edges) + len(nontrivial_edges) == len(edges):
            print("Edges accounted for, okay to proceed")
        else:
            print("Something is wrong")

        for edge in trivial_edges:
            edge_with_value = edge.copy()
            edge_with_value.append(0)

        all_edges = np.concatenate((nontrivial_edges, trivial_edges,), axis=0)
        all_values = np.concatenate((fixed_cocycle_values, np.zeros(len(trivial_edges))), axis=0)
        # print(fixed_cocycle)
        # print(all_edges)
        # print(all_values)

    if return_ripser == 1:
        return all_edges, all_values, D, cocycle, thresh, result
    else:
        return all_edges, all_values, D, cocycle, thresh


def southern_hemisphere_map(theta):
    return np.pi / 4 + 0.5 * theta


def furthest_point(lines, x_min, x_max, y_min, y_max, n):
    pts = np.array(np.meshgrid(np.linspace(x_min, x_max, int(n * (x_max - x_min))),
                               np.linspace(y_min, y_max, int(n * (y_max - y_min)))))
    pts = np.transpose(np.reshape(pts, [2, int(n * (x_max - x_min)) * int(n * (y_max - y_min))]))
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(lines)
    distances, indices = nbrs.kneighbors(pts)
    max_ind = np.argmax(distances)
    return pts[max_ind, :]


def calculate_circle_geodesic(phi, theta):
    a = phi % (2 * np.pi)
    b = theta % (2 * np.pi)
    if a > b:
        x = a - b
    elif a <= b:
        x = b - a
    y = 2 * np.pi - x
    if abs(x) <= abs(y):
        if a > b:
            return x
        else:
            return -x
    else:
        if a > b:
            return y
        else:
            return -y


def integrate_circle_distances(array, smoothness_tolerance=np.pi / 3):
    bits_of_harmonic_energy = []
    for i in range(0, len(array) - 1):
        x = calculate_circle_geodesic(array[i + 1], array[i])
        if abs(x) < smoothness_tolerance:
            bits_of_harmonic_energy.append(x)
        else:
            print("Throwing out value at index", i)
    sum_of_bits = np.sum(bits_of_harmonic_energy)
    # I changed this to just take the remainder when divided by 2pi, then take sum, subtract remainder,
    # divide by 2pi and take nearest integer.
    partial = remainder(sum_of_bits, 2 * np.pi)
    degree = round((sum_of_bits - partial) / (2 * np.pi))
    harmonic_energy = (degree, partial)
    # reconstructed_sum = degree*2*np.pi + partial
    # if math.isclose(sum_of_bits, reconstructed_sum, rel_tol=0.01) == True:
    #    return harmonic_energy
    # else:
    #    raise Exception ("Something went wrong in calculating harmonic energy.")
    return sum_of_bits, harmonic_energy


def pointwise_add_alpha_beta_no_mod(alpha_x, beta_x, p_1, p_2, min_alpha, max_alpha, min_beta, max_beta, step_size):
    t = 0
    x_coord = alpha_x
    y_coord = beta_x
    x_delta = alpha_x - p_1
    y_delta = beta_x - p_2
    while x_coord < max_alpha and x_coord > min_alpha and y_coord < max_beta and y_coord > min_beta:
        t = t + step_size
        x_coord = x_coord + t * x_delta
        y_coord = y_coord + t * y_delta
    if x_coord >= max_alpha or x_coord <= min_alpha:
        return y_coord  # depending on situation: % (2*np.pi)
    elif y_coord >= max_beta or y_coord <= min_beta:
        return x_coord  # depending on situation: % (2*np.pi)
    else:
        raise Exception("Something has gone awry.")


def pointwise_add_alpha_beta_with_mod(alpha_x, beta_x, p_1, p_2, step_size=0.01):
    t = 0
    x_coord = alpha_x
    y_coord = beta_x
    x_delta = alpha_x - p_1
    y_delta = beta_x - p_2
    while x_coord < 2 * np.pi and x_coord > 0 and y_coord < 2 * np.pi and y_coord > 0:
        t = t + step_size
        x_coord = x_coord + t * x_delta
        y_coord = y_coord + t * y_delta
    if x_coord >= 2 * np.pi or x_coord <= 0:
        return y_coord % (2 * np.pi)
    elif y_coord >= 2 * np.pi or y_coord <= 0:
        return x_coord % (2 * np.pi)
    else:
        raise Exception("Something has gone awry.")


def deg_map(n, theta):
    return n * theta


def cohomotopy_sum(image_alpha, image_beta, number_of_pieces,
                   proj_point_threshold_distance=0.5, step_size=0.01):
    # Populate array with image of alpha x beta over whole domain
    image_alpha_cross_beta = np.array((image_alpha, image_beta)).T

    # These are your ultimate S^1 coordinates
    ultimate_S1_coords = []

    max_alpha = max(image_alpha)
    min_alpha = min(image_alpha)
    max_beta = max(image_beta)
    min_beta = min(image_beta)

    list_of_centers = [[np.pi / 2, np.pi / 2],
                       [np.pi, np.pi / 2],
                       [3 * np.pi / 2, np.pi / 2],
                       [np.pi / 2, np.pi],
                       [np.pi, np.pi],
                       [3 * np.pi / 2, np.pi],
                       [np.pi / 2, 3 * np.pi / 2],
                       [np.pi, 3 * np.pi / 2],
                       [3 * np.pi / 2, 3 * np.pi / 2]]

    energies_with_all_info = []

    for i in range(0, len(list_of_centers)):
        center_x = list_of_centers[i][0]
        center_y = list_of_centers[i][1]
        check_for_too_close = []
        for j in range(0, number_of_pieces):
            rect_dist = np.abs(center_x - image_alpha[j]) + np.abs(center_y - image_beta[j])
            check_for_too_close.append(rect_dist)
        min_dist = min(check_for_too_close)
        if min_dist < proj_point_threshold_distance:
            # print("Center", i, "was too close.")
            continue
        aux_y_axis_with_mod = []
        for k in range(0, number_of_pieces):
            aux_y_axis_with_mod.append(
                pointwise_add_alpha_beta_with_mod(
                    image_alpha[k],
                    image_beta[k],
                    center_x,
                    center_y,
                    step_size))
        nrg = integrate_circle_distances(aux_y_axis_with_mod, np.pi)
        energies_with_all_info.append([i, nrg])

    center_index = min(energies_with_all_info, key=lambda x: np.abs(x[1][0]))[0]
    optimal_center = list_of_centers[center_index]
    optimal_projection_energy = min(energies_with_all_info, key=lambda x: np.abs(x[1][0]))[1]

    for k in range(number_of_pieces):
        ultimate_S1_coords.append(
            pointwise_add_alpha_beta_with_mod(
                image_alpha[k],
                image_beta[k],
                optimal_center[0],
                optimal_center[1],
                step_size))

    foo, winds = integrate_circle_distances(ultimate_S1_coords, np.pi)
    return pointwise_add_alpha_beta_with_mod(image_alpha[0], image_beta[0], optimal_center[0], optimal_center[1],
                                             step_size), pointwise_add_alpha_beta_with_mod(
        image_alpha[number_of_pieces - 1], image_beta[number_of_pieces - 1], optimal_center[0], optimal_center[1],
        step_size), winds


def get_values(result, idx, n_pts):
    diagrams = result['dgms']
    cocycles = result['cocycles']
    D = result['dperm2all']
    # PD diagram
    dgm1 = diagrams[1]

    # Compute Cycles
    print('Doing Niko Stuff')
    cocycle = cocycles[1][idx]
    # I'm going to change this 6/1 to a lower threshold so we have fewer edges
    # Original: thresh = dgm1[idx, 1] - 0.0001 # Project cocycle onto edges STRICLY LESS THAN death time
    thresh = (1 - .4) * dgm1[idx, 0] + .4 * dgm1[idx, 1]

    # Fix cycles
    fixed_cocycle = []
    for x in range(len(cocycle)):
        i = cocycle[x][0]
        j = cocycle[x][1]
        if D[i, j] <= thresh:
            fixed_cocycle.append([i, j, cocycle[x][2]])
    # print(fixed_cocycle)
    fixed_cocycle_values = []
    for (x, y, z) in fixed_cocycle:
        fixed_cocycle_values.append(z)

    edges = []
    for i in range(n_pts):
        for j in range(i):
            if D[i, j] <= thresh:
                edges.append([i, j])

    # print("Edges with nontrivial cohomology value:")
    nontrivial_edges = []
    for (x, y, z) in fixed_cocycle:
        nontrivial_edges.append([x, y])
    # print(nontrivial_edges)

    # print('Trivial Edges')
    trivial_edges = edges.copy()
    for x in edges:
        for k in range(len(nontrivial_edges)):
            if x == nontrivial_edges[k]:
                trivial_edges.remove(x)
    # print(trivial_edges)

    if len(trivial_edges) + len(nontrivial_edges) == len(edges):
        print("Edges accounted for, okay to proceed")
    else:
        print("Something is wrong")

    for edge in trivial_edges:
        edge_with_value = edge.copy()
        edge_with_value.append(0)

    all_edges = np.concatenate((nontrivial_edges, trivial_edges,), axis=0)
    all_values = np.concatenate((fixed_cocycle_values, np.zeros(len(trivial_edges))), axis=0)

    return all_edges, all_values, D, cocycle, thresh


def DataLoader(type, n_pts=75, plot=0):
    if type == 0:  # triangle
        x = np.array([[0, 0],
                      [2, 0],
                      [1, 1]])
        dim = 2

    if type == 1:
        x = np.array([[-0.99993339, -0.01154223],
                      [-0.34820481, -0.93741848],
                      [0.86358501, -0.50420326],
                      [-0.10509239, 0.89446246],
                      [0.92380802, 0.38285603]])
        dim = 2

    if type == 2:
        number_of_points = 12
        np.random.seed(9)
        x = tadasets.dsphere(n=number_of_points, d=1, noise=0.1)
        dim = 2

    if type == 3:
        number_of_points = 120
        np.random.seed(9)
        x = tadasets.dsphere(n=number_of_points, d=1, noise=0.2)
        dim = 2

    if type == 4:
        n = n_pts
        x = np.zeros([n, 2])
        t = np.linspace(0, 2 * np.pi - ((2 * np.pi) / n), n)
        x[:, 0] = np.sin(t)
        x[:, 1] = np.cos(t)
        dim = 2

    if type == 5:
        n = n_pts
        x = np.zeros([n, 3])
        t = np.linspace(0, 2 * np.pi - ((2 * np.pi) / n), n)
        x[:, 0] = np.sin(t) + 2 * np.sin(2 * t)
        x[:, 1] = np.cos(t) - 2 * np.cos(2 * t)
        x[:, 2] = np.sin(3 * t)
        dim = 3
        x = 10 * x

    if type == 6:
        n = n_pts
        dim = 5
        x = np.zeros([n, 5])
        t = np.linspace(0, 2 * np.pi - ((2 * np.pi) / n), n)
        x[:, 0] = 5 * np.sin(t)
        x[:, 1] = np.cos(t)
        P = scipy.stats.ortho_group.rvs(dim)
        x = x.dot(P) + .01 * np.random.randn(n, dim)
        # x = x + 0.01 * np.random.randn(n, dim)

    if type == 7:
        size = 128
        x = np.zeros([72, size ** 2])
        # base_path = 'C:/Users/sscho/Dropbox/Research_UCD/hypersphericalcoordinates'
        base_path = 'C:/Users/Stefan/Dropbox/Research_UCD/sphericalcoordinates'
        for i in range(72):
            fname = base_path + '/Coil/obj1__' + str(i) + '.png'
            im_frame = Image.open(fname)
            img = np.reshape(np.array(im_frame.getdata()), [128, 128])
            scale = size / 128
            img = rescale(img, scale, anti_aliasing=False)
            img = np.reshape(img, [size ** 2])
            x[i, :] = img
        dim = size ** 2
        c = np.arange(72) / 72

    if type == 8:
        dim = 2
        z1 = np.linspace(.15, 2 * np.pi - .15, 25)
        z2 = np.linspace(0, 2 * np.pi, 10)
        pts1 = np.array([10 * np.cos(z1), 10 * np.sin(z1)])
        pts2 = np.array([np.cos(z2) + 10, np.sin(z2)])
        x = np.vstack([pts1.T, pts2.T])
        c = np.hstack([z1, np.arctan(pts2[1, :] / pts2[0, :])])

    if dim == 2:
        c = np.arctan2(x[:, 1], x[:, 0])
    elif dim in [3, 4, 5]:
        c = t

    if plot == 1:
        if dim == 2:
            plt.scatter(x[:, 0], x[:, 1], s=50, c=c, cmap='hsv')
            plt.axis('equal')
            plt.title('Data')
            plt.show()
        elif dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=100, c=t, cmap='hsv')
            plt.title('Data')
            plt.show()

    return x, c


def load_kitten_pics():
    base_path = 'C:/Users/Stefan/Dropbox/Research_UCD/sphericalcoordinates'
    count = 0
    size = 128
    n_load = 36
    pts = np.zeros([36 * 36, size ** 2])
    angles = np.zeros([36 * 36, 2])
    for i in range(n_load):
        for j in range(n_load):
            fname = base_path + '/KittenPics/k' + str(i) + '_' + str(j) + '.png'
            im_frame = Image.open(fname)
            img = np.reshape(np.array(im_frame.getdata()), [354, 354, 4])
            img = rescale(0.2989 * img[30:314, 30:314, 0] + 0.5870 * img[30:314, 30:314, 1]
                          + 0.1140 * img[30:314, 30:314, 2], size / 284, anti_aliasing=False)
            img = np.reshape(img, [size ** 2])
            pts[count, :] = img
            angles[count, :] = [i, j]
            count = count + 1
    return pts, angles


def load_bunny_pics(type=0, scale=0.2, n_load=100):
    if type == 0:
        base_path = 'C:/Users/Stefan/Dropbox/Research_UCD/sphericalcoordinates/Bunny'
        key = '/bunny'
        n_total = 2048
    elif type == 1:
        base_path = 'C:/Users/Stefan/Dropbox/Research_UCD/sphericalcoordinates/Bunny2'
        key = '/bunny'
        n_total = 4096
    elif type == 2:
        base_path = 'C:/Users/Stefan/Dropbox/Research_UCD/sphericalcoordinates/Chair'
        key = '/chair'
        n_total = 4096
    elif type == 3:
        base_path = 'C:/Users/Stefan/Dropbox/Research_UCD/sphericalcoordinates/Letter'
        key = '/letterdata'
        n_total = 4096

    pts = []
    idx = np.random.choice(n_total, n_load, replace=False)
    for i in range(n_load):
        fname = base_path + key + str(idx[i]) + '.png'
        img_raw = imread(fname)
        img_small = rescale(img_raw, scale)
        img_flat = np.squeeze(np.reshape(img_small, [img_small.shape[0] * img_small.shape[1]]))
        pts.append(img_flat)
    size = [img_small.shape[0], img_small.shape[1]]
    return np.array(pts), size


def PlotCircleandPts(coords, title='Coordinates', c=0, newfig=1):
    # draw circle
    if newfig == 1:
        plt.figure()
    theta = np.linspace(0, 2 * np.pi, 150)
    a = np.cos(theta)
    b = np.sin(theta)
    plt.plot(a, b)

    # plot pts
    x = np.cos(coords)
    y = np.sin(coords)
    plt.scatter(x, y, 200, marker='*', c=c, cmap='hsv')
    plt.axis('equal')
    plt.axis('off')
    plt.title(title)


def ComputeDistance(pt1, pt2, cycle, orient):
    if orient == 1:
        if pt2 >= pt1:
            D = pt2 - pt1 + 2 * np.pi * cycle
        else:
            D = 2 * np.pi - (pt1 - pt2) + 2 * np.pi * cycle
    elif orient == -1:
        if pt1 >= pt2:
            D = pt1 - pt2 + 2 * np.pi * cycle
        else:
            D = 2 * np.pi - (pt2 - pt1) + 2 * np.pi * cycle
    else:
        print('Orientation broken')
        D = 0
    return D


def ComputeHarmAndGrad(coords, edges, cycles, orient):
    nrg = 0
    grad = np.zeros(len(coords))
    S = np.zeros(len(edges))
    for i in range(len(edges)):
        D = ComputeDistance(coords[edges[i][0]], coords[edges[i][1]], cycles[i], orient[i])
        S[i] = D
        nrg += .5 * (D ** 2)
        grad[edges[i][0]] += -D * orient[i]
        grad[edges[i][1]] += D * orient[i]
    return nrg, grad, S


def ComputeSpringAndGrad(coords, edges, cycles, orient, rest_length, k=1):
    nrg = 0
    grad = np.zeros(len(coords))
    S = np.zeros(len(edges))
    for i in range(len(edges)):
        D = ComputeDistance(coords[edges[i][0]], coords[edges[i][1]], cycles[i], orient[i])
        Displacement = D - rest_length
        S[i] = Displacement
        nrg += .5 * (k * Displacement ** 2)
        grad[edges[i][0]] += -k * Displacement * orient[i]
        grad[edges[i][1]] += k * Displacement * orient[i]
    return nrg, grad, S


def CoordUpdate(coords, edges, cycles, orient, distances, grad, step_size):
    # get new coords
    coords_new = coords - step_size * grad
    edges = np.array(edges)
    # fix orientations and cycles
    for i in range(len(edges)):
        old_dist = (coords[edges[i][1]] - coords[edges[i][0]])
        new_dist = (coords_new[edges[i][1]] - coords_new[edges[i][0]])
        old_sign = np.sign(old_dist)
        new_sign = np.sign(new_dist)
        if old_sign != new_sign:
            if cycles[i] > 0:
                cycles[i] = cycles[i] - 1
            else:
                orient[i] = new_sign
    # update coordinates
    for i in range(len(coords_new)):
        if coords_new[i] < 0:
            coords_new[i] = coords_new[i] + 2 * np.pi
        if coords_new[i] > 2 * np.pi:
            coords_new[i] = coords_new[i] - 2 * np.pi
    # Check for orientation errors---FIX THIS
    for i in range(len(edges)):
        D_new = ComputeDistance(coords_new[edges[i][0]], coords_new[edges[i][1]], cycles[i], orient[i])
        if D_new - distances[i] > np.pi:
            orient[i] = -1 * orient[i]
    return coords_new, cycles, orient


# def furthest_point(lines, x_lim, y_lim, n):
#     pts = np.array(np.meshgrid(np.linspace(0, x_lim, int(n * x_lim)),
#                                np.linspace(0, y_lim, int(n * y_lim))))
#     pts = np.transpose(np.reshape(pts, [2, int(n * x_lim) * int(n * y_lim)]))
#     nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(lines)
#     distances, indices = nbrs.kneighbors(pts)
#     max_ind = np.argmax(distances)
#     return pts[max_ind, :]


def furthest_point(lines, x_min, x_max, y_min, y_max, n):
    pts = np.array(np.meshgrid(np.linspace(x_min, x_max, int(n * (x_max - x_min))),
                               np.linspace(y_min, y_max, int(n * (y_max - y_min)))))
    pts = np.transpose(np.reshape(pts, [2, int(n * (x_max - x_min)) * int(n * (y_max - y_min))]))
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(lines)
    distances, indices = nbrs.kneighbors(pts)
    max_ind = np.argmax(distances)
    return pts[max_ind, :]


def draw_arc(a, b, orient, r=1):
    t = 0
    if orient == 1:
        if b > a:
            t = np.linspace(a, b, 100)
        else:
            t = np.linspace(a, b + 2 * np.pi, 100)
    elif orient == -1:
        if b > a:
            t = np.linspace(b, a + 2 * np.pi, 100)
        else:
            t = np.linspace(b, a, 100)
    xs = r * np.cos(t)
    ys = r * np.sin(t)
    return plt.plot(xs, ys)


def compute_arc(a, b, orient):
    t = 0
    if orient == 1:
        if b > a:
            t = np.linspace(a, b, 100)
        else:
            t = np.linspace(a, b + 2 * np.pi, 100)
    elif orient == -1:
        if b > a:
            t = np.linspace(b, a + 2 * np.pi, 100)
        else:
            t = np.linspace(b, a, 100)
    return t


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_closest(array, value):
    array = np.asarray(array)
    distances = np.linalg.norm(array - value, axis=1)
    idx = distances.argmin()
    return idx

