# import plotly.graph_objects as go
# import numpy as np
# from scipy.stats import multivariate_normal

# # Generate grid
# x = np.linspace(-3, 3, 100)
# y = np.linspace(-3, 3, 100)
# X, Y = np.meshgrid(x, y)
# pos = np.dstack((X, Y))

# # Multivariate normal distribution
# rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])

# # Calculate Z values (PDF)
# Z = rv.pdf(pos)

# # Create the figure
# fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])

# # Add title and format the axes
# fig.update_layout(
#     title='3D Gaussian Distribution',
#     scene=dict(
#         xaxis_title='X Axis',
#         yaxis_title='Y Axis',
#         zaxis_title='Probability Density'
#     )
# )

# # Show the plot
# fig.show()
# fig.write_image("3d_gaussian_surface.png")





###############
# Authored by Weisheng Jiang
# Book 6  |  From Basic Arithmetic to Machine Learning
# Published and copyrighted by Tsinghua University Press
# Beijing, China, 2022
###############

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from matplotlib import cm # Colormaps
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from scipy.stats import multivariate_normal
# from scipy.stats import norm
# from mpl_toolkits.mplot3d import axes3d

# def fcn_Y_given_X (mu_X, mu_Y, sigma_X, sigma_Y, rho, X, Y):
    
#     coeff = 1/sigma_Y/np.sqrt(1 - rho**2)/np.sqrt(2*np.pi)
#     sym_axis = mu_Y + rho*sigma_Y/sigma_X*(X - mu_X)
    
#     quad  = -1/2*((Y - sym_axis)/sigma_Y/np.sqrt(1 - rho**2))**2
    
#     f_Y_given_X  = coeff*np.exp(quad)
    
#     return f_Y_given_X

# # parameters

# rho     = 0.5
# sigma_X = 1
# sigma_Y = 1

# mu_X = 0
# mu_Y = 0

# width = 3
# X = np.linspace(-width,width,31)
# Y = np.linspace(-width,width,31)

# XX, YY = np.meshgrid(X, Y)

# f_Y_given_X = fcn_Y_given_X (mu_X, mu_Y, sigma_X, sigma_Y, rho, XX, YY)

# fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

# ax.plot_wireframe(XX, YY, f_Y_given_X,
#                   color = [0.3,0.3,0.3],
#                   linewidth = 0.25)

# ax.set_xlabel('$x$')
# ax.set_ylabel('$y$')
# ax.set_zlabel('$f_{Y|X}(y|x)$')
# ax.set_proj_type('ortho')
# ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
# ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
# ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

# ax.set_xlim(-width, width)
# ax.set_ylim(-width, width)
# ax.set_zlim(f_Y_given_X.min(),f_Y_given_X.max())
# plt.tight_layout()
# ax.view_init(azim=-120, elev=30)
# plt.show()

# #%% surface projected along X to Y-Z plane

# fig = plt.figure()
# ax = fig.add_subplot(projection = '3d')

# ax.plot_wireframe(XX, YY, f_Y_given_X, rstride=0, cstride=1,
#                   color = [0.3,0.3,0.3],
#                   linewidth = 0.25)

# ax.contour(XX, YY, f_Y_given_X, 
#            levels = 20, zdir='x', \
#             offset=YY.max(), cmap=cm.RdYlBu_r)

# ax.set_xlabel('$x$')
# ax.set_ylabel('$y$')
# ax.set_zlabel('$f_{Y|X}(y|x)$')
# ax.set_proj_type('ortho')
# ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
# ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
# ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

# ax.set_xlim(-width, width)
# ax.set_ylim(-width, width)
# ax.set_zlim(f_Y_given_X.min(),f_Y_given_X.max())
# plt.tight_layout()
# ax.view_init(azim=-120, elev=30)
# plt.show()


# # add X marginal

# f_Y = norm.pdf(Y, loc=mu_Y, scale=sigma_Y)

# fig, ax = plt.subplots()

# colors = plt.cm.RdYlBu_r(np.linspace(0,1,len(X)))

# for i in np.linspace(1,len(X),len(X)):
#     plt.plot(Y,f_Y_given_X[:,int(i)-1],
#              color = colors[int(i)-1])

# plt.plot(Y,f_Y, color = 'k')

# plt.xlabel('y')
# plt.ylabel('$f_{Y|X}(y|x)$')
# ax.set_xlim(-width, width)
# ax.set_ylim(0, f_Y_given_X.max())

# #%% surface projected along Z to X-Y plane

# fig = plt.figure()
# ax = fig.add_subplot(projection = '3d')

# ax.plot_wireframe(XX, YY, f_Y_given_X,
#                   color = [0.3,0.3,0.3],
#                   linewidth = 0.25)

# ax.contour3D(XX,YY,f_Y_given_X,12,
#               cmap = 'RdYlBu_r')

# ax.set_xlabel('$x$')
# ax.set_ylabel('$y$')
# ax.set_zlabel('$f_{Y|X}(y|x)$')
# ax.set_proj_type('ortho')
# ax.xaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
# ax.yaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})
# ax.zaxis._axinfo["grid"].update({"linewidth":0.25, "linestyle" : ":"})

# ax.set_xlim(-width, width)
# ax.set_ylim(-width, width)
# ax.set_zlim(f_Y_given_X.min(),f_Y_given_X.max())
# plt.tight_layout()
# ax.view_init(azim=-120, elev=30)
# plt.show()

# # Plot filled contours

# E_Y_given_X = mu_Y + rho*sigma_Y/sigma_X*(X - mu_X)

# from matplotlib.patches import Rectangle

# fig, ax = plt.subplots(figsize=(7, 7))

# # Plot bivariate normal
# plt.contourf(XX, YY, f_Y_given_X, 12, cmap=cm.RdYlBu_r)
# plt.plot(X,E_Y_given_X, color = 'k', linewidth = 1.25)
# plt.axvline(x = mu_X, color = 'k', linestyle = '--')
# plt.axhline(y = mu_Y, color = 'k', linestyle = '--')

# x = np.linspace(-width,width,num = 201)
# y = np.linspace(-width,width,num = 201)

# xx,yy = np.meshgrid(x,y);

# ellipse = ((xx/sigma_X)**2 - 
#            2*rho*(xx/sigma_X)*(yy/sigma_Y) + 
#            (yy/sigma_Y)**2)/(1 - rho**2);

# plt.contour(xx,yy,ellipse,levels = [1], colors = 'k')

# rect = Rectangle(xy = [- sigma_X, - sigma_Y] , 
#                  width = 2*sigma_X, 
#                  height = 2*sigma_Y,
#                  edgecolor = 'k',facecolor="none")

# ax.add_patch(rect)

# ax.set_xlabel('$x$')
# ax.set_ylabel('$y$')


# plt.savefig("gaussian.jpg")





# ###############
# # Authored by Weisheng Jiang
# # Book 6  |  From Basic Arithmetic to Machine Learning
# # Published and copyrighted by Tsinghua University Press
# # Beijing, China, 2022
# ###############

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from matplotlib import cm # Colormaps
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from scipy.stats import multivariate_normal
# from scipy.stats import norm

# rho     = 0.5
# sigma_X = 1.5
# sigma_Y = 1

# mu_X = 0
# mu_Y = 0
# mu    = [mu_X, mu_Y]

# Sigma = [[sigma_X**2, sigma_X*sigma_Y*rho], 
#         [sigma_X*sigma_Y*rho, sigma_Y**2]]

# width = 4
# X = np.linspace(-width,width,81)
# Y = np.linspace(-width,width,81)

# XX, YY = np.meshgrid(X, Y)

# XXYY = np.dstack((XX, YY))
# bi_norm = multivariate_normal(mu, Sigma)

# #%% visualize PDF

# y_cond_i = 60 # 20, 30, 40, 50, 60, index


# f_X_Y_joint = bi_norm.pdf(XXYY)

# # Plot the tional distributions
# fig = plt.figure(figsize=(7, 7))
# gs = gridspec.GridSpec(2, 2, 
#                        width_ratios=[3, 1], 
#                        height_ratios=[3, 1])

# # Plot surface on top left
# ax1 = plt.subplot(gs[0])

# # Plot bivariate normal
# ax1.contour(XX, YY, f_X_Y_joint, 20, cmap=cm.RdYlBu_r)
# ax1.axvline(x = mu_X, color = 'k', linestyle = '--')
# ax1.axhline(y = mu_Y, color = 'k', linestyle = '--')
# ax1.axhline(y = Y[y_cond_i], color = 'r', linestyle = '--')

# x_sym_axis = mu_X + rho*sigma_X/sigma_Y*(Y[y_cond_i] - mu_Y)
# ax1.axvline(x = x_sym_axis, color = 'r', linestyle = '--')

# ax1.set_xlabel('$X$')
# ax1.set_ylabel('$Y$')
# ax1.yaxis.set_label_position('right')
# ax1.set_xticks([])
# ax1.set_yticks([])

# # Plot Y marginal
# ax2 = plt.subplot(gs[1])
# f_Y = norm.pdf(Y, loc=mu_Y, scale=sigma_Y)

# ax2.plot(f_Y, Y, 'k', label='$f_{Y}(y)$')
# ax2.axhline(y = mu_Y, color = 'k', linestyle = '--')
# ax2.axhline(y = Y[y_cond_i], color = 'r', linestyle = '--')
# ax2.plot(f_Y[y_cond_i], Y[y_cond_i], marker = 'x', markersize = 15)
# plt.title('$f_{Y}(y_{} = %.2f) = %.2f$'
#           %(Y[y_cond_i],f_Y[y_cond_i]))

# ax2.fill_between(f_Y,Y, 
#                  edgecolor = 'none', 
#                  facecolor = '#D9D9D9')
# ax2.legend(loc=0)
# ax2.set_xlabel('PDF')
# ax2.set_ylim(-width, width)
# ax2.set_xlim(0, 0.5)
# ax2.invert_xaxis()
# ax2.yaxis.tick_right()

# # Plot X and Y joint

# ax3 = plt.subplot(gs[2])
# f_X_Y_cond_i = f_X_Y_joint[y_cond_i,:]

# ax3.plot(X, f_X_Y_cond_i, 'r', 
#          label='$f_{X,Y}(x,y_{} = %.2f)$' %(Y[y_cond_i]))


# ax3.axvline(x = mu_X, color = 'k', linestyle = '--')
# ax3.axvline(x = x_sym_axis, color = 'r', linestyle = '--')

# ax3.legend(loc=0)
# ax3.set_ylabel('PDF')
# ax3.yaxis.set_label_position('left')
# ax3.set_xlim(-width, width)
# ax3.set_ylim(0, 0.5)
# ax3.set_yticks([0, 0.25, 0.5])

# ax4 = plt.subplot(gs[3])
# ax4.set_visible(False)

# plt.show()

# #%% compare joint, marginal and tional

# f_X = norm.pdf(X, loc=mu_X, scale=sigma_X)

# fig, ax = plt.subplots()

# colors = plt.cm.RdYlBu_r(np.linspace(0,1,len(Y)))

# f_X_given_Y_cond_i = f_X_Y_cond_i/f_Y[y_cond_i]

# plt.plot(X,f_X, color = 'k',
#          label='$f_{X}(x)$') # marginal
# ax.axvline(x = mu_X, color = 'k', linestyle = '--')


# plt.plot(X,f_X_Y_cond_i, color = 'r',
#          label='$f_{X,Y}(x,y_{} = %.2f$)' %(Y[y_cond_i])) # joint
# ax.axvline(x = x_sym_axis, color = 'r', linestyle = '--')

# plt.plot(X,f_X_given_Y_cond_i, color = 'b',
#          label='$f_{X|Y}(x|y_{} = %.2f$)' %(Y[y_cond_i])) # tional

# ax.fill_between(X,f_X_given_Y_cond_i, 
#                 edgecolor = 'none', 
#                 facecolor = '#DBEEF3')

# ax.fill_between(X,f_X_Y_cond_i, 
#                 edgecolor = 'none',
#                 hatch='/')


# plt.xlabel('X')
# plt.ylabel('PDF')
# ax.set_xlim(-width, width)
# ax.set_ylim(0, 0.35)
# plt.title('$f_{Y}(y_{} = %.2f) = %.2f$'
#           %(Y[y_cond_i],f_Y[y_cond_i]))
# ax.legend()

# plt.savefig('gaussian_con.jpg')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from matplotlib.tri import Triangulation


def plot_guassian_2d(mvn, ax):
    # Generate a grid of points where the PDF will be evaluated
    x = np.linspace(-4, 4, 20)
    y = np.linspace(-0.4, 0.4, 20)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    # Evaluate the PDF of the multivariate normal distribution over the grid
    Z = mvn.pdf(pos)

    # Flatten the grid data for triangulation
    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()

    # Create a triangulation from the flattened grid data
    triangulation = Triangulation(x_flat, y_flat)

    ax.view_init(elev=30, azim=50)
    # Plot the surface with a transparent triangulation
    # edgecolor='none' makes the grid lines transparent
    ax.plot_trisurf(triangulation, z_flat, cmap='viridis', linewidth=0, edgecolor='none')

    ax.grid(True)
    # Label the axes and set the title
    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    ax.set_zlabel('Probability Density')
    # ax.set_title('3D Gaussian Distribution')
    # Make the panes transparent
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Make the grid lines transparent
    # ax.xaxis._axinfo["grid"]['color'] = (1,1,1,0)
    # ax.yaxis._axinfo["grid"]['color'] = (1,1,1,0)
    # ax.zaxis._axinfo["grid"]['color'] = (1,1,1,0)

    # ax.set_xticks(np.linspace(np.min(x_flat), np.max(x_flat), 6))
    # ax.set_yticks(np.linspace(np.min(y_flat), np.max(y_flat), 6))
    # ax.set_zticks(np.linspace(np.min(z_flat), np.max(z_flat), 17))
    ax.set_xticks(np.linspace(-4, 4, 5))
    ax.set_yticks(np.linspace(-0.4, 0.4, 5))
    # if np.max(y_flat) <= 0.1:
    #     ax.set_yticks(np.linspace(-0.1, 0.1, 6))
    # else:
    #     ax.set_yticks(np.linspace(np.min(y_flat), np.max(y_flat), 6))
    ax.set_zticks(np.linspace(0, 8.0, 5))
    ax.set_zlim([0, 8.0])

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # Show the plot
    plt.show()

if __name__ == '__main__':
    # # Generate example data
    # data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=500)

    # # Calculate mean and covariance of the data
    # mean = np.mean(data, axis=0)
    # covariance = np.cov(data.T)

    # # Create a multivariate normal distribution object
    # mvn = multivariate_normal(mean, covariance)
    # # Create a figure with a 3D axis
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')
    # x = np.linspace(-3, 3, 20)
    # y = np.linspace(-3, 3, 20)
    # plot_guassian_2d(mvn, x, y, ax)
    # plt.savefig('gaussian_new1.jpg', dpi=300)

    print(max([5.12, 10.03]))