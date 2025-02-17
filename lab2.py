import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.transforms import Affine2D

def Rot2D(X, Y, phi):
    X_r = X*np.cos(phi) - Y*np.sin(phi)
    Y_r = X*np.sin(phi) + Y*np.cos(phi)
    return X_r, Y_r

def main():
    time_steps_amount = 1000
    t = np.linspace(0, 4*np.pi, time_steps_amount)

    l0 = 4.35
    w_box = 0.5
    h_box = 0.2
    l1 = 2
    l2 = 2.2


    phi = 1*np.sin(1.7*t) + 2*np.cos(t)
    psi = 2*np.sin(1.7*t) + 3*np.cos(3*t)

    xM1 = l1*np.cos(phi)
    yM1 = l1*np.sin(phi)
    xM2 = xM1 + l2*np.cos(psi)
    yM2 = yM1 + l2*np.sin(psi)

    xbox = np.array([-w_box/2, w_box/2, w_box/2, -w_box/2, -w_box/2])
    ybox = np.array([ h_box/2, h_box/2, -h_box/2, -h_box/2,  h_box/2])

    figure = plt.figure(figsize=[10, 10])
    ax = figure.add_subplot(111)
    ax.axis("equal")
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])

    # Поворачиваем систему координат на -90°
    rot = Affine2D().rotate_deg(-90)
    ax.transData = rot + ax.transData

    # Стенки, оси, крепление
    ax.plot([-5, 5], [-l0, -l0], color='black')
    ax.plot([-5, 5], [-l0 - h_box, -l0 - h_box], color='black')
    ax.plot([0, 0], [-4.8, 4.8], color=(0.9, 0.9, 0.9), linestyle='-.')
    ax.plot([-4.8, 4.8], [0, 0], color=(0.9, 0.9, 0.9), linestyle='-.')

    ax.plot([-0.3, -0.3], [-5, 5], color='black')
    ax.plot([-0.3, 0, -0.3, -0.3], [-0.2, 0, 0.2, -0.2], color='black')


    box = ax.plot(xbox + xM1[0], ybox - l0 - h_box/2, color='gray')[0]
    lM1 = ax.plot([0, xM1[0]], [0, yM1[0]], color='brown')[0]
    lM2 = ax.plot([xM1[0], xM2[0]], [yM1[0], yM2[0]], color='black')[0]

    n = 24
    h = 0.08
    xP_help = np.linspace(0, 1, 2*n + 1)
    yP_help = h*np.sin((np.pi/2)*np.arange(2*n + 1))
    xP, yP = Rot2D(xP_help, yP_help, np.pi/2)
    spring = ax.plot(xP + xM1[0], yP*(l0 + yM1[0]) - l0, color='brown')[0]

    ax.plot(0, 0, marker='o', markerfacecolor='white')  # точка фиксации
    M1 = ax.plot(xM1[0], yM1[0], marker='o', color='black', ms=8)[0]
    M2 = ax.plot(xM2[0], yM2[0], marker='o', color='black', ms=8)[0]

    def animate(i):
        lM1.set_data([0, xM1[i]], [0, yM1[i]])
        lM2.set_data([xM1[i], xM2[i]], [yM1[i], yM2[i]])
        box.set_data(xbox + xM1[i], ybox - l0 - h_box/2)
        spring.set_data(xP + xM1[i], yP*(l0 + yM1[i]) - l0)
        M1.set_data(xM1[i], yM1[i])
        M2.set_data(xM2[i], yM2[i])
        return lM1, lM2, box, spring, M1, M2

    animation = FuncAnimation(figure, animate, frames=time_steps_amount, interval=100)
    plt.show()

if __name__ == '__main__':
    main()
