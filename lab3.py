import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation
from matplotlib.transforms import Affine2D

def Rot2D(X, Y, phi):
    X_r = X*np.cos(phi) - Y*np.sin(phi)
    Y_r = X*np.sin(phi) + Y*np.cos(phi)
    return X_r, Y_r

def EqOfMovement(y, t, l1, l2, c, m, g):
    """
    y = [phi, psi, dphi, dpsi].
    Возвращает [dphi, dpsi, ddphi, ddpsi].
    """
    phi = y[0]
    psi = y[1]
    dphi = y[2]
    dpsi = y[3]

    a11 = 2*l1
    a12 = l2*np.cos(psi - phi)
    a21 = l1*np.cos(psi - phi)
    a22 = l2

    b1 = -(2*g + (c*l1/m)*np.cos(phi))*np.sin(phi) + l2*(dpsi**2)*np.sin(psi - phi)
    b2 = -g*np.sin(psi) - l1*(dphi**2)*np.sin(psi - phi)

    det = a11*a22 - a12*a21
    ddphi = (a22*b1 - a12*b2)/det
    ddpsi = (-a21*b1 + a11*b2)/det

    return np.array([dphi, dpsi, ddphi, ddpsi], dtype=float)

def acceleration_array(Y, t, l1, l2, c, m, g):
    """
    Подсчёт (ddphi, ddpsi) на каждом шаге (t[i], Y[i]),
    вызывая EqOfMovement для каждого вектора состояния.
    """
    Npoints = len(t)
    out = np.zeros_like(Y)
    for i in range(Npoints):
        out[i] = EqOfMovement(Y[i], t[i], l1, l2, c, m, g)
    return out

def main():
    # Параметры и начальные условия
    m = 0.1
    l1 = 0.5
    l2 = 0.5
    c = 5
    g = 9.81
    phi0 = 0
    psi0 = np.pi
    d_phi0 = 0
    d_psi0 = 0

    # Визуальные параметры
    l0 = 2
    w_box = 0.5
    h_box = 0.2
    xbox = np.array([-w_box/2, w_box/2, w_box/2, -w_box/2, -w_box/2])
    ybox = np.array([ h_box/2, h_box/2, -h_box/2, -h_box/2,  h_box/2])

    # Сетка времени
    t_fin = 20
    Nt = 2001
    t = np.linspace(0, t_fin, Nt)

    # Интегрируем систему
    y0 = [phi0, psi0, d_phi0, d_psi0]
    Y = odeint(EqOfMovement, y0, t, args=(l1, l2, c, m, g))

    phi = Y[:,0]
    psi = Y[:,1]
    d_phi = Y[:,2]
    d_psi = Y[:,3]

    # Получаем ускорения
    Ydot = acceleration_array(Y, t, l1, l2, c, m, g)
    dd_phi = Ydot[:,2]
    dd_psi = Ydot[:,3]

    # Формула для N:
    N = m*(
        g*np.cos(psi)
        - l1*(dd_phi*np.sin(psi - phi) - d_phi**2*np.cos(psi - phi))
        + l2*(d_psi**2)
    )

    # Построим графики φ(t), ψ(t), N(t)
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(2,3,1)
    ax1.plot(t, phi, 'r')
    ax1.set_title("phi(t)")

    ax2 = fig.add_subplot(2,3,2)
    ax2.plot(t, psi, 'g')
    ax2.set_title("psi(t)")

    ax3 = fig.add_subplot(2,3,3)
    ax3.plot(t, N, 'b')
    ax3.set_title("N(t)")

    # Для анимации
    xM1 = l1*np.cos(phi)
    yM1 = l1*np.sin(phi)
    xM2 = xM1 + l2*np.cos(psi)
    yM2 = yM1 + l2*np.sin(psi)

    ax_anim = fig.add_subplot(2,1,2)
    ax_anim.axis("equal")
    ax_anim.set_xlim([-1.5, 1.5])
    ax_anim.set_ylim([-1.5, 1.5])

    # Поворот системы координат на -90°
    rot = Affine2D().rotate_deg(-90)
    ax_anim.transData = rot + ax_anim.transData

    # Стенки, оси, крепление
    ax_anim.plot([-5, 5], [-l0, -l0], color='black')
    ax_anim.plot([-5, 5], [-l0 - h_box, -l0 - h_box], color='black')
    ax_anim.plot([0, 0], [-4.8, 4.8], color=(0.8, 0.8, 0.8), linestyle='-.')
    ax_anim.plot([-4.8, 4.8], [0, 0], color=(0.8, 0.8, 0.8), linestyle='-.')

    ax_anim.plot([-0.3, -0.3], [-5, 5], color='black')
    ax_anim.plot([-0.3, 0, -0.3, -0.3], [-0.2, 0, 0.2, -0.2], color='black')

    # Коробка (серый), пружина (коричневый), стержни (коричневый/чёрный), массы (чёрные)
    box = ax_anim.plot(xbox + xM1[0], ybox - l0 - h_box/2, color='gray')[0]
    lM1 = ax_anim.plot([0, xM1[0]], [0, yM1[0]], color='brown')[0]
    lM2 = ax_anim.plot([xM1[0], xM2[0]], [yM1[0], yM2[0]], color='black')[0]

    n = 24
    hh = 0.08
    xP_help = np.linspace(0, 1, 2*n + 1)
    yP_help = hh*np.sin((np.pi/2)*np.arange(2*n + 1))
    xP, yP = Rot2D(xP_help, yP_help, np.pi/2)
    spring = ax_anim.plot(xP + xM1[0], yP*(l0 + yM1[0]) - l0, color='brown')[0]

    ax_anim.plot(0, 0, marker='o', markerfacecolor='white')
    M1 = ax_anim.plot(xM1[0], yM1[0], marker='o', color='black', ms=8)[0]
    M2 = ax_anim.plot(xM2[0], yM2[0], marker='o', color='black', ms=8)[0]

    def animate(i):
        lM1.set_data([0, xM1[i]], [0, yM1[i]])
        lM2.set_data([xM1[i], xM2[i]], [yM1[i], yM2[i]])
        box.set_data(xbox + xM1[i], ybox - l0 - h_box/2)
        spring.set_data(xP + xM1[i], yP*(l0 + yM1[i]) - l0)
        M1.set_data([xM1[i]], [yM1[i]])
        M2.set_data([xM2[i]], [yM2[i]])
        return lM1, lM2, box, spring, M1, M2

    anim = FuncAnimation(fig, animate, frames=Nt, interval=1)
    plt.show()

if __name__ == "__main__":
    main()
