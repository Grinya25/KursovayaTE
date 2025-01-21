#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import matplotlib.pyplot as plt

class MaterialPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.initial_x = x 
        self.initial_y = y
        self.trajectory = [(x, y)]

    def update_position(self, x, y):
        self.x = x
        self.y = y
        self.trajectory.append((x, y))

class Rectangle:
    def __init__(self, center_x, center_y, width, height, num_points_x, num_points_y):
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.num_points_x = num_points_x
        self.num_points_y = num_points_y
        self.points = self.generate_points()

    def generate_points(self):
        points = []
        x_coords = np.linspace(self.center_x - self.width / 2, self.center_x + self.width / 2, self.num_points_x)
        y_coords = np.linspace(self.center_y - self.height / 2, self.center_y + self.height / 2, self.num_points_y)
        for x in x_coords:
             for y in y_coords:
               points.append(MaterialPoint(x, y))
        return points

class RungeKuttaIntegrator:
    def __init__(self, A, B):
        self.A = A
        self.B = B 

    def compute_velocity(self, t, x, y):
        v1 = -self.A(t) * x
        v2 = self.B(t) * y
        return v1, v2

    def integrate(self, point, t, dt):
        x, y = point.x, point.y

        k1x, k1y = self.compute_velocity(t, x, y)
        k2x, k2y = self.compute_velocity(t + 2/3 * dt, x + 2/3 * dt * k1x, y + 2/3 * dt * k1y)
        k3x, k3y = self.compute_velocity(t + 2/3 * dt, x + dt * (-1/3 * k1x + 1 * k2x), y + dt * (-1/3 * k1y + 1 * k2y))

        new_x = x + dt * (1/4 * k1x + 2/4 * k2x + 1/4 * k3x)
        new_y = y + dt * (1/4 * k1y + 2/4 * k2y + 1/4 * k3y)

        return new_x, new_y

def plot_deformation(rect, integrator, dt, time_limit=5):
    initial_x = [point.initial_x for point in rect.points]
    initial_y = [point.initial_y for point in rect.points]
    
    t = 0
    while t < time_limit:
      for point in rect.points:
        new_x, new_y = integrator.integrate(point, t, dt)
        point.update_position(new_x, new_y)
      t += dt

    final_x = [point.x for point in rect.points]
    final_y = [point.y for point in rect.points]
    plt.figure(figsize=(8, 8))
    plt.title("Деформация тела")
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.xlim(-20, 0)
    plt.ylim(0, 40)
    plt.scatter(initial_x, initial_y, color='blue', label='Начальная форма', s=10)
    plt.scatter(final_x, final_y, color='green', label=f'Форма через {time_limit} сек', s=10)


    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def plot_trajectories(rect, integrator, dt, trajectory_duration = 5):
    times = np.arange(0, trajectory_duration, dt)

    for t in times:
        for point in rect.points:
            new_x, new_y = integrator.integrate(point, t, dt)
            point.update_position(new_x, new_y)

    plt.figure(figsize=(8, 8))
    plt.title("Траектории материальных точек")
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.xlim(-20, 0)
    plt.ylim(0, 40)
    for point in rect.points:
        trajectory = np.array(point.trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1], linewidth=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def plot_streamlines_and_velocities(integrator, times):
    x = np.linspace(-20, 0, 30)
    y = np.linspace(0, 20, 30)
    X, Y = np.meshgrid(x, y)

    fig, axs = plt.subplots(len(times), 2, figsize=(12, 5 * len(times)))

    for i, t in enumerate(times):
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        for ix in range(X.shape[0]):
            for iy in range(X.shape[1]):
                U[ix, iy], V[ix, iy] = integrator.compute_velocity(t, X[ix, iy], Y[ix, iy])

        axs[i, 0].streamplot(x, y, U, V, color='black', density=1.5)
        axs[i, 0].set_title(f"Линии тока при t = {t:.2f}")
        axs[i, 0].set_xlabel("x")
        axs[i, 0].set_ylabel("y")
        axs[i, 0].set_xlim(-20, 0)
        axs[i, 0].set_ylim(0, 20)

        speed = np.sqrt(U**2 + V**2)
        im = axs[i, 1].imshow(speed, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='viridis')
        axs[i, 1].set_title(f"Распределение скоростей при t = {t:.2f}")
        axs[i, 1].set_xlabel("x")
        axs[i, 1].set_ylabel("y")
        fig.colorbar(im, ax=axs[i, 1], label='Модуль скорости')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    center_x = -10
    center_y = 10
    width = 4
    height = 2
    num_points_x = 10
    num_points_y = 5
    dt = 0.02
    time_limit = 1

    A = lambda t: 1 + 0.1 * t
    B = lambda t: 1 + 0.2 * t


    rect = Rectangle(center_x, center_y, width, height, num_points_x, num_points_y)
    integrator = RungeKuttaIntegrator(A, B)

    plot_deformation(rect, integrator, dt, time_limit)

    plot_trajectories(rect, integrator, dt)

    times = [0, 1, 3]
    plot_streamlines_and_velocities(integrator, times)



# In[ ]:





# In[ ]:




