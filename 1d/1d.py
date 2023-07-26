# # PERSAMAAN PANAS 1 DIMENSI
import deepxde as dde
import numpy as np

# library untuk membuat animasi:
from IPython.display import HTML
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

# pengaturan
plt.rcParams['animation.ffmpeg_path'] = r'C:\FFmpeg\bin\ffmpeg.exe'
dde.config.set_default_float("float64")
dde.optimizers.config.set_LBFGS_options(maxiter=15000)



# # PINN

# solusi analitik persamaan panas 1d
def heat_eq_exact_solution(x, t):
    return np.exp(-(np.pi**2 * a * t)) * np.sin(np.pi * x)

# fungsi untuk menghitung solusi analitik
def gen_exact_solution():
    # jumlah titik tiap dimensi:
    x_dim, t_dim = (1000, 1000)

    # batas dari x dan t:
    x_min, t_min = (0, 0.0)
    x_max, t_max = (L, maxtime)

    # titik titik yg akan dicari solusinya:
    t = np.linspace(t_min, t_max, num=t_dim).reshape(t_dim, 1)
    x = np.linspace(x_min, x_max, num=x_dim).reshape(x_dim, 1)
    usol = np.zeros((x_dim, t_dim)).reshape(x_dim, t_dim)

    # peroleh solusi tiap titik:
    for i in range(x_dim):
        for j in range(t_dim):
            usol[i][j] = heat_eq_exact_solution(x[i], t[j])

    # Save:
    np.savez("heat_eq_data", x=x, t=t, usol=usol)

# fungsi untuk membaca data analitik
def gen_testdata():
    # Load data:
    data = np.load("heat_eq_data.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    # ratakan data:
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y

# Problem parameters:
a = 0.4  # Thermal diffusivity
L = 1  # panjang interval
maxtime = 1

# peroleh solusi analitik
gen_exact_solution()

def pde(x, y):
    # persamaan diferensial parsial: persamaan panas 1d
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - a * dy_xx

# geometri domain permasalahan
geom = dde.geometry.Interval(0, L)
timedomain = dde.geometry.TimeDomain(0, maxtime)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Initial and boundary conditions:
bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(
    geomtime,
    lambda x: np.sin(np.pi * x[:, 0:1]),
    lambda _, on_initial: on_initial,
)

# definisi PDP dan konfigurasi neural network
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    num_domain=2540,
    num_boundary=80,
    num_initial=160,
    num_test=2540,
)
net = dde.nn.FNN([2] + [32] * 4 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

# Bangun dan latih model dengan 2 metode optimasi
model.compile("adam", lr=1e-3, loss_weights=[1, 1, 1])
model.train(iterations=10000)
model.compile("L-BFGS", loss_weights=[1, 1, 1])
losshistory, train_state = model.train()

# Plot/print hasil
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
X, y_true = gen_testdata()
y_PINN = model.predict(X)
print("Mean squared error:", dde.metrics.mean_squared_error(y_true, y_PINN))
np.savetxt("test.dat", np.hstack((X, y_true, y_PINN)))



# Numeric FTCS

# library untuk menghitung waktu running FTCS
import time
start = time.time()

# deklarasi ukuran diferensial
dt = 1e-4
dx = 0.01

# inisialisasi domain
xs = np.arange(0, L, dx)
ts = np.arange(0, maxtime, dt)
u = np.sin(np.pi*xs)     #initial condition

u[0] = 0   #boundary condition
u[-1] = 0

y_num = np.empty(3)

for n in ts:  #iterasi melalui waktu
    un = u.copy()
    u[1:-1] = un[1:-1] + a * dt / dx**2 * (un[2:] - 2 * un[1:-1] + un[0:-2])
    u[0] = 0
    u[-1] = 0
    temp = np.hstack((xs.reshape(-1,1), np.full(len(xs), n).reshape(-1,1), u.reshape(-1,1)))
    y_num = np.vstack((y_num, temp))
    
y_num = np.delete(y_num, 0, 0)
end = time.time()

print('total waktu kalkulasi FTCS', end-start)



# plot hasil dari persamaan panas

fig, axs = plt.subplots(3, 2, figsize=(10,13))
figTime = 0
fig.suptitle("Perbandingan Ketiga Metode")
for ax in axs.flat:
    ax.grid()
    ax.set(xlabel='$x$', ylabel='$u(x, t = {:.2f})$'.format(time[figTime]))
    ax.set_ylim([-.1, 1.1])
    markersize = [10 for i in range(len(y_num[np.where(np.isclose(y_num[:, 1],time[figTime]))[0][0]:np.where(np.isclose(y_num[:, 1],time[figTime]))[0][-1], 0]))]
    ax.plot(result[np.where(result[:, 1] == time[figTime])[0][0]:np.where(result[:, 1] == time[figTime])[0][-1], 0], result[np.where(result[:, 1] == time[figTime])[0][0]:np.where(result[:, 1] == time[figTime])[0][-1], 2], 'r-', linewidth=4, label = 'Analitik', zorder = 0)
    ax.plot(result[np.where(result[:, 1] == time[figTime])[0][0]:np.where(result[:, 1] == time[figTime])[0][-1], 0], result[np.where(result[:, 1] == time[figTime])[0][0]:np.where(result[:, 1] == time[figTime])[0][-1], 3], 'k--', linewidth=2, label = 'PINN', zorder = 5)
    ax.scatter(y_num[np.where(np.isclose(y_num[:, 1],time[figTime]))[0][0]:np.where(np.isclose(y_num[:, 1],time[figTime]))[0][-1], 0], y_num[np.where(np.isclose(y_num[:, 1],time[figTime]))[0][0]:np.where(np.isclose(y_num[:, 1],time[figTime]))[0][-1], 2], markersize, marker='x', c='b', label = 'FTCS', zorder = 10)
    ax.set_title("$t = {:.2f}$".format(time[figTime]))
    figTime += 199

axs[0,1].legend()

plt.tight_layout()
plt.savefig("result.png", dpi=300)



# Animasi hasil

result = np.hstack((X, y_true, y_PINN))
time = np.unique(result[:, 1])
time2 = np.unique(y_num[:, 1])

fig, ax = plt.subplots(figsize=(7,5))

def update(frame):
    ax.clear()
    ax.set_title(" ")
    ax.set_ylim([-.1, 1.1])
    ax.set_ylabel("Temperatur")
    ax.set_xlabel("x")
    ax.grid()
    a = np.where(result[:, 1] == time[frame])[0][0]
    b = np.where(result[:, 1] == time[frame])[0][-1]
    c = np.where(np.isclose(y_num[:, 1], time[frame], atol=5e-4))[0][0]
    d = np.where(np.isclose(y_num[:, 1], time[frame], atol=5e-4))[0][-1]
    ax.plot(result[a:b, 0], result[a:b, 2], 'r-', linewidth=4, label = 'Analitik', zorder = 0)
    ax.plot(result[a:b, 0], result[a:b, 3], 'k--', linewidth=2, label = 'PINN', zorder = 5)
    ax.scatter(y_num[c:d, 0], y_num[c:d, 2], marker='x', label = 'FTCS', zorder = 10)
    ax.set_title("Persebaran panas t = {:.2f}".format(result[a, 1]))
    ax.legend()

ani = FuncAnimation(fig, update, frames=int(len(time)))

# Save animasi sebagai gif
ani.save("heat eq 1d.gif", dpi=200, fps=30)


# mengukur MSE tiap metode
from sklearn.metrics import mean_squared_error as MSE
print('\033[1m' + "Method         Peak                   MSE" + '\033[0m' + 
      "\nAnalytic      ", heat_eq_exact_solution(result[:, 0], result[:, 1]).max(),
      "\nFTCS          ", y_num[:,2].max(),"   ", MSE(y_num[:,2], heat_eq_exact_solution(y_num[:,0], y_num[:,1])), 
      "\nPINN          ", result[:, 3].max(),"   ", MSE(result[:, 3], heat_eq_exact_solution(result[:, 0], result[:, 1]))
     )



# simpan plot loss history
loss_train = np.sum(losshistory.loss_train, axis=1)
loss_test = np.sum(losshistory.loss_test, axis=1)

plt.figure(figsize=(6,4))
plt.grid()
plt.title("Perkembangan $loss$ $function$ tiap iterasi")
plt.semilogy(losshistory.steps, loss_train, label="Train loss")
plt.semilogy(losshistory.steps, loss_test, label="Test loss")
for i in range(len(losshistory.metrics_test[0])):
    plt.semilogy(
        loss_history.steps,
        np.array(loss_history.metrics_test)[:, i],
        label="Test metric",
    )
plt.xlabel("Banyak iterasi")
plt.ylabel("$\mathcal{L}$")
plt.legend()
plt.savefig('loss.png', dpi=300)
