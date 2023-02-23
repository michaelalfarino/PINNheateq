# # PERSAMAAN PANAS 2 DIMENSI DENGAN SUMBER PANAS (VARIASI PANAS DI TENGAH DOMAIN)
import deepxde as dde
import numpy as np
from deepxde.backend import tf

# library untuk membuat animasi:
from IPython.display import HTML
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from matplotlib import cm

# pengaturan
plt.rcParams['animation.ffmpeg_path'] = r'C:\FFmpeg\bin\ffmpeg.exe'
dde.config.set_default_float("float64")
dde.optimizers.config.set_LBFGS_options(maxiter=15000)


# # PINN
# fungsi untuk memperoleh titik2 domain
def gen_points():
    # jumlah titik tiap dimensi:
    x_dim, y_dim, t_dim = (50, 50, 1000)

    # batas dari x dan t:
    x_min, y_min, t_min = (0, 0, 0.0)
    x_max, y_max, t_max = (L, L, maxtime)

    # titik-titik domain
    t = np.linspace(t_min, t_max, num=t_dim).reshape(t_dim, 1)
    x = np.linspace(x_min, x_max, num=x_dim).reshape(x_dim, 1)
    y = np.linspace(y_min, y_max, num=y_dim).reshape(y_dim, 1)

    # Save hasil
    np.savez("heat_eq_data", x=x, y=y, t=t)

# baca titik2 domain
def gen_testdata():
    # Load data:
    data = np.load("heat_eq_data.npz")
    t, x, y = data["t"], data["x"], data["y"]
    # ratakan data
    yy, tt, xx = np.meshgrid(y, t, x)
    X = np.vstack((np.ravel(xx), np.ravel(yy), np.ravel(tt))).T
    return X

# Problem parameters:
a = 2.225e-5  # Thermal diffusivity
L = 1  # panjang interval
maxtime = 10000
var = .05
mean = .25

# peroleh titik2 domain
gen_points()


def pde(x, u):
    # persamaan diferensial parsial: persamaan panas 2d dengan sumber panas
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    du_yy = dde.grad.hessian(u, x, i=1, j=1)
    du_t = dde.grad.jacobian(u, x, i=0, j=2)
    K = 0.00085*1000/63.65
    S = K / (var*var*(2*np.pi)) * tf.exp(-0.5*(((x[:, 0:1]-mean)/var)**2 + ((x[:, 1:2]-mean)/var)**2))
    return du_t - (a * (du_xx + du_yy)) - S

# geometri domain permasalahan
geom = dde.geometry.Rectangle((0,0), (L,L))
timedomain = dde.geometry.TimeDomain(0, maxtime)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Initial and boundary conditions:
bc = dde.icbc.DirichletBC(geomtime, lambda x: 25, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(
    geomtime,
    lambda x: 25,
    lambda _, on_initial: on_initial,
                    )

# definisi PDP dan konfigurasi neural network
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    num_domain=5000,
    num_boundary=400,
    num_initial=200,
    num_test=5000,
)
net = dde.nn.FNN([3] + [32] * 4 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)


# fungsi feature transform
def feature_transform(X):
  return tf.concat([tf.reshape(X[:, 0:1], (-1,1)), tf.reshape(X[:, 1:2], (-1,1)), tf.reshape(X[:, 2:], (-1,1))*1e-3], axis=1)

# tambahkan layer feature transform
net.apply_feature_transform(feature_transform)

# tambahkan layer output transform
net.apply_output_transform(lambda x, y: y*100)

# Bangun dan latih model dengan 2 metode optimasi
model.compile("adam", lr=1e-3, loss_weights=[1e3,1e-2,1e-2])
model.train(iterations=10000)
model.compile("L-BFGS", loss_weights=[1e3,1e-2,1e-2])
losshistory, train_state = model.train()

# Plot/print hasil
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
X = gen_testdata()
y_PINN = model.predict(X)
np.savetxt("test.dat", np.hstack((X, y_PINN)))



# plot metode PINN
fig = plt.figure(figsize=(11,11))
plt.suptitle("Metode PINN")

ax = fig.add_subplot(3, 2, 1, projection='3d')
surf = ax.scatter(X[2500:5000, 0], X[2500:5000, 1], y_PINN[2500:5000], c=y_PINN[2500:5000], cmap=cm.coolwarm)
ax.set_title("t = {:.0f}".format(X[0, 2]))
ax.set_zlim(20, 250)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$Temperatur$')
fig.colorbar(surf, shrink=.5, location='left')

ax = fig.add_subplot(3, 2, 2, projection='3d')
surf = ax.scatter(X[10000:12500, 0], X[10000:12500, 1], y_PINN[10000:12500], c=y_PINN[10000:12500], cmap=cm.coolwarm)
ax.set_title("t = {:.0f}".format(X[10000, 2]))
ax.set_zlim(20, 250)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$Temperatur$')
fig.colorbar(surf, shrink=.5, location='left')

ax = fig.add_subplot(3, 2, 3, projection='3d')
surf = ax.scatter(X[100000:102500, 0], X[100000:102500, 1], y_PINN[100000:102500], c=y_PINN[100000:102500], cmap=cm.coolwarm)
ax.set_title(" \n \nt = {:.0f}".format(X[100000, 2]))
ax.set_zlim(20, 250)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$Temperatur$')
fig.colorbar(surf, shrink=.5, location='left')

ax = fig.add_subplot(3, 2, 4, projection='3d')
surf = ax.scatter(X[500000:502500, 0], X[500000:502500, 1], y_PINN[500000:502500], c=y_PINN[500000:502500], cmap=cm.coolwarm)
ax.set_title(" \n \nt = 2000")#{:.0f}".format(X[500000, 2]))
ax.set_zlim(20, 250)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$Temperatur$')
fig.colorbar(surf, shrink=.5, location='left')

ax = fig.add_subplot(3, 2, 5, projection='3d')
surf = ax.scatter(X[1497500:1500000, 0], X[1497500:1500000, 1], y_PINN[1497500:1500000], c=y_PINN[1497500:1500000], cmap=cm.coolwarm)
ax.set_title(" \n \nt = 6000")#:.0f}".format(X[1497500, 2]))
ax.set_zlim(20, 250)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$Temperatur$')
fig.colorbar(surf, shrink=.5, location='left')

ax = fig.add_subplot(3, 2, 6, projection='3d')
surf = ax.scatter(X[2497500:2500000, 0], X[2497500:2500000, 1], y_PINN[2497500:2500000], c=y_PINN[2497500:2500000], cmap=cm.coolwarm)
ax.set_title(" \n \nt = {:.0f}".format(X[2497500, 2]))
ax.set_zlim(20, 250)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$Temperatur$')
fig.colorbar(surf, shrink=.5, location='left')

plt.tight_layout()
plt.savefig("PINN2dhsvar2.png", dpi=300)



# PLot temperatur terhadap waktu pada 6 titik
idx1 = np.where(np.isclose(X[:, 0], 1) & (np.isclose(X[:, 0], X[:, 1])))[0]
idx2 = np.where((X[:, 0] >= 0.84) & (X[:, 0] <= 0.86) & (np.isclose(X[:, 0], X[:, 1])))[0]
idx3 = np.where((X[:, 0] >= 0.68) & (X[:, 0] <= 0.7) & (np.isclose(X[:, 0], X[:, 1])))[0]
idx4 = np.where((X[:, 0] >= 0.54) & (X[:, 0] <= 0.56) & (np.isclose(X[:, 0], X[:, 1])))[0]
idx5 = np.where((X[:, 0] >= 0.38) & (X[:, 0] <= 0.4) & (np.isclose(X[:, 0], X[:, 1])))[0]
idx6 = np.where((X[:, 0] >= 0.24) & (X[:, 0] <= 0.26) & (np.isclose(X[:, 0], X[:, 1])))[0]

result1 = X[idx1]
result2 = X[idx2]
result3 = X[idx3]
result4 = X[idx4]
result5 = X[idx5]
result6 = X[idx6]

fig = plt.figure(figsize=(10,7))
plt.suptitle("Perkembangan Temperatur terhadap Waktu pada Beberapa Titik")

ax = fig.add_subplot(2, 3, 1)
ax.set_ylabel("Temperatur")
ax.set_xlabel("t")
ax.plot(result1[:, 2], y_PINN[idx1])
plt.title("(x, y) = ({:.2f}, {:.2f})".format(result1[0, 1], result1[0, 1]))
plt.ylim((0,100))

ax = fig.add_subplot(2, 3, 2)
ax.set_ylabel("Temperatur")
ax.set_xlabel("t")
ax.plot(result2[:, 2], y_PINN[idx2])
plt.title("(x, y) = ({:.2f}, {:.2f})".format(result2[0, 1], result2[0, 1]))
plt.ylim((0,100))

ax = fig.add_subplot(2, 3, 3)
ax.set_ylabel("Temperatur")
ax.set_xlabel("t")
ax.plot(result3[:, 2], y_PINN[idx3])
plt.title("(x, y) = ({:.2f}, {:.2f})".format(result3[0, 1], result3[0, 1]))
plt.ylim((0,100))

ax = fig.add_subplot(2, 3, 4)
ax.set_ylabel("Temperatur")
ax.set_xlabel("t")
ax.plot(result4[:, 2], y_PINN[idx4])
plt.title("(x, y) = ({:.2f}, {:.2f})".format(result4[0, 1], result4[0, 1]))
plt.ylim((0,100))

ax = fig.add_subplot(2, 3, 5)
ax.set_ylabel("Temperatur")
ax.set_xlabel("t")
ax.plot(result5[:, 2], y_PINN[idx5])
plt.title("(x, y) = ({:.2f}, {:.2f})".format(result5[0, 1], result5[0, 1]))
plt.ylim((0,150))

ax = fig.add_subplot(2, 3, 6)
ax.set_ylabel("Temperatur")
ax.set_xlabel("t")
ax.plot(result6[:, 2], y_PINN[idx6])
plt.title("(x, y) = ({:.2f}, {:.2f})".format(result6[0, 1], result6[0, 1]))
plt.ylim((0,250))

plt.tight_layout()
plt.savefig("PINN2dhstimevar2.png", dpi=300)



# Animasi hasil
result = np.hstack((X, y_PINN))
time = np.unique(result[:, 2])

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(1, 1, 1, projection='3d')

def update(frame):
    ax.clear()
    ax.set_title(" ")
    ax.set_zlim([20, 150])
    ax.set_zlabel("Temperatur")
    ax.set_ylabel("y")
    ax.set_xlabel("x")
    ax.grid()
    a = np.where(result[:, 2] == time[frame])[0][0]
    b = np.where(result[:, 2] == time[frame])[0][-1]
    surface = ax.scatter(result[a:b, 0], result[a:b, 1], result[a:b, 3], c=result[a:b, 3], cmap=cm.coolwarm)
    ax.set_title("Persebaran panas t = {:.0f}".format(result[a, 2]))

ani = FuncAnimation(fig, update, frames=int(len(time)))

# Save animasi sebagai video
ani.save("heat eq 2dhsvar2.mp4", bitrate=6000, dpi=200, fps=30)



# simpan plot loss history
loss_train = np.sum(losshistory.loss_train, axis=1)
loss_test = np.sum(losshistory.loss_test, axis=1)

plt.figure(figsize=(6,4))
plt.grid()
plt.title("Perkembangan $loss$ $function$ tiap iterasi")
plt.semilogy(losshistory.steps, loss_train, label="Train loss")
plt.semilogy(np.concatenate((losshistory.steps[:12], losshistory.steps[-1:])), np.concatenate((loss_test[:12], loss_test[-1:])), label="Test loss")
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