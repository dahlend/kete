import matplotlib.pyplot as plt
import numpy as np

import kete

lon_steps = np.linspace(0, 360, 1000)
equatorial = [kete.Vector.from_ra_dec(step, 0) for step in lon_steps]
ecliptic = [kete.Vector.from_lat_lon(0, step) for step in lon_steps]
galactic = [
    kete.Vector.from_el_az(0, step, 1, frame=kete.Frames.Galactic) for step in lon_steps
]

vectors_frames = [equatorial, ecliptic, galactic]

fig = plt.figure(figsize=(6, 8))
plt.subplot(3, 1, 1, projection="mollweide")
plt.title("Equatorial Frame")
for vectors in vectors_frames:
    frame = str(vectors[0].frame).split(".")[1]
    vecs = [v.as_equatorial for v in vectors]
    pos = np.array([[v.az, v.el] for v in vecs])

    # roll the indices so that it plots pretty
    # this is just to make the visualization nice
    pos[:, 0] = (pos[:, 0] + 180) % 360 - 180
    idx = np.argmax(abs(np.diff(pos[:, 0])))
    pos = np.roll(pos, -idx - 1, axis=0)
    pos = np.radians(pos)

    # plot the results
    plt.plot(*pos.T, label=frame)
    plt.grid()


plt.subplot(3, 1, 2, projection="mollweide")
plt.title("Ecliptic Frame")
for vectors in vectors_frames:
    frame = str(vectors[0].frame).split(".")[1]
    vecs = [v.as_ecliptic for v in vectors]
    pos = np.array([[v.az, v.el] for v in vecs])

    # roll the indices so that it plots pretty
    # this is just to make the visualization nice
    pos[:, 0] = (pos[:, 0] + 180) % 360 - 180
    idx = np.argmax(abs(np.diff(pos[:, 0])))
    pos = np.roll(pos, -idx - 1, axis=0)
    pos = np.radians(pos)

    # plot the results
    plt.plot(*pos.T, label=frame)
    plt.grid()


plt.subplot(3, 1, 3, projection="mollweide")
plt.title("Galactic Frame")
for vectors in vectors_frames:
    frame = str(vectors[0].frame).split(".")[1]
    vecs = [v.as_galactic for v in vectors]
    pos = np.array([[v.az, v.el] for v in vecs])

    # roll the indices so that it plots pretty
    # this is just to make the visualization nice
    pos[:, 0] = (pos[:, 0] + 180) % 360 - 180
    idx = np.argmax(abs(np.diff(pos[:, 0])))
    pos = np.roll(pos, -idx - 1, axis=0)
    pos = np.radians(pos)

    # plot the results
    plt.plot(*pos.T, label=frame)
    plt.grid()
plt.tight_layout()
plt.legend(bbox_to_anchor=(1.05, 1.29), framealpha=1)
plt.savefig("data/background_frames.png")
