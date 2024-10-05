import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
steps = 9900  # Total simulation steps (make sure these match your saved files)
NX, NY = 400, 50  # Grid size

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.imshow(np.zeros((NY, NX)), cmap='jet', vmin=0, vmax=1)
plt.colorbar(cax)
ax.set_title('Velocity Magnitude Over Time')

def update(frame):
    # Load the velocity field for the current step
    u = np.load(f'animation/velocity_magnitude_step_{frame}.npy')  # Ensure these files exist
    U = u[:, :, 0]
    V = u[:, :, 1]

    # Calculate the velocity magnitude
    velocity_magnitude = np.sqrt(U**2 + V**2)

    vmin = np.min(velocity_magnitude)
    vmax = np.max(velocity_magnitude)

    # Update the image data
    cax.set_array(velocity_magnitude.T)  # Transpose for correct orientation
    cax.set_clim(vmin, vmax)  # Set color limits
    ax.set_title(f'Velocity Magnitude at Step {frame}')
    return cax,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=range(0, steps, 100), blit=True, interval=100)
fps = 1000 / 100

# Save the animation as a video
ani.save('velocity_magnitude.mp4', writer='ffmpeg', fps=fps)

plt.show()
