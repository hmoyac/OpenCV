import cv2
import matplotlib.pyplot as plt
#%matplotlib inline

# %% [markdown]
# ## Exercise 2: Template Matching
# Task: Use template matching to locate a small region (e.g., a car wheel) in the image.

# %% [code] {"execution":{"iopub.status.busy":"2025-06-02T19:35:37.262152Z","iopub.execute_input":"2025-06-02T19:35:37.262512Z","iopub.status.idle":"2025-06-02T19:35:37.693897Z","shell.execute_reply.started":"2025-06-02T19:35:37.262487Z","shell.execute_reply":"2025-06-02T19:35:37.692802Z"}}
# Load image and template
img = cv2.imread('car.png')
if img is None:
    print("Error: Image not found.")
    exit()
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
template = cv2.imread('car mirror.png', cv2.IMREAD_GRAYSCALE)
if template is None:
    print("Error: Template not found.")
    exit()

# Convert image to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Perform template matching
result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Get template dimensions
w, h = template.shape[::-1]

# Draw rectangle around matched region
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img_rgb, top_left, bottom_right, (255, 0, 0), 8)

# Display result
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 2)
plt.title('Template Matching Result')
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1, 2, 1)
plt.title('Template')
plt.imshow(template,cmap='gray')
plt.axis('off')

